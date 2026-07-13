# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Micro-benchmark for color-code inference post-processing optimizations.

Measures isolated operations and their optimized replacements:

  1. GPU bit-packing: residual.cpu().numpy() + np.packbits  →  GPU packbits + smaller transfer
  2. Boundary detector transfer: numpy slice → to(device) per batch  →  pre-loaded GPU tensor
  3. Reshape: .contiguous().view()  →  .reshape()
  4. Chromobius decode time vs GPU work time (overlap feasibility)
  5. Parallel batch decode: splitting B samples across N Python threads.
     FINDING: No benefit (~1.0x).  Chromobius is already internally multi-threaded
     and saturates CPU cores; Python-level splits add overhead without parallelism.
  6. Within-batch overlap: baseline Chromobius (CPU thread) vs realistic fake GPU
     forward pass, sized to match a 3D-CNN model at each D/B.
     This mirrors the optimization in run_inference_and_decode_color where the
     baseline decode is submitted to a ThreadPoolExecutor before the model forward.

Run on a machine with a CUDA GPU for meaningful results on ops #1, #4, #6.
On CPU, only #2 and #3 produce meaningful comparisons.

Usage:
    python code/benchmarks/bench_color_inference_postproc.py
    python code/benchmarks/bench_color_inference_postproc.py --device cuda
    python code/benchmarks/bench_color_inference_postproc.py --device cpu --warmup 5 --reps 50
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow importing from repo root when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub optional training-time deps so the benchmark can import color-code helpers
# in lean inference environments. If a future import adds another optional dep,
# add it to this list rather than making the benchmark require the full stack.
from unittest.mock import MagicMock

_flax_mock = MagicMock()
for _mod in ["flax", "flax.linen", "optax"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _flax_mock

try:
    import chromobius
    import stim
    _HAS_CHROMOBIUS = True
except ImportError:
    _HAS_CHROMOBIUS = False

# ---------------------------------------------------------------------------
# Representative color-code dimensions
# (num_plaq, num_data, T) for distances 5, 7, 9 with standard round counts
# ---------------------------------------------------------------------------
CONFIGS = [
    dict(D=5, num_plaq=18, num_data=25, n_rows=7, n_cols=5, T=13),
    dict(D=7, num_plaq=36, num_data=49, n_rows=11, n_cols=7, T=21),
    dict(D=9, num_plaq=60, num_data=81, n_rows=15, n_cols=9, T=29),
]
BATCH_SIZES = [256, 1024]


# ---------------------------------------------------------------------------
# GPU bit-packing (little-endian, matching np.packbits bitorder='little')
# ---------------------------------------------------------------------------
def packbits_gpu(t: torch.Tensor) -> torch.Tensor:
    """Pack (B, N) uint8 bits into (B, ceil(N/8)) uint8 on the same device."""
    B, N = t.shape
    pad = (8 - N % 8) % 8
    if pad:
        t = F.pad(t, (0, pad))
    t = t.view(B, -1, 8).to(torch.int32)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int32, device=t.device)
    return (t * powers).sum(dim=2).to(torch.uint8)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timer(device):
    _sync(device)
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def bench(fn, warmup, reps, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps * 1e3  # ms per call


def run_benchmarks(device: torch.device, warmup: int, reps: int):
    on_gpu = device.type == "cuda"
    sep = "─" * 70

    print(f"\n{'='*70}")
    print(
        f"  Device: {device}{'  (GPU transfer savings visible here)' if on_gpu else '  (CPU-only: op #1 not meaningful)'}"
    )
    print(f"  Warmup: {warmup}  Reps: {reps}")
    print(f"{'='*70}")

    for cfg in CONFIGS:
        D = cfg["D"]
        num_plaq = cfg["num_plaq"]
        T = cfg["T"]
        n_rows = cfg["n_rows"]
        n_cols = cfg["n_cols"]
        num_boundary = (3 * (D * D - 1)) // 8
        # Total detectors: num_plaq*(2T-1) + num_boundary
        N_det = num_plaq * (2 * T - 1) + num_boundary

        print(f"\nD={D}  num_plaq={num_plaq}  T={T}  N_det={N_det}")
        print(sep)

        for B in BATCH_SIZES:
            # ----------------------------------------------------------------
            # Shared inputs
            # ----------------------------------------------------------------
            residual = torch.randint(0, 2, (B, N_det), dtype=torch.uint8, device=device)
            stim_dets = np.random.randint(0, 2, (B * 4, N_det), dtype=np.uint8)
            z_data_corr = torch.randint(
                0, 2, (B, 1, T, n_rows, n_cols), dtype=torch.int32, device=device
            )

            # Pre-loaded boundary tensor (optimization #2)
            boundary_gpu = torch.from_numpy(stim_dets[:, -num_boundary:]).to(device)

            print(
                f"\n  B={B}  residual {residual.shape}  ({residual.numel()/1024:.1f} KB  packed: {residual.numel()//8/1024:.1f} KB)"
            )

            # ----------------------------------------------------------------
            # Op #1: GPU→CPU transfer + packbits
            # ----------------------------------------------------------------
            def op1_baseline():
                r_np = residual.cpu().numpy()
                return np.packbits(r_np, axis=1, bitorder="little")

            def op1_optimized():
                packed = packbits_gpu(residual)
                return packed.cpu().numpy()

            t_base = bench(op1_baseline, warmup, reps, device)
            t_opt = bench(op1_optimized, warmup, reps, device)
            tag = "" if on_gpu else "  [needs GPU]"
            print(
                f"  #1 transfer+pack  baseline: {t_base:.3f} ms   optimized: {t_opt:.3f} ms   speedup: {t_base/t_opt:.2f}x{tag}"
            )

            # ----------------------------------------------------------------
            # Op #2: boundary detector transfer
            # ----------------------------------------------------------------
            offset = 0

            def op2_baseline():
                nonlocal offset
                bd = stim_dets[offset:offset + B, -num_boundary:]
                result = torch.from_numpy(bd).to(device)
                offset = (offset + B) % (B * 4)
                return result

            slice_start = 0

            def op2_optimized():
                nonlocal slice_start
                result = boundary_gpu[slice_start:slice_start + B]
                slice_start = (slice_start + B) % (B * 4)
                return result

            t_base = bench(op2_baseline, warmup, reps, device)
            t_opt = bench(op2_optimized, warmup, reps, device)
            print(
                f"  #2 boundary xfer  baseline: {t_base:.3f} ms   optimized: {t_opt:.3f} ms   speedup: {t_base/t_opt:.2f}x"
            )

            # ----------------------------------------------------------------
            # Op #3: contiguous().view() vs reshape()
            # ----------------------------------------------------------------
            def op3_baseline():
                return z_data_corr.permute(0, 2, 3, 4, 1).contiguous().view(B, T, n_rows * n_cols)

            def op3_optimized():
                return z_data_corr.permute(0, 2, 3, 4, 1).reshape(B, T, n_rows * n_cols)

            t_base = bench(op3_baseline, warmup, reps, device)
            t_opt = bench(op3_optimized, warmup, reps, device)
            print(
                f"  #3 reshape        baseline: {t_base:.3f} ms   optimized: {t_opt:.3f} ms   speedup: {t_base/t_opt:.2f}x"
            )

            # ----------------------------------------------------------------
            # Op #4: Chromobius decode time vs GPU work — overlap feasibility
            # Build a real color-code circuit and decoder; measure decode time
            # at two syndrome densities (baseline ~10%, residual ~1%).
            # Also measures the threaded overlap vs sequential.
            # ----------------------------------------------------------------
            if _HAS_CHROMOBIUS:
                from qec.color_code.reference_superdense_noise import build_color_memory_circuit
                circ_obj = build_color_memory_circuit(
                    distance=D,
                    n_rounds=T,
                    basis="X",
                    p_error=0.001,
                    noise_model_family="legacy",
                    noise_instruction_semantics="current",
                )
                circuit = circ_obj.stim_circuit
                dem = circuit.detector_error_model(
                    decompose_errors=False,
                    approximate_disjoint_errors=True,
                    ignore_decomposition_failures=True,
                )
                decoder = chromobius.compile_decoder_for_dem(dem)
                N_det_real = dem.num_detectors

                # Sample physically valid syndromes at two error rates
                sampler_hi = build_color_memory_circuit(
                    distance=D,
                    n_rounds=T,
                    basis="X",
                    p_error=0.005,
                    noise_model_family="legacy",
                    noise_instruction_semantics="current",
                ).stim_circuit.compile_detector_sampler()
                sampler_lo = build_color_memory_circuit(
                    distance=D,
                    n_rounds=T,
                    basis="X",
                    p_error=0.0005,
                    noise_model_family="legacy",
                    noise_instruction_semantics="current",
                ).stim_circuit.compile_detector_sampler()

                shots_hi = sampler_hi.sample(B).astype(np.uint8)
                shots_lo = sampler_lo.sample(B).astype(np.uint8)
                packed_hi = np.packbits(shots_hi, axis=1, bitorder="little")
                packed_lo = np.packbits(shots_lo, axis=1, bitorder="little")

                # Warmup
                for _ in range(warmup):
                    decoder.predict_obs_flips_from_dets_bit_packed(packed_hi)
                    decoder.predict_obs_flips_from_dets_bit_packed(packed_lo)

                # Sequential timing
                t0 = time.perf_counter()
                for _ in range(reps):
                    decoder.predict_obs_flips_from_dets_bit_packed(packed_hi)
                t_chromo_hi = (time.perf_counter() - t0) / reps * 1e3

                t0 = time.perf_counter()
                for _ in range(reps):
                    decoder.predict_obs_flips_from_dets_bit_packed(packed_lo)
                t_chromo_lo = (time.perf_counter() - t0) / reps * 1e3

                # Simulate GPU work: a dummy kernel that takes ~t_gpu_sim ms
                # (stand-in for model forward + postproc; adjust to your model's cost)
                dummy_gpu = torch.zeros(B, 64, T, n_rows, n_cols, device=device)

                def gpu_work():
                    out = (dummy_gpu + 1).sum()
                    _sync(device)
                    return out

                t_gpu = bench(gpu_work, warmup, reps, device)

                # Threaded overlap: baseline decode concurrent with GPU work
                executor = ThreadPoolExecutor(max_workers=1)

                def overlap_baseline():
                    fut = executor.submit(decoder.predict_obs_flips_from_dets_bit_packed, packed_hi)
                    gpu_work()
                    return fut.result()

                for _ in range(warmup):
                    overlap_baseline()

                t0 = time.perf_counter()
                for _ in range(reps):
                    overlap_baseline()
                t_overlap = (time.perf_counter() - t0) / reps * 1e3

                executor.shutdown(wait=False)

                t_sequential = t_chromo_hi + t_gpu
                print(
                    f"  #4 Chromobius hi-density:  {t_chromo_hi:.2f} ms   "
                    f"lo-density: {t_chromo_lo:.2f} ms   GPU work: {t_gpu:.2f} ms"
                )
                print(
                    f"     sequential (baseline+GPU): {t_sequential:.2f} ms   "
                    f"overlapped: {t_overlap:.2f} ms   "
                    f"speedup: {t_sequential/t_overlap:.2f}x"
                )

                # ----------------------------------------------------------------
                # Op #5: Parallel batch decode — split B samples across N workers.
                # Variant A: shared decoder (tests GIL release + internal locking).
                # Variant B: one decoder instance per worker (avoids internal locks).
                # Each sample is independent; true parallel execution requires GIL
                # release in the Chromobius C++ extension.
                # ----------------------------------------------------------------
                for n_workers in [2, 4]:
                    chunk_size = max(1, B // n_workers)
                    chunks_hi = [packed_hi[i:i + chunk_size] for i in range(0, B, chunk_size)]

                    # Variant A: shared decoder
                    par_executor = ThreadPoolExecutor(max_workers=n_workers)

                    def parallel_decode_shared(chunks=chunks_hi, ex=par_executor):
                        futs = [
                            ex.submit(decoder.predict_obs_flips_from_dets_bit_packed, c)
                            for c in chunks
                        ]
                        return np.concatenate([f.result() for f in futs], axis=0)

                    for _ in range(warmup):
                        parallel_decode_shared()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        parallel_decode_shared()
                    t_par_shared = (time.perf_counter() - t0) / reps * 1e3
                    par_executor.shutdown(wait=False)

                    # Variant B: independent decoder per worker
                    decoders_n = [chromobius.compile_decoder_for_dem(dem) for _ in range(n_workers)]
                    par_executor2 = ThreadPoolExecutor(max_workers=n_workers)

                    def parallel_decode_indep(chunks=chunks_hi, decs=decoders_n, ex=par_executor2):
                        futs = [
                            ex.submit(d.predict_obs_flips_from_dets_bit_packed, c)
                            for d, c in zip(decs, chunks)
                        ]
                        return np.concatenate([f.result() for f in futs], axis=0)

                    for _ in range(warmup):
                        parallel_decode_indep()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        parallel_decode_indep()
                    t_par_indep = (time.perf_counter() - t0) / reps * 1e3
                    par_executor2.shutdown(wait=False)

                    print(
                        f"  #5 n_workers={n_workers}  shared: {t_par_shared:.2f} ms "
                        f"({t_chromo_hi/t_par_shared:.2f}x)  "
                        f"indep: {t_par_indep:.2f} ms ({t_chromo_hi/t_par_indep:.2f}x)"
                    )
                # ----------------------------------------------------------------
                # Op #6: Within-batch overlap — baseline Chromobius (CPU thread)
                # concurrent with a realistic fake GPU forward pass.
                # The fake forward uses a 3D conv-like workload sized to approximate
                # a 4-layer, 128-filter CNN on the color-code input at this D/B.
                # ----------------------------------------------------------------
                n_filters = 128
                fake_input = torch.zeros(B, 4, T, n_rows, n_cols, device=device)
                fake_w = torch.zeros(n_filters, 4, 3, 3, 3, device=device)

                def fake_gpu_forward(x=fake_input, w=fake_w):
                    out = F.conv3d(x, w, padding=1)
                    _sync(device)
                    return out

                t_gpu_real = bench(fake_gpu_forward, warmup, reps, device)

                # Sequential: baseline decode, then GPU forward
                def sequential_op6():
                    decoder.predict_obs_flips_from_dets_bit_packed(packed_hi)
                    fake_gpu_forward()

                for _ in range(warmup):
                    sequential_op6()
                t0 = time.perf_counter()
                for _ in range(reps):
                    sequential_op6()
                t_seq_op6 = (time.perf_counter() - t0) / reps * 1e3

                # Overlapped: baseline decode in thread, GPU forward in main
                overlap_ex = ThreadPoolExecutor(max_workers=1)

                def overlapped_op6(ex=overlap_ex):
                    fut = ex.submit(decoder.predict_obs_flips_from_dets_bit_packed, packed_hi)
                    fake_gpu_forward()
                    return fut.result()

                for _ in range(warmup):
                    overlapped_op6()
                t0 = time.perf_counter()
                for _ in range(reps):
                    overlapped_op6()
                t_ov_op6 = (time.perf_counter() - t0) / reps * 1e3
                overlap_ex.shutdown(wait=False)

                print(
                    f"  #6 within-batch overlap  GPU forward: {t_gpu_real:.2f} ms  "
                    f"sequential: {t_seq_op6:.2f} ms  "
                    f"overlapped: {t_ov_op6:.2f} ms  "
                    f"speedup: {t_seq_op6/t_ov_op6:.2f}x"
                )
            else:
                print("  #4 Chromobius timing: skipped (chromobius not installed)")
                print("  #5 parallel decode:   skipped (chromobius not installed)")
                print("  #6 within-batch overlap: skipped (chromobius not installed)")

    print(f"\n{sep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_benchmarks(device, args.warmup, args.reps)
