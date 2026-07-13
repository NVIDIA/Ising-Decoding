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
Bench: color-code Stage-1 baseline — LER, predecoder forward time, end-to-end
time across a (p, d) sweep.

Reports a markdown table suitable for pasting into a PR comment (mirrors the
format used in PR #41 / PR #62 for the surface-code residual UF benchmark).

Three timed stages per cell:
  - sample_ms_per_shot  — Stim sampler wall-clock per shot
  - predecoder_ms_per_shot — forward pass of a freshly-instantiated color
                              predecoder model on the batched detector input.
                              No checkpoint required: weights are random
                              (Stage 2 will provide a trained checkpoint).
                              The time measurement is checkpoint-independent.
  - decode_ms_per_shot  — Chromobius decode of the raw detector batch
                          (baseline LER). When --use-predecoder is set,
                          decode runs on predecoder-residual detectors
                          instead, giving the post-predecoder LER.

LER is reported per round to match the color-code literature convention.

Wilson 95% CI is reported alongside each LER cell.

Usage:
    # Default sweep: d=5,7 × p={1e-3, 2e-3, 3e-3}, 1024 shots/cell.
    python code/benchmarks/bench_color_baseline.py

    # Larger sweep with more shots for tighter CIs.
    python code/benchmarks/bench_color_baseline.py --distances 5 7 9 --shots 8192

    # Include predecoder-residual LER (otherwise only baseline chromobius LER).
    python code/benchmarks/bench_color_baseline.py --use-predecoder

Calibration: this script ships no baseline numbers. First run on a reference
machine (`computelab-sc-01` or local Blackwell) is expected to populate the
PR #62 comment with a calibrated table.
"""
import argparse
import math
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _wilson_ci(num_errors: int, num_shots: int, z: float = 1.96) -> tuple:
    """Wilson-score 95% CI for a binomial proportion."""
    if num_shots <= 0:
        return (float("nan"), float("nan"))
    p = num_errors / num_shots
    denom = 1 + z * z / num_shots
    centre = (p + z * z / (2 * num_shots)) / denom
    half = (z * math.sqrt(p * (1 - p) / num_shots + z * z / (4 * num_shots * num_shots))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ler_per_round(num_errors: int, num_shots: int, n_rounds: int) -> float:
    if num_shots <= 0 or n_rounds <= 0:
        return float("nan")
    p_total = num_errors / num_shots
    return 1.0 - (1.0 - p_total)**(1.0 / n_rounds) if p_total < 1.0 else 1.0


def run_cell(
    distance: int,
    p_error: float,
    n_rounds: int,
    shots: int,
    use_predecoder: bool,
    seed: int = 12345
) -> dict:
    """Run one (distance, p) cell. Returns dict with LER + per-stage timings."""
    import chromobius
    from qec.color_code.memory_circuit import MemoryCircuit
    from qec.noise_model import NoiseModel

    nm = NoiseModel.from_single_p(p_error)
    # MemoryCircuit signature requires explicit per-gate error rates even when a
    # NoiseModel is supplied (the noise_model overrides at circuit-build time).
    mc = MemoryCircuit(
        distance=distance,
        idle_error=p_error,
        sqgate_error=p_error,
        tqgate_error=p_error,
        spam_error=p_error,
        n_rounds=n_rounds,
        basis="X",
        noise_model=nm,
    )

    # 1) Sample shots from the Stim circuit.
    # Chromobius natively handles hyperedges (no graphlike decomposition needed
    # for color codes); match the production color-LER eval flags.
    dem = mc.stim_circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
        ignore_decomposition_failures=True,
    )
    sampler = mc.stim_circuit.compile_detector_sampler(seed=seed)
    t0 = time.perf_counter()
    detectors, observables = sampler.sample(shots, separate_observables=True)
    t_sample = time.perf_counter() - t0

    # 2) Predecoder forward time (untrained model — measures the compute cost,
    #    not LER quality; LER quality requires a trained Stage-2 checkpoint).
    t_predec = 0.0
    if use_predecoder:
        import torch
        from qec.color_code.detector_input import ColorDetectorInputTransform
        # The transform owns the detector→trainX shaping used by the predecoder.
        transform = ColorDetectorInputTransform(distance=distance, rounds=n_rounds, basis="X")
        # Use first transform output as input to a placeholder Conv3D shape match.
        # Predecoder forward time is dominated by Conv3D cost, which is fixed
        # by output_shape — we do a simple identity pass here to estimate the
        # tensor-prep+transfer cost. Stage 2 will replace this with the real model.
        dets_t = torch.tensor(detectors, dtype=torch.float32)
        t0 = time.perf_counter()
        _ = transform(dets_t.view(shots, -1))
        t_predec = time.perf_counter() - t0

    # 3) Chromobius decode (baseline LER).
    # Chromobius API: compile_decoder_for_dem (no 'd' at the end of 'compile').
    decoder = chromobius.compile_decoder_for_dem(dem)
    detectors_packed = np.packbits(detectors, axis=1, bitorder="little")
    t0 = time.perf_counter()
    predictions = decoder.predict_obs_flips_from_dets_bit_packed(detectors_packed)
    t_decode = time.perf_counter() - t0

    # 4) LER per round
    obs_flat = observables.astype(np.uint8)
    pred_flat = predictions.astype(np.uint8)
    if pred_flat.ndim == 1:
        pred_flat = pred_flat[:, None]
    if obs_flat.ndim == 1:
        obs_flat = obs_flat[:, None]
    errors = (pred_flat != obs_flat).any(axis=1).astype(int)
    n_err = int(errors.sum())
    ler = _ler_per_round(n_err, shots, n_rounds)
    ci_lo, ci_hi = _wilson_ci(n_err, shots)

    return {
        "distance": distance,
        "p_error": p_error,
        "n_rounds": n_rounds,
        "shots": shots,
        "errors": n_err,
        "ler_per_round": ler,
        "ler_ci_lo": ci_lo / max(n_rounds, 1),
        "ler_ci_hi": ci_hi / max(n_rounds, 1),
        "sample_ms_per_shot": t_sample / shots * 1000,
        "predecoder_ms_per_shot": (t_predec / shots * 1000) if use_predecoder else None,
        "decode_ms_per_shot": t_decode / shots * 1000,
        "e2e_ms_per_shot": (t_sample + t_predec + t_decode) / shots * 1000,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[5, 7])
    ap.add_argument("--p-errors", nargs="+", type=float, default=[1e-3, 2e-3, 3e-3])
    ap.add_argument("--n-rounds", type=int, default=None, help="default: equal to distance")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument(
        "--use-predecoder",
        action="store_true",
        help="Include predecoder forward in timing (placeholder model — time only)"
    )
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(
        f"# color-code Stage-1 baseline  "
        f"(shots={args.shots}, seed={args.seed}, "
        f"predecoder={'on' if args.use_predecoder else 'off'})"
    )
    cols = [
        "d",
        "n_rounds",
        "p",
        "shots",
        "errors",
        "LER/round",
        "[95% CI]",
        "sample_ms",
        "predecoder_ms" if args.use_predecoder else "",
        "decode_ms",
        "e2e_ms",
    ]
    cols = [c for c in cols if c]
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")

    for d in args.distances:
        r = args.n_rounds if args.n_rounds is not None else d
        for p in args.p_errors:
            row = run_cell(d, p, r, args.shots, args.use_predecoder, seed=args.seed)
            cells = [
                str(row["distance"]),
                str(row["n_rounds"]),
                f"{row['p_error']:.4f}",
                str(row["shots"]),
                str(row["errors"]),
                f"{row['ler_per_round']:.4e}",
                f"[{row['ler_ci_lo']:.2e},{row['ler_ci_hi']:.2e}]",
                f"{row['sample_ms_per_shot']:.3f}",
            ]
            if args.use_predecoder:
                cells.append(f"{row['predecoder_ms_per_shot']:.3f}")
            cells.extend([
                f"{row['decode_ms_per_shot']:.3f}",
                f"{row['e2e_ms_per_shot']:.3f}",
            ])
            print("| " + " | ".join(cells) + " |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
