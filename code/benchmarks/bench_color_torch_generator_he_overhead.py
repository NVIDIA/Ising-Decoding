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
"""Bench: end-to-end ColorQCDataGeneratorTorch with HE on vs off.

Measures ``generate_batch`` wall-clock latency for the production color-code
Torch+cuStabilizer training generator with spacelike HE enabled vs disabled.
Quantifies the per-batch overhead the spacelike HE pipeline adds to training.

Requires a precomputed augmented DEM bundle on disk:

    python -m qec.precompute_dem --code color --distance 9 --rounds 5 \
        --basis X --out /path/to/bundles

(Run for both ``--basis X`` and ``--basis Z`` if ``--meas-basis both``.)

Usage::

    python code/benchmarks/bench_color_torch_generator_he_overhead.py \
        --bundles-dir /path/to/bundles --distances 9 13 --rounds 5 \
        --batches 256 1024 4096
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _percentile(samples, q):
    if not samples:
        return float("nan")
    s = sorted(samples)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _bench_generator(gen, B: int, warmup: int, iters: int):
    import torch
    torch.cuda.synchronize()
    for step in range(warmup):
        gen.generate_batch(step=step, batch_size=B)
    torch.cuda.synchronize()
    times = []
    for step in range(warmup, warmup + iters):
        t0 = time.perf_counter()
        gen.generate_batch(step=step, batch_size=B)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundles-dir", required=True, help="dir containing color_d*_r*_*_*.npz bundles"
    )
    ap.add_argument("--distances", type=int, nargs="+", default=[9, 13])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--meas-basis", choices=("X", "Z", "both"), default="both")
    ap.add_argument("--schedule", default="nearest-neighbor")
    ap.add_argument("--he-iters", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("[bench] CUDA not available; skipping.")
        return 0

    from data.generator_torch_color import ColorQCDataGeneratorTorch

    print(f"# Torch CUDA: {torch.cuda.get_device_name(0)}")
    print(
        f"# bundles_dir={args.bundles_dir}  meas_basis={args.meas_basis}  schedule={args.schedule}"
    )
    print(f"# warmup={args.warmup} iters={args.iters} he_iters={args.he_iters}")
    print(
        f"{'d':>3} {'r':>2} {'B':>6}   {'HE-off median (p10/p90)':<35}  "
        f"{'HE-on median (p10/p90)':<35}  overhead"
    )
    print("-" * 130)

    for d in args.distances:
        for B in args.batches:
            kwargs = dict(
                distance=d,
                n_rounds=args.rounds,
                schedule=args.schedule,
                measure_basis=args.meas_basis,
                precomputed_frames_dir=args.bundles_dir,
                rank=0,
                global_rank=0,
                base_seed=42,
                device=torch.device("cuda"),
                he_max_iterations=args.he_iters,
            )
            gen_off = ColorQCDataGeneratorTorch(apply_spacelike_he=False, **kwargs)
            gen_on = ColorQCDataGeneratorTorch(apply_spacelike_he=True, **kwargs)

            t_off = _bench_generator(gen_off, B, args.warmup, args.iters)
            t_on = _bench_generator(gen_on, B, args.warmup, args.iters)

            mo = statistics.median(t_off)
            mn = statistics.median(t_on)
            overhead = (mn - mo) / mo * 100.0 if mo > 0 else float("nan")
            off_str = (
                f"{mo:8.3f}ms (p10={_percentile(t_off,0.1):7.3f} "
                f"p90={_percentile(t_off,0.9):7.3f})"
            )
            on_str = (
                f"{mn:8.3f}ms (p10={_percentile(t_on,0.1):7.3f} "
                f"p90={_percentile(t_on,0.9):7.3f})"
            )
            print(
                f"{d:>3} {args.rounds:>2} {B:>6}   {off_str:<35}  {on_str:<35}  {overhead:+6.1f}%"
            )

            del gen_off, gen_on
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
