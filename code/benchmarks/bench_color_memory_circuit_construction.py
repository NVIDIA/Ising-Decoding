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
Bench: color-code MemoryCircuit construction time across distances.

Measures wall-clock time to construct the Stim-based color-code MemoryCircuit
for a range of distances. The Stim path is CPU-bound; numbers are stable on any
machine. Reported as median over `--reps` reps after `--warmup` discarded reps.

Usage:
    python code/benchmarks/bench_color_memory_circuit_construction.py
    python code/benchmarks/bench_color_memory_circuit_construction.py --distances 5 7 9 11 13 --reps 10

Baseline calibration: TBD on computelab-sc-01. First run prints raw numbers;
populate this docstring with the per-distance median once a reference exists.
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.color_code.memory_circuit import MemoryCircuit
from qec.noise_model import NoiseModel


def time_construction(distance: int, n_rounds: int, p: float, reps: int, warmup: int) -> dict:
    """Return median + min/max construction time (seconds) for one (d, r, p)."""
    noise_model = NoiseModel.from_single_p(p)
    samples = []
    for _ in range(warmup):
        MemoryCircuit(distance=distance, n_rounds=n_rounds, basis="X", noise_model=noise_model)
    for _ in range(reps):
        t0 = time.perf_counter()
        MemoryCircuit(distance=distance, n_rounds=n_rounds, basis="X", noise_model=noise_model)
        samples.append(time.perf_counter() - t0)
    return {
        "distance": distance,
        "n_rounds": n_rounds,
        "median_s": median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "samples": len(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=[5, 7, 9, 11, 13],
        help="Distances to benchmark."
    )
    ap.add_argument(
        "--n-rounds", type=int, default=None, help="n_rounds (default: each distance's own value)."
    )
    ap.add_argument("--p", type=float, default=1e-3, help="Physical error rate.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    print(f"# color-code MemoryCircuit construction bench  (p={args.p})")
    print(f"# warmup={args.warmup} reps={args.reps}")
    print(f"{'distance':>8}  {'n_rounds':>8}  {'median_ms':>10}  {'min_ms':>9}  {'max_ms':>9}")
    for d in args.distances:
        r = args.n_rounds if args.n_rounds is not None else d
        out = time_construction(d, r, args.p, args.reps, args.warmup)
        print(
            f"{out['distance']:>8}  {out['n_rounds']:>8}  "
            f"{out['median_s']*1000:>10.2f}  "
            f"{out['min_s']*1000:>9.2f}  "
            f"{out['max_s']*1000:>9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
