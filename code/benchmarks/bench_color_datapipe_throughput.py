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
Bench: color-code Stim-based datapipe throughput.

Measures samples/sec for `data.datapipe_stim_color.QCDataPipePreDecoder_ColorCode_inference`.
The Stim sampler is CPU-bound; ChromobiusLift can be the bottleneck depending on
distance/n_rounds. Reports per-pull median latency and overall throughput.

Usage:
    python code/benchmarks/bench_color_datapipe_throughput.py
    python code/benchmarks/bench_color_datapipe_throughput.py --distances 5 7 --num-samples 1024
    python code/benchmarks/bench_color_datapipe_throughput.py --batch 32 --workers 0 2 4

Baseline calibration: TBD on computelab-sc-01. Datapipe throughput is the
inference-latency floor for color-code workflows that go through the
ChromobiusLift path.
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distances", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--n-rounds", type=int, default=None, help="default: distance")
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--p", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[0],
        help="DataLoader num_workers values to sweep"
    )
    ap.add_argument("--warmup", type=int, default=2, help="warmup batches")
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
    except Exception as exc:
        print(f"[bench] failed to import datapipe: {exc}")
        return 1

    print(f"# color-code datapipe throughput bench  (p={args.p}, num_samples={args.num_samples})")
    print(f"# warmup={args.warmup}")
    header = (
        f"{'distance':>8}  {'n_rounds':>8}  {'workers':>7}  {'batch':>5}  "
        f"{'med_pull_ms':>11}  {'samples/s':>10}"
    )
    print(header)

    for d in args.distances:
        r = args.n_rounds if args.n_rounds is not None else d
        dataset = QCDataPipePreDecoder_ColorCode_inference(
            distance=d,
            n_rounds=r,
            num_samples=args.num_samples,
            error_mode="circuit_level_color_code",
            p_error=args.p,
            measure_basis="X",
        )
        for w in args.workers:
            loader = DataLoader(dataset, batch_size=args.batch, num_workers=w, shuffle=False)
            # warmup
            for i, _ in enumerate(loader):
                if i + 1 >= args.warmup:
                    break
            times = []
            n_seen = 0
            t_total_start = time.perf_counter()
            for batch in loader:
                t0 = time.perf_counter()
                # Touch tensor to force any lazy materialization.
                _ = batch[next(iter(batch))].shape if isinstance(batch, dict) else batch[0].shape
                times.append(time.perf_counter() - t0)
                # Estimate batch size from any returned tensor
                n_seen += args.batch
            wall = time.perf_counter() - t_total_start
            tput = n_seen / wall if wall > 0 else float("inf")
            print(
                f"{d:>8}  {r:>8}  {w:>7}  {args.batch:>5}  "
                f"{median(times)*1000:>11.3f}  {tput:>10.1f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
