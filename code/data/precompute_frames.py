#!/usr/bin/env python3
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
Precompute DEM bundles (H, p, A) for MemoryCircuitTorch.

Usage:
    # Precompute for a single configuration
    python precompute_frames.py --distance 13 --n_rounds 13 --basis X --rotation O1

    # Precompute for multiple configurations
    python precompute_frames.py --distance 5 9 13 --basis X Z --rotation O1

    # Specify output directory (default: ../frames_data/)
    python precompute_frames.py --distance 13 --output_dir /path/to/frames

Outputs (in --dem_output_dir / --output_dir):
  - surface_d{d}_r{r}_{basis}_frame_predecoder.X.npz
  - surface_d{d}_r{r}_{basis}_frame_predecoder.Z.npz
  - surface_d{d}_r{r}_{basis}_frame_predecoder.p.npz
  - surface_d{d}_r{r}_{basis}_frame_predecoder.A.npz
"""

import argparse
from pathlib import Path
import sys
import time

import torch

# Add code root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.precompute_dem import precompute_dem_bundle_surface_code


def _normalize_rotation(rotation: str) -> str:
    rotation = str(rotation).upper()
    public_to_internal = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}
    internal_to_public = {v: k for k, v in public_to_internal.items()}
    internal_rot = public_to_internal.get(rotation, rotation)
    if internal_rot not in internal_to_public:
        raise ValueError(f"Invalid rotation={rotation!r}. Use O1..O4 (preferred) or XV/XH/ZV/ZH.")
    return internal_rot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute DEM bundles for MemoryCircuitTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=int,
        nargs="+",
        required=True,
        help="Code distance(s) to precompute (e.g., 5 9 13 17 21)",
    )
    parser.add_argument(
        "--n_rounds",
        "-r",
        type=int,
        nargs="+",
        default=None,
        help="Number of rounds (default: same as distance)",
    )
    parser.add_argument(
        "--basis",
        "-b",
        type=str,
        nargs="+",
        default=["X", "Z"],
        choices=["X", "Z"],
        help="Measurement basis (default: X Z)",
    )
    parser.add_argument(
        "--rotation",
        "--rot",
        type=str,
        default="O1",
        choices=["O1", "O2", "O3", "O4", "XV", "XH", "ZV", "ZH"],
        help="Surface code rotation/orientation (default: O1). Public names: O1..O4.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.01,
        help="Scalar p for exporting single-p marginals",
    )
    parser.add_argument(
        "--dem_output_dir",
        type=str,
        default=None,
        help="Output directory (default: ../frames_data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Deprecated alias for --dem_output_dir",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cuda, cuda:0, cpu (default: auto)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if args.dem_output_dir is None:
        args.dem_output_dir = args.output_dir
    if args.dem_output_dir is None:
        args.dem_output_dir = str(Path(__file__).parent.parent / "frames_data")

    if args.n_rounds is None:
        args.n_rounds = args.distance
    if len(args.n_rounds) == 1 and len(args.distance) > 1:
        args.n_rounds = args.n_rounds * len(args.distance)
    if len(args.n_rounds) != len(args.distance):
        print(
            "Error: Number of --n_rounds values must match --distance values (or be a single value)"
        )
        sys.exit(1)

    verbose = not args.quiet
    device = (
        torch.device(args.device) if args.device is not None else
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    internal_rot = _normalize_rotation(args.rotation)

    if verbose:
        print(f"\n{'#' * 60}")
        print(f"# DEM Precompute")
        print(f"# Distances: {args.distance}")
        print(f"# Rounds: {args.n_rounds}")
        print(f"# Bases: {args.basis}")
        print(f"# Rotation: {args.rotation} (internal={internal_rot})")
        print(f"# Output: {args.dem_output_dir}")
        print(f"{'#' * 60}")

    total_t0 = time.time()
    output_dirs = set()

    for d, r in zip(args.distance, args.n_rounds):
        for basis in args.basis:
            try:
                dem_dir = precompute_dem_bundle_surface_code(
                    distance=int(d),
                    n_rounds=int(r),
                    basis=str(basis),
                    code_rotation=internal_rot,
                    p_scalar=float(args.p),
                    dem_output_dir=str(args.dem_output_dir),
                    device=device,
                    export=True,
                )
                output_dirs.add(str(dem_dir))
            except Exception as e:
                print(
                    f"\n✗ Error precomputing d={d}, r={r}, basis={basis}, rotation={args.rotation}: {e}"
                )
                import traceback
                traceback.print_exc()

    total_elapsed = time.time() - total_t0
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"All done! Total time: {total_elapsed:.1f}s")
        print("Output directories:")
        for d in sorted(output_dirs):
            print(f"  - {d}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
