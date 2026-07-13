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
Compare current vs reference Si1000 noise semantics on fixed superdense/CX circuits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.reference_color_baseline import compare_current_vs_reference
from qec.color_code.reference_superdense_noise import (
    build_color_memory_circuit,
    summarize_reference_noise_semantics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Compare current vs reference Si1000 semantics for color-code Chromobius baselines."
    )
    parser.add_argument("--distance", type=int, required=True, help="Code distance")
    parser.add_argument("--n_rounds", type=int, required=True, help="Number of rounds")
    parser.add_argument("--p", type=float, required=True, help="Physical error rate")
    parser.add_argument("--basis", choices=["X", "Z"], default="X", help="Memory basis")
    parser.add_argument("--num_shots", type=int, default=20000, help="Shots for the LER comparison")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for sampling")
    parser.add_argument(
        "--gidney_style_noise_current",
        action="store_true",
        help="Enable gidney_style_noise on the current-semantics backend only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="color_noise_semantics_diff.json",
        help="JSON output path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    current_circ = build_color_memory_circuit(
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        p_error=args.p,
        noise_model_family="si1000",
        noise_instruction_semantics="current",
        gidney_style_noise=bool(args.gidney_style_noise_current),
        add_boundary_detectors=True,
    )
    reference_circ = build_color_memory_circuit(
        distance=args.distance,
        n_rounds=args.n_rounds,
        basis=args.basis,
        p_error=args.p,
        noise_model_family="si1000",
        noise_instruction_semantics="reference",
        gidney_style_noise=False,
        add_boundary_detectors=True,
    )

    payload = {
        "params":
            {
                "distance": args.distance,
                "n_rounds": args.n_rounds,
                "p": args.p,
                "basis": args.basis,
                "num_shots": args.num_shots,
                "batch_size": args.batch_size,
                "gidney_style_noise_current": bool(args.gidney_style_noise_current),
            },
        "instruction_summary":
            {
                "current":
                    summarize_reference_noise_semantics(current_circ.stim_circuit_raw, p=args.p),
                "reference":
                    summarize_reference_noise_semantics(reference_circ.stim_circuit_raw, p=args.p),
            },
        "ler_comparison":
            compare_current_vs_reference(
                distance=args.distance,
                p=args.p,
                n_rounds=args.n_rounds,
                basis=args.basis,
                num_shots=args.num_shots,
                batch_size=args.batch_size,
                gidney_style_noise=bool(args.gidney_style_noise_current),
            ),
    }

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote semantic diff report to {output_path}")
    print(json.dumps(payload["instruction_summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["ler_comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
