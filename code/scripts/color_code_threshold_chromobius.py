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
Standalone threshold plot for superdense color code — Chromobius decoder only (no pre-decoder).

Sweeps over distances and physical error rates, computes LER per round,
and generates a threshold-style plot.

Usage:
    # Default sweep (d=3,5,7,9  p=5e-4,1e-3,2e-3,5e-3  100k shots):
    python code/scripts/color_code_threshold_chromobius.py

    # Custom parameters:
    python code/scripts/color_code_threshold_chromobius.py \
        --distances 3 5 7 9 \
        --p_values 5e-4 1e-3 2e-3 5e-3 \
        --num_shots 100000 \
        --bases X Z \
        --output threshold_color_code.png

    # Quick smoke test:
    python code/scripts/color_code_threshold_chromobius.py \
        --distances 3 5 --p_values 1e-3 5e-3 --num_shots 10000
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import chromobius
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add repo code/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec.color_code.reference_superdense_noise import (
    PAPER_SUPERDENSE_SI1000_ORACLE,
)
from evaluation.reference_color_baseline import (
    compare_results_to_paper,
    compute_chromobius_ler,
)
from qec.noise_model import (
    normalize_noise_instruction_semantics,
    normalize_noise_mode,
    normalize_noise_model_family,
)

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_ler_chromobius(
    distance: int,
    p: float,
    n_rounds: int,
    basis: str,
    num_shots: int,
    batch_size: int = 100_000,
    noise_model_family: str = "legacy",
    noise_instruction_semantics: str = "current",
    gidney_style_noise: bool = False,
) -> dict:
    """
    Compute LER (and LER per round) for one (d, p, basis) configuration.

    Uses Stim to build a superdense color-code memory circuit, samples
    detector + observable data, and decodes with Chromobius.

    Args:
        distance:   Code distance (odd, >= 3).
        p:          Physical error rate (circuit-level depolarising).
        n_rounds:   Number of QEC rounds (including prep + final meas).
        basis:      Measurement basis, 'X' or 'Z'.
        num_shots:  Total number of shots.
        batch_size: Shots per sampling batch (memory control).

    Returns:
        dict with keys: ler_per_round, ler_total, stderr, num_errors, num_shots.
    """
    row = compute_chromobius_ler(
        distance=distance,
        p=p,
        n_rounds=n_rounds,
        basis=basis,
        num_shots=num_shots,
        batch_size=batch_size,
        noise_model_family=noise_model_family,
        noise_instruction_semantics=noise_instruction_semantics,
        gidney_style_noise=gidney_style_noise,
    )
    return {
        'ler_per_round': row['ler_per_round'],
        'ler_total': row['ler_total'],
        'stderr': row['stderr'],
        'num_errors': row['num_errors'],
        'num_shots': row['num_shots'],
    }


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


def run_sweep(
    distances,
    p_values,
    num_shots,
    bases,
    n_rounds_list=None,
    batch_size: int = 100_000,
    *,
    noise_model_family: str = "legacy",
    noise_instruction_semantics: str = "current",
    gidney_style_noise: bool = False,
):
    """
    Run the full (d, p, basis) sweep and return a results table.

    Args:
        distances:    List of code distances.
        p_values:     List of physical error rates.
        num_shots:    Shots per configuration.
        bases:        List of bases, e.g. ['X', 'Z'].
        n_rounds_list: Rounds per distance (default: n_rounds = 4*d).

    Returns:
        list of dicts, one per (d, p, basis).
    """
    if n_rounds_list is None:
        n_rounds_list = [4 * d for d in distances]

    results = []
    total_configs = len(distances) * len(p_values) * len(bases)
    done = 0

    for d, n_r in zip(distances, n_rounds_list):
        for p in p_values:
            for basis in bases:
                done += 1
                tag = f"[{done}/{total_configs}] d={d}, r={n_r}, p={p:.1e}, basis={basis}"
                print(f"\n{tag}")
                t0 = time.time()

                res = compute_ler_chromobius(
                    d,
                    p,
                    n_r,
                    basis,
                    num_shots,
                    batch_size=batch_size,
                    noise_model_family=noise_model_family,
                    noise_instruction_semantics=noise_instruction_semantics,
                    gidney_style_noise=gidney_style_noise,
                )

                elapsed = time.time() - t0
                print(
                    f"  LER/round = {res['ler_per_round']:.4e}  "
                    f"(errors={res['num_errors']}/{res['num_shots']})  "
                    f"[{elapsed:.1f}s]"
                )

                results.append(
                    {
                        'distance':
                            d,
                        'n_rounds':
                            n_r,
                        'p':
                            p,
                        'basis':
                            basis,
                        'noise_model_family':
                            normalize_noise_model_family(
                                noise_model_family,
                                fallback_noise_mode=noise_model_family,
                            ),
                        'noise_instruction_semantics':
                            normalize_noise_instruction_semantics(noise_instruction_semantics),
                        'gidney_style_noise':
                            bool(gidney_style_noise),
                        'ler_per_round':
                            res['ler_per_round'],
                        'ler_total':
                            res['ler_total'],
                        'stderr':
                            res['stderr'],
                        'num_errors':
                            res['num_errors'],
                        'num_shots':
                            res['num_shots'],
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MARKERS = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']


def make_threshold_plot(results, output_path, bases):
    """
    Generate threshold-style plot(s): LER/round vs p, one curve per d.

    Layout:
    - If both X and Z bases: three panels (X, Z, average).
    - If single basis: one panel.
    """
    distances = sorted(set(r['distance'] for r in results))
    p_values = sorted(set(r['p'] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(distances)))

    n_panels = len(bases) + (1 if len(bases) == 2 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5), squeeze=False)
    axes = axes.ravel()

    panel_labels = list(bases) + (['avg(X,Z)'] if len(bases) == 2 else [])

    for panel_idx, label in enumerate(panel_labels):
        ax = axes[panel_idx]

        for di, d in enumerate(distances):
            ler_arr = []
            se_arr = []

            for p in p_values:
                if label.startswith('avg'):
                    # Average X and Z
                    rows = [r for r in results if r['distance'] == d and r['p'] == p]
                    ler_vals = [r['ler_per_round'] for r in rows]
                    se_vals = [r['stderr'] for r in rows]
                    ler_arr.append(np.mean(ler_vals))
                    # Propagate SE: SE_avg = sqrt(sum(se^2)) / N
                    se_arr.append(np.sqrt(np.sum(np.array(se_vals)**2)) / len(se_vals))
                else:
                    row = [
                        r for r in results
                        if r['distance'] == d and r['p'] == p and r['basis'] == label
                    ]
                    if row:
                        ler_arr.append(row[0]['ler_per_round'])
                        se_arr.append(row[0]['stderr'])
                    else:
                        ler_arr.append(np.nan)
                        se_arr.append(np.nan)

            ler_arr = np.array(ler_arr)
            se_arr = np.array(se_arr)
            valid = ~np.isnan(ler_arr) & (ler_arr > 0)

            ax.errorbar(
                np.array(p_values)[valid],
                ler_arr[valid],
                yerr=se_arr[valid],
                marker=MARKERS[di % len(MARKERS)],
                markersize=7,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                label=f'd={d}',
                color=colors[di],
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical error rate  $p$', fontsize=13)
        ax.set_ylabel('Logical error rate / round', fontsize=13)
        ax.set_title(
            f'{label}-basis' if not label.startswith('avg') else 'Average (X+Z)/2', fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.25)
        ax.tick_params(labelsize=11)

    fig.suptitle('Superdense color code — Chromobius only', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_csv(results, csv_path):
    """Write results table to CSV for later re-plotting."""
    fieldnames = [
        'distance', 'n_rounds', 'p', 'basis', 'noise_model_family', 'noise_instruction_semantics',
        'gidney_style_noise', 'ler_per_round', 'ler_total', 'stderr', 'num_errors', 'num_shots'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV  saved to {csv_path}")


def save_results_json(results, json_path):
    """Write the raw sweep table to JSON."""
    payload = {
        "results": results,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"JSON saved to {json_path}")


def build_paper_comparison(results):
    """
    Compare X-memory sweep results against the frozen local PGF oracle.
    """
    return compare_results_to_paper(results, series="chromobius")


def save_paper_comparison(results, comparison_path):
    payload = build_paper_comparison(results)
    with open(comparison_path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Paper comparison JSON saved to {comparison_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description='Superdense color-code threshold plot (Chromobius only)'
    )
    parser.add_argument(
        '--distances',
        type=int,
        nargs='+',
        default=[3, 5, 7, 9],
        help='Code distances (default: 3 5 7 9)'
    )
    parser.add_argument(
        '--p_values',
        type=float,
        nargs='+',
        default=[5e-4, 1e-3, 2e-3, 5e-3],
        help='Physical error rates (default: 5e-4 1e-3 2e-3 5e-3)'
    )
    parser.add_argument(
        '--num_shots',
        type=int,
        default=100_000,
        help='Shots per (d, p, basis) configuration (default: 100000)'
    )
    parser.add_argument(
        '--bases',
        nargs='+',
        default=['X', 'Z'],
        choices=['X', 'Z'],
        help='Measurement bases (default: X Z)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='threshold_color_code_chromobius.png',
        help='Output plot filename (default: threshold_color_code_chromobius.png)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Optional CSV output path (default: same stem as --output with .csv)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100_000, help='Sampling batch size (default: 100000)'
    )
    parser.add_argument(
        '--noise_model_family',
        type=str,
        default=None,
        help="Noise model family: 'legacy' or 'si1000' (preferred new axis)"
    )
    parser.add_argument(
        '--noise_instruction_semantics',
        type=str,
        default='current',
        help="Noise instruction semantics: 'current' or 'reference'"
    )
    parser.add_argument(
        '--noise_mode',
        type=str,
        default='legacy',
        help="Backward-compatible alias for noise_model_family"
    )
    parser.add_argument(
        '--gidney_style_noise',
        action='store_true',
        help='Use Gidney-style superdense noise structure (separate from noise_mode)'
    )
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Optional JSON output path (default: same stem as --output with .json)'
    )
    parser.add_argument(
        '--paper_comparison_json',
        type=str,
        default=None,
        help='Optional paper-vs-ours comparison JSON path'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    noise_model_family = normalize_noise_model_family(
        args.noise_model_family,
        fallback_noise_mode=args.noise_mode,
    )
    noise_instruction_semantics = normalize_noise_instruction_semantics(
        args.noise_instruction_semantics
    )

    print("=" * 70)
    print("  Superdense color code threshold — Chromobius only")
    print("=" * 70)
    n_rounds = [4 * d for d in args.distances]
    print(f"  distances   : {args.distances}")
    print(f"  n_rounds    : {n_rounds}  (= 4*d)")
    print(f"  p_values    : {args.p_values}")
    print(f"  bases       : {args.bases}")
    print(f"  num_shots   : {args.num_shots:,}")
    print(f"  noise_family: {noise_model_family}")
    print(f"  semantics   : {noise_instruction_semantics}")
    print(f"  noise_mode  : {normalize_noise_mode(args.noise_mode)}  (compat)")
    print(f"  gidney_style: {bool(args.gidney_style_noise)}")
    print(f"  output      : {args.output}")
    print("=" * 70)

    t_start = time.time()
    results = run_sweep(
        args.distances,
        args.p_values,
        args.num_shots,
        args.bases,
        batch_size=args.batch_size,
        noise_model_family=noise_model_family,
        noise_instruction_semantics=noise_instruction_semantics,
        gidney_style_noise=args.gidney_style_noise,
    )
    t_total = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"  Sweep complete in {t_total:.1f}s")
    print(f"{'=' * 70}")

    # --- Summary table ---
    print(
        f"\n{'d':>3s} {'r':>3s} {'p':>9s} {'basis':>5s} "
        f"{'LER/round':>12s} {'stderr':>12s} {'errors':>8s} {'shots':>8s}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['distance']:3d} {r['n_rounds']:3d} {r['p']:9.1e} {r['basis']:>5s} "
            f"{r['ler_per_round']:12.4e} {r['stderr']:12.4e} "
            f"{r['num_errors']:8d} {r['num_shots']:8d}"
        )

    # --- Save CSV ---
    csv_path = args.csv or str(Path(args.output).with_suffix('.csv'))
    save_csv(results, csv_path)
    json_path = args.json or str(Path(args.output).with_suffix('.json'))
    save_results_json(results, json_path)
    if noise_model_family == "si1000":
        comparison_path = (
            args.paper_comparison_json or
            str(Path(args.output).with_suffix('.paper-comparison.json'))
        )
        save_paper_comparison(results, comparison_path)

    # --- Plot ---
    make_threshold_plot(results, args.output, args.bases)


if __name__ == '__main__':
    main()
