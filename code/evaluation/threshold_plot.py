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
Compute logical error rate vs physical error rate (threshold plot).

This script loads the best model and computes LER across multiple physical error rates
to generate a threshold plot showing how logical error rate scales with physical error rate.

Usage:
    Set cfg.workflow.task = "inference" in run.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend

# Import the computation functions (Stim-based)
try:
    from evaluation.logical_error_rate import (
        count_logical_errors_with_errorbar as compute_logical_error_rate_stim,
    )
    HAS_LER_MODULE = True
except ImportError:
    HAS_LER_MODULE = False
    compute_logical_error_rate_stim = None


def compute_ler_for_p_range(model, device, dist, cfg, p_values, distance, rank=0, n_rounds=None):
    """
    Compute LER for multiple physical error rates.
    
    Args:
        model: Trained model
        device: Device to run on
        dist: DistributedManager
        cfg: Configuration object
        p_values: List of physical error rates to test
        distance: Code distance to use
        rank: Process rank
        n_rounds: Number of QEC rounds (defaults to distance if not specified)
        
    Returns:
        dict: Results for X and Z bases
            {'p_values': [...], 'X': {...}, 'Z': {...}}
    """
    # Default n_rounds to distance if not specified
    if n_rounds is None:
        n_rounds = distance
    verbose = bool(getattr(cfg.test, "verbose_inference", False)
                  ) or bool(getattr(cfg.test, "verbose", False))
    if not HAS_LER_MODULE:
        if rank == 0:
            print("[Threshold] Error: LER module not available")
        return None
    compute_logical_error_rate = compute_logical_error_rate_stim
    if verbose and rank == 0:
        print("[Inference] Using Stim-based computation")

    results = {
        'p_values': p_values,
        'X': {
            'ler': [],
            'ler_err': [],
            'pymatch_ler': [],
            'pymatch_ler_err': []
        },
        'Z': {
            'ler': [],
            'ler_err': [],
            'pymatch_ler': [],
            'pymatch_ler_err': []
        },
    }

    # Save original error rate, distance, and n_rounds
    original_p = cfg.test.p_error
    original_distance = cfg.distance
    original_n_rounds = cfg.n_rounds

    # Set the distance and n_rounds for this run
    cfg.distance = distance
    cfg.n_rounds = n_rounds

    for p in p_values:
        test_nm_mode = str(getattr(getattr(cfg, "test", None), "noise_model", "train")).lower()
        has_explicit_nm = getattr(cfg.data, "noise_model", None) is not None

        if rank == 0:
            if p is None and test_nm_mode == "train" and has_explicit_nm:
                print(f"\n[Inference] d={distance}, n_rounds={n_rounds}, noise_model=train")
            else:
                print(f"\n[Inference] d={distance}, n_rounds={n_rounds}, p={float(p):.4f}")
            # Column header for the compact summary below
            label_w = 40
            print(f"  {'':<{label_w}}{'No pre-decoder':>15}  {'After pre-decoder':>17}")

        # Update config for this p value (legacy single-p only)
        if p is not None:
            cfg.test.p_error = float(p)

        try:
            result = compute_logical_error_rate(model, device, dist, cfg)

            if isinstance(result, dict) and 'X' in result and 'Z' in result:
                # Extract X basis results
                x_ler = result['X'].get('logical error ratio (mean)')
                x_ler_err = result['X'].get('logical error ratio (standard error)')
                x_pymatch_ler = result['X'].get('logical error ratio (pymatch mean)')
                x_pymatch_ler_err = result['X'].get('logical error ratio (pymatch standard error)')
                x_lat_base = result['X'].get('pymatch latency (baseline µs/round)')
                x_lat_post = result['X'].get('pymatch latency (after predecoder µs/round)')

                # Extract Z basis results
                z_ler = result['Z'].get('logical error ratio (mean)')
                z_ler_err = result['Z'].get('logical error ratio (standard error)')
                z_pymatch_ler = result['Z'].get('logical error ratio (pymatch mean)')
                z_pymatch_ler_err = result['Z'].get('logical error ratio (pymatch standard error)')
                z_lat_base = result['Z'].get('pymatch latency (baseline µs/round)')
                z_lat_post = result['Z'].get('pymatch latency (after predecoder µs/round)')

                if all(
                    v is not None for v in [
                        x_ler, x_ler_err, z_ler, z_ler_err, x_pymatch_ler, x_pymatch_ler_err,
                        z_pymatch_ler, z_pymatch_ler_err
                    ]
                ):
                    results['X']['ler'].append(x_ler)
                    results['X']['ler_err'].append(x_ler_err)
                    results['X']['pymatch_ler'].append(x_pymatch_ler)
                    results['X']['pymatch_ler_err'].append(x_pymatch_ler_err)
                    results['Z']['ler'].append(z_ler)
                    results['Z']['ler_err'].append(z_ler_err)
                    results['Z']['pymatch_ler'].append(z_pymatch_ler)
                    results['Z']['pymatch_ler_err'].append(z_pymatch_ler_err)

                    if rank == 0:

                        def _avg(a, b):
                            vals = [v for v in (a, b) if v is not None and np.isfinite(v)]
                            return float(np.mean(vals)) if vals else float("nan")

                        label_w = 40

                        # Latency (µs/round)
                        x_lat_base_f = float(x_lat_base) if x_lat_base is not None else float("nan")
                        x_lat_post_f = float(x_lat_post) if x_lat_post is not None else float("nan")
                        z_lat_base_f = float(z_lat_base) if z_lat_base is not None else float("nan")
                        z_lat_post_f = float(z_lat_post) if z_lat_post is not None else float("nan")
                        avg_lat_base = _avg(x_lat_base, z_lat_base)
                        avg_lat_post = _avg(x_lat_post, z_lat_post)

                        print(
                            f"  {'PyMatching latency - X basis (µs/round):':<{label_w}}{x_lat_base_f:>15.3f}  {x_lat_post_f:>17.3f}"
                        )
                        print(
                            f"  {'PyMatching latency - Z basis (µs/round):':<{label_w}}{z_lat_base_f:>15.3f}  {z_lat_post_f:>17.3f}"
                        )
                        print(
                            f"  {'PyMatching latency - Avg (µs/round):':<{label_w}}{avg_lat_base:>15.3f}  {avg_lat_post:>17.3f}"
                        )

                        # LER (unitless)
                        avg_ler_base = _avg(x_pymatch_ler, z_pymatch_ler)
                        avg_ler_post = _avg(x_ler, z_ler)
                        print(
                            f"  {'LER - X basis:':<{label_w}}{float(x_pymatch_ler):>15.6f}  {float(x_ler):>17.6f}"
                        )
                        print(
                            f"  {'LER - Z basis:':<{label_w}}{float(z_pymatch_ler):>15.6f}  {float(z_ler):>17.6f}"
                        )
                        print(
                            f"  {'LER - Avg:':<{label_w}}{avg_ler_base:>15.6f}  {avg_ler_post:>17.6f}"
                        )
                else:
                    if rank == 0:
                        print(f"  Warning: Could not extract LER values for p={p}")
                    results['X']['ler'].append(np.nan)
                    results['X']['ler_err'].append(np.nan)
                    results['X']['pymatch_ler'].append(np.nan)
                    results['X']['pymatch_ler_err'].append(np.nan)
                    results['Z']['ler'].append(np.nan)
                    results['Z']['ler_err'].append(np.nan)
                    results['Z']['pymatch_ler'].append(np.nan)
                    results['Z']['pymatch_ler_err'].append(np.nan)
            else:
                if rank == 0:
                    print(f"  Warning: Unexpected result format for p={p}")
                results['X']['ler'].append(np.nan)
                results['X']['ler_err'].append(np.nan)
                results['X']['pymatch_ler'].append(np.nan)
                results['X']['pymatch_ler_err'].append(np.nan)
                results['Z']['ler'].append(np.nan)
                results['Z']['ler_err'].append(np.nan)
                results['Z']['pymatch_ler'].append(np.nan)
                results['Z']['pymatch_ler_err'].append(np.nan)

        except Exception as e:
            if rank == 0:
                print(f"  Error computing LER for p={p}: {e}")
                import traceback
                traceback.print_exc()
            results['X']['ler'].append(np.nan)
            results['X']['ler_err'].append(np.nan)
            results['X']['pymatch_ler'].append(np.nan)
            results['X']['pymatch_ler_err'].append(np.nan)
            results['Z']['ler'].append(np.nan)
            results['Z']['ler_err'].append(np.nan)
            results['Z']['pymatch_ler'].append(np.nan)
            results['Z']['pymatch_ler_err'].append(np.nan)

    # Restore original error rate, distance, and n_rounds
    cfg.test.p_error = original_p
    cfg.distance = original_distance
    cfg.n_rounds = original_n_rounds

    # Convert to numpy arrays
    results['X']['ler'] = np.array(results['X']['ler'])
    results['X']['ler_err'] = np.array(results['X']['ler_err'])
    results['X']['pymatch_ler'] = np.array(results['X']['pymatch_ler'])
    results['X']['pymatch_ler_err'] = np.array(results['X']['pymatch_ler_err'])
    results['Z']['ler'] = np.array(results['Z']['ler'])
    results['Z']['ler_err'] = np.array(results['Z']['ler_err'])
    results['Z']['pymatch_ler'] = np.array(results['Z']['pymatch_ler'])
    results['Z']['pymatch_ler_err'] = np.array(results['Z']['pymatch_ler_err'])
    return results


def create_threshold_plot(all_results, distances, output_path, rank=0):
    """
    Create threshold plots showing both normalized (ratio) and absolute LER vs physical error rate.
    Generates two separate plots:
    1. Normalized: PyMatching/Model ratio
    2. Absolute: Model LER and PyMatching LER in log scale (new plot)
    
    Args:
        all_results: Dictionary mapping distance -> results from compute_ler_for_p_range
        distances: List of distances to plot
        output_path: Path to save the normalized plot (absolute plot will have '_absolute' suffix)
        rank: Process rank
    """
    if rank != 0:
        return  # Only rank 0 creates plots

    # Get p_values from first result (should be the same for all distances)
    p_values = np.array(all_results[distances[0]]['p_values'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Generate colors for each distance
    colors = plt.cm.tab10(np.linspace(0, 1, len(distances)))

    # ===== X-basis subplot =====
    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]

        x_ler = results['X']['ler']
        x_pymatch_ler = results['X']['pymatch_ler']
        # Compute ratio: PyMatching / Model (>1 means we win)
        # Handle divide by zero: if either is 0, return 1
        x_ratio = np.ones_like(x_ler)
        for j in range(len(x_ler)):
            if x_ler[j] > 0 and x_pymatch_ler[j] > 0:
                x_ratio[j] = x_pymatch_ler[j] / x_ler[j]
            # else: keep as 1.0

        # Plot ratio on left axis
        valid_x = ~np.isnan(x_ratio)
        if np.any(valid_x):
            ax1.plot(
                p_values[valid_x],
                x_ratio[valid_x],
                marker='o',
                markersize=10,
                linewidth=2,
                label=f'd={distance} (LER)',
                color=color,
                alpha=0.8
            )

    ax1.set_xlabel('Physical error rate, p', fontsize=14)
    ax1.set_ylabel('PyMatching/Model ratio (>1 is better)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('linear')
    ax1.tick_params(axis='x', labelcolor='black')

    # Set x-axis ticks to match p_values
    ax1.set_xticks(p_values)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(plt.NullFormatter())

    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title('X-basis', fontsize=14)
    ax1.set_ylim([0.5, 2.5])

    ax1.legend(fontsize=9, loc='best')

    # ===== Z-basis subplot =====
    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]

        z_ler = results['Z']['ler']
        z_pymatch_ler = results['Z']['pymatch_ler']
        # Compute ratio: PyMatching / Model (>1 means we win)
        # Handle divide by zero: if either is 0, return 1
        z_ratio = np.ones_like(z_ler)
        for j in range(len(z_ler)):
            if z_ler[j] > 0 and z_pymatch_ler[j] > 0:
                z_ratio[j] = z_pymatch_ler[j] / z_ler[j]
            # else: keep as 1.0

        # Plot ratio on left axis
        valid_z = ~np.isnan(z_ratio)
        if np.any(valid_z):
            ax2.plot(
                p_values[valid_z],
                z_ratio[valid_z],
                marker='o',
                markersize=10,
                linewidth=2,
                label=f'd={distance} (LER)',
                color=color,
                alpha=0.8
            )

    ax2.set_xlabel('Physical error rate, p', fontsize=14)
    ax2.set_ylabel('PyMatching/Model ratio (>1 is better)', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('linear')
    ax2.tick_params(axis='x', labelcolor='black')

    # Set x-axis ticks to match p_values
    ax2.set_xticks(p_values)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(plt.NullFormatter())

    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_title('Z-basis', fontsize=14)
    ax2.set_ylim([0.5, 2.5])

    ax2.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Normalized threshold plot saved to: {output_path}")

    # ========================================================================
    # GENERATE ABSOLUTE LER PLOT (WITHOUT NORMALIZATION)
    # ========================================================================

    fig2, (ax1_abs, ax2_abs) = plt.subplots(1, 2, figsize=(16, 6))

    # ===== X-basis absolute LER subplot =====
    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]

        x_ler = results['X']['ler']
        x_ler_err = results['X']['ler_err']
        x_pymatch_ler = results['X']['pymatch_ler']

        # Plot model LER with error bars (solid line)
        valid_x = ~np.isnan(x_ler)
        if np.any(valid_x):
            ax1_abs.errorbar(
                p_values[valid_x],
                x_ler[valid_x],
                yerr=x_ler_err[valid_x],
                marker='o',
                markersize=7,
                linewidth=2,
                capsize=6,
                capthick=2,
                elinewidth=2,
                label=f'd={distance} (Model)',
                color=color,
                alpha=0.8
            )

        # Plot PyMatching-only baseline (dashed line, no error bars)
        valid_pm = ~np.isnan(x_pymatch_ler)
        if np.any(valid_pm):
            ax1_abs.plot(
                p_values[valid_pm],
                x_pymatch_ler[valid_pm],
                linestyle='--',
                linewidth=2,
                marker='x',
                markersize=5,
                label=f'd={distance} (PyMatch)',
                color=color,
                alpha=0.6
            )

    ax1_abs.set_xlabel('Physical error rate, p', fontsize=14)
    ax1_abs.set_ylabel('Logical Error Rate', fontsize=14)
    ax1_abs.set_xscale('log')
    ax1_abs.set_yscale('log')  # Log scale for small LER values
    ax1_abs.tick_params(axis='both', labelsize=12)

    # Set x-axis ticks to match p_values
    ax1_abs.set_xticks(p_values)
    ax1_abs.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1_abs.get_xaxis().set_minor_formatter(plt.NullFormatter())

    ax1_abs.grid(True, alpha=0.3, which='both')
    ax1_abs.set_title('X-basis (Absolute LER)', fontsize=14)
    ax1_abs.legend(fontsize=8, loc='best', ncol=2)

    # ===== Z-basis absolute LER subplot =====
    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]

        z_ler = results['Z']['ler']
        z_ler_err = results['Z']['ler_err']
        z_pymatch_ler = results['Z']['pymatch_ler']

        # Plot model LER with error bars (solid line)
        valid_z = ~np.isnan(z_ler)
        if np.any(valid_z):
            ax2_abs.errorbar(
                p_values[valid_z],
                z_ler[valid_z],
                yerr=z_ler_err[valid_z],
                marker='o',
                markersize=7,
                linewidth=2,
                capsize=6,
                capthick=2,
                elinewidth=2,
                label=f'd={distance} (Model)',
                color=color,
                alpha=0.8
            )

        # Plot PyMatching-only baseline (dashed line, no error bars)
        valid_pm = ~np.isnan(z_pymatch_ler)
        if np.any(valid_pm):
            ax2_abs.plot(
                p_values[valid_pm],
                z_pymatch_ler[valid_pm],
                linestyle='--',
                linewidth=2,
                marker='x',
                markersize=5,
                label=f'd={distance} (PyMatch)',
                color=color,
                alpha=0.6
            )

    ax2_abs.set_xlabel('Physical error rate, p', fontsize=14)
    ax2_abs.set_ylabel('Logical Error Rate', fontsize=14)
    ax2_abs.set_xscale('log')
    ax2_abs.set_yscale('log')  # Log scale for small LER values
    ax2_abs.tick_params(axis='both', labelsize=12)

    # Set x-axis ticks to match p_values
    ax2_abs.set_xticks(p_values)
    ax2_abs.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2_abs.get_xaxis().set_minor_formatter(plt.NullFormatter())

    ax2_abs.grid(True, alpha=0.3, which='both')
    ax2_abs.set_title('Z-basis (Absolute LER)', fontsize=14)
    ax2_abs.legend(fontsize=8, loc='best', ncol=2)

    plt.tight_layout()

    # Save absolute LER plot with '_absolute' suffix
    output_path_absolute = output_path.replace('.png', '_absolute.png')
    plt.savefig(output_path_absolute, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Absolute LER threshold plot saved to: {output_path_absolute}")


def run_threshold_plot(model, device, dist, cfg):
    """
    Main function to compute threshold plot.
    
    Supports multi-GPU processing: each GPU processes a partition of the samples,
    and statistics are aggregated across all GPUs via all_reduce.
    
    Args:
        model: Trained model
        device: Device to run on
        dist: DistributedManager
        cfg: Configuration object
    """
    import torch

    rank = dist.rank if dist else 0
    world_size = dist.world_size if dist else 1
    verbose = bool(getattr(cfg.test, "verbose_inference", False)
                  ) or bool(getattr(cfg.test, "verbose", False))

    if verbose and rank == 0:
        print("\n" + "=" * 80)
        print("THRESHOLD PLOT COMPUTATION")
        print("=" * 80)

        # Show distributed configuration
        print(f"\n📊 Distributed Configuration:")
        print(f"  World size (GPUs): {world_size}")
        print(f"  Total samples per p (cfg): {cfg.test.num_samples}")
        if world_size > 1:
            print(f"  Samples per GPU: {cfg.test.num_samples // world_size}")

        # Show precomputed frames configuration
        precomputed_frames_dir = getattr(cfg.data, 'precomputed_frames_dir', None)
        print(f"\n🗃️ Frame Data Configuration:")
        if precomputed_frames_dir:
            print(f"  Precomputed frames dir: {precomputed_frames_dir}")
            print(f"  (Will load precomputed data if available, ~100x faster init)")
        else:
            print(f"  Precomputed frames: DISABLED (computing on-the-fly)")
            print(f"  (Set data.precomputed_frames_dir to speed up initialization)")

        # Show PyMatching configuration
        enable_correlated = getattr(cfg.data, 'enable_correlated_pymatching', False)
        print(f"\n🔗 PyMatching Configuration:")
        print(
            f"  Correlated matching: {'ENABLED (two-pass, ~2.7x slower)' if enable_correlated else 'DISABLED (standard, faster)'}"
        )

        # Show sampling configuration
        sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
        print(f"\n🎯 Sampling Configuration:")
        print(f"  Mode: {sampling_mode}")
        if sampling_mode == "temperature":
            temperature = float(getattr(cfg.test, "temperature", 1.0))
            temperature_data = getattr(cfg.test, "temperature_data", None)
            temperature_syn = getattr(cfg.test, "temperature_syn", None)
            temperature_data = float(
                temperature_data
            ) if temperature_data is not None else temperature
            temperature_syn = float(temperature_syn) if temperature_syn is not None else temperature
            print(f"  Temperature (data): {temperature_data}")
            print(f"  Temperature (syndrome): {temperature_syn}")
        else:
            th_data = float(getattr(cfg.test, "th_data", 0.0))
            th_syn = float(getattr(cfg.test, "th_syn", 0.0))
            print(f"  Threshold (data): {th_data}")
            print(f"  Threshold (syndrome): {th_syn}")

    # Public config behavior:
    # `config_public.yaml` does not define a sweep, but hidden defaults include a `threshold:` block.
    # For public inference, we want a single evaluation point (d, n_rounds, p), not a sweep.
    is_public_cfg = hasattr(cfg, "model_id")

    if is_public_cfg:
        d_eff = getattr(cfg.test, "distance", None)
        r_eff = getattr(cfg.test, "n_rounds", None)
        d_eff = int(d_eff) if d_eff is not None else int(cfg.distance)
        r_eff = int(r_eff) if r_eff is not None else int(cfg.n_rounds)
        distances = [d_eff]
        n_rounds_list = [r_eff]
    else:
        # Sweep behavior (legacy / internal-style configs)
        if hasattr(cfg, 'threshold') and hasattr(cfg.threshold, 'distances'):
            distances = list(cfg.threshold.distances)
        else:
            distances = [cfg.distance]

        if hasattr(cfg, 'threshold'
                  ) and hasattr(cfg.threshold, 'n_rounds') and cfg.threshold.n_rounds is not None:
            n_rounds_list = cfg.threshold.n_rounds
            if not isinstance(n_rounds_list, (list, tuple)):
                n_rounds_list = [n_rounds_list] * len(distances)
        else:
            n_rounds_list = distances.copy()

    if verbose and rank == 0:
        print(f"\nTesting {len(distances)} distance(s):")
        for d, r in zip(distances, n_rounds_list):
            print(f"  d={d}, n_rounds={r}")

    # Define p values to test
    if is_public_cfg:
        # Public inference is a single point.
        # If test.noise_model='train' and data.noise_model is provided, p_error is a legacy placeholder:
        # do not sweep/print p-values and do not override cfg.test.p_error.
        test_nm_mode = str(getattr(getattr(cfg, "test", None), "noise_model", "train")).lower()
        has_explicit_nm = getattr(cfg.data, "noise_model", None) is not None
        if test_nm_mode == "train" and has_explicit_nm:
            p_values = [None]
        else:
            p_values = [float(cfg.test.p_error)]
    else:
        if hasattr(cfg, 'threshold') and hasattr(cfg.threshold, 'p_values'):
            p_values = list(cfg.threshold.p_values)
        else:
            p_values = [cfg.test.p_error]

    if verbose and rank == 0:
        print(f"\nTesting {len(p_values)} physical error rates:")
        print(f"  p = {p_values[0]:.4f} to {p_values[-1]:.4f}")
        print(f"  Number of samples per p: {cfg.test.num_samples}")

    # Compute LER for all distances and p values
    all_results = {}
    for i, distance in enumerate(distances):
        n_rounds = n_rounds_list[i] if i < len(n_rounds_list) else distance

        if verbose and rank == 0:
            print(f"\n{'='*60}")
            print(f"Computing for distance d={distance}, n_rounds={n_rounds}")
            print(f"{'='*60}")

        results = compute_ler_for_p_range(
            model, device, dist, cfg, p_values, distance, rank, n_rounds=n_rounds
        )

        if results is None:
            if rank == 0:
                print(f"\n❌ Failed to compute threshold data for d={distance}")
            continue

        all_results[distance] = results

    if not all_results:
        if rank == 0:
            print("\n❌ Failed to compute threshold data for any distance")
        return

    # Synchronize all GPUs before plotting (ensure all computations are complete)
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        if verbose and rank == 0:
            print(f"\n✅ All {world_size} GPUs synchronized, proceeding to plot generation")

    # Create output directory
    if hasattr(cfg, 'threshold') and hasattr(cfg.threshold, 'output_dir'):
        output_dir = cfg.threshold.output_dir
    else:
        output_dir = os.path.join(cfg.output, "plots")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

        # Generate plot filename with metadata
        num_samples = cfg.test.num_samples
        # Format num_samples nicely (e.g., 100k, 1M, 10M)
        if num_samples >= 1_000_000:
            samples_str = f"{num_samples // 1_000_000}M"
        elif num_samples >= 1_000:
            samples_str = f"{num_samples // 1_000}k"
        else:
            samples_str = str(num_samples)

        # Include distance range in filename
        dist_str = f"d{distances[0]}-{distances[-1]}" if len(distances) > 1 else f"d{distances[0]}"
        output_filename = f"threshold_{dist_str}_{samples_str}shots.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n{'='*80}")
        print("GENERATING THRESHOLD PLOT")
        print(f"{'='*80}")

        # Create the plot
        create_threshold_plot(all_results, distances, output_path, rank)

        print(f"\n{'='*80}")
        print("✅ THRESHOLD PLOT COMPLETE")
        print(f"{'='*80}")
        print(f"\nPlot saved to: {output_path}")
        print()

    # Final synchronization to ensure all processes exit cleanly
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
