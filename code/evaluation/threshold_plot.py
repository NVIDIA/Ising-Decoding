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
    Set cfg.workflow.task = "threshold" in run.py
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
        compute_syndrome_density_reduction as compute_syndrome_density_reduction_stim,
        get_global_decoder_config,
    )
    HAS_LER_MODULE = True
except ImportError:
    HAS_LER_MODULE = False
    compute_logical_error_rate_stim = None
    compute_syndrome_density_reduction_stim = None
    get_global_decoder_config = None


def _empty_basis_results():
    return {
        "ler": [],
        "ler_err": [],
        "baseline_ler": [],
        "baseline_ler_err": [],
        "pymatch_ler": [],
        "pymatch_ler_err": [],
        "speedup": [],
        "baseline_single_shot_us_per_round": [],
        "posterior_single_shot_us_per_round": [],
    }


def _empty_threshold_results(decoder_labels):
    return {
        "p_values": [],
        "decoder_labels": dict(decoder_labels),
        "X": _empty_basis_results(),
        "Z": _empty_basis_results(),
        "syndrome_density": {
            "X": [],
            "Z": []
        },
    }


def _append_nan_result_point(results):
    for basis in ("X", "Z"):
        for key in (
            "ler",
            "ler_err",
            "baseline_ler",
            "baseline_ler_err",
            "pymatch_ler",
            "pymatch_ler_err",
            "speedup",
            "baseline_single_shot_us_per_round",
            "posterior_single_shot_us_per_round",
        ):
            results[basis][key].append(np.nan)
        results["syndrome_density"][basis].append(np.nan)


def compute_ler_for_p_range(model, device, dist, cfg, p_values, distance, rank=0, n_rounds=None):
    """
    Compute LER and syndrome density for multiple physical error rates.
    
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
            {
                'p_values': [...],
                'X': {'ler': [...], 'ler_err': [...]},
                'Z': {'ler': [...], 'ler_err': [...]},
                'syndrome_density': {'X': [...], 'Z': [...]}
            }
    """
    # Default n_rounds to distance if not specified
    if n_rounds is None:
        n_rounds = distance

    decoder_config = get_global_decoder_config(cfg) if get_global_decoder_config is not None else {
        "baseline_name": "uncorr_pm",
        "posterior_name": "uncorr_pm",
        "baseline_label": "Uncorr PM",
        "posterior_label": "Uncorr PM",
    }
    decoder_labels = {
        "baseline_name": decoder_config["baseline_name"],
        "posterior_name": decoder_config["posterior_name"],
        "baseline_label": decoder_config["baseline_label"],
        "posterior_label": decoder_config["posterior_label"],
    }

    if not HAS_LER_MODULE:
        if rank == 0:
            print("[Threshold] Error: LER module not available")
        return None
    compute_logical_error_rate = compute_logical_error_rate_stim
    compute_syndrome_density_reduction = compute_syndrome_density_reduction_stim
    if rank == 0:
        print("[Threshold] Using Stim-based computation (TRUE PARALLEL)")

    results = _empty_threshold_results(decoder_labels)
    results["p_values"] = p_values
    results["distance"] = int(distance)
    results["n_rounds"] = int(n_rounds)

    # Save original error rate, distance, and n_rounds
    original_p = cfg.test.p_error
    original_distance = cfg.distance
    original_n_rounds = cfg.n_rounds

    # Set the distance and n_rounds for this run
    cfg.distance = distance
    cfg.n_rounds = n_rounds

    # Store original num_samples for restoration after low-p points
    original_num_samples = cfg.test.num_samples

    for p in p_values:
        # For d >= 13 and p <= 0.002, multiply samples by 8x for better statistics
        # (errors are rare at low p, need more samples to get reliable LER estimates)
        if distance >= 13 and p <= 0.002:
            cfg.test.num_samples = original_num_samples * 8
            samples_multiplier = 8
        else:
            cfg.test.num_samples = original_num_samples
            samples_multiplier = 1

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Testing d={distance}, n_rounds={distance}, p={p:.4f}")
            if samples_multiplier > 1:
                print(f"  [Low-p boost: {samples_multiplier}x samples = {cfg.test.num_samples}]")
            print(f"{'='*60}")

        # Update config for this p value
        cfg.test.p_error = p

        try:
            result = compute_logical_error_rate(model, device, dist, cfg)

            if isinstance(result, dict) and 'X' in result and 'Z' in result:
                # Extract X basis results
                x_ler = result['X'].get('logical error ratio (mean)')
                x_ler_err = result['X'].get('logical error ratio (standard error)')
                x_baseline_ler = (
                    result['X'].get('logical error ratio (baseline mean)') or
                    result['X'].get('logical error ratio (pymatch mean)')
                )
                x_baseline_ler_err = (
                    result['X'].get('logical error ratio (baseline standard error)') or
                    result['X'].get('logical error ratio (pymatch standard error)')
                )
                x_speedup = result['X'].get('decode speedup (single-shot)')
                x_baseline_single_shot = result['X'].get(
                    'baseline decode time (single-shot us/round)'
                )
                x_posterior_single_shot = result['X'].get(
                    'posterior decode time (single-shot us/round)'
                )

                # Extract Z basis results
                z_ler = result['Z'].get('logical error ratio (mean)')
                z_ler_err = result['Z'].get('logical error ratio (standard error)')
                z_baseline_ler = (
                    result['Z'].get('logical error ratio (baseline mean)') or
                    result['Z'].get('logical error ratio (pymatch mean)')
                )
                z_baseline_ler_err = (
                    result['Z'].get('logical error ratio (baseline standard error)') or
                    result['Z'].get('logical error ratio (pymatch standard error)')
                )
                z_speedup = result['Z'].get('decode speedup (single-shot)')
                z_baseline_single_shot = result['Z'].get(
                    'baseline decode time (single-shot us/round)'
                )
                z_posterior_single_shot = result['Z'].get(
                    'posterior decode time (single-shot us/round)'
                )

                if all(
                    v is not None for v in [
                        x_ler,
                        x_ler_err,
                        z_ler,
                        z_ler_err,
                        x_baseline_ler,
                        x_baseline_ler_err,
                        z_baseline_ler,
                        z_baseline_ler_err,
                    ]
                ):
                    results['X']['ler'].append(x_ler)
                    results['X']['ler_err'].append(x_ler_err)
                    results['X']['baseline_ler'].append(x_baseline_ler)
                    results['X']['baseline_ler_err'].append(x_baseline_ler_err)
                    results['X']['pymatch_ler'].append(x_baseline_ler)
                    results['X']['pymatch_ler_err'].append(x_baseline_ler_err)
                    results['X']['speedup'].append(x_speedup if x_speedup is not None else np.nan)
                    results['X']['baseline_single_shot_us_per_round'].append(
                        x_baseline_single_shot if x_baseline_single_shot is not None else np.nan
                    )
                    results['X']['posterior_single_shot_us_per_round'].append(
                        x_posterior_single_shot if x_posterior_single_shot is not None else np.nan
                    )
                    results['Z']['ler'].append(z_ler)
                    results['Z']['ler_err'].append(z_ler_err)
                    results['Z']['baseline_ler'].append(z_baseline_ler)
                    results['Z']['baseline_ler_err'].append(z_baseline_ler_err)
                    results['Z']['pymatch_ler'].append(z_baseline_ler)
                    results['Z']['pymatch_ler_err'].append(z_baseline_ler_err)
                    results['Z']['speedup'].append(z_speedup if z_speedup is not None else np.nan)
                    results['Z']['baseline_single_shot_us_per_round'].append(
                        z_baseline_single_shot if z_baseline_single_shot is not None else np.nan
                    )
                    results['Z']['posterior_single_shot_us_per_round'].append(
                        z_posterior_single_shot if z_posterior_single_shot is not None else np.nan
                    )

                    if rank == 0:
                        print(f"  X-basis LER: {x_ler:.6f} ± {x_ler_err:.6f}")
                        print(f"  Z-basis LER: {z_ler:.6f} ± {z_ler_err:.6f}")
                        print(
                            f"  X-basis baseline LER ({decoder_labels['baseline_label']}): "
                            f"{x_baseline_ler:.6f} ± {x_baseline_ler_err:.6f}"
                        )
                        print(
                            f"  Z-basis baseline LER ({decoder_labels['baseline_label']}): "
                            f"{z_baseline_ler:.6f} ± {z_baseline_ler_err:.6f}"
                        )
                        if x_speedup is not None:
                            print(f"  X-basis single-shot speedup: {x_speedup:.4f}x")
                        if z_speedup is not None:
                            print(f"  Z-basis single-shot speedup: {z_speedup:.4f}x")

                    try:
                        syn_result = compute_syndrome_density_reduction(model, device, dist, cfg)

                        # The result directly contains the reduction factors (no 'stim' wrapper)
                        if isinstance(syn_result, dict):
                            syn_x = syn_result.get('reduction factor (X)')
                            syn_z = syn_result.get('reduction factor (Z)')

                            if syn_x is not None and syn_z is not None:
                                results['syndrome_density']['X'].append(syn_x)
                                results['syndrome_density']['Z'].append(syn_z)
                                if rank == 0:
                                    print(
                                        f"  Syndrome density reduction - X: {syn_x:.4f}, Z: {syn_z:.4f}"
                                    )
                            else:
                                results['syndrome_density']['X'].append(np.nan)
                                results['syndrome_density']['Z'].append(np.nan)
                        else:
                            if rank == 0:
                                print(f"  Warning: Unexpected syndrome result format")
                            results['syndrome_density']['X'].append(np.nan)
                            results['syndrome_density']['Z'].append(np.nan)
                    except Exception as syn_e:
                        if rank == 0:
                            print(
                                f"  Warning: Could not compute syndrome density for p={p}: {syn_e}"
                            )
                            import traceback
                            traceback.print_exc()
                        results['syndrome_density']['X'].append(np.nan)
                        results['syndrome_density']['Z'].append(np.nan)
                else:
                    if rank == 0:
                        print(f"  Warning: Could not extract LER values for p={p}")
                    _append_nan_result_point(results)
            else:
                if rank == 0:
                    print(f"  Warning: Unexpected result format for p={p}")
                _append_nan_result_point(results)

        except Exception as e:
            if rank == 0:
                print(f"  Error computing LER for p={p}: {e}")
                import traceback
                traceback.print_exc()
            _append_nan_result_point(results)

    # Restore original error rate, distance, n_rounds, and num_samples
    cfg.test.p_error = original_p
    cfg.distance = original_distance
    cfg.n_rounds = original_n_rounds
    cfg.test.num_samples = original_num_samples

    # Convert to numpy arrays
    results['p_values'] = np.array(results['p_values'])
    for basis in ("X", "Z"):
        for key in results[basis]:
            results[basis][key] = np.array(results[basis][key])
    results['syndrome_density']['X'] = np.array(results['syndrome_density']['X'])
    results['syndrome_density']['Z'] = np.array(results['syndrome_density']['Z'])

    return results


def create_threshold_plot(all_results, distances, output_path, rank=0):
    """
    Create threshold plots for explicit baseline/posterior decoder comparisons.
    """
    if rank != 0:
        return

    first_result = all_results[distances[0]]
    p_values = np.array(first_result["p_values"])
    decoder_labels = first_result.get("decoder_labels", {})
    baseline_label = decoder_labels.get("baseline_label", "Baseline decoder")
    posterior_label = decoder_labels.get("posterior_label", "Posterior decoder")
    posterior_display = f"PD + {posterior_label}"
    any_positive_x = any(
        np.any(np.asarray(all_results[d]["X"]["ler"]) > 0) or
        np.any(np.asarray(all_results[d]["X"]["baseline_ler"]) > 0) for d in distances
    )
    any_positive_z = any(
        np.any(np.asarray(all_results[d]["Z"]["ler"]) > 0) or
        np.any(np.asarray(all_results[d]["Z"]["baseline_ler"]) > 0) for d in distances
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(distances)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        x_ler = results["X"]["ler"]
        x_baseline_ler = results["X"]["baseline_ler"]
        x_ratio = np.ones_like(x_ler)
        for j in range(len(x_ler)):
            if x_ler[j] > 0 and x_baseline_ler[j] > 0:
                x_ratio[j] = x_baseline_ler[j] / x_ler[j]
        valid_x = ~np.isnan(x_ratio)
        if np.any(valid_x):
            ax1.plot(
                p_values[valid_x],
                x_ratio[valid_x],
                marker="o",
                markersize=10,
                linewidth=2,
                label=f"d={distance} (LER ratio)",
                color=color,
                alpha=0.8,
            )

    ax1.set_xlabel("Physical error rate, p", fontsize=14)
    ax1.set_ylabel(
        f"{baseline_label} / ({posterior_display}) LER ratio (>1 is better)", fontsize=14
    )
    ax1.set_xscale("log")
    ax1.set_yscale("linear")
    ax1.tick_params(axis="x", labelcolor="black")
    ax1.set_xticks(p_values)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("X-basis", fontsize=14)
    ax1.set_ylim([0.5, 2.5])

    ax1_right = ax1.twinx()
    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        x_syn = results["syndrome_density"]["X"]
        valid_x_syn = ~np.isnan(x_syn)
        if np.any(valid_x_syn):
            ax1_right.plot(
                p_values[valid_x_syn],
                x_syn[valid_x_syn],
                marker="s",
                markersize=6,
                linewidth=1.5,
                linestyle="--",
                color=color,
                alpha=0.5,
                label=f"d={distance} (Syn. density)",
            )
    ax1_right.set_ylabel("Syndrome density reduction", fontsize=14)
    ax1_right.set_yscale("linear")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")

    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        z_ler = results["Z"]["ler"]
        z_baseline_ler = results["Z"]["baseline_ler"]
        z_ratio = np.ones_like(z_ler)
        for j in range(len(z_ler)):
            if z_ler[j] > 0 and z_baseline_ler[j] > 0:
                z_ratio[j] = z_baseline_ler[j] / z_ler[j]
        valid_z = ~np.isnan(z_ratio)
        if np.any(valid_z):
            ax2.plot(
                p_values[valid_z],
                z_ratio[valid_z],
                marker="o",
                markersize=10,
                linewidth=2,
                label=f"d={distance} (LER ratio)",
                color=color,
                alpha=0.8,
            )

    ax2.set_xlabel("Physical error rate, p", fontsize=14)
    ax2.set_ylabel(
        f"{baseline_label} / ({posterior_display}) LER ratio (>1 is better)", fontsize=14
    )
    ax2.set_xscale("log")
    ax2.set_yscale("linear")
    ax2.tick_params(axis="x", labelcolor="black")
    ax2.set_xticks(p_values)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("Z-basis", fontsize=14)
    ax2.set_ylim([0.5, 2.5])

    ax2_right = ax2.twinx()
    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        z_syn = results["syndrome_density"]["Z"]
        valid_z_syn = ~np.isnan(z_syn)
        if np.any(valid_z_syn):
            ax2_right.plot(
                p_values[valid_z_syn],
                z_syn[valid_z_syn],
                marker="s",
                markersize=6,
                linewidth=1.5,
                linestyle="--",
                color=color,
                alpha=0.5,
                label=f"d={distance} (Syn. density)",
            )
    ax2_right.set_ylabel("Syndrome density reduction", fontsize=14)
    ax2_right.set_yscale("linear")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    if lines1 or lines2:
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Normalized threshold plot saved to: {output_path}")

    fig2, (ax1_abs, ax2_abs) = plt.subplots(1, 2, figsize=(16, 6))

    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        x_ler = results["X"]["ler"]
        x_ler_err = results["X"]["ler_err"]
        x_baseline_ler = results["X"]["baseline_ler"]
        valid_x = ~np.isnan(x_ler)
        if np.any(valid_x):
            ax1_abs.errorbar(
                p_values[valid_x],
                x_ler[valid_x],
                yerr=x_ler_err[valid_x],
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=6,
                capthick=2,
                elinewidth=2,
                label=f"d={distance} ({posterior_display})",
                color=color,
                alpha=0.8,
            )
        valid_base = ~np.isnan(x_baseline_ler)
        if np.any(valid_base):
            ax1_abs.plot(
                p_values[valid_base],
                x_baseline_ler[valid_base],
                linestyle="--",
                linewidth=2,
                marker="x",
                markersize=5,
                label=f"d={distance} ({baseline_label})",
                color=color,
                alpha=0.6,
            )

    ax1_abs.set_xlabel("Physical error rate, p", fontsize=14)
    ax1_abs.set_ylabel("Logical Error Rate", fontsize=14)
    ax1_abs.set_xscale("log")
    if any_positive_x:
        ax1_abs.set_yscale("log")
    ax1_abs.tick_params(axis="both", labelsize=12)
    ax1_abs.set_xticks(p_values)
    ax1_abs.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1_abs.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax1_abs.grid(True, alpha=0.3, which="both")
    ax1_abs.set_title("X-basis (Absolute LER)", fontsize=14)
    handles, labels = ax1_abs.get_legend_handles_labels()
    if handles:
        ax1_abs.legend(fontsize=8, loc="best", ncol=2)

    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        z_ler = results["Z"]["ler"]
        z_ler_err = results["Z"]["ler_err"]
        z_baseline_ler = results["Z"]["baseline_ler"]
        valid_z = ~np.isnan(z_ler)
        if np.any(valid_z):
            ax2_abs.errorbar(
                p_values[valid_z],
                z_ler[valid_z],
                yerr=z_ler_err[valid_z],
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=6,
                capthick=2,
                elinewidth=2,
                label=f"d={distance} ({posterior_display})",
                color=color,
                alpha=0.8,
            )
        valid_base = ~np.isnan(z_baseline_ler)
        if np.any(valid_base):
            ax2_abs.plot(
                p_values[valid_base],
                z_baseline_ler[valid_base],
                linestyle="--",
                linewidth=2,
                marker="x",
                markersize=5,
                label=f"d={distance} ({baseline_label})",
                color=color,
                alpha=0.6,
            )

    ax2_abs.set_xlabel("Physical error rate, p", fontsize=14)
    ax2_abs.set_ylabel("Logical Error Rate", fontsize=14)
    ax2_abs.set_xscale("log")
    if any_positive_z:
        ax2_abs.set_yscale("log")
    ax2_abs.tick_params(axis="both", labelsize=12)
    ax2_abs.set_xticks(p_values)
    ax2_abs.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2_abs.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax2_abs.grid(True, alpha=0.3, which="both")
    ax2_abs.set_title("Z-basis (Absolute LER)", fontsize=14)
    handles, labels = ax2_abs.get_legend_handles_labels()
    if handles:
        ax2_abs.legend(fontsize=8, loc="best", ncol=2)

    plt.tight_layout()
    output_path_absolute = output_path.replace(".png", "_absolute.png")
    plt.savefig(output_path_absolute, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Absolute LER threshold plot saved to: {output_path_absolute}")

    fig3, (ax1_speed, ax2_speed) = plt.subplots(1, 2, figsize=(16, 6))

    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        x_speed = results["X"]["speedup"]
        valid_x = ~np.isnan(x_speed)
        if np.any(valid_x):
            ax1_speed.plot(
                p_values[valid_x],
                x_speed[valid_x],
                marker="o",
                markersize=7,
                linewidth=2,
                color=color,
                label=f"d={distance}",
            )

    ax1_speed.set_xlabel("Physical error rate, p", fontsize=14)
    ax1_speed.set_ylabel(f"{baseline_label} / ({posterior_display}) speedup", fontsize=14)
    ax1_speed.set_xscale("log")
    ax1_speed.set_yscale("linear")
    ax1_speed.set_xticks(p_values)
    ax1_speed.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1_speed.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax1_speed.grid(True, alpha=0.3, which="both")
    ax1_speed.set_title("X-basis (Single-shot decode speedup)", fontsize=14)
    handles, labels = ax1_speed.get_legend_handles_labels()
    if handles:
        ax1_speed.legend(fontsize=8, loc="best")

    for i, distance in enumerate(distances):
        results = all_results[distance]
        color = colors[i]
        z_speed = results["Z"]["speedup"]
        valid_z = ~np.isnan(z_speed)
        if np.any(valid_z):
            ax2_speed.plot(
                p_values[valid_z],
                z_speed[valid_z],
                marker="o",
                markersize=7,
                linewidth=2,
                color=color,
                label=f"d={distance}",
            )

    ax2_speed.set_xlabel("Physical error rate, p", fontsize=14)
    ax2_speed.set_ylabel(f"{baseline_label} / ({posterior_display}) speedup", fontsize=14)
    ax2_speed.set_xscale("log")
    ax2_speed.set_yscale("linear")
    ax2_speed.set_xticks(p_values)
    ax2_speed.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2_speed.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax2_speed.grid(True, alpha=0.3, which="both")
    ax2_speed.set_title("Z-basis (Single-shot decode speedup)", fontsize=14)
    handles, labels = ax2_speed.get_legend_handles_labels()
    if handles:
        ax2_speed.legend(fontsize=8, loc="best")

    plt.tight_layout()
    output_path_speedup = output_path.replace(".png", "_speedup.png")
    plt.savefig(output_path_speedup, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Speedup threshold plot saved to: {output_path_speedup}")


def run_threshold_plot(model, device, dist, cfg):
    """
    Main function to compute threshold plot.
    
    Supports multi-GPU processing: each GPU processes a partition of the samples,
    and statistics are aggregated across all GPUs via all_reduce.

    Noise model mode:
        When ``test.noise_model`` is set to a dict (or ``"train"``), the sweep runs
        at a fixed noise model instead of iterating over scalar p values.  An optional
        ``threshold.noise_model_scale_factors`` list can be used to sweep scaled
        versions of the base noise model (e.g. ``[0.5, 1.0, 1.5]``).
    
    Args:
        model: Trained model
        device: Device to run on
        dist: DistributedManager
        cfg: Configuration object
    """
    import torch
    from qec.noise_model import resolve_test_noise_model

    rank = dist.rank if dist else 0
    world_size = dist.world_size if dist else 1

    # Resolve noise model early so we can report it and decide the sweep strategy.
    noise_model_obj, nm_mode = resolve_test_noise_model(cfg)
    use_noise_model = noise_model_obj is not None and nm_mode != "none"

    if rank == 0:
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

        decoder_config = get_global_decoder_config(
            cfg
        ) if get_global_decoder_config is not None else None
        print(f"\n🔗 Decoder Configuration:")
        if decoder_config is not None:
            print(
                f"  Baseline global decoder (GD2): "
                f"{decoder_config['baseline_label']} [{decoder_config['baseline_name']}]"
            )
            print(
                f"  Posterior global decoder (GD1): "
                f"{decoder_config['posterior_label']} [{decoder_config['posterior_name']}]"
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

        if use_noise_model:
            print(f"\n🔧 Noise Model Mode: {nm_mode}")
            print(f"  {noise_model_obj!r}")

    # Get distances from config
    if hasattr(cfg, 'threshold') and hasattr(cfg.threshold, 'distances'):
        distances = list(cfg.threshold.distances)
    else:
        # Fallback: use cfg.distance as the single distance
        distances = [cfg.distance]

    # Get n_rounds for each distance
    if hasattr(cfg, 'threshold'
              ) and hasattr(cfg.threshold, 'n_rounds') and cfg.threshold.n_rounds is not None:
        n_rounds_cfg = cfg.threshold.n_rounds
        if hasattr(n_rounds_cfg, '__iter__') and not isinstance(n_rounds_cfg, (str, bytes)):
            n_rounds_list = list(n_rounds_cfg)
        else:
            n_rounds_list = [n_rounds_cfg] * len(distances)
    else:
        # Default: n_rounds = distance for each
        n_rounds_list = distances.copy()

    if rank == 0:
        print(f"\nTesting {len(distances)} distance(s):")
        for d, r in zip(distances, n_rounds_list):
            print(f"  d={d}, n_rounds={r}")

    # -------------------------------------------------------------------
    # Determine sweep axis: p_values (legacy) or noise_model scale factors
    # -------------------------------------------------------------------
    scale_factors = None
    if use_noise_model:
        threshold_cfg = getattr(cfg, 'threshold', None)
        scale_factors = (
            list(threshold_cfg.noise_model_scale_factors) if threshold_cfg is not None and
            getattr(threshold_cfg, 'noise_model_scale_factors', None) is not None else None
        )
        if scale_factors is not None:
            p_values = scale_factors
            if rank == 0:
                print(f"\nNoise model scale factor sweep: {scale_factors}")
                print(f"  Number of samples per point: {cfg.test.num_samples}")
        else:
            p_placeholder = float(noise_model_obj.get_max_probability())
            p_values = [p_placeholder]
            if rank == 0:
                print(f"\nFixed noise model (single point per distance)")
                print(f"  Number of samples per point: {cfg.test.num_samples}")
    else:
        if hasattr(cfg, 'threshold') and hasattr(cfg.threshold, 'p_values'):
            p_values = list(cfg.threshold.p_values)
        else:
            p_values = [cfg.test.p_error]

        if rank == 0:
            print(f"\nTesting {len(p_values)} physical error rates:")
            print(f"  p = {p_values[0]:.4f} to {p_values[-1]:.4f}")
            print(f"  Number of samples per p: {cfg.test.num_samples}")

    # Compute LER for all distances and p values
    all_results = {}
    for i, distance in enumerate(distances):
        n_rounds = n_rounds_list[i] if i < len(n_rounds_list) else distance

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Computing for distance d={distance}, n_rounds={n_rounds}")
            print(f"{'='*60}")

        if use_noise_model and scale_factors is not None:
            from omegaconf import OmegaConf
            saved_nm_cfg = cfg.test.noise_model
            sweep_p_values = []
            for sf in scale_factors:
                scaled_nm = noise_model_obj.scale(sf) if sf != 1.0 else noise_model_obj
                cfg.test.noise_model = OmegaConf.create(scaled_nm.to_config_dict())
                cfg.test.p_error = float(scaled_nm.get_max_probability())
                sweep_p_values.append(cfg.test.p_error)

                if rank == 0:
                    print(f"\n  --- d={distance}, scale={sf} (p_max={cfg.test.p_error:.6f}) ---")

                results = compute_ler_for_p_range(
                    model,
                    device,
                    dist,
                    cfg,
                    [cfg.test.p_error],
                    distance,
                    rank,
                    n_rounds=n_rounds,
                )
                if results is None:
                    continue
                # Relabel: replace the scalar p key with the scale factor label
                all_results.setdefault(
                    distance,
                    _empty_threshold_results(results.get("decoder_labels", {})),
                )
                agg = all_results[distance]
                agg['p_values'].append(sf)
                for basis in ('X', 'Z'):
                    for key in agg[basis]:
                        agg[basis][key].append(
                            results[basis][key][0] if len(results[basis][key]) >
                            0 else float('nan')
                        )
                for basis in ('X', 'Z'):
                    agg['syndrome_density'][basis].append(
                        results['syndrome_density'][basis][0]
                        if len(results['syndrome_density'][basis]) > 0 else float('nan')
                    )

            # Restore original noise model config
            cfg.test.noise_model = saved_nm_cfg

            # Convert to numpy arrays for consistency with the standard path
            if distance in all_results:
                agg = all_results[distance]
                agg['p_values'] = np.array(agg['p_values'])
                for basis in ('X', 'Z'):
                    for key in agg[basis]:
                        agg[basis][key] = np.array(agg[basis][key])
                    agg['syndrome_density'][basis] = np.array(agg['syndrome_density'][basis])
        else:
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

    # Print summary table
    if rank == 0:
        print("\n" + "=" * 80)
        print("THRESHOLD RESULTS SUMMARY")
        print("=" * 80)
        x_label = "scale" if (use_noise_model and scale_factors) else "p"
        for distance, results in all_results.items():
            print(f"\n  d={distance}:")
            decoder_labels = results.get("decoder_labels", {})
            baseline_label = decoder_labels.get("baseline_label", "baseline")
            posterior_label = decoder_labels.get("posterior_label", "posterior")
            for j, pv in enumerate(results['p_values']):
                x_ler = results['X']['ler'][j] if j < len(results['X']['ler']) else float('nan')
                z_ler = results['Z']['ler'][j] if j < len(results['Z']['ler']) else float('nan')
                x_base = results['X']['baseline_ler'][j] if j < len(results['X']['baseline_ler']
                                                                   ) else float('nan')
                z_base = results['Z']['baseline_ler'][j] if j < len(results['Z']['baseline_ler']
                                                                   ) else float('nan')
                x_speed = results['X']['speedup'][j] if j < len(results['X']['speedup']
                                                               ) else float('nan')
                z_speed = results['Z']['speedup'][j] if j < len(results['Z']['speedup']
                                                               ) else float('nan')
                print(
                    f"    {x_label}={pv:.4f}  "
                    f"X: {posterior_label}={x_ler:.6f} {baseline_label}={x_base:.6f} speedup={x_speed:.4f}x  "
                    f"Z: {posterior_label}={z_ler:.6f} {baseline_label}={z_base:.6f} speedup={z_speed:.4f}x"
                )

        # Save JSON results.  Include the rounds mode in the filename so root-level
        # threshold artifacts are not ambiguous when comparing d-round and 4d-round runs.
        import json
        rounds_pairs = []
        for distance, results in all_results.items():
            rounds_pairs.append((int(distance), int(results.get("n_rounds", distance))))
        if rounds_pairs and all(r == d for d, r in rounds_pairs):
            rounds_mode = "n_rounds_eq_d"
        elif rounds_pairs and all(r == 4 * d for d, r in rounds_pairs):
            rounds_mode = "n_rounds_eq_4d"
        else:
            rounds_mode = "n_rounds_custom"
        result_path = os.path.join(cfg.output, f"threshold_results_{rounds_mode}.json")
        os.makedirs(cfg.output, exist_ok=True)
        json_results = {}
        for distance, results in all_results.items():
            json_results[str(distance)] = {
                'distance': int(distance),
                'n_rounds': int(results.get("n_rounds", distance)),
                'p_values': np.asarray(results['p_values']).tolist(),
                'decoder_labels': dict(results.get('decoder_labels', {})),
                'X': {
                    k: np.asarray(v).tolist() for k, v in results['X'].items()
                },
                'Z': {
                    k: np.asarray(v).tolist() for k, v in results['Z'].items()
                },
                'syndrome_density':
                    {
                        k: np.asarray(v).tolist() for k, v in results['syndrome_density'].items()
                    },
            }
            if use_noise_model:
                json_results[str(distance)]['noise_model'] = noise_model_obj.to_config_dict()
                json_results[str(distance)]['noise_model_mode'] = nm_mode
                if scale_factors:
                    json_results[str(distance)]['noise_model_scale_factors'] = scale_factors
        with open(result_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {result_path}")

    # Synchronize all GPUs before plotting (ensure all computations are complete)
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        if rank == 0:
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
        nm_tag = "_noisemodel" if use_noise_model else ""
        output_filename = f"threshold_{dist_str}_{samples_str}shots{nm_tag}.png"
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
