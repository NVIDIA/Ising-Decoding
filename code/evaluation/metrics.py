# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Evaluation Metrics Module

Provides wrapper functions for computing model evaluation metrics during training:
- Logical Error Rate (LER) via compute_validation_ler()

These wrappers handle:
- Multi-pair mode (multiple distances/rounds)
- Error handling and logging
- Metric extraction from raw results

The underlying computations are performed by:
- evaluation/logical_error_rate.py (Stim-based, Torch-only)
"""

from typing import Any

# Import logical error rate functions (Stim-based)
try:
    from evaluation.logical_error_rate import (
        count_logical_errors_with_errorbar as compute_logical_error_rate_stim,
        compute_syndrome_density_reduction as compute_syndrome_density_reduction_stim
    )
    HAS_LER_MODULE = True
except ImportError:
    HAS_LER_MODULE = False
    compute_logical_error_rate_stim = None
    compute_syndrome_density_reduction_stim = None

# Active computation functions (set via configure_metrics)
compute_logical_error_rate = None
compute_syndrome_density_reduction = None


def _safe_ratio(numer: Any, denom: Any) -> float:
    """
    Safe ratio helper for metrics.

    Requirements:
    - 0/0 is ill-defined; default to 1.0 (neutral "no change" factor)
    - x/0 for x>0 returns +inf (signals "infinite" improvement/degradation)
    """
    try:
        n = float(numer)
        d = float(denom)
    except Exception:
        return float("nan")
    if d == 0.0:
        return 1.0 if n == 0.0 else float("inf")
    return n / d


def configure_metrics(rank=0):
    """
    Configure which metric computation functions to use.
    
    Args:
        rank: Process rank for printing messages
        
    Returns:
        Tuple of (compute_logical_error_rate, compute_syndrome_density_reduction)
    """
    global compute_logical_error_rate, compute_syndrome_density_reduction

    compute_logical_error_rate = compute_logical_error_rate_stim
    compute_syndrome_density_reduction = compute_syndrome_density_reduction_stim
    if rank == 0 and HAS_LER_MODULE:
        print("[Evaluation] Using Stim-based validation functions")

    return compute_logical_error_rate, compute_syndrome_density_reduction


def _extract_reduction_factor(result, rank=0):
    """Helper to extract reduction factor from syndrome density result."""
    reduction_factor = None

    if isinstance(result, dict):
        data = result.get('stim', result)
        x_reduction = data.get('reduction factor (X)')
        z_reduction = data.get('reduction factor (Z)')

        if x_reduction is not None and z_reduction is not None:
            reduction_factor = (x_reduction + z_reduction) / 2.0
        elif x_reduction is not None:
            reduction_factor = x_reduction
        elif z_reduction is not None:
            reduction_factor = z_reduction
        else:
            for key in ['reduction factor (X/Z)', 'reduction_factor', 'reduction factor']:
                if key in data:
                    reduction_factor = data[key]
                    break
    elif isinstance(result, (float, int)):
        reduction_factor = float(result)

    return reduction_factor


def _factor_to_percent(factor):
    """Convert an SDR reduction factor (X times) to a percentage reduction.

    factor = input_density / residual_density, so the fraction of syndromes
    removed is  1 - 1/factor  and the percentage is  (1 - 1/factor) * 100.

    Examples:
        2.0x  -> 50.0%   (half the syndromes remain)
        3.7x  -> 73.0%   (~27% remain)
        10.0x -> 90.0%   (10% remain)
        inf   -> 100.0%  (perfect correction)
        1.0x  -> 0.0%    (no improvement)
        nan   -> nan
    """
    import math
    if factor is None:
        return None
    if math.isnan(factor):
        return float('nan')
    if math.isinf(factor):
        return 100.0
    if factor <= 0:
        return float('nan')
    return (1.0 - 1.0 / factor) * 100.0


def compute_syndrome_density(
    model, device, dist, cfg, generator=None, rank=0, sdr_as_percent=False
):
    """Compute syndrome density reduction factor for validation.

    Args:
        sdr_as_percent: If True, return SDR as a percentage reduction (0-100)
            instead of the default X-times multiplier.  A 3.7x factor becomes
            ~73%.  Controlled by cfg.sdr_as_percent when called from training.
    """
    if not HAS_LER_MODULE or compute_syndrome_density_reduction is None:
        if rank == 0:
            print(
                "[Syndrome Density] Warning: Syndrome density module not available, skipping computation"
            )
        return None

    is_multi = generator is not None and hasattr(generator,
                                                 'is_multi_pair') and generator.is_multi_pair()

    if is_multi:
        if rank == 0:
            print(f"[Syndrome Density] Multi-pair mode: computing for each distance separately...")

        results = {}
        pairs_and_gens = generator.get_all_generators()

        for (d, r), single_gen in pairs_and_gens:
            if rank == 0:
                print(f"\n[Syndrome Density] Computing for (d={d}, r={r})...")

            try:
                orig_d, orig_r = cfg.distance, cfg.n_rounds
                cfg.distance, cfg.n_rounds = d, r

                result = compute_syndrome_density_reduction(
                    model, device, dist, cfg, generator=single_gen
                )
                reduction_factor = _extract_reduction_factor(result, rank)

                if reduction_factor is not None:
                    value = _factor_to_percent(
                        reduction_factor
                    ) if sdr_as_percent else reduction_factor
                    results[(d, r)] = value
                    if rank == 0:
                        unit = "%" if sdr_as_percent else "x"
                        print(f"[Syndrome Density] (d={d}, r={r}): {value:.4f}{unit}")

                cfg.distance, cfg.n_rounds = orig_d, orig_r

            except Exception as e:
                if rank == 0:
                    print(f"[Syndrome Density] Error for (d={d}, r={r}): {e}")
                    import traceback
                    traceback.print_exc()

        return results if results else None

    else:
        try:
            if rank == 0:
                print(f"[Syndrome Density] Computing syndrome density reduction...")

            result = compute_syndrome_density_reduction(model, device, dist, cfg)

            reduction_factor = _extract_reduction_factor(result, rank)

            if reduction_factor is not None:
                value = _factor_to_percent(reduction_factor) if sdr_as_percent else reduction_factor
                if rank == 0:
                    unit = "%" if sdr_as_percent else "x"
                    print(f"[Syndrome Density] Reduction: {value:.4f}{unit}")
                return float(value)
            else:
                if rank == 0:
                    print(
                        f"[Syndrome Density] Warning: Could not extract reduction factor from result"
                    )
                return None

        except Exception as e:
            if rank == 0:
                print(f"[Syndrome Density] Error computing syndrome density: {e}")
                import traceback
                traceback.print_exc()
            return None


def _compute_single_ler(model, device, dist, cfg, generator, rank):
    """Helper function to compute LER for a single (distance, rounds) pair."""
    try:
        if rank == 0:
            print(f"[LER Validation] Computing logical error rate...")

        result = compute_logical_error_rate(model, device, dist, cfg)

        # Extract PyMatching decode speedup (baseline / after pre-decoder), averaged across X/Z when available.
        pymatching_speedup_avg = None
        # Also extract the underlying (single-shot) latencies so we can report them in logs.
        pymatching_latency_baseline_avg = None
        pymatching_latency_after_avg = None
        if isinstance(result, dict) and "X" in result and "Z" in result:

            def _extract_latencies(basis_dict):
                if not isinstance(basis_dict, dict):
                    return (None, None)
                base = basis_dict.get("pymatch latency (baseline µs/round)")
                post = basis_dict.get("pymatch latency (after predecoder µs/round)")
                try:
                    base_f = float(base)
                    post_f = float(post)
                except Exception:
                    return (None, None)
                return (base_f, post_f)

            def _extract_speedup(basis_dict):
                if not isinstance(basis_dict, dict):
                    return None
                base = basis_dict.get("pymatch latency (baseline µs/round)")
                post = basis_dict.get("pymatch latency (after predecoder µs/round)")
                try:
                    base_f = float(base)
                    post_f = float(post)
                except Exception:
                    return None
                if not (post_f > 0.0):
                    return None
                return base_f / post_f

            x_base_lat, x_post_lat = _extract_latencies(result.get("X"))
            z_base_lat, z_post_lat = _extract_latencies(result.get("Z"))
            x_speedup = _extract_speedup(result.get("X"))
            z_speedup = _extract_speedup(result.get("Z"))
            if x_speedup is not None and z_speedup is not None:
                pymatching_speedup_avg = 0.5 * (x_speedup + z_speedup)
            elif x_speedup is not None:
                pymatching_speedup_avg = x_speedup
            elif z_speedup is not None:
                pymatching_speedup_avg = z_speedup

            # Average latencies across bases when available.
            base_vals = [v for v in (x_base_lat, z_base_lat) if v is not None]
            post_vals = [v for v in (x_post_lat, z_post_lat) if v is not None]
            if base_vals:
                pymatching_latency_baseline_avg = float(sum(base_vals) / len(base_vals))
            if post_vals:
                pymatching_latency_after_avg = float(sum(post_vals) / len(post_vals))

        if isinstance(result, dict):
            ler_value = None
            for key in [
                'logical_error_rate', 'ler', 'error_rate', 'avg_ler', 'logical error ratio (mean)'
            ]:
                if key in result:
                    ler_value = result[key]
                    break

            if ler_value is None:
                if 'X' in result and isinstance(result['X'], dict):
                    x_ler = (
                        result['X'].get('logical error ratio (mean)') or
                        result['X'].get('logical_error_rate') or result['X'].get('ler')
                    )
                    if x_ler is not None:
                        ler_value = x_ler

                if ler_value is None and 'Z' in result and isinstance(result['Z'], dict):
                    z_ler = (
                        result['Z'].get('logical error ratio (mean)') or
                        result['Z'].get('logical_error_rate') or result['Z'].get('ler')
                    )
                    if z_ler is not None:
                        ler_value = z_ler

                if 'X' in result and 'Z' in result and isinstance(result['X'], dict) and isinstance(
                    result['Z'], dict
                ):
                    x_ler = (
                        result['X'].get('logical error ratio (mean)') or
                        result['X'].get('logical_error_rate') or result['X'].get('ler')
                    )
                    z_ler = (
                        result['Z'].get('logical error ratio (mean)') or
                        result['Z'].get('logical_error_rate') or result['Z'].get('ler')
                    )

                    if x_ler is not None and z_ler is not None:
                        ler_value = (x_ler + z_ler) / 2.0
        elif isinstance(result, (float, int)):
            ler_value = float(result)
        else:
            ler_value = None

        # Extract LER reduction factor
        ler_reduction_factor = None
        if isinstance(result, dict) and 'X' in result and 'Z' in result:
            x_data, z_data = result['X'], result['Z']

            if isinstance(x_data, dict) and isinstance(z_data, dict):
                x_logical_errors = x_data.get('logical errors')
                x_pymatch_flips = x_data.get('pymatch flips')
                z_logical_errors = z_data.get('logical errors')
                z_pymatch_flips = z_data.get('pymatch flips')

                if all(
                    v is not None
                    for v in [x_logical_errors, x_pymatch_flips, z_logical_errors, z_pymatch_flips]
                ):
                    x_reduction = _safe_ratio(x_pymatch_flips, x_logical_errors)
                    z_reduction = _safe_ratio(z_pymatch_flips, z_logical_errors)
                    # Average when both are finite/defined; otherwise fall back to whichever is not-NaN.
                    vals = [v for v in (x_reduction, z_reduction) if v == v]  # exclude NaN
                    ler_reduction_factor = (sum(vals) / len(vals)) if vals else float("nan")

        if ler_value is not None:
            if rank == 0:
                msg = f"[LER Validation] Logical error rate: {ler_value:.6f}"
                if ler_reduction_factor is not None:
                    msg += f" | LER reduction factor: {float(ler_reduction_factor):.4f}x"
                if pymatching_speedup_avg is not None:
                    msg += f" | PyMatching speedup: {float(pymatching_speedup_avg):.3f}x"
                if pymatching_latency_baseline_avg is not None and pymatching_latency_after_avg is not None:
                    msg += (
                        f" | PyMatching latency (µs/round, single-shot): "
                        f"{pymatching_latency_baseline_avg:.3f} -> {pymatching_latency_after_avg:.3f}"
                    )
                print(msg)
            return (float(ler_value), ler_reduction_factor, pymatching_speedup_avg)
        else:
            if rank == 0:
                print(f"[LER Validation] Warning: Could not extract LER value from result")
            return (None, None, pymatching_speedup_avg)

    except Exception as e:
        if rank == 0:
            print(f"[LER Validation] Error computing LER: {e}")
            import traceback
            traceback.print_exc()
        return (None, None, None)


def compute_validation_ler(model, device, dist, cfg, generator=None, rank=0):
    """Compute logical error rate for validation/early stopping."""
    if not HAS_LER_MODULE:
        if rank == 0:
            print("[LER Validation] Warning: LER module not available, skipping LER computation")
        return None

    is_multi = generator is not None and hasattr(generator,
                                                 'is_multi_pair') and generator.is_multi_pair()

    if is_multi:
        if rank == 0:
            print(f"[LER Validation] Multi-pair mode: computing for each distance separately...")

        results = {}
        pairs_and_gens = generator.get_all_generators()

        for (d, r), single_gen in pairs_and_gens:
            if rank == 0:
                print(f"\n[LER Validation] Computing for (d={d}, r={r})...")

            try:
                orig_d, orig_r = cfg.distance, cfg.n_rounds
                cfg.distance, cfg.n_rounds = d, r

                ler_result = _compute_single_ler(model, device, dist, cfg, single_gen, rank)

                if ler_result is not None:
                    results[(d, r)] = ler_result
                    ler_val, ler_red, _ = ler_result
                    if rank == 0:
                        if ler_red is not None:
                            print(
                                f"[LER Validation] (d={d}, r={r}): LER={ler_val:.6f}, Reduction={ler_red:.4f}x"
                            )
                        else:
                            print(f"[LER Validation] (d={d}, r={r}): LER={ler_val:.6f}")

                cfg.distance, cfg.n_rounds = orig_d, orig_r

            except Exception as e:
                if rank == 0:
                    print(f"[LER Validation] Error for (d={d}, r={r}): {e}")
                    import traceback
                    traceback.print_exc()

        return results if results else None

    else:
        return _compute_single_ler(model, device, dist, cfg, generator, rank)
