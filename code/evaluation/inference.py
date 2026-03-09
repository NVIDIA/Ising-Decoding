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
Single-point public inference runner.

Public inference should:
- evaluate at cfg.distance / cfg.n_rounds (user-specified evaluation targets)
- use cfg.data.noise_model (circuit-level 25p) when cfg.test.noise_model == "train"
- report only:
  - LER (PyMatching baseline vs after pre-decoder), X/Z/Avg
  - PyMatching decode speedup (baseline latency / after-predecoder latency), averaged across X/Z
- NO plots and NO syndrome density reduction (SDR)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os


def _detect_shm_bytes() -> Optional[int]:
    try:
        st = os.statvfs("/dev/shm")
        return int(st.f_frsize * st.f_blocks)
    except Exception:
        return None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_basis_metrics(basis_dict: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Returns:
        ler_after: model+predecoder LER ("logical error ratio (mean)")
        ler_baseline: pymatching baseline LER ("logical error ratio (pymatch mean)")
        lat_after: pymatching latency after predecoder (µs/round)
        lat_baseline: pymatching latency baseline (µs/round)
    """
    ler_after = _safe_float(basis_dict.get("logical error ratio (mean)"))
    ler_baseline = _safe_float(basis_dict.get("logical error ratio (pymatch mean)"))
    lat_baseline = _safe_float(basis_dict.get("pymatch latency (baseline µs/round)"))
    lat_after = _safe_float(basis_dict.get("pymatch latency (after predecoder µs/round)"))
    return ler_after, ler_baseline, lat_after, lat_baseline


def run_inference(model, device, dist, cfg) -> None:
    """
    Run a single inference evaluation and print a compact summary.
    """
    # Optional overrides (env-based) to keep inference lightweight.
    try:
        test_cfg = getattr(cfg, "test", None)
        if test_cfg is not None:
            env_samples = os.environ.get("PREDECODER_INFERENCE_NUM_SAMPLES")
            if env_samples:
                test_cfg.num_samples = int(env_samples)
            env_latency = os.environ.get("PREDECODER_INFERENCE_LATENCY_SAMPLES")
            if env_latency:
                test_cfg.latency_num_samples = int(env_latency)
            env_basis = os.environ.get("PREDECODER_INFERENCE_MEAS_BASIS")
            if env_basis:
                test_cfg.meas_basis_test = str(env_basis)
    except Exception:
        pass

    result: Optional[Dict[str, Any]] = None

    # Inference-only safety: allow overriding DataLoader workers to avoid container shm issues.
    try:
        dl_cfg = getattr(getattr(cfg, "test", None), "dataloader", None)
        if dl_cfg is not None:
            override_workers = os.environ.get("PREDECODER_INFERENCE_NUM_WORKERS")
            is_container = os.path.exists("/.dockerenv")
            if override_workers is not None:
                dl_cfg.num_workers = int(override_workers)
            elif is_container and int(getattr(dl_cfg, "num_workers", 0)) > 0:
                shm_bytes = _detect_shm_bytes()
                if shm_bytes is not None and shm_bytes < 1_000_000_000:
                    dl_cfg.num_workers = 0
                    print(
                        f"[Inference] Detected small /dev/shm "
                        f"({shm_bytes / (1024 ** 2):.1f} MiB); "
                        "setting num_workers=0. "
                        "Override with PREDECODER_INFERENCE_NUM_WORKERS."
                    )
            if int(getattr(dl_cfg, "num_workers", 0)) == 0:
                dl_cfg.persistent_workers = False
                if hasattr(dl_cfg, "prefetch_factor"):
                    dl_cfg.prefetch_factor = None
    except Exception:
        pass

    if dist.rank == 0:
        total_samples = int(getattr(getattr(cfg, "test", None), "num_samples", 0) or 0)
        latency_n = getattr(getattr(cfg, "test", None), "latency_num_samples", 10_000)
        try:
            latency_n = int(latency_n)
        except Exception:
            latency_n = 10_000
        if latency_n < 0:
            latency_n = 0
        ws = int(getattr(dist, "world_size", 1) or 1)
        mode = "Stim"
        # Users often think the process is stuck here; make it explicit that heavy eval work is starting.
        print(
            f"[Inference] Starting {mode} evaluation (may take a bit). "
            f"Shots per basis: {total_samples:,} (sharded across {ws} GPU(s)); "
            f"latency timing subset: {latency_n:,} single-shot decodes."
        )

    from evaluation.logical_error_rate import (
        count_logical_errors_with_errorbar as _count_ler,
    )
    result = _count_ler(model, device, dist, cfg)

    if dist.rank != 0:
        return

    test_nm_mode = str(getattr(getattr(cfg, "test", None), "noise_model", "train")).lower()
    has_explicit_nm = getattr(getattr(cfg, "data", None), "noise_model", None) is not None
    if test_nm_mode == "train" and has_explicit_nm:
        print(f"\n[Inference] d={int(cfg.distance)}, n_rounds={int(cfg.n_rounds)}, noise_model=25p")
        # Print the explicit 25-parameter noise model once (rank 0 only).
        try:
            from omegaconf import OmegaConf
            from qec.noise_model import NoiseModel

            nm_cfg = getattr(getattr(cfg, "data", None), "noise_model", None)
            nm_dict = OmegaConf.to_container(nm_cfg,
                                             resolve=True) if hasattr(nm_cfg, "items") else nm_cfg
            if nm_dict is not None:
                nm_obj = NoiseModel.from_config_dict(dict(nm_dict))
                print(f"[Inference] Using explicit noise_model (25p): {nm_obj!r}")
        except Exception:
            # Keep inference readable; if conversion fails, don't crash printing.
            pass
    else:
        # Legacy / internal-style configs only; keep it readable.
        p = getattr(getattr(cfg, "test", None), "p_error", None)
        if p is None:
            print(f"\n[Inference] d={int(cfg.distance)}, n_rounds={int(cfg.n_rounds)}")
        else:
            print(
                f"\n[Inference] d={int(cfg.distance)}, n_rounds={int(cfg.n_rounds)}, p={float(p):.4f}"
            )

    if not isinstance(result, dict) or "X" not in result or "Z" not in result:
        print(
            "[Inference] Warning: unexpected inference result format; expected dict with 'X' and 'Z'."
        )
        return

    x_after, x_base, x_lat_after, x_lat_base = _extract_basis_metrics(result["X"])
    z_after, z_base, z_lat_after, z_lat_base = _extract_basis_metrics(result["Z"])

    def _avg(a: float, b: float) -> float:
        vals = [v for v in (a, b) if v == v]  # NaN check
        return sum(vals) / len(vals) if vals else float("nan")

    avg_lat_base = _avg(x_lat_base, z_lat_base)
    avg_lat_after = _avg(x_lat_after, z_lat_after)

    # Speedup = baseline latency / after latency
    x_speedup = (x_lat_base / x_lat_after) if (x_lat_after > 0.0) else float("nan")
    z_speedup = (z_lat_base / z_lat_after) if (z_lat_after > 0.0) else float("nan")
    avg_speedup = _avg(x_speedup, z_speedup)

    label_w = 40
    print(f"  {'':<{label_w}}{'No pre-decoder':>15}  {'After pre-decoder':>17}")
    print(
        f"  {'PyMatching latency - X basis (µs/round):':<{label_w}}{x_lat_base:>15.3f}  {x_lat_after:>17.3f}"
    )
    print(
        f"  {'PyMatching latency - Z basis (µs/round):':<{label_w}}{z_lat_base:>15.3f}  {z_lat_after:>17.3f}"
    )
    print(
        f"  {'PyMatching latency - Avg (µs/round):':<{label_w}}{avg_lat_base:>15.3f}  {avg_lat_after:>17.3f}"
    )
    print(f"  {'LER - X basis:':<{label_w}}{x_base:>15.6f}  {x_after:>17.6f}")
    print(f"  {'LER - Z basis:':<{label_w}}{z_base:>15.6f}  {z_after:>17.6f}")
    print(
        f"  {'LER - Avg:':<{label_w}}{_avg(x_base, z_base):>15.6f}  {_avg(x_after, z_after):>17.6f}"
    )
    print(f"  {'PyMatching speedup (Avg X/Z):':<{label_w}}{avg_speedup:>15.3f}x")
