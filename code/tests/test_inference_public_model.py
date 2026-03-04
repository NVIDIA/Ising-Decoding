#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Ensure repo's code/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model
from evaluation.logical_error_rate import count_logical_errors_with_errorbar
from training.distributed import DistributedManager

# Tolerance for "LER after <= baseline + tolerance".
# At d=13 with 262k shots and LER ~2e-4, the standard error per basis is
# ~sqrt(p/N) ≈ 2.8e-5; comparing two independent estimates (before/after)
# gives combined SE ~4e-5, so 1e-4 is a ~2.5-sigma guard against flakes.
LER_IMPROVEMENT_TOLERANCE = 1e-4


def _run_inference_rtest(distance: int, n_rounds: int):
    """Load public config and model v1.0.94, run inference at given distance/n_rounds; returns result dict."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "conf" / "config_public.yaml"
    cfg = OmegaConf.load(str(cfg_path))

    cfg.model_id = 1
    cfg.distance = distance
    cfg.n_rounds = n_rounds
    cfg.workflow.task = "inference"

    model_spec = validate_public_config(cfg)
    merged = apply_public_defaults_and_model(cfg, model_spec)

    model_file = (repo_root / "models" / "PreDecoderModelMemory_v1.0.94.pt").resolve()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_file}. It must be in the repo (Git LFS). Run 'git lfs pull' or restore the file."
        )

    merged.model_checkpoint_dir = str(model_file.parent)
    merged.test.use_model_checkpoint = 94
    merged.test.latency_num_samples = 0
    merged.test.verbose_inference = False
    if "dataloader" in merged.test:
        merged.test.dataloader.num_workers = 0

    env_overrides = {
        "PREDECODER_INFERENCE_NUM_SAMPLES": None,
        "PREDECODER_INFERENCE_LATENCY_SAMPLES": None,
        "PREDECODER_INFERENCE_MEAS_BASIS": None,
        "PREDECODER_INFERENCE_NUM_WORKERS": "0",
    }
    old_env = {k: os.environ.get(k) for k in env_overrides}
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(merged, dist)
        return count_logical_errors_with_errorbar(model, dist.device, dist, merged)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_ler_improvement_vs_baseline(self, result, tolerance: float = LER_IMPROVEMENT_TOLERANCE):
    """Assert LER after pre-decoder <= baseline + tolerance for both X and Z."""
    for basis in ("X", "Z"):
        ler_after = float(result[basis]["logical error ratio (mean)"])
        ler_baseline = float(result[basis]["logical error ratio (pymatch mean)"])
        self.assertLessEqual(
            ler_after,
            ler_baseline + tolerance,
            msg=f"{basis}: LER after ({ler_after:.6f}) > baseline ({ler_baseline:.6f}) + {tolerance}",
        )


# Required model file; must exist in repo so tests fail (not skip) if it is removed.
REQUIRED_MODEL_FILE = (Path(__file__).resolve().parents[2] / "models" / "PreDecoderModelMemory_v1.0.94.pt")


class TestPublicInferenceModelV1(unittest.TestCase):
    def test_required_model_file_present(self):
        """Fail if the tracked model file is missing (e.g. removed from repo). Must not skip."""
        self.assertTrue(
            REQUIRED_MODEL_FILE.exists(),
            msg=f"Required model file missing: {REQUIRED_MODEL_FILE}. "
            "It must be in the repo (Git LFS). Run 'git lfs pull' or restore the file.",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d13_noise25p_ler_quality(self):
        """d=13, 25p noise: LER avg, per-basis (X/Z) in range, and improvement over baseline."""
        result = _run_inference_rtest(13, 13)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        # 1) Average LER (existing check)
        self.assertAlmostEqual(ler_avg, 2e-4, delta=1e-4)

        # 2) Per-basis LER in expected range (same band as avg)
        self.assertAlmostEqual(ler_x, 2e-4, delta=1e-4, msg="LER X out of range")
        self.assertAlmostEqual(ler_z, 2e-4, delta=1e-4, msg="LER Z out of range")

        # 3) Pre-decoder not worse than baseline (LER after <= baseline + tolerance)
        _assert_ler_improvement_vs_baseline(self, result)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d9_noise25p_ler_quality(self):
        """d=9, n_rounds=9, 25p noise: LER avg in conservative range, per-basis, improvement.
        Baseline range can be tightened by running: python3 code/tests/measure_d9_ler.py"""
        result = _run_inference_rtest(9, 9)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        # Conservative upper bound for d=9 (smaller code -> higher LER than d=13).
        LER_AVG_D9_MAX = 5e-3
        LER_BASIS_D9_MAX = 1e-2
        self.assertGreaterEqual(ler_avg, 0.0, msg="LER avg negative")
        self.assertLessEqual(ler_avg, LER_AVG_D9_MAX, msg=f"LER avg > {LER_AVG_D9_MAX}")
        self.assertGreaterEqual(ler_x, 0.0)
        self.assertLessEqual(ler_x, LER_BASIS_D9_MAX, msg="LER X too high")
        self.assertGreaterEqual(ler_z, 0.0)
        self.assertLessEqual(ler_z, LER_BASIS_D9_MAX, msg="LER Z too high")

        _assert_ler_improvement_vs_baseline(self, result)


if __name__ == "__main__":
    unittest.main()
