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

import os
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model
from evaluation.logical_error_rate import count_logical_errors_with_errorbar
from training.distributed import DistributedManager

# Tolerance for "LER after <= baseline + tolerance".
# At d=13 with 262k shots and LER ~2e-4, the standard error per basis is
# ~sqrt(p/N) ≈ 2.8e-5; comparing two independent estimates (before/after)
# gives combined SE ~4e-5, so 2e-4 is a ~5-sigma guard against flakes.
LER_IMPROVEMENT_TOLERANCE = 2e-4

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

MODEL_R9 = {
    "filename": "PreDecoderModelMemory_r9_v1.0.77.pt",
    "checkpoint": 77,
    "model_id": 1,
}
MODEL_R13 = {
    "filename": "PreDecoderModelMemory_r13_v1.0.86.pt",
    "checkpoint": 86,
    "model_id": 4,
}

REQUIRED_MODEL_FILES = [
    MODELS_DIR / MODEL_R9["filename"],
    MODELS_DIR / MODEL_R13["filename"],
]


def _run_inference_rtest(distance: int, n_rounds: int, model_info: dict):
    """Load public config and a specific model, run inference; returns result dict."""
    cfg_path = REPO_ROOT / "conf" / "config_public.yaml"
    cfg = OmegaConf.load(str(cfg_path))

    cfg.model_id = model_info["model_id"]
    cfg.distance = distance
    cfg.n_rounds = n_rounds
    cfg.workflow.task = "inference"

    model_spec = validate_public_config(cfg)
    merged = apply_public_defaults_and_model(cfg, model_spec)

    model_file = (MODELS_DIR / model_info["filename"]).resolve()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_file}. It must be in the repo (Git LFS). Run 'git lfs pull' or restore the file."
        )

    merged.model_checkpoint_dir = str(model_file.parent)
    merged.test.use_model_checkpoint = model_info["checkpoint"]
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
            msg=
            f"{basis}: LER after ({ler_after:.6f}) > baseline ({ler_baseline:.6f}) + {tolerance}",
        )


class TestPublicInferenceModels(unittest.TestCase):
    """Tests for pre-trained model files (r9 and r13)."""

    def test_required_model_files_present(self):
        """Fail if any tracked model file is missing. Must not skip."""
        for model_file in REQUIRED_MODEL_FILES:
            self.assertTrue(
                model_file.exists(),
                msg=f"Required model file missing: {model_file}. "
                "It must be in the repo (Git LFS). Run 'git lfs pull' or restore the file.",
            )

    # -- R=9 model tests --

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d9_r9_ler_quality(self):
        """d=9, n_rounds=9, r9 model: LER avg in range, per-basis, improvement."""
        result = _run_inference_rtest(9, 9, MODEL_R9)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        LER_AVG_D9_MAX = 5e-3
        LER_BASIS_D9_MAX = 1e-2
        self.assertGreaterEqual(ler_avg, 0.0, msg="LER avg negative")
        self.assertLessEqual(ler_avg, LER_AVG_D9_MAX, msg=f"LER avg > {LER_AVG_D9_MAX}")
        self.assertGreaterEqual(ler_x, 0.0)
        self.assertLessEqual(ler_x, LER_BASIS_D9_MAX, msg="LER X too high")
        self.assertGreaterEqual(ler_z, 0.0)
        self.assertLessEqual(ler_z, LER_BASIS_D9_MAX, msg="LER Z too high")

        _assert_ler_improvement_vs_baseline(self, result, tolerance=5e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d13_r9_ler_quality(self):
        """d=13, n_rounds=13, r9 model: evaluate at distance larger than R."""
        result = _run_inference_rtest(13, 13, MODEL_R9)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        self.assertAlmostEqual(ler_avg, 2e-4, delta=1.5e-4)
        self.assertAlmostEqual(ler_x, 2e-4, delta=2e-4, msg="LER X out of range")
        self.assertAlmostEqual(ler_z, 2e-4, delta=2e-4, msg="LER Z out of range")

        _assert_ler_improvement_vs_baseline(self, result)

    # -- R=13 model tests --

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d13_r13_ler_quality(self):
        """d=13, n_rounds=13, r13 model: LER avg, per-basis, improvement over baseline."""
        result = _run_inference_rtest(13, 13, MODEL_R13)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        self.assertAlmostEqual(ler_avg, 2e-4, delta=1.5e-4)
        self.assertAlmostEqual(ler_x, 2e-4, delta=2e-4, msg="LER X out of range")
        self.assertAlmostEqual(ler_z, 2e-4, delta=2e-4, msg="LER Z out of range")

        _assert_ler_improvement_vs_baseline(self, result)


if __name__ == "__main__":
    unittest.main()
