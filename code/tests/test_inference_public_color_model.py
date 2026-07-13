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
"""Inference tests for a pre-trained Color-Code Model 5 checkpoint.

Mirrors `test_inference_public_model.py` (surface-code) but for color:
loads `models/Ising-Decoder-ColorCode-5.pt`, runs `count_logical_errors_color`
at d=9, R=9 with a small sample count, and asserts the result schema +
sane LER values.

Unlike the surface-code checkpoints, no pre-trained color-code checkpoint is
distributed with this release, so these tests skip when the model file is
absent — except in CI, where `.github/actions/fetch-models` provides it and a
missing file must fail loudly.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.logical_error_rate_color import count_logical_errors_color
from training.distributed import DistributedManager
from workflows.run import _load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = REPO_ROOT / "models" / "Ising-Decoder-ColorCode-5.pt"


def _build_color_inference_cfg(distance: int, n_rounds: int, num_samples: int, p_error: float):
    """Build a minimal OmegaConf cfg for color-code inference.

    Matches the Color_Model_5 architecture (6-layer conv [256x5, 4], kernel 3)
    and the training-time data settings (superdense, nearest-neighbor schedule).
    No public-config validation: color stays on the internal config schema.
    """
    cfg = OmegaConf.create(
        {
            "exp_tag": "test_color_inference",
            "output": "",  # filled by the test
            "code": "color",
            "distance": distance,
            "n_rounds": n_rounds,
            "meas_basis": "both",
            "workflow": {
                "task": "inference"
            },
            "enable_fp16": False,
            "enable_bf16": False,
            "enable_matmul_tf32": True,
            "enable_cudnn_tf32": True,
            "torch_compile": False,
            "data":
                {
                    "superdense": True,
                    "schedule": "nearest-neighbor",
                    "enable_z_feedforward": True,
                    "timelike_he": False,
                    "num_he_cycles": 1,
                    "use_weight2_timelike": False,
                    "use_weight3_timelike": False,
                    "max_passes_w1": 8,
                    "max_passes_w2": 4,
                    "max_passes_w3": 4,
                    "decompose_y": False,
                    "p_error": None,
                    "p_min": 0.0009,
                    "p_max": 0.0011,
                    "error_mode": "circuit_level_color_code",
                    "precomputed_frames_dir": None,
                },
            "model":
                {
                    "version": "predecoder_memory_v1",
                    "dropout_p": 0.01,
                    "activation": "gelu",
                    "num_filters": [256, 256, 256, 256, 256, 4],
                    "kernel_size": [3, 3, 3, 3, 3, 3],
                    "input_channels": 4,
                    "out_channels": 4,
                },
            "test":
                {
                    "num_samples": num_samples,
                    "trials": 1,
                    "distance": distance,
                    "n_rounds": n_rounds,
                    "noise_model_family": "legacy",
                    "noise_instruction_semantics": "current",
                    "noise_mode": "legacy",
                    "gidney_style_noise": False,
                    "noise_model": "none",
                    "p_error": p_error,
                    "meas_basis_test": "X",
                    "use_model_checkpoint": -1,
                    "th_data": 0.0,
                    "th_syn": 0.0,
                    "sampling_mode": "threshold",
                    "temperature": 1.0,
                    "dataloader":
                        {
                            "batch_size": 64,
                            "num_workers": 0,
                            "persistent_workers": False,
                            "prefetch_factor": None,
                            "pin_memory": False,
                        },
                },
            "datapipe": "memory",
        }
    )
    return cfg


_MODEL_FILE_MISSING_MSG = (
    f"Model file missing: {MODEL_FILE}. No pre-trained color-code checkpoint "
    "is distributed with this release — train one and place it at this path "
    "(see README, 'Color code support'); in CI it is fetched by "
    ".github/actions/fetch-models."
)


class TestPublicInferenceColorModel(unittest.TestCase):

    def test_required_model_file_present(self):
        """In CI the checkpoint must be present — fail loudly; skip elsewhere."""
        if MODEL_FILE.exists():
            return
        if os.environ.get("GITHUB_ACTIONS") == "true":
            self.fail(_MODEL_FILE_MISSING_MSG)
        self.skipTest(_MODEL_FILE_MISSING_MSG)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for color inference test.")
    def test_color_inference_d9_r9_runs_and_returns_schema(self):
        """Run a small inference sweep at d=9, R=9, X basis; check result schema + sane LER."""
        if not MODEL_FILE.exists() and os.environ.get("GITHUB_ACTIONS") != "true":
            self.skipTest(_MODEL_FILE_MISSING_MSG)
        with tempfile.TemporaryDirectory(prefix="color_inference_test_") as tmpdir:
            cfg = _build_color_inference_cfg(distance=9, n_rounds=9, num_samples=256, p_error=1e-3)
            cfg.output = tmpdir
            cfg.model_checkpoint_file = str(MODEL_FILE)

            DistributedManager.initialize()
            dist = DistributedManager()
            model = _load_model(cfg, dist)
            result = count_logical_errors_color(model, dist.device, dist, cfg, log_summary=False)

        # Single-basis run → one entry keyed by "X".
        self.assertIn("X", result)
        entry = result["X"]
        for key in (
            "num_shots",
            "n_rounds",
            "logical_errors",
            "chromobius_errors",
            "logical_error_rate (mean)",
            "logical_error_rate (stderr)",
            "chromobius_error_rate (mean)",
            "chromobius_error_rate (stderr)",
        ):
            self.assertIn(key, entry, msg=f"Missing key {key!r} in result['X']")

        self.assertEqual(int(entry["num_shots"]), 256)
        self.assertEqual(int(entry["n_rounds"]), 9)

        ler = float(entry["logical_error_rate (mean)"])
        chrom_ler = float(entry["chromobius_error_rate (mean)"])
        # Per-round LERs must be finite and within [0, 1/n_rounds] (the upper
        # bound is the all-shots-failed case divided by n_rounds).
        self.assertGreaterEqual(ler, 0.0)
        self.assertLessEqual(ler, 1.0 / 9.0)
        self.assertGreaterEqual(chrom_ler, 0.0)
        self.assertLessEqual(chrom_ler, 1.0 / 9.0)


if __name__ == "__main__":
    unittest.main()
