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

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

from omegaconf import OmegaConf

# Ensure repo's code/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.registry import compute_receptive_field, get_model_spec
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config


class TestPublicConfig(unittest.TestCase):

    def test_registry_receptive_field_formula(self):
        self.assertEqual(compute_receptive_field([3, 3, 3, 3]), 9)
        self.assertEqual(compute_receptive_field([5, 5, 5, 5]), 17)
        self.assertEqual(compute_receptive_field([3, 3, 3, 3, 3, 3]), 13)

    def test_model_id_validation(self):
        spec = get_model_spec(1)
        self.assertEqual(spec.model_id, 1)
        self.assertEqual(spec.receptive_field, 9)

        with self.assertRaises(ValueError):
            get_model_spec(0)
        with self.assertRaises(ValueError):
            get_model_spec(6)

    def test_validate_rejects_hidden_fields(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "enable_fp16": True,  # forbidden
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_validate_rejects_any_train_overrides(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "train": {
                    "epochs": 1
                },
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_validate_rejects_output_and_resume_dir_overrides(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "output": "/tmp/out",
                "resume_dir": "/tmp/out/models",
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_validate_rejects_extra_data_fields(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "data": {
                    "p_min": 0.001
                },  # forbidden key
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_validate_rejects_precomputed_frames_dir_override(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "data": {
                    "precomputed_frames_dir": "/tmp/frames"
                },  # hidden in public release
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_validate_accepts_noise_model(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "data":
                    {
                        "noise_model":
                            {
                                "p_prep_X": 0.001,
                                "p_prep_Z": 0.001,
                                "p_meas_X": 0.001,
                                "p_meas_Z": 0.001,
                                "p_idle_cnot_X": 0.001,
                                "p_idle_cnot_Y": 0.001,
                                "p_idle_cnot_Z": 0.001,
                                "p_idle_spam_X": 0.001,
                                "p_idle_spam_Y": 0.001,
                                "p_idle_spam_Z": 0.001,
                                "p_cnot_IX": 0.0001,
                                "p_cnot_IY": 0.0001,
                                "p_cnot_IZ": 0.0001,
                                "p_cnot_XI": 0.0001,
                                "p_cnot_XX": 0.0001,
                                "p_cnot_XY": 0.0001,
                                "p_cnot_XZ": 0.0001,
                                "p_cnot_YI": 0.0001,
                                "p_cnot_YX": 0.0001,
                                "p_cnot_YY": 0.0001,
                                "p_cnot_YZ": 0.0001,
                                "p_cnot_ZI": 0.0001,
                                "p_cnot_ZX": 0.0001,
                                "p_cnot_ZY": 0.0001,
                                "p_cnot_ZZ": 0.0001,
                            }
                    },
            }
        )
        spec = validate_public_config(cfg)
        merged = apply_public_defaults_and_model(cfg, spec)
        self.assertIsNotNone(merged.data.noise_model)
        # Hidden default: always points to <repo>/frames_data, independent of user cwd.
        repo_root = Path(__file__).resolve().parents[2]
        expected_frames_dir = (repo_root / "frames_data").resolve()
        self.assertEqual(Path(merged.data.precomputed_frames_dir).resolve(), expected_frames_dir)

    def test_validate_rejects_optimizer_subfields(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "optimizer": {
                    "lr": 1e-4,
                    "beta2": 0.9
                },  # forbidden key
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_optimizer_lr_is_hardcoded_by_model_id(self):
        # Values must match the public release mapping in config_validator.py
        expected = {
            1: 3e-4,
            2: 2e-4,
            3: 1e-4,
            4: 2e-4,
            5: 1e-4,
        }
        for model_id, lr in expected.items():
            cfg = OmegaConf.create({"model_id": model_id, "distance": 9, "n_rounds": 9})
            spec = validate_public_config(cfg)
            merged = apply_public_defaults_and_model(cfg, spec)
            self.assertAlmostEqual(float(merged.optimizer.lr), float(lr), places=12)

    def test_validate_rejects_any_test_overrides(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "test": {
                    "trials": 7
                },
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_validate_rejects_any_val_overrides(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 9,
                "n_rounds": 9,
                "val": {
                    "num_samples": 1
                },
            }
        )
        with self.assertRaises(ValueError):
            validate_public_config(cfg)

    def test_distance_rounds_clamping_model1(self):
        cfg = OmegaConf.create({"model_id": 1, "distance": 11, "n_rounds": 11})
        spec = validate_public_config(cfg)
        merged = apply_public_defaults_and_model(cfg, spec)
        # Training always uses the receptive field (R=9 for model 1)
        self.assertEqual(int(merged.distance), 9)
        self.assertEqual(int(merged.n_rounds), 9)
        # Evaluation uses the user-specified targets
        self.assertEqual(int(merged.test.distance), 11)
        self.assertEqual(int(merged.test.n_rounds), 11)
        self.assertEqual(int(merged.test.num_samples), 262144)
        # Validation uses a fixed sample count
        self.assertEqual(int(merged.val.num_samples), 65536)

    def test_distance_rounds_clamping_model3_asymmetric(self):
        cfg = OmegaConf.create({"model_id": 3, "distance": 15, "n_rounds": 19})
        spec = validate_public_config(cfg)
        merged = apply_public_defaults_and_model(cfg, spec)
        # Training always uses the receptive field (R=17 for model 3)
        self.assertEqual(int(merged.distance), 17)
        self.assertEqual(int(merged.n_rounds), 17)
        # Evaluation uses the user-specified targets
        self.assertEqual(int(merged.test.distance), 15)
        self.assertEqual(int(merged.test.n_rounds), 19)
        self.assertEqual(int(merged.test.num_samples), 262144)

    def test_inference_uses_user_distance_rounds(self):
        cfg = OmegaConf.create(
            {
                "model_id": 1,
                "distance": 11,
                "n_rounds": 13,
                "workflow": {
                    "task": "inference"
                },
            }
        )
        spec = validate_public_config(cfg)
        merged = apply_public_defaults_and_model(cfg, spec)
        # In inference mode, top-level distance/n_rounds are the evaluation targets.
        self.assertEqual(int(merged.distance), 11)
        self.assertEqual(int(merged.n_rounds), 13)
        # And the hidden test config matches those values.
        self.assertEqual(int(merged.test.distance), 11)
        self.assertEqual(int(merged.test.n_rounds), 13)
        self.assertEqual(int(merged.test.num_samples), 262144)

    def test_hidden_defaults_are_populated(self):
        # Minimal public cfg should still yield a training-ready merged config.
        cfg = OmegaConf.create({"model_id": 2, "distance": 9, "n_rounds": 9})
        spec = validate_public_config(cfg)
        merged = apply_public_defaults_and_model(cfg, spec)

        # Precision + tf32 enforced
        self.assertFalse(bool(merged.enable_fp16))
        self.assertFalse(bool(merged.enable_bf16))
        self.assertTrue(bool(merged.enable_matmul_tf32))
        self.assertTrue(bool(merged.enable_cudnn_tf32))

        # Always both bases
        self.assertEqual(str(merged.meas_basis), "both")

        # Training expects these blocks to exist (taken from hidden defaults)
        self.assertIn("optimizer_type", merged)
        self.assertIn("lr_scheduler", merged)
        self.assertIn("batch_schedule", merged)
        self.assertIn("ema", merged)
        self.assertIn("time_based_early_stopping", merged)

        # And required leaf fields for training code paths
        self.assertIn("lr", merged.optimizer)
        self.assertIn("weight_decay", merged.optimizer)
        self.assertIn("beta2", merged.optimizer)
        self.assertIn("type", merged.lr_scheduler)
        self.assertIn("enabled", merged.batch_schedule)
        self.assertIn("use_ema", merged.ema)

        # Training epochs are fixed in the public release.
        self.assertEqual(int(merged.train.epochs), 100)

    def test_torch_compile_env_override(self):
        cfg = OmegaConf.create({"model_id": 1, "distance": 9, "n_rounds": 9})
        with patch.dict(
            "os.environ", {
                "PREDECODER_TORCH_COMPILE": "0",
                "PREDECODER_TORCH_COMPILE_MODE": "reduce-overhead"
            }
        ):
            spec = validate_public_config(cfg)
            merged = apply_public_defaults_and_model(cfg, spec)
            self.assertFalse(bool(merged.torch_compile))
            self.assertEqual(str(merged.torch_compile_mode), "reduce-overhead")

    def test_code_rotation_o1_to_o4_accepted_and_normalized(self):
        """All four public orientations O1..O4 are accepted and normalized to internal names."""
        expected = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}
        for public, internal in expected.items():
            with self.subTest(code_rotation=public):
                cfg = OmegaConf.create(
                    {
                        "model_id": 1,
                        "distance": 9,
                        "n_rounds": 9,
                        "data": {
                            "code_rotation": public
                        },
                    }
                )
                spec = validate_public_config(cfg)
                merged = apply_public_defaults_and_model(cfg, spec)
                self.assertEqual(
                    str(merged.data.code_rotation),
                    internal,
                    f"data.code_rotation {public} should normalize to {internal}",
                )

    def test_code_rotation_internal_aliases_accepted(self):
        """Internal names XV, XH, ZV, ZH are also accepted (e.g. for compatibility)."""
        for rotation in ["XV", "XH", "ZV", "ZH"]:
            with self.subTest(code_rotation=rotation):
                cfg = OmegaConf.create(
                    {
                        "model_id": 1,
                        "distance": 9,
                        "n_rounds": 9,
                        "data": {
                            "code_rotation": rotation
                        },
                    }
                )
                spec = validate_public_config(cfg)
                merged = apply_public_defaults_and_model(cfg, spec)
                self.assertEqual(str(merged.data.code_rotation), rotation)


if __name__ == "__main__":
    unittest.main()
