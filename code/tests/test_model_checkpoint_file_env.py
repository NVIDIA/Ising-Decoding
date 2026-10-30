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
"""PREDECODER_MODEL_CHECKPOINT_FILE selects the checkpoint that _load_model loads (CPU only)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from model.factory import ModelFactory
from training.distributed import DistributedManager
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model

REPO_ROOT = _repo_code.parent


def _inference_cfg():
    cfg = OmegaConf.load(str(REPO_ROOT / "conf" / "config_public.yaml"))
    cfg.workflow.task = "inference"
    return apply_public_defaults_and_model(cfg, validate_public_config(cfg))


def _save_random_checkpoint(cfg, path: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    state_dict = ModelFactory.create_model(cfg).state_dict()
    torch.save(state_dict, path)
    return state_dict


class TestModelCheckpointFileEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # initialize() only creates a process group when WORLD_SIZE > 1; tear down
        # only a group this class created, so other tests in the process are unaffected.
        cls._owns_process_group = not DistributedManager.is_initialized()
        DistributedManager.initialize()
        cls._owns_process_group = cls._owns_process_group and DistributedManager.is_initialized()
        cls.dist = DistributedManager()

    @classmethod
    def tearDownClass(cls):
        if cls._owns_process_group:
            torch.distributed.destroy_process_group()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cfg = _inference_cfg()
        self.cfg.enable_fp16 = False

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_weights_equal(self, model, expected_state_dict):
        loaded = model.state_dict()
        self.assertEqual(set(loaded), set(expected_state_dict))
        for name, tensor in expected_state_dict.items():
            self.assertTrue(torch.equal(loaded[name].cpu(), tensor.cpu()), msg=name)

    def test_env_var_selects_checkpoint(self):
        path = self.tmp / "from_env.pt"
        expected = _save_random_checkpoint(self.cfg, path, seed=1)
        env = {"PREDECODER_MODEL_CHECKPOINT_FILE": str(path)}
        with patch.dict(os.environ, env):
            os.environ.pop("PREDECODER_SAFETENSORS_CHECKPOINT", None)  # restored by patch.dict
            model = _load_model(self.cfg, self.dist)
        self._assert_weights_equal(model, expected)

    def test_env_var_overrides_config(self):
        env_path = self.tmp / "from_env.pt"
        cfg_path = self.tmp / "from_cfg.pt"
        expected = _save_random_checkpoint(self.cfg, env_path, seed=1)
        _save_random_checkpoint(self.cfg, cfg_path, seed=2)
        self.cfg.model_checkpoint_file = str(cfg_path)
        env = {"PREDECODER_MODEL_CHECKPOINT_FILE": str(env_path)}
        with patch.dict(os.environ, env):
            os.environ.pop("PREDECODER_SAFETENSORS_CHECKPOINT", None)  # restored by patch.dict
            model = _load_model(self.cfg, self.dist)
        self._assert_weights_equal(model, expected)

    def test_missing_env_checkpoint_raises(self):
        env = {"PREDECODER_MODEL_CHECKPOINT_FILE": str(self.tmp / "missing.pt")}
        with patch.dict(os.environ, env):
            os.environ.pop("PREDECODER_SAFETENSORS_CHECKPOINT", None)  # restored by patch.dict
            with self.assertRaises(FileNotFoundError):
                _load_model(self.cfg, self.dist)


if __name__ == "__main__":
    unittest.main()
