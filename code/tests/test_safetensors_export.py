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
"""Round-trip tests for SafeTensors export/load utilities."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from export.safetensors_utils import save_safetensors, load_safetensors, _build_minimal_cfg
from model.factory import ModelFactory


class TestSafeTensorsRoundTrip(unittest.TestCase):
    """Test save_safetensors / load_safetensors round-trip for fp32 and fp16."""

    # model_id=1 is the smallest public model — fast to instantiate on CPU.
    MODEL_ID = 1

    def _make_model(self, dtype: str) -> torch.nn.Module:
        cfg = _build_minimal_cfg(self.MODEL_ID)
        model = ModelFactory.create_model(cfg)
        if dtype == "fp16":
            model = model.half()
        return model

    def _assert_state_dicts_close(self, a: dict, b: dict, atol: float):
        self.assertEqual(set(a.keys()), set(b.keys()))
        for key in a:
            torch.testing.assert_close(a[key].float(), b[key].float(), atol=atol, rtol=0)

    def test_round_trip_fp32(self):
        model = self._make_model("fp32")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        save_safetensors(model, path, model_id=self.MODEL_ID, dtype="fp32")
        loaded, metadata = load_safetensors(path, device="cpu")

        self.assertEqual(metadata["quant_format"], "fp32")
        self.assertEqual(int(metadata["model_id"]), self.MODEL_ID)
        self.assertEqual(next(iter(loaded.parameters())).dtype, torch.float32)
        self._assert_state_dicts_close(model.state_dict(), loaded.state_dict(), atol=0.0)

    def test_round_trip_fp16(self):
        model = self._make_model("fp16")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        save_safetensors(model, path, model_id=self.MODEL_ID, dtype="fp16")
        loaded, metadata = load_safetensors(path, device="cpu")

        self.assertEqual(metadata["quant_format"], "fp16")
        self.assertEqual(int(metadata["model_id"]), self.MODEL_ID)
        self.assertEqual(next(iter(loaded.parameters())).dtype, torch.float16)
        # fp16 round-trip: weights should be bit-exact (no lossy conversion)
        self._assert_state_dicts_close(model.state_dict(), loaded.state_dict(), atol=0.0)

    def test_metadata_model_id_auto_detect(self):
        """load_safetensors with model_id=None should resolve from file metadata."""
        model = self._make_model("fp32")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        save_safetensors(model, path, model_id=self.MODEL_ID, dtype="fp32")
        loaded, metadata = load_safetensors(path, model_id=None, device="cpu")
        self.assertEqual(int(metadata["model_id"]), self.MODEL_ID)
        self.assertIsNotNone(loaded)

    def test_missing_model_id_raises(self):
        """load_safetensors should raise if metadata has no model_id and none provided."""
        from safetensors.torch import save_file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        dummy = {"weight": torch.zeros(4)}
        save_file(dummy, path, metadata={"quant_format": "fp32"})  # no model_id key

        with self.assertRaises(ValueError):
            load_safetensors(path, model_id=None, device="cpu")

    def test_invalid_dtype_raises(self):
        model = self._make_model("fp32")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        with self.assertRaises(ValueError):
            save_safetensors(model, path, model_id=self.MODEL_ID, dtype="int8")


class TestSafeTensorsRunPyIntegration(unittest.TestCase):
    """Test PREDECODER_SAFETENSORS_CHECKPOINT code path in workflows/run.py."""

    MODEL_ID = 1

    def _make_dist(self):
        dist = types.SimpleNamespace(rank=0, device=torch.device("cpu"))
        return dist

    def _make_cfg(self):
        from omegaconf import OmegaConf
        cfg = _build_minimal_cfg(self.MODEL_ID)
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "workflow": {
                    "task": "inference"
                },
                "enable_fp16": False,
            }),
        )
        return cfg

    def _run_load_model(self, safetensors_path, cfg=None, dist=None):
        from workflows.run import _load_model
        if cfg is None:
            cfg = self._make_cfg()
        if dist is None:
            dist = self._make_dist()
        # _ensure_inference_io_channels requires a full data pipeline; patch it out
        # since we are only testing the SafeTensors loading code path.
        with patch("workflows.run._ensure_inference_io_channels"), \
             patch.dict(os.environ, {"PREDECODER_SAFETENSORS_CHECKPOINT": safetensors_path}):
            return _load_model(cfg, dist), cfg

    def test_env_var_loads_fp32_model(self):
        """PREDECODER_SAFETENSORS_CHECKPOINT causes _load_model to load from .safetensors."""
        model = ModelFactory.create_model(_build_minimal_cfg(self.MODEL_ID))
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        save_safetensors(model, path, model_id=self.MODEL_ID, dtype="fp32")
        loaded, cfg = self._run_load_model(path)

        self.assertEqual(next(iter(loaded.parameters())).dtype, torch.float32)
        self.assertFalse(cfg.enable_fp16)
        # Weights must match the original model
        for key in model.state_dict():
            torch.testing.assert_close(
                model.state_dict()[key].float(),
                loaded.state_dict()[key].float(),
                atol=0.0,
                rtol=0,
            )

    def test_env_var_loads_fp16_model_and_sets_flag(self):
        """fp16 .safetensors sets cfg.enable_fp16 and returns a half-precision model."""
        model = ModelFactory.create_model(_build_minimal_cfg(self.MODEL_ID)).half()
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)

        save_safetensors(model, path, model_id=self.MODEL_ID, dtype="fp16")
        loaded, cfg = self._run_load_model(path)

        self.assertEqual(next(iter(loaded.parameters())).dtype, torch.float16)
        self.assertTrue(cfg.enable_fp16)

    def test_empty_env_var_skips_safetensors_path(self):
        """When env var is unset, _load_model falls through to the normal .pt path."""
        from workflows.run import _load_model
        cfg = self._make_cfg()
        dist = self._make_dist()

        with patch("workflows.run._ensure_inference_io_channels"), \
             patch("workflows.run.ModelFactory") as mock_factory, \
             patch.dict(os.environ, {"PREDECODER_SAFETENSORS_CHECKPOINT": ""}):
            mock_model = unittest.mock.MagicMock()
            mock_factory.create_model.return_value = mock_model
            mock_model.to.return_value = mock_model
            # The normal path raises because no checkpoint dir exists; that is expected.
            try:
                _load_model(cfg, dist)
            except Exception:
                pass
            # Key assertion: ModelFactory.create_model was called (not the safetensors path)
            mock_factory.create_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
