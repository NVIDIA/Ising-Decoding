# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Round-trip tests for SafeTensors export/load utilities."""

import sys
import tempfile
import unittest
from pathlib import Path

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

        save_safetensors(model, path, model_id=self.MODEL_ID, dtype="fp32")
        loaded, metadata = load_safetensors(path, model_id=None, device="cpu")
        self.assertEqual(int(metadata["model_id"]), self.MODEL_ID)
        self.assertIsNotNone(loaded)

    def test_missing_model_id_raises(self):
        """load_safetensors should raise if metadata has no model_id and none provided."""
        from safetensors.torch import save_file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name

        dummy = {"weight": torch.zeros(4)}
        save_file(dummy, path, metadata={"quant_format": "fp32"})  # no model_id key

        with self.assertRaises(ValueError):
            load_safetensors(path, model_id=None, device="cpu")

    def test_invalid_dtype_raises(self):
        model = self._make_model("fp32")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name

        with self.assertRaises(ValueError):
            save_safetensors(model, path, model_id=self.MODEL_ID, dtype="int8")


if __name__ == "__main__":
    unittest.main()
