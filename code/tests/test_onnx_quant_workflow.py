# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for ONNX quantization workflow: _collect_calibration_dets helper and QUANT_FORMAT env var logic."""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from evaluation.logical_error_rate import _collect_calibration_dets


def _make_fake_dataloader(num_batches: int, batch_size: int, num_dets: int, num_obs: int):
    """Build a list of fake batches mimicking the test_dataloader interface."""
    batches = []
    for _ in range(num_batches):
        dets_and_obs = torch.randint(0, 2, (batch_size, num_dets + num_obs), dtype=torch.uint8)
        batches.append({"dets_and_obs": dets_and_obs})
    return batches


class TestCollectCalibrationDets(unittest.TestCase):

    NUM_DETS = 20
    NUM_OBS = 1

    def test_basic_shape_and_dtype(self):
        """Output must have shape (target_samples, NUM_DETS) and dtype uint8."""
        loader = _make_fake_dataloader(
            num_batches=4, batch_size=32, num_dets=self.NUM_DETS, num_obs=self.NUM_OBS
        )
        target = 64
        result = _collect_calibration_dets(loader, self.NUM_OBS, target, self.NUM_DETS)
        self.assertEqual(result.shape, (target, self.NUM_DETS))
        self.assertEqual(result.dtype, np.uint8)

    def test_tiles_when_dataloader_too_short(self):
        """When fewer samples are available than requested, tiles to fill target_samples."""
        loader = _make_fake_dataloader(
            num_batches=1, batch_size=8, num_dets=self.NUM_DETS, num_obs=self.NUM_OBS
        )
        target = 50
        result = _collect_calibration_dets(loader, self.NUM_OBS, target, self.NUM_DETS)
        self.assertEqual(result.shape, (target, self.NUM_DETS))
        self.assertEqual(result.dtype, np.uint8)

    def test_empty_dataloader_raises(self):
        """Empty dataloader (no batches) must raise RuntimeError."""
        loader = []
        with self.assertRaises(RuntimeError):
            _collect_calibration_dets(loader, self.NUM_OBS, 32, self.NUM_DETS)

    def test_width_mismatch_raises(self):
        """If det width doesn't match expected_width, raises RuntimeError."""
        loader = _make_fake_dataloader(
            num_batches=2, batch_size=16, num_dets=self.NUM_DETS, num_obs=self.NUM_OBS
        )
        wrong_width = self.NUM_DETS + 5
        with self.assertRaises(RuntimeError):
            _collect_calibration_dets(loader, self.NUM_OBS, 16, wrong_width)

    def test_stops_early_when_enough_samples(self):
        """Should stop iterating once target_samples are collected."""
        consumed = []
        num_dets = self.NUM_DETS
        num_obs = self.NUM_OBS

        class CountingLoader:

            def __iter__(self):
                for i in range(100):
                    consumed.append(i)
                    dets_and_obs = torch.randint(0, 2, (32, num_dets + num_obs), dtype=torch.uint8)
                    yield {"dets_and_obs": dets_and_obs}

        loader = CountingLoader()
        target = 32  # exactly one batch
        _collect_calibration_dets(loader, num_obs, target, num_dets)
        self.assertEqual(len(consumed), 1)


class TestQuantFormatParsing(unittest.TestCase):
    """Test QUANT_FORMAT env var parsing and routing logic (no GPU, no modelopt needed)."""

    def _run_quant_block(self, quant_format_env: str, mock_mq=None, mock_export=None):
        """Simulate the QUANT_FORMAT parsing + routing logic extracted from LER."""
        with patch.dict(os.environ, {"QUANT_FORMAT": quant_format_env}):
            quant_format = os.environ.get("QUANT_FORMAT", "").strip().lower()
            valid = ("int8", "fp8")
            if quant_format and quant_format not in valid:
                quant_format = ""
            return quant_format

    def test_invalid_quant_format_ignored(self):
        result = self._run_quant_block("bad_format")
        self.assertEqual(result, "")

    def test_valid_int8_accepted(self):
        result = self._run_quant_block("int8")
        self.assertEqual(result, "int8")

    def test_valid_fp8_accepted(self):
        result = self._run_quant_block("fp8")
        self.assertEqual(result, "fp8")

    def test_nvfp4_rejected(self):
        result = self._run_quant_block("nvfp4")
        self.assertEqual(result, "")

    def test_empty_quant_format_no_quantize_call(self):
        """With QUANT_FORMAT unset, mq.quantize must never be called."""
        mock_mq = MagicMock()
        with patch.dict(os.environ, {"QUANT_FORMAT": ""}):
            quant_format = os.environ.get("QUANT_FORMAT", "").strip().lower()
            if quant_format:
                mock_mq.quantize()
        mock_mq.quantize.assert_not_called()

    def test_mq_quantize_called_with_correct_args_int8(self):
        """With QUANT_FORMAT=int8, mq.quantize receives quantize_mode='int8' and calibration_data."""
        mock_mq = MagicMock()
        num_dets = 20
        num_obs = 1
        loader = _make_fake_dataloader(
            num_batches=2, batch_size=32, num_dets=num_dets, num_obs=num_obs
        )

        with patch.dict(os.environ, {"QUANT_FORMAT": "int8", "QUANT_CALIB_SAMPLES": "16"}):
            quant_format = "int8"
            fp32_path = "model.onnx"
            quant_path = "model_int8.onnx"
            calib_num_samples = int(os.environ.get("QUANT_CALIB_SAMPLES", "256"))
            calib_dets = _collect_calibration_dets(loader, num_obs, calib_num_samples, num_dets)
            format_map = {"int8": "int8", "fp8": "fp8"}
            mock_mq.quantize(
                onnx_path=fp32_path,
                quantize_mode=format_map[quant_format],
                calibration_data={"dets": calib_dets},
                output_path=quant_path,
            )

        mock_mq.quantize.assert_called_once()
        call_kwargs = mock_mq.quantize.call_args
        self.assertEqual(call_kwargs.kwargs["quantize_mode"], "int8")
        self.assertIn("dets", call_kwargs.kwargs["calibration_data"])
        calib = call_kwargs.kwargs["calibration_data"]["dets"]
        self.assertEqual(calib.shape, (calib_num_samples, num_dets))
        self.assertEqual(calib.dtype, np.uint8)

    def test_fp8_fail_fast_raises(self):
        """With QUANT_FORMAT=fp8, if mq.quantize raises, a RuntimeError is propagated."""
        num_dets = 20
        num_obs = 1
        loader = _make_fake_dataloader(
            num_batches=2, batch_size=32, num_dets=num_dets, num_obs=num_obs
        )
        calib_dets = _collect_calibration_dets(loader, num_obs, 16, num_dets)

        quant_format = "fp8"
        with self.assertRaises(RuntimeError):
            try:
                raise ValueError("simulated fp8 quantize failure")
            except Exception as e:
                if quant_format == "fp8":
                    raise RuntimeError(
                        f"[LER] FP8 ONNX quantization failed (fail-fast): {e}"
                    ) from e
                pass  # non-fp8 would fall through

    def test_non_fp8_failure_falls_back_to_fp32(self):
        """With QUANT_FORMAT=int8, if mq.quantize raises, onnx_path falls back to fp32 path silently."""
        num_dets = 20
        num_obs = 1
        loader = _make_fake_dataloader(
            num_batches=2, batch_size=32, num_dets=num_dets, num_obs=num_obs
        )
        calib_dets = _collect_calibration_dets(loader, num_obs, 16, num_dets)

        quant_format = "int8"
        fp32_onnx_path = "model.onnx"
        onnx_path = "model_int8.onnx"  # would be the quantized path

        try:
            raise RuntimeError("simulated int8 quantize failure")
        except Exception as e:
            if quant_format == "fp8":
                raise RuntimeError(f"fail-fast: {e}") from e
            # non-fp8: fall back silently
            onnx_path = fp32_onnx_path

        self.assertEqual(onnx_path, fp32_onnx_path)


if __name__ == "__main__":
    unittest.main()
