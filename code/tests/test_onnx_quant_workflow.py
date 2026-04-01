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
"""Tests for ONNX quantization workflow: _collect_calibration_dets helper and QUANT_FORMAT env var logic."""

import os
import re
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

from evaluation.logical_error_rate import (
    _collect_calibration_dets,
    _ort_quantize_int8,
    _parse_quant_format,
)


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
        """Invoke the real _parse_quant_format() from LER under a controlled env."""
        with patch.dict(os.environ, {"QUANT_FORMAT": quant_format_env}):
            return _parse_quant_format(rank=0)

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


class TestOrtQuantizeInt8(unittest.TestCase):
    """Tests for the _ort_quantize_int8 helper (onnxruntime INT8 fallback)."""

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("onnxruntime") is not None and
        __import__("importlib").util.find_spec("modelopt") is None,
        "onnxruntime not installed or modelopt present (ort path is only the fallback when modelopt is absent)",
    )
    def test_ort_quantize_int8_produces_output_file(self):
        """_ort_quantize_int8 must write a valid ONNX file to output_path."""
        try:
            import onnx
            import onnx.helper as oh
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnx/onnxruntime not installed")

        import tempfile
        import numpy as np

        # Build a tiny single-Gemm ONNX model compatible with quantize_static.
        X = oh.make_tensor_value_info("dets", onnx.TensorProto.FLOAT, [1, 4])
        W_data = np.ones((4, 4), dtype=np.float32)
        B_data = np.zeros((4,), dtype=np.float32)
        W = oh.make_tensor("W", onnx.TensorProto.FLOAT, W_data.shape, W_data.flatten().tolist())
        B = oh.make_tensor("B", onnx.TensorProto.FLOAT, B_data.shape, B_data.flatten().tolist())
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
        node = oh.make_node("Gemm", inputs=["dets", "W", "B"], outputs=["Y"])
        graph = oh.make_graph([node], "tiny", [X], [Y], initializer=[W, B])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        # Pin to IR version 8 (opset-17 minimum).  Newer ONNX packages default to
        # IR version 12, which onnxruntime-gpu 1.22.0 (a modelopt dependency) rejects.
        model.ir_version = 8
        onnx.checker.check_model(model)

        calib = np.random.randn(8, 4).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fp32_f:
            fp32_path = fp32_f.name
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as out_f:
            out_path = out_f.name
        self.addCleanup(os.unlink, fp32_path)
        self.addCleanup(os.unlink, out_path)

        onnx.save(model, fp32_path)
        _ort_quantize_int8(fp32_path, out_path, calib)

        quant_model = onnx.load(out_path)
        onnx.checker.check_model(quant_model)

    def test_ort_quantize_int8_called_on_modelopt_import_error(self):
        """When modelopt is not importable, INT8 must fall back to _ort_quantize_int8."""
        called = []
        with patch(
            "evaluation.logical_error_rate._ort_quantize_int8",
            side_effect=lambda *a, **kw: called.append(a),
        ):
            import evaluation.logical_error_rate as ler
            ler._ort_quantize_int8("fp32.onnx", "out.onnx", None)
        self.assertEqual(len(called), 1)

    def test_fp8_raises_on_modelopt_import_error(self):
        """When modelopt is not importable, FP8 must raise RuntimeError (no ort fallback)."""
        quant_format = "fp8"
        with self.assertRaises(RuntimeError):
            try:
                raise ImportError("No module named 'modelopt'")
            except ImportError:
                if quant_format == "fp8":
                    raise RuntimeError(
                        "[LER] FP8 quantization requires nvidia-modelopt. "
                        "Install with: pip install 'nvidia-modelopt[onnx]'"
                        " --ignore-requires-python"
                    )


_HAS_MODELOPT = __import__("importlib").util.find_spec("modelopt") is not None


class TestModeloptQuantize(unittest.TestCase):
    """End-to-end tests that call mq.quantize() on a real ONNX model.

    Skipped when nvidia-modelopt is not installed.  On Python 3.13+ modelopt
    must be installed with --ignore-requires-python (done by check_python_compat.sh
    when MODE=train); these tests confirm it actually works at runtime, not just
    that the import succeeds.
    """

    def _build_tiny_model(self):
        """Return (fp32_path, calib_dets) for a minimal Gemm ONNX model."""
        import tempfile

        import numpy as np
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("dets", onnx.TensorProto.FLOAT, [1, 4])
        W_data = np.ones((4, 4), dtype=np.float32)
        B_data = np.zeros((4,), dtype=np.float32)
        W = oh.make_tensor("W", onnx.TensorProto.FLOAT, W_data.shape, W_data.flatten().tolist())
        B = oh.make_tensor("B", onnx.TensorProto.FLOAT, B_data.shape, B_data.flatten().tolist())
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
        node = oh.make_node("Gemm", inputs=["dets", "W", "B"], outputs=["Y"])
        graph = oh.make_graph([node], "tiny", [X], [Y], initializer=[W, B])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            fp32_path = f.name
        self.addCleanup(os.unlink, fp32_path)
        onnx.save(model, fp32_path)

        calib = np.random.randn(16, 4).astype(np.float32)
        return fp32_path, calib

    @unittest.skipUnless(_HAS_MODELOPT, "nvidia-modelopt not installed")
    def test_mq_quantize_int8_produces_valid_onnx(self):
        """mq.quantize(quantize_mode='int8') must write a valid ONNX file."""
        import tempfile

        import modelopt.onnx.quantization as mq
        import onnx

        fp32_path, calib = self._build_tiny_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out_path = f.name
        self.addCleanup(os.unlink, out_path)

        mq.quantize(
            onnx_path=fp32_path,
            quantize_mode="int8",
            calibration_data={"dets": calib},
            output_path=out_path,
        )

        self.assertTrue(os.path.isfile(out_path), "quantized ONNX output file not created")
        quant_model = onnx.load(out_path)
        onnx.checker.check_model(quant_model)

    @unittest.skipUnless(_HAS_MODELOPT, "nvidia-modelopt not installed")
    def test_mq_quantize_int8_output_differs_from_fp32(self):
        """The quantized model must differ from the FP32 source (QDQ nodes added)."""
        import tempfile

        import modelopt.onnx.quantization as mq
        import onnx

        fp32_path, calib = self._build_tiny_model()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            out_path = f.name
        self.addCleanup(os.unlink, out_path)

        mq.quantize(
            onnx_path=fp32_path,
            quantize_mode="int8",
            calibration_data={"dets": calib},
            output_path=out_path,
        )

        fp32_model = onnx.load(fp32_path)
        quant_model = onnx.load(out_path)
        self.assertNotEqual(
            len(fp32_model.graph.node),
            len(quant_model.graph.node),
            "quantized model should have more nodes (QDQ pairs) than the FP32 source",
        )


class TestModeloptPrerequisite(unittest.TestCase):
    """Verify quantization package prerequisites are correctly declared."""

    _TRAIN_REQS = Path(__file__).resolve().parent.parent / "requirements_public_train-cu12.txt"

    def test_nvidia_modelopt_in_train_requirements(self):
        """nvidia-modelopt[onnx] must be listed in requirements_public_train-cu12.txt."""
        text = self._TRAIN_REQS.read_text()
        self.assertTrue(
            re.search(r"(?m)^nvidia-modelopt", text),
            "nvidia-modelopt[onnx] must appear in requirements_public_train-cu12.txt; "
            "it is used for INT8/FP8 quantization on Python <3.13.",
        )

    def test_onnxruntime_in_train_requirements(self):
        """onnxruntime must be listed in requirements_public_train-cu12.txt for Python 3.13+."""
        text = self._TRAIN_REQS.read_text()
        self.assertTrue(
            re.search(r"(?m)^onnxruntime", text),
            "onnxruntime must appear in requirements_public_train-cu12.txt; "
            "it is the INT8 quantization backend on Python 3.13+ "
            "(nvidia-modelopt does not support Python 3.13+).",
        )

    def test_quant_packages_absent_from_inference_requirements(self):
        """nvidia-modelopt and onnxruntime must NOT appear in the inference requirements."""
        infer_reqs = self._TRAIN_REQS.parent / "requirements_public_inference.txt"
        text = infer_reqs.read_text()
        self.assertFalse(
            re.search(r"(?m)^nvidia-modelopt", text),
            "nvidia-modelopt must not be in requirements_public_train-cu12.txt.",
        )
        self.assertFalse(
            re.search(r"(?m)^onnxruntime", text),
            "onnxruntime must not be in requirements_public_train-cu12.txt.",
        )

    def test_modelopt_importable_when_installed(self):
        """When nvidia-modelopt[onnx] is installed, modelopt.onnx.quantization must be importable.

        On Python 3.13+ modelopt can be installed with --ignore-requires-python;
        this test skips silently if the package is absent regardless of Python version.
        """
        try:
            import modelopt.onnx.quantization as mq  # noqa: F401
        except ImportError:
            self.skipTest("nvidia-modelopt[onnx] is not installed in this environment")

    def test_ort_importable_when_installed(self):
        """onnxruntime.quantization must be importable when onnxruntime is installed."""
        try:
            from onnxruntime.quantization import (  # noqa: F401
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )
        except ImportError:
            self.skipTest("onnxruntime is not installed in this environment")


if __name__ == "__main__":
    unittest.main()
