# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mixed-precision (AMP) helpers in ``training.precision``.

These are CPU-only: they cover the pure decision logic
(dtype selection, GradScaler / channels-last gating, fp32 BCE targets) and the
memory-format helpers, without requiring a GPU.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

torch = pytest.importorskip("torch")

from training.precision import (
    autocast_for_precision,
    get_amp_dtype,
    input_to_channels_last_3d,
    match_input_to_model_memory_format,
    model_is_channels_last_3d,
    module_to_channels_last_3d,
    should_use_channels_last_3d,
    should_use_grad_scaler,
    targets_for_bce,
    validate_precision_flags,
)

CPU = torch.device("cpu")


def test_validate_precision_flags_mutually_exclusive():
    validate_precision_flags(True, False)  # ok
    validate_precision_flags(False, True)  # ok
    validate_precision_flags(False, False)  # ok
    with pytest.raises(ValueError):
        validate_precision_flags(True, True)


def test_get_amp_dtype():
    assert get_amp_dtype(True, False) is torch.float16
    assert get_amp_dtype(False, True) is torch.bfloat16
    assert get_amp_dtype(False, False) is None


def test_targets_for_bce_is_fp32():
    t = torch.zeros(2, 3, dtype=torch.int64)
    assert targets_for_bce(t).dtype is torch.float32


def test_grad_scaler_gating_cpu():
    # GradScaler is only for CUDA fp16; never enabled on CPU.
    assert should_use_grad_scaler(True, CPU) is False
    assert should_use_grad_scaler(False, CPU) is False


def test_channels_last_gating_cpu():
    # channels_last_3d is a CUDA-only optimization; off on CPU.
    assert should_use_channels_last_3d(True, False, CPU) is False
    assert should_use_channels_last_3d(False, True, CPU) is False
    assert should_use_channels_last_3d(False, False, CPU) is False


def test_autocast_for_precision_contexts():
    # No AMP -> no-op context.
    with autocast_for_precision(CPU, False, False):
        pass
    # CPU + fp16 is unsupported -> falls back to a no-op context (must not raise).
    with autocast_for_precision(CPU, True, False):
        pass
    # CPU + bf16 -> a real autocast context that is usable.
    with autocast_for_precision(CPU, False, True):
        pass


def test_input_to_channels_last_3d_noop_when_disabled():
    x = torch.randn(1, 4, 3, 5, 5)
    assert input_to_channels_last_3d(x, False) is x
    # Non-5D input is left alone even when enabled.
    x4 = torch.randn(1, 4, 5, 5)
    assert input_to_channels_last_3d(x4, True) is x4


def test_input_to_channels_last_3d_converts_5d():
    x = torch.randn(2, 4, 3, 5, 5)
    out = input_to_channels_last_3d(x, True)
    assert out.is_contiguous(memory_format=torch.channels_last_3d)


def test_model_layout_helpers_roundtrip():
    conv = torch.nn.Conv3d(4, 8, kernel_size=3)
    assert model_is_channels_last_3d(conv) is False
    conv = module_to_channels_last_3d(conv, True)
    assert model_is_channels_last_3d(conv) is True

    # A contiguous eval input gets matched to the channels_last_3d model.
    x = torch.randn(1, 4, 3, 5, 5)
    out = match_input_to_model_memory_format(x, conv)
    assert out.is_contiguous(memory_format=torch.channels_last_3d)


def test_match_input_noop_for_contiguous_model():
    conv = torch.nn.Conv3d(4, 8, kernel_size=3)  # default contiguous
    x = torch.randn(1, 4, 3, 5, 5)
    assert match_input_to_model_memory_format(x, conv) is x


# unittest.TestCase so CI's `unittest discover` collects it (the pytest-style
# tests above are only run by pytest).
class TestMatchInputDuringOnnxExport(unittest.TestCase):
    """The layout-match helper must be a no-op inside torch.onnx.export.

    Regression test: an unguarded contiguous(memory_format=channels_last_3d)
    in a traced forward makes the legacy exporter fail with
    "onnx memory_format support is not implemented".
    """

    def test_match_input_skipped_during_onnx_export(self):
        # The legacy TorchScript exporter needs the onnx package to serialize.
        try:
            from onnx.reference import ReferenceEvaluator
        except ImportError:
            self.skipTest("onnx is not installed")

        class Wrap(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.model = module_to_channels_last_3d(torch.nn.Conv3d(4, 8, kernel_size=3), True)

            def forward(self, x):
                x = match_input_to_model_memory_format(x, self.model)
                return self.model(x)

        wrap = Wrap().eval()
        x = torch.randn(1, 4, 3, 8, 8)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "wrap.onnx"
            # Without the in-export guard the legacy exporter raises
            # SymbolicValueError: "onnx memory_format support is not implemented".
            torch.onnx.export(
                wrap, (x,), str(out_path), opset_version=18, input_names=["x"], dynamo=False
            )
            self.assertTrue(out_path.exists())
            # Skipping the layout conversion is value-neutral: the exported
            # graph must reproduce the eager output.
            (onnx_out,) = ReferenceEvaluator(str(out_path)).run(None, {"x": x.numpy()})
        with torch.no_grad():
            eager = wrap(x)
        self.assertTrue(torch.allclose(eager, torch.from_numpy(onnx_out), atol=1e-5))
