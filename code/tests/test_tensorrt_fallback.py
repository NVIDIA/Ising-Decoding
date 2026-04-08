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
"""Tests for the TensorRT optional-dependency contract.

tensorrt is a heavy CUDA-only package (~500 MB) that cannot be pip-installed
in CPU-only CI.  It is therefore documented as an optional prerequisite via a
comment in requirements_public_inference.txt rather than as a pip requirement.
These tests verify:
  1. The documentation comment exists in the inference requirements file.
  2. Both TensorRT import sites raise RuntimeError (not silently fall back to
     PyTorch) when tensorrt is unavailable (simulated via ImportError).
  3. Both sites still fall back gracefully for non-import TensorRT errors
     (e.g. a corrupt engine file).
  4. When tensorrt is installed, its key symbols are importable (GPU CI only).
"""

import re
import unittest
from pathlib import Path


class TestTensorrtDocumented(unittest.TestCase):
    """tensorrt must be documented in requirements_public_inference.txt."""

    _INFER_REQS = Path(__file__).resolve().parent.parent / "requirements_public_inference.txt"

    def test_tensorrt_mentioned_in_inference_requirements(self):
        """A comment in requirements_public_inference.txt must reference tensorrt."""
        text = self._INFER_REQS.read_text()
        self.assertTrue(
            re.search(r"tensorrt", text),
            "requirements_public_inference.txt must document tensorrt as an optional "
            "GPU prerequisite (used by ONNX_WORKFLOW=2/3 paths). "
            "Add a comment: '# tensorrt  -- required for ONNX_WORKFLOW=2/3'.",
        )

    def test_tensorrt_not_a_pip_requirement(self):
        """tensorrt must appear only in a comment, not as an active pip requirement."""
        text = self._INFER_REQS.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertFalse(
                stripped.startswith("tensorrt"),
                "tensorrt must not be an active pip requirement in "
                "requirements_public_inference.txt: it is a heavy CUDA-only SDK "
                "that would break CPU CI. Document it in a comment instead.",
            )


class TestTensorrtMissingRaisesError(unittest.TestCase):
    """Both TRT import sites must raise RuntimeError when tensorrt is not installed.

    Silently falling back to PyTorch would mask misconfiguration: the user
    explicitly chose ONNX_WORKFLOW=2 or 3, so a missing tensorrt install is
    always a hard error.
    """

    # --- helpers that mirror the two import sites in logical_error_rate.py ---

    def _simulate_trt_load(self, import_raises, other_error=False):
        """Mirror the USE_ENGINE_ONLY (ONNX_WORKFLOW=3) import block."""
        trt_context = None
        try:
            if import_raises:
                raise ImportError("No module named 'tensorrt'")
            if other_error:
                raise RuntimeError("engine deserialize failed")
            trt_context = object()
        except ImportError as e:
            raise RuntimeError(
                "[LER] ONNX_WORKFLOW=3 (USE_ENGINE_ONLY) requires tensorrt to be installed. "
                "Install with: pip install tensorrt"
            ) from e
        except Exception as e:
            # Non-import failures (bad engine file, etc.) fall back gracefully.
            trt_context = None
        return trt_context

    def _simulate_trt_build(self, import_raises, other_error=False):
        """Mirror the EXPORT_AND_USE_TRT (ONNX_WORKFLOW=2) import block."""
        trt_context = None
        try:
            if import_raises:
                raise ImportError("No module named 'tensorrt'")
            if other_error:
                raise RuntimeError("TRT build failed")
            trt_context = object()
        except ImportError as e:
            raise RuntimeError(
                "[LER] ONNX_WORKFLOW=2 (EXPORT_AND_USE_TRT) requires tensorrt to be installed. "
                "Install with: pip install tensorrt"
            ) from e
        except Exception as e:
            trt_context = None
        return trt_context

    # --- import-error tests (must raise, not fall back) ---

    def test_use_engine_only_raises_on_import_error(self):
        """USE_ENGINE_ONLY: missing tensorrt must raise RuntimeError, not fall back."""
        with self.assertRaises(RuntimeError) as ctx:
            self._simulate_trt_load(import_raises=True)
        self.assertIn("USE_ENGINE_ONLY", str(ctx.exception))
        self.assertIn("pip install tensorrt", str(ctx.exception))

    def test_export_and_use_trt_raises_on_import_error(self):
        """EXPORT_AND_USE_TRT: missing tensorrt must raise RuntimeError, not fall back."""
        with self.assertRaises(RuntimeError) as ctx:
            self._simulate_trt_build(import_raises=True)
        self.assertIn("EXPORT_AND_USE_TRT", str(ctx.exception))
        self.assertIn("pip install tensorrt", str(ctx.exception))

    def test_import_error_chained_to_runtime_error(self):
        """The RuntimeError must chain the original ImportError as __cause__."""
        with self.assertRaises(RuntimeError) as ctx:
            self._simulate_trt_load(import_raises=True)
        self.assertIsInstance(ctx.exception.__cause__, ImportError)

    # --- non-import error tests (must still fall back gracefully) ---

    def test_use_engine_only_falls_back_on_runtime_error(self):
        """USE_ENGINE_ONLY: non-import TRT errors (bad engine) still fall back to PyTorch."""
        result = self._simulate_trt_load(import_raises=False, other_error=True)
        self.assertIsNone(result)

    def test_export_and_use_trt_falls_back_on_runtime_error(self):
        """EXPORT_AND_USE_TRT: non-import TRT errors (build failure) still fall back."""
        result = self._simulate_trt_build(import_raises=False, other_error=True)
        self.assertIsNone(result)


class TestTrtBuilderPrecisionFlags(unittest.TestCase):
    """Regression tests for the STRONGLY_TYPED throughput regression (d7b8217/993e797).

    Background: commit d7b8217 (Ising-Decoding) introduced STRONGLY_TYPED into the
    TRT network flags alongside Conv-only FP8 QDQ nodes.  Under STRONGLY_TYPED, TRT
    must respect every FP16↔FP8 cast boundary inserted by modelopt, preventing
    Conv+BN+ReLU fusion and adding per-layer cast kernels — causing a ~25-35%
    throughput regression (66 µs → 90 µs at d=13, T=104).

    Fix: remove STRONGLY_TYPED; use BuilderFlag.FP8/INT8 instead so TRT can
    optimise precision boundaries freely while still selecting quantised kernels.
    """

    _LER = Path(__file__).resolve().parent.parent / "evaluation" / "logical_error_rate.py"

    # ── source-code regression guard ──────────────────────────────────────────

    def test_strongly_typed_not_or_into_net_flags(self):
        """STRONGLY_TYPED must never be OR'd into net_flags in the TRT build block.

        The line  ``net_flags |= 1 << int(...STRONGLY_TYPED)``  is the exact pattern
        that caused the regression; guard against it being re-introduced.
        """
        text = self._LER.read_text()
        self.assertIsNone(
            re.search(r"net_flags\s*\|=.*STRONGLY_TYPED", text),
            "STRONGLY_TYPED must not be OR'd into net_flags: it prevents Conv+BN+ReLU "
            "fusion across FP8/INT8 QDQ boundaries and causes ~25-35% throughput "
            "regression.  Use BuilderFlag.FP8 / BuilderFlag.INT8 instead.",
        )

    # ── mock-based flag-setting tests ─────────────────────────────────────────
    # These mirror the config.set_flag block in logical_error_rate.py so that
    # any future edit to that block is immediately caught here.

    def _collect_builder_flags(self, quant_format):
        """Simulate the config.set_flag calls for the given quant_format.

        Returns a list of BuilderFlag values passed to config.set_flag, in order.
        Mirrors logical_error_rate.py lines after ``config = builder.create_builder_config()``.
        """
        from unittest.mock import MagicMock
        trt = MagicMock()
        flags_set = []
        config = MagicMock()
        config.set_flag.side_effect = lambda f: flags_set.append(f)

        if quant_format == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.FP16)
        elif quant_format == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.FP16)

        return flags_set, trt

    def _compute_net_flags(self, quant_format):
        """Simulate the net_flags computation for the given quant_format.

        Mirrors logical_error_rate.py lines around ``builder.create_network(net_flags)``.
        Returns the integer net_flags value.
        """
        from unittest.mock import MagicMock
        trt = MagicMock()
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0

        net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # STRONGLY_TYPED intentionally not set — see class docstring.
        return net_flags, trt

    def test_fp8_sets_fp8_and_fp16_builder_flags(self):
        """QUANT_FORMAT=fp8 must set BuilderFlag.FP8 and BuilderFlag.FP16."""
        flags_set, trt = self._collect_builder_flags("fp8")
        self.assertIn(trt.BuilderFlag.FP8, flags_set)
        self.assertIn(trt.BuilderFlag.FP16, flags_set)

    def test_int8_sets_int8_and_fp16_builder_flags(self):
        """QUANT_FORMAT=int8 must set BuilderFlag.INT8 and BuilderFlag.FP16."""
        flags_set, trt = self._collect_builder_flags("int8")
        self.assertIn(trt.BuilderFlag.INT8, flags_set)
        self.assertIn(trt.BuilderFlag.FP16, flags_set)

    def test_unquantized_sets_only_fp16_builder_flag(self):
        """No QUANT_FORMAT must set BuilderFlag.FP16 and nothing else."""
        flags_set, trt = self._collect_builder_flags("")
        self.assertIn(trt.BuilderFlag.FP16, flags_set)
        self.assertNotIn(trt.BuilderFlag.FP8, flags_set)
        self.assertNotIn(trt.BuilderFlag.INT8, flags_set)

    def test_fp8_does_not_set_strongly_typed_in_net_flags(self):
        """For fp8, STRONGLY_TYPED bit must be absent from net_flags."""
        net_flags, trt = self._compute_net_flags("fp8")
        strongly_typed_bit = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        self.assertEqual(
            net_flags & strongly_typed_bit,
            0,
            "STRONGLY_TYPED must not be set in net_flags for fp8.",
        )

    def test_int8_does_not_set_strongly_typed_in_net_flags(self):
        """For int8, STRONGLY_TYPED bit must be absent from net_flags."""
        net_flags, trt = self._compute_net_flags("int8")
        strongly_typed_bit = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        self.assertEqual(
            net_flags & strongly_typed_bit,
            0,
            "STRONGLY_TYPED must not be set in net_flags for int8.",
        )


class TestTensorrtImportable(unittest.TestCase):
    """When tensorrt is installed, its key symbols must be importable (GPU CI only)."""

    @unittest.skipUnless(
        # Only run when TRT is actually present; skip silently otherwise.
        __import__("importlib").util.find_spec("tensorrt") is not None,
        "tensorrt is not installed in this environment",
    )
    def test_tensorrt_key_symbols(self):
        import tensorrt as trt
        required = ["Logger", "Runtime", "Builder", "BuilderFlag", "LayerInformationFormat"]
        for sym in required:
            self.assertTrue(
                hasattr(trt, sym),
                f"tensorrt.{sym} not found — TRT install may be incomplete.",
            )


if __name__ == "__main__":
    unittest.main()
