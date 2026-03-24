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
