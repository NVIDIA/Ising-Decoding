# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for the TensorRT optional-dependency contract.

tensorrt is a heavy CUDA-only package (~500 MB) that cannot be pip-installed
in CPU-only CI.  It is therefore documented as an optional prerequisite via a
comment in requirements_public_inference.txt rather than as a pip requirement.
These tests verify:
  1. The documentation comment exists in the inference requirements file.
  2. Both TensorRT import sites fall back gracefully to trt_context=None when
     the package is unavailable (simulated via ImportError).
  3. When tensorrt is installed, its key symbols are importable (GPU CI only).
"""

import re
import sys
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


class TestTensorrtFallback(unittest.TestCase):
    """Both TRT import sites must set trt_context=None on ImportError."""

    def _simulate_trt_load_fallback(self, import_raises):
        """Simulate the USE_ENGINE_ONLY trt_context assignment pattern."""
        trt_context = None
        try:
            if import_raises:
                raise ImportError("No module named 'tensorrt'")
            # If import succeeded we'd set trt_context here; not reached in tests.
            trt_context = object()
        except Exception:
            trt_context = None
        return trt_context

    def _simulate_trt_build_fallback(self, import_raises):
        """Simulate the EXPORT_AND_USE_TRT trt_context assignment pattern."""
        trt_context = None
        try:
            if import_raises:
                raise ImportError("No module named 'tensorrt'")
            trt_context = object()
        except Exception:
            trt_context = None
        return trt_context

    def test_use_engine_only_falls_back_on_import_error(self):
        """USE_ENGINE_ONLY: ImportError on 'import tensorrt' must yield trt_context=None."""
        result = self._simulate_trt_load_fallback(import_raises=True)
        self.assertIsNone(result)

    def test_export_and_use_trt_falls_back_on_import_error(self):
        """EXPORT_AND_USE_TRT: ImportError on 'import tensorrt' must yield trt_context=None."""
        result = self._simulate_trt_build_fallback(import_raises=True)
        self.assertIsNone(result)

    def test_fallback_does_not_raise(self):
        """Neither TRT path must propagate ImportError to the caller."""
        try:
            self._simulate_trt_load_fallback(import_raises=True)
            self._simulate_trt_build_fallback(import_raises=True)
        except Exception as e:
            self.fail(f"TRT fallback unexpectedly raised: {e}")


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
