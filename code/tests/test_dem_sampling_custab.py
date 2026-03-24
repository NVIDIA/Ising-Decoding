# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Test dem_sampling with cuST (BitMatrixSampler) using the same data as the C++ test
DEMSamplingCPU.SingleShot.
"""

import sys
import unittest
from pathlib import Path

# Ensure repo's code/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import qec.dem_sampling as _dem_mod
from qec.dem_sampling import _custab_available


@unittest.skipUnless(_custab_available(), "custabilizer (cuquantum.stabilizer) not present")
class TestDEMSamplingCustabSingleShot(unittest.TestCase):
    """Python equivalent of C++ TEST(DEMSamplingCPU, SingleShot)."""

    def test_single_shot_checks_match_cpp(self) -> None:
        # From DEMSamplingCPU.SingleShot:
        # num_shots = 1, num_error_mechanisms = 4, num_checks = 2
        # h_check_matrix = {1, 1, 0, 0, 0, 0, 1, 1} -> (num_checks, num_error_mechanisms) = (2, 4)
        H_CPP = [
            [1, 1, 0, 0],  # Check 0: e0 XOR e1
            [0, 0, 1, 1],  # Check 1: e2 XOR e3
        ]
        h_probs = [1.0, 0.0, 1.0, 0.0]
        # C++ expected: cpu_checks = [1, 1] (e0 XOR e1 = 1, e2 XOR e3 = 1)
        expected_checks = [1, 1]

        # dem_sampling expects H (2*num_detectors, num_errors) = (num_checks, num_errors) = (2, 4)
        H = torch.tensor(H_CPP, dtype=torch.uint8)
        p = torch.tensor(h_probs, dtype=torch.float32)
        num_shots = 1

        from qec.dem_sampling import dem_sampling

        # Use cuST path (default when available; ensure not disabled)
        frames_xz = dem_sampling(H, p, num_shots)

        self.assertEqual(frames_xz.shape, (num_shots, 2), "samples shape (num_shots, num_checks)")
        cpu_checks = frames_xz[0].tolist()
        self.assertEqual(
            cpu_checks, expected_checks, f"expected cpu_checks {expected_checks}, got {cpu_checks}"
        )

    def test_single_shot_torch_fallback_matches_checks(self) -> None:
        """Same data via torch path should give same syndrome (deterministic p)."""
        H_CPP = [[1, 1, 0, 0], [0, 0, 1, 1]]
        h_probs = [1.0, 0.0, 1.0, 0.0]
        expected_checks = [1, 1]

        H = torch.tensor(H_CPP, dtype=torch.uint8)
        p = torch.tensor(h_probs, dtype=torch.float32)

        import os
        from qec.dem_sampling import dem_sampling, _reset_use_custab_cache, _reset_sampler_cache

        _reset_sampler_cache()
        _reset_use_custab_cache()
        os.environ["USE_CUSTAB"] = "0"
        try:
            frames_xz = dem_sampling(H, p, 1)
        finally:
            os.environ.pop("USE_CUSTAB", None)
            _reset_use_custab_cache()
        cpu_checks = frames_xz[0].tolist()
        self.assertEqual(
            cpu_checks, expected_checks, f"torch path: expected {expected_checks}, got {cpu_checks}"
        )


class TestDEMSamplingTorchFallback(unittest.TestCase):
    """
    Test the torch fallback path (USE_CUSTAB=0) only.
    Not skipped by custab availability — runs in CI without GPU/custab.
    """

    def test_torch_fallback_shape_and_deterministic_checks(self) -> None:
        """Torch path: same small H/p as C++ SingleShot; check shape and syndrome."""
        import os
        from qec.dem_sampling import dem_sampling, _reset_use_custab_cache, _reset_sampler_cache

        _reset_sampler_cache()
        _reset_use_custab_cache()
        os.environ["USE_CUSTAB"] = "0"
        try:
            H_CPP = [[1, 1, 0, 0], [0, 0, 1, 1]]
            h_probs = [1.0, 0.0, 1.0, 0.0]
            expected_checks = [1, 1]
            H = torch.tensor(H_CPP, dtype=torch.uint8)
            p = torch.tensor(h_probs, dtype=torch.float32)
            frames_xz = dem_sampling(H, p, 1)
        finally:
            os.environ.pop("USE_CUSTAB", None)
            _reset_use_custab_cache()

        self.assertEqual(frames_xz.shape, (1, 2), "samples shape (num_shots, num_checks)")
        self.assertEqual(
            frames_xz[0].tolist(), expected_checks,
            f"torch path: expected {expected_checks}, got {frames_xz[0].tolist()}"
        )


@unittest.skipUnless(
    _custab_available() and _dem_mod._CUPY_AVAILABLE and torch.cuda.is_available(),
    "requires custabilizer + CuPy + CUDA GPU",
)
class TestDEMSamplingCupyGPUPath(unittest.TestCase):
    """Tests for the CuPy zero-copy DLPack GPU path in custab_matrix_sampling."""

    def setUp(self) -> None:
        from qec.dem_sampling import _reset_sampler_cache, _reset_use_custab_cache
        _reset_sampler_cache()
        _reset_use_custab_cache()

    def test_cupy_available_flag(self) -> None:
        """_CUPY_AVAILABLE is True when CuPy is importable."""
        self.assertTrue(_dem_mod._CUPY_AVAILABLE)

    def test_gpu_native_path_shape_and_dtype(self) -> None:
        """CuPy path returns (batch_size, num_checks) uint8 tensor on GPU input."""
        from qec.dem_sampling import dem_sampling

        H = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.uint8).cuda()
        p = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32).cuda()

        frames = dem_sampling(H, p, 4)

        self.assertEqual(frames.shape, (4, 2), "expected shape (batch_size, num_checks)")
        self.assertEqual(frames.dtype, torch.uint8)

    def test_gpu_native_path_deterministic_checks(self) -> None:
        """CuPy path: deterministic probabilities produce the expected syndrome."""
        from qec.dem_sampling import dem_sampling

        H = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.uint8).cuda()
        p = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32).cuda()
        expected = [1, 1]

        frames = dem_sampling(H, p, 1)

        self.assertEqual(
            frames[0].tolist(),
            expected,
            f"CuPy path: expected {expected}, got {frames[0].tolist()}",
        )

    def test_gpu_native_matches_torch_fallback(self) -> None:
        """CuPy GPU path and torch CPU fallback produce the same syndrome for deterministic input."""
        import os
        from qec.dem_sampling import dem_sampling, _reset_sampler_cache, _reset_use_custab_cache

        H_data = [[1, 1, 0, 0], [0, 0, 1, 1]]
        p_data = [1.0, 0.0, 1.0, 0.0]

        # GPU / CuPy path
        frames_gpu = dem_sampling(
            torch.tensor(H_data, dtype=torch.uint8).cuda(),
            torch.tensor(p_data, dtype=torch.float32).cuda(),
            1,
        )

        # CPU / torch fallback
        _reset_sampler_cache()
        _reset_use_custab_cache()
        os.environ["USE_CUSTAB"] = "0"
        try:
            frames_cpu = dem_sampling(
                torch.tensor(H_data, dtype=torch.uint8),
                torch.tensor(p_data, dtype=torch.float32),
                1,
            )
        finally:
            os.environ.pop("USE_CUSTAB", None)
            _reset_use_custab_cache()

        self.assertEqual(frames_gpu[0].tolist(), frames_cpu[0].tolist())


if __name__ == "__main__":
    unittest.main()
