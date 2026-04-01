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
"""
Tests for qec.dem_sampling: unit tests, cuStabilizer BitMatrixSampler
verification, and integration tests against a pure-torch reference.
"""

import sys
import unittest
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

import qec.dem_sampling as _dem_mod
from qec.dem_sampling import (
    dem_sampling,
    measure_from_stacked_frames,
    timelike_syndromes,
    _reset_sampler_cache,
)

# DEM sampling is only performed during training. Some of the test
# infrastructure does inference only (and sometimes on CPU only), so it skips
# installation of the cuquantum package. Skip these tests if cuquantum is not
# available.
if not _dem_mod._CUSTAB_AVAILABLE:
    raise unittest.SkipTest("cuquantum>=26.3.0 (stabilizer) not available")

# ============================================================================
# Helpers
# ============================================================================


def compute_syndrome(H_flat: list[int], errors: list[int], num_checks: int,
                     num_errors: int) -> list[int]:
    """Reference syndrome: checks = H @ errors mod 2 (row-major H)."""
    syndrome = [0] * num_checks
    for col in range(num_errors):
        if errors[col]:
            for row in range(num_checks):
                syndrome[row] ^= H_flat[row * num_errors + col]
    return syndrome


def _make_H_p(
    H_2d: list[list[int]], probs: list[float], device: torch.device = torch.device("cpu")
):
    H = torch.tensor(H_2d, dtype=torch.uint8, device=device)
    p = torch.tensor(probs, dtype=torch.float32, device=device)
    return H, p


def _torch_reference_dem_sampling(
    H: torch.Tensor, p: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """
    Pure-torch DEM sampling -- the exact logic from the old fallback on main.

    H: (2*num_detectors, num_errors) uint8
    p: (num_errors,) float32
    Returns: (batch_size, 2*num_detectors) uint8
    """
    num_errors = int(H.shape[1])
    device = H.device
    rand_vals = torch.rand(batch_size, num_errors, device=device, dtype=torch.float32)
    errors = (rand_vals < p[None, :]).to(torch.uint8)
    frames_xz = torch.matmul(errors.to(torch.float32), H.T.to(torch.float32))
    frames_xz = frames_xz.to(torch.uint8) % 2
    return frames_xz


# ============================================================================
# Unit tests: shape, dtype, binary output
# ============================================================================


class TestDemSampling(unittest.TestCase):

    def setUp(self) -> None:
        _reset_sampler_cache()

    def test_dem_sampling_shape_and_dtype(self):
        num_detectors = 4
        num_errors = 6
        batch_size = 10
        H = torch.randint(0, 2, (2 * num_detectors, num_errors), dtype=torch.uint8)
        p = torch.rand(num_errors)
        out = dem_sampling(H, p, batch_size)
        self.assertEqual(out.shape, (batch_size, 2 * num_detectors))
        self.assertEqual(out.dtype, torch.uint8)
        self.assertTrue((out <= 1).all())

    def test_dem_sampling_output_binary(self):
        H = torch.randint(0, 2, (4, 4), dtype=torch.uint8)
        p = torch.tensor([0.5] * 4, dtype=torch.float32)
        out = dem_sampling(H, p, 20)
        self.assertTrue((out <= 1).all(), "output should be binary")
        self.assertEqual(out.dtype, torch.uint8)


class TestMeasureFromStackedFrames(unittest.TestCase):

    def test_measure_from_stacked_frames_shape(self):
        batch_size = 4
        nq = 3
        n_rounds = 2
        D = n_rounds * nq
        frames_xz = torch.randint(0, 2, (batch_size, 2 * D), dtype=torch.uint8)
        meas_qubits = torch.tensor([0, 1], dtype=torch.long)
        meas_bases = torch.tensor([0, 1], dtype=torch.long)
        out = measure_from_stacked_frames(frames_xz, meas_qubits, meas_bases, nq)
        self.assertEqual(out.shape, (batch_size, n_rounds, 2))
        self.assertEqual(out.dtype, torch.uint8)


class TestTimelikeSyndromes(unittest.TestCase):

    def test_timelike_syndromes_xor_effect(self):
        batch_size = 2
        n_rounds = 2
        num_meas = 2
        num_detectors = 4
        frames_xz = torch.randint(0, 2, (batch_size, 2 * num_detectors), dtype=torch.uint8)
        A = torch.zeros(n_rounds * num_meas, 2 * num_detectors, dtype=torch.uint8)
        meas_old = torch.randint(0, 2, (batch_size, n_rounds, num_meas), dtype=torch.uint8)
        meas_new = timelike_syndromes(frames_xz, A, meas_old)
        self.assertEqual(meas_new.shape, meas_old.shape)
        self.assertTrue(torch.equal(meas_new, meas_old))

    def test_timelike_syndromes_nonzero_A_changes_output(self):
        batch_size = 2
        n_rounds = 1
        num_meas = 2
        num_detectors = 2
        frames_xz = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.uint8)
        A = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.uint8)
        meas_old = torch.zeros(batch_size, n_rounds, num_meas, dtype=torch.uint8)
        meas_new = timelike_syndromes(frames_xz, A, meas_old)
        self.assertEqual(meas_new.shape, meas_old.shape)
        self.assertFalse(torch.equal(meas_new, meas_old))


# ============================================================================
# Wiring: verify BitMatrixSampler is the active backend
# ============================================================================


class TestBitMatrixSamplerIsUsed(unittest.TestCase):
    """Verify that dem_sampling actually goes through BitMatrixSampler."""

    def setUp(self) -> None:
        _reset_sampler_cache()

    def test_cuquantum_imports_successfully(self) -> None:
        from cuquantum.stabilizer.dem_sampling import BitMatrixSampler
        from cuquantum.stabilizer.simulator import Options
        self.assertIsNotNone(BitMatrixSampler)
        self.assertIsNotNone(Options)

    def test_sampler_cache_populated_after_call(self) -> None:
        from cuquantum.stabilizer.dem_sampling import BitMatrixSampler

        H, p = _make_H_p([[1, 1, 0, 0], [0, 0, 1, 1]], [1.0, 0.0, 1.0, 0.0])
        dem_sampling(H, p, 1)
        self.assertIsNotNone(_dem_mod._cached_sampler)
        self.assertIsInstance(_dem_mod._cached_sampler, BitMatrixSampler)


# ============================================================================
# CPU tests -- mirrors C++ DemSamplingCPU
# ============================================================================


class TestDemSamplingCPU(unittest.TestCase):

    def setUp(self) -> None:
        _reset_sampler_cache()

    def test_all_zero_probabilities(self) -> None:
        H, p = _make_H_p(
            [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            [0.0, 0.0, 0.0, 0.0],
        )
        frames = dem_sampling(H, p, 10)
        self.assertEqual(frames.shape, (10, 3))
        self.assertTrue((frames == 0).all(), "p=0 everywhere => zero syndrome")

    def test_all_one_probabilities(self) -> None:
        H, p = _make_H_p(
            [[1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0]],
            [1.0, 1.0, 1.0, 1.0],
        )
        frames = dem_sampling(H, p, 5)
        for shot in range(5):
            self.assertEqual(frames[shot].tolist(), [0, 1, 0])

    def test_mixed_deterministic_probs(self) -> None:
        H, p = _make_H_p([[1, 0, 1], [0, 1, 1]], [1.0, 0.0, 1.0])
        frames = dem_sampling(H, p, 8)
        for shot in range(8):
            self.assertEqual(frames[shot].tolist(), [0, 1])

    def test_identity_matrix(self) -> None:
        I5 = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        H, p = _make_H_p(I5, [1.0] * 5)
        frames = dem_sampling(H, p, 3)
        for shot in range(3):
            self.assertEqual(frames[shot].tolist(), [1, 1, 1, 1, 1])

    def test_all_ones_even_columns(self) -> None:
        H, p = _make_H_p([[1] * 4] * 3, [1.0] * 4)
        frames = dem_sampling(H, p, 4)
        for shot in range(4):
            self.assertEqual(frames[shot].tolist(), [0, 0, 0], "Even column count => syndrome 0")

    def test_all_ones_odd_columns(self) -> None:
        H, p = _make_H_p([[1] * 3] * 3, [1.0] * 3)
        frames = dem_sampling(H, p, 4)
        for shot in range(4):
            self.assertEqual(frames[shot].tolist(), [1, 1, 1], "Odd column count => syndrome 1")

    def test_single_column_matrix(self) -> None:
        H, p = _make_H_p([[1], [0], [1]], [1.0])
        frames = dem_sampling(H, p, 6)
        for shot in range(6):
            self.assertEqual(frames[shot].tolist(), [1, 0, 1])

    def test_single_row_matrix(self) -> None:
        H, p = _make_H_p([[1, 1, 0, 1, 0]], [1.0, 0.0, 1.0, 0.0, 1.0])
        frames = dem_sampling(H, p, 4)
        for shot in range(4):
            self.assertEqual(frames[shot].tolist(), [1])

    def test_single_shot(self) -> None:
        H, p = _make_H_p([[1, 1, 0], [0, 1, 1]], [1.0, 1.0, 0.0])
        frames = dem_sampling(H, p, 1)
        self.assertEqual(frames.shape, (1, 2))
        self.assertEqual(frames[0].tolist(), [0, 1])

    def test_repetition_code_parity(self) -> None:
        H, p = _make_H_p(
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            [0.0, 1.0, 0.0, 0.0],
        )
        frames = dem_sampling(H, p, 3)
        for shot in range(3):
            self.assertEqual(frames[shot].tolist(), [1, 1, 0])

    def test_syndrome_shape_and_binary(self) -> None:
        """Stochastic probs: verify output shape and binary values."""
        H_2d = [
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
        ]
        num_checks = 4
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        num_shots = 200

        H, p = _make_H_p(H_2d, probs)
        frames = dem_sampling(H, p, num_shots)
        self.assertEqual(frames.shape, (num_shots, num_checks))
        self.assertTrue(((frames == 0) | (frames == 1)).all(), "All values must be 0 or 1")

    def test_syndrome_consistency_deterministic_crosscheck(self) -> None:
        """Deterministic version: p in {0,1} so we can verify H @ e mod 2."""
        H_2d = [
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
        ]
        num_checks, num_errors = 4, 8
        H_flat = [v for row in H_2d for v in row]
        probs = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        errors = [1, 0, 1, 0, 1, 0, 1, 0]

        expected = compute_syndrome(H_flat, errors, num_checks, num_errors)

        H, p = _make_H_p(H_2d, probs)
        frames = dem_sampling(H, p, 10)
        for shot in range(10):
            self.assertEqual(frames[shot].tolist(), expected, f"Shot {shot}: expected {expected}")


# ============================================================================
# GPU tests -- mirrors C++ DemSamplingGPU
# ============================================================================


@unittest.skipUnless(
    _dem_mod._CUPY_AVAILABLE and torch.cuda.is_available(),
    "requires CuPy + CUDA GPU",
)
class TestDemSamplingGPU(unittest.TestCase):

    def setUp(self) -> None:
        _reset_sampler_cache()

    def _gpu(self, H_2d, probs):
        return _make_H_p(H_2d, probs, torch.device("cuda"))

    def test_all_zero_probabilities(self) -> None:
        H, p = self._gpu(
            [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
            [0.0, 0.0, 0.0, 0.0],
        )
        frames = dem_sampling(H, p, 10)
        self.assertEqual(frames.shape, (10, 3))
        self.assertTrue((frames == 0).all())

    def test_all_one_probabilities(self) -> None:
        H, p = self._gpu(
            [[1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0]],
            [1.0, 1.0, 1.0, 1.0],
        )
        frames = dem_sampling(H, p, 5)
        for shot in range(5):
            self.assertEqual(frames[shot].tolist(), [0, 1, 0])

    def test_mixed_deterministic_probs(self) -> None:
        H, p = self._gpu([[1, 0, 1], [0, 1, 1]], [1.0, 0.0, 1.0])
        frames = dem_sampling(H, p, 8)
        for shot in range(8):
            self.assertEqual(frames[shot].tolist(), [0, 1])

    def test_identity_matrix(self) -> None:
        I5 = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
        H, p = self._gpu(I5, [1.0] * 5)
        frames = dem_sampling(H, p, 3)
        for shot in range(3):
            self.assertEqual(frames[shot].tolist(), [1, 1, 1, 1, 1])

    def test_all_ones_even_columns(self) -> None:
        H, p = self._gpu([[1] * 4] * 3, [1.0] * 4)
        frames = dem_sampling(H, p, 4)
        for shot in range(4):
            self.assertEqual(frames[shot].tolist(), [0, 0, 0], "Even column count => syndrome 0")

    def test_all_ones_odd_columns(self) -> None:
        H, p = self._gpu([[1] * 3] * 3, [1.0] * 3)
        frames = dem_sampling(H, p, 4)
        for shot in range(4):
            self.assertEqual(frames[shot].tolist(), [1, 1, 1], "Odd column count => syndrome 1")

    def test_repetition_code_parity(self) -> None:
        H, p = self._gpu(
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
            [0.0, 1.0, 0.0, 0.0],
        )
        frames = dem_sampling(H, p, 3)
        for shot in range(3):
            self.assertEqual(frames[shot].tolist(), [1, 1, 0])

    def test_syndrome_shape_and_binary(self) -> None:
        H_2d = [
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
        ]
        H, p = self._gpu(H_2d, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        frames = dem_sampling(H, p, 100)
        self.assertEqual(frames.shape, (100, 4))
        self.assertTrue(((frames == 0) | (frames == 1)).all(), "All values must be 0 or 1")

    def test_binary_output_only(self) -> None:
        H, p = self._gpu(
            [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
            [0.2, 0.4, 0.6, 0.8, 0.5],
        )
        frames = dem_sampling(H, p, 200)
        self.assertTrue(((frames == 0) | (frames == 1)).all(), "Syndrome values must be 0 or 1")

    def test_cpu_gpu_cross_validation(self) -> None:
        H_2d = [[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0], [1, 1, 0, 0, 1, 1]]
        probs = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

        _reset_sampler_cache()
        H_cpu, p_cpu = _make_H_p(H_2d, probs, torch.device("cpu"))
        frames_cpu = dem_sampling(H_cpu, p_cpu, 20)

        _reset_sampler_cache()
        H_gpu, p_gpu = _make_H_p(H_2d, probs, torch.device("cuda"))
        frames_gpu = dem_sampling(H_gpu, p_gpu, 20)

        for shot in range(20):
            self.assertEqual(
                frames_gpu[shot].tolist(),
                frames_cpu[shot].tolist(),
                f"Shot {shot}: GPU != CPU",
            )

    def test_bitpack_boundary_32_columns(self) -> None:
        num_checks, num_errors = 4, 32
        H_2d = [
            [1 if (i + j) % 3 == 0 else 0 for j in range(num_errors)] for i in range(num_checks)
        ]
        H_flat = [v for row in H_2d for v in row]
        probs = [1.0] * num_errors
        errors = [1] * num_errors

        expected = compute_syndrome(H_flat, errors, num_checks, num_errors)

        H, p = self._gpu(H_2d, probs)
        frames = dem_sampling(H, p, 50)
        for shot in range(50):
            self.assertEqual(frames[shot].tolist(), expected, f"Shot {shot} (32-col boundary)")

    def test_bitpack_boundary_33_columns(self) -> None:
        num_checks, num_errors = 3, 33
        H_2d = [
            [1 if (i + j) % 2 == 0 else 0 for j in range(num_errors)] for i in range(num_checks)
        ]
        H_flat = [v for row in H_2d for v in row]
        probs = [1.0] * num_errors
        errors = [1] * num_errors

        expected = compute_syndrome(H_flat, errors, num_checks, num_errors)

        H, p = self._gpu(H_2d, probs)
        frames = dem_sampling(H, p, 50)
        for shot in range(50):
            self.assertEqual(frames[shot].tolist(), expected, f"Shot {shot} (33-col boundary)")

    def test_bitpack_boundary_64_columns(self) -> None:
        num_checks, num_errors = 4, 64
        H_2d = [
            [1 if (i * 7 + j * 3) % 5 == 0 else 0
             for j in range(num_errors)]
            for i in range(num_checks)
        ]
        H_flat = [v for row in H_2d for v in row]
        probs = [1.0] * num_errors
        errors = [1] * num_errors

        expected = compute_syndrome(H_flat, errors, num_checks, num_errors)

        H, p = self._gpu(H_2d, probs)
        frames = dem_sampling(H, p, 40)
        for shot in range(40):
            self.assertEqual(frames[shot].tolist(), expected, f"Shot {shot} (64-col boundary)")

    def test_large_scale_deterministic(self) -> None:
        """20 checks x 100 errors, p in {0,1}, verify H @ e mod 2."""
        num_checks, num_errors, num_shots = 20, 100, 50

        rng = random.Random(12345)
        H_2d = [
            [1 if rng.random() < 0.3 else 0 for _ in range(num_errors)] for _ in range(num_checks)
        ]
        H_flat = [v for row in H_2d for v in row]

        probs = [float(rng.random() < 0.5) for _ in range(num_errors)]
        errors = [int(p) for p in probs]
        expected = compute_syndrome(H_flat, errors, num_checks, num_errors)

        H, p = self._gpu(H_2d, probs)
        frames = dem_sampling(H, p, num_shots)
        for shot in range(num_shots):
            self.assertEqual(frames[shot].tolist(), expected, f"Shot {shot} (large scale)")


# ============================================================================
# Integration: end-to-end pipeline via MemoryCircuitTorch
# ============================================================================


class TestDEMSamplingIntegration(unittest.TestCase):
    """Full pipeline: precompute_dem -> MemoryCircuitTorch -> generate_batch using dem_sampling."""

    def test_memory_circuit_torch_generate_batch_uses_dem_sampling(self) -> None:
        """generate_batch produces correct shapes via BitMatrixSampler."""
        from qec.precompute_dem import precompute_dem_bundle_surface_code
        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distance, n_rounds, batch_size = 3, 3, 4
        artifacts = precompute_dem_bundle_surface_code(
            distance=distance,
            n_rounds=n_rounds,
            basis="X",
            code_rotation="XV",
            p_scalar=0.01,
            dem_output_dir=None,
            device=device,
            export=False,
            return_artifacts=True,
        )
        mc = MemoryCircuitTorch(
            distance=distance,
            n_rounds=n_rounds,
            basis="X",
            H=artifacts["H"],
            p=artifacts["p"],
            A=artifacts["A"],
            device=device,
        )
        trainX, trainY = mc.generate_batch(batch_size=batch_size)

        self.assertEqual(trainX.shape, (batch_size, 4, n_rounds, distance, distance))
        self.assertEqual(trainY.shape, (batch_size, 4, n_rounds, distance, distance))
        self.assertEqual(trainX.dtype, torch.float32)
        self.assertEqual(trainY.dtype, torch.float32)


# ============================================================================
# cuST vs torch reference: deterministic
# ============================================================================


class TestCustVsTorchDeterministic(unittest.TestCase):
    """Deterministic: p in {0,1} so both paths must produce identical syndromes."""

    def test_identity_matrix(self) -> None:
        H = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.uint8)
        p = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        expected = [1, 0, 1]

        _reset_sampler_cache()
        custab_out = dem_sampling(H, p, 1)
        torch_out = _torch_reference_dem_sampling(H, p, 1)

        self.assertEqual(custab_out[0].tolist(), expected)
        self.assertEqual(torch_out[0].tolist(), expected)

    def test_repetition_code(self) -> None:
        H = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.uint8)
        p = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        expected = [1, 1, 0]

        _reset_sampler_cache()
        custab_out = dem_sampling(H, p, 1)
        torch_out = _torch_reference_dem_sampling(H, p, 1)

        self.assertEqual(custab_out[0].tolist(), expected)
        self.assertEqual(torch_out[0].tolist(), expected)

    def test_all_fire_even_parity(self) -> None:
        H = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.uint8)
        p = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        expected = [0, 1, 0]

        _reset_sampler_cache()
        custab_out = dem_sampling(H, p, 1)
        torch_out = _torch_reference_dem_sampling(H, p, 1)

        self.assertEqual(custab_out[0].tolist(), expected)
        self.assertEqual(torch_out[0].tolist(), expected)


# ============================================================================
# cuST vs torch reference: statistical
# ============================================================================


class TestCustVsTorchStatistical(unittest.TestCase):
    """
    Statistical comparison: both paths sample from the same distribution.
    With enough shots the per-detector firing rates must agree.
    """

    NUM_SHOTS = 50_000
    ATOL = 0.02

    def _compare_firing_rates(self, H, p, device, label=""):
        _reset_sampler_cache()
        custab_frames = dem_sampling(H.to(device), p.to(device), self.NUM_SHOTS).float()
        torch_frames = _torch_reference_dem_sampling(H.to(device), p.to(device),
                                                     self.NUM_SHOTS).float()

        custab_rates = custab_frames.mean(dim=0).cpu().numpy()
        torch_rates = torch_frames.mean(dim=0).cpu().numpy()

        max_diff = float(np.max(np.abs(custab_rates - torch_rates)))
        self.assertLess(
            max_diff,
            self.ATOL,
            f"{label} max firing-rate diff {max_diff:.4f} >= {self.ATOL} "
            f"(custab_rates={custab_rates}, torch_rates={torch_rates})",
        )

    def test_small_matrix_cpu(self) -> None:
        H = torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
            dtype=torch.uint8,
        )
        p = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
        self._compare_firing_rates(H, p, torch.device("cpu"), "small_cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA GPU")
    def test_small_matrix_gpu(self) -> None:
        H = torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
            dtype=torch.uint8,
        )
        p = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
        self._compare_firing_rates(H, p, torch.device("cuda"), "small_gpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA GPU")
    def test_realistic_dem_from_stim(self) -> None:
        """Use a real DEM from Stim and compare cuST vs torch firing rates."""
        from qec.precompute_dem import precompute_dem_bundle_surface_code

        device = torch.device("cuda")
        artifacts = precompute_dem_bundle_surface_code(
            distance=3,
            n_rounds=3,
            basis="X",
            code_rotation="XV",
            p_scalar=0.003,
            dem_output_dir=None,
            device=device,
            export=False,
            return_artifacts=True,
        )
        H = artifacts["H"]
        p = artifacts["p"]
        self._compare_firing_rates(H, p, device, "stim_d3_r3")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA GPU")
    def test_realistic_dem_d5(self) -> None:
        """Distance-5 DEM: larger matrix, more realistic."""
        from qec.precompute_dem import precompute_dem_bundle_surface_code

        device = torch.device("cuda")
        artifacts = precompute_dem_bundle_surface_code(
            distance=5,
            n_rounds=5,
            basis="Z",
            code_rotation="XV",
            p_scalar=0.003,
            dem_output_dir=None,
            device=device,
            export=False,
            return_artifacts=True,
        )
        H = artifacts["H"]
        p = artifacts["p"]
        self._compare_firing_rates(H, p, device, "stim_d5_r5")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA GPU")
    def test_generate_batch_output_distribution(self) -> None:
        """
        End-to-end: generate_batch with cuST must produce trainX with
        non-trivial syndrome density similar to what the torch path would give.
        """
        from qec.precompute_dem import precompute_dem_bundle_surface_code
        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch

        device = torch.device("cuda")
        distance, n_rounds = 3, 3
        artifacts = precompute_dem_bundle_surface_code(
            distance=distance,
            n_rounds=n_rounds,
            basis="X",
            code_rotation="XV",
            p_scalar=0.003,
            dem_output_dir=None,
            device=device,
            export=False,
            return_artifacts=True,
        )
        mc = MemoryCircuitTorch(
            distance=distance,
            n_rounds=n_rounds,
            basis="X",
            H=artifacts["H"],
            p=artifacts["p"],
            A=artifacts["A"],
            device=device,
        )
        batch_size = 2048
        trainX, trainY = mc.generate_batch(batch_size=batch_size)

        x_channel = trainX[:, 0]
        z_channel = trainX[:, 1]
        x_density = x_channel.abs().mean().item()
        z_density = z_channel.abs().mean().item()
        total_density = (x_density + z_density) / 2

        self.assertGreater(
            total_density,
            0.001,
            f"Syndrome density {total_density:.6f} is suspiciously low -- "
            "cuST may not be generating errors correctly",
        )

        y_density = trainY.abs().mean().item()
        self.assertGreater(
            y_density,
            0.0,
            "trainY is all zeros -- target data may be wrong",
        )


if __name__ == "__main__":
    unittest.main()
