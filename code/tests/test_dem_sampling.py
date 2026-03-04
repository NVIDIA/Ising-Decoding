# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for qec.dem_sampling (DEM sampling and timelike corrections)."""

import sys
import unittest
from pathlib import Path

import torch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from qec.dem_sampling import dem_sampling, measure_from_stacked_frames, timelike_syndromes


class TestDemSampling(unittest.TestCase):
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
        # Odd-weight frames_xz and A so that (frames_xz @ A.T) has odd entries (mod 2)
        frames_xz = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.uint8)
        A = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.uint8)
        meas_old = torch.zeros(batch_size, n_rounds, num_meas, dtype=torch.uint8)
        meas_new = timelike_syndromes(frames_xz, A, meas_old)
        self.assertEqual(meas_new.shape, meas_old.shape)
        self.assertFalse(torch.equal(meas_new, meas_old))
