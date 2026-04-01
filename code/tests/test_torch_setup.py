# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Quick test to verify the Torch setup: imports, DEM sampling, generator init, batch generation.
# Runs in CI with code/tests (PYTHONPATH=code).

import sys
import unittest
from pathlib import Path

# Ensure repo code/ is on path when run via unittest discover (PYTHONPATH=code)
_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch

import qec.dem_sampling as _dem_mod


class TestTorchSetup(unittest.TestCase):
    """Verify Torch-only setup: imports, DEM sampling, generator init, batch generation."""

    def test_import_dem_sampling(self):
        from qec.dem_sampling import dem_sampling, measure_from_stacked_frames, timelike_syndromes
        self.assertTrue(dem_sampling is not None)

    def test_import_homological_equivalence_torch(self):
        from qec.surface_code.homological_equivalence_torch import (
            apply_weight1_timelike_homological_equivalence_torch,
            build_spacelike_he_cache,
            build_timelike_he_cache,
        )
        self.assertTrue(build_spacelike_he_cache is not None)

    def test_import_generator_torch(self):
        from data.generator_torch import QCDataGeneratorTorch
        self.assertTrue(QCDataGeneratorTorch is not None)

    @unittest.skipUnless(_dem_mod._CUSTAB_AVAILABLE, "cuquantum>=26.3.0 (stabilizer) not available")
    def test_dem_sampling_shape(self):
        from qec.dem_sampling import dem_sampling
        torch.manual_seed(42)
        num_detectors, num_errors = 10, 20
        H = torch.randint(0, 2, (2 * num_detectors, num_errors), dtype=torch.uint8)
        p = torch.rand(num_errors) * 0.01
        frames = dem_sampling(H, p, batch_size=4)
        self.assertEqual(frames.shape, (4, 2 * num_detectors))
        self.assertEqual(frames.dtype, torch.uint8)

    def test_measure_from_stacked_frames_shape(self):
        from qec.dem_sampling import measure_from_stacked_frames
        batch_size, n_rounds, nq = 4, 2, 5
        meas_qubits = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        meas_bases = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        frames_xz = torch.randint(0, 2, (batch_size, 2 * n_rounds * nq), dtype=torch.uint8)
        meas = measure_from_stacked_frames(frames_xz, meas_qubits, meas_bases, nq)
        self.assertEqual(meas.shape, (batch_size, n_rounds, len(meas_qubits)))

    def test_timelike_syndromes_shape(self):
        from qec.dem_sampling import timelike_syndromes
        batch_size, n_rounds, num_meas = 4, 2, 4
        A = torch.randint(0, 2, (n_rounds * num_meas, 2 * n_rounds * 5), dtype=torch.uint8)
        meas_old = torch.randint(0, 2, (batch_size, n_rounds, num_meas), dtype=torch.uint8)
        frames_xz = torch.randint(0, 2, (batch_size, 2 * n_rounds * 5), dtype=torch.uint8)
        meas_new = timelike_syndromes(frames_xz, A, meas_old)
        self.assertEqual(meas_new.shape, meas_old.shape)

    @unittest.skipUnless(_dem_mod._CUSTAB_AVAILABLE, "cuquantum>=26.3.0 (stabilizer) not available")
    def test_generator_init_and_batch(self):
        from data.generator_torch import QCDataGeneratorTorch
        torch.manual_seed(42)
        gen = QCDataGeneratorTorch(
            distance=3,
            n_rounds=3,
            p_error=0.004,
            measure_basis="both",
            rank=0,
            mode="train",
            verbose=False,
            timelike_he=True,
            num_he_cycles=1,
            max_passes_w1=8,
            decompose_y=False,
            precomputed_frames_dir=None,
            code_rotation="XV",
            base_seed=42,
        )
        trainX, trainY = gen.generate_batch(step=0, batch_size=2)
        self.assertEqual(trainX.dim(), 5)
        self.assertEqual(trainY.dim(), 5)
