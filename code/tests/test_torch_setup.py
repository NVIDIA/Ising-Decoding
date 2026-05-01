# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Quick test to verify the Torch setup: imports, DEM sampling, generator init, batch generation.
# Runs in CI with code/tests (PYTHONPATH=code).

import sys
import json
import tempfile
import unittest
from pathlib import Path

# Ensure repo code/ is on path when run via unittest discover (PYTHONPATH=code)
_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch
import numpy as np

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

    def test_generator_uses_noise_model_for_in_memory_dem_precompute(self):
        from data.generator_torch import QCDataGeneratorTorch
        from qec.noise_model import NoiseModel, CNOT_ERROR_TYPES

        nm = NoiseModel(
            p_prep_X=0.001,
            p_prep_Z=0.002,
            p_meas_X=0.003,
            p_meas_Z=0.004,
            p_idle_cnot_X=0.0011,
            p_idle_cnot_Y=0.0012,
            p_idle_cnot_Z=0.0013,
            p_idle_spam_X=0.0021,
            p_idle_spam_Y=0.0022,
            p_idle_spam_Z=0.0023,
            **{f"p_cnot_{k}": 0.0001 for k in CNOT_ERROR_TYPES}
        )

        gen = QCDataGeneratorTorch(
            distance=3,
            n_rounds=3,
            p_error=0.004,
            measure_basis="both",
            rank=0,
            mode="train",
            verbose=False,
            timelike_he=False,
            decompose_y=False,
            precomputed_frames_dir=None,
            code_rotation="XV",
            noise_model=nm,
            device=torch.device("cpu"),
        )

        self.assertIs(gen.noise_model, nm)
        p_values = torch.cat([gen.sim_X.p.cpu(), gen.sim_Z.p.cpu()])
        for expected in (nm.p_prep_X, nm.p_prep_Z, nm.p_meas_X, nm.p_meas_Z, nm.p_idle_spam_Z):
            self.assertTrue(
                torch.isclose(p_values, torch.tensor(expected, dtype=p_values.dtype)).any(),
                f"Expected 25p probability {expected} in generated DEM p vector",
            )

    def test_disk_dem_refreshes_probabilities_from_active_noise_model(self):
        from data.generator_torch import QCDataGeneratorTorch
        from qec.noise_model import NoiseModel, CNOT_ERROR_TYPES
        from qec.precompute_dem import (
            DEM_ARTIFACT_METADATA_KEY,
            precompute_dem_bundle_surface_code,
        )

        nm = NoiseModel(
            p_prep_X=0.001,
            p_prep_Z=0.002,
            p_meas_X=0.003,
            p_meas_Z=0.004,
            p_idle_cnot_X=0.0011,
            p_idle_cnot_Y=0.0012,
            p_idle_cnot_Z=0.0013,
            p_idle_spam_X=0.0021,
            p_idle_spam_Y=0.0022,
            p_idle_spam_Z=0.0023,
            **{f"p_cnot_{k}": 0.0001 for k in CNOT_ERROR_TYPES}
        )

        with tempfile.TemporaryDirectory() as tmp:
            precompute_dem_bundle_surface_code(
                distance=3,
                n_rounds=3,
                basis="X",
                code_rotation="XV",
                p_scalar=0.004,
                dem_output_dir=tmp,
                device=torch.device("cpu"),
                export=True,
                noise_model=nm,
            )
            p_path = Path(tmp) / "surface_d3_r3_X_frame_predecoder.p.npz"
            with np.load(p_path, allow_pickle=False) as z:
                p_arr = z["p"].copy()
                p_nominal = z["p_nominal"].copy()
                metadata_json = z[DEM_ARTIFACT_METADATA_KEY].copy()
            sentinel = np.float32(0.987654)
            p_arr[0] = sentinel
            np.savez_compressed(
                p_path,
                p=p_arr,
                p_nominal=p_nominal,
                **{DEM_ARTIFACT_METADATA_KEY: metadata_json},
            )

            gen = QCDataGeneratorTorch(
                distance=3,
                n_rounds=3,
                p_error=0.004,
                measure_basis="X",
                rank=0,
                mode="train",
                verbose=False,
                timelike_he=False,
                decompose_y=False,
                precomputed_frames_dir=tmp,
                code_rotation="XV",
                noise_model=nm,
                device=torch.device("cpu"),
            )

        self.assertIs(gen.noise_model, nm)
        self.assertFalse(torch.isclose(gen.sim.p.cpu(), torch.tensor(float(sentinel))).any())
        self.assertTrue(
            torch.isclose(gen.sim.p.cpu(), torch.tensor(nm.p_prep_X)).any(),
            "Expected cached DEM structure to use probabilities refreshed from the active noise_model",
        )

    def test_mismatched_noise_model_metadata_reuses_structure_with_refreshed_p(self):
        from data.generator_torch import QCDataGeneratorTorch
        from qec.noise_model import NoiseModel, CNOT_ERROR_TYPES
        from qec.precompute_dem import (
            DEM_ARTIFACT_METADATA_KEY,
            precompute_dem_bundle_surface_code,
        )
        from unittest import mock

        nm_disk = NoiseModel(
            p_prep_X=0.001,
            p_prep_Z=0.002,
            p_meas_X=0.003,
            p_meas_Z=0.004,
            p_idle_cnot_X=0.0011,
            p_idle_cnot_Y=0.0012,
            p_idle_cnot_Z=0.0013,
            p_idle_spam_X=0.0021,
            p_idle_spam_Y=0.0022,
            p_idle_spam_Z=0.0023,
            **{f"p_cnot_{k}": 0.0001 for k in CNOT_ERROR_TYPES}
        )
        nm_active = nm_disk.copy()
        nm_active.p_prep_X += 1e-6

        with tempfile.TemporaryDirectory() as tmp:
            precompute_dem_bundle_surface_code(
                distance=3,
                n_rounds=3,
                basis="X",
                code_rotation="XV",
                p_scalar=0.004,
                dem_output_dir=tmp,
                device=torch.device("cpu"),
                export=True,
                noise_model=nm_disk,
            )
            p_path = Path(tmp) / "surface_d3_r3_X_frame_predecoder.p.npz"
            with np.load(p_path, allow_pickle=False) as z:
                p_arr = z["p"].copy()
                p_nominal = z["p_nominal"].copy()
                metadata_json = z[DEM_ARTIFACT_METADATA_KEY].copy()
            sentinel = np.float32(0.987654)
            p_arr[0] = sentinel
            np.savez_compressed(
                p_path,
                p=p_arr,
                p_nominal=p_nominal,
                **{DEM_ARTIFACT_METADATA_KEY: metadata_json},
            )

            with mock.patch("qec.precompute_dem.precompute_dem_bundle_surface_code") as rebuild:
                rebuild.side_effect = AssertionError("should reuse cached structural DEM")
                gen = QCDataGeneratorTorch(
                    distance=3,
                    n_rounds=3,
                    p_error=0.004,
                    measure_basis="X",
                    rank=0,
                    mode="train",
                    verbose=False,
                    timelike_he=False,
                    decompose_y=False,
                    precomputed_frames_dir=tmp,
                    code_rotation="XV",
                    noise_model=nm_active,
                    device=torch.device("cpu"),
                )
                rebuild.assert_not_called()

        self.assertIs(gen.noise_model, nm_active)
        self.assertFalse(torch.isclose(gen.sim.p.cpu(), torch.tensor(float(sentinel))).any())
        self.assertTrue(
            torch.isclose(gen.sim.p.cpu(), torch.tensor(nm_active.p_prep_X)).any(),
            "Expected cached DEM structure to use probabilities refreshed from the active noise_model",
        )

    def test_legacy_scalar_metadata_free_frames_still_load(self):
        from data.generator_torch import QCDataGeneratorTorch
        from qec.precompute_dem import (
            build_probability_vector_surface_code,
            precompute_dem_bundle_surface_code,
        )

        with tempfile.TemporaryDirectory() as tmp:
            precompute_dem_bundle_surface_code(
                distance=3,
                n_rounds=3,
                basis="X",
                code_rotation="XV",
                p_scalar=0.004,
                dem_output_dir=tmp,
                device=torch.device("cpu"),
                export=True,
            )
            p_path = Path(tmp) / "surface_d3_r3_X_frame_predecoder.p.npz"
            with np.load(p_path, allow_pickle=False) as z:
                p_arr = z["p"].copy()
                p_nominal = z["p_nominal"].copy()
            sentinel = np.float32(0.876543)
            p_arr[0] = sentinel
            np.savez_compressed(p_path, p=p_arr, p_nominal=p_nominal)

            gen = QCDataGeneratorTorch(
                distance=3,
                n_rounds=3,
                p_error=0.004,
                measure_basis="X",
                rank=0,
                mode="train",
                verbose=False,
                timelike_he=False,
                decompose_y=False,
                precomputed_frames_dir=tmp,
                code_rotation="XV",
                device=torch.device("cpu"),
            )

        self.assertFalse(torch.isclose(gen.sim.p.cpu(), torch.tensor(float(sentinel))).any())
        # The legacy (metadata-free) artifact must be refreshed from the active
        # scalar p_error rather than reused as-is. The freshly rebuilt scalar p
        # vector is the ground truth here; in particular its max is the ancilla
        # syndrome-readout probability 2*spam*(1-spam) (with spam = 2/3 * p),
        # which is ~5.32e-3 for p=0.004 and therefore strictly larger than p.
        expected_p = build_probability_vector_surface_code(
            distance=3,
            n_rounds=3,
            basis="X",
            code_rotation="XV",
            p_scalar=0.004,
            noise_model=None,
        )
        self.assertTrue(
            torch.allclose(
                gen.sim.p.cpu(),
                torch.from_numpy(expected_p).to(gen.sim.p.cpu().dtype),
            )
        )

    def test_precompute_frames_loads_nested_noise_model_config(self):
        from data.precompute_frames import _load_noise_model
        from qec.noise_model import NoiseModel, CNOT_ERROR_TYPES

        nm = NoiseModel(
            p_prep_X=0.001,
            p_prep_Z=0.002,
            p_meas_X=0.003,
            p_meas_Z=0.004,
            p_idle_cnot_X=0.0011,
            p_idle_cnot_Y=0.0012,
            p_idle_cnot_Z=0.0013,
            p_idle_spam_X=0.0021,
            p_idle_spam_Y=0.0022,
            p_idle_spam_Z=0.0023,
            **{f"p_cnot_{k}": 0.0001 for k in CNOT_ERROR_TYPES}
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "noise_model.json"
            config_path.write_text(
                json.dumps({"data": {
                    "noise_model": nm.to_config_dict()
                }}),
                encoding="utf-8",
            )
            loaded = _load_noise_model(str(config_path))

        self.assertEqual(loaded.sha256(), nm.sha256())
