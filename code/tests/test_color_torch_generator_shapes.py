# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shape / dtype / device smoke tests for `ColorMemoryCircuitTorch` and
`ColorQCDataGeneratorTorch`.

cuStabilizer is mocked so this can run on CPU and on environments without
`cuquantum` installed. Parity vs the legacy generator is not checked here — that's a separate
work item; the focus is verifying the (trainX, trainY) shape contract.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from qec.color_code.color_code import ColorCode  # noqa: E402
from qec.precompute_dem import (  # noqa: E402
    ColorAugmentedDemBundle,
    build_color_augmented_dem_metadata,
    encode_dem_artifact_metadata,
    get_color_augmented_dem_paths,
    DEM_ARTIFACT_METADATA_KEY,
)
from qec.color_code.homological_equivalence_torch import (  # noqa: E402
    apply_homological_equivalence_color_torch,
)


def _write_synthetic_bundle(
    tmp_dir: Path, *, distance: int, n_rounds: int, basis: str, schedule: str
) -> ColorAugmentedDemBundle:
    cc = ColorCode(distance)
    num_data = int(cc.num_data)
    num_z = int(cc.num_plaquettes)
    num_x = int(cc.num_plaquettes)
    num_meas = num_z + num_x
    num_local_errors = 16
    frame_rows = n_rounds * num_data * 2
    meas_old_rows = n_rounds * num_meas
    meas_new_rows = n_rounds * num_meas
    num_rows = frame_rows + meas_old_rows + meas_new_rows
    num_cols = 1 + n_rounds * (num_local_errors - 1)

    rng = np.random.default_rng(0)
    H = rng.integers(0, 2, size=(num_rows, num_cols), dtype=np.uint8)
    p = np.full((num_cols,), 0.001, dtype=np.float32)

    metadata = build_color_augmented_dem_metadata(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=0.001,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
    )
    metadata_json = np.array(encode_dem_artifact_metadata(metadata))
    paths = get_color_augmented_dem_paths(
        tmp_dir,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    np.savez_compressed(
        paths["H"],
        H=H,
        n_rounds=np.array(n_rounds, dtype=np.int64),
        num_local_errors=np.array(num_local_errors, dtype=np.int64),
        num_data=np.array(num_data, dtype=np.int64),
        num_meas=np.array(num_meas, dtype=np.int64),
        num_z=np.array(num_z, dtype=np.int64),
        num_x=np.array(num_x, dtype=np.int64),
        frame_rows=np.array(frame_rows, dtype=np.int64),
        meas_old_rows=np.array(meas_old_rows, dtype=np.int64),
        meas_new_rows=np.array(meas_new_rows, dtype=np.int64),
        use_decomposed_errors=np.array(False, dtype=np.bool_),
        **{DEM_ARTIFACT_METADATA_KEY: metadata_json},
    )
    np.savez_compressed(
        paths["p"],
        p=p,
        p_nominal=np.array(0.001, dtype=np.float32),
        **{DEM_ARTIFACT_METADATA_KEY: metadata_json},
    )
    return ColorAugmentedDemBundle(
        H=torch.from_numpy(H),
        n_rounds=n_rounds,
        num_local_errors=num_local_errors,
        num_data=num_data,
        num_meas=num_meas,
        num_z=num_z,
        num_x=num_x,
        frame_rows=frame_rows,
        meas_old_rows=meas_old_rows,
        meas_new_rows=meas_new_rows,
        use_decomposed_errors=False,
    )


def _fake_dem_sampling(H, p, batch_size, device_id=None, seed=None):
    rng = np.random.default_rng(seed if seed is not None else 0)
    out = rng.integers(0, 2, size=(int(batch_size), int(H.shape[0])), dtype=np.uint8)
    return torch.as_tensor(out, device=H.device, dtype=torch.uint8)


def _fixed_dem_sampling(outcomes: torch.Tensor):

    def _fake(H, p, batch_size, device_id=None, seed=None):
        del p, device_id, seed
        assert int(batch_size) == int(outcomes.shape[0])
        assert int(H.shape[0]) == int(outcomes.shape[1])
        return outcomes.to(device=H.device, dtype=torch.uint8)

    return _fake


class TestColorMemoryCircuitTorchShapes(unittest.TestCase):
    distance = 3
    n_rounds = 2
    schedule = "nearest-neighbor"

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmp_ctx.name)
        for basis in ("X", "Z"):
            _write_synthetic_bundle(
                self._tmp,
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=basis,
                schedule=self.schedule,
            )

    def tearDown(self):
        self._tmp_ctx.cleanup()

    def test_memory_circuit_shapes(self):
        with mock.patch(
            "qec.color_code.memory_circuit_torch.dem_sampling", side_effect=_fake_dem_sampling
        ):
            from qec.color_code.memory_circuit_torch import ColorMemoryCircuitTorch

            mc = ColorMemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis="X",
                schedule=self.schedule,
                precomputed_frames_dir=str(self._tmp),
                device=torch.device("cpu"),
            )
            B = 8
            trainX, trainY = mc.generate_batch(B)
            cc = ColorCode(self.distance)
            expected = (B, 4, self.n_rounds, cc.n_rows, cc.n_cols)
            self.assertEqual(tuple(trainX.shape), expected)
            self.assertEqual(tuple(trainY.shape), expected)
            self.assertEqual(trainX.dtype, torch.float32)
            self.assertEqual(trainY.dtype, torch.float32)

    def test_generator_basis_alternation(self):
        with mock.patch(
            "qec.color_code.memory_circuit_torch.dem_sampling", side_effect=_fake_dem_sampling
        ):
            from data.generator_torch_color import ColorQCDataGeneratorTorch

            gen = ColorQCDataGeneratorTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                schedule=self.schedule,
                measure_basis="both",
                precomputed_frames_dir=str(self._tmp),
                device=torch.device("cpu"),
                rank=0,
                global_rank=0,
                base_seed=42,
            )
            B = 4
            tX0, tY0 = gen.generate_batch(step=0, batch_size=B)
            tX1, tY1 = gen.generate_batch(step=1, batch_size=B)
            cc = ColorCode(self.distance)
            expected = (B, 4, self.n_rounds, cc.n_rows, cc.n_cols)
            for t in (tX0, tY0, tX1, tY1):
                self.assertEqual(tuple(t.shape), expected)

    def test_memory_circuit_applies_spacelike_he_to_error_labels_only(self):
        from qec.color_code.memory_circuit_torch import ColorMemoryCircuitTorch

        distance = 5
        n_rounds = 5
        basis = "X"
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            bundle = _write_synthetic_bundle(
                tmp,
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                schedule=self.schedule,
            )
            B = 4
            rng = np.random.default_rng(12345)
            frame = rng.integers(0, 2, size=(B, n_rounds, bundle.num_data, 2), dtype=np.uint8)
            meas_old = rng.integers(0, 2, size=(B, n_rounds, bundle.num_meas), dtype=np.uint8)
            meas_new = rng.integers(0, 2, size=(B, n_rounds, bundle.num_meas), dtype=np.uint8)
            outcomes_np = np.zeros((B, bundle.H.shape[0]), dtype=np.uint8)
            f_end = bundle.frame_rows
            mo_end = f_end + bundle.meas_old_rows
            outcomes_np[:, :f_end] = frame.reshape(B, -1)
            outcomes_np[:, f_end:mo_end] = meas_old.reshape(B, -1)
            outcomes_np[:, mo_end:] = meas_new.reshape(B, -1)
            outcomes = torch.from_numpy(outcomes_np)

            common = dict(
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                schedule=self.schedule,
                precomputed_frames_dir=str(tmp),
                device=torch.device("cpu"),
            )
            mc_raw = ColorMemoryCircuitTorch(**common, apply_spacelike_he=False)
            mc_he = ColorMemoryCircuitTorch(**common, apply_spacelike_he=True)

            with mock.patch(
                "qec.color_code.memory_circuit_torch.dem_sampling",
                side_effect=_fixed_dem_sampling(outcomes),
            ):
                raw_x, raw_y = mc_raw.generate_batch(B)
                he_x, he_y = mc_he.generate_batch(B)

            x_cum = torch.as_tensor(frame[..., 0], dtype=torch.uint8)
            z_cum = torch.as_tensor(frame[..., 1], dtype=torch.uint8)
            x_pad = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
            z_pad = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
            x_diff = x_pad[:, 1:, :] ^ x_pad[:, :-1, :]
            z_diff = z_pad[:, 1:, :] ^ z_pad[:, :-1, :]
            expected_z, expected_x = apply_homological_equivalence_color_torch(
                z_diff,
                x_diff,
                mc_he._he_cache,
                max_iterations=mc_he.he_max_iterations,
                use_coset_search=mc_he.use_coset_search,
            )

            torch.testing.assert_close(he_x, raw_x)
            torch.testing.assert_close(he_y[:, 2:], raw_y[:, 2:])
            torch.testing.assert_close(he_y[:, 0], mc_he._scatter_data(expected_z))
            torch.testing.assert_close(he_y[:, 1], mc_he._scatter_data(expected_x))
            self.assertFalse(torch.equal(he_y[:, :2], raw_y[:, :2]))


class TestColorFeedforwardConnectivityTorch(unittest.TestCase):
    """Verifies the Torch-side Z-ancilla→data-qubit feedforward connectivity matrix.

    Mirrors the role of the removed legacy feedforward connectivity test.
    """

    def test_connectivity_matrix_matches_direct_cx_neighbors_nn_d3(self):
        from qec.precompute_dem import (
            _build_color_memory_circuit,
            _extract_color_round_layout,
            build_circuit_z_ancilla_connectivity_matrix,
        )

        distance = 3
        n_rounds = 2
        basis = "X"
        schedule = "nearest-neighbor"
        circ = _build_color_memory_circuit(
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            schedule=schedule,
            p_scalar=0.005,
        )
        layout = _extract_color_round_layout(
            circ=circ,
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            schedule=schedule,
        )
        controls = np.asarray(layout["cnot_circuit"][:, :, 0], dtype=np.int32)
        targets = np.asarray(layout["cnot_circuit"][:, :, 1], dtype=np.int32)
        zcheck = np.asarray(layout["zcheck_qubits"], dtype=np.int32).reshape(-1)
        data = np.asarray(layout["data_qubits"], dtype=np.int32).reshape(-1)
        nq = int(layout["nq"])

        mat = build_circuit_z_ancilla_connectivity_matrix(
            controls=controls,
            targets=targets,
            data_qubits=data,
            zcheck_qubits=zcheck,
            nq=nq,
        )

        self.assertEqual(mat.shape, (zcheck.size, nq))
        self.assertEqual(mat.dtype, np.uint8)
        self.assertTrue(np.all((mat == 0) | (mat == 1)))

        # Rebuild the expected mask directly from the per-layer CX pairs:
        # mat[i, q] == 1 iff there exists a CX layer where one endpoint is
        # zcheck[i] and the other endpoint is data qubit q.
        expected = np.zeros_like(mat)
        z_to_row = {int(z): i for i, z in enumerate(zcheck.tolist())}
        data_set = set(int(q) for q in data.tolist())
        c_flat = controls.reshape(-1)
        t_flat = targets.reshape(-1)
        valid = (c_flat >= 0) & (t_flat >= 0)
        for cq, tq in zip(c_flat[valid].tolist(), t_flat[valid].tolist()):
            if cq in z_to_row and tq in data_set:
                expected[z_to_row[cq], tq] = 1
            elif tq in z_to_row and cq in data_set:
                expected[z_to_row[tq], cq] = 1
        np.testing.assert_array_equal(mat, expected)

        # Every row must hit only data qubits (never ancillas), and only zcheck
        # ancillas (never xcheck) are kept.
        x_set = set(int(q) for q in np.asarray(layout["xcheck_qubits"]).reshape(-1).tolist())
        for i, z in enumerate(zcheck.tolist()):
            hit_cols = np.nonzero(mat[i])[0]
            for q in hit_cols.tolist():
                self.assertIn(int(q), data_set, f"row {i} (zcheck={z}) hits non-data qubit {q}")
                self.assertNotIn(int(q), x_set, f"row {i} (zcheck={z}) hits xcheck {q}")

    def test_feedforward_frame_update_equals_mz_times_connectivity(self):
        """Mz × connectivity (mod 2) gives the X-flip pattern on data qubits."""
        from qec.precompute_dem import (
            _build_color_memory_circuit,
            _extract_color_round_layout,
            build_circuit_z_ancilla_connectivity_matrix,
        )

        distance = 3
        n_rounds = 2
        basis = "X"
        schedule = "nearest-neighbor"
        circ = _build_color_memory_circuit(
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            schedule=schedule,
            p_scalar=0.005,
        )
        layout = _extract_color_round_layout(
            circ=circ,
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            schedule=schedule,
        )
        controls = np.asarray(layout["cnot_circuit"][:, :, 0], dtype=np.int32)
        targets = np.asarray(layout["cnot_circuit"][:, :, 1], dtype=np.int32)
        zcheck = np.asarray(layout["zcheck_qubits"], dtype=np.int32).reshape(-1)
        data = np.asarray(layout["data_qubits"], dtype=np.int32).reshape(-1)
        nq = int(layout["nq"])

        ff_mask = build_circuit_z_ancilla_connectivity_matrix(
            controls=controls,
            targets=targets,
            data_qubits=data,
            zcheck_qubits=zcheck,
            nq=nq,
        )

        rng = np.random.default_rng(7)
        B = 5
        mz = rng.integers(0, 2, size=(B, zcheck.size), dtype=np.uint8)

        # Mz × ff_mask under XOR (mod 2): the resulting (B, nq) matrix should
        # have a 1 at (b, q) exactly when an odd number of zcheck ancillas that
        # fired in shot b are CX-coupled to qubit q.
        result = (mz.astype(np.int64) @ ff_mask.astype(np.int64)) % 2
        self.assertEqual(result.shape, (B, nq))

        expected = np.zeros_like(result)
        for b in range(B):
            for i, z in enumerate(zcheck.tolist()):
                if int(mz[b, i]) == 0:
                    continue
                hit = np.nonzero(ff_mask[i])[0]
                for q in hit.tolist():
                    expected[b, int(q)] ^= 1
        np.testing.assert_array_equal(result.astype(np.uint8), expected.astype(np.uint8))

        # Sanity: never updates ancilla qubits, only data qubits.
        ancilla_q = np.array(
            list(
                set(int(q) for q in zcheck.tolist()) |
                set(int(q) for q in np.asarray(layout["xcheck_qubits"]).reshape(-1).tolist())
            ),
            dtype=np.int64,
        )
        if ancilla_q.size > 0:
            np.testing.assert_array_equal(result[:, ancilla_q], np.zeros_like(result[:, ancilla_q]))


class TestColorMemoryCircuitTorchNoiseModelRebuild(unittest.TestCase):
    """Verifies ColorMemoryCircuitTorch rebuilds self.p from a runtime NoiseModel.

    The augmented-DEM H matrix is structural and stays the same; only the
    probability vector should reflect the 25p NoiseModel rates instead of
    the scalar p the bundle was exported with.
    """

    def test_noise_model_overrides_loaded_scalar_p(self):
        from qec.noise_model import CNOT_ERROR_TYPES, NoiseModel
        from qec.color_code.memory_circuit_torch import ColorMemoryCircuitTorch

        distance = 3
        n_rounds = 2
        basis = "X"
        schedule = "nearest-neighbor"

        cnot_probs = {f"p_cnot_{k}": 0.00011 + i * 0.00001 for i, k in enumerate(CNOT_ERROR_TYPES)}
        nm = NoiseModel(
            p_prep_X=0.0011,
            p_prep_Z=0.0022,
            p_meas_X=0.0033,
            p_meas_Z=0.0044,
            p_idle_cnot_X=0.0051,
            p_idle_cnot_Y=0.0052,
            p_idle_cnot_Z=0.0053,
            p_idle_spam_X=0.0061,
            p_idle_spam_Y=0.0062,
            p_idle_spam_Z=0.0063,
            **cnot_probs,
        )

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            _write_synthetic_bundle(
                tmp,
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                schedule=schedule,
            )

            mc_scalar = ColorMemoryCircuitTorch(
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                schedule=schedule,
                precomputed_frames_dir=str(tmp),
                device=torch.device("cpu"),
            )
            mc_nm = ColorMemoryCircuitTorch(
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                schedule=schedule,
                precomputed_frames_dir=str(tmp),
                device=torch.device("cpu"),
                noise_model=nm,
            )

        # Bundle uses scalar p=0.001 (see _write_synthetic_bundle), so the
        # scalar-mode p vector should carry exactly that value (or 0).
        scalar_p = mc_scalar.p.cpu().numpy()
        self.assertTrue((scalar_p == 0.001).any())

        # NoiseModel-mode p vector must surface the per-fault rates from the
        # 25p model and must not carry the scalar-derived placeholders.
        nm_p = mc_nm.p.cpu().numpy()
        for v in (
            nm.p_idle_cnot_X,
            nm.p_idle_cnot_Y,
            nm.p_idle_cnot_Z,
            nm.p_cnot_IX,
            nm.p_cnot_ZZ,
        ):
            self.assertTrue(
                np.any(np.isclose(nm_p, v, rtol=0.0, atol=1e-9)),
                f"Expected 25p value {v} in noise-model-mode p vector",
            )
        for scalar_value in (0.001, 0.001 / 3.0, 0.001 / 15.0, 2.0 * 0.001 / 3.0):
            self.assertFalse(
                np.any(np.isclose(nm_p, scalar_value, rtol=0.0, atol=1e-9)),
                f"Unexpected scalar-derived value {scalar_value} in noise-model-mode p vector",
            )

        # The structural H matrix must be identical regardless of noise mode.
        np.testing.assert_array_equal(
            mc_scalar.bundle.H.cpu().numpy(),
            mc_nm.bundle.H.cpu().numpy(),
        )

    def test_color_generator_threads_noise_model_to_both_basis_sims(self):
        """ColorQCDataGeneratorTorch must forward noise_model to sim_X and sim_Z."""
        from qec.noise_model import NoiseModel
        from data.generator_torch_color import ColorQCDataGeneratorTorch

        nm = NoiseModel.from_single_p(0.005)

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            for basis in ("X", "Z"):
                _write_synthetic_bundle(
                    tmp,
                    distance=3,
                    n_rounds=2,
                    basis=basis,
                    schedule="nearest-neighbor",
                )
            gen = ColorQCDataGeneratorTorch(
                distance=3,
                n_rounds=2,
                schedule="nearest-neighbor",
                measure_basis="both",
                precomputed_frames_dir=str(tmp),
                device=torch.device("cpu"),
                rank=0,
                global_rank=0,
                base_seed=42,
                noise_model=nm,
            )

        self.assertIs(gen.noise_model, nm)
        self.assertIs(gen.sim_X.noise_model, nm)
        self.assertIs(gen.sim_Z.noise_model, nm)


if __name__ == "__main__":
    unittest.main()
