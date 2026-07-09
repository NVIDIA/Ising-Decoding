# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

import numpy as np

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from qec.noise_model import CNOT_ERROR_TYPES, NoiseModel
from qec.precompute_dem import (
    build_single_p_marginal,
    extract_cnot_structure_from_stim_text,
    generate_all_errors_local,
    replicate_metadata_across_rounds,
)
from qec.surface_code.memory_circuit import MemoryCircuit


class TestBuildSinglePMarginalSpamWindows(unittest.TestCase):

    @staticmethod
    def _build_probability_vector(basis):
        distance = 3
        n_rounds = 4
        p_scalar = 0.004
        noise_model = NoiseModel(
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
            **{f"p_cnot_{error_type}": 0.0001 for error_type in CNOT_ERROR_TYPES},
        )

        circuit = MemoryCircuit(
            distance=distance,
            idle_error=p_scalar,
            sqgate_error=p_scalar,
            tqgate_error=p_scalar,
            spam_error=2.0 * p_scalar / 3.0,
            n_rounds=n_rounds,
            basis=basis,
            code_rotation="XV",
            noise_model=noise_model,
        )
        cnot_circuit, cx_times = extract_cnot_structure_from_stim_text(circuit.circuit)
        t_total = int(len(cx_times) + 2)
        nq = int(2 * distance * distance - 1)
        _, metadata_local = generate_all_errors_local(
            t_total=t_total,
            nq=nq,
            controls_by_layer=cnot_circuit,
            cx_times=cx_times,
        )
        metadata_global = replicate_metadata_across_rounds(
            metadata_local=metadata_local,
            n_rounds=n_rounds,
        )

        data_qubits = np.asarray(circuit.code.data_qubits, dtype=np.int32)
        xcheck_qubits = np.asarray(circuit.code.xcheck_qubits, dtype=np.int32)
        zcheck_qubits = np.asarray(circuit.code.zcheck_qubits, dtype=np.int32)
        meas_qubits = np.concatenate([xcheck_qubits, zcheck_qubits]).astype(np.int32)
        meas_bases = np.concatenate(
            [
                np.zeros(len(xcheck_qubits), dtype=np.int32),
                np.ones(len(zcheck_qubits), dtype=np.int32),
            ]
        )
        probabilities = build_single_p_marginal(
            error_metadata_global=metadata_global,
            t_total=t_total,
            n_rounds=n_rounds,
            data_qubits=data_qubits,
            xcheck_qubits=xcheck_qubits,
            zcheck_qubits=zcheck_qubits,
            meas_qubits=meas_qubits,
            meas_bases=meas_bases,
            basis=basis,
            p_scalar=p_scalar,
            noise_model=noise_model,
        )
        return (
            probabilities,
            metadata_global,
            noise_model,
            set(int(q) for q in data_qubits.tolist()),
            t_total,
            n_rounds,
        )

    def test_real_metadata_uses_spam_idle_in_every_non_final_round(self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                probabilities, metadata, noise_model, data_qubits, t_total, n_rounds = (
                    self._build_probability_vector(basis)
                )
                seen = set()
                count = 0
                for error_index, round_index, time_index, qubit, error_type, _ in metadata:
                    if (
                        0 <= round_index < n_rounds - 1 and time_index == t_total - 1 and
                        qubit in data_qubits and len(error_type) == 1
                    ):
                        expected = getattr(noise_model, f"p_idle_spam_{error_type}")
                        self.assertAlmostEqual(
                            float(probabilities[error_index]),
                            float(expected),
                            places=7,
                        )
                        seen.add((round_index, error_type))
                        count += 1

                self.assertEqual(
                    seen,
                    {
                        (round_index, error_type)
                        for round_index in range(n_rounds - 1)
                        for error_type in ("X", "Y", "Z")
                    },
                )
                self.assertEqual(count, len(data_qubits) * 3 * (n_rounds - 1))

    def test_real_metadata_keeps_final_round_quiet_and_bulk_idles_on_cnot_rates(self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                probabilities, metadata, noise_model, data_qubits, t_total, n_rounds = (
                    self._build_probability_vector(basis)
                )
                final_count = 0
                bulk_types = set()
                for error_index, round_index, time_index, qubit, error_type, _ in metadata:
                    if qubit not in data_qubits or len(error_type) != 1:
                        continue
                    if round_index == n_rounds - 1 and time_index == t_total - 1:
                        self.assertEqual(float(probabilities[error_index]), 0.0)
                        final_count += 1
                    elif 0 < time_index < t_total - 1 and round_index < n_rounds - 1:
                        expected = getattr(noise_model, f"p_idle_cnot_{error_type}")
                        self.assertAlmostEqual(
                            float(probabilities[error_index]),
                            float(expected),
                            places=7,
                        )
                        bulk_types.add(error_type)

                self.assertEqual(final_count, len(data_qubits) * 3)
                self.assertEqual(bulk_types, {"X", "Y", "Z"})


class TestBuildSinglePMarginalResetWindowAndFinalRound(unittest.TestCase):

    def test_reset_window_data_idles_carry_no_prior_in_intermediate_rounds(self):
        # The circuit folds the data-qubit idle during the ancilla prep/reset
        # window into the measurement-window SPAM idle and emits no noise at
        # tt == 0 for data qubits outside the boundary rounds.
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                probabilities, metadata, _, data_qubits, _, n_rounds = (
                    TestBuildSinglePMarginalSpamWindows._build_probability_vector(basis)
                )
                count = 0
                for error_index, round_index, time_index, qubit, error_type, _ in metadata:
                    if (
                        0 < round_index < n_rounds - 1 and time_index == 0 and
                        qubit in data_qubits and len(error_type) == 1
                    ):
                        self.assertEqual(float(probabilities[error_index]), 0.0)
                        count += 1

                self.assertEqual(count, len(data_qubits) * 3 * (n_rounds - 2))

    def test_final_round_data_injection_uses_measurement_rate(self):
        # The final perfect round represents noisy data readout as a fake
        # data-measurement flip at the start of the round: Z_ERROR(p_meas_X)
        # for X-basis readout, X_ERROR(p_meas_Z) for Z-basis readout.
        for basis, allowed_type, rate_name in (
            ("X", "Z", "p_meas_X"),
            ("Z", "X", "p_meas_Z"),
        ):
            with self.subTest(basis=basis):
                probabilities, metadata, noise_model, data_qubits, _, n_rounds = (
                    TestBuildSinglePMarginalSpamWindows._build_probability_vector(basis)
                )
                count = 0
                for error_index, round_index, time_index, qubit, error_type, _ in metadata:
                    if (
                        round_index == n_rounds - 1 and time_index == 0 and qubit in data_qubits and
                        len(error_type) == 1
                    ):
                        expected = (
                            getattr(noise_model, rate_name) if error_type == allowed_type else 0.0
                        )
                        self.assertAlmostEqual(
                            float(probabilities[error_index]),
                            float(expected),
                            places=7,
                        )
                        count += 1

                self.assertEqual(count, len(data_qubits) * 3)

    def test_round_zero_data_prep_keeps_prep_rate(self):
        for basis, allowed_type, rate_name in (
            ("X", "Z", "p_prep_X"),
            ("Z", "X", "p_prep_Z"),
        ):
            with self.subTest(basis=basis):
                probabilities, metadata, noise_model, data_qubits, _, _ = (
                    TestBuildSinglePMarginalSpamWindows._build_probability_vector(basis)
                )
                count = 0
                for error_index, round_index, time_index, qubit, error_type, _ in metadata:
                    if (
                        round_index == 0 and time_index == 0 and qubit in data_qubits and
                        len(error_type) == 1
                    ):
                        expected = (
                            getattr(noise_model, rate_name) if error_type == allowed_type else 0.0
                        )
                        self.assertAlmostEqual(
                            float(probabilities[error_index]),
                            float(expected),
                            places=7,
                        )
                        count += 1

                self.assertEqual(count, len(data_qubits) * 3)


class TestSamplerMatchesStimSyndromeDensities(unittest.TestCase):
    """Seeded end-to-end guard against circuit-vs-sampler prior drift.

    Per-round syndrome densities obtained by pushing Bernoulli(p) errors
    through the precomputed H must match direct Stim sampling of the same
    MemoryCircuit. Both paths are seeded, so the comparison is deterministic.
    """

    DISTANCE = 3
    N_ROUNDS = 4
    BASIS = "X"
    SHOTS = 50_000
    CHUNK = 10_000

    @staticmethod
    def _make_noise_model(**overrides):
        params = {
            key: 0.0 for key in (
                "p_prep_X",
                "p_prep_Z",
                "p_meas_X",
                "p_meas_Z",
                "p_idle_cnot_X",
                "p_idle_cnot_Y",
                "p_idle_cnot_Z",
                "p_idle_spam_X",
                "p_idle_spam_Y",
                "p_idle_spam_Z",
            )
        }
        params.update({f"p_cnot_{error_type}": 0.0 for error_type in CNOT_ERROR_TYPES})
        params.update(overrides)
        return NoiseModel(**params)

    def _build_circuit(self, noise_model):
        return MemoryCircuit(
            distance=self.DISTANCE,
            idle_error=0.004,
            sqgate_error=0.004,
            tqgate_error=0.004,
            spam_error=2.0 * 0.004 / 3.0,
            n_rounds=self.N_ROUNDS,
            basis=self.BASIS,
            code_rotation="XV",
            noise_model=noise_model,
        )

    def _model_round_densities(self, noise_model):
        import torch

        from qec.dem_sampling import measure_from_stacked_frames
        from qec.precompute_dem import precompute_dem_bundle_surface_code

        artifacts = precompute_dem_bundle_surface_code(
            distance=self.DISTANCE,
            n_rounds=self.N_ROUNDS,
            basis=self.BASIS,
            code_rotation="XV",
            p_scalar=0.004,
            dem_output_dir=None,
            device=torch.device("cpu"),
            export=False,
            return_artifacts=True,
            noise_model=noise_model,
        )
        H, p, nq = artifacts["H"], artifacts["p"], artifacts["nq"]
        circuit = self._build_circuit(noise_model)
        xcheck_qubits = np.asarray(circuit.code.xcheck_qubits, dtype=np.int64)
        zcheck_qubits = np.asarray(circuit.code.zcheck_qubits, dtype=np.int64)
        meas_qubits = torch.from_numpy(np.concatenate([xcheck_qubits, zcheck_qubits]))
        meas_bases = torch.from_numpy(
            np.concatenate(
                [
                    np.zeros(len(xcheck_qubits), np.int64),
                    np.ones(len(zcheck_qubits), np.int64),
                ]
            )
        )

        generator = torch.Generator().manual_seed(20260709)
        H_t = H.t().float()
        totals = torch.zeros((self.N_ROUNDS, len(meas_qubits)), dtype=torch.float64)
        remaining = self.SHOTS
        while remaining > 0:
            batch = min(self.CHUNK, remaining)
            errors = (torch.rand((batch, H.shape[1]), generator=generator) < p[None, :]).float()
            frames = (errors @ H_t).remainder(2).to(torch.uint8)
            measurements = measure_from_stacked_frames(frames, meas_qubits, meas_bases, nq)
            syndromes = measurements.clone()
            syndromes[:, 1:] ^= measurements[:, :-1]
            totals += syndromes.to(torch.float64).sum(dim=0)
            remaining -= batch
        return (totals / self.SHOTS).numpy(), len(xcheck_qubits)

    def _stim_round_densities(self, noise_model):
        import stim

        circuit = self._build_circuit(noise_model)
        sampler = stim.Circuit(circuit.circuit).compile_sampler(seed=20260709)
        n_ancillas = len(circuit.code.xcheck_qubits) + len(circuit.code.zcheck_qubits)
        measurements = sampler.sample(self.SHOTS)
        measurements = (
            measurements[:, :self.N_ROUNDS *
                         n_ancillas].reshape(self.SHOTS, self.N_ROUNDS,
                                             n_ancillas).astype(np.uint8)
        )
        syndromes = measurements.copy()
        syndromes[:, 1:] ^= measurements[:, :-1]
        return syndromes.mean(axis=0)

    def _assert_densities_match(self, noise_model):
        model, n_xchecks = self._model_round_densities(noise_model)
        reference = self._stim_round_densities(noise_model)
        # Round-0 Z-check outcomes are projection-random in the X basis; the
        # frame-based model measures deviations from the noise-free reference,
        # so exclude those cells.
        valid = np.ones_like(reference, dtype=bool)
        valid[0, n_xchecks:] = False
        for round_index in range(self.N_ROUNDS):
            model_mean = model[round_index][valid[round_index]].mean()
            reference_mean = reference[round_index][valid[round_index]].mean()
            n_cells = int(valid[round_index].sum()) * self.SHOTS
            stderr = float(np.sqrt(max(reference_mean * (1.0 - reference_mean), 1e-12) / n_cells))
            self.assertLess(
                abs(model_mean - reference_mean),
                5.0 * stderr + 1e-4,
                msg=(
                    f"round {round_index}: model syndrome density {model_mean:.6f} "
                    f"vs stim {reference_mean:.6f}"
                ),
            )

    def test_cnot_idle_only_matches_stim(self):
        self._assert_densities_match(
            self._make_noise_model(p_idle_cnot_X=0.002, p_idle_cnot_Y=0.002, p_idle_cnot_Z=0.002)
        )

    def test_measurement_only_matches_stim(self):
        self._assert_densities_match(self._make_noise_model(p_meas_X=0.004, p_meas_Z=0.004))

    def test_full_noise_model_matches_stim(self):
        self._assert_densities_match(
            self._make_noise_model(
                p_prep_X=0.004,
                p_prep_Z=0.004,
                p_meas_X=0.004,
                p_meas_Z=0.004,
                p_idle_cnot_X=0.002,
                p_idle_cnot_Y=0.002,
                p_idle_cnot_Z=0.002,
                p_idle_spam_X=0.003984,
                p_idle_spam_Y=0.003984,
                p_idle_spam_Z=0.003984,
                **{f"p_cnot_{error_type}": 0.0004 for error_type in CNOT_ERROR_TYPES},
            )
        )


if __name__ == "__main__":
    unittest.main()
