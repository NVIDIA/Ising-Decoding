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


if __name__ == "__main__":
    unittest.main()
