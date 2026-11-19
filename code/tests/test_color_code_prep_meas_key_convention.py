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
"""Convention of the four prep/meas noise-model keys in the color-code path.

``p_prep_X``, ``p_prep_Z``, ``p_meas_X`` and ``p_meas_Z`` are named by the
basis of the reset or measurement that fails, not by the Pauli that is
applied: an X-basis failure is a ``Z_ERROR``, a Z-basis failure is an
``X_ERROR``. The surface-code builder, the color-code builder and the
color-code DEM precompute must all read the keys in this sense.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.color_code.memory_circuit import MemoryCircuit as ColorMemoryCircuit
from qec.noise_model import NoiseModel
from qec.precompute_dem import build_single_p_marginal_color_code
from qec.surface_code.memory_circuit import MemoryCircuit as SurfaceMemoryCircuit

PROBABILITY = 0.01
BASES = ("X", "Z")

# key -> (Stim instruction, ancilla family that is reset/measured in that
# basis, memory basis in which the data qubits are reset/measured in it).
EXPECTED = {
    "p_prep_X": ("Z_ERROR", "xcheck", "X"),
    "p_prep_Z": ("X_ERROR", "zcheck", "Z"),
    "p_meas_X": ("Z_ERROR", "xcheck", "X"),
    "p_meas_Z": ("X_ERROR", "zcheck", "Z"),
}


def _build_color(basis, noise_model):
    circuit = ColorMemoryCircuit(
        distance=3,
        idle_error=0.0,
        sqgate_error=0.0,
        tqgate_error=0.0,
        spam_error=0.0,
        n_rounds=3,
        basis=basis,
        noise_model=noise_model,
        add_detectors=False,
        schedule="nearest-neighbor",
    )
    return circuit, circuit.stim_circuit_raw


def _build_surface(basis, noise_model):
    circuit = SurfaceMemoryCircuit(
        distance=3,
        idle_error=0.0,
        sqgate_error=0.0,
        tqgate_error=0.0,
        spam_error=0.0,
        n_rounds=3,
        basis=basis,
        noise_model=noise_model,
    )
    return circuit, circuit.stim_circuit


def _flip_error_sites(build, basis, noise_model):
    """Return {(instruction, probability): qubit families hit} for a circuit."""
    circuit, stim_circuit = build(basis, noise_model)
    families = {
        "data": {int(q) for q in circuit.code.data_qubits},
        "xcheck": {int(q) for q in circuit.code.xcheck_qubits},
        "zcheck": {int(q) for q in circuit.code.zcheck_qubits},
    }
    sites = {}
    for instruction in stim_circuit.flattened():
        if instruction.name not in ("X_ERROR", "Z_ERROR"):
            continue
        targets = {target.value for target in instruction.targets_copy()}
        key = (instruction.name, round(instruction.gate_args_copy()[0], 9))
        hit = sites.setdefault(key, set())
        hit.update(name for name, qubits in families.items() if targets & qubits)
    return {key: frozenset(hit) for key, hit in sites.items()}


def _expected_sites(key, basis):
    instruction, ancilla_family, data_basis = EXPECTED[key]
    families = {ancilla_family}
    if basis == data_basis:
        families.add("data")
    return {(instruction, PROBABILITY): frozenset(families)}


class TestColorCodePrepMeasKeyConvention(unittest.TestCase):

    def test_single_key_maps_to_failing_basis_in_color_circuit(self):
        for key in EXPECTED:
            for basis in BASES:
                with self.subTest(key=key, basis=basis):
                    sites = _flip_error_sites(_build_color, basis, NoiseModel(**{key: PROBABILITY}))
                    self.assertEqual(sites, _expected_sites(key, basis))

    def test_z_basis_memory_uses_z_keys_for_x_errors(self):
        """Z-basis prep/meas failures (X_ERROR) come from the *_Z keys only."""
        for key, other in (("p_prep_Z", "p_prep_X"), ("p_meas_Z", "p_meas_X")):
            with self.subTest(key=key):
                sites = _flip_error_sites(_build_color, "Z", NoiseModel(**{key: PROBABILITY}))
                self.assertIn(("X_ERROR", PROBABILITY), sites)
                sites = _flip_error_sites(_build_color, "Z", NoiseModel(**{other: PROBABILITY}))
                self.assertNotIn("X_ERROR", {instruction for instruction, _ in sites})

    def test_color_and_surface_builders_agree(self):
        for key in EXPECTED:
            for basis in BASES:
                with self.subTest(key=key, basis=basis):
                    noise_model = NoiseModel(**{key: PROBABILITY})
                    self.assertEqual(
                        _flip_error_sites(_build_color, basis, noise_model),
                        _flip_error_sites(_build_surface, basis, noise_model),
                    )

    def test_color_and_surface_builders_agree_with_distinct_values(self):
        noise_model = NoiseModel(p_prep_X=0.011, p_prep_Z=0.022, p_meas_X=0.033, p_meas_Z=0.044)
        for basis in BASES:
            with self.subTest(basis=basis):
                color_sites = _flip_error_sites(_build_color, basis, noise_model)
                self.assertEqual(
                    set(color_sites),
                    {
                        ("Z_ERROR", 0.011),
                        ("X_ERROR", 0.022),
                        ("Z_ERROR", 0.033),
                        ("X_ERROR", 0.044),
                    },
                )
                self.assertEqual(color_sites, _flip_error_sites(_build_surface, basis, noise_model))

    def test_color_dem_marginals_use_basis_labeled_keys(self):
        """The precomputed color p-vector assigns the same key as the Stim circuit."""
        noise_model = NoiseModel(p_prep_X=0.011, p_prep_Z=0.022, p_meas_X=0.033, p_meas_Z=0.044)
        t_total = 4
        last = t_total - 1
        # (error index, round, time step, qubit, Pauli, second qubit); qubit 0 is
        # data, qubit 1 an X-check ancilla, qubit 2 a Z-check ancilla.
        for basis, pauli, p_data_prep, p_data_meas in (
            ("X", "Z", noise_model.p_prep_X, noise_model.p_meas_X),
            ("Z", "X", noise_model.p_prep_Z, noise_model.p_meas_Z),
        ):
            metadata = [
                (0, -1, -1, -1, "I", -1),
                (1, 1, 0, 1, "Z", -1),  # X-check ancilla prep
                (2, 1, last, 1, "Z", -1),  # X-check ancilla measurement
                (3, 1, 0, 2, "X", -1),  # Z-check ancilla prep
                (4, 1, last, 2, "X", -1),  # Z-check ancilla measurement
                (5, 0, 0, 0, pauli, -1),  # data prep (first round)
                (6, 2, 0, 0, pauli, -1),  # data readout flip (final round)
            ]
            with self.subTest(basis=basis):
                probabilities = build_single_p_marginal_color_code(
                    error_metadata_global=metadata,
                    t_total=t_total,
                    n_rounds=3,
                    data_qubits=np.array([0], dtype=np.int32),
                    xcheck_qubits=np.array([1], dtype=np.int32),
                    zcheck_qubits=np.array([2], dtype=np.int32),
                    basis=basis,
                    p_scalar=0.0,
                    noise_model=noise_model,
                )
                np.testing.assert_allclose(
                    probabilities[1:],
                    [
                        noise_model.p_prep_X,
                        noise_model.p_meas_X,
                        noise_model.p_prep_Z,
                        noise_model.p_meas_Z,
                        p_data_prep,
                        p_data_meas,
                    ],
                    rtol=0.0,
                    atol=1e-9,
                )


if __name__ == "__main__":
    unittest.main()
