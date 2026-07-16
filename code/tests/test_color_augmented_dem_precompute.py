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
"""Color-code augmented DEM precompute tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_color_precompute_p_vector_uses_25p_values(basis):
    """Color DEM precompute should build p from the explicit 25p model, not scalar p/3 or p/K."""
    from qec.noise_model import CNOT_ERROR_TYPES, NoiseModel
    from qec.precompute_dem import precompute_dem_bundle_color_code

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
        **cnot_probs
    )

    artifacts = precompute_dem_bundle_color_code(
        distance=3,
        n_rounds=3,
        basis=basis,
        schedule="nearest-neighbor",
        p_scalar=0.1234,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=64,
        buffer_size=1,
        noise_model=nm,
    )
    p_values = artifacts["p"].cpu().numpy()

    expected_values = [
        nm.p_idle_cnot_X,
        nm.p_idle_cnot_Y,
        nm.p_idle_cnot_Z,
        nm.p_cnot_IX,
        nm.p_cnot_ZZ,
    ]
    for expected in expected_values:
        assert np.any(np.isclose(p_values, expected, rtol=0.0, atol=1e-9)
                     ), (f"Expected 25p probability {expected} in color DEM p vector")

    scalar_derived_values = [0.1234 / 3.0, 0.1234 / 15.0, 2.0 * 0.1234 / 3.0]
    for scalar_value in scalar_derived_values:
        assert not np.any(
            np.isclose(p_values, scalar_value, rtol=0.0, atol=1e-9)
        ), (f"Unexpected scalar-derived probability {scalar_value} in 25p color DEM p vector")


@pytest.mark.parametrize(
    (
        "basis",
        "final_error_type",
        "final_probability",
        "data_prep_error_type",
        "data_prep_probability",
    ),
    [
        ("X", "Z", 0.0044, "Z", 0.0022),
        ("Z", "X", 0.0033, "X", 0.0011),
    ],
)
def test_color_single_qubit_noise_probabilities_follow_pauli_labels(
    basis,
    final_error_type,
    final_probability,
    data_prep_error_type,
    data_prep_probability,
):
    from qec.noise_model import NoiseModel
    from qec.precompute_dem import build_single_p_marginal_color_code

    noise_model = NoiseModel(
        p_prep_X=0.0011,
        p_prep_Z=0.0022,
        p_meas_X=0.0033,
        p_meas_Z=0.0044,
        p_idle_spam_X=0.0061,
        p_idle_spam_Y=0.0062,
        p_idle_spam_Z=0.0063,
    )
    error_types = ("X", "Y", "Z")
    metadata = [(0, -1, -1, -1, "I", -1)]
    metadata.extend(
        (index, 1, 3, 0, error_type, -1) for index, error_type in enumerate(error_types, start=1)
    )
    metadata.extend(
        (index, 2, 0, 0, error_type, -1) for index, error_type in enumerate(error_types, start=4)
    )
    metadata.extend(
        [
            (7, 1, 0, 1, "Z", -1),
            (8, 1, 3, 1, "Z", -1),
            (9, 1, 0, 2, "X", -1),
            (10, 1, 3, 2, "X", -1),
            (11, 0, 0, 0, data_prep_error_type, -1),
        ]
    )

    probabilities = build_single_p_marginal_color_code(
        error_metadata_global=metadata,
        t_total=4,
        n_rounds=3,
        data_qubits=np.array([0], dtype=np.int32),
        xcheck_qubits=np.array([1], dtype=np.int32),
        zcheck_qubits=np.array([2], dtype=np.int32),
        basis=basis,
        p_scalar=0.0,
        noise_model=noise_model,
    )

    np.testing.assert_allclose(
        probabilities[1:4],
        [noise_model.p_idle_spam_X, noise_model.p_idle_spam_Y, noise_model.p_idle_spam_Z],
        rtol=0.0,
        atol=1e-9,
    )
    expected_final = np.zeros(3, dtype=np.float32)
    expected_final[error_types.index(final_error_type)] = final_probability
    np.testing.assert_allclose(probabilities[4:7], expected_final, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        probabilities[7:11],
        [
            noise_model.p_prep_Z,
            noise_model.p_meas_Z,
            noise_model.p_prep_X,
            noise_model.p_meas_X,
        ],
        rtol=0.0,
        atol=1e-9,
    )
    assert probabilities[11] == pytest.approx(data_prep_probability)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_color_scalar_data_idle_probability_remains_p_scalar_over_three(basis):
    """The 25p SPAM fix must not change legacy scalar data-idle priors."""
    from qec.precompute_dem import build_single_p_marginal_color_code

    p_scalar = 0.03
    metadata = [(0, -1, -1, -1, "I", -1)]
    metadata.extend(
        (index, 1, 3, 0, error_type, -1)
        for index, error_type in enumerate(("X", "Y", "Z"), start=1)
    )

    probabilities = build_single_p_marginal_color_code(
        error_metadata_global=metadata,
        t_total=4,
        n_rounds=3,
        data_qubits=np.array([0], dtype=np.int32),
        xcheck_qubits=np.array([1], dtype=np.int32),
        zcheck_qubits=np.array([2], dtype=np.int32),
        basis=basis,
        p_scalar=p_scalar,
        noise_model=None,
    )

    np.testing.assert_allclose(
        probabilities[1:4],
        [p_scalar / 3.0] * 3,
        rtol=0.0,
        atol=1e-9,
    )


def test_color_noiseless_context_restores_noise_state_after_error():
    from qec.color_code.memory_circuit import Circuit
    from qec.noise_model import NoiseModel

    circuit = Circuit(np.array([0], dtype=np.int32))
    noise_model = NoiseModel(p_meas_X=0.01)
    original_rates = (0.1, 0.2, 0.3, 0.4)
    circuit.set_noise_model(noise_model)
    circuit.set_error_rates_simple(*original_rates)
    circuit.set_error_rates()
    original_error_rates = dict(circuit.error_rates)

    with pytest.raises(RuntimeError, match="test failure"):
        with circuit._temporarily_noiseless() as (saved_noise_model, saved_rates):
            assert saved_noise_model is noise_model
            assert saved_rates == original_rates
            assert circuit.noise_model is None
            assert (
                circuit.idle_error,
                circuit.sqgate_error,
                circuit.tqgate_error,
                circuit.spam_error,
            ) == (0, 0, 0, 0)
            raise RuntimeError("test failure")

    assert circuit.noise_model is noise_model
    assert (
        circuit.idle_error,
        circuit.sqgate_error,
        circuit.tqgate_error,
        circuit.spam_error,
    ) == original_rates
    assert circuit.error_rates == original_error_rates


def test_color_noiseless_context_restores_noise_state_normally():
    from qec.color_code.memory_circuit import Circuit
    from qec.noise_model import NoiseModel

    circuit = Circuit(np.array([0], dtype=np.int32))
    noise_model = NoiseModel(p_meas_Z=0.02)
    original_rates = (0.11, 0.22, 0.33, 0.44)
    circuit.set_noise_model(noise_model)
    circuit.set_error_rates_simple(*original_rates)
    circuit.set_error_rates()
    original_error_rates = dict(circuit.error_rates)

    with circuit._temporarily_noiseless() as (saved_noise_model, saved_rates):
        assert saved_noise_model is noise_model
        assert saved_rates == original_rates
        assert circuit.noise_model is None
        assert (
            circuit.idle_error,
            circuit.sqgate_error,
            circuit.tqgate_error,
            circuit.spam_error,
        ) == (0, 0, 0, 0)

    assert circuit.noise_model is noise_model
    assert (
        circuit.idle_error,
        circuit.sqgate_error,
        circuit.tqgate_error,
        circuit.spam_error,
    ) == original_rates
    assert circuit.error_rates == original_error_rates


@pytest.mark.parametrize(
    (
        "basis",
        "final_instruction",
        "final_error_type",
        "data_prep_probability",
        "final_probability",
    ),
    [
        ("X", "Z_ERROR", "Z", 0.023, 0.047),
        ("Z", "X_ERROR", "X", 0.019, 0.031),
    ],
)
def test_color_stim_and_dem_data_noise_locations_match(
    basis,
    final_instruction,
    final_error_type,
    data_prep_probability,
    final_probability,
):
    from qec.color_code.memory_circuit import MemoryCircuit
    from qec.noise_model import NoiseModel
    from qec.precompute_dem import (
        _extract_color_round_layout,
        build_single_p_marginal_color_code,
        generate_all_errors_local,
        replicate_metadata_across_rounds,
    )

    noise_model = NoiseModel(
        p_prep_X=0.019,
        p_prep_Z=0.023,
        p_meas_X=0.031,
        p_meas_Z=0.047,
        p_idle_spam_X=0.061,
        p_idle_spam_Y=0.073,
        p_idle_spam_Z=0.089,
    )
    n_rounds = 4
    circuit = MemoryCircuit(
        distance=3,
        idle_error=0.0,
        sqgate_error=0.0,
        tqgate_error=0.0,
        spam_error=0.0,
        n_rounds=n_rounds,
        basis=basis,
        noise_model=noise_model,
        add_detectors=False,
        schedule="nearest-neighbor",
    )
    data_qubits = set(int(q) for q in circuit.code.data_qubits)
    xcheck_qubits = set(int(q) for q in circuit.code.xcheck_qubits)
    zcheck_qubits = set(int(q) for q in circuit.code.zcheck_qubits)

    stim_idle_ops = []
    stim_data_basis_error_ops = []
    stim_xcheck_error_ops = []
    stim_zcheck_error_ops = []
    for instruction in circuit.stim_circuit_raw.flattened():
        name = instruction.name
        arguments = instruction.gate_args_copy()
        argument = arguments[0] if len(arguments) == 1 else arguments
        int_targets = {
            target.value for target in instruction.targets_copy() if target.is_qubit_target
        }
        if name == "PAULI_CHANNEL_1" and int_targets == data_qubits:
            stim_idle_ops.append(argument)
        if (name == final_instruction and int_targets == data_qubits):
            stim_data_basis_error_ops.append(float(argument))
        if name == "Z_ERROR" and int_targets == xcheck_qubits:
            stim_xcheck_error_ops.append(float(argument))
        if name == "X_ERROR" and int_targets == zcheck_qubits:
            stim_zcheck_error_ops.append(float(argument))

    assert len(stim_idle_ops) == n_rounds - 1
    for argument in stim_idle_ops:
        np.testing.assert_allclose(
            argument,
            [
                noise_model.p_idle_spam_X,
                noise_model.p_idle_spam_Y,
                noise_model.p_idle_spam_Z,
            ],
            rtol=0.0,
            atol=1e-10,
        )
    assert sorted(stim_data_basis_error_ops) == pytest.approx(
        sorted([data_prep_probability, final_probability])
    )
    assert sorted(stim_xcheck_error_ops) == pytest.approx(
        sorted([noise_model.p_prep_Z, noise_model.p_meas_Z] * (n_rounds - 1))
    )
    assert sorted(stim_zcheck_error_ops) == pytest.approx(
        sorted([noise_model.p_prep_X, noise_model.p_meas_X] * (n_rounds - 1))
    )

    layout = _extract_color_round_layout(
        circ=circuit,
        distance=3,
        n_rounds=n_rounds,
        basis=basis,
        schedule="nearest-neighbor",
    )
    _, metadata_local = generate_all_errors_local(
        t_total=layout["t_total"],
        nq=layout["nq"],
        controls_by_layer=layout["cnot_circuit"],
        cx_times=layout["cx_times"],
    )
    metadata = replicate_metadata_across_rounds(
        metadata_local=metadata_local,
        n_rounds=n_rounds,
    )
    probabilities = build_single_p_marginal_color_code(
        error_metadata_global=metadata,
        t_total=layout["t_total"],
        n_rounds=n_rounds,
        data_qubits=layout["data_qubits"],
        xcheck_qubits=layout["xcheck_qubits"],
        zcheck_qubits=layout["zcheck_qubits"],
        basis=basis,
        p_scalar=0.0,
        noise_model=noise_model,
    )

    idle_rates = {
        "X": noise_model.p_idle_spam_X,
        "Y": noise_model.p_idle_spam_Y,
        "Z": noise_model.p_idle_spam_Z,
    }
    dem_idle_locations = [
        (round_index, time_index, qubit, error_type, float(probabilities[error_index]))
        for error_index, round_index, time_index, qubit, error_type, _ in metadata
        if round_index < n_rounds - 1 and time_index == layout["t_total"] -
        1 and qubit in data_qubits and error_type in idle_rates
    ]
    assert len(dem_idle_locations) == (n_rounds - 1) * len(data_qubits) * 3
    for _, _, _, error_type, probability in dem_idle_locations:
        assert probability == pytest.approx(idle_rates[error_type])

    dem_final_locations = [
        (error_type, float(probabilities[error_index]))
        for error_index, round_index, time_index, qubit, error_type, _ in metadata
        if round_index == n_rounds -
        1 and time_index == 0 and qubit in data_qubits and float(probabilities[error_index]) != 0.0
    ]
    assert len(dem_final_locations) == len(data_qubits)
    for error_type, probability in dem_final_locations:
        assert error_type == final_error_type
        assert probability == pytest.approx(final_probability)


def test_color_precompute_p_vector_scalar_path_unchanged_when_noise_model_none():
    """Passing noise_model=None must reproduce the legacy scalar p vector bit-for-bit."""
    from qec.precompute_dem import precompute_dem_bundle_color_code

    common_kwargs = dict(
        distance=3,
        n_rounds=2,
        basis="X",
        schedule="nearest-neighbor",
        p_scalar=0.01,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=256,
        buffer_size=1,
    )
    baseline = precompute_dem_bundle_color_code(**common_kwargs)
    with_none = precompute_dem_bundle_color_code(**common_kwargs, noise_model=None)
    np.testing.assert_array_equal(baseline["p"].cpu().numpy(), with_none["p"].cpu().numpy())
    np.testing.assert_array_equal(
        baseline["H"].cpu().numpy(),
        with_none["H"].cpu().numpy(),
    )


def test_color_precompute_export_writes_noise_metadata(tmp_path):
    """Exported color DEM should record noise_mode + sha256 + canonical_parameters."""
    import json

    from qec.noise_model import NoiseModel
    from qec.precompute_dem import (
        DEM_ARTIFACT_METADATA_KEY,
        get_color_augmented_dem_paths,
        precompute_dem_bundle_color_code,
    )

    nm = NoiseModel.from_single_p(0.005)

    distance, n_rounds, basis, schedule = 3, 2, "X", "nearest-neighbor"
    precompute_dem_bundle_color_code(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=0.123,
        dem_output_dir=str(tmp_path),
        device=torch.device("cpu"),
        export=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=64,
        buffer_size=1,
        noise_model=nm,
    )

    paths = get_color_augmented_dem_paths(
        tmp_path,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    with np.load(paths["p"], allow_pickle=False) as z:
        raw = z[DEM_ARTIFACT_METADATA_KEY]
        meta = json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))

    assert meta["noise_mode"] == "noise_model"
    assert meta["noise_model_sha256"] == nm.sha256()
    assert meta["noise_model"] == nm.canonical_parameters()
    assert meta["code"] == "color"


def test_color_precompute_export_scalar_metadata_unchanged(tmp_path):
    """When noise_model is omitted, exported metadata should still record noise_mode=scalar."""
    import json

    from qec.precompute_dem import (
        DEM_ARTIFACT_METADATA_KEY,
        get_color_augmented_dem_paths,
        precompute_dem_bundle_color_code,
    )

    distance, n_rounds, basis, schedule = 3, 2, "Z", "nearest-neighbor"
    precompute_dem_bundle_color_code(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=0.004,
        dem_output_dir=str(tmp_path),
        device=torch.device("cpu"),
        export=True,
        enable_z_feedforward=True,
        apply_data_x_override=True,
        use_decomposed_errors=False,
        chunk_size=64,
        buffer_size=1,
    )

    paths = get_color_augmented_dem_paths(
        tmp_path,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    with np.load(paths["p"], allow_pickle=False) as z:
        raw = z[DEM_ARTIFACT_METADATA_KEY]
        meta = json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))

    assert meta["noise_mode"] == "scalar"
    assert "noise_model_sha256" not in meta
    assert float(meta["p_scalar"]) == pytest.approx(0.004)


def test_color_augmented_dem_rejects_nonproduction_feedforward_modes():
    from qec.precompute_dem import precompute_dem_bundle_color_code

    common = dict(
        distance=3,
        n_rounds=3,
        basis="X",
        schedule="nearest-neighbor",
        p_scalar=0.01,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        use_decomposed_errors=False,
    )
    with pytest.raises(ValueError, match="enable_z_feedforward=True"):
        precompute_dem_bundle_color_code(
            **common,
            enable_z_feedforward=False,
            apply_data_x_override=True,
        )
    with pytest.raises(ValueError, match="apply_data_x_override=True"):
        precompute_dem_bundle_color_code(
            **common,
            enable_z_feedforward=True,
            apply_data_x_override=False,
        )
