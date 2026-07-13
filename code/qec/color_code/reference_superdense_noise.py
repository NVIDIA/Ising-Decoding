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
Reference Si1000 noise semantics for fixed superdense/CX color-code circuits.

This module keeps the existing superdense circuit skeleton and detector layout
from ``MemoryCircuit`` intact, but rebuilds the raw Stim circuit with a
separate, operation-by-operation noise-instruction model intended to match the
paper's Si1000 semantics more faithfully than the current approximate
``NoiseModel`` path.

The implementation goal is narrow:
- keep the current superdense/CX schedule fixed
- keep the current logical-round structure fixed
- only change which noise instructions are injected, and when
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import stim

from qec.color_code.memory_circuit import MemoryCircuit
from qec.noise_model import NoiseModel

# Extracted from the local PGF figure:
# literature/arXiv-2508.15743v1/paper_plots/superdensesi_vs_chromob.pgf
# These are X-memory, logical-error-per-round marker values for the
# superdense Si1000 comparison figure.
PAPER_SUPERDENSE_SI1000_ORACLE = {
    "chromobius":
        {
            5:
                {
                    1.0e-4: 4.06144e-05,
                    5.0e-4: 4.95438e-04,
                    1.0e-3: 1.81149e-03,
                    2.0e-3: 7.99629e-03,
                    3.0e-3: 2.00239e-02,
                    4.0e-3: 3.56376e-02,
                },
            7:
                {
                    1.0e-4: 1.01430e-06,
                    5.0e-4: 7.49778e-05,
                    1.0e-3: 5.73005e-04,
                    2.0e-3: 4.91306e-03,
                    3.0e-3: 1.60765e-02,
                    4.0e-3: 3.56073e-02,
                },
            11:
                {
                    1.0e-4: 9.09089e-09,
                    5.0e-4: 2.61828e-06,
                    1.0e-3: 7.78991e-05,
                    2.0e-3: 2.17557e-03,
                    3.0e-3: 1.35672e-02,
                    4.0e-3: 4.21925e-02,
                },
        },
    "this_work":
        {
            5:
                {
                    1.0e-4: 9.60009e-07,
                    5.0e-4: 7.16870e-05,
                    1.0e-3: 5.43991e-04,
                    2.0e-3: 3.38561e-03,
                    3.0e-3: 1.17826e-02,
                    4.0e-3: 2.44104e-02,
                },
            7:
                {
                    1.0e-4: 7.14286e-08,
                    5.0e-4: 1.10301e-05,
                    1.0e-3: 1.45734e-04,
                    2.0e-3: 2.13128e-03,
                    3.0e-3: 8.57147e-03,
                    4.0e-3: 2.45874e-02,
                },
            11:
                {
                    5.0e-4: 4.68575e-07,
                    1.0e-3: 1.45919e-05,
                    2.0e-3: 7.11701e-04,
                    3.0e-3: 6.11811e-03,
                    4.0e-3: 2.42294e-02,
                },
        },
}


@dataclass(frozen=True)
class Si1000ReferenceNoiseSpec:
    """
    Exact instruction-level Si1000 contract for the fixed current gate set.

    Channel meanings mirror the paper, but are attached to our current
    superdense/CX circuit skeleton:
    - CX 2Q noise: total p depolarizing after each physical CX layer
    - 1Q Clifford noise: total p/10 depolarizing after each explicit 1Q Clifford
    - init/reset error: basis-flip 2p after explicit standalone reset
    - measurement error: basis-flip 5p before measurement
    - idle during gate windows: total p/10 depolarizing on qubits not touched by
      the current physical gate layer
    - idle during measure/reset windows: total 2p depolarizing on qubits not
      participating in the current MR window

    The fixed current circuit family uses fused ``MRX/MRZ`` ancilla operations.
    For the reference-noise/Chromobius backend we expand those to explicit
    ``M`` + ``R`` in the raw Stim circuit so the measurement record stream is
    preserved, but we deliberately do *not* add a second 2p post-``MR`` prep
    flip. That extra reset fault creates unsupported same-basis hyperedges for
    Chromobius on this detector layout. Standalone ``R/RX`` instructions still
    receive the full 2p initialization error.
    """

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if self.measure_error_probability > 1.0:
            raise ValueError(
                f"Si1000 measurement error 5p exceeds 1 for p={self.p}. "
                "Reference semantics require 5p <= 1."
            )
        if self.prep_error_probability > 1.0:
            raise ValueError(
                f"Si1000 reset error 2p exceeds 1 for p={self.p}. "
                "Reference semantics require 2p <= 1."
            )
        if sum(self.measure_reset_idle_args()) > 1.0:
            raise ValueError(
                f"Si1000 measure/reset idle channel exceeds probability 1 for p={self.p}."
            )

    @property
    def prep_error_probability(self) -> float:
        return 2.0 * self.p

    @property
    def measure_error_probability(self) -> float:
        return 5.0 * self.p

    def one_qubit_gate_args(self) -> tuple[float, float, float]:
        q = self.p / 10.0
        return (q / 3.0, q / 3.0, q / 3.0)

    def gate_idle_args(self) -> tuple[float, float, float]:
        q = self.p / 10.0
        return (q / 3.0, q / 3.0, q / 3.0)

    def measure_reset_idle_args(self) -> tuple[float, float, float]:
        q = 2.0 * self.p
        return (q / 3.0, q / 3.0, q / 3.0)

    def two_qubit_gate_args(self) -> tuple[float, ...]:
        return (self.p / 15.0,) * 15


def _convert_targets(targets: Iterable[Any]) -> list[Any]:
    converted: list[Any] = []
    for tgt in targets:
        if isinstance(tgt, tuple) and len(tgt) == 2 and tgt[0] == "rec":
            converted.append(stim.target_rec(int(tgt[1])))
        else:
            converted.append(tgt)
    return converted


def _int_targets(targets: Iterable[Any]) -> list[int]:
    return [int(t) for t in targets if isinstance(t, int)]


def _append_existing_operation(
    circuit: stim.Circuit,
    name: str,
    targets: Iterable[Any],
    arg: Any,
) -> None:
    converted_targets = _convert_targets(targets)
    no_arg_names = {
        "R",
        "RX",
        "RZ",
        "H",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "TICK",
        "CX",
        "M",
        "MR",
        "MX",
        "MRX",
        "MZ",
        "MRZ",
    }
    if name in no_arg_names:
        circuit.append_operation(name, converted_targets)
    else:
        circuit.append_operation(name, converted_targets, arg)


def _append_basis_flip_error(
    circuit: stim.Circuit,
    *,
    qubits: Iterable[int],
    basis: str,
    probability: float,
) -> None:
    q_list = [int(q) for q in qubits]
    if probability <= 0 or not q_list:
        return
    basis = basis.upper()
    if basis == "Z":
        circuit.append_operation("X_ERROR", q_list, probability)
    elif basis == "X":
        circuit.append_operation("Z_ERROR", q_list, probability)
    else:
        raise ValueError(f"Unsupported basis {basis!r}")


def _append_pauli_channel_1(
    circuit: stim.Circuit,
    *,
    qubits: Iterable[int],
    args: tuple[float, float, float],
) -> None:
    q_list = [int(q) for q in qubits]
    if not q_list or sum(args) <= 0:
        return
    circuit.append_operation("PAULI_CHANNEL_1", q_list, list(args))


def _append_pauli_channel_2(
    circuit: stim.Circuit,
    *,
    flat_pairs: Iterable[int],
    args: tuple[float, ...],
) -> None:
    q_list = [int(q) for q in flat_pairs]
    if not q_list or sum(args) <= 0:
        return
    circuit.append_operation("PAULI_CHANNEL_2", q_list, list(args))


def summarize_reference_noise_semantics(
    stim_circuit_raw: stim.Circuit,
    *,
    p: float | None = None,
) -> dict[str, int | float]:
    """
    Count key instruction classes for comparing current vs reference semantics.
    """

    summary = {
        "two_qubit_noise_ops": 0,
        "two_qubit_noise_targets": 0,
        "gate_idle_noise_ops": 0,
        "gate_idle_noise_targets": 0,
        "measure_reset_idle_noise_ops": 0,
        "measure_reset_idle_noise_targets": 0,
        "other_pauli1_noise_ops": 0,
        "other_pauli1_noise_targets": 0,
        "prep_flip_ops": 0,
        "prep_flip_targets": 0,
        "meas_flip_ops": 0,
        "meas_flip_targets": 0,
        "reset_flip_after_mr_ops": 0,
        "reset_flip_after_mr_targets": 0,
    }
    expected_gate_idle_total = (p / 10.0) if p is not None else None
    expected_measure_reset_idle_total = (2.0 * p) if p is not None else None
    expected_prep_error = (2.0 * p) if p is not None else None
    expected_meas_error = (5.0 * p) if p is not None else None

    prev_name: str | None = None
    prev_prev_name: str | None = None
    tol = 1e-8
    for name, targets, arg in stim_circuit_raw.flattened_operations():
        if name == "PAULI_CHANNEL_2":
            summary["two_qubit_noise_ops"] += 1
            summary["two_qubit_noise_targets"] += len(_int_targets(targets))
        elif name in ("PAULI_CHANNEL_1", "DEPOLARIZE1"):
            if name == "PAULI_CHANNEL_1":
                total = float(sum(arg)) if isinstance(arg, list) and len(arg) == 3 else None
            else:
                total = float(arg) if isinstance(arg, (int, float)) else None
            if (
                total is not None and expected_measure_reset_idle_total is not None and
                abs(total - expected_measure_reset_idle_total) < tol
            ):
                summary["measure_reset_idle_noise_ops"] += 1
                summary["measure_reset_idle_noise_targets"] += len(_int_targets(targets))
            elif (
                total is not None and expected_gate_idle_total is not None and
                abs(total - expected_gate_idle_total) < tol
            ):
                summary["gate_idle_noise_ops"] += 1
                summary["gate_idle_noise_targets"] += len(_int_targets(targets))
            else:
                summary["other_pauli1_noise_ops"] += 1
                summary["other_pauli1_noise_targets"] += len(_int_targets(targets))
        elif name in ("X_ERROR", "Z_ERROR"):
            q = float(arg) if isinstance(arg, (int, float)) else None
            target_count = len(_int_targets(targets))
            if q is not None and abs(q) > 0:
                if expected_prep_error is not None and abs(q - expected_prep_error) < tol:
                    summary["prep_flip_ops"] += 1
                    summary["prep_flip_targets"] += target_count
                    if prev_name in ("MR", "MRX", "MRZ") or (
                        prev_name in ("R", "RX", "RZ") and prev_prev_name in ("M", "MX", "MZ")
                    ):
                        summary["reset_flip_after_mr_ops"] += 1
                        summary["reset_flip_after_mr_targets"] += target_count
                elif expected_meas_error is not None and abs(q - expected_meas_error) < tol:
                    summary["meas_flip_ops"] += 1
                    summary["meas_flip_targets"] += target_count

        prev_prev_name = prev_name
        prev_name = name

    return summary


class ReferenceNoiseMemoryCircuit:
    """
    Wrapper around ``MemoryCircuit`` with a separate reference Si1000 injector.

    The underlying noiseless superdense/CX circuit, detector layout, and
    feedforward structure come from ``MemoryCircuit``. This wrapper replaces
    only the raw noise instructions.
    """

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        basis: str,
        p: float,
        schedule: str = "nearest-neighbor",
        add_boundary_detectors: bool = True,
        add_physical_coords: bool = False,
        flip_triangle: bool = False,
    ) -> None:
        self.reference_spec = Si1000ReferenceNoiseSpec(float(p))
        self.base_circuit = MemoryCircuit(
            distance=distance,
            idle_error=0.0,
            sqgate_error=0.0,
            tqgate_error=0.0,
            spam_error=0.0,
            n_rounds=n_rounds,
            basis=basis,
            add_boundary_detectors=add_boundary_detectors,
            schedule=schedule,
            add_physical_coords=add_physical_coords,
            flip_triangle=flip_triangle,
            noise_model=None,
            gidney_style_noise=False,
        )

        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.basis = str(basis).upper()
        self.schedule = schedule
        self.code = self.base_circuit.code
        self.circuit = str(self._build_reference_raw_circuit())
        self.stim_circuit_raw = stim.Circuit(self.circuit)
        self.stim_circuit = self.stim_circuit_raw.with_inlined_feedback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_circuit, name)

    def _z_connected_data_by_z_ancilla(self) -> dict[int, list[int]]:
        return self.base_circuit._z_connected_data_by_z_ancilla()

    def _build_reference_raw_circuit(self) -> stim.Circuit:
        spec = self.reference_spec
        result = stim.Circuit()

        all_qubits = frozenset(int(q) for q in self.base_circuit.all_qubits)
        xcheck_set = frozenset(int(q) for q in self.code.xcheck_qubits)
        zcheck_set = frozenset(int(q) for q in self.code.zcheck_qubits)

        # ``MemoryCircuit`` defines n_rounds as the full number of stabilizer
        # rounds including state-prep and logical-measurement rounds.
        round_idx = 0

        for name, targets, arg in self.base_circuit.stim_circuit_raw.flattened_operations():
            int_targets = _int_targets(targets)
            int_target_set = frozenset(int_targets)
            in_logical_measurement_round = round_idx == (self.n_rounds - 1)

            if name in ("RX", "R", "RZ"):
                _append_existing_operation(result, name, targets, arg)
                reset_basis = "X" if name == "RX" else "Z"
                _append_basis_flip_error(
                    result,
                    qubits=int_targets,
                    basis=reset_basis,
                    probability=spec.prep_error_probability,
                )
                continue

            if name == "CX":
                _append_existing_operation(result, name, targets, arg)

                # Classical feedforward uses record targets; treat it as noiseless.
                if any(isinstance(t, tuple) and len(t) == 2 and t[0] == "rec" for t in targets):
                    continue

                if in_logical_measurement_round:
                    continue

                _append_pauli_channel_2(
                    result,
                    flat_pairs=int_targets,
                    args=spec.two_qubit_gate_args(),
                )
                idle_targets = sorted(all_qubits - int_target_set)
                _append_pauli_channel_1(
                    result,
                    qubits=idle_targets,
                    args=spec.gate_idle_args(),
                )
                continue

            if name in ("H", "S", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG"):
                _append_existing_operation(result, name, targets, arg)
                if in_logical_measurement_round:
                    continue
                _append_pauli_channel_1(
                    result,
                    qubits=int_targets,
                    args=spec.one_qubit_gate_args(),
                )
                idle_targets = sorted(all_qubits - int_target_set)
                _append_pauli_channel_1(
                    result,
                    qubits=idle_targets,
                    args=spec.gate_idle_args(),
                )
                continue

            if name in ("MR", "MRX", "MRZ"):
                meas_basis = "X" if name.endswith("X") else "Z"
                meas_name = "MX" if meas_basis == "X" else "MZ"
                reset_name = "RX" if meas_basis == "X" else "R"
                if not in_logical_measurement_round:
                    _append_basis_flip_error(
                        result,
                        qubits=int_targets,
                        basis=meas_basis,
                        probability=spec.measure_error_probability,
                    )

                # Model reference semantics as explicit measurement followed by reset.
                result.append_operation(meas_name, int_targets)
                result.append_operation(reset_name, int_targets)

                if not in_logical_measurement_round:
                    idle_targets = sorted(all_qubits - int_target_set)
                    _append_pauli_channel_1(
                        result,
                        qubits=idle_targets,
                        args=spec.measure_reset_idle_args(),
                    )

                if name in ("MRX",) and int_target_set == xcheck_set:
                    round_idx += 1
                elif name in ("MR",) and int_target_set == xcheck_set:
                    round_idx += 1
                elif name in ("MRZ",) and int_target_set == zcheck_set and not xcheck_set:
                    round_idx += 1

                continue

            if name in ("M", "MX", "MZ"):
                meas_basis = "X" if name.endswith("X") else "Z"
                if not in_logical_measurement_round:
                    _append_basis_flip_error(
                        result,
                        qubits=int_targets,
                        basis=meas_basis,
                        probability=spec.measure_error_probability,
                    )

                _append_existing_operation(result, name, targets, arg)

                continue

            _append_existing_operation(result, name, targets, arg)

        return result


def build_reference_noise_memory_circuit(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    p: float,
    schedule: str = "nearest-neighbor",
    add_boundary_detectors: bool = True,
    add_physical_coords: bool = False,
    flip_triangle: bool = False,
) -> ReferenceNoiseMemoryCircuit:
    return ReferenceNoiseMemoryCircuit(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        p=p,
        schedule=schedule,
        add_boundary_detectors=add_boundary_detectors,
        add_physical_coords=add_physical_coords,
        flip_triangle=flip_triangle,
    )


def build_color_memory_circuit(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    p_error: float,
    noise_model_family: str,
    noise_instruction_semantics: str,
    noise_model: NoiseModel | None = None,
    gidney_style_noise: bool = False,
    schedule: str = "nearest-neighbor",
    add_boundary_detectors: bool = True,
) -> MemoryCircuit | ReferenceNoiseMemoryCircuit:
    """
    Build the appropriate color-code circuit for the chosen noise semantics.
    """
    if noise_instruction_semantics not in ("current", "reference"):
        raise ValueError(f"Unsupported noise_instruction_semantics={noise_instruction_semantics!r}")
    if noise_model_family not in ("legacy", "si1000"):
        raise ValueError(f"Unsupported noise_model_family={noise_model_family!r}")

    if noise_instruction_semantics == "reference":
        if noise_model_family != "si1000":
            raise ValueError(
                "reference noise semantics currently require "
                "noise_model_family='si1000'."
            )
        return build_reference_noise_memory_circuit(
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            p=p_error,
            schedule=schedule,
            add_boundary_detectors=add_boundary_detectors,
        )

    resolved_noise_model = noise_model
    if resolved_noise_model is None and noise_model_family == "si1000":
        resolved_noise_model = NoiseModel.from_si1000(float(p_error))

    p_placeholder = (
        float(resolved_noise_model.get_max_probability())
        if resolved_noise_model is not None else float(p_error)
    )
    return MemoryCircuit(
        distance=distance,
        idle_error=p_placeholder,
        sqgate_error=p_placeholder,
        tqgate_error=p_placeholder,
        spam_error=(2.0 / 3.0) * p_placeholder,
        n_rounds=n_rounds,
        basis=basis,
        noise_model=resolved_noise_model,
        gidney_style_noise=gidney_style_noise,
        schedule=schedule,
        add_boundary_detectors=add_boundary_detectors,
    )
