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

import numpy as np


def unfold_repeat_instruction(circuit_lines: list, ignore_detectors: bool = False):
    repeat_block = []
    read_block = False
    num_times = 1
    for line in circuit_lines:
        if read_block:
            repeat_block.append(line.strip())
        if "REPEAT" in line:
            num_times = int(line.split("REPEAT ")[1].split(' {')[0])
            read_block = True

    unfolded_circuit_lines = []
    read_line = True
    for line in circuit_lines:
        if "REPEAT" in line:
            read_line = False
        if read_line:
            unfolded_circuit_lines.append(line.strip())

    # Assume num_times is the code distance: we thus have num_times = code_distance**2 - 1

    for round in range(1, num_times + 1):
        for line in repeat_block[:-1]:  # Ignore last item: it is the closing brace
            if "DETECTOR" not in line:
                unfolded_circuit_lines.append(line)
            else:
                if not ignore_detectors:
                    # Figure out the correct detector ids based on the round number
                    det1 = line.split("rec[")[1].split("]")[0]
                    det2 = line.split("rec[")[2].split("]")[0]
                    shifted_det1 = int(det1) - (num_times - round) * (num_times**2 - 1)
                    shifted_det2 = int(det2) - (num_times - round) * (num_times**2 - 1)

                    unfolded_circuit_lines.append(
                        f"DETECTOR rec[{shifted_det1}] rec[{shifted_det2}]"
                    )

    # unfolded_circuit_lines.extend(repeat_block[:-1] * num_times) # Ignore last item: it is the closing brace

    return unfolded_circuit_lines


def add_instruction(flips: dict):
    margin = ""
    for gate, qubit_list in flips.items():
        instruction = " ".join(qubit_list.astype(str))
        if instruction:
            if 'I' in gate:
                margin += gate.replace('I', '') + " " + instruction + '\n'
            elif len(gate) == 1:
                margin += gate + " " + instruction + '\n'
            else:
                instruction = qubit_list.reshape(-1, 2)
                q1 = instruction[:, 0]
                q2 = instruction[:, 1]
                ins1 = " ".join(q1.astype(str))
                ins2 = " ".join(q2.astype(str))
                margin += gate[0] + " " + ins1 + '\n'
                margin += gate[1] + " " + ins2 + '\n'
    return margin


def extract_circuit_realization(
    circuit_lines: list, ignore_detectors: bool = False, ignore_tick: bool = False
):
    # print(len(circuit_lines))
    for i, line in enumerate(circuit_lines):
        if ignore_tick:
            if "TICK" in line:
                circuit_lines[i] = ""
                continue
        if ignore_detectors:
            if "DETECTOR" in line:
                circuit_lines[i] = ""
                continue

        if "X_ERROR" in line:
            flips = {"X": []}
            p = float(line.split("(")[1].split(")")[0])
            # print("X_ERROR line")
            qbts = line.split(") ")[1].split(" ")
            qbts = [int(qbt) for qbt in qbts if qbt != ""]
            # print(qbts)
            flip_idx = np.random.choice(2, size=len(qbts), p=[1 - p, p])
            flip_idx = np.where(flip_idx)[0]
            flip_idx = flip_idx.astype(int)
            # print(flip_idx, type(flip_idx))
            flip_qbts = np.take(qbts, flip_idx)
            flips["X"] = flip_qbts
            # flip_qbts = qbts[flip_idx]
            # print(f"X {" ".join(flip_qbts.astype(str))}")

            margin = add_instruction(flips)
            margin = margin.strip()
            # print(margin)
            if margin:
                circuit_lines[i] = margin
            else:
                circuit_lines[i] = ""

        if "DEPOLARIZE1" in line:
            flips = {"X": [], "Y": [], "Z": []}
            # print("DEPOLARIZE1 line")
            p = float(line.split("(")[1].split(")")[0])
            qbts = line.split(") ")[1].split(" ")
            qbts = np.array([int(qbt) for qbt in qbts if qbt != ""])
            paulis = np.random.choice(4, size=len(qbts), p=[1 - p, p / 3, p / 3, p / 3])
            for j, gate in enumerate(flips.keys()):
                flips[gate] = qbts[(np.where(paulis == j + 1)[0]).astype(int)]

            margin = add_instruction(flips)
            margin = margin.strip()
            # print(margin)
            if margin:
                circuit_lines[i] = margin
            else:
                circuit_lines[i] = ""

        if "DEPOLARIZE2" in line:
            # print("DEPOLARIZE2 line")
            flips = {
                "IX": [],
                "IY": [],
                "IZ": [],
                "XI": [],
                "XX": [],
                "XY": [],
                "XZ": [],
                "YI": [],
                "YX": [],
                "YY": [],
                "YZ": [],
                "ZI": [],
                "ZX": [],
                "ZY": [],
                "ZZ": []
            }
            p = float(line.split("(")[1].split(")")[0])
            # p = 0.5
            qbts = line.split(") ")[1].split(" ")
            qbts = np.array([int(qbt) for qbt in qbts if qbt != ""]).reshape(-1, 2)
            # qbts = []
            paulis = np.random.choice(16, size=len(qbts), p=[1 - p] + [p / 15] * 15)
            for j, gate in enumerate(flips.keys()):
                if gate[0] == 'I':
                    flips[gate] = qbts[(np.where(paulis == j + 1)[0]).astype(int)][:, 1].reshape(-1)
                elif gate[1] == 'I':
                    flips[gate] = qbts[(np.where(paulis == j + 1)[0]).astype(int)][:, 0].reshape(-1)
                else:
                    flips[gate] = qbts[(np.where(paulis == j + 1)[0]).astype(int)].reshape(-1)

            margin = add_instruction(flips)
            margin = margin.strip()
            # print(margin)
            if margin:
                circuit_lines[i] = margin
            else:
                circuit_lines[i] = ""

    circuit_lines = [line for line in circuit_lines if line != ""]
    circuit_lines = [line for line in circuit_lines if line != "}"]
    circuit_lines = "\n".join(circuit_lines)
    return circuit_lines


if __name__ == "__main__":
    import stim
    from memory_circuit import MemoryCircuit
    import time

    d = 3
    p = 0
    n_rounds = d

    circ = MemoryCircuit(
        distance=d,
        idle_error=p,
        sqgate_error=p,
        tqgate_error=p,
        spam_error=p,
        n_rounds=n_rounds,
        add_tick=False,
        add_detectors=False
    )

    # circuit_lines = circ.circuit.split('\n')
    # unfolded_circuit_lines = unfold_repeat_instruction(circuit_lines, ignore_detectors=True)
    # extracted_circuit = extract_circuit_realization(unfolded_circuit_lines, ignore_detectors=True, ignore_tick=True)
    # print(extracted_circuit)
    # C = to_circuit_matrix_C(extracted_circuit.split('\n'), d, ignore_H=True)
    # print(C[:6].T)
