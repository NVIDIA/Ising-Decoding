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
import stim
import copy
from typing import Optional

# Import NoiseModel (will be available after noise_model.py is created)
try:
    from qec.noise_model import NoiseModel
except ImportError:
    NoiseModel = None


class SurfaceCode:
    """
    Surface code class.
    
    This class generates a rotated surface code with customizable boundary conditions.
    Currently only odd distances are supported; support for even distances will be 
    added in the future.
    
    Args:
        distance (int): The distance of the surface code (must be odd).
        first_bulk_syndrome_type (str, optional): Type of the first bulk syndrome 
            qubit ('X' or 'Z'). Defaults to 'X'.
        rotated_type (str, optional): Orientation of the northwestern-most boundary
            ('H' for horizontal or 'V' for vertical). This parameter actually 
            exchanges the direction of the corresponding logical operator. For example,
            if first_bulk_syndrome_type='X' and rotated_type='H', this creates a 
            vertical X logical operator. Defaults to 'H'.
    
    Attributes:
        distance (int): The distance of the surface code.
        first_bulk_syndrome_type (str): Type of the first bulk syndrome qubit.
        rotated_type (str): Orientation of the northwestern-most boundary.
        logical_direction (str): Direction of the logical operators (e.g., 'XV', 'ZH').
        data_qubits (dict): Dictionary containing data qubit information.
        xcheck_qubits (dict): Dictionary containing X-type syndrome qubit information.
        zcheck_qubits (dict): Dictionary containing Z-type syndrome qubit information.
    """

    def __init__(self, distance, first_bulk_syndrome_type='X', rotated_type='V'):
        assert distance % 2 == 1, "Distance must be odd"
        self.distance = distance
        self.first_bulk_syndrome_type = first_bulk_syndrome_type
        self.rotated_type = rotated_type
        self.logical_direction = first_bulk_syndrome_type + ('V' if rotated_type == 'H' else 'H')
        self.code_dict = self._generate_code()
        self.data_qubits_dict = self.code_dict["data"]
        self.xcheck_qubits_dict = self.code_dict["syndrome_X"]
        self.zcheck_qubits_dict = self.code_dict["syndrome_Z"]
        self.all_qubits = np.array(
            range(
                len(self.data_qubits_dict) + len(self.xcheck_qubits_dict) +
                len(self.zcheck_qubits_dict)
            )
        )
        self.data_qubits = np.array(list(self.data_qubits_dict.keys()))
        self.xcheck_qubits = np.array(list(self.xcheck_qubits_dict.keys()))
        self.zcheck_qubits = np.array(list(self.zcheck_qubits_dict.keys()))
        self.hx = np.zeros((len(self.xcheck_qubits), len(self.data_qubits)))
        self.hz = np.zeros((len(self.zcheck_qubits), len(self.data_qubits)))
        self.lx = np.zeros((self.distance, self.distance))
        self.lz = np.zeros((self.distance, self.distance))
        self.lx[0, :self.distance] = 1
        self.lz[0, :self.distance] = 1

        for x in self.xcheck_qubits:
            for y in self.xcheck_qubits_dict[x]["plaquette"]["qubit_id"]:
                if y != -1:
                    self.hx[x - len(self.data_qubits), y] = 1
        for z in self.zcheck_qubits:
            for y in self.zcheck_qubits_dict[z]["plaquette"]["qubit_id"]:
                if y != -1:
                    self.hz[z - len(self.data_qubits) - len(self.xcheck_qubits), y] = 1

        if self.logical_direction == "XH":
            self.lx = self.lx.reshape(1, -1)
            self.lz = self.lz.T.reshape(1, -1)
        elif self.logical_direction == "XV":
            self.lx = self.lx.T.reshape(1, -1)
            self.lz = self.lz.reshape(1, -1)
        elif self.logical_direction == "ZH":
            self.lx = self.lx.T.reshape(1, -1)
            self.lz = self.lz.reshape(1, -1)
        elif self.logical_direction == "ZV":
            self.lx = self.lx.reshape(1, -1)
            self.lz = self.lz.T.reshape(1, -1)

    def _generate_code(self):
        """Generate dictionary with data qubits in odd rows and syndrome qubits in even rows.
        
        Args:
            distance: Surface code distance (D)
            first_bulk_syndrome_type: First bulk syndrome qubit type ('X' or 'Z')
            rotated_type: Choose where the boundary syndrome qubits are placed ('H' or 'V')
            
        Returns:
            Dictionary with 'data', 'syndrome_X' and 'syndrome_Z' keys containing qubit coordinates
        """
        code_dict = {
            'data': {
                q: {
                    "coord": []
                } for q in range(int(self.distance**2))
            },
            'syndrome_X':
                {
                    x: {
                        "coord": [],
                        "plaquette": {
                            "coord": [],
                            "qubit_id": []
                        },
                        "type": ""
                    } for x in range(
                        int(self.distance**2), int(self.distance**2 + (self.distance**2 - 1) / 2)
                    )
                },
            'syndrome_Z':
                {
                    z: {
                        "coord": [],
                        "plaquette": {
                            "coord": [],
                            "qubit_id": []
                        },
                        "type": ""
                    } for z in range(
                        int(self.distance**2 + (self.distance**2 - 1) /
                            2), int(self.distance**2 + (self.distance**2 - 1))
                    )
                }
        }

        qubit_coord_dict = {'data': [], 'syndrome_X': [], 'syndrome_Z': []}
        # Data qubits at odd coordinates:
        # + + + + + ...
        # + D + D + ...
        # + + + + + ...
        # + D + D + ...
        # + + + + + ...
        # ...
        X_flag = False if self.first_bulk_syndrome_type.upper() == 'X' else True
        for i in range(self.distance):
            X_flag = not X_flag  # toggle X_flag for each row
            for j in range(self.distance):
                x = 1 + 2 * i
                y = 1 + 2 * j
                qubit_coord_dict['data'].append([x, y])
                if i < self.distance - 1 and j < self.distance - 1:
                    # Add bulk syndrome coordinates
                    if X_flag:
                        qubit_coord_dict['syndrome_X'].append([x + 1, y + 1])
                    else:
                        qubit_coord_dict['syndrome_Z'].append([x + 1, y + 1])
                    X_flag = not X_flag

        # Syndrome qubits at even coordinates
        # H: First syndrome qubit appears at the leftmost boundary of the grid
        # + + + + + ...
        # + D + D + ...
        # S + S + S ...
        # + D + D + ...
        # ...
        # V: First syndrome qubit appears at the top boundary of the grid
        # + + S + ...
        # + D + D ...
        # + + S + ...
        # + D + D ...
        # ...
        position = [0, 0]
        assert self.rotated_type.upper() in ['H', 'V'], "rotated_type must be 'H' or 'V'"
        keep_flag = 0 if self.rotated_type.upper() == 'H' else 1

        X_flag = (self.rotated_type.upper() == 'H') * (
            self.first_bulk_syndrome_type.upper() == 'X'
        ) + (self.rotated_type.upper() == 'V') * (self.first_bulk_syndrome_type.upper() == 'Z')
        # X_flag = True if keep_flag == 0 else False

        for _ in range(4 * self.distance - 1):
            position = self.hop(position, self.distance)
            keep_flag += 1
            keep_flag %= 2

            # print(position, keep_flag)

            # Add boundary syndrome coordinates
            keep = not (keep_flag % 2)
            if keep:
                if position not in [
                    [0, 0], [0, 2 * self.distance], [2 * self.distance, 0],
                    [2 * self.distance, 2 * self.distance]
                ]:
                    if X_flag:
                        qubit_coord_dict['syndrome_X'].append(position)
                    else:
                        qubit_coord_dict['syndrome_Z'].append(position)

            if position in [
                [0, 0], [0, 2 * self.distance], [2 * self.distance, 0],
                [2 * self.distance, 2 * self.distance]
            ]:
                keep_flag -= 1
                keep_flag %= 2
                X_flag = not X_flag

        qubit_coord_dict["data"] = sorted(qubit_coord_dict["data"])
        qubit_coord_dict["syndrome_X"] = sorted(qubit_coord_dict["syndrome_X"])
        temp = [[y, x] for x, y in qubit_coord_dict["syndrome_Z"]]
        temp = sorted(temp)
        qubit_coord_dict["syndrome_Z"] = [[y, x] for x, y in temp]
        # print(f"sorted qubit_coord_dict: {qubit_coord_dict}")
        self._qubit_coord_dict = qubit_coord_dict

        for i, q in enumerate(qubit_coord_dict["data"]):
            code_dict["data"][i]["coord"] = q

        for i, x in enumerate(qubit_coord_dict["syndrome_X"]):
            code_dict["syndrome_X"][i + int(self.distance**2)]["coord"] = x
            if 0 in x or 2 * self.distance in x:
                code_dict["syndrome_X"][i + int(self.distance**2)]["type"] = "boundary"
            else:
                code_dict["syndrome_X"][i + int(self.distance**2)]["type"] = "bulk"

        for i, z in enumerate(qubit_coord_dict["syndrome_Z"]):
            code_dict["syndrome_Z"][i + int(self.distance**2) +
                                    int((self.distance**2 - 1) / 2)]["coord"] = z
            if 0 in z or 2 * self.distance in z:
                code_dict["syndrome_Z"][i + int(self.distance**2) +
                                        int((self.distance**2 - 1) / 2)]["type"] = "boundary"
            else:
                code_dict["syndrome_Z"][i + int(self.distance**2) +
                                        int((self.distance**2 - 1) / 2)]["type"] = "bulk"

        for x in code_dict["syndrome_X"]:
            i, j = code_dict["syndrome_X"][x]["coord"]
            if self.logical_direction == "XH" or self.logical_direction == "ZV":
                candidates = [[i - 1, j + 1], [i + 1, j + 1], [i - 1, j - 1], [i + 1, j - 1]]
            else:
                candidates = [[i - 1, j + 1], [i - 1, j - 1], [i + 1, j + 1], [i + 1, j - 1]]

            for c in candidates:
                if c in qubit_coord_dict["data"]:
                    code_dict["syndrome_X"][x]["plaquette"]["coord"].append(c)
                    code_dict["syndrome_X"][x]["plaquette"]["qubit_id"].append(
                        qubit_coord_dict["data"].index(c)
                    )
                else:
                    code_dict["syndrome_X"][x]["plaquette"]["coord"].append([-1, -1])
                    code_dict["syndrome_X"][x]["plaquette"]["qubit_id"].append(-1)
        # print("qubit_coord_dict: ", qubit_coord_dict["data"])
        for z in code_dict["syndrome_Z"]:
            i, j = code_dict["syndrome_Z"][z]["coord"]
            if self.logical_direction == "XH" or self.logical_direction == "ZV":
                candidates = [[i - 1, j + 1], [i - 1, j - 1], [i + 1, j + 1], [i + 1, j - 1]]
            else:
                candidates = [[i - 1, j + 1], [i + 1, j + 1], [i - 1, j - 1], [i + 1, j - 1]]

            for c in candidates:
                if c in qubit_coord_dict["data"]:
                    code_dict["syndrome_Z"][z]["plaquette"]["coord"].append(c)
                    code_dict["syndrome_Z"][z]["plaquette"]["qubit_id"].append(
                        qubit_coord_dict["data"].index(c)
                    )
                else:
                    code_dict["syndrome_Z"][z]["plaquette"]["coord"].append([-1, -1])
                    code_dict["syndrome_Z"][z]["plaquette"]["qubit_id"].append(-1)
        # for key, value in code_dict.items():
        #     print(key)
        #     for key2, value2 in value.items():
        #         print(key2)
        #         print(value2)
        return code_dict

    def hop(self, position, D):
        """
        Hop to the next position in the boundary of the grid where there is a syndrome qubit.
        
        Args:
            position: Current position in the grid
            D: Distance of the surface code
            
        Returns:
            Next position in the boundary of the grid where there is a syndrome qubit (clock-wise rotation).
        """
        x, y = position
        if x == 0 and y == 0:
            return [0, 2]
        else:
            if x == 0:
                # hop to the right unless y == 2*D
                if y == 2 * D:
                    return [x + 2, y]
                else:
                    return [x, y + 2]
            elif x == 2 * D:
                # hop to the left unless y == 0
                if y == 0:
                    return [x - 2, y]
                else:
                    return [x, y - 2]
            if y == 0:
                # hop up unless x == 0
                if x == 0:
                    return [x, y + 2]
                else:
                    return [x - 2, y]
            if y == 2 * D:
                # hop down unless x == 2*D
                if x == 2 * D:
                    return [x, y - 2]
                else:
                    return [x + 2, y]


# Class adapted (and expanded) from Mingyu Kang's quits package: https://github.com/mkangquantum/quits/blob/main/src/quits/circuit.py
class Circuit:
    '''
    Class containing helper functions for writing Stim circuits (https://github.com/quantumlib/Stim)
    
    Supports two noise modes:
    1. Simple mode: Single error rates (idle_error, sqgate_error, tqgate_error, spam_error)
    2. NoiseModel mode: 22-parameter explicit noise model
    '''

    def __init__(self, all_qubits):

        self.circuit = ''
        self.margin = ''
        self.all_qubits = all_qubits
        self.idle_error = 0.
        self.sqgate_error = 0.
        self.tqgate_error = 0.
        self.spam_error = 0.
        self.noise_model = None  # Optional 22-parameter noise model

    def set_all_qubits(self, all_qubits):
        self.all_qubits = all_qubits

    def set_noise_model(self, noise_model: 'NoiseModel') -> None:
        """
        Set the 22-parameter noise model for circuit generation.
        
        When a NoiseModel is set, the circuit will use:
        - X_ERROR/Z_ERROR for prep/meas with explicit probabilities
        - PAULI_CHANNEL_1 for idle errors (instead of DEPOLARIZE1)
        - PAULI_CHANNEL_2 for CNOT errors (instead of DEPOLARIZE2)
        
        Args:
            noise_model: NoiseModel instance with 22 parameters
        """
        self.noise_model = noise_model
        # Also set simple error rates for backwards compatibility
        if noise_model is not None:
            # Use max probabilities for simple rate fallbacks
            self.spam_error = max(
                noise_model.p_prep_X, noise_model.p_prep_Z, noise_model.p_meas_X,
                noise_model.p_meas_Z
            )
            # In 25p semantics we have two idle families; for legacy scalar placeholders,
            # keep a conservative value.
            self.idle_error = max(
                noise_model.get_total_idle_cnot_probability(),
                noise_model.get_total_idle_spam_probability()
            )
            self.tqgate_error = noise_model.get_total_cnot_probability()

    def set_error_rates_simple(self, idle_error, sqgate_error, tqgate_error, spam_error):
        self.idle_error = idle_error
        self.sqgate_error = sqgate_error
        self.tqgate_error = tqgate_error
        self.spam_error = spam_error

    def start_loop(self, num_rounds):
        c = 'REPEAT %d {\n' % num_rounds
        self.circuit += c
        self.margin = '    '
        return c

    def end_loop(self):
        c = '}\n'
        self.circuit += c
        self.margin = ''
        return c

    def add_tick(self):
        c = self.margin + 'TICK\n'
        self.circuit += c
        return c

    def add_reset(self, qubits, basis='Z'):
        basis = basis.upper()

        c = self.margin
        if basis == 'Z':
            c += 'RZ '  # Reset to |0>
        elif basis == 'X':
            c += 'RX '  # Reset to |+>
        for q in qubits:
            c += '%d ' % q
        c += '\n'

        # Apply preparation errors (basis-labeled semantics):
        # - Z-basis prep (|0>): X flips outcome -> use p_prep_Z (Z-basis prep failure)
        # - X-basis prep (|+>): Z flips outcome -> use p_prep_X (X-basis prep failure)
        if self.noise_model is not None:
            # Use explicit X_ERROR/Z_ERROR with the appropriate basis-labeled probability.
            if basis == 'Z' and self.noise_model.p_prep_Z > 0:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.noise_model.p_prep_Z
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
            elif basis == 'X' and self.noise_model.p_prep_X > 0:
                c += self.margin
                c += 'Z_ERROR(%.10f) ' % self.noise_model.p_prep_X
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
        elif self.spam_error > 0.:
            # Fallback to simple mode
            c += self.margin
            if basis == 'Z':
                c += 'X_ERROR(%.10f) ' % self.spam_error
            elif basis == 'X':
                c += 'Z_ERROR(%.10f) ' % self.spam_error
            for q in qubits:
                c += '%d ' % q
            c += '\n'

        self.circuit += c
        return c

    def add_single_error(self, qubits, error_type):
        """Add a single-qubit error (X or Z) to specified qubits."""
        # Determine error probability
        if self.noise_model is not None:
            # Basis-labeled semantics:
            # - Applying X corresponds to Z-basis prep failure -> p_prep_Z
            # - Applying Z corresponds to X-basis prep failure -> p_prep_X
            if error_type == 'X':
                error_prob = self.noise_model.p_prep_Z
            elif error_type == 'Z':
                error_prob = self.noise_model.p_prep_X
            else:
                error_prob = 0.
        else:
            error_prob = self.spam_error

        if error_prob == 0.:
            return ''

        c = self.margin
        if error_type == 'X':
            c += 'X_ERROR(%.10f) ' % error_prob
        elif error_type == 'Z':
            c += 'Z_ERROR(%.10f) ' % error_prob
        for q in qubits:
            c += '%d ' % q
        c += '\n'

        self.circuit += c
        return c

    def add_idle(self, qubits, logical_measurement=False, idle_kind: str = "cnot"):
        """
        Add idle errors to specified qubits.
        
        When NoiseModel is set, uses PAULI_CHANNEL_1 with explicit (p_X, p_Y, p_Z).
        In 25p noise-model semantics, idle_kind chooses which idle family to apply:
          - idle_kind='cnot': idle during bulk/CNOT layers (default)
          - idle_kind='spam': idle during ancilla prep/reset window for data qubits
        Otherwise uses DEPOLARIZE1 for backwards compatibility.
        """
        if self.noise_model is not None:
            # Use 25-parameter noise model: PAULI_CHANNEL_1(p_X, p_Y, p_Z)
            if idle_kind == "spam":
                p_X, p_Y, p_Z = self.noise_model.to_stim_pauli_channel_1_args_spam()
            else:
                p_X, p_Y, p_Z = self.noise_model.to_stim_pauli_channel_1_args_cnot()
            total_prob = p_X + p_Y + p_Z
            if total_prob == 0.:
                return ''

            c = self.margin
            if not logical_measurement:
                c += 'PAULI_CHANNEL_1(%.10f, %.10f, %.10f) ' % (p_X, p_Y, p_Z)
            else:
                # For logical measurement round, only apply basis-relevant error
                if self.basis == 'X':
                    c += 'Z_ERROR(%.10f) ' % p_Z
                else:
                    c += 'X_ERROR(%.10f) ' % p_X
            for q in qubits:
                c += '%d ' % q
            c += '\n'

            self.circuit += c
            return c
        else:
            # Fallback to simple mode
            if self.idle_error == 0.:
                return ''

            c = self.margin
            if not logical_measurement:
                c += 'DEPOLARIZE1(%.10f) ' % self.idle_error
            else:
                c += 'Z_ERROR(%.10f) ' % self.idle_error if self.basis == 'X' else 'X_ERROR(%.10f) ' % self.idle_error
            for q in qubits:
                c += '%d ' % q
            c += '\n'

            self.circuit += c
            return c

    def add_hadamard(self, qubits):
        """
        Add Hadamard gates with depolarizing errors.
        
        When NoiseModel is set, uses PAULI_CHANNEL_1 (same as idle).
        Otherwise uses DEPOLARIZE1 for backwards compatibility.
        """
        c = self.margin
        c += 'H '
        for q in qubits:
            c += '%d ' % q
        c += '\n'

        if self.noise_model is not None:
            # Use 22-parameter noise model: PAULI_CHANNEL_1 for single-qubit gate error
            p_X, p_Y, p_Z = self.noise_model.to_stim_pauli_channel_1_args()
            total_prob = p_X + p_Y + p_Z
            if total_prob > 0.:
                c += self.margin
                c += 'PAULI_CHANNEL_1(%.10f, %.10f, %.10f) ' % (p_X, p_Y, p_Z)
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
        elif self.sqgate_error > 0.:
            # Fallback to simple mode
            c += self.margin
            c += 'DEPOLARIZE1(%.10f) ' % self.sqgate_error
            for q in qubits:
                c += '%d ' % q
            c += '\n'

        self.circuit += c
        return c

    def add_hadamard_layer(self, qubits, before_measurement=False, add_tick=True):
        c1 = self.add_hadamard(qubits)
        if not before_measurement:
            other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        else:
            # Only consider syndrome qubits to apply idling before measurement
            other_qubits = np.delete(
                np.concatenate([self.code.xcheck_qubits, self.code.zcheck_qubits]),
                np.where(
                    np.isin(
                        np.concatenate([self.code.xcheck_qubits, self.code.zcheck_qubits]), qubits
                    )
                )[0]
            )
        c2 = self.add_idle(other_qubits)
        if add_tick:
            c3 = self.add_tick()
        else:
            c3 = ''
        return c1 + c2 + c3

    def add_cnot(self, qubits):
        """
        Add CNOT gates with errors to specified qubit pairs.
        
        When NoiseModel is set, uses PAULI_CHANNEL_2 with 15 explicit probabilities.
        Otherwise uses DEPOLARIZE2 for backwards compatibility.
        
        Convention: For CNOT from control to target, error "AB" means:
        A is applied to control, B is applied to target.
        """
        c = self.margin
        c += 'CX '
        for q in qubits:
            c += '%d ' % q
        c += '\n'

        if self.noise_model is not None:
            # Use 22-parameter noise model: PAULI_CHANNEL_2 with 15 probabilities
            # Order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
            probs = self.noise_model.to_stim_pauli_channel_2_args()
            total_prob = sum(probs)
            if total_prob > 0.:
                c += self.margin
                # Format: PAULI_CHANNEL_2(pIX, pIY, pIZ, pXI, pXX, ..., pZZ) ctrl tgt
                prob_str = ', '.join('%.10f' % p for p in probs)
                c += 'PAULI_CHANNEL_2(%s) ' % prob_str
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
        elif self.tqgate_error > 0.:
            # Fallback to simple mode
            c += self.margin
            c += 'DEPOLARIZE2(%.10f) ' % self.tqgate_error
            for q in qubits:
                c += '%d ' % q
            c += '\n'

        self.circuit += c
        return c

    def add_cnot_layer(self, qubits, add_tick=True):
        c1 = self.add_cnot(qubits)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        if add_tick:
            c3 = self.add_tick()
        else:
            c3 = ''
        return c1 + c2 + c3

    def add_measure_reset(self, qubits, error_free_reset=False):
        """
        Add measure-and-reset with errors (Z-basis measurement, reset to |0>).
        
        When NoiseModel is set, uses explicit measurement and prep error probabilities.
        """
        c = ''

        # Measurement error (before measurement)
        if self.noise_model is not None:
            # Z-basis measurement failure is modeled as a pre-measurement X flip.
            if self.noise_model.p_meas_Z > 0:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.noise_model.p_meas_Z
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
        elif self.spam_error > 0.:
            c += self.margin
            c += 'X_ERROR(%.10f) ' % self.spam_error
            for q in qubits:
                c += '%d ' % q
            c += '\n'

        c += self.margin
        c += 'MR '  # Measure and reset to |0>
        for q in qubits:
            c += '%d ' % q
        c += '\n'

        # Reset error (after reset, if not error-free)
        if not error_free_reset:
            if self.noise_model is not None:
                # Z-basis reset failure is modeled as a post-reset X flip.
                if self.noise_model.p_prep_Z > 0:
                    c += self.margin
                    c += 'X_ERROR(%.10f) ' % self.noise_model.p_prep_Z
                    for q in qubits:
                        c += '%d ' % q
                    c += '\n'
            elif self.spam_error > 0.:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.spam_error
                for q in qubits:
                    c += '%d ' % q
                c += '\n'

        self.circuit += c
        return c

    def add_measure_reset_layer(self, qubits, error_free_reset=False, add_tick=True):
        c1 = self.add_measure_reset(qubits, error_free_reset)
        other_qubits = np.delete(self.all_qubits, np.where(np.isin(self.all_qubits, qubits))[0])
        c2 = self.add_idle(other_qubits)
        if add_tick:
            c3 = self.add_tick()
        else:
            c3 = ''
        return c1 + c2 + c3

    def add_measure(self, qubits, basis='Z', include_reset=False):
        """
        Add measurement with errors to specified qubits.
        
        When NoiseModel is set, uses explicit X_ERROR/Z_ERROR with measurement probabilities.
        Otherwise uses spam_error for backwards compatibility.
        
        Convention:
        - Z-basis measurement: X error before measurement flips the outcome
        - X-basis measurement: Z error before measurement flips the outcome
        """
        basis = basis.upper()

        c = ''
        # Apply measurement errors (before measurement)
        if self.noise_model is not None:
            # Basis-labeled semantics:
            # - Z-basis measurement failure -> apply X with prob p_meas_Z
            # - X-basis measurement failure -> apply Z with prob p_meas_X
            if basis == 'Z' and self.noise_model.p_meas_Z > 0:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.noise_model.p_meas_Z
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
            elif basis == 'X' and self.noise_model.p_meas_X > 0:
                c += self.margin
                c += 'Z_ERROR(%.10f) ' % self.noise_model.p_meas_X
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
        elif self.spam_error > 0.:
            # Fallback to simple mode
            c += self.margin
            if basis == 'Z':
                c += 'X_ERROR(%.10f) ' % self.spam_error
            elif basis == 'X':
                c += 'Z_ERROR(%.10f) ' % self.spam_error
            for q in qubits:
                c += '%d ' % q
            c += '\n'

        c += self.margin

        if basis == 'Z':
            if include_reset:
                c += 'MRZ '
            else:
                c += 'MZ '
        elif basis == 'X':
            if include_reset:
                c += 'MRX '
            else:
                c += 'MX '
        for q in qubits:
            c += '%d ' % q
        c += '\n'

        self.circuit += c
        return c

    def add_detector(self, inds):
        c = self.margin + 'DETECTOR '
        for ind in inds:
            c += 'rec[-%d] ' % ind
        c += '\n'

        self.circuit += c

    def add_observable(self, observable_no, inds):
        c = self.margin + 'OBSERVABLE_INCLUDE(%d) ' % observable_no
        for ind in inds:
            c += 'rec[-%d] ' % ind
        c += '\n'

        self.circuit += c
        return c

    def add_qubit_coordinates(self, code_dict):
        for qubit in code_dict["data"]:
            c = 'QUBIT_COORDS'
            c += f"({code_dict['data'][qubit]['coord'][0]}, {code_dict['data'][qubit]['coord'][1]}) {qubit}"
            c += '\n'
            self.circuit += c
        for qubit in code_dict["syndrome_X"]:
            c = 'QUBIT_COORDS'
            c += f"({code_dict['syndrome_X'][qubit]['coord'][0]}, {code_dict['syndrome_X'][qubit]['coord'][1]}) {qubit}"
            c += '\n'
            self.circuit += c
        for qubit in code_dict["syndrome_Z"]:
            c = 'QUBIT_COORDS'
            c += f"({code_dict['syndrome_Z'][qubit]['coord'][0]}, {code_dict['syndrome_Z'][qubit]['coord'][1]}) {qubit}"
            c += '\n'
            self.circuit += c
        return c


class MemoryCircuit(Circuit):
    """
    Memory circuit for surface code quantum error correction.
    
    This class generates a complete quantum circuit for implementing a surface code
    memory experiment, including state preparation, stabilizer measurements, and
    logical measurements. The circuit can be configured for different code orientations and includes circuit level noise modeling.
    
    Args:
        distance (int): The distance of the surface code (must be odd).
        idle_error (float): Error rate for idle operations.
        sqgate_error (float): Error rate for single-qubit gates.
        tqgate_error (float): Error rate for two-qubit gates.
        spam_error (float): State preparation and measurement error rate.
        n_rounds (int): Number of stabilizer measurement rounds.
        basis (str, optional): Logical basis for the memory experiment ('Z' or 'X'). 
            Defaults to 'X'.
        get_all_detectors (bool, optional): Whether to include all detector types.
            Defaults to True.
        noisy_init (bool, optional): Whether to include noise in initialization.
            Defaults to True.
        noisy_meas (bool, optional): Whether to include noise in measurements.
            Defaults to False.
        add_tick (bool, optional): Whether to add timing ticks to the circuit.
            Defaults to True.
        add_detectors (bool, optional): Whether to add detector annotations.
            Defaults to True.
        code_rotation (str, optional): Determines the orientation of the surface code patch and its logical operators. Defaults to 'XV'.
            This is a two-character string where:
            - First character ('X' or 'Z'): Type of the first (upper-left-most) bulk syndrome qubit
            - Second character ('H' or 'V'): Orientation of the surface code patch. 
              It determines whether the upper-left-most boundary syndrome qubit is placed horizontally or vertically to the upper-left-most bulk syndrome qubit.
            
            Options and their meaning:
            - 'XV': X-type first bulk syndrome + Vertically placed syndrome boundary qubit → X logical runs horizontally
            - 'XH': X-type first bulk syndrome + Horizontally placed syndrome boundary qubit → X logical runs vertically  
            - 'ZV': Z-type first bulk syndrome + Vertically placed syndrome boundary qubit → Z logical runs horizontally
            - 'ZH': Z-type first bulk syndrome + Horizontally placed syndrome boundary qubit → Z logical runs vertically
            
            Explicitly:
            - 'XV':       SZ    ...
                       D1    D2 ...
                          SX    ...
                        
            - 'XH':             ...
                       D1    D2 ...
                    SZ    SX    ...
                    
            - 'ZV':       SX    ...
                       D1    D2 ...
                          SZ    ...
                        
            - 'ZH':             ...
                       D1    D2 ...
                    SX    SZ    ...
    
    Attributes:
        circuit (str): The complete Stim circuit as a string.
        distance (int): The distance of the surface code.
        n_rounds (int): Number of stabilizer measurement rounds.
        basis (str): Logical basis for the memory experiment.
        code (SurfaceCode): The underlying surface code object.
        
    Example:
        >>> # Create a distance-3 surface code memory circuit
        >>> circ = MemoryCircuit(
        ...     distance=3,
        ...     idle_error=1e-3,
        ...     sqgate_error=1e-3, 
        ...     tqgate_error=1e-3,
        ...     spam_error=1e-3,
        ...     n_rounds=3,
        ...     basis='Z',
        ...     code_rotation='XV'
        ... )
        >>> print(circ.circuit)  # Print the generated Stim circuit
    """
    def __init__(self, distance, idle_error, sqgate_error, tqgate_error, spam_error, n_rounds,\
                          basis='X', get_all_detectors=True, noisy_init=True, noisy_meas=False,
                          add_tick=True, add_detectors=True, code_rotation='XV', noise_model=None,
                          add_boundary_detectors=False):
        """
        Initialize a MemoryCircuit for surface code quantum error correction.
        
        Args:
            distance: Code distance
            idle_error: Idle error rate (used if noise_model is None)
            sqgate_error: Single-qubit gate error rate (used if noise_model is None)
            tqgate_error: Two-qubit gate error rate (used if noise_model is None)
            spam_error: SPAM error rate (used if noise_model is None)
            n_rounds: Number of stabilizer measurement rounds
            basis: Logical basis ('X' or 'Z')
            get_all_detectors: Whether to include all detector types
            noisy_init: Whether to include noise in initialization
            noisy_meas: Whether to include noise in measurements
            add_tick: Whether to add timing ticks
            add_detectors: Whether to add detector annotations
            code_rotation: Code orientation ('XV', 'XH', 'ZV', or 'ZH')
            noise_model: Optional NoiseModel for 22-parameter noise model.
                         If provided, uses explicit per-type probabilities instead
                         of deriving from single error rates.
            add_boundary_detectors: Whether to add boundary detectors comparing final
                         data qubit measurements to last ancilla measurements.
                         Required for proper decoding with PAULI_CHANNEL_2 noise.
        """
        self.circuit = ''
        self.margin = ''
        self.distance = distance
        self.n_rounds = n_rounds  # n_rounds is defined as the number of stabilizer rounds: counting state prep and logical measurement rounds
        self._add_tick = add_tick
        self._add_detectors = add_detectors
        self._add_boundary_detectors = add_boundary_detectors
        self.basis = basis
        get_Z_detectors = True if basis == 'Z' or get_all_detectors else False
        get_X_detectors = True if basis == 'X' or get_all_detectors else False

        self.code = SurfaceCode(
            distance,
            first_bulk_syndrome_type=code_rotation[0],  # X or Z
            rotated_type=code_rotation[1]
        )  # H or V

        super().__init__(self.code.all_qubits)

        # Set error rates: use noise_model if provided, otherwise use simple rates
        if noise_model is not None:
            self.set_noise_model(noise_model)
        else:
            self.set_error_rates_simple(idle_error, sqgate_error, tqgate_error, spam_error)
        self.set_error_rates()

        ################## Logical state prep ##################
        self.add_reset(
            self.code.data_qubits, basis
        )  # Reset data qubits to either |0> or |+> depending on basis

        self._add_stabilizer_round(
            state_prep=True, combine_reset_and_measure=True
        )  # Add stabilizer round for state prep
        # The state_prep flag is used to avoid adding idle locations for data qubits in the first stabilizer round

        ################# Adding detectors for the first stabilizer round ##################
        if basis == 'X':
            if get_X_detectors:
                # e.g. d=3
                # 4 checks for X, 4 for Z
                # First we measure X, then Z

                for i in range(1, len(self.code.xcheck_qubits) + 1)[::-1]:
                    ind = len(self.code.zcheck_qubits) + i
                    # ind = 8, 7, 6, 5
                    # add_detector[ind] = rec[-8], rec[-7], rec[-6], rec[-5]
                    # These are the X measurements, correct.
                    self.add_detector([ind])
        elif basis == 'Z':
            if get_Z_detectors:
                for i in range(1, len(self.code.zcheck_qubits) + 1)[::-1]:
                    # Here we add rec[-4], rec[-3], rec[-2], rec[-1]
                    # These are the Z measurements, correct.
                    self.add_detector([i])

        ############## Logical memory w/ noise ###############
        if (self.n_rounds - 2) > 0:
            self.start_loop(
                self.n_rounds - 2
            )  # -2 because we already did one stabilizer round for state prep and one for logical measurement

            self._add_stabilizer_round(combine_reset_and_measure=True)

            if self._add_detectors:
                if get_Z_detectors:
                    for i in range(1, len(self.code.zcheck_qubits) + 1)[::-1]:
                        ind = len(self.code.xcheck_qubits) + i
                        self.add_detector(
                            [
                                ind,
                                ind + len(self.code.xcheck_qubits) + len(self.code.zcheck_qubits)
                            ]
                        )
                if get_X_detectors:
                    for i in range(1, len(self.code.xcheck_qubits) + 1)[::-1]:
                        self.add_detector(
                            [i, i + len(self.code.xcheck_qubits) + len(self.code.zcheck_qubits)]
                        )

            self.end_loop()

        ################## Logical measurement ##################
        # Our convention for this task: perform one final perfect stabilizer round (with reset errors),
        # add detectors for all stabilizers (paired against the immediately previous round),
        # then measure all data qubits in the logical basis without adding detectors, and add one logical observable.

        self._add_stabilizer_round(logical_measurement=True, combine_reset_and_measure=True)

        if self._add_detectors:
            # Pair final perfect round checks vs previous round checks
            if get_Z_detectors:
                for i in range(1, len(self.code.zcheck_qubits) + 1)[::-1]:
                    ind = len(self.code.xcheck_qubits) + i
                    self.add_detector(
                        [ind, ind + len(self.code.xcheck_qubits) + len(self.code.zcheck_qubits)]
                    )
            if get_X_detectors:
                for i in range(1, len(self.code.xcheck_qubits) + 1)[::-1]:
                    self.add_detector(
                        [i, i + len(self.code.xcheck_qubits) + len(self.code.zcheck_qubits)]
                    )

        # Finally, measure all data qubits in the chosen basis (no detectors here)
        orig = (self.idle_error, self.sqgate_error, self.tqgate_error, self.spam_error)

        # Ignore errors here
        self.set_error_rates_simple(0, 0, 0, 0)
        self.set_error_rates()

        self.add_measure(self.code.data_qubits, basis=self.basis)

        # Restore original error rates
        self.set_error_rates_simple(*orig)
        self.set_error_rates()

        # Add boundary detectors if requested (before observable)
        # These compare final data qubit measurements to last ancilla measurements
        if self._add_detectors and self._add_boundary_detectors:
            self._add_boundary_detectors_to_circuit()

        # Add a single logical observable from data measurements
        if self._add_detectors:
            num_data = len(self.code.data_qubits)
            data_qubits_list = list(self.code.data_qubits)

            def data_rec_offset_for_qubit(qid: int) -> int:
                pos = data_qubits_list.index(qid)
                return num_data - pos

            if self.basis.upper() == 'X':
                lx_positions = [
                    idx for idx, v in enumerate(self.code.lx.flatten().tolist()) if v == 1
                ]
                obs_inds = [data_rec_offset_for_qubit(qid) for qid in lx_positions]
                if len(obs_inds) > 0:
                    self.add_observable(0, obs_inds)
            elif self.basis.upper() == 'Z':
                lz_positions = [
                    idx for idx, v in enumerate(self.code.lz.flatten().tolist()) if v == 1
                ]
                obs_inds = [data_rec_offset_for_qubit(qid) for qid in lz_positions]
                if len(obs_inds) > 0:
                    self.add_observable(0, obs_inds)

        self.stim_circuit = stim.Circuit(self.circuit)

    def _add_boundary_detectors_to_circuit(self):
        """
        Add boundary detectors comparing final data qubit measurements to last ancilla measurements.
        
        For X-basis memory: adds detectors using X-stabilizer parity (hx matrix)
        For Z-basis memory: adds detectors using Z-stabilizer parity (hz matrix)
        
        Each boundary detector XORs:
        - The data qubits in a stabilizer's support (from final data measurement)
        - The corresponding ancilla's last measurement
        
        This "closes" the detection graph, enabling proper decoding with PAULI_CHANNEL_2 noise.
        """
        num_data = len(self.code.data_qubits)
        num_x = len(self.code.xcheck_qubits)
        num_z = len(self.code.zcheck_qubits)

        # Select appropriate stabilizer parity matrix based on measurement basis
        if self.basis.upper() == 'X':
            # X-basis memory: use X-stabilizers
            parity = self.code.hx
            # Measurement order in final round: MRX (X-ancillas), MR (Z-ancillas), M (data)
            # Data qubits are at rec[-1] to rec[-num_data]
            # Z-ancillas are at rec[-(num_data+1)] to rec[-(num_data+num_z)]
            # X-ancillas are at rec[-(num_data+num_z+1)] to rec[-(num_data+num_z+num_x)]
            ancilla_base_from_end = num_data + num_z
            num_ancillas = num_x
        else:
            # Z-basis memory: use Z-stabilizers
            parity = self.code.hz
            # For Z-basis, Z-ancillas come after X-ancillas
            ancilla_base_from_end = num_data
            num_ancillas = num_z

        # Add boundary detector for each stabilizer
        for stab_idx in range(parity.shape[0]):
            # Find data qubits in this stabilizer's support
            support = [i for i in range(num_data) if parity[stab_idx, i] == 1]
            if not support:
                continue

            # Data qubit rec indices: rec[-1] is the last data qubit measured
            # data_qubits are measured in order, so qubit i is at rec[-(num_data - i)]
            data_rec_indices = [num_data - i for i in support]

            # Ancilla rec index: ancillas are measured before data qubits
            # Stabilizer index maps to ancilla in reverse order due to how measurements are recorded
            ancilla_rec_index = ancilla_base_from_end + (num_ancillas - stab_idx)

            # Build detector string
            all_rec_indices = data_rec_indices + [ancilla_rec_index]
            self.add_detector(all_rec_indices)

    def _add_stabilizer_round(
        self, logical_measurement=False, state_prep=False, combine_reset_and_measure=False
    ):
        if logical_measurement:

            # --- save original error rates and noise_model ---
            orig = (self.idle_error, self.sqgate_error, self.tqgate_error, self.spam_error)
            orig_noise_model = self.noise_model

            # Final round (logical measurement): noiseless EXCEPT fake data-prep SPAM injection on data qubits.
            # We temporarily clear noise_model so it doesn't inject idle/CNOT/measurement noise.
            self.noise_model = None
            # Set all legacy scalar rates to 0 so no other noise is injected in this round.
            self.set_error_rates_simple(0, 0, 0, 0)
            self.set_error_rates()

        if not combine_reset_and_measure:

            self.add_reset(self.code.xcheck_qubits, basis='X')
            self.add_reset(self.code.zcheck_qubits, basis='Z')
        else:
            if state_prep:
                self.add_reset(self.code.xcheck_qubits, basis='X')
                self.add_reset(self.code.zcheck_qubits, basis='Z')
            else:
                self.add_single_error(self.code.xcheck_qubits, 'Z')
                self.add_single_error(self.code.zcheck_qubits, 'X')

        if not state_prep:
            if logical_measurement and orig_noise_model is not None:
                # Inject ONLY a "fake data-measurement SPAM" error on data qubits.
                #
                # Why measurement rates? In this circuit, the *actual* final data-qubit measurement is
                # forced noiseless (errors are set to 0 before measuring data qubits). The bookkeeping
                # convention represents noisy data readout via a time-reversed injection at the start
                # of the final perfect stabilizer round.
                #
                # Basis-labeled semantics:
                # - X-basis data measurement failure -> apply Z with prob p_meas_X
                # - Z-basis data measurement failure -> apply X with prob p_meas_Z
                if self.basis.upper() == 'X':
                    p_fake = float(orig_noise_model.p_meas_X)
                    if p_fake > 0:
                        c = self.margin + 'Z_ERROR(%.10f) ' % p_fake
                        for q in self.code.data_qubits:
                            c += '%d ' % q
                        c += '\n'
                        self.circuit += c
                else:  # 'Z'
                    p_fake = float(orig_noise_model.p_meas_Z)
                    if p_fake > 0:
                        c = self.margin + 'X_ERROR(%.10f) ' % p_fake
                        for q in self.code.data_qubits:
                            c += '%d ' % q
                        c += '\n'
                        self.circuit += c
            else:
                if self.noise_model is not None:
                    # NoiseModel semantics (drift/decomposition): IGNORE data-idle during ancilla prep/reset.
                    # We instead apply SPAM-idle during the ancilla *measurement* window (see below).
                    pass
                else:
                    self.add_idle(self.code.data_qubits, logical_measurement=logical_measurement)

        if logical_measurement:
            # Keep all scalar error rates at 0 for the rest of the logical-measurement round.
            # (noise_model is already cleared at the start of logical_measurement)
            self.set_error_rates_simple(0, 0, 0, 0)
            self.set_error_rates()

        # self.add_hadamard_layer(self.code.xcheck_qubits, add_tick=self._add_tick)

        # FIRST TICK
        if self._add_tick:
            self.add_tick()

        for i in range(4):
            layer = []
            for xcheck, zcheck in zip(self.code.xcheck_qubits_dict, self.code.zcheck_qubits_dict):
                x_data_qubit = self.code.xcheck_qubits_dict[xcheck]['plaquette']['qubit_id'][i]
                z_data_qubit = self.code.zcheck_qubits_dict[zcheck]['plaquette']['qubit_id'][i]
                if x_data_qubit != -1:
                    layer.extend([xcheck, x_data_qubit])
                if z_data_qubit != -1:
                    layer.extend([z_data_qubit, zcheck])
            self.add_cnot_layer(
                layer, add_tick=self._add_tick
            )  # This adds a tick after the CNOT layer
        # self.add_hadamard_layer(self.code.xcheck_qubits, before_measurement=True, add_tick=self._add_tick)

        # ADD TICK BEFORE MEASUREMENT. IMPORTANT!
        if self._add_tick:
            self.add_tick()

        self.add_measure(
            self.code.xcheck_qubits, basis='X', include_reset=combine_reset_and_measure
        )
        self.add_measure(
            self.code.zcheck_qubits, basis='Z', include_reset=combine_reset_and_measure
        )
        # After ancilla measurement, data qubits are idle.
        # - Legacy single-p: apply idle noise here (bulk idle)
        # - NoiseModel: apply SPAM-idle here (and ignore the prep/reset window idle above)
        if self.noise_model is None:
            self.add_idle(self.code.data_qubits, logical_measurement=logical_measurement)
        else:
            self.add_idle(
                self.code.data_qubits, logical_measurement=logical_measurement, idle_kind="spam"
            )

        if logical_measurement:
            # --- restore original error rates and noise_model before exiting ---
            self.noise_model = orig_noise_model
            self.set_error_rates_simple(*orig)
            self.set_error_rates()

    def set_error_rates(self):
        self.error_rates = {
            "errRateIdle1": self.idle_error,
            "errRatePrepX": self.spam_error,
            "errRatePrepZ": self.spam_error,
            "errRateMeasX": self.spam_error,
            "errRateMeasZ": self.spam_error,
            "errRateCNOT": self.tqgate_error,
            "errRateHad": self.sqgate_error,
            "errRateS": self.sqgate_error,  # This is not used in the code
        }


if __name__ == "__main__":

    d = 5
    p = 1e-1
    shots = 128
    circ = MemoryCircuit(
        distance=d,
        idle_error=p,
        sqgate_error=p,
        tqgate_error=p,
        spam_error=2. / 3. * p,
        n_rounds=d,
        basis='X'
    )

    meas = circ.stim_circuit.compile_sampler().sample(shots=shots)

    # drop final D*D data-qubit measurements and reshape to (shots, n_rounds, D^2-1)
    meas = meas[..., :-(d * d)].reshape(shots, d, d * d - 1)

    print(meas)
