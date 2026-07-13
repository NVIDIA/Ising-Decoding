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

import stim
import numpy as np
from qec.noise_model import NoiseModel
from qec.color_code import ColorCode

# --------------------------------------------------------------------------------------
# Legacy-equivalent triangular color-code scheduling (reimplemented locally).
#
# The legacy reference implementation (`qec.surface_code.memory_circuit.triangular_color_code_circuit`)
# is used for verification where available, but *circuit generation here does not depend on it*.
# --------------------------------------------------------------------------------------


def _legacy_triangular_color_code_cnot_layers(d: int) -> list[list[tuple[int, int]]]:
    """
    Reproduce the 8 CNOT layers (time steps 1..8) from the legacy triangular color-code circuit.

    Returns:
        layers: length-8 list; each element is a list of (control, target) pairs in *legacy qubit indices*:
          - data qubits: 0..num_data-1
          - ancillas: num_data..num_data+2*num_plaquettes-1, grouped as (a1,a2) = (num_data+2p, num_data+2p+1)
    """
    d = int(d)
    num_data = (3 * d * d + 1) // 4
    num_plaquettes = (3 * (d * d - 1)) // 8
    num_rows_data_qubits = (3 * d - 1) // 2

    layers: list[list[tuple[int, int]]] = []

    # tt=1..8 correspond to 8 layers
    for tt in range(1, 9):
        edges: list[tuple[int, int]] = []

        if tt == 1 or tt == 8:
            # CNOT between the two ancilla qubits: control |+> (a1), target |0> (a2)
            index = num_data
            for _ in range(num_plaquettes):
                a1 = index
                a2 = index + 1
                edges.append((a1, a2))
                index += 2

        elif tt == 2:
            # First sequence (data = control)
            data_qubit_index = d
            index_ancilla = num_data
            cols = d - 1
            for rr in range(num_rows_data_qubits - 1):
                if rr % 3 == 0:
                    for _ in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 1
                elif rr % 3 == 1:
                    for cc in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        if cc == cols - 1:
                            index_ancilla += 3
                        else:
                            index_ancilla += 1
                else:  # rr % 3 == 2
                    for _ in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 1

        elif tt == 3:
            # Second sequence (data = control)
            data_qubit_index = 0
            index_ancilla = num_data
            cols = d - 1
            for rr in range(num_rows_data_qubits - 1):
                if rr % 3 == 0:
                    for cc in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        index_ancilla += 1
                        if cc == cols - 1:
                            data_qubit_index += 3
                        else:
                            data_qubit_index += 1
                    cols -= 1
                elif rr % 3 == 1:
                    for cc in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        if cc == cols - 1:
                            index_ancilla += 3
                        else:
                            index_ancilla += 1
                else:  # rr % 3 == 2
                    for _ in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 1

        elif tt == 4:
            # Third sequence (data = control)
            data_qubit_index = 1
            index_ancilla = num_data + d - 1
            cols = d - 1
            for rr in range(num_rows_data_qubits - 2):
                if rr % 3 == 0:
                    for _ in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        index_ancilla += 1
                elif rr % 3 == 1:
                    for _ in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 2
                else:  # rr % 3 == 2
                    for cc in range(cols):
                        edges.append((data_qubit_index, index_ancilla))
                        index_ancilla += 1
                        if cc == cols - 1:
                            data_qubit_index += 3
                        else:
                            data_qubit_index += 1

        elif tt == 5:
            # First sequence (data = target) => ancilla is control
            data_qubit_index = d
            index_ancilla = num_data
            cols = d - 1
            for rr in range(num_rows_data_qubits - 1):
                if rr % 3 == 0:
                    for _ in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 1
                elif rr % 3 == 1:
                    for cc in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        if cc == cols - 1:
                            index_ancilla += 3
                        else:
                            index_ancilla += 1
                else:  # rr % 3 == 2
                    for _ in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 1

        elif tt == 6:
            # Second sequence (data = target) => ancilla is control
            data_qubit_index = 0
            index_ancilla = num_data
            cols = d - 1
            for rr in range(num_rows_data_qubits - 1):
                if rr % 3 == 0:
                    for cc in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        index_ancilla += 1
                        if cc == cols - 1:
                            data_qubit_index += 3
                        else:
                            data_qubit_index += 1
                    cols -= 1
                elif rr % 3 == 1:
                    for cc in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        if cc == cols - 1:
                            index_ancilla += 3
                        else:
                            index_ancilla += 1
                else:  # rr % 3 == 2
                    for _ in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 1

        else:  # tt == 7
            # Third sequence (data = target) => ancilla is control
            data_qubit_index = 1
            index_ancilla = num_data + d - 1
            cols = d - 1
            for rr in range(num_rows_data_qubits - 2):
                if rr % 3 == 0:
                    for _ in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        index_ancilla += 1
                elif rr % 3 == 1:
                    for _ in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        data_qubit_index += 1
                        index_ancilla += 1
                    cols -= 2
                else:  # rr % 3 == 2
                    for cc in range(cols):
                        edges.append((index_ancilla, data_qubit_index))
                        index_ancilla += 1
                        if cc == cols - 1:
                            data_qubit_index += 3
                        else:
                            data_qubit_index += 1

        layers.append(sorted(edges))

    return layers


def new_superdense_cnot_layers(cc: ColorCode) -> list[list[tuple[int, int]]]:
    """
    New (correctness-first) superdense schedule in OUR qubit numbering.

    Design goal:
      - Keep the *global 8-layer, clash-free parallel schedule* (the hard part)
      - Enforce per-plaquette 50/50 split for *all* plaquettes (including weight-4 boundaries)

    Approach:
      - Start from the legacy 8-layer edge-coloring in *legacy qubit indices* (implemented locally in
        `_legacy_triangular_color_code_cnot_layers(d)`), which is known to be parallel (no clashes).
      - Detect weight-4 plaquettes that have a 3/1 split between their two ancillas.
      - For each such plaquette, move exactly one data qubit from the heavy ancilla to the light ancilla by
        swapping BOTH its forward-half and reverse-half CNOT endpoints (preserving the layer/time index),
        only when doing so does not introduce a clash.
      - Map the corrected legacy layers into OUR qubit numbering via the same bijection used by
        `_legacy_superdense_cx_layers` (data ordering + plaquette matching by data sets).

    Raises:
      AssertionError if correction is not possible without clashes (should not happen for odd d>=3).
    """

    d = int(cc.distance)
    num_data = int(cc.num_data)
    num_plaquettes = int(cc.num_plaquettes)

    # Legacy layers in legacy qubit IDs: 8 layers, where layers 1..3 are "forward", 4..6 are "reverse".
    legacy_layers: list[list[tuple[int, int]]] = _legacy_triangular_color_code_cnot_layers(d)

    # Build fast lookup: for each layer index, map (u,v) unordered to the directed (c,t) edge.
    layer_edge_map = []
    for edges in legacy_layers:
        m = {}
        for c, t in edges:
            m[tuple(sorted((c, t)))] = (c, t)
        layer_edge_map.append(m)

    def layer_used_qubits(li: int) -> set[int]:
        used = set()
        for c, t in legacy_layers[li]:
            used.add(c)
            used.add(t)
        return used

    # Helper: find the layer indices (forward and reverse) where data d_q interacts with ancilla a_q.
    def find_layers_for_pair(data_q: int, anc_q: int) -> tuple[int, int]:
        forward = -1
        reverse = -1
        key = tuple(sorted((data_q, anc_q)))
        for li in (1, 2, 3):
            if key in layer_edge_map[li]:
                forward = li
                break
        for li in (4, 5, 6):
            if key in layer_edge_map[li]:
                reverse = li
                break
        if forward < 0 or reverse < 0:
            raise AssertionError(
                f"Expected both forward and reverse edges for (data={data_q}, anc={anc_q})"
            )
        return forward, reverse

    # For each plaquette, compute its data split in legacy (by scanning edges touching its a1/a2).
    for p in range(num_plaquettes):
        a1 = num_data + 2 * p
        a2 = num_data + 2 * p + 1

        a1_conn = set()
        a2_conn = set()
        for li in range(8):
            for c, t in legacy_layers[li]:
                if c == a1 and t < num_data:
                    a1_conn.add(t)
                elif t == a1 and c < num_data:
                    a1_conn.add(c)
                if c == a2 and t < num_data:
                    a2_conn.add(t)
                elif t == a2 and c < num_data:
                    a2_conn.add(c)

        data_union = a1_conn | a2_conn
        w = len(data_union)
        if w == 6:
            # already 3/3 in legacy
            continue
        if w != 4:
            raise AssertionError(
                f"Unexpected plaquette weight={w} in legacy schedule at p={p}, d={d}"
            )

        if len(a1_conn) == 2 and len(a2_conn) == 2:
            continue

        # Determine heavy/light ancillas.
        if len(a1_conn) > len(a2_conn):
            heavy_anc, light_anc = a1, a2
            heavy_set = set(a1_conn)
        else:
            heavy_anc, light_anc = a2, a1
            heavy_set = set(a2_conn)

        # Try moving one data qubit from heavy_anc to light_anc.
        moved = False
        for dq in sorted(heavy_set):
            # dq must currently connect to heavy_anc in both halves.
            f_li, r_li = find_layers_for_pair(dq, heavy_anc)

            # Light ancilla must be unused in those layers (to avoid ancilla clash).
            used_f = layer_used_qubits(f_li)
            used_r = layer_used_qubits(r_li)
            if light_anc in used_f or light_anc in used_r:
                continue

            # Also ensure dq isn't already interacting with something else in those layers (it is, with heavy_anc).
            # We'll replace that edge, so dq stays used once.

            def replace_edge(li: int, old_anc: int, new_anc: int, dq_: int) -> None:
                key = tuple(sorted((dq_, old_anc)))
                c, t = layer_edge_map[li][key]
                # Replace endpoint old_anc -> new_anc, preserving direction.
                if c == old_anc and t == dq_:
                    new_edge = (new_anc, dq_)
                elif c == dq_ and t == old_anc:
                    new_edge = (dq_, new_anc)
                else:
                    raise AssertionError("Unexpected directed edge orientation")
                # mutate legacy_layers + maps
                legacy_layers[li].remove((c, t))
                layer_edge_map[li].pop(key)
                layer_edge_map[li][tuple(sorted(new_edge))] = new_edge
                legacy_layers[li].append(new_edge)

            replace_edge(f_li, heavy_anc, light_anc, dq)
            replace_edge(r_li, heavy_anc, light_anc, dq)
            moved = True
            break

        if not moved:
            raise AssertionError(
                f"Could not rebalance weight-4 plaquette p={p} at d={d} without clashes"
            )

    # Final sanity: ensure each legacy layer is still disjoint.
    for li, edges in enumerate(legacy_layers):
        used = set()
        for c, t in edges:
            if c in used or t in used:
                raise AssertionError(f"Post-fix clash in legacy layer {li} at d={d}")
            used.add(c)
            used.add(t)

    # --- Map legacy qubits -> our qubits (same method as _legacy_superdense_cx_layers) ---
    ordered_our_data = sorted(
        range(num_data), key=lambda q: (cc.qubit_to_coord[q][0], cc.qubit_to_coord[q][1])
    )
    our_data_to_legacy = {q: i for i, q in enumerate(ordered_our_data)}
    legacy_data_to_our = {i: q for q, i in our_data_to_legacy.items()}

    # Match legacy plaquettes by data sets (in legacy data IDs)
    legacy_plaq_data_set_to_p = {}
    for p in range(num_plaquettes):
        a1 = num_data + 2 * p
        a2 = num_data + 2 * p + 1
        ds = set()
        for edges in legacy_layers:
            for c, t in edges:
                if c < num_data and t in (a1, a2):
                    ds.add(c)
                if c in (a1, a2) and t < num_data:
                    ds.add(t)
        legacy_plaq_data_set_to_p[tuple(sorted(ds))] = p

    legacy_to_our = dict(legacy_data_to_our)
    for plaq in cc.plaquettes:
        key = tuple(sorted(our_data_to_legacy[q] for q in plaq["data_qubits"]))
        p = legacy_plaq_data_set_to_p.get(key)
        if p is None:
            raise AssertionError(f"Could not match plaquette data set to legacy after fix: {key}")
        legacy_to_our[num_data + 2 * p] = int(plaq["x_ancilla"])
        legacy_to_our[num_data + 2 * p + 1] = int(plaq["z_ancilla"])

    # Convert layers to OUR IDs
    out_layers: list[list[tuple[int, int]]] = []
    for li in range(8):
        mapped = [(legacy_to_our[c], legacy_to_our[t]) for c, t in legacy_layers[li]]
        # deterministic order
        out_layers.append(sorted((int(c), int(t)) for c, t in mapped))

    return out_layers


# author of the following class is Mingyu Kang: https://github.com/mkangquantum/quits/blob/main/src/quits/circuit.py
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
        self.noise_model = None  # Optional 25-parameter noise model

    def set_all_qubits(self, all_qubits):
        self.all_qubits = all_qubits

    def set_noise_model(self, noise_model: 'NoiseModel') -> None:
        """
        Set the 25-parameter noise model for circuit generation.
        
        When a NoiseModel is set, the circuit will use:
        - X_ERROR/Z_ERROR for prep/meas with explicit probabilities
        - PAULI_CHANNEL_1 for idle errors (instead of DEPOLARIZE1)
        - PAULI_CHANNEL_2 for CNOT errors (instead of DEPOLARIZE2)
        
        Args:
            noise_model: NoiseModel instance with 25 parameters
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

    def set_error_rates(self):
        """
        Populate a legacy error-rate dictionary used by some downstream utilities/tests.

        Note: The circuit generation methods primarily use (idle_error, sqgate_error, tqgate_error, spam_error)
        directly (or an explicit NoiseModel). This mapping mirrors the surface-code implementation for
        compatibility.
        """
        self.error_rates = {
            "errRateIdle1": self.idle_error,
            "errRateIdle2": self.idle_error,
            "errRateIdle7": self.idle_error,
            "errRateIdle8": self.idle_error,
            "errRatePrepX": self.spam_error,
            "errRatePrepZ": self.spam_error,
            "errRateMeasX": self.spam_error,
            "errRateMeasZ": self.spam_error,
            "errRateCNOT": self.tqgate_error,
            "errRateHad": self.sqgate_error,
            "errRateS": self.sqgate_error,  # not used
        }
        return self.error_rates

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

        # Apply preparation errors
        if self.noise_model is not None:
            # Use 22-parameter noise model: explicit X_ERROR and Z_ERROR
            # For Z-basis prep (|0>): X error flips to |1>
            # For X-basis prep (|+>): Z error flips to |->
            if basis == 'Z' and self.noise_model.p_prep_X > 0:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.noise_model.p_prep_X
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
            elif basis == 'X' and self.noise_model.p_prep_Z > 0:
                c += self.margin
                c += 'Z_ERROR(%.10f) ' % self.noise_model.p_prep_Z
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
            if error_type == 'X':
                error_prob = self.noise_model.p_prep_X
            elif error_type == 'Z':
                error_prob = self.noise_model.p_prep_Z
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
            if self.noise_model.p_meas_X > 0:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.noise_model.p_meas_X
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
                if self.noise_model.p_prep_X > 0:
                    c += self.margin
                    c += 'X_ERROR(%.10f) ' % self.noise_model.p_prep_X
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
            # Use 22-parameter noise model: explicit X_ERROR or Z_ERROR
            if basis == 'Z' and self.noise_model.p_meas_X > 0:
                c += self.margin
                c += 'X_ERROR(%.10f) ' % self.noise_model.p_meas_X
                for q in qubits:
                    c += '%d ' % q
                c += '\n'
            elif basis == 'X' and self.noise_model.p_meas_Z > 0:
                c += self.margin
                c += 'Z_ERROR(%.10f) ' % self.noise_model.p_meas_Z
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

    def add_detector(self, inds, coords=None):
        """
        Add a detector comparing measurement records.
        
        Args:
            inds: List of measurement record indices (positive integers, rec[-ind])
            coords: Optional tuple of coordinates (x, y, t) or (x, y, t, chromobius_annotation)
                    For chromobius compatibility, the 4th coordinate encodes basis and color:
                    0=RedX, 1=GreenX, 2=BlueX, 3=RedZ, 4=GreenZ, 5=BlueZ
        """
        c = self.margin + 'DETECTOR'
        if coords is not None:
            coord_str = ', '.join(str(x) for x in coords)
            c += f'({coord_str}) '
        else:
            c += ' '
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

    def add_qubit_coordinates_from_layout(self, layout: dict[int, tuple[int, int]]):
        """
        Add Stim `QUBIT_COORDS(r, c) q` annotations from a provided qubit->(r,c) mapping.

        This is useful for embedding the circuit on a 2D nearest-neighbor grid and for downstream
        visualization/compilation tools that consume Stim coordinates.
        """
        # Deterministic order
        for q in sorted(layout.keys()):
            r, c = layout[q]
            self.circuit += f"QUBIT_COORDS({r}, {c}) {q}\n"
        return self.circuit


class MemoryCircuit(Circuit):
    """
    Memory circuit for color code quantum error correction.
    
    This class generates a complete quantum circuit for implementing a color code
    memory experiment, including state preparation, stabilizer measurements, and
    logical measurements. Includes circuit level noise modeling.
    We follow https://arxiv.org/pdf/2312.08813 for the circuit structure and implement the superdense color code circuit.
    We use an upper triangular lattice structure for the color code.
    
    Args:
        distance (int): The distance of the color code (must be odd).
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
    
    Attributes:
        circuit (str): The complete Stim circuit as a string.
        distance (int): The distance of the color code.
        n_rounds (int): Number of stabilizer measurement rounds.
        basis (str): Logical basis for the memory experiment.
        code (ColorCode): The underlying color code object.
        
    Example:
        >>> # Create a distance-3 superdense color code memory circuit
        >>> circ = MemoryCircuit(
        ...     distance=3,
        ...     idle_error=1e-3,
        ...     sqgate_error=1e-3, 
        ...     tqgate_error=1e-3,
        ...     spam_error=1e-3,
        ...     n_rounds=3,
        ...     basis='X',
        ... )
        >>> print(circ.circuit)  # Print the generated Stim circuit
    """

    # Superdense schedule template for *weight-6* plaquettes.
    #
    # Each entry is a "macro-step". For each plaquette, we materialize all pairs in that macro-step
    # using `ColorCode.superdense_plaquette(plaq_idx)` which provides a1/a2/q1..q6.
    #
    # Each entry is a "macro-step" listing (control_key, target_key) pairs.
    #
    # This sequence is the one specified by the user (8 macro-steps):
    # 0. a1 -> a2
    # 1. q1 -> a1, q6 -> a2
    # 2. q2 -> a1, q5 -> a2
    # 3. q3 -> a1, q4 -> a2
    # 4. a1 -> q1, a2 -> q6
    # 5. a1 -> q2, a2 -> q5
    # 6. a1 -> q3, a2 -> q4
    # 7. a1 -> a2
    SUPERDENSE_W6_STEPS = [
        [("a1", "a2")],
        [("q1", "a1"), ("q6", "a2")],
        [("q2", "a1"), ("q5", "a2")],
        [("q3", "a1"), ("q4", "a2")],
        [("a1", "q1"), ("a2", "q6")],
        [("a1", "q2"), ("a2", "q5")],
        [("a1", "q3"), ("a2", "q4")],
        [("a1", "a2")],
    ]

    @staticmethod
    def _pack_edges_greedily(edges):
        """
        Greedily pack a list of (control, target) edges into disjoint CX layers.

        Returns:
            List[List[int]] where each inner list is a flat [c1,t1,c2,t2,...] suitable for add_cnot_layer.
        """
        remaining = list(edges)
        layers = []
        while remaining:
            used = set()
            layer_pairs = []
            next_remaining = []
            for c, t in remaining:
                if c in used or t in used:
                    next_remaining.append((c, t))
                    continue
                used.add(c)
                used.add(t)
                layer_pairs.append((c, t))
            flat = []
            for c, t in layer_pairs:
                flat.extend([int(c), int(t)])
            layers.append(flat)
            remaining = next_remaining
        return layers

    def _legacy_superdense_cx_layers(self):
        """
        Return the legacy superdense CNOT layers (8 layers) mapped into *this* ColorCode qubit numbering.

        This uses a local reimplementation of the legacy schedule (`_legacy_triangular_color_code_cnot_layers`)
        and builds an explicit bijection:
          - data qubits: legacy inverted-triangle row-major order ↔ our upright numbering
          - ancillas: match plaquettes by their (mapped) data-qubit sets, then map (a1,a2) ↔ (x_ancilla,z_ancilla)

        Returns:
            List[List[int]] where each inner list is a flat [c1,t1,c2,t2,...] suitable for add_cnot_layer.
        """
        d = int(self.distance)
        num_data = int(self.code.num_data)
        num_plaquettes = int(self.code.num_plaquettes)

        # --- Map OUR data qubits -> LEGACY data qubits ---
        # Legacy uses an inverted triangle; empirically this matches ordering OUR data bottom-to-top, left-to-right.
        ordered_our_data = sorted(
            range(num_data),
            key=lambda q: (self.code.qubit_to_coord[q][0], self.code.qubit_to_coord[q][1]),
        )
        our_data_to_legacy = {q: i for i, q in enumerate(ordered_our_data)}
        legacy_data_to_our = {i: q for q, i in our_data_to_legacy.items()}

        # --- Reconstruct legacy plaquette data sets (in legacy data IDs) ---
        legacy_edges_by_tt = _legacy_triangular_color_code_cnot_layers(d)

        legacy_plaq_data_set_to_p = {}
        for p in range(num_plaquettes):
            a1_legacy = num_data + 2 * p
            a2_legacy = num_data + 2 * p + 1
            ds = set()
            for edges in legacy_edges_by_tt:
                for c, t in edges:
                    if c < num_data and t in (a1_legacy, a2_legacy):
                        ds.add(c)
                    if c in (a1_legacy, a2_legacy) and t < num_data:
                        ds.add(t)
            key = tuple(sorted(ds))
            legacy_plaq_data_set_to_p[key] = p

        # --- Map legacy ancillas -> our ancillas by matching plaquettes via data sets ---
        legacy_qubit_to_our = dict(legacy_data_to_our)
        for plaq in self.code.plaquettes:
            ours_set_legacy_ids = tuple(sorted(our_data_to_legacy[q] for q in plaq["data_qubits"]))
            p = legacy_plaq_data_set_to_p.get(ours_set_legacy_ids)
            if p is None:
                raise AssertionError(
                    f"Could not match our plaquette data set to a legacy plaquette: {ours_set_legacy_ids}"
                )
            a1_legacy = num_data + 2 * p
            a2_legacy = num_data + 2 * p + 1
            legacy_qubit_to_our[a1_legacy] = int(plaq["x_ancilla"])
            legacy_qubit_to_our[a2_legacy] = int(plaq["z_ancilla"])

        # --- Translate each legacy CNOT layer into our qubit IDs ---
        mapped_layers = []
        for edges in legacy_edges_by_tt:
            flat = []
            for c, t in sorted(edges):
                flat.extend([int(legacy_qubit_to_our[c]), int(legacy_qubit_to_our[t])])
            mapped_layers.append(flat)
        return mapped_layers

    def _z_connected_data_by_z_ancilla(self) -> dict[int, list[int]]:
        """
        Determine, for the current schedule, which data qubits are connected (via any CX in the 8-layer schedule)
        to each plaquette's Z-ancilla.

        Returns:
            Dict[z_ancilla_id -> sorted list of data qubit ids]
        """
        # Build the 8 CX layers in OUR qubit IDs (same representation used to emit the circuit).
        sched = getattr(self, "schedule", "nearest-neighbor")
        if sched == "legacy":
            sched = "nearest-neighbor"
        if sched == "new":
            sched = "long-range"

        if sched == "nearest-neighbor":
            flat_layers = self._legacy_superdense_cx_layers()
            layers = [
                [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)] for flat in flat_layers
            ]
        elif sched == "long-range":
            layers = new_superdense_cnot_layers(self.code)
        else:
            raise ValueError(f"Unsupported schedule for feedforward: {sched}")

        z_to_data: dict[int, set[int]] = {}
        # Use plaquette supports to avoid accidentally picking up edges to other plaquettes' data.
        for plaq in self.code.plaquettes:
            z = int(plaq["z_ancilla"])
            z_to_data.setdefault(z, set())

        for plaq in self.code.plaquettes:
            data_set = set(int(q) for q in plaq["data_qubits"])
            z = int(plaq["z_ancilla"])
            touched = z_to_data[z]
            for layer in layers:
                for c, t in layer:
                    if c == z and t in data_set:
                        touched.add(int(t))
                    elif t == z and c in data_set:
                        touched.add(int(c))

        return {z: sorted(ds) for z, ds in z_to_data.items()}

    def _x_connected_data_by_x_ancilla(self) -> dict[int, list[int]]:
        """
        Determine, for the current schedule, which data qubits are connected (via any CX in the 8-layer schedule)
        to each plaquette's X-ancilla.

        Returns:
            Dict[x_ancilla_id -> sorted list of data qubit ids]
        """
        sched = getattr(self, "schedule", "nearest-neighbor")
        if sched == "legacy":
            sched = "nearest-neighbor"
        if sched == "new":
            sched = "long-range"

        if sched == "nearest-neighbor":
            flat_layers = self._legacy_superdense_cx_layers()
            layers = [
                [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)] for flat in flat_layers
            ]
        elif sched == "long-range":
            layers = new_superdense_cnot_layers(self.code)
        else:
            raise ValueError(f"Unsupported schedule for feedforward/noise bookkeeping: {sched}")

        x_to_data: dict[int, set[int]] = {}
        for plaq in self.code.plaquettes:
            x = int(plaq["x_ancilla"])
            x_to_data.setdefault(x, set())

        for plaq in self.code.plaquettes:
            data_set = set(int(q) for q in plaq["data_qubits"])
            x = int(plaq["x_ancilla"])
            touched = x_to_data[x]
            for layer in layers:
                for c, t in layer:
                    if c == x and t in data_set:
                        touched.add(int(t))
                    elif t == x and c in data_set:
                        touched.add(int(c))

        return {x: sorted(ds) for x, ds in x_to_data.items()}

    def _add_z_feedforward_x_corrections(self) -> None:
        """
        After measuring Z-ancillas, apply X^b to each data qubit that interacted with that plaquette's Z-ancilla,
        where b is that Z-ancilla's measurement result.

        Stim encoding:
          CX rec[-k] q   (apply X on q iff referenced measurement result is 1)
        """
        z_qubits = [int(q) for q in self.code.zcheck_qubits]
        if not z_qubits:
            return

        z_to_data = self._z_connected_data_by_z_ancilla()

        # Measurement order is exactly the order we pass to add_measure for zcheck_qubits.
        # After measuring all Z ancillas, rec[-1] corresponds to the *last* Z ancilla in this list.
        # In general, z_qubits[i] corresponds to rec[-(len(z_qubits)-i)].
        n = len(z_qubits)
        for i, z in enumerate(z_qubits):
            k = n - i  # positive integer for rec[-k]
            targets = z_to_data.get(z, [])
            for dq in targets:
                self.circuit += f"CX rec[-{k}] {dq}\n"

    def _add_post_round_data_idle_noise(self, *, logical_measurement: bool) -> None:
        """
        Apply a final "data idling" noise step after syndrome extraction + feedforward.

        All data qubits are idle during the measurement window (waiting for ancilla measurement
        outcomes), so they all receive idle noise. This matches the legacy simulator behavior where
        all non-measured qubits at measurement time receive idle errors.

        Note: The feedforward correction (CX rec[-k] q) is a Pauli frame update, not a physical
        gate that would "occupy" the data qubit. Data qubits remain idle throughout.
        """
        idle_targets = sorted(self.code.data_qubits)

        if idle_targets:
            if self.noise_model is None:
                self.add_idle(idle_targets, logical_measurement=logical_measurement)
            else:
                self.add_idle(
                    idle_targets, logical_measurement=logical_measurement, idle_kind="spam"
                )

    def _add_superdense_layers(self):
        """
        Add the superdense CNOT schedule across *all plaquettes*, using global disjoint-layer packing.

        Weight-6 plaquettes participate in all 8 macro-steps.
        Weight-4 plaquettes are embedded into the q1..q6 frame by `ColorCode.superdense_plaquette` with q3/q4=-1,
        so they naturally skip macro-steps that reference missing qubits.
        """
        # Schedule names:
        # - "nearest-neighbor": legacy-equivalent schedule that is NN-embeddable on our circuit-only physical grid
        # - "legacy": backward-compatible alias for "nearest-neighbor"
        sched = getattr(self, "schedule", "nearest-neighbor")
        if sched == "legacy":
            sched = "nearest-neighbor"

        # - "long-range": allows non-NN couplers (e.g. neutral atoms / trapped ions); enforces 50/50 split per plaquette
        # - "new": backward-compatible alias for "long-range"
        if sched == "new":
            sched = "long-range"

        if sched == "nearest-neighbor":
            flat_layers = self._legacy_superdense_cx_layers()
        elif sched == "long-range":
            # new_superdense_cnot_layers returns list of (c,t) pairs; flatten for add_cnot_layer
            flat_layers = [
                [q
                 for (c, t) in layer
                 for q in (c, t)]
                for layer in new_superdense_cnot_layers(self.code)
            ]
        else:
            raise ValueError(
                "schedule must be 'nearest-neighbor' (or alias 'legacy') or 'long-range' (or alias 'new')"
            )

        for layer in flat_layers:
            if layer:
                self.add_cnot_layer(layer, add_tick=self._add_tick)

    def _add_stabilizer_round(
        self, logical_measurement=False, state_prep=False, combine_reset_and_measure=False
    ):
        """
        Add one stabilizer-measurement round for the color code.

        Implements the superdense schedule with feedforward correction:
        1. Z ancillas measured first
        2. Feedforward X^b applied to data qubits (where b is Z measurement)
        3. X ancillas measured
        
        The feedforward is absorbed by stim.Circuit.with_inlined_feedback() at circuit construction end.
        """
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

        # Reset ancillas (or model reset-via-measurement depending on combine_reset_and_measure)
        if not combine_reset_and_measure:
            self.add_reset(self.code.xcheck_qubits, basis='X')  # a1 family
            self.add_reset(self.code.zcheck_qubits, basis='Z')  # a2 family
        else:
            if state_prep:
                self.add_reset(self.code.xcheck_qubits, basis='X')
                self.add_reset(self.code.zcheck_qubits, basis='Z')
            else:
                # Inject a prep-like error instead of explicit reset when using MR* for subsequent rounds.
                self.add_single_error(self.code.xcheck_qubits, 'Z')
                self.add_single_error(self.code.zcheck_qubits, 'X')

        # Data idle during ancilla reset/prep window (legacy vs NoiseModel semantics handled like surface code)
        if not state_prep:
            if logical_measurement and orig_noise_model is not None:
                # Inject ONLY a "fake data-measurement SPAM" error on data qubits.
                if self.basis.upper() == 'X':
                    p_fake = float(orig_noise_model.p_meas_Z)
                    if p_fake > 0:
                        c = self.margin + 'Z_ERROR(%.10f) ' % p_fake
                        for q in self.code.data_qubits:
                            c += '%d ' % q
                        c += '\n'
                        self.circuit += c
                else:  # 'Z'
                    p_fake = float(orig_noise_model.p_meas_X)
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
                elif self._gidney_style_noise:
                    # Gidney-style noise: skip idle noise on data qubits during ancilla prep.
                    # This matches Gidney's superdense circuit which has no extra idle on data qubits.
                    pass
                elif logical_measurement:
                    # Single-p mode, final round: inject data measurement SPAM error.
                    # Use orig spam_error (saved before zeroing) to mimic measurement error on data qubits.
                    # X-basis measurement -> Z errors flip the outcome; Z-basis -> X errors flip the outcome.
                    p_fake = orig[3]  # orig = (idle_error, sqgate_error, tqgate_error, spam_error)
                    if p_fake > 0:
                        if self.basis.upper() == 'X':
                            c = self.margin + 'Z_ERROR(%.10f) ' % p_fake
                        else:
                            c = self.margin + 'X_ERROR(%.10f) ' % p_fake
                        for q in self.code.data_qubits:
                            c += '%d ' % q
                        c += '\n'
                        self.circuit += c
                else:
                    self.add_idle(self.code.data_qubits, logical_measurement=logical_measurement)

        if logical_measurement:
            # Keep all scalar error rates at 0 for the rest of the logical-measurement round.
            self.set_error_rates_simple(0, 0, 0, 0)
            self.set_error_rates()

        # FIRST TICK
        if self._add_tick:
            self.add_tick()

        # --- Superdense entangling schedule (weight-6 only for now) ---
        self._add_superdense_layers()

        # NOTE: Last CNOT layer already adds a TICK via add_cnot_layer(add_tick=True).
        # This provides the needed separation between CX and measurement layers.
        # DO NOT add another TICK here - it creates a double-TICK that adds extra idle noise.

        # Measure ancillas with feedforward:
        # 1. Measure Z ancillas first
        # 2. Apply feedforward X^b to connected data qubits
        # 3. Measure X ancillas
        # This order + feedforward makes both X and Z detectors deterministic after with_inlined_feedback()

        self.add_measure(
            self.code.zcheck_qubits, basis='Z', include_reset=combine_reset_and_measure
        )

        # Feedforward: CX from Z measurement to connected data qubits
        # Z measurements are at rec[-num_plaq], ..., rec[-1] (in order of zcheck_qubits)
        num_plaq = self.code.num_plaquettes
        for i, z_anc in enumerate(self.code.zcheck_qubits):
            rec_idx = num_plaq - i  # rec[-num_plaq] for first, rec[-1] for last
            # Get data qubits connected to this Z ancilla
            data_qubits = self._z_anc_to_data.get(int(z_anc), [])
            for data_q in sorted(data_qubits):
                self.circuit += f'{self.margin}CX rec[-{rec_idx}] {data_q}\n'

        self.add_measure(
            self.code.xcheck_qubits, basis='X', include_reset=combine_reset_and_measure
        )

        # NOTE: Feedforward is already applied after Z measurement (before X measurement) at lines above.
        # The _add_z_feedforward_x_corrections() call was removed because it was incorrectly placed
        # after X measurement, where rec indices no longer point to Z measurements.

        # After ancilla measurement + feedforward, apply the requested post-round data-idle noise step.
        # Skip if using Gidney-style noise (matches Gidney's superdense circuit structure).
        if not self._gidney_style_noise:
            self._add_post_round_data_idle_noise(logical_measurement=logical_measurement)

        if logical_measurement:
            # --- restore original error rates and noise_model before exiting ---
            self.noise_model = orig_noise_model
            self.set_error_rates_simple(*orig)
            self.set_error_rates()

    def _add_boundary_detectors_to_circuit(self):
        """
        Add boundary detectors comparing final data qubit measurements to last ancilla measurements.
        
        For X-basis memory: adds detectors using X-stabilizer parity (comparing to X-ancilla measurements)
        For Z-basis memory: adds detectors using Z-stabilizer parity (comparing to Z-ancilla measurements)
        
        Each boundary detector XORs:
        - The data qubits in a stabilizer's support (from final data measurement)
        - The corresponding ancilla's last measurement
        
        This "closes" the detection graph, enabling proper decoding with PAULI_CHANNEL_2 noise.
        
        Measurement order in color code (most recent first):
        - Data qubits: rec[-1] to rec[-num_data]
        - X ancillas: rec[-(num_data+1)] to rec[-(num_data+num_stabs)]
        - Z ancillas: rec[-(num_data+num_stabs+1)] to rec[-(num_data+2*num_stabs)]
        """
        num_data = self.code.num_data
        num_stabs = self.code.num_plaquettes

        # Stabilizer parity matrix (same for X and Z in color code)
        parity = self.code.hx  # hx == hz for CSS color code

        # Determine which ancillas to use based on measurement basis
        if self.basis.upper() == 'X':
            # X-basis memory: use X-ancilla measurements
            # X-ancillas are at rec[-(num_data+1)] to rec[-(num_data+num_stabs)]
            ancilla_base_from_end = num_data
            coord_key = 'x'  # Use X chromobius coordinate
        else:
            # Z-basis memory: use Z-ancilla measurements
            # Z-ancillas are at rec[-(num_data+num_stabs+1)] to rec[-(num_data+2*num_stabs)]
            ancilla_base_from_end = num_data + num_stabs
            coord_key = 'z'  # Use Z chromobius coordinate

        # Boundary detectors are at time t=1 (after final round at t=0)
        t = 1

        # Add boundary detector for each stabilizer
        for stab_idx in range(num_stabs):
            # Find data qubits in this stabilizer's support
            support = [i for i in range(num_data) if parity[stab_idx, i] == 1]
            if not support:
                continue

            # Data qubit rec indices: data qubits are measured in order
            # rec[-1] is last data qubit, rec[-num_data] is first
            # data_qubits list order matches measurement order
            data_qubits_list = list(self.code.data_qubits)
            data_rec_indices = [num_data - data_qubits_list.index(qid) for qid in support]

            # Ancilla rec index
            # Ancillas are measured in plaquette order (0, 1, 2, ...)
            # Plaquette stab_idx is at rec[-(ancilla_base_from_end + (num_stabs - stab_idx))]
            ancilla_rec_index = ancilla_base_from_end + (num_stabs - stab_idx)

            # Get chromobius coordinates for this plaquette
            coords = self._plaq_chromobius_coords[stab_idx]
            det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords[coord_key])

            # Build detector with coordinates
            all_rec_indices = data_rec_indices + [ancilla_rec_index]
            self.add_detector(all_rec_indices, coords=det_coords)

    def __init__(self, distance, idle_error, sqgate_error, tqgate_error, spam_error, n_rounds,\
                          basis='X', get_all_detectors=True, noisy_init=True, noisy_meas=False,
                          add_tick=True, add_detectors=True, noise_model=None, schedule: str = "nearest-neighbor",\
                          add_physical_coords: bool = False, flip_triangle: bool = False,
                          gidney_style_noise: bool = False, add_boundary_detectors: bool = False):
        """
        Initialize a MemoryCircuit for color code quantum error correction.
        
        Args:
            distance: Code distance
            idle_error: Idle error rate (used if noise_model is None)
            sqgate_error: Single-qubit gate error rate (used if noise_model is None)
            tqgate_error: Two-qubit gate error rate (used if noise_model is None)
            spam_error: SPAM error rate (used if noise_model is None)
            n_rounds: Number of stabilizer measurement rounds
            basis: Logical basis ('X' or 'Z')
            get_all_detectors: Whether to include all detector types
            add_boundary_detectors: Whether to add boundary detectors comparing final data
                measurements to last ancilla measurements. Required for proper decoding
                with PAULI_CHANNEL_2 (25-parameter) noise model.
            noisy_init: Whether to include noise in initialization
            noisy_meas: Whether to include noise in measurements
            add_tick: Whether to add timing ticks
            add_detectors: Whether to add detector annotations
            noise_model: Optional NoiseModel for 25-parameter noise model.
                         If provided, uses explicit per-type probabilities instead
                         of deriving from single error rates.
            gidney_style_noise: If True, use Gidney's noise model structure:
                         - No extra idle noise on data qubits during ancilla prep
                         - No post-round idle noise on data qubits
                         This matches the noise structure in Gidney's superdense color code paper.
        """
        self.circuit = ''
        self.margin = ''
        self.distance = distance
        self.n_rounds = n_rounds  # n_rounds is defined as the number of stabilizer rounds: counting state prep and logical measurement rounds
        self._add_tick = add_tick
        self._add_detectors = add_detectors
        self._gidney_style_noise = gidney_style_noise
        self.add_boundary_detectors = add_boundary_detectors
        self.basis = basis
        # Public schedule names:
        # - "nearest-neighbor" (alias: "legacy")
        # - "long-range" (alias: "new")
        if schedule == "legacy":
            schedule = "nearest-neighbor"
        if schedule == "new":
            schedule = "long-range"
        self.schedule = schedule
        get_Z_detectors = True if basis == 'Z' or get_all_detectors else False
        get_X_detectors = True if basis == 'X' or get_all_detectors else False

        self.code = ColorCode(distance)

        super().__init__(self.code.all_qubits)

        # Optional: attach a 2D NN physical embedding as QUBIT_COORDS in the Stim text.
        # This uses the circuit-only physical grid (independent from ColorCode's logical/CNN embeddings).
        if add_physical_coords:
            layout = self.code.get_circuit_physical_layout(
                id_order="rtl", flip_rows=bool(flip_triangle)
            )
            self.add_qubit_coordinates_from_layout(layout)

        # Set error rates: use noise_model if provided, otherwise use simple rates
        if noise_model is not None:
            self.set_noise_model(noise_model)
        else:
            self.set_error_rates_simple(idle_error, sqgate_error, tqgate_error, spam_error)
        self.set_error_rates()

        # Compute Z ancilla to data qubit connectivity (for feedforward)
        # This maps each Z ancilla to the data qubits it interacts with in the CNOT schedule
        # We use the legacy CNOT layers and map to our qubit numbering
        self._z_anc_to_data = {int(z): set() for z in self.code.zcheck_qubits}

        # Get the CNOT layers in OUR qubit numbering (via the mapping in _legacy_superdense_cx_layers)
        # For now, compute directly from legacy layers with mapping
        legacy_layers = _legacy_triangular_color_code_cnot_layers(distance)
        num_data = self.code.num_data
        num_plaq = self.code.num_plaquettes

        # Build legacy to our qubit mapping (same logic as _legacy_superdense_cx_layers)
        ordered_our_data = sorted(
            range(num_data),
            key=lambda q: (self.code.qubit_to_coord[q][0], self.code.qubit_to_coord[q][1])
        )
        our_data_to_legacy = {q: i for i, q in enumerate(ordered_our_data)}
        legacy_data_to_our = {i: q for q, i in our_data_to_legacy.items()}

        # Match plaquettes by data sets
        legacy_plaq_data_set_to_p = {}
        for p in range(num_plaq):
            a1_legacy = num_data + 2 * p
            a2_legacy = num_data + 2 * p + 1
            ds = set()
            for layer in legacy_layers:
                for c, t in layer:
                    if c < num_data and t in (a1_legacy, a2_legacy):
                        ds.add(c)
                    if c in (a1_legacy, a2_legacy) and t < num_data:
                        ds.add(t)
            legacy_plaq_data_set_to_p[tuple(sorted(ds))] = p

        # Map legacy ancillas to our ancillas
        legacy_to_our = dict(legacy_data_to_our)
        for plaq in self.code.plaquettes:
            key = tuple(sorted(our_data_to_legacy[q] for q in plaq["data_qubits"]))
            p = legacy_plaq_data_set_to_p.get(key)
            if p is not None:
                legacy_to_our[num_data + 2 * p] = int(plaq["x_ancilla"])
                legacy_to_our[num_data + 2 * p + 1] = int(plaq["z_ancilla"])

        # Now trace connectivity in our numbering
        z_ancillas_our = set(int(z) for z in self.code.zcheck_qubits)
        for layer in legacy_layers:
            for ctrl, tgt in layer:
                ctrl_our = legacy_to_our.get(ctrl, ctrl)
                tgt_our = legacy_to_our.get(tgt, tgt)
                if ctrl_our < num_data and tgt_our in z_ancillas_our:
                    self._z_anc_to_data[tgt_our].add(ctrl_our)
                if ctrl_our in z_ancillas_our and tgt_our < num_data:
                    self._z_anc_to_data[ctrl_our].add(tgt_our)

        # Convert sets to sorted lists for deterministic order
        for z_anc in self._z_anc_to_data:
            self._z_anc_to_data[z_anc] = sorted(self._z_anc_to_data[z_anc])

        # Precompute chromobius 4th coordinates for each plaquette
        # color_index: red=0, green=1, blue=2
        # chromobius_annotation = color_index + xz*3 (xz=0 for X, xz=1 for Z)
        self._plaq_chromobius_coords = []
        color_map = {'red': 0, 'green': 1, 'blue': 2}
        for plaq in self.code.plaquettes:
            color_idx = color_map[plaq['color']]
            # Store (x_coord, z_coord) for each plaquette
            # x_coord = color_idx + 0*3 = color_idx
            # z_coord = color_idx + 1*3 = color_idx + 3
            self._plaq_chromobius_coords.append(
                {
                    'x': color_idx,  # X-check chromobius annotation
                    'z': color_idx + 3,  # Z-check chromobius annotation
                    'grid_pos': plaq['grid_pos'],  # (row, col) for spatial coords
                }
            )

        # Number of stabilizers (same for X and Z)
        num_stabs = self.code.num_plaquettes

        ################## Logical state prep ##################
        self.add_reset(
            self.code.data_qubits, basis
        )  # Reset data qubits to either |0> or |+> depending on basis

        # Stabilizer rounds (superdense scaffold)
        # State-prep round
        self._add_stabilizer_round(state_prep=True, combine_reset_and_measure=True)

        ################# Adding detectors for the first stabilizer round ##################
        # Only add detectors for basis-matched stabilizers in first round:
        # - X basis (|+⟩ prep): Only X stabilizers are deterministic
        # - Z basis (|0⟩ prep): Only Z stabilizers are deterministic
        #
        # Measurement order (with feedforward): Z first, then X
        # After first round:
        #   Z measurements: rec[-(2*num_stabs)] to rec[-(num_stabs+1)]
        #   X measurements: rec[-num_stabs] to rec[-1]

        if self._add_detectors:
            t = 0  # Time coordinate for first round

            if basis == 'X' and get_X_detectors:
                # X stabilizers are deterministic on |+⟩
                for plaq_idx in range(num_stabs):
                    # X measurements are at rec[-num_stabs] to rec[-1]
                    x_ind = num_stabs - plaq_idx
                    coords = self._plaq_chromobius_coords[plaq_idx]
                    det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords['x'])
                    self.add_detector([x_ind], coords=det_coords)

            elif basis == 'Z' and get_Z_detectors:
                # Z stabilizers are deterministic on |0⟩
                for plaq_idx in range(num_stabs):
                    # Z measurements are at rec[-(2*num_stabs)] to rec[-(num_stabs+1)]
                    z_ind = 2 * num_stabs - plaq_idx
                    coords = self._plaq_chromobius_coords[plaq_idx]
                    det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords['z'])
                    self.add_detector([z_ind], coords=det_coords)

        ############## Logical memory w/ noise ###############
        # Bulk memory rounds
        if (self.n_rounds - 2) > 0:
            self.start_loop(self.n_rounds - 2)  # -2: one for state prep, one for final

            self._add_stabilizer_round(combine_reset_and_measure=True)

            if self._add_detectors:
                # Middle round detectors: compare current to previous
                # With feedforward, BOTH X and Z detectors are deterministic
                # Offset between rounds = 2 * num_stabs
                round_offset = 2 * num_stabs
                t = 0  # Time coordinate (relative within REPEAT block)

                # X detectors
                if get_X_detectors:
                    for plaq_idx in range(num_stabs):
                        x_ind = num_stabs - plaq_idx  # X measurement (most recent in round)
                        coords = self._plaq_chromobius_coords[plaq_idx]
                        det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords['x'])
                        self.add_detector([x_ind, x_ind + round_offset], coords=det_coords)

                # Z detectors
                if get_Z_detectors:
                    for plaq_idx in range(num_stabs):
                        z_ind = 2 * num_stabs - plaq_idx  # Z measurement (older in round)
                        coords = self._plaq_chromobius_coords[plaq_idx]
                        det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords['z'])
                        self.add_detector([z_ind, z_ind + round_offset], coords=det_coords)

            self.end_loop()

        ################## Logical measurement ##################
        # Final perfect stabilizer round (with reset errors but no other noise)
        self._add_stabilizer_round(logical_measurement=True, combine_reset_and_measure=True)

        if self._add_detectors:
            # Final round detectors: compare final perfect round to previous
            # With feedforward, BOTH X and Z detectors are deterministic
            round_offset = 2 * num_stabs
            t = 0  # Final round time

            # X detectors
            if get_X_detectors:
                for plaq_idx in range(num_stabs):
                    x_ind = num_stabs - plaq_idx
                    coords = self._plaq_chromobius_coords[plaq_idx]
                    det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords['x'])
                    self.add_detector([x_ind, x_ind + round_offset], coords=det_coords)

            # Z detectors
            if get_Z_detectors:
                for plaq_idx in range(num_stabs):
                    z_ind = 2 * num_stabs - plaq_idx
                    coords = self._plaq_chromobius_coords[plaq_idx]
                    det_coords = (coords['grid_pos'][0], coords['grid_pos'][1], t, coords['z'])
                    self.add_detector([z_ind, z_ind + round_offset], coords=det_coords)

        # Final data measurement (noiseless)
        orig = (self.idle_error, self.sqgate_error, self.tqgate_error, self.spam_error)
        self.set_error_rates_simple(0, 0, 0, 0)
        self.set_error_rates()

        self.add_measure(self.code.data_qubits, basis=self.basis)

        # Restore original error rates
        self.set_error_rates_simple(*orig)
        self.set_error_rates()

        # Add boundary detectors if requested
        # These compare final data qubit measurements to last ancilla measurements
        # Required for proper decoding with PAULI_CHANNEL_2 noise model
        if self._add_detectors and self.add_boundary_detectors:
            self._add_boundary_detectors_to_circuit()

        # Logical observable: follows Gidney's superdense convention
        # X-basis: ALL data qubits (transversal X)
        # Z-basis: bottom edge only (minimal weight Z)
        # Measurements are at rec[-1], rec[-2], ..., rec[-num_data]
        # Data qubit i is at rec[-(num_data - i)]
        if self._add_detectors:
            if self.basis.upper() == 'X':
                # X-basis memory: use ALL data qubits for observable
                obs_inds = list(range(1, num_data + 1))
            else:
                # Z-basis memory: use bottom edge only (d qubits)
                obs_inds = [num_data - q for q in self.code.logical_qubits]
            self.add_observable(0, obs_inds)

        # Build raw circuit (with feedforward CX gates)
        self.stim_circuit_raw = stim.Circuit(self.circuit)

        # Apply with_inlined_feedback() to absorb feedforward into detector/observable definitions
        # This removes the feedforward CX gates and rewrites detectors to be deterministic
        self.stim_circuit = self.stim_circuit_raw.with_inlined_feedback()
