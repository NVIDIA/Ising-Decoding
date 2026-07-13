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

import unittest
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure `import qec...` works when running via unittest discovery.
sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.color_code import ColorCode
from qec.color_code.memory_circuit import MemoryCircuit
from qec.surface_code import memory_circuit as sc_mc
from qec.color_code.memory_circuit import _legacy_triangular_color_code_cnot_layers


def _parse_cx_layers_from_stim_text(circuit_text: str) -> List[List[Tuple[int, int]]]:
    """
    Extract CX layers from a Stim circuit string.
    Returns a list of layers, each layer is a list of (control, target) pairs.
    """
    layers = []
    for line in circuit_text.splitlines():
        line = line.strip()
        if not line.startswith("CX "):
            continue
        parts = [p for p in line.split(" ") if p]
        # Ignore classically-controlled X feedforward like: "CX rec[-k] q".
        if any(p.startswith("rec[") for p in parts[1:]):
            continue
        qubits = list(map(int, parts[1:]))
        assert len(qubits) % 2 == 0, f"CX line must have an even number of qubit args: {line}"
        pairs = [(qubits[i], qubits[i + 1]) for i in range(0, len(qubits), 2)]
        layers.append(pairs)
    return layers


def _first_superdense_round_layers(circuit_text: str,
                                   *,
                                   n_layers: int = 8) -> List[List[Tuple[int, int]]]:
    """
    MemoryCircuit builds multiple stabilizer rounds even when n_rounds=1 (state-prep + final logical-meas round).
    Each round contributes an 8-layer superdense schedule, so the raw Stim text can contain 16+ CX lines.

    These schedule-structure tests are intended to validate a *single* 8-layer schedule instance, so we
    slice out the first N non-feedback CX layers.
    """
    layers = _parse_cx_layers_from_stim_text(circuit_text)
    if len(layers) < n_layers:
        raise AssertionError(f"Expected at least {n_layers} CX layers, got {len(layers)}")
    return layers[:n_layers]


def _legacy_cx_layers(d: int) -> List[List[Tuple[int, int]]]:
    """
    Extract legacy CX layers from triangular_color_code_circuit(d).
    Returns 8 layers (tt=1..8), each a list of (control, target) pairs.
    """
    C = sc_mc.triangular_color_code_circuit(d)
    layers = []
    for tt in range(1, C.shape[1] - 1):
        edges = []
        for q in range(C.shape[0]):
            v = int(C[q, tt])
            if v > 10000:
                tgt = (v - 10000) - 1
                edges.append((q, tgt))
        layers.append(sorted(edges))
    return layers


def _local_legacy_cx_layers(d: int) -> List[List[Tuple[int, int]]]:
    return _legacy_triangular_color_code_cnot_layers(d)


def _compute_legacy_to_ours_qubit_map(cc: ColorCode) -> Dict[int, int]:
    """
    Build a qubit-ID mapping from legacy -> ours for the triangular color code.

    This mapping is derived exactly as used by MemoryCircuit._legacy_superdense_cx_layers():
    - Data qubits: legacy inverted-triangle row-major order corresponds to our data sorted bottom-to-top, left-to-right.
    - Ancillas: match plaquettes by their data-qubit sets after mapping, then map (a1,a2) -> (x_ancilla,z_ancilla).
    """
    d = int(cc.distance)
    C = sc_mc.triangular_color_code_circuit(d)
    num_data = int(cc.num_data)
    num_plaquettes = int(cc.num_plaquettes)

    # Our data -> legacy data by position
    ordered_our_data = sorted(
        range(num_data), key=lambda q: (cc.qubit_to_coord[q][0], cc.qubit_to_coord[q][1])
    )
    our_data_to_legacy = {q: i for i, q in enumerate(ordered_our_data)}
    legacy_data_to_our = {i: q for q, i in our_data_to_legacy.items()}

    legacy_layers = _legacy_cx_layers(d)

    # Reconstruct legacy plaquette data sets (in legacy data IDs)
    legacy_plaq_data_set_to_p = {}
    for p in range(num_plaquettes):
        a1_legacy = num_data + 2 * p
        a2_legacy = num_data + 2 * p + 1
        ds = set()
        for edges in legacy_layers:
            for c, t in edges:
                if c < num_data and t in (a1_legacy, a2_legacy):
                    ds.add(c)
                if c in (a1_legacy, a2_legacy) and t < num_data:
                    ds.add(t)
        legacy_plaq_data_set_to_p[tuple(sorted(ds))] = p

    # Build full mapping
    legacy_to_our = dict(legacy_data_to_our)
    for plaq in cc.plaquettes:
        ours_set_legacy_ids = tuple(sorted(our_data_to_legacy[q] for q in plaq["data_qubits"]))
        p = legacy_plaq_data_set_to_p.get(ours_set_legacy_ids)
        if p is None:
            raise AssertionError(
                f"Could not match plaquette data set to legacy: {ours_set_legacy_ids}"
            )
        a1_legacy = num_data + 2 * p
        a2_legacy = num_data + 2 * p + 1
        legacy_to_our[a1_legacy] = int(plaq["x_ancilla"])
        legacy_to_our[a2_legacy] = int(plaq["z_ancilla"])

    return legacy_to_our


@unittest.skipUnless(
    hasattr(sc_mc, 'triangular_color_code_circuit'),
    "legacy reference circuit generator not available in this distribution"
)
class TestColorCodeLegacyCnotMapping(unittest.TestCase):

    def _assert_layers_match_under_mapping(self, d: int) -> None:
        cc = ColorCode(d)

        # Generate our circuit with a single stabilizer round and no ticks to keep CX lines aligned to layers.
        circ = MemoryCircuit(
            distance=d,
            idle_error=0.0,
            sqgate_error=0.0,
            tqgate_error=0.0,
            spam_error=0.0,
            n_rounds=1,
            basis="X",
            add_tick=False,
            add_detectors=False,
            schedule="nearest-neighbor",
        )
        our_layers = _first_superdense_round_layers(circ.circuit, n_layers=8)
        legacy_layers = _legacy_cx_layers(d)

        # Build legacy->ours mapping, then invert it to map our->legacy for comparison.
        legacy_to_our = _compute_legacy_to_ours_qubit_map(cc)
        our_to_legacy = {v: k for k, v in legacy_to_our.items()}

        # We expect exactly 8 CNOT layers (tt=1..8) in legacy, and we compare against the first 8 layers
        # of our circuit (one superdense schedule instance).
        self.assertEqual(len(legacy_layers), 8, "Legacy must have 8 CX layers (tt=1..8)")
        self.assertEqual(
            len(our_layers), 8,
            f"Expected 8 CX layers for the first schedule instance; got {len(our_layers)} for d={d}"
        )

        # Compare layer-by-layer as sets of directed edges after mapping our -> legacy.
        for layer_idx in range(8):
            ours_mapped = sorted(
                (our_to_legacy[c], our_to_legacy[t]) for c, t in our_layers[layer_idx]
            )
            self.assertEqual(
                ours_mapped,
                legacy_layers[layer_idx],
                f"Mismatch at d={d} layer={layer_idx}",
            )

    def test_d3(self):
        self._assert_layers_match_under_mapping(3)

    def test_d5(self):
        self._assert_layers_match_under_mapping(5)

    def test_d7(self):
        self._assert_layers_match_under_mapping(7)

    def test_local_schedule_matches_legacy_raw_layers(self):
        """
        Ensure our local schedule generator reproduces the legacy raw CX layers exactly in legacy qubit IDs.
        """
        for d in (3, 5, 7):
            self.assertEqual(
                _local_legacy_cx_layers(d), _legacy_cx_layers(d), f"Raw schedule mismatch at d={d}"
            )
