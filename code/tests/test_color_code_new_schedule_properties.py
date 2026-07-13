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
from typing import List, Tuple

# Ensure `import qec...` works when running via unittest discovery.
sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.color_code import ColorCode
from qec.color_code.memory_circuit import MemoryCircuit

Pair = Tuple[int, int]


def _parse_cx_pairs(stim_text: str) -> List[Pair]:
    pairs: List[Pair] = []
    for line in stim_text.splitlines():
        line = line.strip()
        if not line.startswith("CX "):
            continue
        parts = [p for p in line.split(" ") if p]
        # Ignore classically-controlled X feedforward like: "CX rec[-k] q".
        if any(p.startswith("rec[") for p in parts[1:]):
            continue
        qs = list(map(int, parts[1:]))
        assert len(qs) % 2 == 0, f"CX line must have even number of args: {line}"
        pairs.extend([(qs[i], qs[i + 1]) for i in range(0, len(qs), 2)])
    return pairs


def _parse_cx_layers(stim_text: str) -> List[List[Pair]]:
    layers: List[List[Pair]] = []
    for line in stim_text.splitlines():
        line = line.strip()
        if not line.startswith("CX "):
            continue
        parts = [p for p in line.split(" ") if p]
        # Ignore classically-controlled X feedforward like: "CX rec[-k] q".
        if any(p.startswith("rec[") for p in parts[1:]):
            continue
        qs = list(map(int, parts[1:]))
        pairs = [(qs[i], qs[i + 1]) for i in range(0, len(qs), 2)]
        layers.append(pairs)
    return layers


def _first_superdense_round_layers(stim_text: str, *, n_layers: int = 8) -> List[List[Pair]]:
    """
    MemoryCircuit builds multiple stabilizer rounds even when n_rounds=1 (state-prep + final logical-meas round).
    Each round contributes an 8-layer superdense schedule, so the raw Stim text can contain 16+ CX lines.

    These schedule-structure tests are intended to validate a *single* 8-layer schedule instance, so we
    slice out the first N non-feedback CX layers.
    """
    layers = _parse_cx_layers(stim_text)
    if len(layers) < n_layers:
        raise AssertionError(f"Expected at least {n_layers} CX layers, got {len(layers)}")
    return layers[:n_layers]


class TestColorCodeNewScheduleProperties(unittest.TestCase):
    """
    Validate the long-range schedule ('schedule=\"long-range\"', alias: \"new\") satisfies:
      - 50/50 partition per plaquette: weight-4 -> 2/2, weight-6 -> 3/3
      - no data qubit connects to both ancillas of the same plaquette
      - each CX layer is disjoint (no qubit used twice in the same layer)
    """

    def _assert_for_d(self, d: int) -> None:
        cc = ColorCode(d)
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
            schedule="long-range",
        )

        cx_pairs = _parse_cx_pairs(circ.circuit)
        cx_layers = _first_superdense_round_layers(circ.circuit, n_layers=8)

        # Must be exactly 8 layers by construction.
        self.assertEqual(
            len(cx_layers), 8,
            f"d={d}: expected exactly 8 CX layers for one schedule instance (first round)"
        )

        # Per-layer disjointness.
        for li, pairs in enumerate(cx_layers):
            used = set()
            for c, t in pairs:
                self.assertNotIn(c, used, f"d={d}: layer {li} reuses qubit {c}")
                self.assertNotIn(t, used, f"d={d}: layer {li} reuses qubit {t}")
                used.add(c)
                used.add(t)

        # Per-plaquette 50/50 + no-overlap + coverage.
        for plaq_idx, plaq in enumerate(cc.plaquettes):
            data_set = set(int(q) for q in plaq["data_qubits"])
            x = int(plaq["x_ancilla"])
            z = int(plaq["z_ancilla"])
            w = int(plaq["weight"])
            expected_half = w // 2

            x_conn = set()
            z_conn = set()
            for c, t in cx_pairs:
                if (c == x and t in data_set) or (t == x and c in data_set):
                    x_conn.add(t if c == x else c)
                if (c == z and t in data_set) or (t == z and c in data_set):
                    z_conn.add(t if c == z else c)

            self.assertEqual(
                x_conn & z_conn, set(),
                f"d={d} plaq={plaq_idx}: data overlaps between X and Z ancillas"
            )
            self.assertEqual(
                x_conn | z_conn, data_set,
                f"d={d} plaq={plaq_idx}: not all data qubits are connected"
            )
            self.assertEqual(
                len(x_conn), expected_half,
                f"d={d} plaq={plaq_idx}: expected {expected_half} X-connected data qubits"
            )
            self.assertEqual(
                len(z_conn), expected_half,
                f"d={d} plaq={plaq_idx}: expected {expected_half} Z-connected data qubits"
            )

    def test_odd_distances_through_21(self):
        for d in range(3, 22, 2):
            self._assert_for_d(d)
