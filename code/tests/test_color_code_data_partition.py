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


class TestColorCodeDataAncillaPartition(unittest.TestCase):
    """
    For each plaquette, each data qubit must connect via CNOTs to exactly one of:
      - the plaquette's X ancilla (a1)
      - the plaquette's Z ancilla (a2)

    i.e., no data qubit in a plaquette is allowed to CNOT with both ancillas.
    (For now we only enforce the **no-overlap** property plus coverage; boundary plaquettes
    can have 1/3 or 3/1 splits depending on the legacy schedule.)
    """

    def _assert_partition_property(self, d: int) -> None:
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
            schedule="nearest-neighbor",
        )
        cx_pairs = _parse_cx_pairs(circ.circuit)

        # Build connectivity maps per plaquette.
        for plaq_idx, plaq in enumerate(cc.plaquettes):
            data_set = set(int(q) for q in plaq["data_qubits"])
            x = int(plaq["x_ancilla"])
            z = int(plaq["z_ancilla"])
            w = int(plaq["weight"])
            self.assertIn(w, (4, 6), f"Unexpected plaquette weight w={w} at d={d} plaq={plaq_idx}")

            x_connected = set()
            z_connected = set()

            for c, t in cx_pairs:
                # Ignore ancilla-ancilla or unrelated edges quickly.
                if c == x or t == x:
                    other = t if c == x else c
                    if other in data_set:
                        x_connected.add(other)
                if c == z or t == z:
                    other = t if c == z else c
                    if other in data_set:
                        z_connected.add(other)

            # No data qubit may connect to both ancillas for the same plaquette.
            inter = x_connected & z_connected
            self.assertEqual(
                inter,
                set(),
                f"d={d} plaq={plaq_idx} has data qubits connected to BOTH X and Z ancillas: {sorted(inter)}",
            )

            # Must cover all data qubits in the plaquette.
            union = x_connected | z_connected
            self.assertEqual(
                union,
                data_set,
                f"d={d} plaq={plaq_idx} missing connectivity for data qubits: "
                f"expected={sorted(data_set)} got={sorted(union)} (X={sorted(x_connected)} Z={sorted(z_connected)})",
            )

    def test_odd_distances_through_21(self):
        for d in range(3, 22, 2):
            self._assert_partition_property(d)


@unittest.skipUnless(
    hasattr(sc_mc, 'triangular_color_code_circuit'),
    "legacy reference circuit generator not available in this distribution"
)
class TestLegacyTriangularColorCodeNoOverlap(unittest.TestCase):
    """
    Run the same **no-overlap** invariant directly on the legacy reference circuit generator:
    `qec.surface_code.memory_circuit.triangular_color_code_circuit(d)`.

    For each plaquette p (with ancillas a1 = num_data + 2p and a2 = num_data + 2p + 1),
    no data qubit is allowed to CNOT with both a1 and a2 anywhere in the stabilizer schedule.
    """

    @staticmethod
    def _legacy_cx_pairs(d: int) -> List[Pair]:
        C = sc_mc.triangular_color_code_circuit(d)
        pairs: List[Pair] = []
        # tt=1..8 are the CNOT layers; entries >10000 encode controls
        for tt in range(1, C.shape[1] - 1):
            for q in range(C.shape[0]):
                v = int(C[q, tt])
                if v > 10000:
                    tgt = (v - 10000) - 1
                    pairs.append((q, tgt))
        return pairs

    def test_no_overlap_odd_distances_through_21(self):
        for d in range(3, 22, 2):
            num_data = (3 * d * d + 1) // 4
            num_plaquettes = (3 * (d * d - 1)) // 8
            pairs = self._legacy_cx_pairs(d)

            for p in range(num_plaquettes):
                a1 = num_data + 2 * p
                a2 = num_data + 2 * p + 1

                a1_connected = set()
                a2_connected = set()

                for c, t in pairs:
                    # data <-> a1
                    if c == a1 and t < num_data:
                        a1_connected.add(t)
                    if t == a1 and c < num_data:
                        a1_connected.add(c)
                    # data <-> a2
                    if c == a2 and t < num_data:
                        a2_connected.add(t)
                    if t == a2 and c < num_data:
                        a2_connected.add(c)

                inter = a1_connected & a2_connected
                self.assertEqual(
                    inter,
                    set(),
                    f"legacy d={d} plaquette={p} has data qubits connected to BOTH a1 and a2: {sorted(inter)}",
                )
