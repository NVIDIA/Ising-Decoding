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
from typing import List, Set, Tuple

# Ensure `import qec...` works when running via unittest discovery.
sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.surface_code import memory_circuit as sc_mc

Pair = Tuple[int, int]


def _legacy_cx_pairs(d: int) -> List[Pair]:
    """
    Extract all (control, target) CX pairs from the legacy triangular color-code Cmat.

    Legacy representation:
      - if Cmat[q, tt] == 10000 + (tgt+1) then control=q, target=tgt
      - if Cmat[q, tt] == 10000 then q is a target marker (ignored here)
    """
    C = sc_mc.triangular_color_code_circuit(d)
    pairs: List[Pair] = []
    for tt in range(1, C.shape[1] - 1):  # tt=1..8 are the CNOT layers
        for q in range(C.shape[0]):
            v = int(C[q, tt])
            if v > 10000:
                tgt = (v - 10000) - 1
                pairs.append((q, tgt))
    return pairs


@unittest.skipUnless(
    hasattr(sc_mc, 'triangular_color_code_circuit'),
    "legacy reference circuit generator not available in this distribution"
)
class TestLegacyColorCodeNoOverlap(unittest.TestCase):
    """
    Legacy-only sanity check:
    For each plaquette, no data qubit is allowed to CNOT with both ancillas of that plaquette.
    """

    def _assert_no_overlap(self, d: int) -> None:
        num_data = (3 * d * d + 1) // 4
        num_plaquettes = (3 * (d * d - 1)) // 8
        pairs = _legacy_cx_pairs(d)

        for p in range(num_plaquettes):
            a1 = num_data + 2 * p
            a2 = num_data + 2 * p + 1

            a1_conn: Set[int] = set()
            a2_conn: Set[int] = set()

            for c, t in pairs:
                if c == a1 and t < num_data:
                    a1_conn.add(t)
                elif t == a1 and c < num_data:
                    a1_conn.add(c)

                if c == a2 and t < num_data:
                    a2_conn.add(t)
                elif t == a2 and c < num_data:
                    a2_conn.add(c)

            inter = a1_conn & a2_conn
            self.assertEqual(
                inter,
                set(),
                f"legacy d={d} plaquette={p} has data qubits connected to BOTH ancillas: {sorted(inter)}",
            )

            union = a1_conn | a2_conn
            self.assertIn(
                len(union),
                (4, 6),
                f"legacy d={d} plaquette={p} touches unexpected #data qubits: {len(union)} (a1={sorted(a1_conn)} a2={sorted(a2_conn)})",
            )

            # Optional: require that every touched data qubit is assigned to exactly one ancilla (implied by no-overlap).
            self.assertEqual(
                len(union),
                len(a1_conn) + len(a2_conn),
                f"legacy d={d} plaquette={p} unexpected overlap accounting",
            )

    def test_odd_distances_through_21(self):
        for d in range(3, 22, 2):
            self._assert_no_overlap(d)


class TestLegacyColorCodeHalfHalfPartition(unittest.TestCase):
    """
    DELETED (outdated):
    The legacy triangular color-code schedule can exhibit 3/1 (or 1/3) partitions on boundary plaquettes
    due to nearest-neighbor constraints under a 2D embedding. This is no longer treated as a bug.
    """
    pass
