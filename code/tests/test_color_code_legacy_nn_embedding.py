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

from qec.color_code import ColorCode
from qec.color_code.memory_circuit import MemoryCircuit


def _parse_cx_pairs(stim_text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for line in stim_text.splitlines():
        line = line.strip()
        if not line.startswith("CX "):
            continue
        parts = [p for p in line.split(" ") if p]
        # Ignore classically-controlled X feedforward like: "CX rec[-k] q".
        if any(p.startswith("rec[") for p in parts[1:]):
            continue
        qs = list(map(int, parts[1:]))
        assert len(qs) % 2 == 0
        pairs += [(qs[i], qs[i + 1]) for i in range(0, len(qs), 2)]
    return pairs


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class TestLegacyScheduleNearestNeighborEmbedding(unittest.TestCase):

    def test_legacy_schedule_is_nn_on_physical_grid_under_flip(self):
        # The legacy schedule can be embedded on a 2D NN grid using the circuit-only physical layout.
        # Flipping the triangle orientation (row reflection) must not break NN adjacency.
        for d in range(3, 22, 2):
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
            pairs = _parse_cx_pairs(circ.circuit)

            for flip_rows in (False, True):
                pos = cc.get_circuit_physical_layout(id_order="rtl", flip_rows=flip_rows)
                bad = []
                for c, t in pairs:
                    md = _manhattan(pos[c], pos[t])
                    if md != 1:
                        bad.append((md, c, t, pos[c], pos[t]))
                self.assertEqual(
                    bad,
                    [],
                    msg=
                    f"d={d} flip_rows={flip_rows} had NN violations (showing up to 5): {bad[:5]}",
                )


if __name__ == "__main__":
    unittest.main()
