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


class TestColorCodeNewScheduleSupport(unittest.TestCase):
    """
    Validate stabilizer *support* for schedule='long-range' (alias: 'new'):
    - For each plaquette p, the only data qubits that ever CNOT with p's ancillas are exactly p['data_qubits'].
    - No ancilla is allowed to CNOT with data outside its plaquette.

    This does not attempt to validate the Pauli type/sign of the measured stabilizer yet.
    """

    def _assert_support_for_d(self, d: int) -> None:
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
        pairs = _parse_cx_pairs(circ.circuit)

        # Precompute a fast map from ancilla -> plaquette index and expected support.
        x_to_plaq = {int(p["x_ancilla"]): i for i, p in enumerate(cc.plaquettes)}
        z_to_plaq = {int(p["z_ancilla"]): i for i, p in enumerate(cc.plaquettes)}

        # Track which data each plaquette's ancillas ever touched.
        touched_x = [set() for _ in cc.plaquettes]
        touched_z = [set() for _ in cc.plaquettes]

        for c, t in pairs:
            # Ancilla-ancilla edges are allowed (X->Z inside a plaquette).
            if c in x_to_plaq and t in z_to_plaq:
                self.assertEqual(
                    x_to_plaq[c],
                    z_to_plaq[t],
                    f"d={d}: found X->Z CNOT across different plaquettes: X{c}(p{x_to_plaq[c]}) -> Z{t}(p{z_to_plaq[t]})",
                )
                continue

            # X ancilla with data
            if c in x_to_plaq and t < cc.num_data:
                pi = x_to_plaq[c]
                self.assertIn(
                    t,
                    cc.plaquettes[pi]["data_qubits"],
                    f"d={d}: X ancilla {c} (plaq {pi}) touched out-of-support data {t}",
                )
                touched_x[pi].add(t)
            if t in x_to_plaq and c < cc.num_data:
                pi = x_to_plaq[t]
                self.assertIn(
                    c,
                    cc.plaquettes[pi]["data_qubits"],
                    f"d={d}: X ancilla {t} (plaq {pi}) touched out-of-support data {c}",
                )
                touched_x[pi].add(c)

            # Z ancilla with data
            if c in z_to_plaq and t < cc.num_data:
                pi = z_to_plaq[c]
                self.assertIn(
                    t,
                    cc.plaquettes[pi]["data_qubits"],
                    f"d={d}: Z ancilla {c} (plaq {pi}) touched out-of-support data {t}",
                )
                touched_z[pi].add(t)
            if t in z_to_plaq and c < cc.num_data:
                pi = z_to_plaq[t]
                self.assertIn(
                    c,
                    cc.plaquettes[pi]["data_qubits"],
                    f"d={d}: Z ancilla {t} (plaq {pi}) touched out-of-support data {c}",
                )
                touched_z[pi].add(c)

        # Now enforce exact support coverage per plaquette.
        for pi, plaq in enumerate(cc.plaquettes):
            expected = set(int(q) for q in plaq["data_qubits"])
            observed = touched_x[pi] | touched_z[pi]
            self.assertEqual(
                observed,
                expected,
                f"d={d}: plaq {pi} observed support != expected support; observed={sorted(observed)} expected={sorted(expected)} "
                f"(X-touched={sorted(touched_x[pi])}, Z-touched={sorted(touched_z[pi])})",
            )

    def test_odd_distances_through_21(self):
        for d in range(3, 22, 2):
            self._assert_support_for_d(d)
