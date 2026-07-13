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
from collections import defaultdict
from typing import Dict, List, Tuple

# Ensure `import qec...` works when running via unittest discovery.
sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.color_code import ColorCode
from qec.color_code.memory_circuit import MemoryCircuit

Pair = Tuple[int, int]


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
        assert len(qs) % 2 == 0, f"CX line must have even number of args: {line}"
        layers.append([(qs[i], qs[i + 1]) for i in range(0, len(qs), 2)])
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


class TestColorCodeNewScheduleDoubleConnections(unittest.TestCase):
    """
    For schedule='long-range' (alias: 'new'):
    - For each plaquette, each data qubit connects to exactly one of {X-ancilla, Z-ancilla}.
    - For that chosen ancilla, the data-ancilla connection occurs exactly twice:
        1) in the forward half (layers 1..3): D -> anc
        2) in the reverse half (layers 4..6): anc -> D
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
        layers = _first_superdense_round_layers(circ.circuit, n_layers=8)
        self.assertEqual(
            len(layers), 8, f"d={d}: expected 8 CX layers for one schedule instance (first round)"
        )

        # For fast membership queries
        flat_pairs: List[Tuple[int, int, int]] = []  # (layer_idx, c, t)
        for li, layer in enumerate(layers):
            for c, t in layer:
                flat_pairs.append((li, c, t))

        for plaq_idx, plaq in enumerate(cc.plaquettes):
            data_set = set(int(q) for q in plaq["data_qubits"])
            x = int(plaq["x_ancilla"])
            z = int(plaq["z_ancilla"])

            occ: Dict[Tuple[int, int], List[Tuple[int, int,
                                                  int]]] = defaultdict(list)  # (dq,anc)-> events
            for li, c, t in flat_pairs:
                if c == x and t in data_set:
                    occ[(t, x)].append((li, c, t))
                if t == x and c in data_set:
                    occ[(c, x)].append((li, c, t))
                if c == z and t in data_set:
                    occ[(t, z)].append((li, c, t))
                if t == z and c in data_set:
                    occ[(c, z)].append((li, c, t))

            for dq in data_set:
                hasx = (dq, x) in occ
                hasz = (dq, z) in occ
                self.assertFalse(
                    hasx and hasz,
                    f"d={d} plaq={plaq_idx} data={dq} connects to BOTH X and Z ancillas",
                )
                self.assertTrue(
                    hasx or hasz,
                    f"d={d} plaq={plaq_idx} data={dq} connects to neither ancilla",
                )
                anc = x if hasx else z
                events = occ[(dq, anc)]
                self.assertEqual(
                    len(events),
                    2,
                    f"d={d} plaq={plaq_idx} data={dq} anc={anc} expected exactly 2 CNOTs, got {events}",
                )

                f = [e for e in events if 1 <= e[0] <= 3]
                r = [e for e in events if 4 <= e[0] <= 6]
                self.assertEqual(
                    len(f),
                    1,
                    f"d={d} plaq={plaq_idx} data={dq} anc={anc} expected 1 forward-half event in layers 1..3, got {events}",
                )
                self.assertEqual(
                    len(r),
                    1,
                    f"d={d} plaq={plaq_idx} data={dq} anc={anc} expected 1 reverse-half event in layers 4..6, got {events}",
                )

                (fli, fc, ft) = f[0]
                (rli, rc, rt) = r[0]
                self.assertEqual(
                    (fc, ft),
                    (dq, anc),
                    f"d={d} plaq={plaq_idx} forward event must be D->anc (layer {fli}), got {f[0]}",
                )
                self.assertEqual(
                    (rc, rt),
                    (anc, dq),
                    f"d={d} plaq={plaq_idx} reverse event must be anc->D (layer {rli}), got {r[0]}",
                )

            # Stronger ordering check per ancilla:
            # All forward D->anc interactions must happen before any reverse anc->D interaction for that ancilla.
            for anc in (x, z):
                forward_layers = []
                reverse_layers = []
                for dq in data_set:
                    key = (dq, anc)
                    if key not in occ:
                        continue
                    for (li, c, t) in occ[key]:
                        if c == dq and t == anc and 1 <= li <= 3:
                            forward_layers.append(li)
                        if c == anc and t == dq and 4 <= li <= 6:
                            reverse_layers.append(li)
                # If anc participates in this plaquette at all, it should have both halves.
                if forward_layers or reverse_layers:
                    self.assertTrue(
                        forward_layers,
                        f"d={d} plaq={plaq_idx} anc={anc} missing forward D->anc events"
                    )
                    self.assertTrue(
                        reverse_layers,
                        f"d={d} plaq={plaq_idx} anc={anc} missing reverse anc->D events"
                    )
                    self.assertLess(
                        max(forward_layers),
                        min(reverse_layers),
                        f"d={d} plaq={plaq_idx} anc={anc} reverse started before forward finished: "
                        f"forward_layers={sorted(forward_layers)} reverse_layers={sorted(reverse_layers)}",
                    )

    def test_odd_distances_through_21(self):
        for d in range(3, 22, 2):
            self._assert_for_d(d)
