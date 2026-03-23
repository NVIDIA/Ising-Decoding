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
"""Tests for qec.surface_code.stim_utils (REPEAT unfolding, add_instruction)."""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from qec.surface_code import stim_utils


class TestUnfoldRepeatInstruction(unittest.TestCase):

    def test_no_repeat_returns_unchanged(self):
        lines = ["R 0 1", "H 0", "M 0"]
        out = stim_utils.unfold_repeat_instruction(lines[:], ignore_detectors=True)
        self.assertEqual(out, ["R 0 1", "H 0", "M 0"])

    def test_repeat_block_unfolded_with_detectors(self):
        # REPEAT 2 { ... } -> round 1 and 2 body; detector ids shifted
        lines = [
            "R 0 1",
            "REPEAT 2 {",
            "H 0",
            "DETECTOR rec[0] rec[1]",
            "}",
        ]
        out = stim_utils.unfold_repeat_instruction(lines, ignore_detectors=False)
        self.assertIn("H 0", out)
        self.assertIn("DETECTOR rec[0] rec[1]", out)
        detector_lines = [l for l in out if "DETECTOR" in l]
        self.assertEqual(len(detector_lines), 2)

    def test_repeat_ignore_detectors_drops_detector_lines(self):
        lines = [
            "REPEAT 2 {",
            "H 0",
            "DETECTOR rec[0] rec[1]",
            "}",
        ]
        out = stim_utils.unfold_repeat_instruction(lines, ignore_detectors=True)
        for line in out:
            self.assertNotIn("DETECTOR", line)


class TestAddInstruction(unittest.TestCase):

    def test_single_qubit_gate(self):
        import numpy as np
        flips = {"X": np.array([0, 2])}
        margin = stim_utils.add_instruction(flips)
        self.assertIn("X 0 2", margin)

    def test_empty_gate_no_instruction(self):
        import numpy as np
        flips = {"X": np.array([])}
        margin = stim_utils.add_instruction(flips)
        self.assertEqual(margin.strip(), "")

    def test_two_qubit_gate_splits_lines(self):
        import numpy as np
        # XX (no 'I') triggers reshape path: q1=[0,2], q2=[1,3]
        flips = {"XX": np.array([0, 1, 2, 3])}
        margin = stim_utils.add_instruction(flips)
        self.assertIn("X ", margin)
        self.assertIn("0 2", margin)
        self.assertIn("1 3", margin)
