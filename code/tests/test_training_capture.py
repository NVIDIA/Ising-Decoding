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
"""Tests for training.capture (_StaticCapture)."""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from training.capture import _StaticCapture


class TestStaticCapture(unittest.TestCase):

    def test_state_dict_empty(self):
        _StaticCapture.load_state_dict({"amp_scalers": {}, "amp_scaler_checkpoints": {}})
        d = _StaticCapture.state_dict()
        self.assertIn("amp_scalers", d)
        self.assertIn("amp_scaler_checkpoints", d)
        self.assertEqual(d["amp_scalers"], {})
        self.assertEqual(d["amp_scaler_checkpoints"], {})

    def test_load_state_dict_empty_none(self):
        _StaticCapture.load_state_dict(None)

    def test_load_state_dict_empty_dict(self):
        _StaticCapture.load_state_dict({})

    def test_load_state_dict_roundtrip(self):
        state = {"amp_scalers": {"k": 1}, "amp_scaler_checkpoints": {"k": 2}}
        _StaticCapture.load_state_dict(state)
        d = _StaticCapture.state_dict()
        self.assertEqual(d["amp_scalers"], {"k": 1})
        self.assertEqual(d["amp_scaler_checkpoints"], {"k": 2})
        _StaticCapture.load_state_dict({"amp_scalers": {}, "amp_scaler_checkpoints": {}})
