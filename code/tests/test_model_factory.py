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
"""Tests for model.factory (ModelFactory)."""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from model.factory import ModelFactory
from model.predecoder import get_mock_config


class TestModelFactory(unittest.TestCase):

    def test_invalid_code_raises(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(code="invalid")
        with self.assertRaises(ValueError):
            ModelFactory.create_model(cfg)

    def test_invalid_model_version_raises(self):
        cfg = get_mock_config()
        cfg.code = "surface"
        cfg.model.version = "unknown_version"
        with self.assertRaises(ValueError):
            ModelFactory.create_model(cfg)

    def test_create_surface_model_v1(self):
        cfg = get_mock_config()
        cfg.code = "surface"
        cfg.model.version = "predecoder_memory_v1"
        model = ModelFactory.create_model(cfg)
        self.assertIsNotNone(model)
        self.assertEqual(model.distance, cfg.distance)
        self.assertEqual(model.n_rounds, cfg.n_rounds)
