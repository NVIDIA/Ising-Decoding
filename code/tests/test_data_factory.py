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
"""Tests for data.factory (DatapipeFactory)."""

import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from data.factory import DatapipeFactory


class TestDatapipeFactoryCreateDatapipe(unittest.TestCase):

    def test_surface_memory_returns_none_none(self):
        cfg = OmegaConf.create({
            "code": "surface",
            "datapipe": "memory",
        })
        a, b = DatapipeFactory.create_datapipe(cfg)
        self.assertIsNone(a)
        self.assertIsNone(b)

    def test_invalid_code_raises(self):
        cfg = OmegaConf.create({"code": "invalid"})
        with self.assertRaises(ValueError):
            DatapipeFactory.create_datapipe(cfg)

    def test_surface_non_memory_datapipe_raises(self):
        cfg = OmegaConf.create({
            "code": "surface",
            "datapipe": "other",
        })
        with self.assertRaises(ValueError):
            DatapipeFactory.create_datapipe(cfg)


class TestDatapipeFactoryCreateDatapipeInference(unittest.TestCase):

    def test_invalid_code_raises(self):
        cfg = OmegaConf.create({"code": "invalid"})
        with self.assertRaises(ValueError):
            DatapipeFactory.create_datapipe_inference(cfg)

    def test_surface_non_memory_datapipe_raises(self):
        cfg = OmegaConf.create({
            "code": "surface",
            "datapipe": "other",
        })
        with self.assertRaises(ValueError):
            DatapipeFactory.create_datapipe_inference(cfg)

    def test_surface_memory_creates_dataset_with_minimal_cfg(self):
        cfg = OmegaConf.create(
            {
                "code": "surface",
                "datapipe": "memory",
                "distance": 5,
                "n_rounds": 5,
                "data": {
                    "error_mode": "circuit_level_surface_custom",
                    "code_rotation": "XV"
                },
                "test":
                    {
                        "num_samples": 100,
                        "p_error": 0.01,
                        "meas_basis_test": "X",
                        "noise_model": "none",
                    },
            }
        )
        pipe = DatapipeFactory.create_datapipe_inference(cfg)
        self.assertIsNotNone(pipe)
        self.assertTrue(hasattr(pipe, "__iter__") or callable(getattr(pipe, "__getitem__", None)))
