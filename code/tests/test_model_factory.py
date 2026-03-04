# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for model.factory (ModelFactory)."""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from model.factory import ModelFactory
from model.predecoder import get_mock_config, get_mock_config_v2


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

    def test_create_surface_model_v2(self):
        cfg = get_mock_config_v2()
        cfg.code = "surface"
        cfg.model.version = "predecoder_memory_v2"
        model = ModelFactory.create_model(cfg)
        self.assertIsNotNone(model)
        self.assertEqual(model.distance, cfg.distance)
        self.assertEqual(model.n_rounds, cfg.n_rounds)
