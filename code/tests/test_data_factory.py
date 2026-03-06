# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
