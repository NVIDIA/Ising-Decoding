# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model contract for the color-code accurate cascade configuration.

Verifies that conf/config_color_accurate_d17.yaml instantiates exactly the
intended architecture:
  1. 25,397,764 parameters
  2. a plain Conv3d stem (no bottleneck stem)
  3. 11 standard cascade bottleneck blocks + 1x1x1 output projection
  4. SiLU activations inside the cascade blocks
"""

import sys
import unittest
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.factory import ModelFactory
from model.predecoder import CascadeBottleneckBlock3D

CONFIG_PATH = Path(__file__).parent.parent.parent / "conf" / "config_color_accurate_d17.yaml"
EXPECTED_PARAMS = 25_397_764


def _build_model():
    cfg = OmegaConf.load(CONFIG_PATH)
    return ModelFactory.create_model(cfg)


class TestColorAccurateModelContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _build_model()

    def test_exact_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total, EXPECTED_PARAMS)

    def test_plain_conv3d_stem(self):
        self.assertIsInstance(self.model.input_block, nn.Conv3d)

    def test_eleven_cascade_blocks_and_output_projection(self):
        self.assertEqual(len(self.model.blocks), 11)
        for block in self.model.blocks:
            self.assertIsInstance(block, CascadeBottleneckBlock3D)
        self.assertIsInstance(self.model.output_projection, nn.Conv3d)
        self.assertEqual(self.model.output_projection.kernel_size, (1, 1, 1))

    def test_cascade_blocks_use_silu(self):
        activations = [
            m for m in self.model.blocks.modules()
            if isinstance(m, (nn.SiLU, nn.GELU, nn.ReLU, nn.Tanh))
        ]
        self.assertTrue(activations, "expected activation modules in cascade blocks")
        for act in activations:
            self.assertIsInstance(act, nn.SiLU)


if __name__ == "__main__":
    unittest.main()
