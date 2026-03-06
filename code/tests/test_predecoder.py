# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Tests for model/predecoder: forward pass shape (v1 and v2). Catches breakage from architecture/config changes."""

import unittest
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.predecoder import (
    PreDecoderModelMemory_v1,
    PreDecoderModelMemory_v2,
    get_mock_config,
    get_mock_config_v2,
)


class TestPreDecoderModelMemoryV1(unittest.TestCase):

    def test_forward_shape(self):
        cfg = get_mock_config()
        model = PreDecoderModelMemory_v1(cfg)
        B, C, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D)
        out = model(x)
        self.assertEqual(out.shape, (B, cfg.model.out_channels, T, D, D))


class TestPreDecoderModelMemoryV2(unittest.TestCase):

    def test_forward_shape(self):
        cfg = get_mock_config_v2()
        model = PreDecoderModelMemory_v2(cfg)
        B, C, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D)
        out = model(x)
        self.assertEqual(out.shape, (B, cfg.model.out_channels, T, D, D))


if __name__ == "__main__":
    unittest.main()
