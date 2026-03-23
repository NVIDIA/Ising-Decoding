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
"""Tests for model/predecoder: forward pass shape (v1). Catches breakage from architecture/config changes."""

import unittest
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.predecoder import (
    PreDecoderModelMemory_v1,
    get_mock_config,
)


class TestPreDecoderModelMemoryV1(unittest.TestCase):

    def test_forward_shape(self):
        cfg = get_mock_config()
        model = PreDecoderModelMemory_v1(cfg)
        B, C, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D)
        out = model(x)
        self.assertEqual(out.shape, (B, cfg.model.out_channels, T, D, D))


if __name__ == "__main__":
    unittest.main()
