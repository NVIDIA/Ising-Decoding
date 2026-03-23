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
"""Tests for evaluation/logical_error_rate: sample_predictions (LER pipeline) and interleave_XZ_residuals (decoder input ordering)."""

import unittest
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.logical_error_rate import (
    sample_predictions,
    interleave_XZ_residuals,
)


class TestSamplePredictions(unittest.TestCase):
    """sample_predictions drives LER evaluation; wrong threshold/temperature behavior changes reported metrics."""

    def test_threshold_mode_deterministic_and_reproducible(self):
        logits = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]])
        out = sample_predictions(logits, threshold=0.0, sampling_mode="threshold")
        self.assertEqual(out.dtype, torch.int32)
        expected = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int32)
        self.assertTrue(torch.equal(out, expected))
        # Same input must give same output (no randomness)
        out2 = sample_predictions(logits, threshold=0.0, sampling_mode="threshold")
        self.assertTrue(torch.equal(out, out2))

    def test_temperature_mode_produces_binary_and_respects_extreme_logits(self):
        torch.manual_seed(42)
        # Very large magnitude logits: temperature scaling should still push to near 0/1
        logits = torch.tensor([[10.0, -10.0]])
        out = sample_predictions(logits, sampling_mode="temperature", temperature=0.5)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.all((out == 0) | (out == 1)))
        self.assertEqual(out[0, 0].item(), 1)
        self.assertEqual(out[0, 1].item(), 0)


class TestInterleaveXZResiduals(unittest.TestCase):
    """Interleave order must match what the decoder expects (round-major, X then Z per round)."""

    def test_ordering_matches_round_major_x_then_z(self):
        B, nX, nZ, T = 2, 2, 2, 2
        R_X = torch.tensor([[[1, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=torch.float32)
        R_Z = torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=torch.float32)
        out = interleave_XZ_residuals(R_X, R_Z)
        self.assertEqual(out.shape, (B, T * (nX + nZ)))
        # Round 0: R_X[0,:,0] then R_Z[0,:,0] -> [1,0,0,1]
        self.assertTrue(torch.equal(out[0, :4], torch.tensor([1.0, 0.0, 0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
