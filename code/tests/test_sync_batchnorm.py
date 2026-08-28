# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SyncBatchNorm conversion in distributed training.

Verifies:
  1. conversion is opt-in and only applies with more than one rank
  2. every BatchNorm in the cascade blocks is converted
  3. state_dict keys are unchanged, so checkpoints load in either direction
     between single-GPU (BatchNorm) and multi-GPU (SyncBatchNorm) runs
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.predecoder import CascadeBottleneckBlock3D
from training.train import maybe_convert_sync_batchnorm


def _small_bn_model():
    """Small stand-in mirroring the pre-decoder's Conv3d+BatchNorm3d blocks."""
    return nn.Sequential(
        nn.Conv3d(2, 4, kernel_size=3, padding=1),
        nn.BatchNorm3d(4),
        nn.SiLU(),
        nn.Conv3d(4, 2, kernel_size=3, padding=1),
        nn.BatchNorm3d(2),
    )


def _batchnorm_types(module):
    return [
        type(m) for m in module.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


class TestSyncBatchNormConversion(unittest.TestCase):
    def test_conversion_is_opt_in(self):
        unchanged = maybe_convert_sync_batchnorm(
            _small_bn_model(), enabled=False, world_size=4
        )
        self.assertTrue(all(t is nn.BatchNorm3d for t in _batchnorm_types(unchanged)))

    def test_conversion_is_distributed_only(self):
        unchanged = maybe_convert_sync_batchnorm(
            _small_bn_model(), enabled=True, world_size=1
        )
        self.assertTrue(all(t is nn.BatchNorm3d for t in _batchnorm_types(unchanged)))

    def test_all_batchnorms_converted_when_enabled(self):
        converted = maybe_convert_sync_batchnorm(
            _small_bn_model(), enabled=True, world_size=2
        )
        types_ = _batchnorm_types(converted)
        self.assertTrue(types_, "expected BatchNorm layers in the model")
        self.assertTrue(all(t is nn.SyncBatchNorm for t in types_))

    def test_cascade_block_converts(self):
        block = maybe_convert_sync_batchnorm(
            CascadeBottleneckBlock3D(in_channels=8, out_channels=8),
            enabled=True,
            world_size=2,
        )
        types_ = _batchnorm_types(block)
        self.assertTrue(types_, "expected BatchNorm layers in CascadeBottleneckBlock3D")
        self.assertTrue(all(t is nn.SyncBatchNorm for t in types_))

    def test_state_dict_keys_unchanged(self):
        plain = _small_bn_model()
        synced = maybe_convert_sync_batchnorm(
            _small_bn_model(), enabled=True, world_size=2
        )
        self.assertEqual(set(plain.state_dict().keys()), set(synced.state_dict().keys()))

    def test_checkpoint_roundtrip_between_variants(self):
        plain = _small_bn_model()
        # Perturb running stats so the load is meaningful
        with torch.no_grad():
            plain(torch.randn(2, 2, 4, 4, 4))
        synced = maybe_convert_sync_batchnorm(
            _small_bn_model(), enabled=True, world_size=2
        )
        synced.load_state_dict(plain.state_dict())
        back = _small_bn_model()
        back.load_state_dict(synced.state_dict())
        for key, value in plain.state_dict().items():
            if value.dtype.is_floating_point:
                torch.testing.assert_close(back.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
