# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SyncBatchNorm conversion in distributed training.

Verifies:
  1. conversion is opt-in and only applies with more than one rank
  2. every BatchNorm in the cascade blocks is converted
  3. state_dict keys are unchanged, so checkpoints load in either direction
     between single-GPU (BatchNorm) and multi-GPU (SyncBatchNorm) runs
"""

import os
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
    return [type(m) for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class TestSyncBatchNormConversion(unittest.TestCase):

    def test_conversion_is_opt_in(self):
        unchanged = maybe_convert_sync_batchnorm(_small_bn_model(), enabled=False, world_size=4)
        self.assertTrue(all(t is nn.BatchNorm3d for t in _batchnorm_types(unchanged)))

    def test_conversion_is_distributed_only(self):
        unchanged = maybe_convert_sync_batchnorm(_small_bn_model(), enabled=True, world_size=1)
        self.assertTrue(all(t is nn.BatchNorm3d for t in _batchnorm_types(unchanged)))

    def test_all_batchnorms_converted_when_enabled(self):
        converted = maybe_convert_sync_batchnorm(_small_bn_model(), enabled=True, world_size=2)
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
        synced = maybe_convert_sync_batchnorm(_small_bn_model(), enabled=True, world_size=2)
        self.assertEqual(set(plain.state_dict().keys()), set(synced.state_dict().keys()))

    def test_checkpoint_roundtrip_between_variants(self):
        plain = _small_bn_model()
        # Perturb running stats so the load is meaningful
        with torch.no_grad():
            plain(torch.randn(2, 2, 4, 4, 4))
        synced = maybe_convert_sync_batchnorm(_small_bn_model(), enabled=True, world_size=2)
        synced.load_state_dict(plain.state_dict())
        back = _small_bn_model()
        back.load_state_dict(synced.state_dict())
        for key, value in plain.state_dict().items():
            if value.dtype.is_floating_point:
                torch.testing.assert_close(back.state_dict()[key], value)


def _fresh_rendezvous_file():
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".rendezvous")
    os.close(fd)
    os.remove(path)
    return path


def _worker_sync_batchnorm_stats(rank, world_size, init_file, result_path):
    """Run one rank of a SyncBatchNorm forward/backward on distinct data."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import torch.distributed as dist
    from training.train import maybe_convert_sync_batchnorm

    # Both ranks share cuda:0 so this exercises the real sync path on a
    # single-GPU machine; gloo carries the CUDA tensors.
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    device = torch.device("cuda", 0)
    torch.manual_seed(0)

    model = nn.Sequential(nn.Conv3d(2, 4, kernel_size=3, padding=1), nn.BatchNorm3d(4))
    model = maybe_convert_sync_batchnorm(model, enabled=True, world_size=world_size)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)

    # Offset each rank's data so per-rank statistics would diverge without sync.
    x = (torch.randn(4, 2, 3, 3, 3) + rank * 10.0).to(device)
    model(x).sum().backward()

    bn = next(m for m in model.modules() if isinstance(m, nn.SyncBatchNorm))
    if rank == 0:
        torch.save(bn.running_mean.cpu(), result_path + ".0")
    else:
        torch.save(bn.running_mean.cpu(), result_path + ".1")
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for SyncBatchNorm")
class TestSyncBatchNormAcrossRanks(unittest.TestCase):

    def test_running_stats_are_identical_across_ranks(self):
        """Ranks fed different data must end with the same running statistics."""
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            result_path = os.path.join(tmp, "running_mean")
            mp.spawn(
                _worker_sync_batchnorm_stats,
                args=(2, _fresh_rendezvous_file(), result_path),
                nprocs=2,
                join=True,
            )
            rank0 = torch.load(result_path + ".0")
            rank1 = torch.load(result_path + ".1")

        torch.testing.assert_close(rank0, rank1)
        # Guard against the stats being trivially zero (i.e. never updated).
        self.assertGreater(float(rank0.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
