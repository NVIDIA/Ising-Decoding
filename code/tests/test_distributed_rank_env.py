# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher environment contract for multi-node runs.

A launcher must set global RANK separately from node-local LOCAL_RANK. If
RANK were derived from LOCAL_RANK, every node would use the same ranks, and
the per-rank generator seeds (keyed on global rank) would repeat across
nodes so all nodes trained on identical samples.

Verifies that DistributedManager reads the two independently and derives the
CUDA device from the local index.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.distributed import DistributedManager

RANK_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


def _with_rank_env(**overrides):
    env = {k: v for k, v in os.environ.items() if k not in RANK_ENV_KEYS}
    env.update(overrides)
    return patch.dict(os.environ, env, clear=True)


class TestDistributedRankEnv(unittest.TestCase):

    def test_global_rank_is_independent_of_local_rank(self):
        # Node 3 of an 8-node x 4-GPU job, second GPU on that node.
        with _with_rank_env(RANK="13", LOCAL_RANK="1", WORLD_SIZE="32"):
            dist_manager = DistributedManager()
        self.assertEqual(dist_manager.rank, 13)
        self.assertEqual(dist_manager.local_rank, 1)
        self.assertEqual(dist_manager.world_size, 32)

    def test_local_rank_defaults_to_global_rank_when_unset(self):
        with _with_rank_env(RANK="2", WORLD_SIZE="4"):
            dist_manager = DistributedManager()
        self.assertEqual(dist_manager.rank, 2)
        self.assertEqual(dist_manager.local_rank, 2)

    def test_single_process_defaults(self):
        with _with_rank_env():
            dist_manager = DistributedManager()
        self.assertEqual(dist_manager.rank, 0)
        self.assertEqual(dist_manager.local_rank, 0)
        self.assertEqual(dist_manager.world_size, 1)

    def test_device_index_follows_local_rank(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to check device placement")
        with _with_rank_env(RANK="13", LOCAL_RANK="0", WORLD_SIZE="32"):
            dist_manager = DistributedManager()
        self.assertEqual(dist_manager.device, torch.device("cuda", 0))


if __name__ == "__main__":
    unittest.main()
