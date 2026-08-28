# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample accounting for the accurate d17 reference run on 32 ranks.

The reference run uses 8 nodes x 4 GPUs, 8,388,608 global samples per epoch,
gradient accumulation 2, and a per-rank batch schedule of 32 (epoch 0) then
128 (epoch 1 onward). This pins the resulting microbatch-iteration and
optimizer-update counts so a config or accounting change that silently
alters epoch length is caught.
"""

import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import get_current_per_device_batch_size
from training.utils import compute_batches_per_epoch

CONFIG_PATH = Path(__file__).parent.parent.parent / "conf" / "config_color_accurate_d17.yaml"
WORLD_SIZE = 32


class TestReferenceAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = OmegaConf.load(CONFIG_PATH)

    def _plan(self, epoch):
        per_device = get_current_per_device_batch_size(epoch, self.cfg)
        iterations = compute_batches_per_epoch(
            self.cfg.train.num_samples, per_device, WORLD_SIZE
        )
        updates = iterations // self.cfg.train.accumulate_steps
        return per_device, iterations, updates

    def test_epoch_zero_plan(self):
        per_device, iterations, updates = self._plan(0)
        self.assertEqual(per_device, 32)
        self.assertEqual(iterations, 8192)
        self.assertEqual(updates, 4096)

    def test_steady_state_plan(self):
        for epoch in (1, 2, 10):
            per_device, iterations, updates = self._plan(epoch)
            self.assertEqual(per_device, 128, f"epoch {epoch}")
            self.assertEqual(iterations, 2048, f"epoch {epoch}")
            self.assertEqual(updates, 1024, f"epoch {epoch}")

    def test_accumulation_windows_are_complete(self):
        """No tail window at either scheduled batch size."""
        for epoch in (0, 1):
            _, iterations, _ = self._plan(epoch)
            self.assertEqual(iterations % self.cfg.train.accumulate_steps, 0)

    def test_epoch_consumes_every_configured_sample(self):
        for epoch in (0, 1):
            per_device, iterations, _ = self._plan(epoch)
            self.assertEqual(
                iterations * per_device * WORLD_SIZE, self.cfg.train.num_samples
            )

    def test_superbatch_divides_both_scheduled_batch_sizes(self):
        superbatch_size = int(self.cfg.data.superbatch_size)
        for epoch in (0, 1):
            per_device, _, _ = self._plan(epoch)
            self.assertEqual(
                superbatch_size % per_device, 0,
                f"superbatch {superbatch_size} must divide per-rank batch {per_device}",
            )


if __name__ == "__main__":
    unittest.main()
