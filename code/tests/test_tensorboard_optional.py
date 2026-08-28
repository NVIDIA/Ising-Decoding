# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training must work when tensorboard is not installed.

`SummaryWriter` is imported under try/except and the writer is left as None
when the import fails, so every logging call site has to tolerate None. The
call sites predate that optional import and were guarded on rank only, which
made a run without tensorboard die partway through the first epoch:

    AttributeError: 'NoneType' object has no attribute 'add_scalar'

CI never caught this because every job that actually trains installs
requirements_public_train-cu*.txt, which pulls in tensorboard.

This drives train_epoch with tb_writer=None, so it fails on the unguarded
code and passes once the guards are in place.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import train_epoch


def _device():
    """Run on CUDA when present: the profiling path synchronizes the CUDA
    device whenever CUDA is available, regardless of the tensor device."""
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


class _TinyGenerator:
    """Minimal generator matching the train_epoch contract.

    Real generators emit batches already on the training device, so this does
    the same rather than relying on train_epoch to move them.
    """

    def __init__(self, batch_shape, label_shape, device):
        self.batch_shape = batch_shape
        self.label_shape = label_shape
        self.device = device

    def generate_batch(self, step, batch_size, return_timing=False, **kwargs):
        torch.manual_seed(step)
        x = torch.randn(batch_size, *self.batch_shape, device=self.device)
        y = torch.randint(0, 2, (batch_size, *self.label_shape), device=self.device).float()
        if return_timing:
            return x, y, {}
        return x, y


class TestTrainingWithoutTensorboard(unittest.TestCase):

    def test_train_epoch_runs_with_null_writer(self):
        device = _device()
        batch_size = 2
        steps_per_epoch = 4
        accumulate_steps = 2

        model = nn.Conv3d(2, 2, kernel_size=1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generator = _TinyGenerator((2, 2, 2, 2), (2, 2, 2, 2), device)

        avg_loss, global_step = train_epoch(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            cumulative_steps_before_epoch=0,
            epoch_number=0,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            tb_writer=None,  # the point of the test
            device=device,
            enable_fp16=False,
            rank=0,
            global_step=0,
            accumulate_steps=accumulate_steps,
        )

        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))
        # One optimizer update per accumulation window.
        self.assertEqual(global_step, steps_per_epoch // accumulate_steps)

    def test_train_epoch_runs_with_null_writer_while_profiling(self):
        """The profiling branch logs its own timing scalars — also guard those."""
        device = _device()
        model = nn.Conv3d(2, 2, kernel_size=1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generator = _TinyGenerator((2, 2, 2, 2), (2, 2, 2, 2), device)

        avg_loss, _ = train_epoch(
            generator=generator,
            steps_per_epoch=2,
            batch_size=2,
            cumulative_steps_before_epoch=0,
            epoch_number=0,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            tb_writer=None,
            device=device,
            enable_fp16=False,
            rank=0,
            global_step=0,
            accumulate_steps=1,
            profile_enabled=True,
            profile_log_every=1,
            profile_warmup_steps=0,
        )

        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))


if __name__ == "__main__":
    unittest.main()
