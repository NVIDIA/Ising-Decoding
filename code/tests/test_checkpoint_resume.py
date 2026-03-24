# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint save/resume correctness.

Verifies:
  1. Epoch numbering: no off-by-one on resume (checkpoint stores next epoch)
  2. LR scheduler: no double-counting of steps on resume
  3. TensorBoard: no duplicate logging at the same epoch step
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.utils import save_checkpoint, load_checkpoint
from training.optimizers import get_lr_scheduler
from training.train import get_current_per_device_batch_size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model():
    """A trivially small model for fast test iteration."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _make_cfg(
    epochs=6,
    lr=1e-3,
    scheduler_type="warmup_then_decay",
    milestones=None,
    gamma=0.7,
    warmup_steps=0,
    batch_initial=8,
    batch_final=8,
    batch_schedule_enabled=False,
    batch_start_epoch=1,
    batch_end_epoch=2,
    accumulate_steps=1,
    num_samples=64,
    checkpoint_interval=1,
):
    if milestones is None:
        milestones = [0.25, 0.5]
    return OmegaConf.create(
        {
            "train":
                {
                    "epochs": epochs,
                    "num_samples": num_samples,
                    "accumulate_steps": accumulate_steps,
                    "checkpoint_interval": checkpoint_interval,
                },
            "optimizer": {
                "lr": lr
            },
            "lr_scheduler":
                {
                    "type": scheduler_type,
                    "milestones": milestones,
                    "gamma": gamma,
                    "warmup_steps": warmup_steps,
                },
            "batch_schedule":
                {
                    "enabled": batch_schedule_enabled,
                    "initial": batch_initial,
                    "final": batch_final,
                    "start_epoch": batch_start_epoch,
                    "end_epoch": batch_end_epoch,
                },
        }
    )


def _compute_total_steps(cfg, world_size=1):
    """Mirror the total_steps calculation from train.py."""
    total = 0
    for epoch in range(cfg.train.epochs):
        bs = get_current_per_device_batch_size(epoch, cfg)
        acc = cfg.train.accumulate_steps
        batches = cfg.train.num_samples // (bs * world_size)
        steps = max(1, math.ceil(batches / acc))
        total += steps
    return total


def _simulate_training_steps(model, optimizer, scheduler, num_steps):
    """Simulate `num_steps` optimizer steps (no real data needed)."""
    loss_fn = nn.MSELoss()
    for _ in range(num_steps):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointEpochNumbering(unittest.TestCase):
    """Verify that resume starts at the correct epoch (no off-by-one)."""

    def test_resume_starts_at_next_epoch(self):
        """After completing epoch N, checkpoint stores N+1 so resume skips N."""
        cfg = _make_cfg(epochs=10)
        total_steps = _compute_total_steps(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- First run: complete epochs 0, 1, 2 ---
            model = _tiny_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
            scheduler = get_lr_scheduler(cfg, optimizer, total_steps)

            steps_per_epoch = _compute_total_steps(
                OmegaConf.create(
                    {
                        **OmegaConf.to_container(cfg), "train":
                            {
                                **OmegaConf.to_container(cfg.train), "epochs": 1
                            }
                    }
                )
            )
            global_step = 0

            completed_epochs = 3
            for epoch in range(completed_epochs):
                _simulate_training_steps(model, optimizer, scheduler, steps_per_epoch)
                global_step += steps_per_epoch

            # Save checkpoint the way train.py now does: epoch + 1
            save_checkpoint(
                path=tmpdir,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=completed_epochs,  # epoch_number + 1 = 2 + 1 = 3
                global_step=global_step,
            )

            # --- Resume ---
            model2 = _tiny_model()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=cfg.optimizer.lr)
            scheduler2 = get_lr_scheduler(cfg, optimizer2, total_steps)

            init_epoch, loaded_global_step = load_checkpoint(
                tmpdir,
                models=model2,
                optimizer=optimizer2,
                scheduler=scheduler2,
                device="cpu",
            )

            # init_epoch should be 3 (the next epoch to run)
            self.assertEqual(init_epoch, completed_epochs)
            self.assertEqual(loaded_global_step, global_step)

            # The training loop `for epoch in range(init_epoch, total_epochs):`
            # should start at epoch 3, not re-run epoch 2
            resumed_epochs = list(range(init_epoch, cfg.train.epochs))
            self.assertEqual(resumed_epochs[0], completed_epochs)
            self.assertNotIn(completed_epochs - 1, resumed_epochs)


class TestSchedulerNoDoubleCount(unittest.TestCase):
    """Verify that the scheduler is NOT double-advanced on resume."""

    def test_scheduler_position_after_resume(self):
        """After loading checkpoint, scheduler.last_epoch == global_step (not 2x)."""
        cfg = _make_cfg(epochs=10, milestones=[0.25, 0.5], gamma=0.7)
        total_steps = _compute_total_steps(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- First run ---
            model = _tiny_model()
            opt = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
            sched = get_lr_scheduler(cfg, opt, total_steps)

            steps_per_epoch = _compute_total_steps(
                OmegaConf.create(
                    {
                        **OmegaConf.to_container(cfg), "train":
                            {
                                **OmegaConf.to_container(cfg.train), "epochs": 1
                            }
                    }
                )
            )
            global_step = 0

            for epoch in range(3):
                _simulate_training_steps(model, opt, sched, steps_per_epoch)
                global_step += steps_per_epoch

            # Record the LR and scheduler position before saving
            lr_before_save = sched.get_last_lr()[0]
            last_epoch_before_save = sched.last_epoch

            self.assertEqual(
                last_epoch_before_save, global_step,
                "Scheduler last_epoch should equal global_step after training"
            )

            save_checkpoint(
                path=tmpdir,
                models=model,
                optimizer=opt,
                scheduler=sched,
                epoch=3,
                global_step=global_step,
            )

            # --- Resume (correct way: load state_dict only, no replay) ---
            model2 = _tiny_model()
            opt2 = torch.optim.Adam(model2.parameters(), lr=cfg.optimizer.lr)
            sched2 = get_lr_scheduler(cfg, opt2, total_steps)

            _, loaded_global_step = load_checkpoint(
                tmpdir,
                models=model2,
                optimizer=opt2,
                scheduler=sched2,
                device="cpu",
            )

            # The scheduler should be at global_step, NOT 2 * global_step
            self.assertEqual(
                sched2.last_epoch, global_step,
                f"Scheduler should be at {global_step} but is at {sched2.last_epoch} "
                f"(double-count would give {2 * global_step})"
            )

            # LR should match what it was before saving
            lr_after_load = sched2.get_last_lr()[0]
            self.assertAlmostEqual(
                lr_after_load,
                lr_before_save,
                places=10,
                msg="LR after resume should match LR before checkpoint save"
            )

    def test_scheduler_milestones_hit_at_correct_step(self):
        """LR decay happens at the intended fraction of total_steps, not earlier."""
        total_steps = 100
        cfg = _make_cfg(epochs=10, milestones=[0.5], gamma=0.7, warmup_steps=0)

        # Milestone at step 50
        model = _tiny_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = get_lr_scheduler(cfg, opt, total_steps)

        base_lr = 1e-3

        # Advance to step 49 (just before milestone)
        for _ in range(49):
            sched.step()
        self.assertAlmostEqual(
            sched.get_last_lr()[0],
            base_lr,
            places=10,
            msg="LR should be unchanged before milestone"
        )

        # Step 50 (at milestone)
        sched.step()
        self.assertAlmostEqual(
            sched.get_last_lr()[0],
            base_lr * 0.7,
            places=10,
            msg="LR should decay by gamma at milestone"
        )

        # Now simulate resume from step 30
        with tempfile.TemporaryDirectory() as tmpdir:
            model2 = _tiny_model()
            opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            sched2 = get_lr_scheduler(cfg, opt2, total_steps)

            # Advance to step 30 and save
            for _ in range(30):
                sched2.step()
            save_checkpoint(
                path=tmpdir,
                optimizer=opt2,
                scheduler=sched2,
                epoch=3,
                global_step=30,
            )

            # Resume
            model3 = _tiny_model()
            opt3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
            sched3 = get_lr_scheduler(cfg, opt3, total_steps)

            load_checkpoint(
                tmpdir,
                optimizer=opt3,
                scheduler=sched3,
                device="cpu",
            )

            # Scheduler should be at step 30 — LR should still be base_lr
            self.assertEqual(sched3.last_epoch, 30)
            self.assertAlmostEqual(
                sched3.get_last_lr()[0],
                base_lr,
                places=10,
                msg="LR should be base_lr at step 30 (milestone is at 50)"
            )

            # Advance 20 more steps to reach milestone at step 50
            for _ in range(20):
                sched3.step()
            self.assertEqual(sched3.last_epoch, 50)
            self.assertAlmostEqual(
                sched3.get_last_lr()[0],
                base_lr * 0.7,
                places=10,
                msg="LR should decay at step 50 after resume (not before)"
            )

    def test_double_count_regression(self):
        """Explicitly verify the old bug: load_state_dict + replay = 2x steps."""
        total_steps = 100
        cfg = _make_cfg(milestones=[0.5], gamma=0.7, warmup_steps=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train to step 30, save
            model = _tiny_model()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = get_lr_scheduler(cfg, opt, total_steps)
            for _ in range(30):
                sched.step()
            save_checkpoint(path=tmpdir, scheduler=sched, epoch=3, global_step=30)

            # Simulate the OLD buggy resume: load state_dict THEN replay
            opt_buggy = torch.optim.Adam(_tiny_model().parameters(), lr=1e-3)
            sched_buggy = get_lr_scheduler(cfg, opt_buggy, total_steps)
            load_checkpoint(tmpdir, scheduler=sched_buggy, device="cpu")

            # After correct load: scheduler is at step 30
            self.assertEqual(sched_buggy.last_epoch, 30, "After load_state_dict, should be at 30")

            # Simulate old bug: replay global_step more times
            for _ in range(30):
                sched_buggy.step()

            # Bug: scheduler now thinks it's at step 60 — past the 50-step milestone!
            self.assertEqual(sched_buggy.last_epoch, 60, "Old bug would put scheduler at 60 (2x30)")
            self.assertAlmostEqual(
                sched_buggy.get_last_lr()[0],
                1e-3 * 0.7,
                places=10,
                msg="Old bug triggers premature LR decay (milestone 50 hit at real step 30)"
            )


class TestBatchScheduleWithEpochFix(unittest.TestCase):
    """Verify batch schedule uses correct epoch numbers after the fix."""

    def test_batch_size_schedule(self):
        cfg = _make_cfg(
            batch_schedule_enabled=True,
            batch_initial=8,
            batch_final=32,
            batch_start_epoch=1,
            batch_end_epoch=3,
        )
        # epoch 0: initial (8)
        self.assertEqual(get_current_per_device_batch_size(0, cfg), 8)
        # epoch 1: still initial (at start_epoch boundary)
        self.assertEqual(get_current_per_device_batch_size(1, cfg), 8)
        # epoch 2: midpoint
        bs2 = get_current_per_device_batch_size(2, cfg)
        self.assertGreater(bs2, 8)
        self.assertLess(bs2, 32)
        # epoch 3: boundary, should reach final
        bs3 = get_current_per_device_batch_size(3, cfg)
        self.assertEqual(bs3, 32)
        # epoch 4+: final
        self.assertEqual(get_current_per_device_batch_size(4, cfg), 32)

    def test_resumed_epoch_uses_correct_batch_size(self):
        """After resume from epoch 2, training should start epoch 2
        with the correct batch size (not re-use epoch 1's batch size)."""
        cfg = _make_cfg(
            epochs=6,
            batch_schedule_enabled=True,
            batch_initial=8,
            batch_final=32,
            batch_start_epoch=1,
            batch_end_epoch=3,
        )

        # Simulate: training completed epochs 0, 1; checkpoint saved with epoch=2
        init_epoch = 2  # The fix saves epoch_number + 1

        # The resumed loop: for epoch in range(init_epoch, cfg.train.epochs)
        resumed_epochs = list(range(init_epoch, cfg.train.epochs))
        self.assertEqual(resumed_epochs, [2, 3, 4, 5])

        # Epoch 2 should get the ramped batch size, NOT epoch 1's
        bs_epoch2 = get_current_per_device_batch_size(2, cfg)
        bs_epoch1 = get_current_per_device_batch_size(1, cfg)
        self.assertGreater(
            bs_epoch2, bs_epoch1, "Resumed epoch 2 should use a larger batch than epoch 1"
        )


class TestTensorBoardNoDuplicates(unittest.TestCase):
    """Verify that correct epoch numbering prevents duplicate TB entries."""

    def test_no_overlapping_epoch_logged(self):
        """Simulate two sequential runs and verify no epoch is logged twice."""
        cfg = _make_cfg(epochs=10)

        # Job 1: completes epochs 0, 1, 2 → saves checkpoint with epoch=3
        job1_logged_epochs = [0, 1, 2]
        saved_epoch = 3  # epoch_number + 1

        # Job 2: resumes from init_epoch=3
        init_epoch = saved_epoch
        job2_logged_epochs = list(range(init_epoch, cfg.train.epochs))

        # No overlap
        overlap = set(job1_logged_epochs) & set(job2_logged_epochs)
        self.assertEqual(overlap, set(), f"Epochs {overlap} would be logged twice!")

        # Together they cover all epochs
        all_epochs = sorted(job1_logged_epochs + job2_logged_epochs)
        self.assertEqual(all_epochs, list(range(cfg.train.epochs)))


class TestEndToEndMiniTraining(unittest.TestCase):
    """Full integration: train → save → resume → verify state consistency."""

    def test_full_save_resume_cycle(self):
        """Train a few epochs, save, resume, continue, and verify
        the global_step and scheduler are seamlessly continuous."""
        cfg = _make_cfg(
            epochs=6,
            milestones=[0.5],
            gamma=0.7,
            warmup_steps=0,
            num_samples=32,
            batch_initial=8,
        )
        total_steps = _compute_total_steps(cfg)
        steps_per_epoch = total_steps // cfg.train.epochs  # uniform (no batch schedule)

        with tempfile.TemporaryDirectory() as tmpdir:
            # === Phase 1: train epochs 0, 1, 2 ===
            model = _tiny_model()
            opt = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
            sched = get_lr_scheduler(cfg, opt, total_steps)
            global_step = 0

            for epoch in range(3):
                _simulate_training_steps(model, opt, sched, steps_per_epoch)
                global_step += steps_per_epoch

            sched_position_phase1 = sched.last_epoch
            lr_phase1 = sched.get_last_lr()[0]

            # Save (with the fix: epoch + 1)
            save_checkpoint(
                path=tmpdir,
                models=model,
                optimizer=opt,
                scheduler=sched,
                epoch=3,  # next epoch to run
                global_step=global_step,
            )

            # === Phase 2: resume and continue epochs 3, 4, 5 ===
            model2 = _tiny_model()
            opt2 = torch.optim.Adam(model2.parameters(), lr=cfg.optimizer.lr)
            sched2 = get_lr_scheduler(cfg, opt2, total_steps)

            init_epoch, loaded_gs = load_checkpoint(
                tmpdir,
                models=model2,
                optimizer=opt2,
                scheduler=sched2,
                device="cpu",
            )

            # Verify seamless state
            self.assertEqual(init_epoch, 3)
            self.assertEqual(loaded_gs, global_step)
            self.assertEqual(sched2.last_epoch, sched_position_phase1)
            self.assertAlmostEqual(sched2.get_last_lr()[0], lr_phase1, places=10)

            # Continue training
            global_step2 = loaded_gs
            for epoch in range(init_epoch, cfg.train.epochs):
                _simulate_training_steps(model2, opt2, sched2, steps_per_epoch)
                global_step2 += steps_per_epoch

            # Final global_step should equal total_steps
            self.assertEqual(
                global_step2, total_steps,
                f"After full training: global_step={global_step2} != total_steps={total_steps}"
            )
            self.assertEqual(sched2.last_epoch, total_steps)

    def test_multiple_resumes_no_drift(self):
        """Simulate 3 consecutive resume cycles; verify scheduler never drifts."""
        cfg = _make_cfg(
            epochs=9,
            milestones=[0.5],
            gamma=0.7,
            warmup_steps=0,
            num_samples=32,
            batch_initial=8,
        )
        total_steps = _compute_total_steps(cfg)
        steps_per_epoch = total_steps // cfg.train.epochs

        # Train in 3 jobs of 3 epochs each; use a single persistent dir so
        # checkpoints survive between iterations (TemporaryDirectory per iteration
        # would be deleted before the next job's load).
        global_step = 0
        init_epoch = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for job_idx in range(3):
                model = _tiny_model()
                opt = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
                sched = get_lr_scheduler(cfg, opt, total_steps)

                if job_idx > 0:
                    # Resume from the directory we last saved to
                    init_epoch, global_step = load_checkpoint(
                        tmpdir,
                        models=model,
                        optimizer=opt,
                        scheduler=sched,
                        device="cpu",
                    )

                # Verify no drift
                self.assertEqual(
                    sched.last_epoch, global_step,
                    f"Job {job_idx+1}: scheduler at {sched.last_epoch}, "
                    f"expected {global_step}"
                )

                # Train 3 epochs
                for epoch in range(init_epoch, init_epoch + 3):
                    _simulate_training_steps(model, opt, sched, steps_per_epoch)
                    global_step += steps_per_epoch

                # Save into the same dir for next job
                save_checkpoint(
                    path=tmpdir,
                    models=model,
                    optimizer=opt,
                    scheduler=sched,
                    epoch=init_epoch + 3,
                    global_step=global_step,
                )

        # After all 3 jobs: 9 epochs × steps_per_epoch = total_steps
        self.assertEqual(global_step, total_steps)


if __name__ == "__main__":
    unittest.main()
