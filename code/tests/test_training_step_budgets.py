# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler accounting and seed budgets for fresh and resumed training."""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generator_torch_color import ColorQCDataGeneratorTorch
from training import train


def _config():
    return OmegaConf.load(
        Path(__file__).resolve().parents[2] / "conf/config_color_accurate_d17.yaml"
    )


def _generator(stride, *, interleave=False):
    gen = object.__new__(ColorQCDataGeneratorTorch)
    gen.rank_seed_stride = stride
    gen.device = torch.device("cpu")
    gen._mixed = True
    gen.measure_basis = "both"
    gen.sim = gen.sim_X = gen.sim_Z = None
    gen._init_superbatching(
        superbatch_size=1024 if interleave else 0,
        prefetch=False,
        interleave_bases=interleave,
    )
    return gen


def test_resume_counts_remaining_batches_but_keeps_full_scheduler():
    cfg = _config()
    scheduler_steps, train_steps, val_steps = train._calculate_step_budgets(cfg, 16, 300)
    assert scheduler_steps == 825_344
    assert train_steps == 409_600
    assert val_steps == 32
    _generator(1_000_003).check_step_budget(train_steps)


def test_fresh_run_still_rejects_an_exhausted_training_budget():
    scheduler_steps, train_steps, val_steps = train._calculate_step_budgets(_config(), 16, 0)
    assert (scheduler_steps, train_steps, val_steps) == (825_344, 1_650_688, 128)
    with pytest.raises(ValueError, match="1,650,688 generator steps"):
        _generator(1_000_003).check_step_budget(train_steps)


@pytest.mark.parametrize("init_epoch", [400, 401])
def test_finished_resume_has_no_generator_budget(init_epoch):
    assert train._calculate_step_budgets(_config(), 32, init_epoch) == (412_672, 0, 0)


def test_validation_can_exceed_its_budget_when_training_fits():
    cfg = _config()
    cfg.train.epochs = 1
    cfg.train.num_samples = 32_000
    cfg.val.num_samples = 131_072
    scheduler_steps, train_steps, val_steps = train._calculate_step_budgets(cfg, 2, 0)
    assert (scheduler_steps, train_steps, val_steps) == (250, 500, 2048)
    _generator(1000).check_step_budget(train_steps)
    with pytest.raises(ValueError, match="2,048 generator steps"):
        _generator(1000).check_step_budget(val_steps)


@pytest.mark.parametrize("samples, expected", [(0, 0), (1, 0), (64, 1), (65, 1), (66, 2)])
def test_validation_uses_rank_floor_then_batch_ceiling(samples, expected):
    cfg = _config()
    cfg.train.epochs = 1
    cfg.val.num_samples = samples
    assert train._calculate_step_budgets(cfg, 2, 0)[2] == expected


def test_validation_uses_direct_budget_and_does_not_sum_repeated_epochs():
    cfg = _config()
    cfg.train.epochs = 3
    cfg.batch_schedule.enabled = False
    cfg.batch_schedule.initial = 32
    cfg.val.num_samples = 9 * 32 * 2
    val_steps = train._calculate_step_budgets(cfg, 2, 0)[2]
    assert val_steps == 9
    _generator(9).check_step_budget(val_steps)
    with pytest.raises(ValueError):
        _generator(9, interleave=True).check_step_budget(val_steps)
