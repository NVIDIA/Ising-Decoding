# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation's shared step offset does not consume the per-rank seed budget."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generator_torch_color import ColorQCDataGeneratorTorch
from data.seed_layout import VALIDATION_SEED_OFFSET
from training.train import validation_step


class _RecordingGenerator:

    def __init__(self, rank):
        self.steps = []
        self.seeds = []
        self.seeder = ColorQCDataGeneratorTorch.__new__(ColorQCDataGeneratorTorch)
        self.seeder.base_seed = 42 + VALIDATION_SEED_OFFSET
        self.seeder._step_seed_offset = (rank + 1) * 9

    def generate_batch(self, step, batch_size):
        self.steps.append(step)
        self.seeds.append(self.seeder._seed_for_step(step))
        values = torch.zeros(batch_size, 1, dtype=torch.float32)
        return values, values


@pytest.mark.parametrize("batch_count, expected_overlap", [(9, 0), (10, 1)])
def test_actual_validation_seed_window_at_rank_stride(batch_count, expected_overlap):
    generators = [_RecordingGenerator(rank) for rank in range(2)]
    for generator in generators:
        validation_step(
            generator=generator,
            model=torch.nn.Identity(),
            num_samples=2 * batch_count - 1,
            batch_size=2,
            device=torch.device("cpu"),
            enable_fp16=False,
            rank=1,  # Suppress progress output; generator ranks remain distinct.
        )
        assert generator.steps == list(range(10_000, 10_000 + batch_count))

    overlap = set(generators[0].seeds) & set(generators[1].seeds)
    assert len(overlap) == expected_overlap
