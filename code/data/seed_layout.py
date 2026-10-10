# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout of the per-rank seed ranges shared by the training and validation generators.

Relative to the run's base seed, training rank ``r`` draws its step seeds from
``[(r+1)*S, (r+2)*S)`` with ``S = rank_seed_stride``; the validation generator
uses the same layout shifted by ``VALIDATION_SEED_OFFSET``. Seeds are reduced
modulo 2**31. ``check_seed_layout`` rejects strides for which those ranges
collide, which ``check_step_budget`` (one rank against its neighbour) cannot
see.
"""
from __future__ import annotations

# Added to the base seed for the validation generator (train.py convention).
VALIDATION_SEED_OFFSET = 100_000_000

SEED_MODULUS = 2**31


def check_seed_layout(
    world_size: int,
    rank_seed_stride: int,
    *,
    validation_seed_offset: int = VALIDATION_SEED_OFFSET,
) -> None:
    """Fail fast if ``world_size`` ranks with this stride would share seeds.

    Two independent conditions, both reported when both fail:

    * the highest training range must end before the lowest validation range
      begins, i.e. ``world_size * stride <= validation_seed_offset``;
      otherwise training rank ``offset // stride`` reuses validation rank 0's
      seeds (exactly when the offset is a multiple of the stride, overlapping
      ranges otherwise);
    * every range must stay below the 2**31 seed modulus, i.e.
      ``validation_seed_offset + (world_size + 1) * stride < 2**31``;
      otherwise the modular reduction folds a high rank onto a low one.
    """
    world_size = int(world_size)
    stride = int(rank_seed_stride)
    offset = int(validation_seed_offset)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if stride <= 0:
        raise ValueError("rank_seed_stride must be positive")

    problems = []
    if world_size * stride > offset:
        alias_rank = offset // stride
        problems.append(
            f"{world_size} ranks span {world_size * stride:,} seeds, past the validation "
            f"offset {offset:,}: training rank {alias_rank} would draw the same seeds as "
            f"validation rank 0 (need world_size * stride <= {offset:,})"
        )
    span = offset + (world_size + 1) * stride
    if span >= SEED_MODULUS:
        period = max(1, SEED_MODULUS // stride)
        problems.append(
            f"the ranges need {span:,} distinct seeds, more than the 2**31 seed modulus "
            f"({SEED_MODULUS:,}): ranks {period} apart (e.g. 0 and {period}) would wrap "
            "onto the same range"
        )
    if problems:
        raise ValueError(
            f"rank_seed_stride={stride:,} is not a valid seed layout for {world_size} ranks:\n  - "
            + "\n  - ".join(problems)
        )


__all__ = ["VALIDATION_SEED_OFFSET", "SEED_MODULUS", "check_seed_layout"]
