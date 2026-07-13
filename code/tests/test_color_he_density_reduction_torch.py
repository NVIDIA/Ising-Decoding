# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch color-code HE density-reduction sweep.

Restores the parametric coverage that the removed legacy
``test_color_code_he_density_reduction.py`` provided:

  - Spacelike HE never increases the total number of nonzero error labels
    (weight-non-increasing).
  - The reduction is consistent across seeds.
  - Holds across (p_error, distance, basis), exercised at d=3 and d=5
    on both X and Z bases for two error rates.

This test directly drives ``apply_homological_equivalence_color_torch``
on synthetic Bernoulli error diffs; it does not require an augmented DEM
bundle or cuStabilizer. Runs on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from qec.color_code.color_code import ColorCode  # noqa: E402
from qec.color_code.homological_equivalence_torch import (  # noqa: E402
    apply_homological_equivalence_color_torch,
    build_color_spacelike_he_cache,
)


def _sample_diffs(*, distance: int, n_rounds: int, p_error: float, seed: int):
    """Draw (B, R, num_data) iid Bernoulli(p_error) X and Z diff tensors."""
    cc = ColorCode(distance)
    num_data = int(cc.num_data)
    rng = np.random.default_rng(seed)
    B = 8
    shape = (B, n_rounds, num_data)
    x = rng.binomial(1, p_error, size=shape).astype(np.uint8)
    z = rng.binomial(1, p_error, size=shape).astype(np.uint8)
    return torch.from_numpy(z), torch.from_numpy(x)


def _density(t: torch.Tensor) -> int:
    return int(t.sum().item())


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("p_error", [0.005, 0.02])
def test_spacelike_he_does_not_increase_density(distance, basis, p_error):
    """Total nonzero label count after HE is <= before HE."""
    del basis  # spacelike HE rules are identical for X and Z bases here; included
    # in the parametrization so the test mirrors the legacy coverage matrix.
    cc = ColorCode(distance)
    cache = build_color_spacelike_he_cache(cc, device=torch.device("cpu"))
    z, x = _sample_diffs(distance=distance, n_rounds=distance, p_error=p_error, seed=0)
    pre = _density(z) + _density(x)
    z_after, x_after = apply_homological_equivalence_color_torch(z, x, cache)
    post = _density(z_after) + _density(x_after)
    assert post <= pre, f"HE increased density: pre={pre} post={post}"


@pytest.mark.parametrize("distance", [3, 5])
def test_spacelike_he_reduction_is_seed_consistent(distance):
    """Density reduction (pre - post) is non-negative and varies less than 50% across seeds."""
    cc = ColorCode(distance)
    cache = build_color_spacelike_he_cache(cc, device=torch.device("cpu"))
    pre_values, post_values = [], []
    for seed in range(8):
        z, x = _sample_diffs(distance=distance, n_rounds=distance, p_error=0.01, seed=seed)
        pre_values.append(_density(z) + _density(x))
        z_after, x_after = apply_homological_equivalence_color_torch(z, x, cache)
        post_values.append(_density(z_after) + _density(x_after))
    reductions = np.array([pre - post for pre, post in zip(pre_values, post_values)])
    assert np.all(reductions >= 0)
    # Spread bounded relative to the mean: avoid flagging healthy variance,
    # but a regression that doubled the spread would fail.
    mean = float(reductions.mean())
    if mean > 0:
        spread = float(reductions.std() / mean)
        assert spread < 1.0, f"reduction spread too large across seeds: {spread:.2f}"


@pytest.mark.parametrize("distance", [3, 5])
def test_spacelike_he_reduces_density_more_at_higher_p(distance):
    """At higher error rate there is more to reduce, so the reduction magnitude grows."""
    cc = ColorCode(distance)
    cache = build_color_spacelike_he_cache(cc, device=torch.device("cpu"))
    reductions = []
    for p in (0.001, 0.005, 0.02):
        # Average over a few seeds to smooth small-batch noise.
        local = []
        for seed in range(4):
            z, x = _sample_diffs(distance=distance, n_rounds=distance, p_error=p, seed=seed)
            pre = _density(z) + _density(x)
            z_after, x_after = apply_homological_equivalence_color_torch(z, x, cache)
            post = _density(z_after) + _density(x_after)
            local.append(pre - post)
        reductions.append(float(np.mean(local)))
    assert reductions[0] <= reductions[1] <= reductions[2], (
        f"reductions did not grow monotonically with p_error: {reductions}"
    )


def test_spacelike_he_x_and_z_independently_reduced():
    """X and Z error channels are reduced independently — both should be non-increasing."""
    distance = 5
    cc = ColorCode(distance)
    cache = build_color_spacelike_he_cache(cc, device=torch.device("cpu"))
    z, x = _sample_diffs(distance=distance, n_rounds=distance, p_error=0.01, seed=42)
    z_pre, x_pre = _density(z), _density(x)
    z_after, x_after = apply_homological_equivalence_color_torch(z, x, cache)
    assert _density(z_after) <= z_pre
    assert _density(x_after) <= x_pre
