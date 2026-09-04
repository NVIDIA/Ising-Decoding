# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense augmented-DEM memory guard is a parameter, not a hard limit.

d = r = 17 needs a ~6 GiB dense H, above the 2 GiB default that was sized
for PID-scale configs. Large codes must be able to raise the ceiling
explicitly (API ``max_dense_bytes``, CLI ``--max_dense_gib``) while the
default keeps protecting small machines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from qec import precompute_dem  # noqa: E402


def _build(max_dense_bytes):
    return precompute_dem.precompute_dem_bundle_color_code(
        distance=5,
        n_rounds=2,
        basis="X",
        p_scalar=0.003,
        dem_output_dir=None,
        device=torch.device("cpu"),
        export=False,
        return_artifacts=True,
        max_dense_bytes=max_dense_bytes,
    )


def test_guard_trips_when_the_ceiling_is_below_the_dense_size():
    with pytest.raises(MemoryError, match="max_dense_gib"):
        _build(max_dense_bytes=1)


def test_default_ceiling_admits_a_pid_scale_config():
    artifacts = _build(max_dense_bytes=None)
    assert artifacts["H"].ndim == 2


def test_explicit_ceiling_above_the_dense_size_admits_the_build():
    artifacts = _build(max_dense_bytes=64 * 1024**3)
    assert artifacts["H"].ndim == 2


def test_cli_exposes_the_ceiling_in_gib(monkeypatch):
    seen = {}

    def fake_color(**kwargs):
        seen.update(kwargs)
        return Path(".")

    monkeypatch.setattr(precompute_dem, "precompute_dem_bundle_color_code", fake_color)
    monkeypatch.setattr(
        sys, "argv", [
            "precompute_dem", "--code", "color", "-d", "17", "-r", "17", "-b", "X", "--p", "0.004",
            "--max_dense_gib", "8", "--no_save"
        ]
    )
    precompute_dem.main()
    assert seen["max_dense_bytes"] == 8 * 1024**3
    assert seen["p_scalar"] == 0.004
