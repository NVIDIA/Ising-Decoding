# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic Torch superbatch buffering."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import torch

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data.superbatch import SuperbatchBuffer  # noqa: E402


class _Builder:

    def __init__(self) -> None:
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, start_step, sample_count, collect_timing):
        with self.lock:
            self.calls.append((start_step, sample_count, collect_timing))
        values = (torch.arange(sample_count, dtype=torch.float32) + int(start_step) * 100)
        return values[:, None], (values + 0.5)[:, None], {"build": 1.0}


def test_resident_slices_and_prefetched_refill_follow_steps():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=4,
        prefetch=True,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        x0, y0 = buffer.generate(step=10, batch_size=2, return_timing=False, direct_fn=builder)
        x1, _ = buffer.generate(step=11, batch_size=2, return_timing=False, direct_fn=builder)
        x2, _ = buffer.generate(step=12, batch_size=2, return_timing=False, direct_fn=builder)

        assert x0[:, 0].tolist() == [1000.0, 1001.0]
        assert y0[:, 0].tolist() == [1000.5, 1001.5]
        assert x1[:, 0].tolist() == [1002.0, 1003.0]
        assert x2[:, 0].tolist() == [1200.0, 1201.0]
        assert builder.calls[:2] == [(10, 4, False), (12, 4, False)]
    finally:
        buffer.close()


def test_step_discontinuity_discards_stale_resident_data():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=4,
        prefetch=False,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        buffer.generate(step=3, batch_size=2, return_timing=False, direct_fn=builder)
        jumped, _ = buffer.generate(step=20, batch_size=2, return_timing=False, direct_fn=builder)
        assert jumped[:, 0].tolist() == [2000.0, 2001.0]
        assert [call[0] for call in builder.calls] == [3, 20]
    finally:
        buffer.close()


def test_disabled_buffer_preserves_direct_path_and_timing():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=0,
        prefetch=True,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    x, y, timing = buffer.generate(step=7, batch_size=3, return_timing=True, direct_fn=builder)
    assert x[:, 0].tolist() == [700.0, 701.0, 702.0]
    assert y[:, 0].tolist() == [700.5, 701.5, 702.5]
    assert timing == {"build": 1.0}
    assert builder.calls == [(7, 3, True)]


@pytest.mark.parametrize("batch_size", [3, 8])
def test_invalid_enabled_batch_size_fails_loudly(batch_size):
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=4,
        prefetch=False,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        with pytest.raises(ValueError, match="superbatch_size"):
            buffer.generate(
                step=0,
                batch_size=batch_size,
                return_timing=False,
                direct_fn=builder,
            )
    finally:
        buffer.close()


def test_timing_is_attributed_to_first_slice_only():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=4,
        prefetch=False,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        _, _, first = buffer.generate(step=0, batch_size=2, return_timing=True, direct_fn=builder)
        _, _, second = buffer.generate(step=1, batch_size=2, return_timing=True, direct_fn=builder)
        assert first["build"] == 1.0
        assert first["superbatch_generate_s"] >= 0
        assert first["superbatch_wait_s"] >= 0
        assert second["superbatch_wait_s"] == 0
        assert "build" not in second
    finally:
        buffer.close()


def test_color_superbatch_balances_mixed_bases_in_every_slice():
    from data.generator_torch_color import ColorQCDataGeneratorTorch

    class _BasisSim:

        def __init__(self, marker):
            self.marker = marker

        def generate_batch(self, batch_size, *, seed):
            del seed
            x = torch.full((batch_size, 1), self.marker)
            return x, x + 10

    generator = object.__new__(ColorQCDataGeneratorTorch)
    generator.base_seed = 7
    generator._step_seed_offset = 1_000_003
    generator._mixed = True
    generator.sim_X = _BasisSim(0)
    generator.sim_Z = _BasisSim(1)
    generator.device = torch.device("cpu")
    generator._superbatch_buffer = SuperbatchBuffer(
        superbatch_size=8,
        prefetch=False,
        device=generator.device,
        build_fn=generator._build_superbatch,
    )
    try:
        first_x, first_y = generator.generate_batch(0, 4)
        second_x, _ = generator.generate_batch(1, 4)
        direct_x, _ = generator.generate_direct_batch(1, 4)

        assert first_x[:, 0].tolist() == [0, 1, 0, 1]
        assert first_y[:, 0].tolist() == [10, 11, 10, 11]
        assert second_x[:, 0].tolist() == [0, 1, 0, 1]
        assert direct_x[:, 0].tolist() == [1, 1, 1, 1]
    finally:
        generator.close()


def test_color_superbatch_replay_is_deterministic():
    """Two identically seeded generators produce the same buffered sequence."""
    from data.generator_torch_color import ColorQCDataGeneratorTorch

    class _SeedSim:

        def generate_batch(self, batch_size, *, seed):
            g = torch.Generator().manual_seed(int(seed))
            x = torch.rand((batch_size, 1), generator=g)
            return x, x + 0.5

    def _make(superbatch_size):
        gen = object.__new__(ColorQCDataGeneratorTorch)
        gen.base_seed = 7
        gen._step_seed_offset = 1_000_003
        gen._mixed = False
        gen.sim = _SeedSim()
        gen.sim_X = gen.sim
        gen.sim_Z = gen.sim
        gen.device = torch.device("cpu")
        gen._superbatch_buffer = SuperbatchBuffer(
            superbatch_size=superbatch_size,
            prefetch=False,
            device=gen.device,
            build_fn=gen._build_superbatch,
        )
        return gen

    first = _make(8)
    second = _make(8)
    try:
        for step in range(4):
            ax, ay = first.generate_batch(step, 4)
            bx, by = second.generate_batch(step, 4)
            assert torch.equal(ax, bx)
            assert torch.equal(ay, by)
    finally:
        first.close()
        second.close()


def test_color_superbatch_slices_do_not_overlap():
    """Consecutive buffered steps return disjoint, ordered slices."""
    from data.generator_torch_color import ColorQCDataGeneratorTorch

    class _CountingSim:

        def generate_batch(self, batch_size, *, seed):
            base = torch.arange(batch_size, dtype=torch.float32) + int(seed) % 1000
            return base[:, None], base[:, None] + 0.5

    gen = object.__new__(ColorQCDataGeneratorTorch)
    gen.base_seed = 7
    gen._step_seed_offset = 1_000_003
    gen._mixed = False
    gen.sim = _CountingSim()
    gen.sim_X = gen.sim
    gen.sim_Z = gen.sim
    gen.device = torch.device("cpu")
    gen._superbatch_buffer = SuperbatchBuffer(
        superbatch_size=8,
        prefetch=False,
        device=gen.device,
        build_fn=gen._build_superbatch,
    )
    try:
        first, _ = gen.generate_batch(0, 4)
        second, _ = gen.generate_batch(1, 4)
        assert set(first[:, 0].tolist()).isdisjoint(second[:, 0].tolist())
        assert first[:, 0].tolist() == sorted(first[:, 0].tolist())
        assert second[:, 0].tolist() == sorted(second[:, 0].tolist())
    finally:
        gen.close()
