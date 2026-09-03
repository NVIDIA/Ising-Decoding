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


class _MarkerSim:
    """Emits a constant per basis so slices reveal their basis composition."""

    def __init__(self, marker):
        self.marker = marker

    def generate_batch(self, batch_size, *, seed):
        del seed
        x = torch.full((batch_size, 1), float(self.marker))
        return x, x + 10


class _SeedSim:
    """Output depends on seed and batch size: seeded rand."""

    def generate_batch(self, batch_size, *, seed):
        g = torch.Generator().manual_seed(int(seed))
        x = torch.rand((batch_size, 1), generator=g)
        return x, x + 0.5


class _RecordingSim:
    """Records every (marker, seed) the generator asks for."""

    def __init__(self, marker, log, lock):
        self.marker = marker
        self.log = log
        self.lock = lock

    def generate_batch(self, batch_size, *, seed):
        with self.lock:
            self.log.append((self.marker, int(seed)))
        x = torch.full((batch_size, 1), 0.0 if self.marker == "X" else 1.0)
        return x, x + 10


def _color_generator(*, sim_x, sim_z=None, superbatch_size, prefetch=False, interleave=False):
    """Generator shell exposing only the seeding/buffering logic under test."""
    from data.generator_torch_color import ColorQCDataGeneratorTorch

    gen = object.__new__(ColorQCDataGeneratorTorch)
    gen.base_seed = 7
    gen._step_seed_offset = 1_000_003
    gen.device = torch.device("cpu")
    if sim_z is None:
        gen._mixed = False
        gen.measure_basis = "X"
        gen.sim = sim_x
        gen.sim_X = sim_x
        gen.sim_Z = None
    else:
        gen._mixed = True
        gen.measure_basis = "both"
        gen.sim = None
        gen.sim_X = sim_x
        gen.sim_Z = sim_z
    gen._init_superbatching(
        superbatch_size=superbatch_size, prefetch=prefetch, interleave_bases=interleave
    )
    return gen


def test_basis_local_default_keeps_every_minibatch_single_basis():
    """Default mixed mode: even steps are all-X, odd steps all-Z, like direct."""
    gen = _color_generator(sim_x=_MarkerSim(0), sim_z=_MarkerSim(1), superbatch_size=8)
    try:
        for step in range(8):
            buffered, _ = gen.generate_batch(step, 4)
            direct, _ = gen.generate_direct_batch(step, 4)
            expected = [0.0] * 4 if step % 2 == 0 else [1.0] * 4
            assert buffered[:, 0].tolist() == expected, step
            assert torch.equal(buffered, direct), step
        assert set(gen._superbatch_buffers) == {"X", "Z"}
    finally:
        gen.close()


@pytest.mark.parametrize("batch_size", [2, 4])
def test_basis_local_superbatch_equal_to_batch_reproduces_direct_samples(batch_size):
    """superbatch_size == batch_size: buffered and direct runs are sample-identical."""
    buffered = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=batch_size)
    direct = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=0)
    try:
        for step in range(8):
            bx, by = buffered.generate_batch(step, batch_size)
            dx, dy = direct.generate_batch(step, batch_size)
            assert torch.equal(bx, dx), step
            assert torch.equal(by, dy), step
    finally:
        buffered.close()
        direct.close()


def test_basis_local_chunks_are_seeded_by_their_first_step():
    """Each chunk uses the direct seed of the first step it serves; no seed repeats."""
    log, lock = [], threading.Lock()
    gen = _color_generator(
        sim_x=_RecordingSim("X", log, lock),
        sim_z=_RecordingSim("Z", log, lock),
        superbatch_size=8,
    )
    try:
        for step in range(16):
            gen.generate_batch(step, 2)  # 4 slices per chunk -> chunk starts at X 0, 8; Z 1, 9
        expected = {
            ("X", gen._seed_for_step(0)),
            ("X", gen._seed_for_step(8)),
            ("Z", gen._seed_for_step(1)),
            ("Z", gen._seed_for_step(9)),
        }
        assert set(log) == expected
        assert len(log) == len(set(log))
    finally:
        gen.close()


@pytest.mark.parametrize("superbatch_size", [2, 4, 8])
def test_interleaved_superbatch_seeds_never_repeat_across_builds(superbatch_size):
    """Interleaved mode: each (basis, seed) pair is drawn once, including size == batch.

    With one minibatch per superbatch the start step advances by one per build;
    seeding the two bases with start_step and start_step + 1 made consecutive
    builds hand the same seed to the same simulator, so every sample was
    generated twice.
    """
    log, lock = [], threading.Lock()
    gen = _color_generator(
        sim_x=_RecordingSim("X", log, lock),
        sim_z=_RecordingSim("Z", log, lock),
        superbatch_size=superbatch_size,
        interleave=True,
    )
    try:
        for step in range(8):
            gen.generate_batch(step, 2)
        assert len(log) == len(set(log)), f"duplicate (basis, seed) draws: {log}"
        assert sum(1 for m, _ in log if m == "X") == sum(1 for m, _ in log if m == "Z")
    finally:
        gen.close()


def test_interleaved_superbatch_seeds_are_disjoint_from_direct_generation():
    log, lock = [], threading.Lock()
    gen = _color_generator(
        sim_x=_RecordingSim("X", log, lock),
        sim_z=_RecordingSim("Z", log, lock),
        superbatch_size=4,
        interleave=True,
    )
    try:
        for step in range(8):
            gen.generate_batch(step, 2)
        buffered = set(log)
        log.clear()
        for step in range(8):
            gen.generate_direct_batch(step, 2)
        assert buffered.isdisjoint(set(log))
    finally:
        gen.close()


def test_interleaved_superbatch_balances_mixed_bases_in_every_slice():
    """Opt-in interleaving: every slice is half X, half Z (unlike direct)."""
    gen = _color_generator(
        sim_x=_MarkerSim(0), sim_z=_MarkerSim(1), superbatch_size=8, interleave=True
    )
    try:
        first_x, first_y = gen.generate_batch(0, 4)
        second_x, _ = gen.generate_batch(1, 4)
        direct_x, _ = gen.generate_direct_batch(1, 4)

        assert first_x[:, 0].tolist() == [0, 1, 0, 1]
        assert first_y[:, 0].tolist() == [10, 11, 10, 11]
        assert second_x[:, 0].tolist() == [0, 1, 0, 1]
        assert direct_x[:, 0].tolist() == [1, 1, 1, 1]
        assert set(gen._superbatch_buffers) == {"ALL"}
    finally:
        gen.close()


def test_interleave_flag_is_ignored_for_a_single_basis():
    gen = _color_generator(sim_x=_SeedSim(), superbatch_size=8, interleave=True)
    try:
        assert gen._superbatch_interleave is False
        assert set(gen._superbatch_buffers) == {"ALL"}
    finally:
        gen.close()


def test_color_superbatch_replay_is_deterministic():
    """Two identically seeded generators produce the same buffered sequence."""
    first = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=8)
    second = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=8)
    try:
        for step in range(8):
            ax, ay = first.generate_batch(step, 4)
            bx, by = second.generate_batch(step, 4)
            assert torch.equal(ax, bx)
            assert torch.equal(ay, by)
    finally:
        first.close()
        second.close()


def test_color_superbatch_slices_do_not_overlap():
    """Consecutive buffered steps of one basis return disjoint, ordered slices."""

    class _CountingSim:

        def generate_batch(self, batch_size, *, seed):
            base = torch.arange(batch_size, dtype=torch.float32) + int(seed) % 1000
            return base[:, None], base[:, None] + 0.5

    gen = _color_generator(sim_x=_CountingSim(), superbatch_size=8)
    try:
        first, _ = gen.generate_batch(0, 4)
        second, _ = gen.generate_batch(1, 4)
        assert set(first[:, 0].tolist()).isdisjoint(second[:, 0].tolist())
        assert first[:, 0].tolist() == sorted(first[:, 0].tolist())
        assert second[:, 0].tolist() == sorted(second[:, 0].tolist())
    finally:
        gen.close()


def test_color_release_reprimes_both_bases_across_an_epoch_boundary():
    """release(keep_pending) between epochs: both basis buffers continue the sequence."""
    resumed = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=8, prefetch=True)
    straight = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=8)
    try:
        for step in range(8):  # epoch 0: fills X (steps 0,2,4,6) and Z (1,3,5,7) once each
            resumed.generate_batch(step, 2)
            straight.generate_batch(step, 2)
        resumed.release_superbatch(wait=True, keep_pending=True)
        for buffer in resumed._superbatch_buffers.values():
            assert buffer._resident is None
        for step in range(8, 16):  # epoch 1 continues the step sequence
            rx, ry = resumed.generate_batch(step, 2)
            sx, sy = straight.generate_batch(step, 2)
            assert torch.equal(rx, sx), step
            assert torch.equal(ry, sy), step
    finally:
        resumed.close()
        straight.close()


def test_color_release_without_keep_pending_then_reprime_matches_direct():
    """A hard release (no pending kept) re-primes from the requested step."""
    gen = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=4)
    direct = _color_generator(sim_x=_SeedSim(), sim_z=_SeedSim(), superbatch_size=0)
    try:
        for step in range(4):
            gen.generate_batch(step, 4)
        gen.release_superbatch(wait=True)
        # superbatch_size == batch_size here, so values must equal direct's.
        for step in range(4, 8):
            gx, _ = gen.generate_batch(step, 4)
            dx, _ = direct.generate_batch(step, 4)
            assert torch.equal(gx, dx), step
    finally:
        gen.close()
        direct.close()


def test_release_keep_pending_reuses_prefetched_build_at_epoch_boundary():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=4,
        prefetch=True,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        buffer.generate(step=0, batch_size=2, return_timing=False, direct_fn=builder)
        buffer.generate(step=1, batch_size=2, return_timing=False, direct_fn=builder)
        # Epoch boundary: drop resident data, keep the prefetch for step 2.
        buffer.release(wait=True, keep_pending=True)
        assert buffer._resident is None
        assert buffer._future_key == (2, 2)

        x, _ = buffer.generate(step=2, batch_size=2, return_timing=False, direct_fn=builder)
        assert x[:, 0].tolist() == [200.0, 201.0]
        # No rebuild: the builds are the initial one and the single prefetch
        # for step 2 (plus the prefetch for step 4 that serving step 2 queues).
        assert [c[0] for c in builder.calls][:2] == [0, 2]
        assert builder.calls.count((2, 4, False)) == 1
    finally:
        buffer.close()


def test_release_keep_pending_rebuilds_when_batch_size_changes():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=8,
        prefetch=True,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        for step in range(4):
            buffer.generate(step=step, batch_size=2, return_timing=False, direct_fn=builder)
        buffer.release(wait=True, keep_pending=True)
        assert buffer._future_key == (4, 2)

        # Batch schedule moves 2 -> 4: the pending build no longer matches.
        x, _ = buffer.generate(step=4, batch_size=4, return_timing=False, direct_fn=builder)
        assert x[:, 0].tolist() == [400.0, 401.0, 402.0, 403.0]
        assert buffer._resident is not None and buffer._resident.batch_size == 4
    finally:
        buffer.close()


def test_release_without_keep_pending_drops_the_prefetch():
    builder = _Builder()
    buffer = SuperbatchBuffer(
        superbatch_size=4,
        prefetch=True,
        device=torch.device("cpu"),
        build_fn=builder,
    )
    try:
        buffer.generate(step=0, batch_size=2, return_timing=False, direct_fn=builder)
        buffer.release(wait=True)
        assert buffer._resident is None and buffer._future is None and buffer._future_key is None
    finally:
        buffer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_buffer_hands_producer_stream_data_to_the_consumer():
    """The build runs on the producer stream; the consumer must see complete data."""
    device = torch.device("cuda", torch.cuda.current_device())
    seen_streams = []

    def build(start_step, sample_count, collect_timing):
        seen_streams.append(torch.cuda.current_stream(device))
        # A long dependent chain on the producer stream, so a missing event
        # wait would let the consumer read unfinished values.
        values = torch.full((sample_count, 1 << 16), float(start_step), device=device)
        for _ in range(64):
            values = torch.sin(values) + 1.0
        # The sin chain converges to a fixed point; the start step is added
        # afterwards so builds stay distinguishable.
        marker = values.mean(dim=1, keepdim=True) + float(start_step)
        return marker, marker + 0.5, {}

    buffer = SuperbatchBuffer(superbatch_size=8, prefetch=True, device=device, build_fn=build)
    try:
        x0, y0 = buffer.generate(step=0, batch_size=4, return_timing=False, direct_fn=build)
        assert buffer._resident is not None and buffer._resident.ready_event is not None
        # Consumer-side reference computed on the default stream.
        ref = torch.full((1, 1 << 16), 0.0, device=device)
        for _ in range(64):
            ref = torch.sin(ref) + 1.0
        ref = ref.mean() + 0.0
        torch.cuda.synchronize(device)
        assert torch.allclose(x0[:, 0], ref.expand(4), atol=1e-6)
        assert torch.allclose(y0[:, 0] - x0[:, 0], torch.full((4,), 0.5, device=device))
        assert seen_streams and seen_streams[0] == buffer._producer_stream
        assert seen_streams[0] != torch.cuda.default_stream(device)
        # Slices from the same build and from the prefetched next build.
        x1, _ = buffer.generate(step=1, batch_size=4, return_timing=False, direct_fn=build)
        x2, _ = buffer.generate(step=2, batch_size=4, return_timing=False, direct_fn=build)
        torch.cuda.synchronize(device)
        assert torch.equal(x1, x0)
        assert not torch.equal(x2, x0)
    finally:
        buffer.close()
