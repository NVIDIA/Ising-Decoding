# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MultiQCDataGeneratorTorch.

Covers the round-robin dispatch, code-family routing (surface vs color),
the `is_multi_pair`/`get_current_pair`/`get_generator_for_pair`
/`get_all_generators` contract, the per-pair `batch_size` list form,
and validation of bad arguments. Inner generators are mocked so this is a
CPU-only unit test of the manager class.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))


class _FakeInnerGen:
    """Stand-in for QCDataGeneratorTorch / ColorQCDataGeneratorTorch.

    Records every generate_batch call so tests can assert the manager passes
    the right (step, batch_size) through.
    """

    def __init__(self, tag, **kwargs):
        self.tag = tag
        self.init_kwargs = kwargs
        self.calls = []

    def generate_batch(self, step, batch_size, return_timing=False, **_):
        self.calls.append((int(step), int(batch_size), bool(return_timing)))
        return ("X", self.tag, step, batch_size)


def _patched_surface_factory(_calls):

    def _make(**kwargs):
        gen = _FakeInnerGen("surface", **kwargs)
        _calls.append(("surface", kwargs))
        return gen

    return _make


def _patched_color_factory(_calls):

    def _make(**kwargs):
        gen = _FakeInnerGen("color", **kwargs)
        _calls.append(("color", kwargs))
        return gen

    return _make


def _build(code, distances, rounds, **extra):
    """Construct MultiQCDataGeneratorTorch with the two inner factories mocked."""
    inner_calls = []
    with mock.patch(
        "data.generator_torch.QCDataGeneratorTorch",
        side_effect=_patched_surface_factory(inner_calls)
    ), mock.patch(
        "data.generator_torch_color.ColorQCDataGeneratorTorch",
        side_effect=_patched_color_factory(inner_calls),
    ):
        from data.generator_torch_multi import MultiQCDataGeneratorTorch
        gen = MultiQCDataGeneratorTorch(
            distances=distances,
            rounds=rounds,
            code=code,
            **extra,
        )
    return gen, inner_calls


def test_surface_dispatch_creates_qcdata_generator_torch_per_pair():
    distances = [5, 7, 9]
    rounds = [5, 7, 9]
    gen, inner_calls = _build("surface", distances, rounds)
    assert len(gen._gens) == 3
    assert [c[0] for c in inner_calls] == ["surface", "surface", "surface"]
    # Surface dispatch passes per-pair (distance, n_rounds).
    for (label, kwargs), d, r in zip(inner_calls, distances, rounds):
        assert label == "surface"
        assert kwargs["distance"] == d
        assert kwargs["n_rounds"] == r


def test_color_dispatch_requires_precomputed_frames_dir():
    """Color path can't build a DEM bundle on the fly; precomputed dir is required."""
    from data.generator_torch_multi import MultiQCDataGeneratorTorch
    with pytest.raises(ValueError, match="precomputed_frames_dir"):
        MultiQCDataGeneratorTorch(
            distances=[3],
            rounds=[3],
            code="color",
            precomputed_frames_dir=None,
        )


def test_color_dispatch_creates_color_generator_per_pair():
    distances = [3, 5]
    rounds = [3, 5]
    gen, inner_calls = _build(
        "color", distances, rounds, precomputed_frames_dir="/tmp/does-not-matter"
    )
    assert len(gen._gens) == 2
    assert [c[0] for c in inner_calls] == ["color", "color"]
    for (label, kwargs), d, r in zip(inner_calls, distances, rounds):
        assert label == "color"
        assert kwargs["distance"] == d
        assert kwargs["n_rounds"] == r


def test_round_robin_index_spends_two_steps_per_pair():
    """Step 0,1 -> pair 0; step 2,3 -> pair 1; step 4,5 -> pair 0 (cycles)."""
    gen, _ = _build("surface", [5, 7], [5, 7])
    assert [gen._index_for_step(s) for s in range(8)] == [0, 0, 1, 1, 0, 0, 1, 1]


def test_generate_batch_forwards_to_correct_inner_generator():
    gen, _ = _build("surface", [5, 7], [5, 7])
    out = gen.generate_batch(step=3, batch_size=16)
    # step=3 -> idx 1 (second pair, "7")
    assert out[1] == "surface"
    assert gen._gens[1].calls == [(0, 16, False)]
    assert gen._gens[0].calls == []
    # Local step counter is independent per inner generator.
    # step=7 -> idx 1 again (7//2=3, 3%2=1).
    gen.generate_batch(step=7, batch_size=16)
    assert gen._gens[1].calls == [(0, 16, False), (1, 16, False)]
    assert gen._gens[0].calls == []


def test_generate_batch_per_pair_batch_size_list():
    gen, _ = _build("surface", [5, 7], [5, 7])
    gen.generate_batch(step=0, batch_size=[8, 32])  # pair 0 -> 8
    gen.generate_batch(step=2, batch_size=[8, 32])  # pair 1 -> 32
    assert gen._gens[0].calls == [(0, 8, False)]
    assert gen._gens[1].calls == [(0, 32, False)]


def test_is_multi_pair_returns_true():
    gen, _ = _build("surface", [5, 7], [5, 7])
    assert gen.is_multi_pair() is True


def test_get_current_pair_tracks_round_robin():
    gen, _ = _build("surface", [5, 7, 9], [5, 7, 9])
    assert gen.get_current_pair(0) == (5, 5)
    assert gen.get_current_pair(2) == (7, 7)
    assert gen.get_current_pair(4) == (9, 9)
    assert gen.get_current_pair(6) == (5, 5)  # wraps


def test_get_generator_for_pair_returns_matching_or_raises():
    gen, _ = _build("surface", [5, 7], [5, 7])
    assert gen.get_generator_for_pair(7, 7) is gen._gens[1]
    with pytest.raises(ValueError, match="No generator for"):
        gen.get_generator_for_pair(11, 11)


def test_get_all_generators_returns_zipped_pairs_and_gens():
    gen, _ = _build("surface", [5, 7], [5, 7])
    pairs_and_gens = gen.get_all_generators()
    assert len(pairs_and_gens) == 2
    assert pairs_and_gens[0][0] == (5, 5)
    assert pairs_and_gens[1][0] == (7, 7)
    assert pairs_and_gens[0][1] is gen._gens[0]
    assert pairs_and_gens[1][1] is gen._gens[1]


def test_get_info_reports_mode_and_pair_list():
    gen, _ = _build("surface", [5, 7], [5, 7], mode="train")
    info = gen.get_info()
    assert info["mode"] == "train"
    assert info["num_pairs"] == 2
    assert info["pairs"] == [
        {
            "distance": 5,
            "n_rounds": 5
        },
        {
            "distance": 7,
            "n_rounds": 7
        },
    ]


def test_mismatched_or_empty_lengths_raise():
    from data.generator_torch_multi import MultiQCDataGeneratorTorch
    with pytest.raises(ValueError, match="same non-zero length"):
        MultiQCDataGeneratorTorch(distances=[5, 7], rounds=[5])
    with pytest.raises(ValueError, match="same non-zero length"):
        MultiQCDataGeneratorTorch(distances=[], rounds=[])
    with pytest.raises(TypeError, match="lists or tuples"):
        MultiQCDataGeneratorTorch(distances=5, rounds=5)


def test_noise_model_is_threaded_through_to_per_pair_generators():
    nm = object()  # opaque placeholder; the manager just forwards it
    gen, inner_calls = _build("surface", [5, 7], [5, 7], noise_model=nm)
    for label, kwargs in inner_calls:
        assert kwargs["noise_model"] is nm


def test_seed_offset_is_decorrelated_across_pairs():
    """Surface and color generators get distinct base seeds derived from idx."""
    gen, inner_calls = _build("surface", [5, 7, 9], [5, 7, 9], base_seed=42)
    seed_offsets = [kwargs["seed_offset"] for _label, kwargs in inner_calls]
    assert seed_offsets == [0, 10_000_000, 20_000_000]
    # Color version uses base_seed+offset directly (no seed_offset kwarg).
    gen, inner_calls = _build(
        "color",
        [3, 5],
        [3, 5],
        precomputed_frames_dir="/tmp/x",
        base_seed=42,
    )
    base_seeds = [kwargs["base_seed"] for _label, kwargs in inner_calls]
    assert base_seeds == [42, 42 + 10_000_000]
