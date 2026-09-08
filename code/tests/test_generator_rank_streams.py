# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RNG stream separation for the color Torch generator.

Multi-node training must give every global rank its own data stream. The
per-step seed therefore has to key off the *global* rank, not the node-local
one: with one process per GPU across N nodes, local ranks repeat on every
node, so a local-rank-keyed seed would make all nodes train on identical
samples.

Verifies:
  1. same base seed + global rank + step replays exactly
  2. different global ranks never share a step seed, including ranks that
     collide on their node-local index
  3. train and validation generators (different base seeds) are separate
  4. the X and Z basis streams within one generator are distinct
"""

import sys
import unittest
from pathlib import Path

import torch

# Ensure repo's code/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generator_torch_color import ColorQCDataGeneratorTorch

# Mirrors the train.py convention for the validation generator's base seed.
VAL_SEED_OFFSET = 100_000_000


def _seeder(*, base_seed, global_rank, mixed=True):
    """Build a generator shell exposing only the seeding logic under test."""
    gen = object.__new__(ColorQCDataGeneratorTorch)
    gen.base_seed = int(base_seed)
    gen.global_rank = int(global_rank)
    gen.rank_seed_stride = 1_000_003
    gen._step_seed_offset = (int(global_rank) + 1) * gen.rank_seed_stride
    gen._mixed = bool(mixed)
    gen.device = torch.device("cpu")
    return gen


class TestGeneratorRankStreams(unittest.TestCase):

    def test_same_rank_and_step_replays(self):
        first = _seeder(base_seed=42, global_rank=5)
        second = _seeder(base_seed=42, global_rank=5)
        for step in range(8):
            self.assertEqual(first._seed_for_step(step), second._seed_for_step(step))

    def test_different_global_ranks_never_share_a_stream(self):
        steps = range(64)
        seeds_by_rank = {
            rank: {_seeder(base_seed=42, global_rank=rank)._seed_for_step(s) for s in steps}
            for rank in range(32)
        }
        for rank_a, seeds_a in seeds_by_rank.items():
            for rank_b, seeds_b in seeds_by_rank.items():
                if rank_a < rank_b:
                    self.assertTrue(
                        seeds_a.isdisjoint(seeds_b),
                        f"global ranks {rank_a} and {rank_b} share step seeds",
                    )

    def test_ranks_colliding_on_local_index_stay_separate(self):
        # 8 nodes x 4 GPUs: local rank 0 appears as global ranks 0, 4, 8, ...
        local_zero_ranks = [node * 4 for node in range(8)]
        seeds = [
            _seeder(base_seed=42, global_rank=rank)._seed_for_step(0) for rank in local_zero_ranks
        ]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_train_and_validation_streams_are_distinct(self):
        train = _seeder(base_seed=42, global_rank=3)
        val = _seeder(base_seed=42 + VAL_SEED_OFFSET, global_rank=3)
        train_seeds = {train._seed_for_step(s) for s in range(256)}
        val_seeds = {val._seed_for_step(s) for s in range(256)}
        self.assertTrue(train_seeds.isdisjoint(val_seeds))

    def test_x_and_z_streams_are_distinct(self):

        class _BasisSim:

            def __init__(self, marker):
                self.marker = marker

            def generate_batch(self, batch_size, *, seed):
                x = torch.full((batch_size, 1), float(self.marker))
                return x, x

        gen = _seeder(base_seed=42, global_rank=1)
        gen.sim_X = _BasisSim(0)
        gen.sim_Z = _BasisSim(1)
        gen.sim = None

        even_x, _, _ = gen._generate_direct(0, 2, False)
        odd_x, _, _ = gen._generate_direct(1, 2, False)
        self.assertEqual(even_x[:, 0].tolist(), [0.0, 0.0])
        self.assertEqual(odd_x[:, 0].tolist(), [1.0, 1.0])
        self.assertNotEqual(gen._seed_for_step(0), gen._seed_for_step(1))


if __name__ == "__main__":
    unittest.main()


class TestStepBudget(unittest.TestCase):
    """The per-rank seed ranges are finite; a run must be told when it would leave them."""

    def _with_buffer(self, gen, superbatch_size, interleave=False):
        gen.measure_basis = "both" if gen._mixed else "X"
        gen.sim = gen.sim_X = gen.sim_Z = None
        gen._init_superbatching(
            superbatch_size=superbatch_size, prefetch=False, interleave_bases=interleave
        )
        return gen

    def test_direct_budget_is_one_stride(self):
        from data.generator_torch_color import RANK_STREAM_STRIDE
        gen = self._with_buffer(_seeder(base_seed=42, global_rank=3), 0)
        self.assertEqual(gen.max_generator_steps(), RANK_STREAM_STRIDE)
        gen.check_step_budget(RANK_STREAM_STRIDE)
        with self.assertRaises(ValueError):
            gen.check_step_budget(RANK_STREAM_STRIDE + 1)

    def test_interleaved_superbatch_budget_is_half_a_stride(self):
        from data.generator_torch_color import RANK_STREAM_STRIDE
        gen = self._with_buffer(
            _seeder(base_seed=42, global_rank=3, mixed=True), 8, interleave=True
        )
        self.assertEqual(gen.max_generator_steps(), RANK_STREAM_STRIDE // 2)
        gen.check_step_budget(RANK_STREAM_STRIDE // 2)
        with self.assertRaises(ValueError):
            gen.check_step_budget(RANK_STREAM_STRIDE // 2 + 1)

    def test_basis_local_superbatch_keeps_the_direct_budget(self):
        """Basis-local chunks reuse direct step seeds, so the budget is one stride."""
        from data.generator_torch_color import RANK_STREAM_STRIDE
        gen = self._with_buffer(_seeder(base_seed=42, global_rank=3, mixed=True), 8)
        self.assertEqual(gen.max_generator_steps(), RANK_STREAM_STRIDE)

    def test_single_basis_superbatch_keeps_the_direct_budget(self):
        from data.generator_torch_color import RANK_STREAM_STRIDE
        gen = self._with_buffer(_seeder(base_seed=42, global_rank=3, mixed=False), 8)
        self.assertEqual(gen.max_generator_steps(), RANK_STREAM_STRIDE)

    def test_the_guarded_boundary_is_a_real_collision(self):
        """One step past the budget, rank r's seed equals rank r+1's first seed."""
        from data.generator_torch_color import RANK_STREAM_STRIDE
        lo = _seeder(base_seed=42, global_rank=3)
        hi = _seeder(base_seed=42, global_rank=4)
        self.assertNotEqual(lo._seed_for_step(RANK_STREAM_STRIDE - 1), hi._seed_for_step(0))
        self.assertEqual(lo._seed_for_step(RANK_STREAM_STRIDE), hi._seed_for_step(0))
        # Interleaved builds: two seeds per start step, so the crossing is at half.
        half = RANK_STREAM_STRIDE // 2
        self.assertEqual(
            lo._seed_for_superbatch_basis(half, 1),
            hi._seed_for_superbatch_basis(0, 0),
        )
        self.assertNotEqual(
            lo._seed_for_superbatch_basis(half - 1, 1),
            hi._seed_for_superbatch_basis(0, 0),
        )

    def test_custom_stride_scales_the_budget_and_the_offsets(self):
        """Fewer ranks need more steps each; data.rank_seed_stride makes room."""
        gen = _seeder(base_seed=42, global_rank=3)
        gen.rank_seed_stride = 4_000_003
        gen._step_seed_offset = (3 + 1) * gen.rank_seed_stride
        gen.measure_basis = "both"
        gen.sim = gen.sim_X = gen.sim_Z = None
        gen._init_superbatching(superbatch_size=8, prefetch=False, interleave_bases=False)
        self.assertEqual(gen.max_generator_steps(), 4_000_003)
        gen.check_step_budget(1_650_688)  # 400 epochs on 16 ranks
        nxt = _seeder(base_seed=42, global_rank=4)
        nxt._step_seed_offset = (4 + 1) * 4_000_003
        self.assertEqual(gen._seed_for_step(4_000_003), nxt._seed_for_step(0))

    def test_reference_run_fits_the_budget(self):
        """400 epochs on 32 ranks: 8,192 + 399 x 2,048 = 825,344 generator steps.

        Fits the basis-local budget (one stride); it would exceed the interleaved
        budget (half a stride), which is one reason basis-local is the default.
        """
        steps = 8192 + 399 * 2048
        gen = self._with_buffer(_seeder(base_seed=42, global_rank=31, mixed=True), 1024)
        gen.check_step_budget(steps)
        gen = self._with_buffer(
            _seeder(base_seed=42, global_rank=31, mixed=True), 1024, interleave=True
        )
        with self.assertRaises(ValueError):
            gen.check_step_budget(steps)
