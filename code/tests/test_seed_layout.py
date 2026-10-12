# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``data.seed_layout.check_seed_layout``: rank seed ranges vs validation offset and 2**31.

``check_step_budget`` only checks one rank against its neighbour. Two further
ways for ``rank_seed_stride`` to make streams collide:

  * training ranges reaching the validation generator's offset
    (``stride=10_000_000``: training rank 10 == validation rank 0);
  * ranges wrapping at the 2**31 seed modulus (``stride=2**28``: ranks 0 and 8).

Both are caught here with ``world_size`` in hand, and the strides actually used
by the reference runs pass.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.seed_layout import (  # noqa: E402
    SEED_MODULUS,
    VALIDATION_SEED_OFFSET,
    check_seed_layout,
)


def _seed(base, offset, rank, stride, step=0):
    return (base + offset + (rank + 1) * stride + step) % SEED_MODULUS


class TestSeedLayout(unittest.TestCase):

    def test_reference_strides_pass(self):
        check_seed_layout(32, 1_000_003)  # 8-node reference run
        check_seed_layout(16, 4_000_003)  # 4-node continuation
        check_seed_layout(1, 1_000_003)

    def test_training_rank_aliasing_validation_rank_is_rejected(self):
        stride = 10_000_000
        # The collision the check must name: training rank 10 vs validation rank 0.
        self.assertEqual(_seed(42, 0, 10, stride), _seed(42, VALIDATION_SEED_OFFSET, 0, stride))
        with self.assertRaises(ValueError) as ctx:
            check_seed_layout(16, stride)
        self.assertIn("training rank 10", str(ctx.exception))
        # Ranks 0..9 end exactly at the validation offset: accepted. Rank 10 collides.
        check_seed_layout(10, stride)
        with self.assertRaises(ValueError):
            check_seed_layout(11, stride)

    def test_wraparound_at_the_seed_modulus_is_rejected(self):
        stride = 2**28
        self.assertEqual(_seed(42, 0, 0, stride), _seed(42, 0, 8, stride))
        with self.assertRaises(ValueError) as ctx:
            check_seed_layout(16, stride)
        msg = str(ctx.exception)
        # Such a stride breaks both rules; the message names both.
        self.assertIn("2**31", msg)
        self.assertIn("0 and 8", msg)
        self.assertIn("validation", msg)

    def test_modulus_boundary_is_exact(self):
        # Keep the validation rule satisfied (offset == world * stride) so only
        # the modulus rule is in play: span = (2 * world + 1) * stride.
        world = 4
        stride = (SEED_MODULUS - 1) // (2 * world + 1)
        check_seed_layout(world, stride, validation_seed_offset=world * stride)
        with self.assertRaises(ValueError) as ctx:
            check_seed_layout(world, stride + 1, validation_seed_offset=world * (stride + 1))
        self.assertIn("2**31", str(ctx.exception))
        self.assertNotIn("validation rank", str(ctx.exception))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            check_seed_layout(0, 1_000_003)
        with self.assertRaises(ValueError):
            check_seed_layout(4, 0)

    def test_generator_method_delegates(self):
        import torch  # noqa: F401  (generator module imports torch)
        from data.generator_torch_color import ColorQCDataGeneratorTorch
        gen = object.__new__(ColorQCDataGeneratorTorch)
        gen.rank_seed_stride = 10_000_000
        with self.assertRaises(ValueError):
            gen.check_seed_layout(world_size=16)
        gen.rank_seed_stride = 1_000_003
        gen.check_seed_layout(world_size=32)


if __name__ == "__main__":
    unittest.main()
