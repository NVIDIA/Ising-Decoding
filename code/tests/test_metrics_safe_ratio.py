# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from pathlib import Path
import sys

# Ensure repo's code/ is on sys.path when running via unittest discovery
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import _safe_ratio


class TestMetricsSafeRatio(unittest.TestCase):
    def test_safe_ratio_zero_over_zero_is_one(self):
        self.assertEqual(_safe_ratio(0, 0), 1.0)
        self.assertEqual(_safe_ratio(0.0, 0.0), 1.0)

    def test_safe_ratio_positive_over_zero_is_inf(self):
        v = _safe_ratio(1, 0)
        self.assertTrue(v == float("inf"))

    def test_safe_ratio_normal(self):
        self.assertAlmostEqual(_safe_ratio(2, 4), 0.5)


if __name__ == "__main__":
    unittest.main()

