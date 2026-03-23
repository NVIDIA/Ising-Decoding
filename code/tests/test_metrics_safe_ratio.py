# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
