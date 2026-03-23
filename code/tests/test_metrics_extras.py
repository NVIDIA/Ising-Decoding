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
"""Tests for evaluation.metrics (configure_metrics, _extract_reduction_factor)."""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from evaluation.metrics import configure_metrics, _extract_reduction_factor


class TestConfigureMetrics(unittest.TestCase):

    def test_configure_metrics_returns_two_callables(self):
        a, b = configure_metrics(rank=0)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)

    def test_configure_metrics_rank_one_no_print(self):
        # Should not raise
        configure_metrics(rank=1)


class TestExtractReductionFactor(unittest.TestCase):

    def test_extract_from_dict_stim_x_z(self):
        result = {"stim": {"reduction factor (X)": 2.0, "reduction factor (Z)": 4.0}}
        self.assertEqual(_extract_reduction_factor(result), 3.0)

    def test_extract_from_dict_stim_x_only(self):
        result = {"stim": {"reduction factor (X)": 2.0}}
        self.assertEqual(_extract_reduction_factor(result), 2.0)

    def test_extract_from_dict_stim_reduction_factor_key(self):
        result = {"stim": {"reduction factor": 1.5}}
        self.assertEqual(_extract_reduction_factor(result), 1.5)

    def test_extract_from_float(self):
        self.assertEqual(_extract_reduction_factor(3.14), 3.14)

    def test_extract_from_int(self):
        self.assertEqual(_extract_reduction_factor(2), 2.0)

    def test_extract_from_empty_dict_returns_none(self):
        self.assertIsNone(_extract_reduction_factor({"stim": {}}))

    def test_extract_from_nested_stim(self):
        result = {"other": 1, "stim": {"reduction factor (X/Z)": 2.5}}
        self.assertEqual(_extract_reduction_factor(result), 2.5)
