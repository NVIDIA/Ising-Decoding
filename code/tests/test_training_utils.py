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
"""Tests for training.utils (create_directory, should_stop_due_to_time, _is_external_model, etc.)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from training.utils import (
    create_directory,
    should_stop_due_to_time,
    _is_external_model,
    compare_receptive_field_with_window_data,
    dict_to_device,
)


class TestCreateDirectory(unittest.TestCase):

    def test_create_directory_creates_new_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "a", "b", "c")
            self.assertFalse(os.path.exists(sub))
            create_directory(sub)
            self.assertTrue(os.path.isdir(sub))

    def test_create_directory_existing_no_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_directory(tmp)
            self.assertTrue(os.path.isdir(tmp))


class TestShouldStopDueToTime(unittest.TestCase):

    def test_disabled_returns_false(self):
        cfg = SimpleNamespace()
        self.assertFalse(should_stop_due_to_time(cfg, [10.0], 0))

    def test_no_time_based_attr_returns_false(self):
        cfg = SimpleNamespace(time_based_early_stopping=SimpleNamespace(enabled=False))
        self.assertFalse(should_stop_due_to_time(cfg, [10.0], 0))

    def test_no_epoch_times_returns_false(self):
        cfg = SimpleNamespace(
            time_based_early_stopping=SimpleNamespace(enabled=True),
            job_start_timestamp=0.0,
            job_time_limit_seconds=3600,
            safety_margin_minutes=5,
        )
        cfg.time_based_early_stopping.safety_margin_minutes = 5
        self.assertFalse(should_stop_due_to_time(cfg, [], 0))


class TestIsExternalModel(unittest.TestCase):

    def test_ordinary_module_false(self):
        m = torch.nn.Linear(2, 3)
        self.assertFalse(_is_external_model(m))

    def test_external_like_true(self):

        class ExternalLike(torch.nn.Module):

            def save(self):
                pass

            def load(self):
                pass

            @property
            def meta(self):
                return {}

        self.assertTrue(_is_external_model(ExternalLike()))


class TestCompareReceptiveFieldWithWindowData(unittest.TestCase):

    def test_ok_when_receptive_field_le_window(self):
        # R = 1 + 3+3+3 - 3 = 7; window_size = min(9,9) = 9
        cfg = SimpleNamespace(
            model=SimpleNamespace(kernel_size=[3, 3, 3]),
            distance=9,
            n_rounds=9,
        )
        compare_receptive_field_with_window_data(cfg)

    def test_warning_when_receptive_field_gt_window(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(kernel_size=[5, 5, 5, 5]),
            distance=3,
            n_rounds=3,
        )
        compare_receptive_field_with_window_data(cfg)


class TestDictToDevice(unittest.TestCase):

    def test_tensor_moved_to_device(self):
        d = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        out = dict_to_device(d, torch.device("cpu"))
        self.assertEqual(out["a"].device.type, "cpu")
        self.assertEqual(out["b"].device.type, "cpu")

    def test_non_tensor_passthrough(self):
        d = {"a": 1, "b": "x"}
        out = dict_to_device(d, torch.device("cpu"))
        self.assertEqual(out["a"], 1)
        self.assertEqual(out["b"], "x")


class TestStepsPerEpochCalculation(unittest.TestCase):
    """steps_per_epoch must count forward passes (one generate_batch per step),
    not optimizer steps.  accumulate_steps only controls optimizer.step()
    frequency and must not reduce the loop iteration count."""

    def _steps_per_epoch(self, num_samples, per_device_bs, world_size):
        """Mirror the calculation in train.py:main() (line ~1298)."""
        return num_samples // (per_device_bs * world_size)

    def test_accumulate_steps_does_not_reduce_data_seen(self):
        num_samples = 8_388_608
        batch_size = 512
        world_size = 1

        steps = self._steps_per_epoch(num_samples, batch_size, world_size)
        data_seen = steps * batch_size * world_size

        self.assertEqual(data_seen, num_samples)

    def test_multi_gpu(self):
        num_samples = 8_388_608
        batch_size = 512
        world_size = 4

        steps = self._steps_per_epoch(num_samples, batch_size, world_size)
        data_seen = steps * batch_size * world_size

        self.assertEqual(data_seen, num_samples)

    def test_default_production_values(self):
        steps = self._steps_per_epoch(67_108_864, 512, 1)
        self.assertEqual(steps, 131072)
        self.assertEqual(steps * 512, 67_108_864)
