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

import math
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from evaluation.logical_error_rate import _maybe_warmup_compile, _time_single_shot_latency_stim


class _FakeMatcher:

    def __init__(self) -> None:
        self.calls = []

    def decode(self, syndromes: np.ndarray) -> int:
        self.calls.append(np.asarray(syndromes))
        return 0


class TestInferenceLatencyTiming(unittest.TestCase):

    def test_time_single_shot_latency_scales_per_round(self) -> None:
        matcher = _FakeMatcher()
        baseline = np.zeros((5, 4), dtype=np.uint8)
        residual = np.zeros((3, 4), dtype=np.uint8)

        clock = {"t": 0.0}

        def _fake_perf_counter() -> float:
            clock["t"] += 1.0
            return clock["t"]

        with patch(
            "evaluation.logical_error_rate.time.perf_counter",
            side_effect=_fake_perf_counter,
        ), patch("evaluation.logical_error_rate.torch.cuda.is_available", return_value=False):
            baseline_us, predecoder_us = _time_single_shot_latency_stim(
                matcher=matcher,
                baseline_syndromes=baseline,
                residual_syndromes=residual,
                n_rounds=5,
                warmup_iterations=50,
            )

        expected_us_per_round = 1e6 / 5.0
        self.assertAlmostEqual(baseline_us, expected_us_per_round, places=6)
        self.assertAlmostEqual(predecoder_us, expected_us_per_round, places=6)

        n_samples = 3
        warmup_n = 3
        expected_decode_calls = 2 * warmup_n + 2 * n_samples
        self.assertEqual(len(matcher.calls), expected_decode_calls)

    def test_time_single_shot_latency_handles_empty(self) -> None:
        matcher = _FakeMatcher()
        baseline = np.zeros((0, 4), dtype=np.uint8)
        residual = np.zeros((0, 4), dtype=np.uint8)

        with patch("evaluation.logical_error_rate.torch.cuda.is_available", return_value=False):
            baseline_us, predecoder_us = _time_single_shot_latency_stim(
                matcher=matcher,
                baseline_syndromes=baseline,
                residual_syndromes=residual,
                n_rounds=5,
            )

        self.assertTrue(math.isnan(baseline_us))
        self.assertTrue(math.isnan(predecoder_us))
        self.assertEqual(len(matcher.calls), 0)


class TestMaybeWarmupCompile(unittest.TestCase):

    def _cpu_device(self):
        return torch.device("cpu")

    def test_calls_pipeline_module_when_compile_active(self):
        pipeline = MagicMock(return_value=torch.zeros(1))
        dets = np.zeros((4, 8), dtype=np.uint8)
        _maybe_warmup_compile(
            pipeline, dets, self._cpu_device(), trt_context=None, applied_compile=True
        )
        self.assertEqual(pipeline.call_count, 1)
        tensor_arg = pipeline.call_args[0][0]
        self.assertEqual(tensor_arg.shape[0], 1)
        self.assertEqual(tensor_arg.dtype, torch.float32)

    def test_skipped_when_compile_not_applied(self):
        pipeline = MagicMock()
        dets = np.zeros((4, 8), dtype=np.uint8)
        _maybe_warmup_compile(
            pipeline, dets, self._cpu_device(), trt_context=None, applied_compile=False
        )
        pipeline.assert_not_called()

    def test_skipped_when_trt_context_present(self):
        pipeline = MagicMock()
        dets = np.zeros((4, 8), dtype=np.uint8)
        _maybe_warmup_compile(
            pipeline, dets, self._cpu_device(), trt_context=object(), applied_compile=True
        )
        pipeline.assert_not_called()

    def test_cuda_sync_called_on_gpu_device(self):
        pipeline = MagicMock(return_value=torch.zeros(1))
        dets = np.zeros((4, 8), dtype=np.uint8)
        gpu_device = MagicMock(spec=torch.device)
        gpu_device.type = "cuda"
        with patch("evaluation.logical_error_rate.torch.as_tensor", return_value=torch.zeros(1)) as _mock_tensor, \
             patch("evaluation.logical_error_rate.torch.cuda.synchronize") as mock_sync:
            _maybe_warmup_compile(
                pipeline, dets, gpu_device, trt_context=None, applied_compile=True
            )
        mock_sync.assert_called_once()

    def test_cuda_sync_not_called_on_cpu_device(self):
        pipeline = MagicMock(return_value=torch.zeros(1))
        dets = np.zeros((4, 8), dtype=np.uint8)
        with patch("evaluation.logical_error_rate.torch.cuda.synchronize") as mock_sync:
            _maybe_warmup_compile(
                pipeline, dets, self._cpu_device(), trt_context=None, applied_compile=True
            )
        mock_sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
