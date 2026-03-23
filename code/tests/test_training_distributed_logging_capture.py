# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for training.distributed, training.logging, training.capture (no physicsnemo).
# Some tests are GPU-only (DistributedManager when CUDA required); CI runs CPU and GPU separately.
"""Tests for local distributed/logging/capture utilities."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch


class TestPythonLogger(unittest.TestCase):
    """Tests for training.logging.PythonLogger (CPU)."""

    def test_logger_creates_and_logs(self):
        from training.logging import PythonLogger
        log = PythonLogger("test_training_logging")
        log.info("info message")
        log.warning("warning message")
        log.success("success message")
        log.error("error message")


class TestStaticCapture(unittest.TestCase):
    """Tests for training.capture._StaticCapture (CPU)."""

    def test_state_dict_roundtrip(self):
        from training.capture import _StaticCapture
        d = _StaticCapture.state_dict()
        self.assertIn("amp_scalers", d)
        self.assertIn("amp_scaler_checkpoints", d)
        _StaticCapture.load_state_dict(d)
        _StaticCapture.load_state_dict(None)

    def test_load_state_dict(self):
        from training.capture import _StaticCapture
        _StaticCapture.load_state_dict(
            {
                "amp_scalers": {
                    "k": 1
                },
                "amp_scaler_checkpoints": {
                    "k": 2
                },
            }
        )
        d = _StaticCapture.state_dict()
        self.assertEqual(d["amp_scalers"], {"k": 1})
        self.assertEqual(d["amp_scaler_checkpoints"], {"k": 2})


class TestDistributedManagerNoGpu(unittest.TestCase):
    """DistributedManager without GPU uses CPU device (CPU CI)."""

    def test_init_uses_cpu_without_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            from training.distributed import DistributedManager
            dist = DistributedManager()
            self.assertEqual(dist.device.type, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "GPU required for DistributedManager device tests")
class TestDistributedManagerWithGpu(unittest.TestCase):
    """DistributedManager with GPU: device is cuda (GPU CI or local with GPU)."""

    def test_init_sets_cuda_device(self):
        from training.distributed import DistributedManager
        dist = DistributedManager()
        self.assertEqual(dist.device.type, "cuda")
        self.assertIn(dist.local_rank, (0, dist.rank))
