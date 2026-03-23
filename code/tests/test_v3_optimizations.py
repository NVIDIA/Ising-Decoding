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
"""Tests for V3 optimization features in evaluation and training modules."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from evaluation.logical_error_rate import _get_env_bool, CUDAPrefetcher


# ---------------------------------------------------------------------------
# _get_env_bool
# ---------------------------------------------------------------------------
class TestGetEnvBool(unittest.TestCase):

    def test_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_get_env_bool("NONEXISTENT_VAR"))
            self.assertTrue(_get_env_bool("NONEXISTENT_VAR", True))

    def test_truthy_values(self):
        for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON", "anything"):
            with patch.dict(os.environ, {"TEST_VAR": val}):
                self.assertTrue(_get_env_bool("TEST_VAR"), f"Expected True for {val!r}")

    def test_falsy_values(self):
        for val in ("0", "false", "False", "FALSE", "no", "NO", "off", "OFF", ""):
            with patch.dict(os.environ, {"TEST_VAR": val}):
                self.assertFalse(_get_env_bool("TEST_VAR"), f"Expected False for {val!r}")

    def test_whitespace_stripped(self):
        with patch.dict(os.environ, {"TEST_VAR": "  false  "}):
            self.assertFalse(_get_env_bool("TEST_VAR"))
        with patch.dict(os.environ, {"TEST_VAR": "  1  "}):
            self.assertTrue(_get_env_bool("TEST_VAR"))


# ---------------------------------------------------------------------------
# CUDAPrefetcher (CPU fallback behavior)
# ---------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CUDAPrefetcher")
class TestCUDAPrefetcher(unittest.TestCase):

    def _make_loader(self, n_batches=3):
        """Simulate a DataLoader yielding dicts of tensors."""
        batches = []
        for i in range(n_batches):
            batches.append({
                "x": torch.randn(4, 8),
                "y": torch.randint(0, 2, (4,)),
            })
        return batches

    def test_iterates_all_batches(self):
        batches = self._make_loader(5)
        prefetcher = CUDAPrefetcher(batches, torch.device("cuda"))
        collected = list(prefetcher)
        self.assertEqual(len(collected), 5)

    def test_moves_to_cuda(self):
        original = self._make_loader(2)
        prefetcher = CUDAPrefetcher(original, torch.device("cuda"))
        for batch in prefetcher:
            self.assertTrue(batch["x"].is_cuda)
            self.assertTrue(batch["y"].is_cuda)

    def test_preserves_tensor_values(self):
        original = self._make_loader(2)
        prefetcher = CUDAPrefetcher(original, torch.device("cuda"))
        for got, expected in zip(prefetcher, original):
            self.assertTrue(torch.equal(got["x"].cpu(), expected["x"]))
            self.assertTrue(torch.equal(got["y"].cpu(), expected["y"]))

    def test_empty_loader(self):
        prefetcher = CUDAPrefetcher([], torch.device("cuda"))
        self.assertEqual(list(prefetcher), [])

    def test_single_batch(self):
        batches = self._make_loader(1)
        prefetcher = CUDAPrefetcher(batches, torch.device("cuda"))
        collected = list(prefetcher)
        self.assertEqual(len(collected), 1)


# ---------------------------------------------------------------------------
# Vectorized EMA update vs per-parameter fallback
# ---------------------------------------------------------------------------
class TestEMAUpdate(unittest.TestCase):

    def _make_models(self):
        m1 = nn.Linear(8, 4, bias=True)
        m2 = nn.Linear(8, 4, bias=True)
        with torch.no_grad():
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                p2.data.copy_(p1.data)
        return m1, m2

    def test_foreach_matches_loop(self):
        """Vectorized _foreach update must produce identical results to the per-parameter loop."""
        ema_decay = 0.999
        model, _ = self._make_models()
        ema_loop, _ = self._make_models()
        ema_vec, _ = self._make_models()

        # Initialize both EMA copies identically
        with torch.no_grad():
            for p_el, p_ev in zip(ema_loop.parameters(), ema_vec.parameters()):
                p_ev.data.copy_(p_el.data)

        # Simulate a model update
        with torch.no_grad():
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * 0.01)

        # Per-parameter loop (fallback)
        with torch.no_grad():
            for ema_param, param in zip(ema_loop.parameters(), model.parameters()):
                if ema_param.dtype.is_floating_point:
                    ema_param.data.mul_(ema_decay).add_(
                        param.data.to(ema_param.dtype), alpha=1.0 - ema_decay
                    )

        # Vectorized (V3)
        with torch.no_grad():
            ema_params = [p.data for p in ema_vec.parameters() if p.dtype.is_floating_point]
            model_params = [
                p.data.to(ep.dtype)
                for p, ep in zip(model.parameters(), ema_vec.parameters())
                if ep.dtype.is_floating_point
            ]
            torch._foreach_mul_(ema_params, ema_decay)
            torch._foreach_add_(ema_params, model_params, alpha=1.0 - ema_decay)

        for p_loop, p_vec in zip(ema_loop.parameters(), ema_vec.parameters()):
            self.assertTrue(
                torch.allclose(p_loop.data, p_vec.data, atol=1e-7),
                f"EMA mismatch: max diff={float((p_loop.data - p_vec.data).abs().max())}"
            )


# ---------------------------------------------------------------------------
# SDR threshold gate
# ---------------------------------------------------------------------------
class TestSDRThresholdGate(unittest.TestCase):

    def _gate(self, sdr_value, threshold, current_metric, best_vloss):
        """Replicate the SDR gate logic from train.py."""
        syndrome_qualifies = (sdr_value is not None and sdr_value >= threshold)
        sdr_not_computed = sdr_value is None
        is_improvement = (current_metric < best_vloss and (syndrome_qualifies or sdr_not_computed))
        return is_improvement

    def test_improvement_when_sdr_qualifies(self):
        self.assertTrue(
            self._gate(sdr_value=2.0, threshold=1.5, current_metric=0.1, best_vloss=0.2)
        )

    def test_no_improvement_when_sdr_below_threshold(self):
        self.assertFalse(
            self._gate(sdr_value=1.2, threshold=1.5, current_metric=0.1, best_vloss=0.2)
        )

    def test_no_improvement_when_metric_worse(self):
        self.assertFalse(
            self._gate(sdr_value=2.0, threshold=1.5, current_metric=0.3, best_vloss=0.2)
        )

    def test_bypass_when_sdr_not_computed(self):
        self.assertTrue(
            self._gate(sdr_value=None, threshold=1.5, current_metric=0.1, best_vloss=0.2)
        )

    def test_no_bypass_when_metric_worse_and_sdr_none(self):
        self.assertFalse(
            self._gate(sdr_value=None, threshold=1.5, current_metric=0.3, best_vloss=0.2)
        )

    def test_percent_mode_threshold(self):
        self.assertTrue(
            self._gate(sdr_value=40.0, threshold=33.3, current_metric=0.1, best_vloss=0.2)
        )
        self.assertFalse(
            self._gate(sdr_value=20.0, threshold=33.3, current_metric=0.1, best_vloss=0.2)
        )

    def test_exact_threshold_qualifies(self):
        self.assertTrue(
            self._gate(sdr_value=1.5, threshold=1.5, current_metric=0.1, best_vloss=0.2)
        )


if __name__ == "__main__":
    unittest.main()
