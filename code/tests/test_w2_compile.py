# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Quick test: exercise the compiled weight-2 path on CUDA at multiple sizes.
# Skipped in CI when no CUDA (unit-tests job is CPU-only).

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch

try:
    from qec.surface_code.homological_equivalence_torch import (
        _get_compiled_w2_loop,
        _simplify_time_w2_step_nobreak,
    )
    _HAS_W2_COMPILE = True
except ImportError:
    _HAS_W2_COMPILE = False


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for w2 compile test")
@unittest.skipUnless(_HAS_W2_COMPILE, "w2 compile API not in this branch")
class TestW2Compile(unittest.TestCase):
    """Exercise compiled weight-2 path on CUDA at multiple sizes."""

    def test_w2_loop_d5(self):
        self._run_test(d=5, B=32, num_passes=2)

    def test_w2_loop_d7(self):
        self._run_test(d=7, B=32, num_passes=2)

    def test_w2_loop_d9(self):
        self._run_test(d=9, B=32, num_passes=2)

    def test_w2_loop_d13(self):
        self._run_test(d=13, B=32, num_passes=2)

    def _run_test(self, d, B=32, num_passes=2):
        torch._dynamo.config.cache_size_limit = 64
        from qec.surface_code.homological_equivalence_torch import (
            _get_compiled_w2_loop,
            _simplify_time_w2_step_nobreak,
        )
        device = torch.device("cuda")
        Q = d * d
        S = (d - 1) * d // 2
        T = d + 1
        torch.manual_seed(42)
        x_work = (torch.rand(B, Q, T, device=device) > 0.9).float()
        z_work = (torch.rand(B, Q, T, device=device) > 0.9).float()
        sz_work = (torch.rand(B, S, T, device=device) > 0.8).float()
        sx_work = (torch.rand(B, S, T, device=device) > 0.8).float()
        pf_Z = torch.rand(S, Q, device=device).round()
        pf_X = torch.rand(S, Q, device=device).round()
        prs_Z = pf_Z.sum(dim=1)
        prs_X = pf_X.sum(dim=1)
        P = min(15, S)
        max_anti = 4
        q1 = torch.randint(0, Q, (P,), device=device)
        q2 = torch.randint(0, Q, (P,), device=device)
        anti_idx_X = torch.randint(0, S, (P, max_anti), device=device)
        anti_valid_X = torch.ones(P, max_anti, device=device)
        prs_anti_X = prs_Z[anti_idx_X.reshape(-1)].reshape(P, max_anti)
        anti_idx_Z = torch.randint(0, S, (P, max_anti), device=device)
        anti_valid_Z = torch.ones(P, max_anti, device=device)
        prs_anti_Z = prs_X[anti_idx_Z.reshape(-1)].reshape(P, max_anti)
        num_z_t = torch.tensor(S, device=device)
        num_x_t = torch.tensor(S, device=device)
        max_t = T - 1
        min_t_x, min_t_z = 0, 0
        compiled_w2 = _get_compiled_w2_loop(max_t, min_t_x, min_t_z, num_passes)
        xw, zw, szw, sxw = compiled_w2(
            x_work.clone(), z_work.clone(), sz_work.clone(), sx_work.clone(),
            pf_Z, prs_Z,
            q1, q2, anti_idx_X, anti_valid_X, prs_anti_X,
            pf_X, prs_X,
            q1, q2, anti_idx_Z, anti_valid_Z, prs_anti_Z,
            num_z_t, num_x_t,
        )
        torch.cuda.synchronize()
        self.assertEqual(xw.shape, (B, Q, T))
        self.assertFalse(torch.isnan(xw).any())
