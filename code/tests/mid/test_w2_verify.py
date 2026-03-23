# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Verify weight-2 step-level density guarantee and full pipeline trainY weight.
# Runs in CI (CPU); uses test_homological_equivalence._make_dem_artifacts.

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch

from test_homological_equivalence import _make_dem_artifacts

# w2 step API may not exist in this branch
try:
    from qec.surface_code.homological_equivalence_torch import (
        _simplify_time_w2_step,
        build_weight2_timelike_cache,
    )
    _HAS_W2_STEP = True
except ImportError:
    _HAS_W2_STEP = False

from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch


@unittest.skipUnless(_HAS_W2_STEP, "w2 step API not in this branch")
class TestW2Verify(unittest.TestCase):
    """Verify weight-2 step density and full pipeline."""

    def test_w2_step_density_non_increasing(self):
        d, R = 5, 5
        H, p = _make_dem_artifacts(distance=d, n_rounds=R, rotation="XV", num_errors=64, seed=42)
        gen = MemoryCircuitTorch(
            distance=d,
            n_rounds=R,
            basis="X",
            code_rotation="XV",
            timelike_he=True,
            num_he_cycles=1,
            max_passes_w1=8,
            use_weight2=False,
            device=torch.device("cpu"),
            H=H,
            p=p,
            A=None,
        )
        B = 128
        D2 = d * d
        num_z = gen.parity_Z.shape[0]
        pZ_f = gen.parity_Z.float()
        cache_X = build_weight2_timelike_cache(
            gen.parity_X, gen.parity_Z, d, "X", torch.device("cpu")
        )
        torch.manual_seed(42)
        for trial in range(3):
            x_err = (torch.rand(B, D2, 2) > 0.85).float()
            sz = (torch.rand(B, num_z, 2) > 0.8).float()
            dens_before = x_err + torch.einsum("bst,sd->bdt", sz, pZ_f)
            total_before = dens_before.sum(dim=(1, 2))
            x_out, sz_out, _ = _simplify_time_w2_step(x_err, sz, pZ_f, cache_X)
            dens_after = x_out + torch.einsum("bst,sd->bdt", sz_out, pZ_f)
            total_after = dens_after.sum(dim=(1, 2))
            violations = (total_after > total_before + 1e-6).sum().item()
            self.assertEqual(violations, 0, f"trial {trial}: density should not increase")

    def test_full_pipeline_w2_reproducible(self):
        d, R = 5, 5
        H, p = _make_dem_artifacts(distance=d, n_rounds=R, rotation="XV", num_errors=64, seed=42)
        gen_w2 = MemoryCircuitTorch(
            distance=d,
            n_rounds=R,
            basis="X",
            code_rotation="XV",
            timelike_he=True,
            num_he_cycles=1,
            max_passes_w1=8,
            use_weight2=True,
            max_passes_w2=4,
            device=torch.device("cpu"),
            H=H,
            p=p,
            A=None,
        )
        B = 128
        torch.manual_seed(100)
        tX_a, tY_a = gen_w2.generate_batch(batch_size=B)
        torch.manual_seed(100)
        tX_b, tY_b = gen_w2.generate_batch(batch_size=B)
        self.assertTrue(torch.allclose(tX_a, tX_b), "trainX should match for same seed")
        self.assertTrue(torch.allclose(tY_a, tY_b), "trainY should match for same seed")
