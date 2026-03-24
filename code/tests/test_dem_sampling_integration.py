# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Integration test: dem_sampling -> MemoryCircuitTorch.generate_batch pipeline.

When custab is present, runs with USE_CUSTAB=1 (cuST path); otherwise USE_CUSTAB=0
(torch fallback) so CI without GPU/custab can still execute the test.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure repo's code/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


class TestDEMSamplingIntegration(unittest.TestCase):
    """Full pipeline: precompute_dem -> MemoryCircuitTorch -> generate_batch using dem_sampling."""

    def test_memory_circuit_torch_generate_batch_uses_dem_sampling(self) -> None:
        """generate_batch runs with custab when available, else torch fallback; check shapes."""
        from qec.dem_sampling import _reset_use_custab_cache
        from qec.precompute_dem import precompute_dem_bundle_surface_code
        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch

        # Rely on default: use custab when available, else torch fallback. Clear any leftover env.
        _reset_use_custab_cache()
        os.environ.pop("USE_CUSTAB", None)
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            distance, n_rounds, batch_size = 3, 3, 4
            artifacts = precompute_dem_bundle_surface_code(
                distance=distance,
                n_rounds=n_rounds,
                basis="X",
                code_rotation="XV",
                p_scalar=0.01,
                dem_output_dir=None,
                device=device,
                export=False,
                return_artifacts=True,
            )
            mc = MemoryCircuitTorch(
                distance=distance,
                n_rounds=n_rounds,
                basis="X",
                H=artifacts["H"],
                p=artifacts["p"],
                A=artifacts["A"],
                device=device,
            )
            trainX, trainY = mc.generate_batch(batch_size=batch_size)
        finally:
            os.environ.pop("USE_CUSTAB", None)

        # trainX/trainY: (B, 4, R, D, D)
        self.assertEqual(trainX.shape, (batch_size, 4, n_rounds, distance, distance))
        self.assertEqual(trainY.shape, (batch_size, 4, n_rounds, distance, distance))
        self.assertEqual(trainX.dtype, torch.float32)
        self.assertEqual(trainY.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
