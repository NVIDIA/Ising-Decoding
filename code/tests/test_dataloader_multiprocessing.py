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
Multi-worker DataLoader tests for the Stim inference datapipe.

Verifies QCDataPipePreDecoder_Memory_inference is pickle-safe and correct
under num_workers > 0 with spawn multiprocessing (CPU-only, no GPU needed).
"""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch
from torch.utils.data import DataLoader

from data.datapipe_stim import QCDataPipePreDecoder_Memory_inference

_D, _T, _N, _BS, _W = 3, 3, 32, 8, 2


def _make_loader(basis, num_workers=_W, **kw):
    ds = QCDataPipePreDecoder_Memory_inference(
        distance=_D,
        n_rounds=_T,
        num_samples=_N,
        error_mode="circuit_level_surface_custom",
        p_error=0.01,
        measure_basis=basis,
        code_rotation="XV",
    )
    opts = dict(batch_size=_BS, shuffle=False)
    if num_workers > 0:
        opts["multiprocessing_context"] = "spawn"
    opts.update(kw)
    return ds, DataLoader(ds, num_workers=num_workers, **opts)


class TestMultiWorkerDataLoader(unittest.TestCase):

    def test_iteration_completes_all_bases(self):
        for basis in ("X", "Z", "both"):
            with self.subTest(basis=basis):
                _, loader = _make_loader(basis)
                total = sum(b["trainX"].shape[0] for b in loader)
                self.assertEqual(total, _N)

    def test_matches_single_worker_all_bases(self):
        for basis in ("X", "Z", "both"):
            with self.subTest(basis=basis):
                ds, _ = _make_loader(basis, num_workers=0)
                loader_0 = DataLoader(ds, batch_size=_BS, shuffle=False)
                loader_n = DataLoader(
                    ds,
                    batch_size=_BS,
                    shuffle=False,
                    num_workers=_W,
                    multiprocessing_context="spawn",
                )
                for b0, bn in zip(loader_0, loader_n):
                    for k in ("trainX", "x_syn_diff", "z_syn_diff", "dets_and_obs"):
                        torch.testing.assert_close(b0[k], bn[k])

    def test_persistent_workers_with_prefetch(self):
        _, loader = _make_loader("X", persistent_workers=True, prefetch_factor=2)
        total = sum(b["trainX"].shape[0] for b in loader)
        self.assertEqual(total, _N)


if __name__ == "__main__":
    unittest.main()
