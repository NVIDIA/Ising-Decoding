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
"""Regression tests for multi-GPU (DDP) device placement in the Torch color path.

These cover two device-placement bugs that made the color code training path crash
under DistributedDataParallel (it had only been exercised single-GPU before):

1. ``ColorSpacelikeHECache`` is built once on a single device, but under DDP each
   rank runs on its own GPU. Homological-equivalence ops (e.g. ``cfg @ parity.T``)
   then mixed the cache's device with the batch's device. The fix aligns the cache
   to the batch device inside ``_cache_on_device`` / ``simplify_color_batched_torch``.

2. The color data generator resolved a missing ``device`` to the *index-less*
   ``torch.device("cuda")``. Because data generation runs in a background prefetch
   thread and CUDA's current device is per-thread (defaulting to 0), freshly created
   tensors landed on ``cuda:0`` while cached tensors sat on the rank's GPU -> a
   ``torch.stack`` device mismatch. The fix resolves the default to a concrete
   ``torch.device("cuda", current_device())``.

The cache tests run on CPU/1-GPU; the cross-GPU and generator tests are gated on
GPU count so they no-op in CPU-only / single-GPU CI but exercise the real fix on
multi-GPU nodes.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import unittest
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qec.color_code.color_code import ColorCode
from qec.color_code.homological_equivalence_torch import (
    _cache_on_device,
    apply_homological_equivalence_color_torch,
    build_color_spacelike_he_cache,
)


def _setup_sys_path():
    """Ensure code/ is importable inside spawned worker processes."""
    code_root = str(Path(__file__).resolve().parents[1])
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def _diffs(distance, n_rounds, device, *, p=0.05, seed=0):
    cc = ColorCode(distance)
    rng = np.random.default_rng(seed)
    shape = (4, n_rounds, int(cc.num_data))
    z = torch.from_numpy(rng.binomial(1, p, size=shape).astype(np.uint8))
    x = torch.from_numpy(rng.binomial(1, p, size=shape).astype(np.uint8))
    return z.to(device), x.to(device)


class TestHECacheDeviceAlignment(unittest.TestCase):
    """Fix #1: the HE cache must be usable from a different device than it was built on."""

    def test_cache_on_device_is_noop_when_already_on_device(self):
        cache = build_color_spacelike_he_cache(ColorCode(3), device=torch.device("cpu"))
        # Same device -> the helper returns the original object (no copy, no regression
        # for single-GPU / CPU runs).
        self.assertIs(_cache_on_device(cache, torch.device("cpu")), cache)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cache_on_device_moves_every_tensor_field(self):
        cache = build_color_spacelike_he_cache(ColorCode(3), device=torch.device("cpu"))
        moved = _cache_on_device(cache, torch.device("cuda", 0))
        for f in dataclasses.fields(moved):
            v = getattr(moved, f.name)
            if torch.is_tensor(v):
                self.assertEqual(v.device.type, "cuda", f"field {f.name} not moved")
            elif isinstance(v, tuple) and len(v) > 0 and all(torch.is_tensor(t) for t in v):
                for i, t in enumerate(v):
                    self.assertEqual(t.device.type, "cuda", f"field {f.name}[{i}] not moved")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_apply_he_with_cpu_cache_and_cuda_input(self):
        # Cache on CPU, batch on cuda:0 -> pre-fix this raised a device-mismatch
        # RuntimeError in weight reduction's matmul. Post-fix it aligns and runs.
        cache = build_color_spacelike_he_cache(ColorCode(3), device=torch.device("cpu"))
        z, x = _diffs(3, 3, torch.device("cuda", 0))
        z_out, x_out = apply_homological_equivalence_color_torch(z, x, cache)
        self.assertEqual(z_out.device.type, "cuda")
        self.assertEqual(x_out.device.type, "cuda")
        self.assertEqual(z_out.shape, z.shape)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "2 CUDA GPUs required")
    def test_apply_he_with_cache_and_input_on_different_gpus(self):
        # The exact DDP scenario: cache built on cuda:0, batch on cuda:1.
        cache = build_color_spacelike_he_cache(ColorCode(3), device=torch.device("cuda", 0))
        z, x = _diffs(3, 3, torch.device("cuda", 1))
        z_out, x_out = apply_homological_equivalence_color_torch(z, x, cache)
        self.assertEqual(z_out.device.index, 1)
        self.assertEqual(x_out.device.index, 1)


# ---------------------------------------------------------------------------
# Fix #2: color data generator places tensors on the rank's GPU under DDP, even
# when generation happens in a background thread (per-thread CUDA current device).
# ---------------------------------------------------------------------------


def _worker_color_generator_device(rank, world_size, init_file, frames_dir):
    _setup_sys_path()
    from concurrent.futures import ThreadPoolExecutor

    from data.generator_torch_color import ColorQCDataGeneratorTorch

    dist.init_process_group(
        backend="nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)

    # device=None on purpose: exercises the default-device resolution. The generator
    # must pin a concrete cuda index so the background prefetch thread below does not
    # silently fall back to cuda:0.
    gen = ColorQCDataGeneratorTorch(
        distance=3,
        n_rounds=3,
        schedule="nearest-neighbor",
        measure_basis="both",
        precomputed_frames_dir=frames_dir,
        device=None,
        rank=rank,
        global_rank=rank,
        base_seed=123 + rank,
    )
    assert gen.device.index == rank, (
        f"Rank {rank}: generator device {gen.device} has no/!=rank index"
    )

    # Generate in a worker thread to mirror the training prefetch path (this is what
    # surfaced the per-thread device bug under DDP).
    with ThreadPoolExecutor(max_workers=1) as ex:
        trainX, trainY = ex.submit(gen.generate_batch, 0, 8).result()

    assert trainX.device.type == "cuda" and trainX.device.index == rank, (
        f"Rank {rank}: trainX on {trainX.device}, expected cuda:{rank}"
    )
    assert trainY.device.index == rank, f"Rank {rank}: trainY on {trainY.device}"
    assert torch.isfinite(trainX.float()).all(), f"Rank {rank}: non-finite trainX"
    dist.destroy_process_group()


def _fresh_rendezvous_file():
    fd, path = tempfile.mkstemp(suffix=".rendezvous")
    os.close(fd)
    os.remove(path)
    return path


@unittest.skipUnless(torch.cuda.device_count() >= 2, "2 CUDA GPUs required")
class TestMultiGPUColorGenerator(unittest.TestCase):
    """Color generator must place per-rank tensors on the rank's GPU under DDP."""

    def test_per_rank_color_tensors_on_correct_device(self):
        from qec.precompute_dem import precompute_dem_bundle_color_code

        with tempfile.TemporaryDirectory() as frames_dir:
            # Small d=3 augmented-DEM bundle for both bases (CPU precompute, fast).
            for basis in ("X", "Z"):
                precompute_dem_bundle_color_code(
                    distance=3,
                    n_rounds=3,
                    basis=basis,
                    schedule="nearest-neighbor",
                    p_scalar=0.004,
                    dem_output_dir=frames_dir,
                    device=torch.device("cpu"),
                    export=True,
                )
            init_file = _fresh_rendezvous_file()
            mp.spawn(
                _worker_color_generator_device,
                args=(2, init_file, frames_dir),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
