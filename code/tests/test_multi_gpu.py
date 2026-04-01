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
"""
Multi-GPU unit tests (requires >=2 CUDA GPUs).

Uses torch.multiprocessing.spawn to launch 2 ranks and validates:
  - NCCL process group initialization and all_reduce
  - DDP forward + backward pass with the predecoder model
  - QCDataGeneratorTorch generating data on the correct device per rank

All classes are gated with @unittest.skipUnless(torch.cuda.device_count() >= 2).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _require_multi_gpu(cls):
    return unittest.skipUnless(torch.cuda.device_count() >= 2, "2 CUDA GPUs required")(cls)


def _setup_sys_path():
    """Ensure code/ is on sys.path inside spawned worker processes."""
    code_root = str(Path(__file__).resolve().parent.parent)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


# ---------------------------------------------------------------------------
# Worker functions executed inside spawned subprocesses
# ---------------------------------------------------------------------------


def _worker_nccl_allreduce(rank, world_size, init_file):
    """All_reduce sum: ranks hold (rank+1), result must equal world_size*(world_size+1)/2."""
    _setup_sys_path()
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    val = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2  # = 3.0 for world_size=2
    assert abs(val.item() - expected) < 1e-5, (
        f"Rank {rank}: all_reduce sum {val.item():.4f} != expected {expected:.4f}"
    )
    dist.destroy_process_group()


def _worker_ddp_step(rank, world_size, init_file):
    """DDP forward + backward; all parameter gradients must be finite."""
    _setup_sys_path()
    from types import SimpleNamespace
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model.predecoder import PreDecoderModelMemory_v1

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Small config to keep the test fast
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace(
        dropout_p=0.0,
        activation="relu",
        input_channels=4,
        out_channels=2,
        num_filters=[4, 2],
        kernel_size=[3, 3],
    )
    cfg.distance = 3
    cfg.n_rounds = 3

    model = PreDecoderModelMemory_v1(cfg).to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    torch.manual_seed(42)
    B, C, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    x = torch.randn(B, C, T, D, D, device=device)
    out = ddp_model(x)
    loss = out.sum()
    loss.backward()

    for name, param in ddp_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), (f"Rank {rank}: non-finite gradient in {name}")

    dist.destroy_process_group()


def _worker_data_generator(rank, world_size, init_file):
    """QCDataGeneratorTorch must place tensors on the rank's own GPU."""
    _setup_sys_path()
    from data.generator_torch import QCDataGeneratorTorch

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    gen = QCDataGeneratorTorch(
        distance=3,
        n_rounds=3,
        p_error=0.004,
        measure_basis="both",
        device=device,
        base_seed=42 + rank,
    )
    trainX, trainY = gen.generate_batch(step=0, batch_size=8)

    assert trainX.device.type == "cuda", f"Rank {rank}: trainX not on CUDA"
    assert trainX.device.index == rank, (
        f"Rank {rank}: trainX on GPU {trainX.device.index}, expected GPU {rank}"
    )
    assert torch.isfinite(trainX.float()).all(), f"Rank {rank}: non-finite values in trainX"
    assert torch.isfinite(trainY.float()).all(), f"Rank {rank}: non-finite values in trainY"

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helper: create a fresh rendezvous file path (must not exist before init)
# ---------------------------------------------------------------------------


def _fresh_rendezvous_file():
    fd, path = tempfile.mkstemp(suffix=".rendezvous")
    os.close(fd)
    os.remove(path)
    return path


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@_require_multi_gpu
class TestNCCLCommunication(unittest.TestCase):
    """NCCL process group initializes and all_reduce produces correct sums."""

    def test_allreduce_sum_two_gpus(self):
        init_file = _fresh_rendezvous_file()
        mp.spawn(_worker_nccl_allreduce, args=(2, init_file), nprocs=2, join=True)


@_require_multi_gpu
class TestDDPForwardBackward(unittest.TestCase):
    """DDP wraps PreDecoder; gradients must be finite after backward across 2 ranks."""

    def test_ddp_gradient_sync(self):
        init_file = _fresh_rendezvous_file()
        mp.spawn(_worker_ddp_step, args=(2, init_file), nprocs=2, join=True)


@_require_multi_gpu
class TestMultiGPUDataGenerator(unittest.TestCase):
    """QCDataGeneratorTorch places output tensors on the correct device per rank."""

    def test_per_rank_tensors_on_correct_device(self):
        init_file = _fresh_rendezvous_file()
        mp.spawn(_worker_data_generator, args=(2, init_file), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
