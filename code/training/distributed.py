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
Minimal distributed utilities to replace physicsnemo DistributedManager.
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.distributed as dist


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


class DistributedManager:
    """
    Lightweight wrapper around torch.distributed with a physicsnemo-like API.
    """

    @staticmethod
    def is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def initialize() -> None:
        if not dist.is_available() or dist.is_initialized():
            return
        world_size = _get_env_int("WORLD_SIZE", 1)
        rank = _get_env_int("RANK", 0)
        local_rank = _get_env_int("LOCAL_RANK", rank)
        if world_size <= 1:
            return
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                device_id=local_rank,
            )
        else:
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    def __init__(self) -> None:
        self.rank = _get_env_int("RANK", 0)
        self.world_size = _get_env_int("WORLD_SIZE", 1)
        local_rank = _get_env_int("LOCAL_RANK", self.rank)
        self.local_rank = local_rank
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")
        self.group_names: List[str] = []
        # Defaults used by DDP; keep configurable via env for compatibility.
        self.broadcast_buffers = bool(_get_env_int("PREDECODER_BROADCAST_BUFFERS", 1))
        self.find_unused_parameters = bool(_get_env_int("PREDECODER_FIND_UNUSED_PARAMETERS", 0))

    def group_rank(self, name: str) -> int:
        if not self.is_initialized():
            return 0
        return dist.get_rank()
