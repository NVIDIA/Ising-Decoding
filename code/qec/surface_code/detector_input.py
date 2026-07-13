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
"""Shared detector-input tensor preparation for surface-code predecoder paths."""

from __future__ import annotations

import torch

from qec._tensor_helpers import _grid_to_padded_stab, _one_hot_map
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
)


class SurfaceDetectorInputTransform(torch.nn.Module):
    """Build surface-code model inputs from flattened detector vectors."""

    def __init__(
        self,
        *,
        distance: int,
        rounds: int,
        basis: str,
        rotation: str = "XV",
        preprocess_strategy: str = "gather",
    ):
        super().__init__()
        self.distance = int(distance)
        self.rounds = int(rounds)
        self.basis = str(basis).upper()
        self.rotation = str(rotation).upper()
        self.preprocess_strategy = str(preprocess_strategy)
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be X or Z, got {basis!r}")
        if self.preprocess_strategy not in ("dense_matmul", "gather"):
            raise ValueError(f"Unsupported preprocess strategy: {preprocess_strategy!r}")

        self.height = self.distance
        self.width = self.distance
        self.num_data = self.distance * self.distance
        self.num_stabs = (self.num_data - 1) // 2
        self.num_main_dets = self.num_stabs * (2 * self.rounds - 1)
        self.detector_width = self.num_main_dets + self.num_stabs

        x_indices = compute_stabX_to_data_index_map(
            self.distance,
            self.rotation,
        ).to(dtype=torch.long)
        z_indices = compute_stabZ_to_data_index_map(
            self.distance,
            self.rotation,
        ).to(dtype=torch.long)
        grid_size = self.height * self.width
        self.register_buffer("x_to_grid", _one_hot_map(x_indices, grid_size), persistent=False)
        self.register_buffer("z_to_grid", _one_hot_map(z_indices, grid_size), persistent=False)
        self.register_buffer(
            "x_grid_to_stab",
            _grid_to_padded_stab(x_indices, grid_size),
            persistent=False,
        )
        self.register_buffer(
            "z_grid_to_stab",
            _grid_to_padded_stab(z_indices, grid_size),
            persistent=False,
        )

        x_present = normalized_weight_mapping_Xstab_memory(
            self.distance,
            self.rotation,
        ).reshape(1, self.height, self.width).repeat(self.rounds, 1, 1)
        z_present = normalized_weight_mapping_Zstab_memory(
            self.distance,
            self.rotation,
        ).reshape(1, self.height, self.width).repeat(self.rounds, 1, 1)
        if self.basis == "X":
            z_present[0] = 0.0
            z_present[-1] = 0.0
        else:
            x_present[0] = 0.0
            x_present[-1] = 0.0
        static_block = torch.stack(
            [
                x_present.to(dtype=torch.float32),
                z_present.to(dtype=torch.float32),
            ], dim=0
        ).unsqueeze(0)
        self.register_buffer("static_block", static_block, persistent=False)

        x_mask = torch.ones((1, self.num_stabs, self.rounds), dtype=torch.float32)
        z_mask = torch.ones((1, self.num_stabs, self.rounds), dtype=torch.float32)
        if self.basis == "X":
            z_mask[:, :, 0] = 0.0
            z_mask[:, :, -1] = 0.0
        else:
            x_mask[:, :, 0] = 0.0
            x_mask[:, :, -1] = 0.0
        self.register_buffer("x_mask", x_mask, persistent=False)
        self.register_buffer("z_mask", z_mask, persistent=False)

    def split_main_dets(self, dets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dets = dets.to(dtype=torch.float32)
        batch = dets.shape[0]
        first = dets[:, :self.num_stabs].reshape(batch, self.num_stabs, 1)
        zeros = torch.zeros_like(first)

        rest = dets[:, self.num_stabs:self.num_main_dets].reshape(
            batch,
            self.rounds - 1,
            2,
            self.num_stabs,
        )
        rest_x = rest[:, :, 0, :].permute(0, 2, 1).contiguous()
        rest_z = rest[:, :, 1, :].permute(0, 2, 1).contiguous()

        if self.basis == "X":
            x_first, z_first = first, zeros
        else:
            x_first, z_first = zeros, first

        x_syn = torch.cat([x_first, rest_x], dim=2) * self.x_mask
        z_syn = torch.cat([z_first, rest_z], dim=2) * self.z_mask
        return x_syn, z_syn

    def _map_syn_to_grid(
        self,
        syn: torch.Tensor,
        to_grid: torch.Tensor,
        grid_to_stab: torch.Tensor,
    ) -> torch.Tensor:
        syn_by_round = syn.permute(0, 2, 1).contiguous()
        if self.preprocess_strategy == "dense_matmul":
            return torch.matmul(syn_by_round, to_grid)

        zero = torch.zeros(
            syn_by_round.shape[0],
            syn_by_round.shape[1],
            1,
            dtype=syn_by_round.dtype,
            device=syn_by_round.device,
        )
        padded = torch.cat([zero, syn_by_round], dim=2)
        gather_index = grid_to_stab.view(1, 1, -1).expand(
            syn_by_round.shape[0],
            syn_by_round.shape[1],
            -1,
        )
        return torch.gather(padded, 2, gather_index)

    def boundary_dets(self, dets: torch.Tensor) -> torch.Tensor:
        return dets[:, self.num_main_dets:self.detector_width]

    def build_train_x(
        self,
        dets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dets = dets.to(dtype=torch.float32)
        x_syn, z_syn = self.split_main_dets(dets)

        x_grid = self._map_syn_to_grid(
            x_syn,
            self.x_to_grid,
            self.x_grid_to_stab,
        ).reshape(dets.shape[0], self.rounds, self.height, self.width)
        z_grid = self._map_syn_to_grid(
            z_syn,
            self.z_to_grid,
            self.z_grid_to_stab,
        ).reshape(dets.shape[0], self.rounds, self.height, self.width)

        dynamic = torch.stack([x_grid, z_grid], dim=1)
        static = self.static_block.expand(dets.shape[0], -1, -1, -1, -1)
        train_x = torch.cat([dynamic, static], dim=1).contiguous()
        return (
            train_x,
            x_syn.to(dtype=torch.int32),
            z_syn.to(dtype=torch.int32),
            self.boundary_dets(dets).to(dtype=torch.int32),
        )

    def forward(self, dets: torch.Tensor) -> torch.Tensor:
        train_x, _, _, _ = self.build_train_x(dets)
        return train_x


__all__ = ["SurfaceDetectorInputTransform"]
