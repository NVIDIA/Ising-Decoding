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
"""Canonical transformation from Stim detector samples to pre-decoder inputs.

This is the single source of truth for converting Stim detector bits into the
``(trainX, x_syn_diff, z_syn_diff)`` tuple that the pre-decoder consumes. Both
the file-based datapipe and the offline reference path use this helper so the
transformation cannot drift between callers.

The GPU-resident :class:`PreDecoderMemoryEvalModule` keeps its own buffer-fused
forward for ONNX/TensorRT export, but its algorithm must match the helper bit
for bit; ``test_offline_stim_decoding.py`` asserts that parity.

Input contract
--------------
- ``dets``: ``(B, 2 * T * half)`` uint8/int tensor where ``half = (D*D - 1)//2``
  and detectors are stored in Stim's emission order, two ``half``-sized groups
  per round (X-stab group followed by Z-stab group).
- ``basis``: ``'X'`` or ``'Z'`` — the memory basis being decoded.
- ``code_rotation``: one of ``'XV'``, ``'XH'``, ``'ZV'``, ``'ZH'``.

Output
------
- ``trainX``: ``(B, 4, T, D, D)`` float32 — channels are
  ``[x_syn_grid, z_syn_grid, x_present, z_present]``.
- ``x_syn_diff``, ``z_syn_diff``: ``(B, half, T)`` int32 — boundary-masked
  syndrome differences, suitable for residual-syndrome arithmetic.
"""

from __future__ import annotations

from typing import Tuple

import torch

from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
)


def dets_to_predecoder_inputs(
    dets: torch.Tensor,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str = "XV",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert detector bits to the pre-decoder's ``(trainX, x_syn_diff, z_syn_diff)``.

    Args:
        dets: ``(B, 2 * T * half)`` tensor of detector bits.
        distance: Surface-code distance ``D`` (must be odd).
        n_rounds: Number of measurement rounds ``T`` (must be >= 1).
        basis: Memory basis being decoded; ``'X'`` or ``'Z'``.
        code_rotation: Code orientation; ``'XV'``, ``'XH'``, ``'ZV'``, ``'ZH'``.

    Returns:
        ``(trainX, x_syn_diff, z_syn_diff)``. See module docstring for shapes.

    Raises:
        ValueError: If ``dets`` does not have ``2 * T * half`` columns, or if
            ``basis`` / ``code_rotation`` are not in the supported set.
    """
    if dets.ndim != 2:
        raise ValueError(f"dets must be 2-D (B, num_detectors); got shape {tuple(dets.shape)}")

    D = int(distance)
    T = int(n_rounds)
    if T < 1:
        raise ValueError(f"n_rounds must be >= 1, got {T}")
    if D < 3 or (D % 2) == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {D}")
    half = (D * D - 1) // 2

    basis_upper = str(basis).strip().upper()
    if basis_upper not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

    rotation = str(code_rotation).strip().upper() if code_rotation else "XV"
    if rotation not in ("XV", "XH", "ZV", "ZH"):
        raise ValueError(f"code_rotation must be one of XV/XH/ZV/ZH, got {code_rotation!r}")

    B, num_dets = dets.shape
    expected = 2 * T * half
    if int(num_dets) != expected:
        raise ValueError(
            f"detector count mismatch: dets has {int(num_dets)} columns, "
            f"expected 2 * n_rounds * (D^2 - 1)/2 = {expected} "
            f"(distance={D}, n_rounds={T})."
        )

    dets_f = dets.to(torch.float32)
    timeline_len = 2 * T

    # (B, half, 2*T) with one extra zero column at index ``timeline_len`` used
    # as a sentinel for boundary rounds that have no corresponding detector.
    dets_timeline = dets_f.view(B, timeline_len, half).permute(0, 2, 1).contiguous()
    zero_col = torch.zeros((B, half, 1), dtype=torch.float32, device=dets_f.device)
    padded = torch.cat([dets_timeline, zero_col], dim=2)
    sentinel_idx = timeline_len

    dev = dets.device
    x_bulk_idx = torch.arange(1, timeline_len - 1, 2, dtype=torch.long, device=dev)
    z_bulk_idx = torch.arange(2, timeline_len, 2, dtype=torch.long, device=dev)
    zero_idx = torch.zeros(1, dtype=torch.long, device=dev)
    sentinel = torch.full((1,), sentinel_idx, dtype=torch.long, device=dev)

    if basis_upper == "X":
        idx_x = torch.cat([zero_idx, x_bulk_idx])
        idx_z = torch.cat([sentinel, z_bulk_idx[:-1], sentinel]) if T > 1 \
            else torch.cat([sentinel, sentinel])
    else:
        idx_z = torch.cat([zero_idx, z_bulk_idx])
        idx_x = torch.cat([sentinel, x_bulk_idx[:-1], sentinel]) if T > 1 \
            else torch.cat([sentinel, sentinel])

    if T == 1:
        # For a single round there is no "bulk" interior; both index lists must
        # contain exactly T = 1 entry. The cross-basis row is the all-zero
        # sentinel; the in-basis row is the t=0 detector group.
        if basis_upper == "X":
            idx_x = zero_idx
            idx_z = sentinel
        else:
            idx_z = zero_idx
            idx_x = sentinel

    x_syn_diff = torch.index_select(padded, 2, idx_x).to(torch.int32).contiguous()
    z_syn_diff = torch.index_select(padded, 2, idx_z).to(torch.int32).contiguous()

    w_map_x = normalized_weight_mapping_Xstab_memory(D, rotation).reshape(D, D)
    w_map_z = normalized_weight_mapping_Zstab_memory(D, rotation).reshape(D, D)
    x_present = w_map_x.to(dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    z_present = w_map_z.to(dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    x_present = x_present.expand(B, T, D, D).contiguous()
    z_present = z_present.expand(B, T, D, D).contiguous()

    if basis_upper == "X":
        if T >= 1:
            z_present = z_present.clone()
            z_present[:, 0] = 0
            if T > 1:
                z_present[:, -1] = 0
    else:
        if T >= 1:
            x_present = x_present.clone()
            x_present[:, 0] = 0
            if T > 1:
                x_present[:, -1] = 0

    idx_map_x = torch.as_tensor(
        compute_stabX_to_data_index_map(D, rotation), dtype=torch.long, device=dev
    )
    idx_map_z = torch.as_tensor(
        compute_stabZ_to_data_index_map(D, rotation), dtype=torch.long, device=dev
    )
    n_stab_x = int(idx_map_x.shape[0])
    n_stab_z = int(idx_map_z.shape[0])

    x_grid = torch.zeros(B, D * D, T, dtype=torch.float32, device=dev)
    z_grid = torch.zeros(B, D * D, T, dtype=torch.float32, device=dev)
    x_grid[:, idx_map_x, :] = x_syn_diff[:, :n_stab_x, :].to(torch.float32)
    z_grid[:, idx_map_z, :] = z_syn_diff[:, :n_stab_z, :].to(torch.float32)

    x_type = x_grid.reshape(B, D, D, T).permute(0, 3, 1, 2).contiguous()
    z_type = z_grid.reshape(B, D, D, T).permute(0, 3, 1, 2).contiguous()
    train_x = torch.cat(
        [
            x_type.unsqueeze(1),
            z_type.unsqueeze(1),
            x_present.unsqueeze(1),
            z_present.unsqueeze(1),
        ],
        dim=1,
    ).contiguous()
    return train_x, x_syn_diff, z_syn_diff


__all__ = ["dets_to_predecoder_inputs"]
