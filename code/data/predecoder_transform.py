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

Single source of truth for converting Stim detector bits into the
``(trainX, x_syn_diff, z_syn_diff)`` tuple that the pre-decoder consumes.
:func:`dets_to_predecoder_inputs` (file-based datapipe) and
:meth:`evaluation.logical_error_rate.PreDecoderMemoryEvalModule._batch_to_trainx_and_syndromes`
(GPU/ONNX/TensorRT export path) both delegate to
:func:`_predecoder_transform_core`; the latter pre-registers the same buffers
the helper rebuilds ad-hoc per call.

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


def _build_scatter_perm(idx_map: torch.Tensor, D2: int, half: int) -> torch.Tensor:
    """Invert ``idx_map`` into a length-``D2`` permutation whose missing entries
    point at ``half`` (a sentinel column that callers keep all-zero)."""
    perm = torch.full((D2,), half, dtype=torch.long, device=idx_map.device)
    perm[idx_map] = torch.arange(idx_map.shape[0], dtype=torch.long, device=idx_map.device)
    return perm


def _predecoder_transform_core(
    dets: torch.Tensor,
    *,
    D: int,
    T: int,
    half: int,
    basis: str,
    scatter_perm_x: torch.Tensor,
    scatter_perm_z: torch.Tensor,
    w_mapXgrid: torch.Tensor,
    w_mapZgrid: torch.Tensor,
    zero_pad_row: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Buffer-fused dets → ``(trainX, x_syn_diff, z_syn_diff)`` transform.

    Returns ``x_syn_diff`` and ``z_syn_diff`` as float32; callers that need
    int32 cast at the boundary. The signature mirrors the buffers that
    :class:`PreDecoderMemoryEvalModule` pre-registers so the eval/export path
    can call this directly without per-call allocations.

    Args:
        dets: ``(B, 2 * T * half)`` detector bits.
        D, T, half, basis: Code / batch geometry. ``basis`` is ``'X'`` or ``'Z'``.
        scatter_perm_x, scatter_perm_z: length-``D*D`` long tensors mapping
            each grid position to a syndrome row (or to the sentinel column
            at index ``half``); see :func:`_build_scatter_perm`.
        w_mapXgrid, w_mapZgrid: ``(1, 1, D, D)`` float32 presence grids.
        zero_pad_row: ``(1, 1, 1)`` float32 — broadcast to fabricate
            sentinel rows/columns for scatter-via-gather.
    """
    B = dets.shape[0]
    timeline_len = 2 * T
    dev = dets.device

    # ── trt_L1: preprocessor (cast, deinterleave, index_select, boundary handling) ──
    # (B, 2*T*half) -> (B, half, 2*T) float32, then pad a sentinel column for
    # boundary rounds that have no corresponding detector.
    dets_timeline = dets.to(torch.float32).view(B, timeline_len, half).permute(0, 2, 1).contiguous()
    zero_col = zero_pad_row.expand(B, half, 1)
    padded = torch.cat([dets_timeline, zero_col], dim=2)  # (B, half, 2*T+1)
    sentinel_idx = timeline_len

    x_bulk_idx = torch.arange(1, timeline_len - 1, 2, dtype=torch.long, device=dev)
    z_bulk_idx = torch.arange(2, timeline_len, 2, dtype=torch.long, device=dev)
    zero_idx = torch.zeros(1, dtype=torch.long, device=dev)
    sentinel = torch.full((1,), sentinel_idx, dtype=torch.long, device=dev)

    if T == 1:
        if basis == "X":
            idx_x, idx_z = zero_idx, sentinel
        else:
            idx_z, idx_x = zero_idx, sentinel
    else:
        if basis == "X":
            idx_x = torch.cat([zero_idx, x_bulk_idx])
            idx_z = torch.cat([sentinel, z_bulk_idx[:-1], sentinel])
        else:
            idx_z = torch.cat([zero_idx, z_bulk_idx])
            idx_x = torch.cat([sentinel, x_bulk_idx[:-1], sentinel])

    x_syn_diff = torch.index_select(padded, 2, idx_x)  # (B, half, T) float32
    z_syn_diff = torch.index_select(padded, 2, idx_z)  # (B, half, T) float32

    # Presence: broadcast-multiply by round mask to zero boundary rounds (no clone/in-place)
    if T == 1:
        boundary_mask = torch.zeros(1, device=dev, dtype=torch.float32)
    else:
        boundary_mask = torch.cat(
            [
                torch.zeros(1, device=dev, dtype=torch.float32),
                torch.ones(T - 2, device=dev, dtype=torch.float32),
                torch.zeros(1, device=dev, dtype=torch.float32),
            ]
        )
    boundary_mask = boundary_mask.view(1, T, 1, 1)
    if basis == "X":
        x_present = w_mapXgrid.expand(B, T, D, D)
        z_present = (w_mapZgrid * boundary_mask).expand(B, T, D, D)
    else:
        x_present = (w_mapXgrid * boundary_mask).expand(B, T, D, D)
        z_present = w_mapZgrid.expand(B, T, D, D)

    # ── trt_L2: trainX assembly (scatter-via-gather → grid reshape → cat) ──
    zero_pad = zero_pad_row.expand(B, 1, T)
    x_grid = torch.index_select(
        torch.cat([x_syn_diff, zero_pad], dim=1), 1, scatter_perm_x
    )  # (B, D², T)
    z_grid = torch.index_select(torch.cat([z_syn_diff, zero_pad], dim=1), 1, scatter_perm_z)
    x_type = x_grid.reshape(B, D, D, T).permute(0, 3, 1, 2).contiguous()  # (B, T, D, D)
    z_type = z_grid.reshape(B, D, D, T).permute(0, 3, 1, 2).contiguous()
    trainX = torch.cat(
        [
            x_type.unsqueeze(1),
            z_type.unsqueeze(1),
            x_present.unsqueeze(1),
            z_present.unsqueeze(1),
        ],
        dim=1,
    ).contiguous()

    return trainX, x_syn_diff, z_syn_diff


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

    num_dets = dets.shape[1]
    expected = 2 * T * half
    if int(num_dets) != expected:
        raise ValueError(
            f"detector count mismatch: dets has {int(num_dets)} columns, "
            f"expected 2 * n_rounds * (D^2 - 1)/2 = {expected} "
            f"(distance={D}, n_rounds={T})."
        )

    dev = dets.device
    idx_map_x = torch.as_tensor(
        compute_stabX_to_data_index_map(D, rotation), dtype=torch.long, device=dev
    )
    idx_map_z = torch.as_tensor(
        compute_stabZ_to_data_index_map(D, rotation), dtype=torch.long, device=dev
    )
    scatter_perm_x = _build_scatter_perm(idx_map_x, D * D, half)
    scatter_perm_z = _build_scatter_perm(idx_map_z, D * D, half)
    w_mapX = normalized_weight_mapping_Xstab_memory(D, rotation).reshape(D, D)
    w_mapZ = normalized_weight_mapping_Zstab_memory(D, rotation).reshape(D, D)
    w_mapXgrid = w_mapX.to(dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    w_mapZgrid = w_mapZ.to(dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0)
    zero_pad_row = torch.zeros(1, 1, 1, dtype=torch.float32, device=dev)

    trainX, x_syn_diff, z_syn_diff = _predecoder_transform_core(
        dets,
        D=D,
        T=T,
        half=half,
        basis=basis_upper,
        scatter_perm_x=scatter_perm_x,
        scatter_perm_z=scatter_perm_z,
        w_mapXgrid=w_mapXgrid,
        w_mapZgrid=w_mapZgrid,
        zero_pad_row=zero_pad_row,
    )
    return (
        trainX,
        x_syn_diff.to(torch.int32).contiguous(),
        z_syn_diff.to(torch.int32).contiguous(),
    )


__all__ = ["dets_to_predecoder_inputs"]
