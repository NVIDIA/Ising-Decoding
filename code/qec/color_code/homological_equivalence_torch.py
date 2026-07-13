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
"""Torch spacelike homological equivalence for triangular color codes.

The original reference spacelike implementation has been removed; this is
the only spacelike HE path now. The same algorithm in Torch:

* flatten ``(batch, round)`` into a single batch dimension;
* apply weight reduction with disjoint plaquette layers and GPU matmuls;
* run fix-equivalence sequentially in plaquette order, vectorized over the
  whole flattened batch;
* keep the implementation spacelike-only. Timelike color-code HE is deliberately
  out of scope for the torch data-generation path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Tuple

import numpy as np
import torch


def _as_uint8_binary(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.uint8)
    if x.dtype == torch.uint8:
        return x
    return torch.remainder(x.to(torch.uint8), 2)


HE_WEIGHT6_PATTERN = np.array(
    [
        [3, 4, 5],
        [2, 4, 5],
        [1, 4, 5],
        [2, 3, 4],
        [0, 3, 4],
        [1, 3, 5],
        [2, 3, 5],
        [0, 3, 5],
        [1, 3, 4],
        [0, 4, 5],
    ],
    dtype=np.int32,
)

HE_WEIGHT6_CANONICAL = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 5],
        [1, 2, 5],
        [0, 2, 4],
        [0, 1, 4],
        [1, 2, 4],
        [0, 2, 5],
        [1, 2, 3],
    ],
    dtype=np.int32,
)

W4_SRC_BLUE = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
W4_DST_BLUE = np.array([[2, 3], [1, 3], [0, 3]], dtype=np.int32)
W4_SRC_GREEN_RED = np.array([[2, 3], [0, 2], [1, 2]], dtype=np.int32)
W4_DST_GREEN_RED = np.array([[0, 1], [1, 3], [0, 3]], dtype=np.int32)
W4_SRC_BY_ORIENT = np.stack([W4_SRC_BLUE, W4_SRC_GREEN_RED, W4_SRC_GREEN_RED], axis=0)
W4_DST_BY_ORIENT = np.stack([W4_DST_BLUE, W4_DST_GREEN_RED, W4_DST_GREEN_RED], axis=0)

_LOCAL_MASK_BITS_6 = torch.arange(6, dtype=torch.int64)
_LOCAL_MASK_POWERS_6 = (1 << _LOCAL_MASK_BITS_6).to(torch.int64)
_LOCAL_MASK_BITS_4 = _LOCAL_MASK_BITS_6[:4]
_LOCAL_MASK_POWERS_4 = _LOCAL_MASK_POWERS_6[:4]
_LOCAL_TO_GLOBAL_4 = torch.tensor([0, 1, 4, 5], dtype=torch.long)


def _build_local_rewrite_lut_np(
    pattern: np.ndarray, canonical: np.ndarray, num_bits: int
) -> np.ndarray:
    powers = (1 << np.arange(num_bits, dtype=np.int32)).astype(np.int32)
    lut = np.arange(1 << num_bits, dtype=np.int32)
    src_mask = np.sum(powers[pattern], axis=-1)
    dst_mask = np.sum(powers[canonical], axis=-1)
    lut[src_mask] = dst_mask
    return lut


def _build_oriented_local_rewrite_lut_np(
    pattern_by_orient: np.ndarray,
    canonical_by_orient: np.ndarray,
    num_bits: int,
) -> np.ndarray:
    powers = (1 << np.arange(num_bits, dtype=np.int32)).astype(np.int32)
    table_size = 1 << num_bits
    lut = np.broadcast_to(
        np.arange(table_size, dtype=np.int32),
        (pattern_by_orient.shape[0], table_size),
    ).copy()
    src_mask = np.sum(powers[pattern_by_orient], axis=-1)
    dst_mask = np.sum(powers[canonical_by_orient], axis=-1)
    for orient in range(pattern_by_orient.shape[0]):
        lut[orient, src_mask[orient]] = dst_mask[orient]
    return lut


HE_W6_REWRITE_LUT = _build_local_rewrite_lut_np(
    HE_WEIGHT6_PATTERN,
    HE_WEIGHT6_CANONICAL,
    num_bits=6,
)
W4_REWRITE_LUT_BY_ORIENT = _build_oriented_local_rewrite_lut_np(
    W4_SRC_BY_ORIENT,
    W4_DST_BY_ORIENT,
    num_bits=4,
)


@dataclass(frozen=True)
class ColorSpacelikeHECache:
    """Precomputed static and tensor metadata for color-code spacelike HE."""

    qubit_indices: torch.Tensor  # (P, 6) int64, q1..q6, -1 padded
    weights: torch.Tensor  # (P,) int64
    valid_mask: torch.Tensor  # (P, 6) bool
    w4_orient: torch.Tensor  # (P,) int64
    parity_matrix: torch.Tensor  # (P, D) uint8
    wr6_layer_parity: torch.Tensor  # (L6, M6, D) float32
    wr6_layer_valid: torch.Tensor  # (L6, M6) bool
    wr6_layer_weights: torch.Tensor  # (L6, M6) int64
    wr4_layer_parity: torch.Tensor  # (L4, M4, D) float32
    wr4_layer_valid: torch.Tensor  # (L4, M4) bool
    wr4_layer_weights: torch.Tensor  # (L4, M4) int64
    fe_layer_q_safe: torch.Tensor  # (Lfe, Mfe, 6) int64
    fe_layer_valid: torch.Tensor  # (Lfe, Mfe, 6) bool
    fe_layer_active: torch.Tensor  # (Lfe, Mfe) bool
    fe_layer_weights: torch.Tensor  # (Lfe, Mfe) int64
    fe_layer_orient: torch.Tensor  # (Lfe, Mfe) int64
    fe_layer_write_q: Tuple[torch.Tensor, ...]  # per layer: (num_valid_qubits,) int64
    fe_layer_write_slot: Tuple[torch.Tensor, ...]  # per layer: flattened (Mfe*6) slots
    fe_layer_kind: Tuple[int, ...]  # 6=W6-only, 4=W4-only, 0=mixed/other
    stabilizer_group: torch.Tensor  # (G, D) uint8, G=1 when not enumerated
    w6_rewrite_lut: torch.Tensor  # (64,) int64
    w4_rewrite_lut_by_orient: torch.Tensor  # (3, 16) int64
    local_mask_bits_6: torch.Tensor  # (6,) int64
    local_mask_powers_6: torch.Tensor  # (6,) int64
    local_mask_bits_4: torch.Tensor  # (4,) int64
    local_mask_powers_4: torch.Tensor  # (4,) int64
    local_to_global4: torch.Tensor  # (4,) int64
    fe_order: Tuple[int, ...]
    weights_tuple: Tuple[int, ...]
    w4_orient_tuple: Tuple[int, ...]
    valid_slots: Tuple[Tuple[int, ...], ...]
    valid_slot_qubits: Tuple[Tuple[Tuple[int, int], ...], ...]
    all_valid6: Tuple[bool, ...]
    all_valid4: Tuple[bool, ...]
    num_plaquettes: int
    num_data: int


def _device_of_color_code() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_HE_CACHE_DEVICE_COPIES: dict = {}


def _cache_on_device(
    cache: "ColorSpacelikeHECache", device: torch.device
) -> "ColorSpacelikeHECache":
    """Return a copy of ``cache`` with all tensor fields on ``device``.

    The cache is built once on a single device (see ``build_color_spacelike_he_cache``),
    but under DistributedDataParallel each rank operates on its own GPU. Aligning the
    cache to the input tensor's device avoids cross-device ops such as ``cfg @ parity.T``.
    The result is memoized per ``(cache, device)`` so the host->device move happens at
    most once per device. When the cache is already on ``device`` (single-GPU / CPU) the
    original object is returned unchanged, so this cannot regress non-distributed runs.
    """
    device = torch.device(device)
    if torch.is_tensor(cache.parity_matrix) and cache.parity_matrix.device == device:
        return cache
    key = (id(cache), str(device))
    hit = _HE_CACHE_DEVICE_COPIES.get(key)
    if hit is not None:
        return hit
    moved = {}
    for f in fields(cache):
        v = getattr(cache, f.name)
        if torch.is_tensor(v):
            moved[f.name] = v.to(device)
        elif isinstance(v, tuple) and len(v) > 0 and all(torch.is_tensor(t) for t in v):
            moved[f.name] = tuple(t.to(device) for t in v)
        else:
            moved[f.name] = v
    new_cache = replace(cache, **moved)
    _HE_CACHE_DEVICE_COPIES[key] = new_cache
    return new_cache


def _build_layers(plaquette_indices: list[int], supports: list[set[int]]) -> list[list[int]]:
    layer_qubits: list[set[int]] = []
    layers: list[list[int]] = []
    for p_idx in plaquette_indices:
        supp = supports[p_idx]
        for lid, used in enumerate(layer_qubits):
            if supp.isdisjoint(used):
                layers[lid].append(p_idx)
                used.update(supp)
                break
        else:
            layers.append([p_idx])
            layer_qubits.append(set(supp))
    return layers


def _build_contiguous_disjoint_layers(
    plaquette_indices: tuple[int, ...],
    supports: list[set[int]],
) -> list[list[int]]:
    """Split an ordered plaquette list into contiguous disjoint-support layers."""
    layers: list[list[int]] = []
    current: list[int] = []
    used: set[int] = set()
    for p_idx in plaquette_indices:
        supp = supports[p_idx]
        if current and not supp.isdisjoint(used):
            layers.append(current)
            current = [p_idx]
            used = set(supp)
        else:
            current.append(p_idx)
            used.update(supp)
    if current:
        layers.append(current)
    return layers


def _pack_layer_parity(
    layers: list[list[int]],
    parity_matrix: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_layers = len(layers)
    max_layer_size = max((len(layer) for layer in layers), default=0)
    num_data = parity_matrix.shape[1]
    parity = np.zeros((num_layers, max_layer_size, num_data), dtype=np.float32)
    valid = np.zeros((num_layers, max_layer_size), dtype=bool)
    layer_weights = np.zeros((num_layers, max_layer_size), dtype=np.int64)
    for lid, layer in enumerate(layers):
        for mid, p_idx in enumerate(layer):
            parity[lid, mid] = parity_matrix[p_idx].astype(np.float32)
            valid[lid, mid] = True
            layer_weights[lid, mid] = int(weights[p_idx])
    return parity, valid, layer_weights


def _pack_fe_layers(
    layers: list[list[int]],
    qubit_indices: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    w4_orient: np.ndarray,
    num_data: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
    Tuple[int, ...],
]:
    num_layers = len(layers)
    max_layer_size = max((len(layer) for layer in layers), default=0)
    q_safe = np.zeros((num_layers, max_layer_size, 6), dtype=np.int64)
    valid = np.zeros((num_layers, max_layer_size, 6), dtype=bool)
    active = np.zeros((num_layers, max_layer_size), dtype=bool)
    layer_weights = np.zeros((num_layers, max_layer_size), dtype=np.int64)
    layer_orient = np.zeros((num_layers, max_layer_size), dtype=np.int64)

    for lid, layer in enumerate(layers):
        for mid, p_idx in enumerate(layer):
            q_safe[lid, mid] = np.clip(qubit_indices[p_idx], 0, num_data - 1)
            valid[lid, mid] = valid_mask[p_idx]
            active[lid, mid] = True
            layer_weights[lid, mid] = int(weights[p_idx])
            layer_orient[lid, mid] = int(max(0, min(2, int(w4_orient[p_idx]))))

    q_t = torch.as_tensor(q_safe, dtype=torch.long, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active, dtype=torch.bool, device=device)
    weights_t = torch.as_tensor(layer_weights, dtype=torch.long, device=device)
    orient_t = torch.as_tensor(layer_orient, dtype=torch.long, device=device)

    write_q: list[torch.Tensor] = []
    write_slot: list[torch.Tensor] = []
    layer_kind: list[int] = []
    for lid in range(num_layers):
        slots: list[int] = []
        qubits: list[int] = []
        active_weights: set[int] = set()
        for mid in range(max_layer_size):
            if not active[lid, mid]:
                continue
            active_weights.add(int(layer_weights[lid, mid]))
            for slot in range(6):
                if valid[lid, mid, slot]:
                    slots.append(mid * 6 + slot)
                    qubits.append(int(q_safe[lid, mid, slot]))
        write_q.append(torch.as_tensor(qubits, dtype=torch.long, device=device))
        write_slot.append(torch.as_tensor(slots, dtype=torch.long, device=device))
        if active_weights == {6}:
            layer_kind.append(6)
        elif active_weights == {4}:
            layer_kind.append(4)
        else:
            layer_kind.append(0)

    return (
        q_t,
        valid_t,
        active_t,
        weights_t,
        orient_t,
        tuple(write_q),
        tuple(write_slot),
        tuple(layer_kind),
    )


def _build_stabilizer_group(parity_matrix: np.ndarray, max_plaquettes: int = 16) -> np.ndarray:
    num_plaquettes, num_data = parity_matrix.shape
    if num_plaquettes > max_plaquettes:
        return np.zeros((1, num_data), dtype=np.uint8)

    group = np.zeros((1 << num_plaquettes, num_data), dtype=np.uint8)
    for mask in range(1, 1 << num_plaquettes):
        bit = (mask & -mask).bit_length() - 1
        group[mask] = group[mask ^ (1 << bit)] ^ parity_matrix[bit]
    return group


def build_color_spacelike_he_cache(
    color_code,
    *,
    device: Optional[torch.device] = None,
) -> ColorSpacelikeHECache:
    """Build Torch metadata matching the reference ``precompute_plaquette_data``."""
    if device is None:
        device = _device_of_color_code()

    from qec.color_code.homological_equivalence import get_plaquette_qubit_labels

    num_plaq = int(color_code.num_plaquettes)
    num_data = int(color_code.num_data)

    qubit_indices = np.full((num_plaq, 6), -1, dtype=np.int64)
    weights = np.zeros(num_plaq, dtype=np.int64)
    valid_mask = np.zeros((num_plaq, 6), dtype=bool)
    w4_orient = np.full(num_plaq, -1, dtype=np.int64)

    for p_idx, plaq in enumerate(color_code.plaquettes):
        weight = int(plaq["weight"])
        weights[p_idx] = weight

        if weight == 4:
            color = str(plaq.get("color", "")).lower()
            if color == "blue":
                w4_orient[p_idx] = 0
            elif color == "green":
                w4_orient[p_idx] = 1
            elif color == "red":
                w4_orient[p_idx] = 2
            else:
                w4_orient[p_idx] = 0

        labels = get_plaquette_qubit_labels(
            list(plaq["data_qubits"]),
            color_code.qubit_to_coord,
            weight,
        )
        for slot, key in enumerate(("q1", "q2", "q3", "q4", "q5", "q6")):
            q = int(labels[key])
            qubit_indices[p_idx, slot] = q
            valid_mask[p_idx, slot] = q >= 0

    parity_matrix = np.zeros((num_plaq, num_data), dtype=np.uint8)
    supports: list[set[int]] = []
    for p_idx in range(num_plaq):
        support: set[int] = set()
        for q in qubit_indices[p_idx].tolist():
            q = int(q)
            if q >= 0:
                parity_matrix[p_idx, q] = 1
                support.add(q)
        supports.append(support)

    w6_plaqs = [i for i in range(num_plaq) if int(weights[i]) == 6]
    w4_plaqs = [i for i in range(num_plaq) if int(weights[i]) == 4]
    wr6_parity, wr6_valid, wr6_weights = _pack_layer_parity(
        _build_layers(w6_plaqs, supports),
        parity_matrix,
        weights,
    )
    wr4_parity, wr4_valid, wr4_weights = _pack_layer_parity(
        _build_layers(w4_plaqs, supports),
        parity_matrix,
        weights,
    )

    fe_order = tuple(w6_plaqs + w4_plaqs)
    fe_layers = _build_contiguous_disjoint_layers(fe_order, supports)
    (
        fe_layer_q_safe,
        fe_layer_valid,
        fe_layer_active,
        fe_layer_weights,
        fe_layer_orient,
        fe_layer_write_q,
        fe_layer_write_slot,
        fe_layer_kind,
    ) = _pack_fe_layers(
        fe_layers,
        qubit_indices,
        valid_mask,
        weights,
        w4_orient,
        num_data,
        device,
    )
    weights_tuple = tuple(int(w) for w in weights.tolist())
    w4_orient_tuple = tuple(int(max(0, min(2, o))) for o in w4_orient.tolist())
    valid_slots = tuple(
        tuple(slot
              for slot in range(6)
              if bool(valid_mask[p_idx, slot]))
        for p_idx in range(num_plaq)
    )
    valid_slot_qubits = tuple(
        tuple((slot, int(qubit_indices[p_idx, slot]))
              for slot in valid_slots[p_idx])
        for p_idx in range(num_plaq)
    )
    all_valid6 = tuple(bool(valid_mask[p_idx].all()) for p_idx in range(num_plaq))
    all_valid4 = tuple(bool(valid_mask[p_idx, [0, 1, 4, 5]].all()) for p_idx in range(num_plaq))

    return ColorSpacelikeHECache(
        qubit_indices=torch.as_tensor(qubit_indices, dtype=torch.long, device=device),
        weights=torch.as_tensor(weights, dtype=torch.long, device=device),
        valid_mask=torch.as_tensor(valid_mask, dtype=torch.bool, device=device),
        w4_orient=torch.as_tensor(w4_orient, dtype=torch.long, device=device),
        parity_matrix=torch.as_tensor(parity_matrix, dtype=torch.uint8, device=device),
        wr6_layer_parity=torch.as_tensor(wr6_parity, dtype=torch.float32, device=device),
        wr6_layer_valid=torch.as_tensor(wr6_valid, dtype=torch.bool, device=device),
        wr6_layer_weights=torch.as_tensor(wr6_weights, dtype=torch.long, device=device),
        wr4_layer_parity=torch.as_tensor(wr4_parity, dtype=torch.float32, device=device),
        wr4_layer_valid=torch.as_tensor(wr4_valid, dtype=torch.bool, device=device),
        wr4_layer_weights=torch.as_tensor(wr4_weights, dtype=torch.long, device=device),
        fe_layer_q_safe=fe_layer_q_safe,
        fe_layer_valid=fe_layer_valid,
        fe_layer_active=fe_layer_active,
        fe_layer_weights=fe_layer_weights,
        fe_layer_orient=fe_layer_orient,
        fe_layer_write_q=fe_layer_write_q,
        fe_layer_write_slot=fe_layer_write_slot,
        fe_layer_kind=fe_layer_kind,
        stabilizer_group=torch.as_tensor(
            _build_stabilizer_group(parity_matrix),
            dtype=torch.uint8,
            device=device,
        ),
        w6_rewrite_lut=torch.as_tensor(HE_W6_REWRITE_LUT, dtype=torch.long, device=device),
        w4_rewrite_lut_by_orient=torch.as_tensor(
            W4_REWRITE_LUT_BY_ORIENT,
            dtype=torch.long,
            device=device,
        ),
        local_mask_bits_6=_LOCAL_MASK_BITS_6.to(device),
        local_mask_powers_6=_LOCAL_MASK_POWERS_6.to(device),
        local_mask_bits_4=_LOCAL_MASK_BITS_4.to(device),
        local_mask_powers_4=_LOCAL_MASK_POWERS_4.to(device),
        local_to_global4=_LOCAL_TO_GLOBAL_4.to(device),
        fe_order=fe_order,
        weights_tuple=weights_tuple,
        w4_orient_tuple=w4_orient_tuple,
        valid_slots=valid_slots,
        valid_slot_qubits=valid_slot_qubits,
        all_valid6=all_valid6,
        all_valid4=all_valid4,
        num_plaquettes=num_plaq,
        num_data=num_data,
    )


def _weight_reduction_layers_torch(
    cfg: torch.Tensor,
    changed: torch.Tensor,
    layer_parity: torch.Tensor,
    layer_valid: torch.Tensor,
    layer_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_parity.shape[0] == 0:
        return cfg, changed

    for lid in range(int(layer_parity.shape[0])):
        parity = layer_parity[lid]  # (M, D)
        valid = layer_valid[lid]  # (M,)
        weights = layer_weights[lid]  # (M,)

        counts = (cfg.to(torch.float32) @ parity.T).to(torch.int32)
        thresholds = torch.where(
            weights == 6,
            torch.full_like(weights, 4),
            torch.where(weights == 4, torch.full_like(weights, 3), torch.full_like(weights, 99)),
        )
        should_flip = valid.unsqueeze(0) & (counts >= thresholds.unsqueeze(0))
        flip_mask = ((should_flip.to(torch.float32) @ parity) > 0).to(torch.uint8)
        new_cfg = cfg ^ flip_mask
        changed = changed | (cfg != new_cfg)
        cfg = new_cfg

    return cfg, changed


def weight_reduction_color_torch(
    errors: torch.Tensor,
    cache: ColorSpacelikeHECache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched color-code spacelike weight reduction on ``(N, num_data)``."""
    cfg = _as_uint8_binary(errors)
    changed = torch.zeros_like(cfg, dtype=torch.bool)
    cfg, changed = _weight_reduction_layers_torch(
        cfg,
        changed,
        cache.wr6_layer_parity,
        cache.wr6_layer_valid,
        cache.wr6_layer_weights,
    )
    cfg, changed = _weight_reduction_layers_torch(
        cfg,
        changed,
        cache.wr4_layer_parity,
        cache.wr4_layer_valid,
        cache.wr4_layer_weights,
    )
    return cfg, changed


def fix_equivalence_color_torch(
    errors: torch.Tensor,
    cache: ColorSpacelikeHECache,
    changed_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched fix-equivalence matching the reference ``fix_equivalence_batched``.

    Plaquettes are processed in contiguous disjoint-support layers that preserve
    the same weight-6-then-weight-4 order as the reference. Each layer is vectorized over
    all rows and over the layer's non-overlapping plaquettes.
    """
    cfg = _as_uint8_binary(errors).clone()
    N, num_data = cfg.shape
    if changed_mask is None:
        changed = torch.zeros((N, num_data), dtype=torch.bool, device=cfg.device)
    else:
        changed = changed_mask.to(device=cfg.device, dtype=torch.bool).clone()

    for lid in range(int(cache.fe_layer_q_safe.shape[0])):
        q_safe = cache.fe_layer_q_safe[lid]  # (M, 6)
        valid = cache.fe_layer_valid[lid]  # (M, 6)
        active = cache.fe_layer_active[lid]  # (M,)
        weights = cache.fe_layer_weights[lid]  # (M,)
        orient = cache.fe_layer_orient[lid]  # (M,)
        M = int(q_safe.shape[0])
        if M == 0:
            continue

        vals = cfg.index_select(1, q_safe.reshape(-1)).reshape(N, M, 6)
        vals = vals * valid.to(torch.uint8).unsqueeze(0)

        layer_kind = cache.fe_layer_kind[lid]
        if layer_kind == 6:
            vals_i64 = vals.to(torch.int64)
            local_mask6 = (vals_i64 * cache.local_mask_powers_6.view(1, 1, 6)).sum(dim=2)
            count6 = vals_i64.sum(dim=2)
            dst_mask6 = cache.w6_rewrite_lut.index_select(0, local_mask6.reshape(-1)).reshape(
                N,
                M,
            )
            valid6 = valid.all(dim=1)
            do_change = (
                (weights == 6).unsqueeze(0) & active.unsqueeze(0) & valid6.unsqueeze(0) &
                (count6 == 3) & (dst_mask6 != local_mask6)
            )
            dst_support = ((dst_mask6.unsqueeze(2) >> cache.local_mask_bits_6.view(1, 1, 6)) &
                           1).to(torch.uint8)
        elif layer_kind == 4:
            vals4 = vals.index_select(2, cache.local_to_global4)
            vals4_i64 = vals4.to(torch.int64)
            local_mask4 = (vals4_i64 * cache.local_mask_powers_4.view(1, 1, 4)).sum(dim=2)
            count4 = vals4_i64.sum(dim=2)
            orient_idx = orient.unsqueeze(0).expand_as(local_mask4)
            dst_mask4 = cache.w4_rewrite_lut_by_orient[orient_idx, local_mask4]
            valid4 = valid.index_select(1, cache.local_to_global4).all(dim=1)
            do_change = (
                (weights == 4).unsqueeze(0) & active.unsqueeze(0) & valid4.unsqueeze(0) &
                (count4 == 2) & (dst_mask4 != local_mask4)
            )
            dst_bits4 = ((dst_mask4.unsqueeze(2) >> cache.local_mask_bits_4.view(1, 1, 4)) &
                         1).to(torch.uint8)
            dst_support = vals.clone()
            dst_support[:, :, 0] = dst_bits4[:, :, 0]
            dst_support[:, :, 1] = dst_bits4[:, :, 1]
            dst_support[:, :, 4] = dst_bits4[:, :, 2]
            dst_support[:, :, 5] = dst_bits4[:, :, 3]
        else:
            vals_i64 = vals.to(torch.int64)

            local_mask6 = (vals_i64 * cache.local_mask_powers_6.view(1, 1, 6)).sum(dim=2)
            count6 = vals_i64.sum(dim=2)
            dst_mask6 = cache.w6_rewrite_lut.index_select(0, local_mask6.reshape(-1)).reshape(
                N,
                M,
            )
            valid6 = valid.all(dim=1)
            do6 = (
                (weights == 6).unsqueeze(0) & active.unsqueeze(0) & valid6.unsqueeze(0) &
                (count6 == 3) & (dst_mask6 != local_mask6)
            )
            dst_support6 = ((dst_mask6.unsqueeze(2) >> cache.local_mask_bits_6.view(1, 1, 6)) &
                            1).to(torch.uint8)

            vals4 = vals.index_select(2, cache.local_to_global4)
            vals4_i64 = vals4.to(torch.int64)
            local_mask4 = (vals4_i64 * cache.local_mask_powers_4.view(1, 1, 4)).sum(dim=2)
            count4 = vals4_i64.sum(dim=2)
            orient_idx = orient.unsqueeze(0).expand_as(local_mask4)
            dst_mask4 = cache.w4_rewrite_lut_by_orient[orient_idx, local_mask4]
            valid4 = valid.index_select(1, cache.local_to_global4).all(dim=1)
            do4 = (
                (weights == 4).unsqueeze(0) & active.unsqueeze(0) & valid4.unsqueeze(0) &
                (count4 == 2) & (dst_mask4 != local_mask4)
            )
            dst_bits4 = ((dst_mask4.unsqueeze(2) >> cache.local_mask_bits_4.view(1, 1, 4)) &
                         1).to(torch.uint8)
            dst_support4 = vals.clone()
            dst_support4[:, :, 0] = dst_bits4[:, :, 0]
            dst_support4[:, :, 1] = dst_bits4[:, :, 1]
            dst_support4[:, :, 4] = dst_bits4[:, :, 2]
            dst_support4[:, :, 5] = dst_bits4[:, :, 3]

            is_w6 = (weights == 6).view(1, M, 1)
            dst_support = torch.where(is_w6, dst_support6, dst_support4)
            do_change = torch.where((weights == 6).unsqueeze(0), do6, do4)

        new_support = torch.where(valid.unsqueeze(0), dst_support, vals)
        actual_change = valid.unsqueeze(0) & (vals != new_support)
        would_change = actual_change.any(dim=2) & do_change
        changed_at_q = changed.index_select(1, q_safe.reshape(-1)).reshape(N, M, 6)
        conflict = (actual_change & changed_at_q).any(dim=2)
        apply_ok = would_change & (~conflict)

        write_q = cache.fe_layer_write_q[lid]
        write_slot = cache.fe_layer_write_slot[lid]
        if write_q.numel() == 0:
            continue

        flat_new_support = new_support.reshape(N, M * 6).index_select(1, write_slot)
        flat_apply = apply_ok.unsqueeze(2).expand(N, M, 6).reshape(N, M * 6).index_select(
            1,
            write_slot,
        )
        old_cols = cfg.index_select(1, write_q)
        new_cols = torch.where(flat_apply, flat_new_support, old_cols)
        write_q_expanded = write_q.unsqueeze(0).expand(N, -1)
        cfg.scatter_(1, write_q_expanded, new_cols)

        old_changed = changed.index_select(1, write_q)
        changed.scatter_(
            1,
            write_q_expanded,
            old_changed | (flat_apply & (old_cols != new_cols)),
        )

    return cfg


def coset_min_weight_batched_torch(
    errors: torch.Tensor,
    stabilizer_group: torch.Tensor,
) -> torch.Tensor:
    """Pick the minimum-weight representative from an enumerated stabilizer coset."""
    cfg = _as_uint8_binary(errors)
    group = _as_uint8_binary(stabilizer_group).to(cfg.device)
    if group.shape[0] <= 1:
        return cfg

    G = int(group.shape[0])
    N = int(cfg.shape[0])
    chunk_size = max(1, min(N, 4096 // G))
    out = torch.empty_like(cfg)
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        batch = cfg[start:end]
        candidates = batch.unsqueeze(1) ^ group.unsqueeze(0)
        weights = candidates.sum(dim=2)
        best = weights.argmin(dim=1)
        out[start:end] = candidates[torch.arange(end - start, device=cfg.device), best]
    return out


def simplify_color_batched_torch(
    errors: torch.Tensor,
    cache: ColorSpacelikeHECache,
    *,
    max_iterations: int = 16,
    use_coset_search: bool = False,
) -> torch.Tensor:
    """Batched spacelike simplification on ``(N, num_data)``."""
    cfg = _as_uint8_binary(errors)
    # Align the (single-device) cache to the batch's device for DDP correctness.
    cache = _cache_on_device(cache, cfg.device)

    def one_pass(x: torch.Tensor) -> torch.Tensor:
        reduced, changed = weight_reduction_color_torch(x, cache)
        return fix_equivalence_color_torch(reduced, cache, changed)

    prev = cfg
    cfg = one_pass(cfg)
    iters = 1
    while iters < int(max_iterations) and not torch.equal(cfg, prev):
        prev = cfg
        cfg = one_pass(cfg)
        iters += 1

    if use_coset_search:
        cfg = coset_min_weight_batched_torch(cfg, cache.stabilizer_group)

    return cfg


def apply_homological_equivalence_color_torch(
    z_diffs: torch.Tensor,
    x_diffs: torch.Tensor,
    cache: ColorSpacelikeHECache,
    *,
    max_iterations: int = 16,
    use_coset_search: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply spacelike color-code HE to ``(B, T, num_data)`` diff tensors."""
    z = _as_uint8_binary(z_diffs)
    x = _as_uint8_binary(x_diffs)
    if z.ndim != 3 or x.ndim != 3:
        raise ValueError("z_diffs and x_diffs must both have shape (B, T, num_data)")
    if z.shape != x.shape:
        raise ValueError(
            f"z_diffs and x_diffs must have matching shapes, got {z.shape} and {x.shape}"
        )

    B, T, D = z.shape
    z_flat = z.reshape(B * T, D)
    x_flat = x.reshape(B * T, D)

    # Color-code X and Z errors share the same spacelike HE rules. Run both
    # halves through one larger batch to amortize the Python loop and kernel
    # launch overhead in weight reduction / fix-equivalence.
    zx_can = simplify_color_batched_torch(
        torch.cat([z_flat, x_flat], dim=0),
        cache,
        max_iterations=max_iterations,
        use_coset_search=use_coset_search,
    )
    z_can, x_can = zx_can.split(B * T, dim=0)
    return z_can.reshape(B, T, D), x_can.reshape(B, T, D)


__all__ = [
    "ColorSpacelikeHECache",
    "build_color_spacelike_he_cache",
    "weight_reduction_color_torch",
    "fix_equivalence_color_torch",
    "coset_min_weight_batched_torch",
    "simplify_color_batched_torch",
    "apply_homological_equivalence_color_torch",
]
