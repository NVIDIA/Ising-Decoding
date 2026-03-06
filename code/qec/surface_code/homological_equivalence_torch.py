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
Torch Implementation of Homological Equivalence Transformations
===============================================================

This module provides a high-performance Torch implementation of the subset of
homological equivalence (HE) used in the data-generation/training pipeline:

- Spacelike HE on *diff* frames (canonicalize each (batch, round) independently)
- Timelike HE, weight-1 only (brickwork / Trotterized time-pair processing)

Scope / Non-goals
-----------------
- Weight-2 timelike HE is intentionally NOT implemented (never used in actual training).

Correctness goal
----------------
For realistic SurfaceCode circuits, outputs are deterministic and consistent
for the same inputs.

Performance strategy
--------------------
- Flatten (B, T) into a single batch dimension for spacelike ops.
- Weight reduction uses matmul against stabilizer support masks (fast on GPU).
- Equivalence fixing is sequential over stabilizers to match overlap semantics,
  but each stabilizer step is fully vectorized over the batch.
- Timelike overlap resolution avoids materializing dense (B, num_stabs, D2)
  tensors by using sparse edge lists + `scatter_reduce_`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


def _as_uint8_binary(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.uint8)
    if x.dtype != torch.uint8:
        x = x.to(torch.uint8)
    return x & 1


# -----------------------------------------------------------------------------
# Spacelike HE caches / helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SpacelikeHECache:
    distance: int
    parity: torch.Tensor  # (num_stabs, D2) uint8
    support_masks: torch.Tensor  # (num_stabs, D2) uint8
    support_sizes: torch.Tensor  # (num_stabs,) int64
    layers: Tuple[torch.Tensor, ...]  # tuple of (L_i,) int64 stabilizer indices

    # Weight-2 boundary stabilizers
    w2_canonical: torch.Tensor  # (num_stabs,) int64, -1 if not weight-2
    w2_other: torch.Tensor  # (num_stabs,) int64, -1 if not weight-2

    # Weight-4 stabilizers corners in coordinate order (tl,tr,bl,br)
    w4_tl: torch.Tensor  # (num_stabs,) int64, -1 if not weight-4
    w4_tr: torch.Tensor
    w4_bl: torch.Tensor
    w4_br: torch.Tensor


def _compute_layers_greedy(support_masks_cpu_bool: torch.Tensor) -> List[List[int]]:
    """
    Deterministically build disjoint layers in original stabilizer order.
    """
    num_stabs, D2 = support_masks_cpu_bool.shape
    layers: List[List[int]] = []
    current_union = torch.zeros((D2,), dtype=torch.bool)
    current_layer: List[int] = []

    for i in range(num_stabs):
        supp = support_masks_cpu_bool[i]
        if bool(torch.any(supp & current_union)):
            layers.append(current_layer)
            current_layer = [i]
            current_union = supp.clone()
        else:
            current_layer.append(i)
            current_union |= supp

    if current_layer:
        layers.append(current_layer)
    return layers


def _precompute_w2_boundary_canonical(pair: Tuple[int, int], distance: int) -> Tuple[int, int]:
    """
    For a weight-2 boundary stabilizer pair (a,b), return (canonical, other)
    using boundary orientation and weight-1 fix logic.
    """
    a, b = pair
    a_alpha, a_beta = divmod(a, distance)
    b_alpha, b_beta = divmod(b, distance)
    is_horizontal = (a_alpha == b_alpha)
    if is_horizontal:
        if a_alpha == 0:
            canonical = a if a_beta > b_beta else b
        else:
            canonical = a if a_beta < b_beta else b
    else:
        if a_beta == 0:
            canonical = a if a_alpha < b_alpha else b
        else:
            canonical = a if a_alpha > b_alpha else b
    other = b if canonical == a else a
    return canonical, other


def build_spacelike_he_cache(
    parity_matrix: torch.Tensor,
    distance: Optional[int] = None,
    *,
    device: Optional[torch.device] = None,
) -> SpacelikeHECache:
    """
    Precompute stabilizer metadata for fast spacelike HE.
    """
    parity_u8 = _as_uint8_binary(parity_matrix)
    num_stabs, D2 = parity_u8.shape
    if distance is None:
        distance = int(int(D2)**0.5)
    d = int(distance)

    if device is None:
        device = parity_u8.device

    # Keep parity on device (used by callers), but do precompute on CPU.
    parity_cpu = parity_u8.to("cpu")
    support_masks_cpu = (parity_cpu == 1).to(torch.uint8)
    support_sizes_cpu = support_masks_cpu.sum(dim=1, dtype=torch.int64)

    layers_list = _compute_layers_greedy((support_masks_cpu == 1))

    # Precompute w2 / w4 metadata on CPU
    w2_canonical_cpu = torch.full((num_stabs,), -1, dtype=torch.int64)
    w2_other_cpu = torch.full((num_stabs,), -1, dtype=torch.int64)

    w4_tl_cpu = torch.full((num_stabs,), -1, dtype=torch.int64)
    w4_tr_cpu = torch.full((num_stabs,), -1, dtype=torch.int64)
    w4_bl_cpu = torch.full((num_stabs,), -1, dtype=torch.int64)
    w4_br_cpu = torch.full((num_stabs,), -1, dtype=torch.int64)

    for s in range(num_stabs):
        ss = int(support_sizes_cpu[s].item())
        if ss == 2:
            idx = torch.nonzero(parity_cpu[s] == 1, as_tuple=False).flatten().tolist()
            if len(idx) != 2:
                continue
            a, b = int(idx[0]), int(idx[1])
            canonical, other = _precompute_w2_boundary_canonical((a, b), d)
            w2_canonical_cpu[s] = canonical
            w2_other_cpu[s] = other
        elif ss == 4:
            idx = torch.nonzero(parity_cpu[s] == 1, as_tuple=False).flatten().tolist()
            if len(idx) != 4:
                continue
            coords = sorted([(i // d, i % d, i) for i in idx])  # (alpha,beta,i)
            w4_tl_cpu[s] = coords[0][2]
            w4_tr_cpu[s] = coords[1][2]
            w4_bl_cpu[s] = coords[2][2]
            w4_br_cpu[s] = coords[3][2]

    # Move to target device
    parity = parity_u8.to(device)
    support_masks = (parity == 1).to(torch.uint8)
    support_sizes = support_masks.sum(dim=1, dtype=torch.int64)
    layers = tuple(torch.tensor(layer, dtype=torch.int64, device=device) for layer in layers_list)

    return SpacelikeHECache(
        distance=d,
        parity=parity,
        support_masks=support_masks,
        support_sizes=support_sizes,
        layers=layers,
        w2_canonical=w2_canonical_cpu.to(device),
        w2_other=w2_other_cpu.to(device),
        w4_tl=w4_tl_cpu.to(device),
        w4_tr=w4_tr_cpu.to(device),
        w4_bl=w4_bl_cpu.to(device),
        w4_br=w4_br_cpu.to(device),
    )


def _weight_reduction(cfg: torch.Tensor, cache: SpacelikeHECache) -> torch.Tensor:
    """
    Weight reduction (parallel within disjoint stabilizer layers).

    cfg: (N, D2) uint8 in {0,1}
    """
    cfg = _as_uint8_binary(cfg)
    # CUDA does not support integer GEMM for these dtypes in PyTorch;
    # use float accumulation (exact here since counts are small integers).
    cfg_f = cfg.to(torch.float32)
    support_masks_f = cache.support_masks.to(torch.float32)  # (S, D2)
    support_sizes = cache.support_sizes  # (S,)

    for layer_idx in cache.layers:
        if layer_idx.numel() == 0:
            continue
        masks = support_masks_f.index_select(0, layer_idx)  # (L, D2) float
        sizes = support_sizes.index_select(0, layer_idx)  # (L,)

        error_counts = (cfg_f @ masks.t()).to(torch.int32)  # (N, L) int32
        act1 = (error_counts == 4) | ((error_counts == 2) & (sizes.unsqueeze(0) == 2))
        act2 = (error_counts == 3)

        set_to_zero_mask = ((act1.to(torch.float32) @ masks) > 0)  # (N, D2)
        flip_mask = ((act2.to(torch.float32) @ masks) > 0) & (~set_to_zero_mask)

        cfg = cfg * (~set_to_zero_mask).to(cfg.dtype)
        cfg = cfg ^ flip_mask.to(cfg.dtype)
        cfg_f = cfg.to(torch.float32)

    return cfg


def _apply_corner_update(
    cfg_col: torch.Tensor,
    *,
    set_one: torch.Tensor,
    set_zero: torch.Tensor,
) -> torch.Tensor:
    # Disjoint masks: set_one and set_zero should not overlap.
    return torch.where(
        set_one, torch.ones_like(cfg_col),
        torch.where(set_zero, torch.zeros_like(cfg_col), cfg_col)
    )


def _fix_equivalence(cfg: torch.Tensor, cache: SpacelikeHECache, *, basis: str) -> torch.Tensor:
    """
    Equivalence fixing with overlap handling (sequential over stabilizers).

    basis selects the diagonal rule for weight-4 stabilizers:
      - basis='X': diagonal TL+BR -> TR+BL
      - basis='Z': diagonal TR+BL -> TL+BR
    """
    cfg = _as_uint8_binary(cfg)
    N, D2 = cfg.shape
    claimed = torch.zeros((N, D2), dtype=torch.bool, device=cfg.device)

    num_stabs = int(cache.support_sizes.numel())
    basis = basis.upper()
    for s in range(num_stabs):
        ss = int(cache.support_sizes[s].item())
        if ss == 2:
            canonical = int(cache.w2_canonical[s].item())
            other = int(cache.w2_other[s].item())
            if canonical < 0 or other < 0:
                continue

            vals = cfg[:, (canonical, other)]  # (N,2)
            error_count = vals.sum(dim=1)
            has_overlap = (vals.bool() & claimed[:, (canonical, other)]).any(dim=1)
            should_process = (error_count == 1) & (~has_overlap)

            # If error is on `other`, move it to canonical
            error_at_canonical = cfg[:, canonical] == 1
            should_move = should_process & (~error_at_canonical)
            if should_move.any():
                cfg[:, canonical] = torch.where(
                    should_move, torch.ones_like(cfg[:, canonical]), cfg[:, canonical]
                )
                cfg[:, other] = torch.where(
                    should_move, torch.zeros_like(cfg[:, other]), cfg[:, other]
                )
                claimed[:, canonical] = claimed[:, canonical] | should_move
                claimed[:, other] = claimed[:, other] | should_move

        elif ss == 4:
            tl = int(cache.w4_tl[s].item())
            if tl < 0:
                continue
            tr = int(cache.w4_tr[s].item())
            bl = int(cache.w4_bl[s].item())
            br = int(cache.w4_br[s].item())

            sub = cfg[:, (tl, tr, bl, br)]  # (N,4)
            error_count = sub.sum(dim=1)
            has_overlap = (sub.bool() & claimed[:, (tl, tr, bl, br)]).any(dim=1)
            should_process = (error_count == 2) & (~has_overlap)
            if not should_process.any():
                continue

            tl1 = sub[:, 0] == 1
            tr1 = sub[:, 1] == 1
            bl1 = sub[:, 2] == 1
            br1 = sub[:, 3] == 1

            if basis == "X":
                # vertical: TL+BL -> TR+BR
                m1 = should_process & tl1 & bl1 & (~tr1) & (~br1)
                # horizontal: BL+BR -> TL+TR
                m2 = should_process & bl1 & br1 & (~tl1) & (~tr1)
                # diagonal: TL+BR -> TR+BL
                m3 = should_process & tl1 & br1 & (~tr1) & (~bl1)

                moved = m1 | m2 | m3
                if moved.any():
                    cfg[:, tl] = _apply_corner_update(cfg[:, tl], set_one=m2, set_zero=m1 | m3)
                    cfg[:, tr] = _apply_corner_update(
                        cfg[:, tr], set_one=m1 | m2 | m3, set_zero=torch.zeros_like(m1)
                    )
                    cfg[:, bl] = _apply_corner_update(cfg[:, bl], set_one=m3, set_zero=m1 | m2)
                    cfg[:, br] = _apply_corner_update(cfg[:, br], set_one=m1, set_zero=m2 | m3)
                    claimed[:, (tl, tr, bl, br)] = claimed[:, (tl, tr, bl, br)] | moved.unsqueeze(1)

            else:  # basis == "Z"
                # vertical: TL+BL -> TR+BR
                m1 = should_process & tl1 & bl1 & (~tr1) & (~br1)
                # horizontal: BL+BR -> TL+TR
                m2 = should_process & bl1 & br1 & (~tl1) & (~tr1)
                # diagonal Z: TR+BL -> TL+BR
                m3 = should_process & tr1 & bl1 & (~tl1) & (~br1)

                moved = m1 | m2 | m3
                if moved.any():
                    cfg[:, tl] = _apply_corner_update(cfg[:, tl], set_one=m2 | m3, set_zero=m1)
                    cfg[:, tr] = _apply_corner_update(cfg[:, tr], set_one=m1 | m2, set_zero=m3)
                    cfg[:, bl] = _apply_corner_update(
                        cfg[:, bl], set_one=torch.zeros_like(m1), set_zero=m1 | m2 | m3
                    )
                    cfg[:, br] = _apply_corner_update(cfg[:, br], set_one=m1 | m3, set_zero=m2)
                    claimed[:, (tl, tr, bl, br)] = claimed[:, (tl, tr, bl, br)] | moved.unsqueeze(1)

    return cfg


def _simplify_spacelike(
    cfg: torch.Tensor,
    cache: SpacelikeHECache,
    *,
    basis: str,
    max_iterations: int = 100,
) -> torch.Tensor:
    cfg = _as_uint8_binary(cfg)
    prev = torch.empty_like(cfg)
    for _ in range(int(max_iterations)):
        prev.copy_(cfg)
        cfg = _weight_reduction(cfg, cache)
        cfg = _fix_equivalence(cfg, cache, basis=basis)
        if torch.equal(cfg, prev):
            break
    return cfg


def apply_homological_equivalence_torch_vmap(
    z_diffs: torch.Tensor,
    x_diffs: torch.Tensor,
    parity_matrix_Z: torch.Tensor,
    parity_matrix_X: torch.Tensor,
    distance: Optional[int] = None,
    *,
    cache_Z: Optional[SpacelikeHECache] = None,
    cache_X: Optional[SpacelikeHECache] = None,
    max_iterations: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Spacelike HE over batch and time: takes diff frames and canonicalizes
    each diff independently.
    """
    z = _as_uint8_binary(z_diffs)
    x = _as_uint8_binary(x_diffs)
    B, T, D2 = z.shape
    if distance is None:
        distance = int(int(D2)**0.5)
    d = int(distance)

    if cache_Z is None:
        cache_Z = build_spacelike_he_cache(parity_matrix_Z, distance=d, device=z.device)
    if cache_X is None:
        cache_X = build_spacelike_he_cache(parity_matrix_X, distance=d, device=x.device)

    z_flat = z.reshape(B * T, D2)
    x_flat = x.reshape(B * T, D2)

    x_can = _simplify_spacelike(x_flat, cache_X, basis="X", max_iterations=max_iterations)
    z_can = _simplify_spacelike(z_flat, cache_Z, basis="Z", max_iterations=max_iterations)

    return z_can.reshape(B, T, D2), x_can.reshape(B, T, D2)


# -----------------------------------------------------------------------------
# Timelike HE (weight-1 only)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelikeHECache:
    edge_stab: torch.Tensor  # (E,) int64
    edge_qubit: torch.Tensor  # (E,) int64
    num_stabs: int
    D2: int


def build_timelike_he_cache(parity_stab_to_qubit: torch.Tensor) -> TimelikeHECache:
    parity = _as_uint8_binary(parity_stab_to_qubit)
    num_stabs, D2 = parity.shape
    idx = torch.nonzero(parity == 1, as_tuple=False)
    return TimelikeHECache(
        edge_stab=idx[:, 0].to(torch.int64),
        edge_qubit=idx[:, 1].to(torch.int64),
        num_stabs=int(num_stabs),
        D2=int(D2),
    )


def _require_scatter_reduce() -> None:
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "Timelike HE requires torch.Tensor.scatter_reduce_ (PyTorch >= 1.12 / 2.x)."
        )


def _timelike_pair_step_torch(
    diffs_bt: torch.Tensor,  # (B2, 2, D2)
    meas_bt: torch.Tensor,  # (B2, 2, num_stabs)
    parity_stab_to_qubit: torch.Tensor,  # (num_stabs, D2)
    *,
    use_tie_breaker: bool = True,
    trainX_bt: Optional[torch.Tensor] = None,  # (B2, 2, num_stabs)
    cache: Optional[TimelikeHECache] = None,
    overlap_chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single timelike weight-1 step for one adjacent time pair, including
    overlap handling.
    """
    _require_scatter_reduce()

    diffs_bt = _as_uint8_binary(diffs_bt)
    meas_bt = _as_uint8_binary(meas_bt)
    parity = _as_uint8_binary(parity_stab_to_qubit)
    if trainX_bt is not None:
        trainX_bt = _as_uint8_binary(trainX_bt)

    B2, _, D2 = diffs_bt.shape
    num_stabs = int(meas_bt.shape[2])

    if cache is None:
        cache = build_timelike_he_cache(parity)

    edge_stab = cache.edge_stab.to(diffs_bt.device)
    edge_qubit = cache.edge_qubit.to(diffs_bt.device)

    # CUDA integer GEMM isn't supported in PyTorch for these dtypes;
    # use float accumulation (exact for these small integer sums).
    parity_f = parity.to(torch.float32)

    meas_contrib = torch.einsum("bts,sd->btd", meas_bt.to(torch.float32), parity_f).to(torch.int32)
    if trainX_bt is not None:
        trainX_contrib = torch.einsum("bts,sd->btd", trainX_bt.to(torch.float32),
                                      parity_f).to(torch.int32)
    else:
        trainX_contrib = torch.zeros_like(meas_contrib)

    old_density_per_round = diffs_bt.to(torch.int32) + meas_contrib + trainX_contrib
    old_density = old_density_per_round.sum(dim=1)  # (B2, D2)

    # Proposed flip: flip diffs at both rounds; flip meas at round 0 only
    new_diffs_bt = (1 - diffs_bt).to(torch.uint8)
    new_meas_bt = meas_bt.clone()
    new_meas_bt[:, 0, :] = 1 - new_meas_bt[:, 0, :]

    new_meas_contrib = torch.einsum("bts,sd->btd", new_meas_bt.to(torch.float32),
                                    parity_f).to(torch.int32)
    new_density_per_round = new_diffs_bt.to(torch.int32) + new_meas_contrib + trainX_contrib
    new_density = new_density_per_round.sum(dim=1)

    accept_raw = new_density < old_density
    if use_tie_breaker:
        density_equal = new_density == old_density
        old_max = torch.maximum(old_density_per_round[:, 0, :], old_density_per_round[:, 1, :])
        new_max = torch.maximum(new_density_per_round[:, 0, :], new_density_per_round[:, 1, :])
        accept_raw = accept_raw | (density_equal & (new_max > old_max))

    # Overlap resolution via sparse edges + scatter_reduce_ (chunked over batch)
    accept_final = torch.zeros_like(accept_raw)

    # Use int16 for min-reductions (D2 is small: D<=25 => D2<=625)
    sentinel = int(D2)
    edge_stab_2d = edge_stab.view(1, -1)
    edge_qubit_2d = edge_qubit.view(1, -1)
    edge_qubit_i16 = edge_qubit.to(torch.int16)

    for start in range(0, B2, int(overlap_chunk_size)):
        end = min(B2, start + int(overlap_chunk_size))
        a = accept_raw[start:end]  # (Bc, D2) bool
        Bc = int(a.shape[0])

        a_edge = a.index_select(1, edge_qubit)  # (Bc, E) bool
        cand = torch.where(
            a_edge,
            edge_qubit_i16.view(1, -1).expand(Bc, -1),
            torch.full((Bc, edge_qubit.numel()), sentinel, device=a.device, dtype=torch.int16),
        )  # (Bc, E) int16

        min_q = torch.full((Bc, num_stabs), sentinel, device=a.device, dtype=torch.int16)
        min_q.scatter_reduce_(
            dim=1,
            index=edge_stab_2d.expand(Bc, -1),
            src=cand,
            reduce="amin",
            include_self=True,
        )

        min_edge = min_q.index_select(1, edge_stab)  # (Bc, E) int16
        ok_edge = (min_edge == edge_qubit_i16.view(1, -1).expand(Bc, -1))  # (Bc, E) bool

        all_ok = torch.ones((Bc, D2), device=a.device, dtype=torch.uint8)
        all_ok.scatter_reduce_(
            dim=1,
            index=edge_qubit_2d.expand(Bc, -1),
            src=ok_edge.to(torch.uint8),
            reduce="amin",
            include_self=True,
        )

        accept_final[start:end] = a & all_ok.bool()

    # Apply updates
    diffs_out = diffs_bt ^ accept_final.to(diffs_bt.dtype).unsqueeze(1)
    flip_counts = (accept_final.to(torch.float32) @ parity_f.t()).to(torch.int32)  # (B2, num_stabs)
    flip_stab = (flip_counts & 1).to(meas_bt.dtype)
    meas_out = meas_bt.clone()
    meas_out[:, 0, :] = meas_out[:, 0, :] ^ flip_stab
    return diffs_out, meas_out


def _timelike_pass_brickwork_torch(
    diffs: torch.Tensor,  # (B, T, D2)
    meas: torch.Tensor,  # (B, T, num_stabs)
    parity_stab_to_qubit: torch.Tensor,
    *,
    exclude_round0: bool = False,
    use_tie_breaker: bool = True,
    trainX: Optional[torch.Tensor] = None,  # (B, T, num_stabs)
    cache: Optional[TimelikeHECache] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diffs = _as_uint8_binary(diffs)
    meas = _as_uint8_binary(meas)
    if trainX is not None:
        trainX = _as_uint8_binary(trainX)

    B, T, D2 = diffs.shape
    num_stabs = int(meas.shape[2])

    start_even = 2 if exclude_round0 else 0

    def process_pass(start_idx: int, d: torch.Tensor, m: torch.Tensor,
                     tX: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        num_pairs = (T - start_idx) // 2
        if num_pairs <= 0:
            return d, m
        slice_len = 2 * num_pairs
        end_idx = start_idx + slice_len

        d_slice = d[:, start_idx:end_idx, :]
        m_slice = m[:, start_idx:end_idx, :]
        d_flat = d_slice.reshape(B * num_pairs, 2, D2)
        m_flat = m_slice.reshape(B * num_pairs, 2, num_stabs)

        tX_flat = None
        if tX is not None:
            tX_slice = tX[:, start_idx:end_idx, :]
            tX_flat = tX_slice.reshape(B * num_pairs, 2, num_stabs)

        d_new, m_new = _timelike_pair_step_torch(
            d_flat,
            m_flat,
            parity_stab_to_qubit,
            use_tie_breaker=use_tie_breaker,
            trainX_bt=tX_flat,
            cache=cache,
        )

        d_out = d.clone()
        m_out = m.clone()
        d_out[:, start_idx:end_idx, :] = d_new.reshape(B, slice_len, D2)
        m_out[:, start_idx:end_idx, :] = m_new.reshape(B, slice_len, num_stabs)
        return d_out, m_out

    diffs, meas = process_pass(start_even, diffs, meas, trainX)
    diffs, meas = process_pass(1, diffs, meas, trainX)
    return diffs, meas


def _apply_timelike_weight1_convergence_torch(
    z_error_diffs: torch.Tensor,
    x_error_diffs: torch.Tensor,
    s1s2_x: torch.Tensor,
    s1s2_z: torch.Tensor,
    parity_matrix_X: torch.Tensor,
    parity_matrix_Z: torch.Tensor,
    *,
    max_passes: int,
    basis: str,
    use_tie_breaker: bool = True,
    trainX_x: Optional[torch.Tensor] = None,
    trainX_z: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = _as_uint8_binary(z_error_diffs)
    x = _as_uint8_binary(x_error_diffs)
    sx = _as_uint8_binary(s1s2_x)
    sz = _as_uint8_binary(s1s2_z)

    tX_x = _as_uint8_binary(trainX_x) if trainX_x is not None else None
    tX_z = _as_uint8_binary(trainX_z) if trainX_z is not None else None

    basis = basis.upper()
    exclude_round0_x = (basis == "X")
    exclude_round0_z = (basis == "Z")

    parity_Z = _as_uint8_binary(parity_matrix_Z).to(z.device)
    parity_X = _as_uint8_binary(parity_matrix_X).to(z.device)
    cache_Z = build_timelike_he_cache(parity_Z)
    cache_X = build_timelike_he_cache(parity_X)

    iters = 0
    prev = None
    while True:
        if iters >= int(max_passes):
            break
        if prev is not None:
            prev_z, prev_x, prev_sx, prev_sz = prev
            if not (
                (z != prev_z).any() | (x != prev_x).any() | (sx != prev_sx).any() |
                (sz != prev_sz).any()
            ):
                break

        prev = (z, x, sx, sz)

        # X errors anticommute with Z stabs -> use trainX_z
        x, sz = _timelike_pass_brickwork_torch(
            x,
            sz,
            parity_Z,
            exclude_round0=exclude_round0_x,
            use_tie_breaker=use_tie_breaker,
            trainX=tX_z,
            cache=cache_Z,
        )
        # Z errors anticommute with X stabs -> use trainX_x
        z, sx = _timelike_pass_brickwork_torch(
            z,
            sx,
            parity_X,
            exclude_round0=exclude_round0_z,
            use_tie_breaker=use_tie_breaker,
            trainX=tX_x,
            cache=cache_X,
        )

        iters += 1

    return z, x, sx, sz, torch.tensor(iters, dtype=torch.int32, device=z.device)


def _cumulative_to_diffs_torch(z_cum: torch.Tensor,
                               x_cum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z_cum = _as_uint8_binary(z_cum)
    x_cum = _as_uint8_binary(x_cum)
    z_pad = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
    x_pad = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
    return (z_pad[:, :-1, :] ^ z_pad[:, 1:, :]), (x_pad[:, :-1, :] ^ x_pad[:, 1:, :])


def apply_weight1_timelike_homological_equivalence_torch(
    z_errors: torch.Tensor,  # (B, T, D2) cumulative
    x_errors: torch.Tensor,  # (B, T, D2) cumulative
    s1s2_x: torch.Tensor,  # (B, T, num_X_stabs)
    s1s2_z: torch.Tensor,  # (B, T, num_Z_stabs)
    parity_matrix_Z: torch.Tensor,
    parity_matrix_X: torch.Tensor,
    distance: int,
    num_he_cycles: int,
    max_passes: int,
    basis: str,
    use_tie_breaker: bool = True,
    trainX_x: Optional[torch.Tensor] = None,  # (B, T, num_X_stabs)
    trainX_z: Optional[torch.Tensor] = None,  # (B, T, num_Z_stabs)
    *,
    cache_Z_spacelike: Optional[SpacelikeHECache] = None,
    cache_X_spacelike: Optional[SpacelikeHECache] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weight-1 timelike homological equivalence over batch and time.

    Flow (all in diffs space after initial conversion):
      1) cumulative -> diffs once
      2) repeat num_he_cycles:
          a) spacelike HE on diffs
          b) timelike weight-1 HE on diffs until convergence
      3) final spacelike cleanup on diffs

    Returns:
        (z_diffs_out, x_diffs_out, s1s2_x_out, s1s2_z_out)
    """
    z_diffs, x_diffs = _cumulative_to_diffs_torch(z_errors, x_errors)
    sx = _as_uint8_binary(s1s2_x)
    sz = _as_uint8_binary(s1s2_z)

    parity_Z = _as_uint8_binary(parity_matrix_Z).to(z_diffs.device)
    parity_X = _as_uint8_binary(parity_matrix_X).to(z_diffs.device)

    # Build spacelike caches once
    if cache_Z_spacelike is None:
        cache_Z_spacelike = build_spacelike_he_cache(
            parity_Z, distance=distance, device=z_diffs.device
        )
    if cache_X_spacelike is None:
        cache_X_spacelike = build_spacelike_he_cache(
            parity_X, distance=distance, device=z_diffs.device
        )

    for _ in range(int(num_he_cycles)):
        # Spacelike HE on diffs
        z_diffs, x_diffs = apply_homological_equivalence_torch_vmap(
            z_diffs,
            x_diffs,
            parity_Z,
            parity_X,
            distance=distance,
            cache_Z=cache_Z_spacelike,
            cache_X=cache_X_spacelike,
        )

        # Timelike weight-1 HE on diffs until convergence
        z_diffs, x_diffs, sx, sz, _ = _apply_timelike_weight1_convergence_torch(
            z_diffs,
            x_diffs,
            sx,
            sz,
            parity_X,
            parity_Z,
            max_passes=max_passes,
            basis=basis,
            use_tie_breaker=use_tie_breaker,
            trainX_x=trainX_x,
            trainX_z=trainX_z,
        )

    # Final spacelike cleanup
    z_diffs, x_diffs = apply_homological_equivalence_torch_vmap(
        z_diffs,
        x_diffs,
        parity_Z,
        parity_X,
        distance=distance,
        cache_Z=cache_Z_spacelike,
        cache_X=cache_X_spacelike,
    )

    return z_diffs, x_diffs, sx, sz
