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
homological equivalence (HE) used in your data-generation/training pipeline:

- Spacelike HE on *diff* frames (canonicalize each (batch, round) independently)
- Timelike HE, weight-1 (brickwork / Trotterized time-pair processing)
- Timelike HE, weight-2 (optional, via ``use_weight2=True``)

Correctness goal
----------------
For realistic SurfaceCode circuits,
this Torch implementation should produce correct HE outputs bit-for-bit when run
on the same inputs.

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

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


def _as_uint8_binary(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x.to(torch.uint8)
    if x.dtype == torch.uint8:
        return x
    return x.to(torch.uint8) & 1


def _ensure_uint8(x: torch.Tensor) -> torch.Tensor:
    """Fast-path for inner functions where the caller already guarantees uint8 {0,1} data."""
    if x.dtype == torch.uint8:
        return x
    return _as_uint8_binary(x)


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

    # Precomputed data for compiled sequential spacelike HE (P2+P3)
    seq_compile_data: Optional[dict] = None


def _compute_layers_greedy(support_masks_cpu_bool: torch.Tensor) -> List[List[int]]:
    """
    Deterministically build disjoint layers in original stabilizer order.
    Mirrors the reference logic used by `weight_reduction_*`.
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
    For a weight-2 boundary stabilizer pair (a,b), return (canonical, other),
    matching `_identify_boundary_orientation` + weight-1 fix logic.
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
    basis: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SpacelikeHECache:
    """Precompute stabilizer metadata for fast spacelike HE."""
    parity_u8 = _as_uint8_binary(parity_matrix)
    num_stabs, D2 = parity_u8.shape
    if distance is None:
        distance = int(int(D2)**0.5)
    d = int(distance)

    if device is None:
        device = parity_u8.device

    parity_cpu = parity_u8.to("cpu")
    support_masks_cpu = (parity_cpu == 1).to(torch.uint8)
    support_sizes_cpu = support_masks_cpu.sum(dim=1, dtype=torch.int64)

    layers_list = _compute_layers_greedy((support_masks_cpu == 1))

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
            coords = sorted([(i // d, i % d, i) for i in idx])
            w4_tl_cpu[s] = coords[0][2]
            w4_tr_cpu[s] = coords[1][2]
            w4_bl_cpu[s] = coords[2][2]
            w4_br_cpu[s] = coords[3][2]

    parity_dev = parity_u8.to(device)
    support_masks = (parity_dev == 1).to(torch.uint8)
    support_sizes = support_masks.sum(dim=1, dtype=torch.int64)
    layers = tuple(torch.tensor(layer, dtype=torch.int64, device=device) for layer in layers_list)

    cache = SpacelikeHECache(
        distance=d,
        parity=parity_dev,
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

    if basis is not None:
        scd = _build_seq_compile_data(cache, basis, device)
        object.__setattr__(cache, "seq_compile_data", scd)

    return cache


# ---------------------------------------------------------------------------
# Coset min-weight search for stuck patterns (NEW-4 / P12)
# ---------------------------------------------------------------------------


def _build_coset_generators(parity: torch.Tensor) -> torch.Tensor:
    """Build the stabilizer generator matrix for coset enumeration.

    Returns (S, D2) uint8 on CPU — each row is one stabilizer's support.
    Only feasible for small S (d <= 7 → S <= 24).
    """
    return _as_uint8_binary(parity).cpu()


def coset_minimum_weight(
    cfg: torch.Tensor,
    parity: torch.Tensor,
    *,
    max_generators: int = 20,
) -> torch.Tensor:
    """Replace each error with the minimum-weight coset representative.

    For each batch element, enumerates all 2^S XOR combinations of stabilizer
    rows and picks the one with the lowest Hamming weight that has the same
    syndrome.

    Args:
        cfg: (N, D2) uint8 binary error patterns
        parity: (S, D2) uint8 parity matrix
        max_generators: Skip if S exceeds this (exponential cost guard)

    Returns:
        (N, D2) uint8 — minimum-weight coset representatives
    """
    cfg = _as_uint8_binary(cfg)
    par = _as_uint8_binary(parity)
    S, D2 = par.shape
    N = cfg.shape[0]

    if S > max_generators or S == 0:
        return cfg

    num_cosets = 1 << S
    # Build all 2^S group elements by iterating Gray code-style XORs
    generators = par.to(cfg.device)
    group = torch.zeros(num_cosets, D2, dtype=torch.uint8, device=cfg.device)
    current = torch.zeros(D2, dtype=torch.uint8, device=cfg.device)
    group[0] = current
    for i in range(1, num_cosets):
        bit = (i & -i).bit_length() - 1
        current = current ^ generators[bit]
        group[i] = current

    # For each batch element, XOR with all group elements: (N, 2^S, D2)
    # Then pick the minimum-weight version.
    # Process in chunks to avoid OOM for larger N.
    chunk_size = max(1, min(N, 4096 // num_cosets))
    result = torch.empty_like(cfg)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        batch = cfg[start:end]  # (C, D2)
        candidates = batch.unsqueeze(1) ^ group.unsqueeze(0)  # (C, 2^S, D2)
        weights = candidates.sum(dim=2)  # (C, 2^S)
        best = weights.argmin(dim=1)  # (C,)
        result[start:end] = candidates[torch.arange(end - start, device=cfg.device), best]

    return result


def _simplify_time_w1_step_nobreak(
    err: torch.Tensor,
    syn: torch.Tensor,
    pf: torch.Tensor,
    pfT: torch.Tensor,
    pf_col_sum: torch.Tensor,
    trainX_d0: torch.Tensor,
    trainX_d1: torch.Tensor,
    accept_fn: "callable",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    torch.compile-friendly single-step timelike w1 reduction on a 2-round window.
    All float, no .item(), uses algebraic identity for round-0 contrib flip.

    err: (B, D2, 2) float  —  note: (B, D2, T) layout for compile (OPT-10)
    syn: (B, S, 2) float
    pf:  (S, D2) float
    pfT: (D2, S) float
    pf_col_sum: (D2,) float
    trainX_d0/d1: (B, D2) float
    accept_fn: overlap resolver, signature (accept_raw, pf) -> accept
    """
    old_contrib = torch.einsum("bst,sd->bdt", syn, pf)
    old_d0 = err[:, :, 0] + old_contrib[:, :, 0] + trainX_d0
    old_d1 = err[:, :, 1] + old_contrib[:, :, 1] + trainX_d1

    new_d0 = (1 - err[:, :, 0]) + (pf_col_sum - old_contrib[:, :, 0]) + trainX_d0
    new_d1 = (1 - err[:, :, 1]) + old_contrib[:, :, 1] + trainX_d1

    old_density = old_d0 + old_d1
    new_density = new_d0 + new_d1

    old_max = torch.maximum(old_d0, old_d1)
    new_max = torch.maximum(new_d0, new_d1)
    accept_raw = (new_density < old_density) | ((new_density == old_density) & (new_max > old_max))

    accept = accept_fn(accept_raw, pf)

    mask = accept.unsqueeze(2)
    err_out = torch.where(mask, 1 - err, err)

    flip_count = accept.float() @ pfT
    should_flip = (flip_count % 2).bool()
    syn_out = syn.clone()
    syn_out[:, :, 0] = torch.where(should_flip, 1 - syn[:, :, 0], syn[:, :, 0])

    return err_out, syn_out


def _weight_reduction(cfg: torch.Tensor, cache: SpacelikeHECache) -> torch.Tensor:
    """
    Weight reduction (parallel within disjoint stabilizer layers).

    cfg: (N, D2) uint8 in {0,1}
    """
    cfg = _ensure_uint8(cfg)
    cfg_i8 = cfg.to(torch.int8)
    support_masks_i8 = cache.support_masks.to(torch.int8)  # (S, D2)
    support_sizes = cache.support_sizes  # (S,)

    # Int8 matmul is ~2x faster than float32 on GPU, but has two failure modes:
    #  1) RuntimeError on backends that don't support int8 GEMM (older PyTorch,
    #     certain devices, or torch.compile Triton failures).
    #  2) Silent overflow when the accumulation dimension D2 >= 128 (distance >= 12),
    #     since int8 accumulators wrap at [-128, 127].
    # The try/except below catches case (1) and falls back to float32 for the
    # rest of the call.  Case (2) is safe here because error_counts values are
    # at most 4 (stabilizer support size) and act1/act2 are bool→int8 with at
    # most L ones, so intermediate sums stay well within int8 range as long as
    # L < 128 (true for practical surface code distances).
    _use_int8 = True

    for layer_idx in cache.layers:
        if layer_idx.numel() == 0:
            continue
        masks_i8 = support_masks_i8.index_select(0, layer_idx)  # (L, D2) int8
        sizes = support_sizes.index_select(0, layer_idx)  # (L,)

        if _use_int8:
            try:
                error_counts = (cfg_i8 @ masks_i8.t()).to(torch.int32)
                act1 = (error_counts == 4) | ((error_counts == 2) & (sizes.unsqueeze(0) == 2))
                act2 = (error_counts == 3)
                set_to_zero_mask = ((act1.to(torch.int8) @ masks_i8).to(torch.int32) > 0)
                flip_mask = ((act2.to(torch.int8) @ masks_i8).to(torch.int32)
                             > 0) & (~set_to_zero_mask)
            except RuntimeError as exc:
                warnings.warn(
                    f"Int8 GEMM failed, falling back to float32 for weight reduction: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _use_int8 = False
                masks_f = cache.support_masks.to(torch.float32).index_select(0, layer_idx)
                error_counts = (cfg.to(torch.float32) @ masks_f.t()).to(torch.int32)
                act1 = (error_counts == 4) | ((error_counts == 2) & (sizes.unsqueeze(0) == 2))
                act2 = (error_counts == 3)
                set_to_zero_mask = ((act1.to(torch.float32) @ masks_f) > 0)
                flip_mask = ((act2.to(torch.float32) @ masks_f) > 0) & (~set_to_zero_mask)
        else:
            masks_f = cache.support_masks.to(torch.float32).index_select(0, layer_idx)
            error_counts = (cfg.to(torch.float32) @ masks_f.t()).to(torch.int32)
            act1 = (error_counts == 4) | ((error_counts == 2) & (sizes.unsqueeze(0) == 2))
            act2 = (error_counts == 3)
            set_to_zero_mask = ((act1.to(torch.float32) @ masks_f) > 0)
            flip_mask = ((act2.to(torch.float32) @ masks_f) > 0) & (~set_to_zero_mask)

        cfg = cfg * (~set_to_zero_mask).to(cfg.dtype)
        cfg = cfg ^ flip_mask.to(cfg.dtype)
        cfg_i8 = cfg.to(torch.int8)

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
    cfg = _ensure_uint8(cfg)
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


# ---------------------------------------------------------------------------
# P2+P3+P5: Compiled sequential spacelike HE
# ---------------------------------------------------------------------------
# These functions replicate the exact sequential algorithm (_weight_reduction
# + _fix_equivalence) but in forms optimized for GPU execution:
#
# Weight-reduction: torch.compile with mode="reduce-overhead" (CUDA graphs)
#   fuses all layer operations into a single kernel dispatch.
#
# Fix-equivalence: Manual CUDA graph capture of the branchless sequential
#   stabilizer loop.  The Python for-loop over stabilizers produces hundreds
#   of tiny kernel launches; recording them into a CUDA graph and replaying
#   eliminates per-launch overhead entirely.
#
# _build_seq_compile_data()        – precompute once per cache (at init time)
# _wr_seq_step_nobreak()           – weight-reduction for all layers (all-float, compiled)
# _fe_seq_step_nobreak()           – fix-equivalence for all stabilizers (all-float, eager)
# _get_compiled_seq_wr()           – returns compiled WR function
# _build_fe_cuda_graph()           – capture FE as CUDA graph (P5)
# _simplify_spacelike_seq_compiled – hybrid compiled-WR + CUDA-graph-FE loop


def _build_seq_compile_data(
    cache: SpacelikeHECache,
    basis: str,
    device: torch.device,
) -> dict:
    """Precompute flat tensors for the compiled sequential spacelike path.

    Called once at cache-build time.  The returned dict contains everything
    needed by _wr_seq_step_nobreak and _fe_seq_step_nobreak so that the
    compiled function takes only tensor arguments (no Python objects).
    """
    basis = basis.upper()
    num_stabs = int(cache.support_sizes.numel())

    # --- Weight-reduction layer data ---
    # Pack all layers into padded tensors so the compiled function can
    # iterate over a fixed number of layers with a static loop.
    layer_masks_list: list[torch.Tensor] = []
    layer_sizes_list: list[torch.Tensor] = []
    max_layer_size = 0
    for layer_idx in cache.layers:
        if layer_idx.numel() == 0:
            continue
        masks = cache.support_masks.index_select(0, layer_idx)  # (L, D2) uint8
        sizes = cache.support_sizes.index_select(0, layer_idx)  # (L,)
        layer_masks_list.append(masks)
        layer_sizes_list.append(sizes)
        max_layer_size = max(max_layer_size, masks.shape[0])

    num_layers = len(layer_masks_list)
    D2 = cache.support_masks.shape[1]

    # Pad to (num_layers, max_layer_size, D2) and (num_layers, max_layer_size)
    # with a valid_mask (num_layers, max_layer_size) to ignore padding.
    padded_masks = torch.zeros(num_layers, max_layer_size, D2, dtype=torch.float32, device=device)
    padded_sizes = torch.zeros(num_layers, max_layer_size, dtype=torch.float32, device=device)
    layer_valid = torch.zeros(num_layers, max_layer_size, dtype=torch.float32, device=device)

    for i, (m, s) in enumerate(zip(layer_masks_list, layer_sizes_list)):
        L = m.shape[0]
        padded_masks[i, :L] = m.to(torch.float32)
        padded_sizes[i, :L] = s.to(torch.float32)
        layer_valid[i, :L] = 1.0

    is_boundary = (padded_sizes == 2.0)  # (num_layers, max_layer_size)

    # --- Fix-equivalence stabilizer data (unified flat layout) ---
    # Walk stabilizers in original order.  Build a flat list where each
    # active stabilizer (w2 or w4) has one entry with:
    #   seq_types[i] = 1.0 (w2) or 0.0 (w4)   — float, no branching
    #   seq_q0..q3[i] = qubit indices (for w2: canonical, other, 0, 0)
    #                    (for w4: tl, tr, bl, br — used for claimed update)
    #   seq_w4_s0..d1[i, 0..2] = src/dst per pattern (for w2: zero-filled)

    d = cache.distance
    seq_types_list: list[float] = []
    seq_q0_list: list[int] = []
    seq_q1_list: list[int] = []
    seq_q2_list: list[int] = []
    seq_q3_list: list[int] = []
    seq_w4_s0_list: list[list[int]] = []
    seq_w4_s1_list: list[list[int]] = []
    seq_w4_d0_list: list[list[int]] = []
    seq_w4_d1_list: list[list[int]] = []

    for s in range(num_stabs):
        ss = int(cache.support_sizes[s].item())
        if ss == 2:
            canonical = int(cache.w2_canonical[s].item())
            other = int(cache.w2_other[s].item())
            if canonical < 0 or other < 0:
                continue
            seq_types_list.append(1.0)
            seq_q0_list.append(canonical)
            seq_q1_list.append(other)
            seq_q2_list.append(0)
            seq_q3_list.append(0)
            seq_w4_s0_list.append([0, 0, 0])
            seq_w4_s1_list.append([0, 0, 0])
            seq_w4_d0_list.append([0, 0, 0])
            seq_w4_d1_list.append([0, 0, 0])
        elif ss == 4:
            tl = int(cache.w4_tl[s].item())
            if tl < 0:
                continue
            tr = int(cache.w4_tr[s].item())
            bl = int(cache.w4_bl[s].item())
            br = int(cache.w4_br[s].item())
            seq_types_list.append(0.0)
            seq_q0_list.append(tl)
            seq_q1_list.append(tr)
            seq_q2_list.append(bl)
            seq_q3_list.append(br)

            if basis == "X":
                seq_w4_s0_list.append([tl, bl, tl])
                seq_w4_s1_list.append([bl, br, br])
                seq_w4_d0_list.append([tr, tl, tr])
                seq_w4_d1_list.append([br, tr, bl])
            else:
                seq_w4_s0_list.append([tl, bl, tr])
                seq_w4_s1_list.append([bl, br, bl])
                seq_w4_d0_list.append([tr, tl, tl])
                seq_w4_d1_list.append([br, tr, br])

    n_entries = len(seq_types_list)

    if n_entries > 0:
        seq_types_t = torch.tensor(seq_types_list, dtype=torch.float32, device=device)
        seq_q0_t = torch.tensor(seq_q0_list, dtype=torch.long, device=device)
        seq_q1_t = torch.tensor(seq_q1_list, dtype=torch.long, device=device)
        seq_q2_t = torch.tensor(seq_q2_list, dtype=torch.long, device=device)
        seq_q3_t = torch.tensor(seq_q3_list, dtype=torch.long, device=device)
        seq_w4_s0_t = torch.tensor(seq_w4_s0_list, dtype=torch.long, device=device)
        seq_w4_s1_t = torch.tensor(seq_w4_s1_list, dtype=torch.long, device=device)
        seq_w4_d0_t = torch.tensor(seq_w4_d0_list, dtype=torch.long, device=device)
        seq_w4_d1_t = torch.tensor(seq_w4_d1_list, dtype=torch.long, device=device)
    else:
        seq_types_t = torch.zeros(0, dtype=torch.float32, device=device)
        seq_q0_t = seq_q1_t = seq_q2_t = seq_q3_t = torch.zeros(0, dtype=torch.long, device=device)
        seq_w4_s0_t = seq_w4_s1_t = torch.zeros(0, 3, dtype=torch.long, device=device)
        seq_w4_d0_t = seq_w4_d1_t = torch.zeros(0, 3, dtype=torch.long, device=device)

    return dict(
        num_layers=num_layers,
        padded_masks=padded_masks,
        is_boundary=is_boundary,
        layer_valid=layer_valid,
        n_entries=n_entries,
        seq_types=seq_types_t,
        seq_q0=seq_q0_t,
        seq_q1=seq_q1_t,
        seq_q2=seq_q2_t,
        seq_q3=seq_q3_t,
        seq_w4_s0=seq_w4_s0_t,
        seq_w4_s1=seq_w4_s1_t,
        seq_w4_d0=seq_w4_d0_t,
        seq_w4_d1=seq_w4_d1_t,
    )


def _wr_seq_step_nobreak(
    error_f: torch.Tensor,
    padded_masks: torch.Tensor,
    is_boundary: torch.Tensor,
    layer_valid: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    """Compile-friendly sequential weight-reduction (all-float, no early exit).

    Processes disjoint layers in order, matching _weight_reduction exactly.
    error_f:       (N, D2) float
    padded_masks:  (num_layers, max_layer_size, D2) float
    is_boundary:   (num_layers, max_layer_size) bool
    layer_valid:   (num_layers, max_layer_size) float
    """
    for i in range(num_layers):
        masks = padded_masks[i]  # (max_layer_size, D2) float
        bnd = is_boundary[i]  # (max_layer_size,) bool
        valid = layer_valid[i]  # (max_layer_size,) float

        counts = error_f @ masks.T  # (N, max_layer_size)
        # Mask invalid entries so they never trigger actions
        counts = counts * valid.unsqueeze(0)

        act1 = ((counts == 4.0) | ((counts == 2.0) & bnd.unsqueeze(0))).float()
        act2 = (counts == 3.0).float()
        # Zero out invalid stabilizers
        act1 = act1 * valid.unsqueeze(0)
        act2 = act2 * valid.unsqueeze(0)

        zero_mask = (act1 @ masks).clamp(max=1.0)
        flip_raw = (act2 @ masks).clamp(max=1.0)
        flip_mask = flip_raw * (1.0 - zero_mask)

        error_f = error_f * (1.0 - zero_mask)
        error_f = error_f - 2.0 * error_f * flip_mask + flip_mask

    return error_f


def _fe_seq_step_nobreak(
    error_f: torch.Tensor,
    seq_types: torch.Tensor,
    seq_q0: torch.Tensor,
    seq_q1: torch.Tensor,
    seq_q2: torch.Tensor,
    seq_q3: torch.Tensor,
    seq_w4_s0: torch.Tensor,
    seq_w4_s1: torch.Tensor,
    seq_w4_d0: torch.Tensor,
    seq_w4_d1: torch.Tensor,
    num_entries: int,
) -> torch.Tensor:
    """Optimized sequential fix-equivalence using precomputed index tensors.

    Processes stabilizers in exact original order with a branching loop (this
    runs eagerly, not compiled). Each w2 entry does ~6 kernel launches and each
    w4 entry does ~18, versus ~40 per entry in the branchless version.
    """
    N, D2 = error_f.shape
    claimed_f = torch.zeros_like(error_f)

    for s in range(num_entries):
        is_w2 = seq_types[s].item() > 0.5

        if is_w2:
            ci_idx = seq_q0[s].unsqueeze(0).unsqueeze(0).expand(N, 1)
            oi_idx = seq_q1[s].unsqueeze(0).unsqueeze(0).expand(N, 1)

            v_can = torch.gather(error_f, 1, ci_idx).squeeze(1)
            v_oth = torch.gather(error_f, 1, oi_idx).squeeze(1)

            error_count = v_can + v_oth
            c_can = torch.gather(claimed_f, 1, ci_idx).squeeze(1)
            c_oth = torch.gather(claimed_f, 1, oi_idx).squeeze(1)
            has_overlap = ((v_can > 0.5) & (c_can > 0.5)) | ((v_oth > 0.5) & (c_oth > 0.5))

            should_process = (error_count == 1.0) & (~has_overlap)
            should_move = should_process & (v_can < 0.5)
            move_f = should_move.float().unsqueeze(1)

            error_f = error_f.scatter(1, ci_idx, (v_can.unsqueeze(1) * (1.0 - move_f) + move_f))
            error_f = error_f.scatter(1, oi_idx, (v_oth.unsqueeze(1) * (1.0 - move_f)))

            proc_f = should_process.float().unsqueeze(1)
            claimed_f = claimed_f.scatter(
                1, ci_idx, (torch.gather(claimed_f, 1, ci_idx) + proc_f).clamp(max=1.0)
            )
            claimed_f = claimed_f.scatter(
                1, oi_idx, (torch.gather(claimed_f, 1, oi_idx) + proc_f).clamp(max=1.0)
            )

        else:
            corner_idx = torch.stack([seq_q0[s], seq_q1[s], seq_q2[s],
                                      seq_q3[s]]).unsqueeze(0).expand(N, 4)
            sub = torch.gather(error_f, 1, corner_idx)
            error_count = sub.sum(dim=1)

            claimed_sub = torch.gather(claimed_f, 1, corner_idx)
            has_overlap = ((sub > 0.5) & (claimed_sub > 0.5)).any(dim=1)

            should_process = (error_count == 2.0) & (~has_overlap)
            sp_f = should_process.float()

            handled = torch.zeros(N, dtype=torch.float32, device=error_f.device)
            for p in range(3):
                s0_idx = seq_w4_s0[s, p].unsqueeze(0).unsqueeze(0).expand(N, 1)
                s1_idx = seq_w4_s1[s, p].unsqueeze(0).unsqueeze(0).expand(N, 1)
                d0_idx = seq_w4_d0[s, p].unsqueeze(0).unsqueeze(0).expand(N, 1)
                d1_idx = seq_w4_d1[s, p].unsqueeze(0).unsqueeze(0).expand(N, 1)

                vs0 = torch.gather(error_f, 1, s0_idx).squeeze(1)
                vs1 = torch.gather(error_f, 1, s1_idx).squeeze(1)
                vd0 = torch.gather(error_f, 1, d0_idx).squeeze(1)
                vd1 = torch.gather(error_f, 1, d1_idx).squeeze(1)

                match = sp_f * (1.0 - handled) * vs0 * vs1 * (1.0 - vd0) * (1.0 - vd1)
                match_1 = match.unsqueeze(1)

                error_f = error_f.scatter(
                    1, s0_idx,
                    torch.gather(error_f, 1, s0_idx) * (1.0 - match_1)
                )
                error_f = error_f.scatter(
                    1, s1_idx,
                    torch.gather(error_f, 1, s1_idx) * (1.0 - match_1)
                )
                error_f = error_f.scatter(
                    1, d0_idx,
                    torch.gather(error_f, 1, d0_idx) * (1.0 - match_1) + match_1
                )
                error_f = error_f.scatter(
                    1, d1_idx,
                    torch.gather(error_f, 1, d1_idx) * (1.0 - match_1) + match_1
                )

                handled = handled + match

            moved_f = (handled > 0.5).float().unsqueeze(1)
            for ci in range(4):
                qi = corner_idx[:, ci:ci + 1]
                claimed_f = claimed_f.scatter(
                    1, qi, (torch.gather(claimed_f, 1, qi) + moved_f).clamp(max=1.0)
                )

    return error_f


_compiled_seq_wr_cache: dict = {}


def _get_compiled_seq_wr(num_layers: int):
    """Return a compiled weight-reduction function for the sequential path.

    Only wraps _wr_seq_step_nobreak (small, regular, ~50 ops per layer).
    Fix-equivalence uses CUDA graph capture instead (see _build_fe_cuda_graph).
    """
    key = ("seq_wr", num_layers)
    if key in _compiled_seq_wr_cache:
        return _compiled_seq_wr_cache[key]

    nl = num_layers

    def _wr_fn(error_f, padded_masks, is_boundary, layer_valid):
        return _wr_seq_step_nobreak(error_f, padded_masks, is_boundary, layer_valid, nl)

    compiled = torch.compile(_wr_fn, mode="reduce-overhead", fullgraph=True)
    _compiled_seq_wr_cache[key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# P5: CUDA-graph-captured fix-equivalence
# ---------------------------------------------------------------------------
# The sequential FE loop launches hundreds of tiny kernels (one per stabilizer
# per operation).  Recording this exact kernel sequence into a CUDA graph at
# warmup time and replaying it eliminates all per-launch overhead.
#
# Requirements for CUDA graph capture:
#   - No dynamic memory allocation (no fancy indexing like cfg[:, (a,b)])
#   - No CPU-GPU synchronization (.item(), .any() used for control flow)
#   - All tensor shapes must be static across replays
#
# The branchless FE function uses direct column access (cfg[:, col]) and
# torch.where instead of conditional writes, satisfying all constraints.

_fe_cuda_graph_cache: dict = {}


def _make_branchless_fe_fn(
    stab_ops: list,
    basis: str,
):
    """Build a branchless FE closure suitable for CUDA graph capture.

    stab_ops is a list of ('w2', canonical, other) or ('w4', tl, tr, bl, br)
    tuples with pure Python integers (no tensor .item() calls at runtime).
    """
    basis = basis.upper()

    def _fe_branchless(cfg: torch.Tensor, claimed: torch.Tensor) -> None:
        for op in stab_ops:
            if op[0] == "w2":
                _, can, oth = op
                v_can = cfg[:, can]
                v_oth = cfg[:, oth]
                ec = v_can.to(torch.int32) + v_oth.to(torch.int32)
                c_can = claimed[:, can]
                c_oth = claimed[:, oth]
                has_ov = (v_can.bool() & c_can) | (v_oth.bool() & c_oth)
                should_process = (ec == 1) & (~has_ov)
                should_move = should_process & (v_can != 1)
                cfg[:, can] = torch.where(should_move, torch.ones_like(v_can), v_can)
                cfg[:, oth] = torch.where(should_move, torch.zeros_like(v_oth), v_oth)
                claimed[:, can] = c_can | should_move
                claimed[:, oth] = c_oth | should_move
            else:
                _, tl, tr, bl, br = op
                vtl = cfg[:, tl]
                vtr = cfg[:, tr]
                vbl = cfg[:, bl]
                vbr = cfg[:, br]
                ec = (
                    vtl.to(torch.int32) + vtr.to(torch.int32) + vbl.to(torch.int32) +
                    vbr.to(torch.int32)
                )
                ctl = claimed[:, tl]
                ctr = claimed[:, tr]
                cbl = claimed[:, bl]
                cbr = claimed[:, br]
                has_ov = (
                    (vtl.bool() & ctl) | (vtr.bool() & ctr) | (vbl.bool() & cbl) |
                    (vbr.bool() & cbr)
                )
                sp = (ec == 2) & (~has_ov)
                tl1 = vtl == 1
                tr1 = vtr == 1
                bl1 = vbl == 1
                br1 = vbr == 1
                if basis == "X":
                    m1 = sp & tl1 & bl1 & (~tr1) & (~br1)
                    m2 = sp & bl1 & br1 & (~tl1) & (~tr1)
                    m3 = sp & tl1 & br1 & (~tr1) & (~bl1)
                    moved = m1 | m2 | m3
                    cfg[:, tl] = torch.where(
                        m2, torch.ones_like(vtl), torch.where(m1 | m3, torch.zeros_like(vtl), vtl)
                    )
                    cfg[:, tr] = torch.where(moved, torch.ones_like(vtr), vtr)
                    cfg[:, bl] = torch.where(
                        m3, torch.ones_like(vbl), torch.where(m1 | m2, torch.zeros_like(vbl), vbl)
                    )
                    cfg[:, br] = torch.where(
                        m1, torch.ones_like(vbr), torch.where(m2 | m3, torch.zeros_like(vbr), vbr)
                    )
                else:
                    m1 = sp & tl1 & bl1 & (~tr1) & (~br1)
                    m2 = sp & bl1 & br1 & (~tl1) & (~tr1)
                    m3 = sp & tr1 & bl1 & (~tl1) & (~br1)
                    moved = m1 | m2 | m3
                    cfg[:, tl] = torch.where(
                        m2 | m3, torch.ones_like(vtl), torch.where(m1, torch.zeros_like(vtl), vtl)
                    )
                    cfg[:, tr] = torch.where(
                        m1 | m2, torch.ones_like(vtr), torch.where(m3, torch.zeros_like(vtr), vtr)
                    )
                    cfg[:, bl] = torch.where(moved, torch.zeros_like(vbl), vbl)
                    cfg[:, br] = torch.where(
                        m1 | m3, torch.ones_like(vbr), torch.where(m2, torch.zeros_like(vbr), vbr)
                    )
                claimed[:, tl] = ctl | moved
                claimed[:, tr] = ctr | moved
                claimed[:, bl] = cbl | moved
                claimed[:, br] = cbr | moved

    return _fe_branchless


def _extract_stab_ops(cache: SpacelikeHECache) -> list:
    """Extract stabilizer operations as pure Python tuples (no tensor .item() at runtime)."""
    num_stabs = int(cache.support_sizes.numel())
    ops: list = []
    for s in range(num_stabs):
        ss = int(cache.support_sizes[s].item())
        if ss == 2:
            canonical = int(cache.w2_canonical[s].item())
            other = int(cache.w2_other[s].item())
            if canonical >= 0 and other >= 0:
                ops.append(("w2", canonical, other))
        elif ss == 4:
            tl = int(cache.w4_tl[s].item())
            if tl >= 0:
                tr = int(cache.w4_tr[s].item())
                bl = int(cache.w4_bl[s].item())
                br = int(cache.w4_br[s].item())
                ops.append(("w4", tl, tr, bl, br))
    return ops


def _build_fe_cuda_graph(
    cache: SpacelikeHECache,
    basis: str,
    N: int,
    device: torch.device,
) -> dict:
    """Capture fix-equivalence as a CUDA graph for the given batch size.

    Returns a dict with 'graph', 'cfg_static', 'claimed_static', 'fe_fn'.
    The graph is replayed by copying input into cfg_static, zeroing
    claimed_static, calling graph.replay(), then reading cfg_static.
    """
    D2 = int(cache.parity.shape[1])
    stab_ops = _extract_stab_ops(cache)
    fe_fn = _make_branchless_fe_fn(stab_ops, basis)

    cfg_static = torch.zeros(N, D2, dtype=torch.uint8, device=device)
    claimed_static = torch.zeros(N, D2, dtype=torch.bool, device=device)

    for _ in range(3):
        fe_fn(cfg_static, claimed_static)
        cfg_static.zero_()
        claimed_static.zero_()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fe_fn(cfg_static, claimed_static)

    return {
        "graph": g,
        "cfg_static": cfg_static,
        "claimed_static": claimed_static,
        "fe_fn": fe_fn,
        "N": N,
        "D2": D2,
    }


def _get_fe_cuda_graph(
    cache: SpacelikeHECache,
    basis: str,
    N: int,
    device: torch.device,
) -> dict:
    """Return a cached CUDA graph for fix-equivalence, building on first call."""
    key = (id(cache), basis.upper(), N)
    if key not in _fe_cuda_graph_cache:
        _fe_cuda_graph_cache[key] = _build_fe_cuda_graph(cache, basis, N, device)
    return _fe_cuda_graph_cache[key]


def _simplify_spacelike_seq_compiled(
    cfg: torch.Tensor,
    cache: SpacelikeHECache,
    max_iterations: int = 100,
    basis: str = "X",
) -> torch.Tensor:
    """Run sequential spacelike HE with compiled WR + CUDA-graph-captured FE.

    Weight-reduction: torch.compile fuses layer ops into one kernel dispatch.
    Fix-equivalence: CUDA graph replay eliminates per-stabilizer launch overhead
    (hundreds of tiny kernels → single graph replay, ~3x faster than eager FE).

    Falls back to eager FE on CPU or if CUDA graph capture fails.
    """
    scd = cache.seq_compile_data
    if scd is None:
        raise ValueError("seq_compile_data not built — pass basis= to build_spacelike_he_cache")

    num_layers = scd["num_layers"]
    wr_fn = _get_compiled_seq_wr(num_layers)

    if cfg.dtype != torch.uint8:
        cfg = _as_uint8_binary(cfg)

    N = cfg.shape[0]
    use_graph = cfg.is_cuda

    fe_graph_data: Optional[dict] = None
    if use_graph:
        try:
            fe_graph_data = _get_fe_cuda_graph(cache, basis, N, cfg.device)
        except Exception:
            fe_graph_data = None

    prev = torch.empty_like(cfg)
    for _ in range(int(max_iterations)):
        prev.copy_(cfg)

        torch.compiler.cudagraph_mark_step_begin()
        cfg_f = wr_fn(
            cfg.to(torch.float32),
            scd["padded_masks"],
            scd["is_boundary"],
            scd["layer_valid"],
        ).clone()
        cfg = cfg_f.round().to(torch.uint8)

        if fe_graph_data is not None:
            gd = fe_graph_data
            gd["cfg_static"].copy_(cfg)
            gd["claimed_static"].zero_()
            gd["graph"].replay()
            cfg.copy_(gd["cfg_static"])
        else:
            cfg = _fix_equivalence(cfg, cache, basis=basis)

        if torch.equal(cfg, prev):
            break

    return cfg


def _simplify_spacelike(
    cfg: torch.Tensor,
    cache: SpacelikeHECache,
    *,
    basis: str,
    max_iterations: int = 100,
    use_compile: bool = False,
    compute_dtype: torch.dtype = torch.float32,
    use_coset_search: bool = False,
    parity: Optional[torch.Tensor] = None,
    coset_max_generators: int = 20,
) -> torch.Tensor:
    if use_compile and cache.seq_compile_data is not None:
        cfg = _simplify_spacelike_seq_compiled(
            cfg, cache, max_iterations=max_iterations, basis=basis
        )
    else:
        if cfg.dtype != torch.uint8:
            cfg = _as_uint8_binary(cfg)
        prev = torch.empty_like(cfg)
        for _ in range(int(max_iterations)):
            prev.copy_(cfg)
            cfg = _weight_reduction(cfg, cache)
            cfg = _fix_equivalence(cfg, cache, basis=basis)
            if torch.equal(cfg, prev):
                break

    if use_coset_search and parity is not None:
        par_u8 = parity if parity.dtype == torch.uint8 else _as_uint8_binary(parity)
        cfg = coset_minimum_weight(cfg, par_u8, max_generators=coset_max_generators)

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
    use_compile: bool = False,
    compute_dtype: torch.dtype = torch.float32,
    use_coset_search: bool = False,
    coset_max_generators: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch spacelike HE implementation.

    Takes diff frames and canonicalizes each diff independently.

    Args:
        use_coset_search: If True, after greedy canonicalization, enumerate all
            coset representatives and pick the minimum-weight one (P12 / NEW-4).
        coset_max_generators: Guard against exponential blowup — skip coset
            search if the stabilizer count exceeds this value.
    """
    z = _as_uint8_binary(z_diffs)
    x = _as_uint8_binary(x_diffs)
    B, T, D2 = z.shape
    if distance is None:
        distance = int(int(D2)**0.5)
    d = int(distance)

    parity_Z = _as_uint8_binary(parity_matrix_Z).to(z.device)
    parity_X = _as_uint8_binary(parity_matrix_X).to(x.device)

    if cache_Z is None:
        cache_Z = build_spacelike_he_cache(parity_Z, distance=d, basis="Z", device=z.device)
    if cache_X is None:
        cache_X = build_spacelike_he_cache(parity_X, distance=d, basis="X", device=x.device)

    z_flat = z.reshape(B * T, D2)
    x_flat = x.reshape(B * T, D2)

    x_can = _simplify_spacelike(
        x_flat,
        cache_X,
        basis="X",
        max_iterations=max_iterations,
        use_compile=use_compile,
        compute_dtype=compute_dtype,
        use_coset_search=use_coset_search,
        parity=parity_X,
        coset_max_generators=coset_max_generators,
    )
    z_can = _simplify_spacelike(
        z_flat,
        cache_Z,
        basis="Z",
        max_iterations=max_iterations,
        use_compile=use_compile,
        compute_dtype=compute_dtype,
        use_coset_search=use_coset_search,
        parity=parity_Z,
        coset_max_generators=coset_max_generators,
    )

    return z_can.reshape(B, T, D2), x_can.reshape(B, T, D2)


# -----------------------------------------------------------------------------
# Timelike HE
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


@dataclass(frozen=True)
class Weight2TimelikeCache:
    """Precomputed data for weight-2 timelike HE reductions.

    For each weight-4 stabilizer in the *same-type* parity matrix, stores the
    qubit pairs and their anticommuting stabilizer indices from the *conjugate*
    parity matrix.
    """
    num_w4: int
    num_patterns: int
    qubit_pairs: torch.Tensor  # (num_w4, num_patterns, 2)
    anti_stab_indices: torch.Tensor  # (num_w4, num_patterns, max_anti)
    anti_stab_counts: torch.Tensor  # (num_w4, num_patterns)


def build_weight2_timelike_cache(
    same_parity: torch.Tensor,
    conjugate_parity: torch.Tensor,
    distance: int,
    error_type: str,
    device: torch.device,
) -> Weight2TimelikeCache:
    """Precompute qubit pairs and anticommuting stabilizer map for weight-2 timelike HE."""
    same_u8 = _as_uint8_binary(same_parity).cpu()
    conj_u8 = _as_uint8_binary(conjugate_parity).cpu()
    num_stabs = same_u8.shape[0]
    d = int(distance)

    all_pairs: List[torch.Tensor] = []
    all_anti_lists: List[list] = []

    for s in range(num_stabs):
        support = torch.nonzero(same_u8[s], as_tuple=True)[0].tolist()
        if len(support) != 4:
            continue

        coords = sorted([(idx // d, idx % d, idx) for idx in support])
        tl, tr, bl, br = coords[0], coords[1], coords[2], coords[3]

        if error_type.upper() == "X":
            pairs = [
                (tr[2], br[2]),  # vertical: right column
                (tl[2], tr[2]),  # horizontal: top row
                (tr[2], bl[2]),  # diagonal: X-canonical
            ]
        else:
            pairs = [
                (tl[2], tr[2]),  # horizontal: top row
                (tr[2], br[2]),  # vertical: right column
                (tl[2], br[2]),  # diagonal: Z-canonical
            ]

        anti_lists = []
        for q1, q2 in pairs:
            overlap = (conj_u8[:, q1].int() + conj_u8[:, q2].int()) % 2
            anti_indices = torch.nonzero(overlap, as_tuple=True)[0].tolist()
            anti_lists.append(anti_indices)

        all_pairs.append(torch.tensor(pairs, dtype=torch.int64))
        all_anti_lists.append(anti_lists)

    num_w4 = len(all_pairs)
    num_patterns = 3

    if num_w4 == 0:
        return Weight2TimelikeCache(
            num_w4=0,
            num_patterns=num_patterns,
            qubit_pairs=torch.zeros((0, num_patterns, 2), dtype=torch.int64, device=device),
            anti_stab_indices=torch.zeros((0, num_patterns, 1), dtype=torch.int64, device=device),
            anti_stab_counts=torch.zeros((0, num_patterns), dtype=torch.int64, device=device),
        )

    qubit_pairs = torch.stack(all_pairs, dim=0)
    max_anti = max(len(al) for anti_lists in all_anti_lists for al in anti_lists)
    max_anti = max(max_anti, 1)

    anti_stab_indices = torch.full((num_w4, num_patterns, max_anti), -1, dtype=torch.int64)
    anti_stab_counts = torch.zeros((num_w4, num_patterns), dtype=torch.int64)

    for i, anti_lists in enumerate(all_anti_lists):
        for p, indices in enumerate(anti_lists):
            anti_stab_counts[i, p] = len(indices)
            for j, idx in enumerate(indices):
                anti_stab_indices[i, p, j] = idx

    return Weight2TimelikeCache(
        num_w4=num_w4,
        num_patterns=num_patterns,
        qubit_pairs=qubit_pairs.to(device),
        anti_stab_indices=anti_stab_indices.to(device),
        anti_stab_counts=anti_stab_counts.to(device),
    )


def _simplify_time_w2_step(
    error_diff: torch.Tensor,
    conj_syndrome: torch.Tensor,
    conj_parity_f: torch.Tensor,
    w2_cache: Weight2TimelikeCache,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """One pass of weight-2 timelike reductions over all weight-4 stabilizers.

    Vectorized: evaluates all (stab, pattern) proposals in one batched operation,
    then applies the best accepted proposal per batch element greedily.

    For Z-errors: error_diff=z, conj_syndrome=s1s2x, conj_parity_f=parity_X.
    For X-errors: error_diff=x, conj_syndrome=s1s2z, conj_parity_f=parity_Z.

    Args:
        error_diff: (B, D2, 2) float error diffs
        conj_syndrome: (B, num_conj_stabs, 2) float conjugate syndrome diffs
        conj_parity_f: (num_conj_stabs, D2) float conjugate parity matrix
        w2_cache: precomputed Weight2TimelikeCache
    """
    if w2_cache.num_w4 == 0:
        return error_diff, conj_syndrome, 0

    B, D2, _ = error_diff.shape
    P = w2_cache.num_w4 * w2_cache.num_patterns

    syn_contrib = torch.einsum("bst,sd->bdt", conj_syndrome, conj_parity_f)
    density = error_diff + syn_contrib
    old_total = density.sum(dim=(1, 2))
    old_r1 = density[:, :, 1].sum(dim=1)

    qpairs = w2_cache.qubit_pairs.reshape(P, 2)
    q1_all, q2_all = qpairs[:, 0], qpairs[:, 1]

    err_q1 = error_diff[:, q1_all, :]
    err_q2 = error_diff[:, q2_all, :]
    err_delta_total = (1.0 - 2.0 * err_q1).sum(dim=2) + (1.0 - 2.0 * err_q2).sum(dim=2)
    err_delta_r1 = (1.0 - 2.0 * err_q1[:, :, 1]) + (1.0 - 2.0 * err_q2[:, :, 1])

    anti_idx = w2_cache.anti_stab_indices.reshape(P, -1)
    anti_cnt = w2_cache.anti_stab_counts.reshape(P)
    max_anti = anti_idx.shape[1]

    parity_row_sums = conj_parity_f.sum(dim=1)

    anti_valid = torch.arange(max_anti,
                              device=error_diff.device).unsqueeze(0) < anti_cnt.unsqueeze(1)
    anti_idx_safe = anti_idx.clamp(min=0)

    syn_r0_all = conj_syndrome[:, :, 0]
    syn_gathered = syn_r0_all[:, anti_idx_safe.reshape(-1)].reshape(B, P, max_anti)
    prs_gathered = parity_row_sums[anti_idx_safe.reshape(-1)].reshape(P, max_anti)

    flip_factors = (1.0 - 2.0 * syn_gathered) * prs_gathered.unsqueeze(0)
    flip_factors = flip_factors * anti_valid.unsqueeze(0).float()
    syn_delta_total = flip_factors.sum(dim=2)

    delta_total = err_delta_total + syn_delta_total
    new_total = old_total.unsqueeze(1) + delta_total
    new_r1 = old_r1.unsqueeze(1) + err_delta_r1

    accept = (new_total < old_total.unsqueeze(1)
             ) | ((new_total == old_total.unsqueeze(1)) & (new_r1 > old_r1.unsqueeze(1)))

    benefit = old_total.unsqueeze(1) - new_total
    benefit = torch.where(accept, benefit, torch.tensor(-float("inf"), device=error_diff.device))
    best_p = benefit.argmax(dim=1)
    any_accepted = benefit.max(dim=1).values > -float("inf")

    total_accepted = int(any_accepted.sum().item())
    if total_accepted == 0:
        return error_diff, conj_syndrome, 0

    mask = any_accepted
    best_q1 = q1_all[best_p[mask]]
    best_q2 = q2_all[best_p[mask]]

    error_diff = error_diff.clone()
    batch_idx = torch.where(mask)[0]
    error_diff[batch_idx, best_q1, :] = 1.0 - error_diff[batch_idx, best_q1, :]
    error_diff[batch_idx, best_q2, :] = 1.0 - error_diff[batch_idx, best_q2, :]

    conj_syndrome = conj_syndrome.clone()
    best_anti_idx = anti_idx[best_p[mask], :]
    best_anti_cnt = anti_cnt[best_p[mask]]
    best_anti_valid = torch.arange(max_anti, device=error_diff.device
                                  ).unsqueeze(0) < best_anti_cnt.unsqueeze(1)
    best_anti_safe = best_anti_idx.clamp(min=0)

    for ai in range(max_anti):
        ai_valid = best_anti_valid[:, ai]
        if not ai_valid.any():
            break
        s_indices = best_anti_safe[ai_valid, ai]
        b_indices = batch_idx[ai_valid]
        conj_syndrome[b_indices, s_indices, 0] = 1.0 - conj_syndrome[b_indices, s_indices, 0]

    return error_diff, conj_syndrome, total_accepted


def _simplify_time_w2_step_nobreak(
    err: torch.Tensor,
    syn: torch.Tensor,
    conj_pf: torch.Tensor,
    parity_row_sums: torch.Tensor,
    q1_all: torch.Tensor,
    q2_all: torch.Tensor,
    anti_idx_safe: torch.Tensor,
    anti_valid_f: torch.Tensor,
    prs_gathered: torch.Tensor,
    num_conj_stabs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torch.compile-friendly weight-2 step. No .item(), no data-dependent branching."""
    B = err.shape[0]
    S = syn.shape[1]

    syn_contrib = torch.einsum("bst,sd->bdt", syn, conj_pf)
    density = err + syn_contrib
    old_total = density.sum(dim=(1, 2))
    old_r1 = density[:, :, 1].sum(dim=1)

    err_q1 = err[:, q1_all, :]
    err_q2 = err[:, q2_all, :]
    err_delta_total = (1.0 - 2.0 * err_q1).sum(dim=2) + (1.0 - 2.0 * err_q2).sum(dim=2)
    err_delta_r1 = (1.0 - 2.0 * err_q1[:, :, 1]) + (1.0 - 2.0 * err_q2[:, :, 1])

    syn_r0 = syn[:, :, 0]
    syn_gathered = syn_r0[:, anti_idx_safe.reshape(-1)].reshape(B, -1, anti_idx_safe.shape[1])
    flip_factors = (1.0 -
                    2.0 * syn_gathered) * prs_gathered.unsqueeze(0) * anti_valid_f.unsqueeze(0)
    syn_delta_total = flip_factors.sum(dim=2)

    delta_total = err_delta_total + syn_delta_total
    new_total = old_total.unsqueeze(1) + delta_total
    new_r1 = old_r1.unsqueeze(1) + err_delta_r1

    accept = (new_total < old_total.unsqueeze(1)
             ) | ((new_total == old_total.unsqueeze(1)) & (new_r1 > old_r1.unsqueeze(1)))

    benefit = (old_total.unsqueeze(1) - new_total) * accept.float() + (-1e9) * (~accept).float()
    best_p = benefit.argmax(dim=1)
    any_accepted = benefit.max(dim=1).values > -1e8

    best_q1 = q1_all[best_p]
    best_q2 = q2_all[best_p]

    b_idx = torch.arange(B, device=err.device)
    new_err = err.clone()
    new_err[b_idx, best_q1, :] = 1.0 - new_err[b_idx, best_q1, :]
    new_err[b_idx, best_q2, :] = 1.0 - new_err[b_idx, best_q2, :]

    best_anti = anti_idx_safe[best_p, :]
    best_anti_v = anti_valid_f[best_p, :]

    flip_counts = torch.zeros(B, S, device=err.device, dtype=err.dtype)
    flip_counts.scatter_add_(1, best_anti, best_anti_v)
    should_flip_syn = ((flip_counts % 2) > 0.5)

    new_syn = syn.clone()
    syn_r0_val = new_syn[:, :, 0]
    new_syn_r0 = torch.where(should_flip_syn, 1.0 - syn_r0_val, syn_r0_val)
    new_syn = torch.cat([new_syn_r0.unsqueeze(2), new_syn[:, :, 1:]], dim=2)

    mask = any_accepted.unsqueeze(1).unsqueeze(2)
    err_out = torch.where(mask, new_err, err)
    syn_out = torch.where(mask, new_syn, syn)

    return err_out, syn_out


def _require_scatter_reduce() -> None:
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "Timelike HE requires torch.Tensor.scatter_reduce_ (PyTorch >= 1.12 / 2.x)."
        )


def _resolve_overlaps(accept_raw: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
    """
    Dense, compile-friendly overlap resolution (OPT-5).

    For each accepted qubit, check that it is the smallest-index accepted qubit
    in every stabilizer it belongs to. This ensures at most one qubit per
    stabilizer is actually flipped, matching the sparse scatter_reduce_ semantics.

    Args:
        accept_raw: (B, D2) bool
        parity:     (S, D2) float
    Returns:
        accept: (B, D2) bool
    """
    D2 = accept_raw.shape[1]
    parity_bool = parity.bool()

    in_support_and_accepted = accept_raw.unsqueeze(1) & parity_bool.unsqueeze(0)  # (B, S, D2)

    qubit_indices = torch.arange(D2, device=accept_raw.device)
    sentinel = D2
    masked = torch.where(in_support_and_accepted, qubit_indices, sentinel)  # (B, S, D2)
    min_per_stab = masked.min(dim=2).values  # (B, S)

    is_min = (min_per_stab.unsqueeze(1) == qubit_indices.unsqueeze(0).unsqueeze(2))  # (B, D2, S)

    qubit_in_stab = parity_bool.T  # (D2, S)
    relevant_check = is_min | ~qubit_in_stab.unsqueeze(0)  # (B, D2, S)

    all_ok = relevant_check.all(dim=2)  # (B, D2)
    return accept_raw & all_ok


def _timelike_pair_step_torch(
    diffs_bt: torch.Tensor,  # (B2, 2, D2)
    meas_bt: torch.Tensor,  # (B2, 2, num_stabs)
    parity_stab_to_qubit: torch.Tensor,  # (num_stabs, D2)
    *,
    use_tie_breaker: bool = True,
    trainX_contrib_bt: Optional[torch.Tensor] = None,  # (B2, 2, D2) precomputed
    cache: Optional[TimelikeHECache] = None,
    parity_f: Optional[torch.Tensor] = None,
    pf_col_sum: Optional[torch.Tensor] = None,
    overlap_chunk_size: int = 2048,
    use_dense_overlap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single timelike weight-1 step for one adjacent time pair.
    Single timelike weight-1 pair step including overlap handling.

    trainX_contrib_bt: precomputed parity @ trainX in (B2, 2, D2) space.
    parity_f / pf_col_sum: precomputed float parity and column sums to
    avoid redundant casts and allow algebraic einsum elimination.
    use_dense_overlap: if True, use dense _resolve_overlaps (compile-friendly, OPT-5).
    """
    if diffs_bt.dtype != torch.uint8:
        diffs_bt = _as_uint8_binary(diffs_bt)
    if meas_bt.dtype != torch.uint8:
        meas_bt = _as_uint8_binary(meas_bt)
    if parity_stab_to_qubit.dtype != torch.uint8:
        parity_stab_to_qubit = _as_uint8_binary(parity_stab_to_qubit)
    parity = parity_stab_to_qubit

    B2, _, D2 = diffs_bt.shape
    num_stabs = int(meas_bt.shape[2])

    if cache is None:
        cache = build_timelike_he_cache(parity)

    if parity_f is None:
        parity_f = parity.to(torch.float32)
    if pf_col_sum is None:
        pf_col_sum = parity_f.sum(dim=0)

    meas_contrib = torch.einsum("bts,sd->btd", meas_bt.to(torch.float32), parity_f).to(torch.int32)
    if trainX_contrib_bt is not None:
        trainX_contrib = trainX_contrib_bt.to(torch.int32)
    else:
        trainX_contrib = torch.zeros_like(meas_contrib)

    old_density_per_round = diffs_bt.to(torch.int32) + meas_contrib + trainX_contrib
    old_density = old_density_per_round.sum(dim=1)

    new_diffs_bt = (1 - diffs_bt).to(torch.uint8)
    new_meas_contrib = meas_contrib.clone()
    new_meas_contrib[:, 0, :] = pf_col_sum.to(torch.int32) - meas_contrib[:, 0, :]

    new_density_per_round = new_diffs_bt.to(torch.int32) + new_meas_contrib + trainX_contrib
    new_density = new_density_per_round.sum(dim=1)

    accept_raw = new_density < old_density
    if use_tie_breaker:
        density_equal = new_density == old_density
        old_max = torch.maximum(old_density_per_round[:, 0, :], old_density_per_round[:, 1, :])
        new_max = torch.maximum(new_density_per_round[:, 0, :], new_density_per_round[:, 1, :])
        accept_raw = accept_raw | (density_equal & (new_max > old_max))

    if use_dense_overlap:
        accept_final = _resolve_overlaps(accept_raw, parity_f)
    else:
        _require_scatter_reduce()
        edge_stab = cache.edge_stab.to(diffs_bt.device)
        edge_qubit = cache.edge_qubit.to(diffs_bt.device)
        accept_final = torch.zeros_like(accept_raw)
        sentinel = int(D2)
        edge_stab_2d = edge_stab.view(1, -1)
        edge_qubit_2d = edge_qubit.view(1, -1)
        edge_qubit_i16 = edge_qubit.to(torch.int16)

        for start in range(0, B2, int(overlap_chunk_size)):
            end = min(B2, start + int(overlap_chunk_size))
            a = accept_raw[start:end]
            Bc = int(a.shape[0])

            a_edge = a.index_select(1, edge_qubit)
            cand = torch.where(
                a_edge,
                edge_qubit_i16.view(1, -1).expand(Bc, -1),
                torch.full((Bc, edge_qubit.numel()), sentinel, device=a.device, dtype=torch.int16),
            )

            min_q = torch.full((Bc, num_stabs), sentinel, device=a.device, dtype=torch.int16)
            min_q.scatter_reduce_(
                dim=1,
                index=edge_stab_2d.expand(Bc, -1),
                src=cand,
                reduce="amin",
                include_self=True,
            )

            min_edge = min_q.index_select(1, edge_stab)
            ok_edge = (min_edge == edge_qubit_i16.view(1, -1).expand(Bc, -1))

            all_ok = torch.ones((Bc, D2), device=a.device, dtype=torch.uint8)
            all_ok.scatter_reduce_(
                dim=1,
                index=edge_qubit_2d.expand(Bc, -1),
                src=ok_edge.to(torch.uint8),
                reduce="amin",
                include_self=True,
            )

            accept_final[start:end] = a & all_ok.bool()

    diffs_out = diffs_bt ^ accept_final.to(diffs_bt.dtype).unsqueeze(1)
    flip_counts = (accept_final.to(torch.float32) @ parity_f.t()).to(torch.int32)
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
    trainX_contrib: Optional[torch.Tensor] = None,  # (B, T, D2) precomputed
    cache: Optional[TimelikeHECache] = None,
    parity_f: Optional[torch.Tensor] = None,
    pf_col_sum: Optional[torch.Tensor] = None,
    use_dense_overlap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if diffs.dtype != torch.uint8:
        diffs = _as_uint8_binary(diffs)
    if meas.dtype != torch.uint8:
        meas = _as_uint8_binary(meas)

    B, T, D2 = diffs.shape
    num_stabs = int(meas.shape[2])

    start_even = 2 if exclude_round0 else 0

    def process_pass(start_idx: int, d: torch.Tensor, m: torch.Tensor,
                     tXc: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """In-place slice update — caller must clone d/m beforehand."""
        num_pairs = (T - start_idx) // 2
        if num_pairs <= 0:
            return d, m
        slice_len = 2 * num_pairs
        end_idx = start_idx + slice_len

        d_slice = d[:, start_idx:end_idx, :]
        m_slice = m[:, start_idx:end_idx, :]
        d_flat = d_slice.reshape(B * num_pairs, 2, D2)
        m_flat = m_slice.reshape(B * num_pairs, 2, num_stabs)

        tXc_flat = None
        if tXc is not None:
            tXc_flat = tXc[:, start_idx:end_idx, :].reshape(B * num_pairs, 2, D2)

        d_new, m_new = _timelike_pair_step_torch(
            d_flat,
            m_flat,
            parity_stab_to_qubit,
            use_tie_breaker=use_tie_breaker,
            trainX_contrib_bt=tXc_flat,
            cache=cache,
            parity_f=parity_f,
            pf_col_sum=pf_col_sum,
            use_dense_overlap=use_dense_overlap,
        )

        d[:, start_idx:end_idx, :] = d_new.reshape(B, slice_len, D2)
        m[:, start_idx:end_idx, :] = m_new.reshape(B, slice_len, num_stabs)
        return d, m

    # Clone once before even pass to avoid mutating caller's tensors
    diffs = diffs.clone()
    meas = meas.clone()
    diffs, meas = process_pass(start_even, diffs, meas, trainX_contrib)
    # Odd pass operates on even pass output — no additional clone needed
    diffs, meas = process_pass(1, diffs, meas, trainX_contrib)
    return diffs, meas


def _apply_timelike_compiled(
    z: torch.Tensor,
    x: torch.Tensor,
    sx: torch.Tensor,
    sz: torch.Tensor,
    parity_Z_f: torch.Tensor,
    parity_X_f: torch.Tensor,
    pf_col_sum_Z: torch.Tensor,
    pf_col_sum_X: torch.Tensor,
    trainX_z_contrib: Optional[torch.Tensor],
    trainX_x_contrib: Optional[torch.Tensor],
    *,
    max_passes: int,
    basis: str,
    compile_chunk_size: int = 2,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Compiled timelike w1 with hybrid early-exit (OPT-4 + NEW-3).

    Runs *compile_chunk_size* passes per compiled call, then checks convergence.
    Repeats until converged or max_passes reached.
    compute_dtype (NEW-2): controls float precision for density computations.
    """
    B, T, D2 = z.shape
    max_t = T - 1
    if max_t <= 0:
        return z, x, sx, sz, 0

    basis_up = basis.upper()
    even_start_x = 2 if basis_up == "X" else 0
    even_start_z = 2 if basis_up == "Z" else 0
    min_t_x = 1 if basis_up == "X" else 0
    min_t_z = 1 if basis_up == "Z" else 0

    dt = compute_dtype
    pfT_Z = parity_Z_f.to(dt).T.contiguous()
    pfT_X = parity_X_f.to(dt).T.contiguous()
    pf_Z_dt = parity_Z_f.to(dt)
    pf_X_dt = parity_X_f.to(dt)
    pf_col_Z_dt = pf_col_sum_Z.to(dt)
    pf_col_X_dt = pf_col_sum_X.to(dt)

    x_work = x.to(dt).transpose(1, 2).contiguous()
    z_work = z.to(dt).transpose(1, 2).contiguous()
    num_stabs_Z = parity_Z_f.shape[0]
    num_stabs_X = parity_X_f.shape[0]
    sz_work = sz.to(dt).transpose(1, 2).contiguous()
    sx_work = sx.to(dt).transpose(1, 2).contiguous()

    dev = z.device
    if trainX_z_contrib is not None:
        tX_z_work = trainX_z_contrib.to(dt).transpose(1, 2).contiguous()
    else:
        tX_z_work = torch.zeros(B, D2, T, device=dev, dtype=dt)
    if trainX_x_contrib is not None:
        tX_x_work = trainX_x_contrib.to(dt).transpose(1, 2).contiguous()
    else:
        tX_x_work = torch.zeros(B, D2, T, device=dev, dtype=dt)

    compiled_fn = _get_compiled_timelike_loop(
        max_t, min_t_x, min_t_z, max_passes, even_start_x, even_start_z
    )

    x_work, z_work, sz_work, sx_work = compiled_fn(
        x_work,
        z_work,
        sz_work,
        sx_work,
        pf_Z_dt,
        pfT_Z,
        pf_col_Z_dt,
        pf_X_dt,
        pfT_X,
        pf_col_X_dt,
        tX_z_work,
        tX_x_work,
    )

    z_out = z_work.transpose(1, 2).round().to(torch.uint8)
    x_out = x_work.transpose(1, 2).round().to(torch.uint8)
    sz_out = sz_work.transpose(1, 2).round().to(torch.uint8)
    sx_out = sx_work.transpose(1, 2).round().to(torch.uint8)

    return z_out, x_out, sx_out, sz_out, max_passes


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
    use_dense_overlap: bool = False,
    use_compile: bool = False,
    compile_chunk_size: int = 2,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = _as_uint8_binary(z_error_diffs)
    x = _as_uint8_binary(x_error_diffs)
    sx = _as_uint8_binary(s1s2_x)
    sz = _as_uint8_binary(s1s2_z)

    basis_up = basis.upper()
    exclude_round0_x = (basis_up == "X")
    exclude_round0_z = (basis_up == "Z")

    parity_Z = _as_uint8_binary(parity_matrix_Z).to(z.device)
    parity_X = _as_uint8_binary(parity_matrix_X).to(z.device)

    parity_Z_f = parity_Z.to(torch.float32)
    parity_X_f = parity_X.to(torch.float32)
    pf_col_sum_Z = parity_Z_f.sum(dim=0)
    pf_col_sum_X = parity_X_f.sum(dim=0)

    trainX_z_contrib: Optional[torch.Tensor] = None
    trainX_x_contrib: Optional[torch.Tensor] = None
    if trainX_z is not None:
        tX_z = _as_uint8_binary(trainX_z)
        trainX_z_contrib = torch.einsum("bts,sd->btd", tX_z.to(torch.float32), parity_Z_f)
    if trainX_x is not None:
        tX_x = _as_uint8_binary(trainX_x)
        trainX_x_contrib = torch.einsum("bts,sd->btd", tX_x.to(torch.float32), parity_X_f)

    if use_compile:
        z, x, sx, sz, iters = _apply_timelike_compiled(
            z,
            x,
            sx,
            sz,
            parity_Z_f,
            parity_X_f,
            pf_col_sum_Z,
            pf_col_sum_X,
            trainX_z_contrib,
            trainX_x_contrib,
            max_passes=max_passes,
            basis=basis_up,
            compile_chunk_size=compile_chunk_size,
            compute_dtype=compute_dtype,
        )
        return z, x, sx, sz, torch.tensor(iters, dtype=torch.int32, device=z.device)

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

        x, sz = _timelike_pass_brickwork_torch(
            x,
            sz,
            parity_Z,
            exclude_round0=exclude_round0_x,
            use_tie_breaker=use_tie_breaker,
            trainX_contrib=trainX_z_contrib,
            cache=cache_Z,
            parity_f=parity_Z_f,
            pf_col_sum=pf_col_sum_Z,
            use_dense_overlap=use_dense_overlap,
        )
        z, sx = _timelike_pass_brickwork_torch(
            z,
            sx,
            parity_X,
            exclude_round0=exclude_round0_z,
            use_tie_breaker=use_tie_breaker,
            trainX_contrib=trainX_x_contrib,
            cache=cache_X,
            parity_f=parity_X_f,
            pf_col_sum=pf_col_sum_X,
            use_dense_overlap=use_dense_overlap,
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


def _apply_weight2_pass(
    z_diffs: torch.Tensor,
    x_diffs: torch.Tensor,
    sx: torch.Tensor,
    sz: torch.Tensor,
    parity_X_f: torch.Tensor,
    parity_Z_f: torch.Tensor,
    cache_X_w2: Weight2TimelikeCache,
    cache_Z_w2: Weight2TimelikeCache,
    *,
    basis: str,
    max_passes_w2: int = 4,
    use_compile: bool = False,
    compile_chunk_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run weight-2 timelike HE passes on (B, T, D2)-layout diffs.

    When use_compile=True, delegates to the CUDA-graph compiled path
    via _apply_weight2_compiled.  Otherwise uses the eager path with
    early-exit based on accepted-count.
    """
    B, T, D2 = z_diffs.shape
    max_t = T - 1
    if max_t <= 0 or (cache_X_w2.num_w4 == 0 and cache_Z_w2.num_w4 == 0):
        return z_diffs, x_diffs, sx, sz

    if use_compile:
        return _apply_weight2_compiled(
            z_diffs,
            x_diffs,
            sx,
            sz,
            parity_X_f,
            parity_Z_f,
            cache_X_w2,
            cache_Z_w2,
            basis=basis,
            max_passes_w2=max_passes_w2,
            compile_chunk_size=compile_chunk_size,
        )

    basis_up = basis.upper()
    min_t_x = 1 if basis_up == "X" else 0
    min_t_z = 1 if basis_up == "Z" else 0

    x_work = x_diffs.float().transpose(1, 2).contiguous()
    z_work = z_diffs.float().transpose(1, 2).contiguous()
    sz_work = sz.float().transpose(1, 2).contiguous()
    sx_work = sx.float().transpose(1, 2).contiguous()

    for _ in range(max_passes_w2):
        total_w2 = 0
        for t in range(max_t):
            if t >= min_t_x and cache_X_w2.num_w4 > 0:
                x_err = x_work[:, :, t:t + 2]
                sz_syn = sz_work[:, :, t:t + 2]
                x_err, sz_syn, n = _simplify_time_w2_step(x_err, sz_syn, parity_Z_f, cache_X_w2)
                total_w2 += n
                x_work[:, :, t:t + 2] = x_err
                sz_work[:, :, t:t + 2] = sz_syn

            if t >= min_t_z and cache_Z_w2.num_w4 > 0:
                z_err = z_work[:, :, t:t + 2]
                sx_syn = sx_work[:, :, t:t + 2]
                z_err, sx_syn, n = _simplify_time_w2_step(z_err, sx_syn, parity_X_f, cache_Z_w2)
                total_w2 += n
                z_work[:, :, t:t + 2] = z_err
                sx_work[:, :, t:t + 2] = sx_syn

        if total_w2 == 0:
            break

    return (
        z_work.transpose(1, 2).to(torch.uint8), x_work.transpose(1, 2).to(torch.uint8),
        sx_work.transpose(1, 2).to(torch.uint8), sz_work.transpose(1, 2).to(torch.uint8)
    )


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
    use_dense_overlap: bool = False,
    use_compile: bool = False,
    compile_chunk_size: int = 2,
    compute_dtype: Optional[torch.dtype] = None,
    use_weight2: bool = False,
    max_passes_w2: int = 4,
    cache_Z_w2: Optional[Weight2TimelikeCache] = None,
    cache_X_w2: Optional[Weight2TimelikeCache] = None,
    use_coset_search: bool = False,
    coset_max_generators: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch HE: spacelike + timelike weight-1 (+ optional weight-2).

    Flow (all in diffs space after initial conversion):
      1) cumulative -> diffs once
      2) repeat num_he_cycles:
          a) spacelike HE on diffs (+ optional coset min-weight search)
          b) timelike weight-1 HE on diffs until convergence
          c) if use_weight2: timelike weight-2 HE, then re-converge weight-1
      3) final spacelike cleanup on diffs

    Args:
        use_weight2: If True, run weight-2 timelike after weight-1 converges (OPT-7).
        max_passes_w2: Max weight-2 passes per cycle.
        cache_Z_w2 / cache_X_w2: Prebuilt Weight2TimelikeCache for Z/X errors.
            Built automatically if use_weight2=True and not provided.
        use_coset_search: If True, after greedy spacelike canonicalization,
            enumerate all coset representatives and pick the minimum-weight one (P12 / NEW-4).
        coset_max_generators: Skip coset search if S exceeds this (exponential guard).
    """
    z_diffs, x_diffs = _cumulative_to_diffs_torch(z_errors, x_errors)
    sx = _as_uint8_binary(s1s2_x)
    sz = _as_uint8_binary(s1s2_z)

    parity_Z = _as_uint8_binary(parity_matrix_Z).to(z_diffs.device)
    parity_X = _as_uint8_binary(parity_matrix_X).to(z_diffs.device)

    dt = compute_dtype if compute_dtype is not None else torch.float32

    if cache_Z_spacelike is None:
        cache_Z_spacelike = build_spacelike_he_cache(
            parity_Z, distance=distance, basis="Z", device=z_diffs.device
        )
    if cache_X_spacelike is None:
        cache_X_spacelike = build_spacelike_he_cache(
            parity_X, distance=distance, basis="X", device=z_diffs.device
        )

    parity_Z_f = parity_Z.to(torch.float32)
    parity_X_f = parity_X.to(torch.float32)

    if use_weight2:
        if cache_X_w2 is None:
            cache_X_w2 = build_weight2_timelike_cache(
                parity_Z, parity_Z, distance, "X", z_diffs.device
            )
        if cache_Z_w2 is None:
            cache_Z_w2 = build_weight2_timelike_cache(
                parity_X, parity_X, distance, "Z", z_diffs.device
            )

    spacelike_kwargs = dict(
        use_compile=use_compile,
        compute_dtype=dt,
        use_coset_search=use_coset_search,
        coset_max_generators=coset_max_generators,
    )

    for _ in range(int(num_he_cycles)):
        z_diffs, x_diffs = apply_homological_equivalence_torch_vmap(
            z_diffs,
            x_diffs,
            parity_Z,
            parity_X,
            distance=distance,
            cache_Z=cache_Z_spacelike,
            cache_X=cache_X_spacelike,
            **spacelike_kwargs,
        )

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
            use_dense_overlap=use_dense_overlap or use_compile,
            use_compile=use_compile,
            compile_chunk_size=compile_chunk_size,
            compute_dtype=dt,
        )

        if use_weight2 and cache_X_w2 is not None and cache_Z_w2 is not None:
            z_diffs, x_diffs, sx, sz = _apply_weight2_pass(
                z_diffs,
                x_diffs,
                sx,
                sz,
                parity_X_f,
                parity_Z_f,
                cache_X_w2,
                cache_Z_w2,
                basis=basis,
                max_passes_w2=max_passes_w2,
                use_compile=use_compile,
                compile_chunk_size=compile_chunk_size,
            )
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
                use_dense_overlap=use_dense_overlap or use_compile,
                use_compile=use_compile,
                compile_chunk_size=compile_chunk_size,
                compute_dtype=dt,
            )

    z_diffs, x_diffs = apply_homological_equivalence_torch_vmap(
        z_diffs,
        x_diffs,
        parity_Z,
        parity_X,
        distance=distance,
        cache_Z=cache_Z_spacelike,
        cache_X=cache_X_spacelike,
        **spacelike_kwargs,
    )

    return z_diffs, x_diffs, sx, sz


# ---------------------------------------------------------------------------
# torch.compile caches and warmup (OPT-6)
# ---------------------------------------------------------------------------

_compiled_timelike_cache: dict = {}
_compiled_weight2_cache: dict = {}


def _get_compiled_timelike_loop(
    max_t: int,
    min_t_x: int,
    min_t_z: int,
    num_passes: int,
    even_start_x: int = 0,
    even_start_z: int = 0,
):
    """
    Return a torch.compiled function that runs num_passes full sweeps
    over all round-pairs using brickwork (even+odd) pattern.
    Cached by (max_t, min_t_x, min_t_z, num_passes, even_start_x, even_start_z).
    """
    key = (max_t, min_t_x, min_t_z, num_passes, even_start_x, even_start_z)
    if key in _compiled_timelike_cache:
        return _compiled_timelike_cache[key]

    def _timelike_loop(
        x_work,
        z_work,
        sz_work,
        sx_work,
        pf_Z,
        pfT_Z,
        pf_col_sum_Z,
        pf_X,
        pfT_X,
        pf_col_sum_X,
        tX_z_work,
        tX_x_work,
    ):
        for _pass in range(num_passes):
            for t in range(even_start_x, max_t, 2):
                x_err = x_work[:, :, t:t + 2]
                sz_syn = sz_work[:, :, t:t + 2]
                x_err, sz_syn = _simplify_time_w1_step_nobreak(
                    x_err, sz_syn, pf_Z, pfT_Z, pf_col_sum_Z, tX_z_work[:, :, t],
                    tX_z_work[:, :, t + 1], _resolve_overlaps
                )
                # Out-of-place cat avoids clone+slice aliasing inside torch.compile
                x_work = torch.cat([x_work[:, :, :t], x_err, x_work[:, :, t + 2:]], dim=2)
                sz_work = torch.cat([sz_work[:, :, :t], sz_syn, sz_work[:, :, t + 2:]], dim=2)

            for t in range(1, max_t, 2):
                if t >= min_t_x:
                    x_err = x_work[:, :, t:t + 2]
                    sz_syn = sz_work[:, :, t:t + 2]
                    x_err, sz_syn = _simplify_time_w1_step_nobreak(
                        x_err, sz_syn, pf_Z, pfT_Z, pf_col_sum_Z, tX_z_work[:, :, t],
                        tX_z_work[:, :, t + 1], _resolve_overlaps
                    )
                    x_work = torch.cat([x_work[:, :, :t], x_err, x_work[:, :, t + 2:]], dim=2)
                    sz_work = torch.cat([sz_work[:, :, :t], sz_syn, sz_work[:, :, t + 2:]], dim=2)

            for t in range(even_start_z, max_t, 2):
                z_err = z_work[:, :, t:t + 2]
                sx_syn = sx_work[:, :, t:t + 2]
                z_err, sx_syn = _simplify_time_w1_step_nobreak(
                    z_err, sx_syn, pf_X, pfT_X, pf_col_sum_X, tX_x_work[:, :, t],
                    tX_x_work[:, :, t + 1], _resolve_overlaps
                )
                z_work = torch.cat([z_work[:, :, :t], z_err, z_work[:, :, t + 2:]], dim=2)
                sx_work = torch.cat([sx_work[:, :, :t], sx_syn, sx_work[:, :, t + 2:]], dim=2)

            for t in range(1, max_t, 2):
                if t >= min_t_z:
                    z_err = z_work[:, :, t:t + 2]
                    sx_syn = sx_work[:, :, t:t + 2]
                    z_err, sx_syn = _simplify_time_w1_step_nobreak(
                        z_err, sx_syn, pf_X, pfT_X, pf_col_sum_X, tX_x_work[:, :, t],
                        tX_x_work[:, :, t + 1], _resolve_overlaps
                    )
                    z_work = torch.cat([z_work[:, :, :t], z_err, z_work[:, :, t + 2:]], dim=2)
                    sx_work = torch.cat([sx_work[:, :, :t], sx_syn, sx_work[:, :, t + 2:]], dim=2)

        return x_work, z_work, sz_work, sx_work

    compiled = torch.compile(_timelike_loop, mode="reduce-overhead", fullgraph=True)
    _compiled_timelike_cache[key] = compiled
    return compiled


def _get_compiled_weight2_loop(
    max_t: int,
    min_t_x: int,
    min_t_z: int,
    num_passes: int,
    has_x_w4: bool,
    has_z_w4: bool,
):
    """Return a torch.compiled weight-2 loop for CUDA graph replay.

    The loop applies ``_simplify_time_w2_step_nobreak`` over all round-pairs
    for ``num_passes`` full sweeps.  Cached by shape parameters.
    """
    key = ("w2", max_t, min_t_x, min_t_z, num_passes, has_x_w4, has_z_w4)
    if key in _compiled_weight2_cache:
        return _compiled_weight2_cache[key]

    def _w2_loop(
        x_work,
        z_work,
        sz_work,
        sx_work,
        conj_pf_Z,
        prs_Z,
        q1_Z,
        q2_Z,
        anti_idx_Z,
        anti_valid_Z,
        prs_g_Z,
        ncs_Z,
        conj_pf_X,
        prs_X,
        q1_X,
        q2_X,
        anti_idx_X,
        anti_valid_X,
        prs_g_X,
        ncs_X,
    ):
        for _p in range(num_passes):
            for t in range(max_t):
                if t >= min_t_x and has_x_w4:
                    x_err = x_work[:, :, t:t + 2]
                    sz_syn = sz_work[:, :, t:t + 2]
                    x_err, sz_syn = _simplify_time_w2_step_nobreak(
                        x_err, sz_syn, conj_pf_Z, prs_Z, q1_Z, q2_Z, anti_idx_Z, anti_valid_Z,
                        prs_g_Z, ncs_Z
                    )
                    x_work = torch.cat([x_work[:, :, :t], x_err, x_work[:, :, t + 2:]], dim=2)
                    sz_work = torch.cat([sz_work[:, :, :t], sz_syn, sz_work[:, :, t + 2:]], dim=2)

                if t >= min_t_z and has_z_w4:
                    z_err = z_work[:, :, t:t + 2]
                    sx_syn = sx_work[:, :, t:t + 2]
                    z_err, sx_syn = _simplify_time_w2_step_nobreak(
                        z_err, sx_syn, conj_pf_X, prs_X, q1_X, q2_X, anti_idx_X, anti_valid_X,
                        prs_g_X, ncs_X
                    )
                    z_work = torch.cat([z_work[:, :, :t], z_err, z_work[:, :, t + 2:]], dim=2)
                    sx_work = torch.cat([sx_work[:, :, :t], sx_syn, sx_work[:, :, t + 2:]], dim=2)

        return x_work, z_work, sz_work, sx_work

    compiled = torch.compile(_w2_loop, mode="max-autotune", fullgraph=True)
    _compiled_weight2_cache[key] = compiled
    return compiled


def _precompute_w2_nobreak_tensors(
    cache: Weight2TimelikeCache,
    conj_pf: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Flatten Weight2TimelikeCache into fixed-shape tensors for _nobreak."""
    P = cache.num_w4 * cache.num_patterns
    qpairs = cache.qubit_pairs.reshape(P, 2)
    q1 = qpairs[:, 0].to(device)
    q2 = qpairs[:, 1].to(device)
    anti_idx = cache.anti_stab_indices.reshape(P, -1)
    anti_cnt = cache.anti_stab_counts.reshape(P)
    max_anti = anti_idx.shape[1]
    parity_row_sums = conj_pf.sum(dim=1).to(dtype)
    anti_valid = (
        torch.arange(max_anti, device=device).unsqueeze(0) < anti_cnt.to(device).unsqueeze(1)
    ).to(dtype)
    anti_idx_safe = anti_idx.clamp(min=0).to(device)
    prs_gathered = parity_row_sums[anti_idx_safe.reshape(-1)].reshape(P, max_anti)
    num_conj_stabs = torch.tensor(conj_pf.shape[0], device=device, dtype=dtype)
    return q1, q2, anti_idx_safe, anti_valid, prs_gathered, parity_row_sums, num_conj_stabs


def _apply_weight2_compiled(
    z_diffs: torch.Tensor,
    x_diffs: torch.Tensor,
    sx: torch.Tensor,
    sz: torch.Tensor,
    parity_X_f: torch.Tensor,
    parity_Z_f: torch.Tensor,
    cache_X_w2: Weight2TimelikeCache,
    cache_Z_w2: Weight2TimelikeCache,
    *,
    basis: str,
    max_passes_w2: int = 4,
    compile_chunk_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled weight-2 pass with hybrid early-exit and CUDA graph replay."""
    B, T, D2 = z_diffs.shape
    max_t = T - 1
    if max_t <= 0:
        return z_diffs, x_diffs, sx, sz

    basis_up = basis.upper()
    min_t_x = 1 if basis_up == "X" else 0
    min_t_z = 1 if basis_up == "Z" else 0
    has_x_w4 = cache_X_w2.num_w4 > 0
    has_z_w4 = cache_Z_w2.num_w4 > 0

    dev = z_diffs.device
    dt = torch.float32

    x_work = x_diffs.to(dt).transpose(1, 2).contiguous()
    z_work = z_diffs.to(dt).transpose(1, 2).contiguous()
    sz_work = sz.to(dt).transpose(1, 2).contiguous()
    sx_work = sx.to(dt).transpose(1, 2).contiguous()

    (q1_Z, q2_Z, ai_Z, av_Z, pg_Z, prs_Z,
     ncs_Z) = _precompute_w2_nobreak_tensors(cache_X_w2, parity_Z_f, dev, dt)
    (q1_X, q2_X, ai_X, av_X, pg_X, prs_X,
     ncs_X) = _precompute_w2_nobreak_tensors(cache_Z_w2, parity_X_f, dev, dt)

    conj_pf_Z = parity_Z_f.to(dt).to(dev)
    conj_pf_X = parity_X_f.to(dt).to(dev)

    compiled_fn = _get_compiled_weight2_loop(
        max_t, min_t_x, min_t_z, max_passes_w2, has_x_w4, has_z_w4
    )

    x_work, z_work, sz_work, sx_work = compiled_fn(
        x_work,
        z_work,
        sz_work,
        sx_work,
        conj_pf_Z,
        prs_Z,
        q1_Z,
        q2_Z,
        ai_Z,
        av_Z,
        pg_Z,
        ncs_Z,
        conj_pf_X,
        prs_X,
        q1_X,
        q2_X,
        ai_X,
        av_X,
        pg_X,
        ncs_X,
    )

    return (
        z_work.transpose(1, 2).round().to(torch.uint8), x_work.transpose(1,
                                                                         2).round().to(torch.uint8),
        sx_work.transpose(1,
                          2).round().to(torch.uint8), sz_work.transpose(1,
                                                                        2).round().to(torch.uint8)
    )


def warmup_he_compile(
    distance: int,
    n_rounds: int,
    basis: str,
    max_passes_w1: int,
    apply_spacelike: bool = True,
    use_weight2: bool = False,
    max_passes_w2: int = 4,
) -> None:
    """Eagerly trigger torch.compile for all HE kernels.

    Call from a background thread while DEM generation runs on the main
    thread. Every ``_get_compiled_*`` helper is cached, so subsequent
    calls (including the real HE path) are instant cache hits.

    This is a no-op on CPU or if ``n_rounds`` is too small.
    """
    R = int(n_rounds)
    max_t = R - 1
    if max_t <= 0:
        return

    min_t_x = 1 if str(basis).upper() == "X" else 0
    min_t_z = 1 if str(basis).upper() == "Z" else 0

    if apply_spacelike:
        for nl in range(max(1, int(distance) - 1), int(distance) + 2):
            _get_compiled_seq_wr(nl)

    even_start_x = 2 if str(basis).upper() == "X" else 0
    even_start_z = 2 if str(basis).upper() == "Z" else 0
    _get_compiled_timelike_loop(
        max_t, min_t_x, min_t_z, int(max_passes_w1), even_start_x, even_start_z
    )

    if use_weight2 and max_passes_w2 > 0:
        _get_compiled_weight2_loop(
            max_t, min_t_x, min_t_z, min(1, int(max_passes_w2)), has_x_w4=True, has_z_w4=True
        )
