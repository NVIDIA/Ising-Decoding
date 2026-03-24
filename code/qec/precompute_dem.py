#!/usr/bin/env python3
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
Torch-only precompute of DEM matrices (H, p, A) for the multi-round detector-frame simulator.

Goal:
  Export the DEM bundle used by `qec/surface_code/memory_circuit_torch.py` without
  relying on presampled simulator state.

Outputs (in --dem_output_dir):
  - surface_d{d}_r{r}_{basis}_frame_predecoder.X.npz  : HX (num_detectors, num_errors) uint8
  - surface_d{d}_r{r}_{basis}_frame_predecoder.Z.npz  : HZ (num_detectors, num_errors) uint8
  - surface_d{d}_r{r}_{basis}_frame_predecoder.p.npz  : p  (num_errors,) float32 (single-p marginal)
  - surface_d{d}_r{r}_{basis}_frame_predecoder.A.npz  : A  (n_rounds*num_meas, 2*num_detectors) uint8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import sys

import numpy as np
import torch

# Ensure `import qec...` works when running as a script.
_CODE_ROOT = Path(__file__).resolve().parents[1]  # .../pre-decoder/code
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

# =============================================================================
# Stim parsing helpers (pure python)
# =============================================================================


def extract_cnot_structure_from_stim_text(circuit_string: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CX layers (before REPEAT) from a Stim circuit string.

    Returns:
        circuit: (num_layers, max_gates, 2) int32 padded with -1.
        cx_times: (num_layers,) int32 time indices (prep is time=0, first CX layer is time=1).
    """
    lines = circuit_string.strip().split("\n")
    cnot_layers: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("REPEAT"):
            break
        if line.startswith("TICK"):
            if current:
                cnot_layers.append(current)
                current = []
            continue
        if line.startswith("CX") or line.startswith("CNOT"):
            parts = [p for p in line.split(" ") if p]
            # Ignore classically-controlled feedforward: "CX rec[-k] q".
            if any(p.startswith("rec[") for p in parts[1:]):
                continue
            qs = list(map(int, parts[1:]))
            for i in range(0, len(qs), 2):
                if i + 1 < len(qs):
                    current.append((qs[i], qs[i + 1]))

    if current:
        cnot_layers.append(current)
    if not cnot_layers:
        raise ValueError("No CX/CNOT layers found before REPEAT")

    max_gates = max(len(layer) for layer in cnot_layers)
    num_layers = len(cnot_layers)
    circuit = np.full((num_layers, max_gates, 2), -1, dtype=np.int32)
    for li, layer in enumerate(cnot_layers):
        for gi, (c, t) in enumerate(layer):
            circuit[li, gi, 0] = int(c)
            circuit[li, gi, 1] = int(t)
    cx_times = np.arange(num_layers, dtype=np.int32) + 1
    return circuit, cx_times


# =============================================================================
# Torch presampling core (frame_predecoder)
# =============================================================================


def _torch_update_pauli_frame_with_layer(
    frame: torch.Tensor,  # (B, nq, 2) uint8
    controls: torch.Tensor,  # (G,) long
    targets: torch.Tensor,  # (G,) long
) -> torch.Tensor:
    # Implements the same Clifford propagation as the legacy implementation:
    #   Z_control ^= Z_target
    #   X_target  ^= X_control
    if controls.numel() == 0:
        return frame
    # Snapshot values before in-place updates.
    z_t = frame.index_select(1, targets)[:, :, 1].clone()
    x_c = frame.index_select(1, controls)[:, :, 0].clone()
    frame[:, controls, 1] ^= z_t
    frame[:, targets, 0] ^= x_c
    return frame


def _torch_inject_errors(errors: torch.Tensor, frame: torch.Tensor, t: int) -> torch.Tensor:
    # errors: (E, nq, 3) int8, where [:,:,0:2] are X/Z bits and [:,:,2] is time index.
    mask = (errors[:, :, 2] == int(t)).to(torch.uint8).unsqueeze(-1)  # (E, nq, 1)
    frame ^= (errors[:, :, :2].to(torch.uint8) * mask)
    return frame


@torch.no_grad()
def presample_frame_single_round_torch(
    *,
    t_total: int,
    nq: int,
    controls_by_layer: np.ndarray,  # (L, G, 2) int32 padded with -1
    cx_times: np.ndarray,  # (L,) int32 in 1..L
    errors: torch.Tensor,  # (E, nq, 3) int8 on device
) -> torch.Tensor:
    """Return per-error end-of-round frames: (E, nq, 2) uint8."""
    dev = errors.device
    frame = torch.zeros((errors.shape[0], int(nq), 2), dtype=torch.uint8, device=dev)

    controls = torch.as_tensor(controls_by_layer[:, :, 0], dtype=torch.long, device=dev)
    targets = torch.as_tensor(controls_by_layer[:, :, 1], dtype=torch.long, device=dev)
    cx_times_t = torch.as_tensor(cx_times, dtype=torch.long, device=dev)

    for tt in range(int(t_total)):
        # Apply ideal CX layer at time tt (if any).
        mask_layer = (cx_times_t == int(tt))
        if bool(mask_layer.any()):
            cs = controls[mask_layer].reshape(-1)
            ts = targets[mask_layer].reshape(-1)
            valid = (cs >= 0) & (ts >= 0)
            cs = cs[valid]
            ts = ts[valid]
            frame = _torch_update_pauli_frame_with_layer(frame, cs, ts)
        # Then inject errors occurring at this time.
        frame = _torch_inject_errors(errors, frame, tt)

    return frame


@torch.no_grad()
def propagate_frame_one_round_torch(
    frame: torch.Tensor,  # (B, nq, 2) uint8
    controls_by_layer: torch.Tensor,  # (L, G) long, -1 padded
    targets_by_layer: torch.Tensor,  # (L, G) long, -1 padded
) -> torch.Tensor:
    out = frame
    for li in range(int(controls_by_layer.shape[0])):
        cs = controls_by_layer[li].reshape(-1)
        ts = targets_by_layer[li].reshape(-1)
        valid = (cs >= 0) & (ts >= 0)
        out = _torch_update_pauli_frame_with_layer(out, cs[valid], ts[valid])
    return out


@torch.no_grad()
def presample_detector_seq_multiround_torch(
    *,
    frame_single_round: torch.Tensor,  # (E, nq, 2) uint8
    controls_by_layer: np.ndarray,  # (L, G, 2) int32
    meas_qubits: np.ndarray,  # (m,) int32
    n_rounds: int,
) -> torch.Tensor:
    """Return detectors_seq: (E, n_rounds, nq, 2) uint8."""
    dev = frame_single_round.device
    E, nq = int(frame_single_round.shape[0]), int(frame_single_round.shape[1])
    R = int(n_rounds)
    if R < 1:
        raise ValueError("n_rounds must be >= 1")

    meas_q = torch.as_tensor(meas_qubits, dtype=torch.long, device=dev).reshape(-1)
    controls = torch.as_tensor(controls_by_layer[:, :, 0], dtype=torch.long, device=dev)
    targets = torch.as_tensor(controls_by_layer[:, :, 1], dtype=torch.long, device=dev)

    outs = torch.zeros((E, R, nq, 2), dtype=torch.uint8, device=dev)
    outs[:, 0] = frame_single_round

    carry = frame_single_round.clone()
    if meas_q.numel() > 0:
        carry[:, meas_q, :] = 0

    for rr in range(1, R):
        out = propagate_frame_one_round_torch(carry, controls, targets)
        outs[:, rr] = out
        carry = out
        if meas_q.numel() > 0:
            carry = carry.clone()
            carry[:, meas_q, :] = 0

    return outs


def _torch_measure(
    frame: torch.Tensor, meas_qubits: np.ndarray, meas_bases: np.ndarray
) -> torch.Tensor:
    # frame: (E, nq, 2) uint8, meas_bases: 0=X, 1=Z, do_measurement reads component (1-basis).
    dev = frame.device
    qs = torch.as_tensor(meas_qubits, dtype=torch.long, device=dev).reshape(-1)
    bases = torch.as_tensor(meas_bases, dtype=torch.long, device=dev).reshape(-1)
    x = frame.index_select(1, qs)[:, :, 0]
    z = frame.index_select(1, qs)[:, :, 1]
    # Z-basis (1) reads X; X-basis (0) reads Z.
    return torch.where(bases[None, :] == 1, x, z).to(torch.uint8)


def _torch_keep_idx(
    measurements: torch.Tensor, frame: torch.Tensor, data_qubits: np.ndarray
) -> torch.Tensor:
    dev = frame.device
    data_q = torch.as_tensor(data_qubits, dtype=torch.long, device=dev).reshape(-1)
    detected = (measurements.sum(dim=-1) > 0)
    data_frames = frame.index_select(1, data_q)  # (E, Nd, 2)
    has_data_error = (data_frames.sum(dim=(1, 2)) > 0)
    keep = detected | (~has_data_error)
    return keep.to(torch.uint8)


@torch.no_grad()
def apply_keep_deferral_to_detectors_torch(
    detector_frame: torch.Tensor,  # (E, D, 2) uint8
    keep_mask: torch.Tensor,  # (E,) uint8
    *,
    origin_round: int,
    nq: int,
) -> torch.Tensor:
    start = int(origin_round) * int(nq)
    end = start + int(nq)
    out = detector_frame.clone()
    keep = keep_mask.to(torch.uint8).view(-1, 1, 1)
    out[:, start:end, :] = out[:, start:end, :] * keep
    return out


# =============================================================================
# Error-basis generation (pure python, converted to torch)
# =============================================================================


def generate_all_errors_local(
    *,
    t_total: int,
    nq: int,
    controls_by_layer: np.ndarray,  # (L,G,2)
    cx_times: np.ndarray,  # (L,)
) -> tuple[np.ndarray, list[tuple[int, int, int, str, int]]]:
    """
    Mirror legacy `generate_all_errors`, but return numpy arrays.

    Returns:
        errors_local: (E, nq, 3) int8
        metadata_local: list of (err_idx, time, qubit, err_type, q2_or_minus1)
    """
    I = (0, 0)
    X = (1, 0)
    Z = (0, 1)
    Y = (1, 1)

    errors: list[np.ndarray] = [np.zeros((nq, 3), dtype=np.int8)]
    metadata: list[tuple[int, int, int, str, int]] = [(0, -1, -1, "I", -1)]
    counter = 1

    # Precompute per-time gate participation sets and per-time CNOTs.
    for tt in range(int(t_total)):
        # Find layer for this time (cx_times start at 1).
        layer_mask = (cx_times == tt)
        pairs: list[tuple[int, int]] = []
        if layer_mask.any():
            layer = controls_by_layer[layer_mask][0]  # (G,2)
            for c, t in layer.tolist():
                if int(c) >= 0 and int(t) >= 0:
                    pairs.append((int(c), int(t)))
        active = set()
        for c, t in pairs:
            active.add(c)
            active.add(t)

        # Two-qubit errors at CNOT locations: one location per CNOT, keyed by (time, control).
        for c_q, t_q in pairs:
            loc = (tt, c_q)
            # Guard against repeated controls in a layer (shouldn't happen).
            # Skip duplicates when repeated.
            # Use a set local to this time step.
            # (This only affects pathological inputs.)
            # We'll just proceed with unique controls.
            # Create all 15 non-identity Paulis in {I,X,Y,Z}^2.
            for e1, e1n in ((I, "I"), (X, "X"), (Z, "Z"), (Y, "Y")):
                for e2, e2n in ((I, "I"), (X, "X"), (Z, "Z"), (Y, "Y")):
                    if e1 == I and e2 == I:
                        continue
                    cur = np.zeros((nq, 3), dtype=np.int8)
                    cur[c_q, 0] = e1[0]
                    cur[c_q, 1] = e1[1]
                    cur[c_q, 2] = np.int8(tt)
                    cur[t_q, 0] = e2[0]
                    cur[t_q, 1] = e2[1]
                    cur[t_q, 2] = np.int8(tt)
                    errors.append(cur)
                    metadata.append((counter, int(tt), int(c_q), f"{e1n}{e2n}", int(t_q)))
                    counter += 1

        # Single-qubit errors for idle qubits
        for q in range(int(nq)):
            if q in active:
                continue
            for e1, e1n in ((X, "X"), (Z, "Z"), (Y, "Y")):
                cur = np.zeros((nq, 3), dtype=np.int8)
                cur[q, 0] = e1[0]
                cur[q, 1] = e1[1]
                cur[q, 2] = np.int8(tt)
                errors.append(cur)
                metadata.append((counter, int(tt), int(q), e1n, -1))
                counter += 1

    return np.stack(errors, axis=0).astype(np.int8), metadata


def replicate_metadata_across_rounds(
    *,
    metadata_local: list[tuple[int, int, int, str, int]],
    n_rounds: int,
) -> list[tuple[int, int, int, int, str, int]]:
    """Build error_metadata_global: (err_idx, round, time, qubit, err_type, q2)."""
    e_local = len(metadata_local)
    non_id = e_local - 1
    out: list[tuple[int, int, int, int, str, int]] = [(0, -1, -1, -1, "I", -1)]
    for r in range(int(n_rounds)):
        base = 1 + r * non_id
        for local_idx in range(1, e_local):
            g = base + (local_idx - 1)
            _li, tt, q, et, q2 = metadata_local[local_idx]
            out.append((int(g), int(r), int(tt), int(q), str(et), int(q2)))
    return out


# =============================================================================
# Timelike map A (dense) from dependency masks (pure numpy)
# =============================================================================


def build_meas_new_masks_from_data_numpy(
    *,
    controls_by_layer: np.ndarray,  # (L,G,2) int32
    nq: int,
    data_qubits: np.ndarray,  # (Nd,)
    meas_qubits: np.ndarray,  # (m,)
    meas_bases: np.ndarray,  # (m,)
) -> np.ndarray:
    """
    Numpy port of `build_meas_new_masks_from_data` producing (m,2,words) uint32.
    """
    data_qubits = np.array(data_qubits, dtype=np.int32).reshape(-1)
    meas_qubits = np.array(meas_qubits, dtype=np.int32).reshape(-1)
    meas_bases = np.array(meas_bases, dtype=np.int32).reshape(-1)
    n_data = int(data_qubits.shape[0])
    words = (n_data + 31) // 32

    x_deps = np.zeros((int(nq), words), dtype=np.uint32)
    z_deps = np.zeros((int(nq), words), dtype=np.uint32)
    for di, q in enumerate(data_qubits.tolist()):
        w = di // 32
        b = di % 32
        bit = np.uint32(1) << np.uint32(b)
        x_deps[int(q), w] ^= bit
        z_deps[int(q), w] ^= bit

    for layer in range(int(controls_by_layer.shape[0])):
        cs = controls_by_layer[layer, :, 0].reshape(-1)
        ts = controls_by_layer[layer, :, 1].reshape(-1)
        valid = (cs >= 0) & (ts >= 0)
        cs = cs[valid]
        ts = ts[valid]
        for c, t in zip(cs.tolist(), ts.tolist()):
            x_deps[int(t), :] ^= x_deps[int(c), :]
            z_deps[int(c), :] ^= z_deps[int(t), :]

    m = int(meas_qubits.shape[0])
    masks = np.zeros((m, 2, words), dtype=np.uint32)
    for j in range(m):
        q = int(meas_qubits[j])
        b = int(meas_bases[j])
        if b == 1:
            masks[j, 0, :] = x_deps[q, :]
        else:
            masks[j, 1, :] = z_deps[q, :]
    return masks


def build_dense_A_from_masks(
    *,
    masks_u32: np.ndarray,  # (m,2,words)
    data_qubits: np.ndarray,  # (Nd,)
    nq: int,
    n_rounds: int,
) -> np.ndarray:
    """
    Build dense A: (n_rounds*m, 2*(n_rounds*nq)) uint8.
    This matches `data/precompute_frames.py` export semantics.
    """
    masks = np.array(masks_u32, dtype=np.uint32)
    data_qubits = np.array(data_qubits, dtype=np.int32).reshape(-1)
    m = int(masks.shape[0])
    Ddet = int(n_rounds) * int(nq)
    A = np.zeros((int(n_rounds) * m, 2 * Ddet), dtype=np.uint8)

    for j in range(m):
        for comp in (0, 1):
            bits = masks[j, comp, :]
            for di in range(int(data_qubits.shape[0])):
                w = di // 32
                b = di % 32
                if (int(bits[w]) >> b) & 1:
                    q = int(data_qubits[di])
                    for rr in range(int(n_rounds)):
                        det = rr * int(nq) + q
                        row = rr * m + j
                        col = det if comp == 0 else (Ddet + det)
                        A[row, col] ^= 1
    return A


# =============================================================================
# p vector export (single-p marginal; copied from tests/print_bell_multiround_frame.py)
# =============================================================================


def build_single_p_marginal(
    *,
    error_metadata_global: list[tuple[int, int, int, int, str, int]],
    t_total: int,
    n_rounds: int,
    data_qubits: np.ndarray,
    xcheck_qubits: np.ndarray,
    zcheck_qubits: np.ndarray,
    meas_qubits: np.ndarray,
    meas_bases: np.ndarray,
    basis: str,
    p_scalar: float,
    noise_model=None,
) -> np.ndarray:
    data_set = set(int(x) for x in np.array(data_qubits).reshape(-1).tolist())
    meas_set = set(int(x) for x in np.array(meas_qubits).reshape(-1).tolist())
    xcheck_set = set(int(x) for x in np.array(xcheck_qubits).reshape(-1).tolist())

    prep_basis_map: dict[tuple[int, int], int] = {}
    data_prep_basis = 0 if str(basis).upper() == "X" else 1
    for r in range(int(n_rounds)):
        if r == 0 or r == int(n_rounds) - 1:
            for q in data_set:
                prep_basis_map[(r, q)] = int(data_prep_basis)
        for q in meas_set:
            prep_basis_map[(r, int(q))] = (0 if int(q) in xcheck_set else 1)

    meas_basis_map: dict[tuple[int, int], int] = {}
    for r in range(int(n_rounds)):
        for q, b in zip(
            np.array(meas_qubits).reshape(-1).tolist(),
            np.array(meas_bases).reshape(-1).tolist()
        ):
            meas_basis_map[(r, int(q))] = int(b)

    use_nm = noise_model is not None

    if use_nm:
        nm = noise_model
        _nm_single = {"X": {}, "Y": {}, "Z": {}}
        _nm_single["X"]["idle_cnot"] = float(nm.p_idle_cnot_X)
        _nm_single["Y"]["idle_cnot"] = float(nm.p_idle_cnot_Y)
        _nm_single["Z"]["idle_cnot"] = float(nm.p_idle_cnot_Z)
        _nm_single["X"]["idle_spam"] = float(nm.p_idle_spam_X)
        _nm_single["Y"]["idle_spam"] = float(nm.p_idle_spam_Y)
        _nm_single["Z"]["idle_spam"] = float(nm.p_idle_spam_Z)
        _nm_cnot = {}
        for ab in [
            "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
        ]:
            _nm_cnot[ab] = float(getattr(nm, f"p_cnot_{ab}"))
        p_prep_X = float(nm.p_prep_X)
        p_prep_Z = float(nm.p_prep_Z)
        p_meas_X = float(nm.p_meas_X)
        p_meas_Z = float(nm.p_meas_Z)
    else:
        spam_error = float(p_scalar) * 2.0 / 3.0
        combined_meas_error = 2.0 * spam_error * (1.0 - spam_error)

    E = int(max(e for (e, *_rest) in error_metadata_global)) + 1
    p_err = np.zeros((E,), dtype=np.float32)
    p_err[0] = 0.0

    for (eidx, r, tt, q, et, q2) in error_metadata_global:
        eidx = int(eidx)
        if eidx == 0:
            continue
        r = int(r)
        tt = int(tt)
        q = int(q)
        et = str(et)

        is_final_round = (r == int(n_rounds) - 1)
        is_prep = (tt == 0) and ((r, q) in prep_basis_map)
        is_meas = (tt == int(t_total) - 1) and ((r, q) in meas_basis_map)

        is_data = (q in data_set)
        is_meas_qubit = (q in meas_set)

        is_ancilla_prep = is_prep and is_meas_qubit
        is_ancilla_meas = is_meas and is_meas_qubit
        is_data_prep = is_prep and is_data
        is_data_meas = (tt == int(t_total) - 1) and is_data and ((r, q) in prep_basis_map)

        if use_nm:
            if is_final_round and not (tt == 0 and is_data):
                p_err[eidx] = 0.0
                continue

            if len(et) == 2:
                # CNOT two-qubit error: direct lookup
                if is_final_round:
                    p_err[eidx] = 0.0
                else:
                    p_err[eidx] = float(_nm_cnot.get(et, 0.0))
            elif len(et) == 1:
                if is_ancilla_prep:
                    p_err[eidx] = 0.0
                elif is_ancilla_meas:
                    meas_basis = int(meas_basis_map[(r, q)])
                    if meas_basis == 0:
                        allowed = (et == "Z")
                    else:
                        allowed = (et == "X")
                    p_err[eidx] = float(
                        p_meas_X if meas_basis == 0 else p_meas_Z
                    ) if allowed else 0.0
                elif is_data_prep:
                    prep_basis = int(prep_basis_map[(r, q)])
                    if prep_basis == 0:
                        allowed = (et == "Z")
                    else:
                        allowed = (et == "X")
                    if is_final_round:
                        p_err[eidx] = float(
                            p_prep_X if prep_basis == 0 else p_prep_Z
                        ) if allowed else 0.0
                    else:
                        p_err[eidx] = float(
                            p_prep_X if prep_basis == 0 else p_prep_Z
                        ) if allowed else 0.0
                elif is_data_meas:
                    if is_final_round:
                        p_err[eidx] = 0.0
                    else:
                        p_err[eidx] = float(_nm_single.get(et, {}).get("idle_spam", 0.0))
                else:
                    # Bulk idle: use idle_cnot or idle_spam depending on time step
                    if is_final_round:
                        p_err[eidx] = 0.0
                    elif is_prep or is_data_meas:
                        p_err[eidx] = float(_nm_single.get(et, {}).get("idle_spam", 0.0))
                    else:
                        p_err[eidx] = float(_nm_single.get(et, {}).get("idle_cnot", 0.0))
            else:
                p_err[eidx] = 0.0
        else:
            # Legacy scalar noise model path (unchanged)
            if is_ancilla_prep:
                p_non_final = 0.0
            elif is_ancilla_meas:
                p_non_final = float(combined_meas_error)
            elif is_data_prep or is_data_meas:
                p_non_final = float(spam_error)
            else:
                p_non_final = float(p_scalar)

            if is_final_round:
                p_adjusted = float(spam_error) if (tt == 0 and is_data) else 0.0
            else:
                p_adjusted = p_non_final

            if len(et) == 1:
                if is_prep:
                    prep_basis = int(prep_basis_map[(r, q)])
                    allowed = (et == ("Z" if prep_basis == 0 else "X"))
                    K = 1
                elif is_meas:
                    meas_basis = int(meas_basis_map[(r, q)])
                    allowed = (et == ("Z" if meas_basis == 0 else "X"))
                    K = 1
                else:
                    allowed = et in ("X", "Z", "Y")
                    K = 3
                p_err[eidx] = (p_adjusted / float(K)) if allowed else 0.0
            else:
                p_err[eidx] = float(p_adjusted) / 15.0

    return p_err


# =============================================================================
# End-to-end entrypoint
# =============================================================================


@torch.no_grad()
def precompute_dem_bundle_surface_code(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    p_scalar: float,
    dem_output_dir: str | None,
    device: torch.device,
    export: bool = True,
    return_artifacts: bool = False,
    noise_model=None,
) -> Path | dict[str, torch.Tensor | int]:
    from qec.surface_code.memory_circuit import MemoryCircuit

    distance = int(distance)
    n_rounds = int(n_rounds)
    basis = str(basis).upper()
    code_rotation = str(code_rotation).upper()
    p_scalar = float(p_scalar)

    # Build circuit (Stim text) and extract CX structure.
    # When an explicit NoiseModel is provided, MemoryCircuit uses its per-type
    # probabilities (PAULI_CHANNEL_1/2) instead of the uniform scalar rates.
    # The scalar rates are still passed as placeholders (MemoryCircuit requires them).
    circ = MemoryCircuit(
        distance=distance,
        idle_error=p_scalar,
        sqgate_error=p_scalar,
        tqgate_error=p_scalar,
        spam_error=2.0 / 3.0 * p_scalar,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=code_rotation,
        noise_model=noise_model,
    )
    circ.set_error_rates()
    cnot_circuit, cx_times = extract_cnot_structure_from_stim_text(circ.circuit)
    t_total = int(len(cx_times) + 2)
    nq = int(2 * distance * distance - 1)

    data_qubits = np.array(circ.code.data_qubits, dtype=np.int32)
    xcheck_qubits = np.array(circ.code.xcheck_qubits, dtype=np.int32)
    zcheck_qubits = np.array(circ.code.zcheck_qubits, dtype=np.int32)
    meas_qubits = np.concatenate([xcheck_qubits, zcheck_qubits]).astype(np.int32)
    meas_bases = np.concatenate(
        [np.zeros(len(xcheck_qubits), np.int32),
         np.ones(len(zcheck_qubits), np.int32)]
    ).astype(np.int32)

    # Generate local error basis + metadata.
    errors_local_np, metadata_local = generate_all_errors_local(
        t_total=t_total, nq=nq, controls_by_layer=cnot_circuit, cx_times=cx_times
    )
    errors_local = torch.from_numpy(errors_local_np).to(device=device, dtype=torch.int8)

    # Single-round frames + keep mask.
    frame_single = presample_frame_single_round_torch(
        t_total=t_total,
        nq=nq,
        controls_by_layer=cnot_circuit,
        cx_times=cx_times,
        errors=errors_local
    )
    m_local = _torch_measure(frame_single, meas_qubits, meas_bases)
    keep_local = _torch_keep_idx(m_local, frame_single, data_qubits)  # (E_local,)

    # Multi-round detector propagation (for single-round basis).
    det_seq = presample_detector_seq_multiround_torch(
        frame_single_round=frame_single,
        controls_by_layer=cnot_circuit,
        meas_qubits=meas_qubits,
        n_rounds=n_rounds,
    )  # (E_local, R, nq, 2)

    # Build global detector-frame tensor frame_predecoder: (E_total, Ddet, 2)
    num_errors_local = int(errors_local_np.shape[0])
    non_id = num_errors_local - 1
    num_errors_total = 1 + int(n_rounds) * non_id
    num_detectors = int(n_rounds) * int(nq)

    det_nonid = det_seq[1:]  # (non_id, R, nq, 2)
    keep_nonid = keep_local[1:]  # (non_id,)

    frames_by_origin: list[torch.Tensor] = []
    for origin in range(int(n_rounds)):
        if origin == 0:
            rounds_full = det_nonid
        else:
            prefix = torch.zeros((non_id, origin, nq, 2), dtype=torch.uint8, device=device)
            tail = det_nonid[:, :(n_rounds - origin), :, :]
            rounds_full = torch.cat([prefix, tail], dim=1)
        flat = rounds_full.reshape(non_id, num_detectors, 2)
        flat_kept = apply_keep_deferral_to_detectors_torch(
            flat, keep_nonid, origin_round=origin, nq=nq
        )
        frames_by_origin.append(flat_kept)

    frame_predecoder = torch.cat(
        [torch.zeros((1, num_detectors, 2), dtype=torch.uint8, device=device)] + frames_by_origin,
        dim=0
    )
    assert int(frame_predecoder.shape[0]) == int(num_errors_total)

    # Export p (single-p marginal)
    metadata_global = replicate_metadata_across_rounds(
        metadata_local=metadata_local, n_rounds=n_rounds
    )
    p_err = build_single_p_marginal(
        error_metadata_global=metadata_global,
        t_total=t_total,
        n_rounds=n_rounds,
        data_qubits=data_qubits,
        xcheck_qubits=xcheck_qubits,
        zcheck_qubits=zcheck_qubits,
        meas_qubits=meas_qubits,
        meas_bases=meas_bases,
        basis=basis,
        p_scalar=p_scalar,
        noise_model=noise_model,
    ).astype(np.float32)

    # Export A (dense timelike map)
    masks = build_meas_new_masks_from_data_numpy(
        controls_by_layer=cnot_circuit,
        nq=nq,
        data_qubits=data_qubits,
        meas_qubits=meas_qubits,
        meas_bases=meas_bases,
    )
    A = build_dense_A_from_masks(masks_u32=masks, data_qubits=data_qubits, nq=nq, n_rounds=n_rounds)

    # Optional: return in-memory artifacts for training without writing files.
    # NOTE: Computing H requires large transposes (can be the dominant allocation).
    if return_artifacts:
        HX = frame_predecoder[:, :, 0].T.contiguous()  # (Ddet, E)
        HZ = frame_predecoder[:, :, 1].T.contiguous()
        H = torch.cat([HX, HZ], dim=0).to(dtype=torch.uint8)  # (2*Ddet, E)
        p_t = torch.from_numpy(p_err).to(device=device, dtype=torch.float32)
        A_t = torch.from_numpy(A.astype(np.uint8)).to(device=device, dtype=torch.uint8)
        return {
            "H": H,
            "p": p_t,
            "A": A_t,
            "nq": int(nq),
            "num_detectors": int(num_detectors),
        }

    if export:
        if dem_output_dir is None:
            raise ValueError("dem_output_dir must be provided when export=True")
        dem_dir = Path(dem_output_dir)
        dem_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"surface_d{distance}_r{n_rounds}_{basis}_frame_predecoder"

        # Export H (detectors x errors). These transposes can be large, so avoid allocating them
        # unless we are actually exporting.
        HX = frame_predecoder[:, :, 0].T.contiguous()  # (Ddet, E)
        HZ = frame_predecoder[:, :, 1].T.contiguous()

        np.savez_compressed(dem_dir / f"{prefix}.X.npz", HX=HX.cpu().numpy().astype(np.uint8))
        np.savez_compressed(dem_dir / f"{prefix}.Z.npz", HZ=HZ.cpu().numpy().astype(np.uint8))
        np.savez_compressed(
            dem_dir / f"{prefix}.p.npz", p=p_err, p_nominal=np.array(p_scalar, dtype=np.float32)
        )
        np.savez_compressed(dem_dir / f"{prefix}.A.npz", A=A.astype(np.uint8))
        return dem_dir

    # Benchmark/no-save mode: do all compute, but don't write artifacts.
    return Path(".")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distance", "-d", type=int, required=True)
    ap.add_argument("--n_rounds", "-r", type=int, default=None)
    ap.add_argument("--basis", "-b", type=str, choices=["X", "Z"], required=True)
    ap.add_argument("--rotation", "--rot", type=str, default="XV", choices=["XV", "XH", "ZV", "ZH"])
    ap.add_argument(
        "--p", type=float, default=0.01, help="Scalar p for exporting single-p marginals"
    )
    ap.add_argument("--dem_output_dir", type=str, default=None)
    ap.add_argument(
        "--no_save", action="store_true", help="Run precompute but do not write any files"
    )
    ap.add_argument(
        "--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu (default: auto)"
    )
    args = ap.parse_args()

    d = int(args.distance)
    r = int(args.n_rounds) if args.n_rounds is not None else d
    dev = (
        torch.device(args.device) if args.device is not None else
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    precompute_dem_bundle_surface_code(
        distance=d,
        n_rounds=r,
        basis=str(args.basis),
        code_rotation=str(args.rotation),
        p_scalar=float(args.p),
        dem_output_dir=(str(args.dem_output_dir) if args.dem_output_dir is not None else None),
        device=dev,
        export=(not bool(args.no_save)),
    )


if __name__ == "__main__":
    main()
