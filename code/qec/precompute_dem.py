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

For `--code color`, this exports an augmented DEM bundle with rows for
predecoder data frames, physical measurements, and s1/s2 measurements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

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

DEM_ARTIFACT_METADATA_VERSION = 1
DEM_ARTIFACT_METADATA_KEY = "metadata_json"


def build_dem_artifact_metadata(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    p_scalar: float,
    noise_model=None,
) -> dict[str, Any]:
    """Build metadata for a DEM artifact.

    The structural keys identify whether cached H/A frame artifacts can be
    reused. The probability keys record provenance for the stored p vector, but
    probability changes alone do not invalidate the structural DEM.
    """
    metadata: dict[str, Any] = {
        "schema_version": DEM_ARTIFACT_METADATA_VERSION,
        "distance": int(distance),
        "n_rounds": int(n_rounds),
        "basis": str(basis).upper(),
        "code_rotation": str(code_rotation).upper(),
    }
    if noise_model is None:
        metadata.update({
            "noise_mode": "scalar",
            "p_scalar": float(p_scalar),
        })
    else:
        metadata.update(
            {
                "noise_mode": "noise_model",
                "p_scalar_placeholder": float(p_scalar),
                "noise_model_sha256": noise_model.sha256(),
                "noise_model": noise_model.canonical_parameters(),
            }
        )
    return metadata


def encode_dem_artifact_metadata(metadata: dict[str, Any]) -> str:
    """Serialize metadata in a deterministic JSON form."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_dem_artifact_metadata(value) -> dict[str, Any]:
    """Decode metadata loaded from an npz scalar/string array."""
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def load_dem_artifact_metadata(p_path: Path) -> dict[str, Any] | None:
    """Return metadata from a .p.npz file, or None for legacy artifacts."""
    with np.load(p_path, allow_pickle=False) as z:
        if DEM_ARTIFACT_METADATA_KEY not in z.files:
            return None
        return decode_dem_artifact_metadata(z[DEM_ARTIFACT_METADATA_KEY])


def dem_artifact_metadata_matches(
    metadata: dict[str, Any] | None,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    p_scalar: float,
    noise_model=None,
) -> tuple[bool, str]:
    """Check whether on-disk DEM metadata matches the requested structure."""
    if metadata is None:
        return True, "legacy artifact without structural metadata"

    expected = build_dem_artifact_metadata(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=code_rotation,
        p_scalar=p_scalar,
        noise_model=noise_model,
    )
    for key in ("schema_version", "distance", "n_rounds", "basis", "code_rotation"):
        if metadata.get(key) != expected.get(key):
            return False, f"metadata {key}={metadata.get(key)!r} != expected {expected.get(key)!r}"

    return True, "structural metadata matches"


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
        # In the explicit noise model, every non-final stabilizer round has a
        # data-idle SPAM window after the ancilla measurement/reset step.  Do
        # not gate this location on prep_basis_map: data qubits are prepared
        # only in the boundary rounds, but they idle in this window every round.
        is_data_spam_idle = (tt == int(t_total) - 1) and is_data

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
                    # Stim emits Z_ERROR(p_prep_X) on X-basis-reset ancillas and
                    # X_ERROR(p_prep_Z) on Z-basis-reset ancillas (see MemoryCircuit.
                    # add_reset / add_single_error). Treat prep as its own one-Pauli
                    # fault channel, consistent with get_grouped_totals' "X/Z prep
                    # and measurement are separate one-Pauli fault channels" rule
                    # and with linear behaviour under uniform noise upscaling.
                    prep_basis = int(prep_basis_map[(r, q)])
                    if prep_basis == 0:
                        allowed = (et == "Z")
                    else:
                        allowed = (et == "X")
                    p_err[eidx] = float(
                        p_prep_X if prep_basis == 0 else p_prep_Z
                    ) if allowed else 0.0
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
                        # The circuit represents noisy data readout as a fake
                        # data-measurement flip injected at the start of the final
                        # perfect round, at the *measurement* rates: Z_ERROR(p_meas_X)
                        # for X-basis readout, X_ERROR(p_meas_Z) for Z-basis (see
                        # MemoryCircuit's logical_measurement injection).
                        p_err[eidx] = float(
                            p_meas_X if prep_basis == 0 else p_meas_Z
                        ) if allowed else 0.0
                    else:
                        p_err[eidx] = float(
                            p_prep_X if prep_basis == 0 else p_prep_Z
                        ) if allowed else 0.0
                elif is_data_spam_idle:
                    if is_final_round:
                        p_err[eidx] = 0.0
                    else:
                        p_err[eidx] = float(_nm_single.get(et, {}).get("idle_spam", 0.0))
                elif tt == 0 and is_data:
                    # Data qubits idling during the ancilla prep/reset window carry
                    # no separate noise location: the circuit folds that idle into
                    # the measurement-window SPAM idle above and emits nothing here
                    # (see MemoryCircuit: "IGNORE data-idle during ancilla
                    # prep/reset"). Boundary rounds never reach this branch: their
                    # tt == 0 data locations are consumed by is_data_prep.
                    p_err[eidx] = 0.0
                else:
                    # Remaining single-qubit locations are bulk/CNOT-layer idles.
                    if is_final_round:
                        p_err[eidx] = 0.0
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


def build_probability_vector_surface_code(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    p_scalar: float,
    noise_model=None,
) -> np.ndarray:
    """Build only the per-error probability vector for a surface-code DEM.

    Cached DEM frame artifacts encode the possible detector responses. The
    active scalar/noise model only determines how likely each error column is,
    so callers can reuse cached H/A artifacts and refresh p with this helper.
    """
    from qec.surface_code.memory_circuit import MemoryCircuit

    distance = int(distance)
    n_rounds = int(n_rounds)
    basis = str(basis).upper()
    code_rotation = str(code_rotation).upper()
    p_scalar = float(p_scalar)

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

    _, metadata_local = generate_all_errors_local(
        t_total=t_total, nq=nq, controls_by_layer=cnot_circuit, cx_times=cx_times
    )
    metadata_global = replicate_metadata_across_rounds(
        metadata_local=metadata_local, n_rounds=n_rounds
    )
    return build_single_p_marginal(
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
        metadata = build_dem_artifact_metadata(
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            code_rotation=code_rotation,
            p_scalar=p_scalar,
            noise_model=noise_model,
        )
        np.savez_compressed(
            dem_dir / f"{prefix}.p.npz",
            p=p_err,
            p_nominal=np.array(p_scalar, dtype=np.float32),
            **{DEM_ARTIFACT_METADATA_KEY: np.array(encode_dem_artifact_metadata(metadata))},
        )
        np.savez_compressed(dem_dir / f"{prefix}.A.npz", A=A.astype(np.uint8))
        return dem_dir

    # Benchmark/no-save mode: do all compute, but don't write artifacts.
    return Path(".")


# =============================================================================
# Color-code augmented DEM export (Torch recurrence)
# =============================================================================

COLOR_AUGMENTED_DEM_METADATA_VERSION = 1
COLOR_AUGMENTED_DEM_MAX_DENSE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class ColorAugmentedDemBundle:
    """Dense augmented color-code response matrix and row layout metadata."""

    H: torch.Tensor
    n_rounds: int
    num_local_errors: int
    num_data: int
    num_meas: int
    num_z: int
    num_x: int
    frame_rows: int
    meas_old_rows: int
    meas_new_rows: int
    use_decomposed_errors: bool

    @property
    def num_rows(self) -> int:
        return int(self.H.shape[0])

    @property
    def num_cols(self) -> int:
        return int(self.H.shape[1])


def color_augmented_dem_prefix(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
) -> str:
    """Stable artifact prefix for color-code augmented DEM bundles."""
    schedule_tag = str(schedule).replace("/", "_").replace(" ", "_")
    return f"color_d{int(distance)}_r{int(n_rounds)}_{str(basis).upper()}_{schedule_tag}_augmented_dem"


def get_color_augmented_dem_paths(
    dem_dir: str | Path,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
) -> dict[str, Path]:
    """Return H/p artifact paths for a color-code augmented DEM bundle."""
    base = Path(dem_dir)
    prefix = color_augmented_dem_prefix(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    return {
        "H": base / f"{prefix}.H.npz",
        "p": base / f"{prefix}.p.npz",
    }


def color_augmented_dem_artifacts_exist(
    dem_dir: str | Path,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
) -> bool:
    paths = get_color_augmented_dem_paths(
        dem_dir,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    return all(path.exists() for path in paths.values())


def build_color_augmented_dem_metadata(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
    p_scalar: float,
    enable_z_feedforward: bool,
    apply_data_x_override: bool,
    use_decomposed_errors: bool,
    noise_model=None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": COLOR_AUGMENTED_DEM_METADATA_VERSION,
        "code": "color",
        "distance": int(distance),
        "n_rounds": int(n_rounds),
        "basis": str(basis).upper(),
        "schedule": str(schedule),
        "enable_z_feedforward": bool(enable_z_feedforward),
        "apply_data_x_override": bool(apply_data_x_override),
        "use_decomposed_errors": bool(use_decomposed_errors),
    }
    if noise_model is None:
        metadata.update({
            "noise_mode": "scalar",
            "p_scalar": float(p_scalar),
        })
    else:
        metadata.update(
            {
                "noise_mode": "noise_model",
                "p_scalar_placeholder": float(p_scalar),
                "noise_model_sha256": noise_model.sha256(),
                "noise_model": noise_model.canonical_parameters(),
            }
        )
    return metadata


def _color_metadata_matches(
    metadata: dict[str, Any] | None,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
    enable_z_feedforward: bool,
    apply_data_x_override: bool,
    use_decomposed_errors: bool,
) -> tuple[bool, str]:
    if metadata is None:
        return False, "missing color augmented DEM metadata"
    # Structural keys identify whether cached H artifacts can be reused; the
    # probability vector encodes its own noise mode separately, so we don't
    # require noise_mode to match here.
    expected = build_color_augmented_dem_metadata(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=float(metadata.get("p_scalar", 0.0)),
        enable_z_feedforward=enable_z_feedforward,
        apply_data_x_override=apply_data_x_override,
        use_decomposed_errors=use_decomposed_errors,
    )
    for key in (
        "schema_version",
        "code",
        "distance",
        "n_rounds",
        "basis",
        "schedule",
        "enable_z_feedforward",
        "apply_data_x_override",
        "use_decomposed_errors",
    ):
        if metadata.get(key) != expected.get(key):
            return False, f"metadata {key}={metadata.get(key)!r} != expected {expected.get(key)!r}"
    return True, "color augmented DEM metadata matches"


def _color_global_col(round_idx: int, local_error_idx: int, num_local_errors: int) -> int:
    if int(local_error_idx) <= 0:
        return 0
    return 1 + int(round_idx) * (int(num_local_errors) - 1) + (int(local_error_idx) - 1)


def _flatten_color_augmented_rows(
    frame_data: torch.Tensor,
    meas_old: torch.Tensor,
    meas_new: torch.Tensor,
) -> torch.Tensor:
    """Flatten Torch raw outputs from (batch, rounds, ...) into (batch, rows)."""
    batch = int(frame_data.shape[0])
    return torch.cat(
        [
            frame_data.reshape(batch, -1),
            meas_old.reshape(batch, -1),
            meas_new.reshape(batch, -1),
        ],
        dim=1,
    ).to(dtype=torch.uint8)


def build_circuit_z_ancilla_connectivity_matrix(
    *,
    controls: np.ndarray,
    targets: np.ndarray,
    data_qubits: np.ndarray,
    zcheck_qubits: np.ndarray,
    nq: int,
) -> np.ndarray:
    """Build the Z-check ancilla to data-qubit feedforward matrix."""
    zcheck_np = np.array(zcheck_qubits, dtype=np.int32).reshape(-1)
    data_np = np.array(data_qubits, dtype=np.int32).reshape(-1)
    mat = np.zeros((zcheck_np.size, int(nq)), dtype=np.uint8)
    if zcheck_np.size == 0 or data_np.size == 0:
        return mat

    z_to_row = {int(z): i for i, z in enumerate(zcheck_np.tolist())}
    data_set = set(int(q) for q in data_np.tolist())

    c = np.array(controls, dtype=np.int32).reshape(-1)
    t = np.array(targets, dtype=np.int32).reshape(-1)
    valid = (c >= 0) & (t >= 0)
    c = c[valid]
    t = t[valid]

    for cq, tq in zip(c.tolist(), t.tolist()):
        cq = int(cq)
        tq = int(tq)
        if cq in z_to_row and tq in data_set:
            mat[z_to_row[cq], tq] = 1
        elif tq in z_to_row and cq in data_set:
            mat[z_to_row[tq], cq] = 1

    return mat


def _torch_zero_qubits(frame: torch.Tensor, qubits: np.ndarray | torch.Tensor) -> torch.Tensor:
    out = frame.clone()
    q = torch.as_tensor(qubits, dtype=torch.long, device=frame.device).reshape(-1)
    if q.numel() > 0:
        out[:, q, :] = 0
    return out


def _torch_apply_data_x_override(
    frame_total: torch.Tensor,
    sampled_frame_full: torch.Tensor,
    rep_carry_data_x: torch.Tensor,
    x_flip_sampled: torch.Tensor,
    data_qubits: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    data_q = torch.as_tensor(data_qubits, dtype=torch.long, device=frame_total.device).reshape(-1)
    sampled_x = sampled_frame_full.index_select(1, data_q)[:, :, 0].to(torch.uint8)
    data_x = rep_carry_data_x.to(torch.uint8) ^ sampled_x ^ x_flip_sampled.to(torch.uint8)
    out = frame_total.clone()
    out[:, data_q, 0] = data_x
    return out


@torch.no_grad()
def _torch_run_batched_shots_with_injected_errors(
    sampled_indices_per_shot: torch.Tensor | np.ndarray,
    frame: torch.Tensor,
    keep: torch.Tensor,
    *,
    buffer_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate explicit local error indices into predecoder/out frames."""
    dev = frame.device
    idx = torch.as_tensor(sampled_indices_per_shot, dtype=torch.long, device=dev)
    if idx.ndim == 1:
        idx = idx[:, None]
    if idx.ndim != 2:
        raise ValueError("sampled_indices_per_shot must have shape (batch, slots)")

    buffer_size = max(1, int(buffer_size))
    if int(idx.shape[1]) > buffer_size:
        raise ValueError(
            f"sampled_indices_per_shot has {int(idx.shape[1])} slots, "
            f"but buffer_size={buffer_size}; increase buffer_size rather than truncating faults"
        )

    active = idx > 0
    safe_idx = torch.where(active, idx, torch.zeros_like(idx))
    batch, slots = int(safe_idx.shape[0]), int(safe_idx.shape[1])

    shot_frames = frame.index_select(0, safe_idx.reshape(-1)
                                    ).reshape(batch, slots, int(frame.shape[1]), 2)
    active_w = active.to(torch.uint8).view(batch, slots, 1, 1)
    keep_w = keep.index_select(0, safe_idx.reshape(-1)).reshape(batch, slots)
    keep_w = keep_w.to(torch.uint8).view(batch, slots, 1, 1)

    predecoder_frame = torch.remainder(
        (shot_frames * active_w * keep_w).sum(dim=1, dtype=torch.int32),
        2,
    ).to(torch.uint8)
    out_frame = torch.remainder(
        (shot_frames * active_w).sum(dim=1, dtype=torch.int32),
        2,
    ).to(torch.uint8)
    return predecoder_frame, out_frame


@torch.no_grad()
def _torch_run_color_rounds_with_injected_errors(
    sampled_indices_per_round: torch.Tensor | np.ndarray,
    frame: torch.Tensor,
    keep: torch.Tensor,
    data_qubits: np.ndarray,
    meas_qubits_per_round: np.ndarray,
    meas_bases_per_round: np.ndarray,
    controls: np.ndarray,
    targets: np.ndarray,
    *,
    buffer_size: int = 64,
    feedforward_mask: np.ndarray | None = None,
    apply_data_x_override: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch port of the color-code injected-error multi-round recurrence."""
    dev = frame.device
    sampled = torch.as_tensor(sampled_indices_per_round, dtype=torch.long, device=dev)
    if sampled.ndim != 3:
        raise ValueError("sampled_indices_per_round must have shape (rounds, batch, slots)")

    n_rounds = int(sampled.shape[0])
    num_shots = int(sampled.shape[1])
    num_qubits = int(frame.shape[1])

    data_q = torch.as_tensor(data_qubits, dtype=torch.long, device=dev).reshape(-1)
    controls_t = torch.as_tensor(controls, dtype=torch.long, device=dev)
    targets_t = torch.as_tensor(targets, dtype=torch.long, device=dev)

    has_feedforward = feedforward_mask is not None
    ff_parity_data = None
    num_z = 0
    if has_feedforward:
        ff_mask = torch.as_tensor(feedforward_mask, dtype=torch.uint8, device=dev)
        num_z = int(ff_mask.shape[0])
        num_data = int(data_q.numel())
        if int(ff_mask.shape[1]) == num_qubits:
            ff_parity_data = ff_mask.index_select(1, data_q)
        elif int(ff_mask.shape[1]) == num_data:
            ff_parity_data = ff_mask
        else:
            raise ValueError(
                f"feedforward_mask shape {tuple(ff_mask.shape)} unsupported; "
                f"expected ({num_z},{num_qubits}) or ({num_z},{num_data})"
            )

    meas_qubits_per_round = np.asarray(meas_qubits_per_round, dtype=np.int32)
    meas_bases_per_round = np.asarray(meas_bases_per_round, dtype=np.int32)

    physical_carry = torch.zeros((num_shots, num_qubits, 2), dtype=torch.uint8, device=dev)
    rep_carry_data_x = torch.zeros((num_shots, int(data_q.numel())), dtype=torch.uint8, device=dev)

    frame_outputs: list[torch.Tensor] = []
    meas_old_outputs: list[torch.Tensor] = []
    meas_new_outputs: list[torch.Tensor] = []

    for round_idx in range(n_rounds):
        meas_q_np = meas_qubits_per_round[round_idx].reshape(-1)
        meas_b_np = meas_bases_per_round[round_idx].reshape(-1)

        accumulated_frame = propagate_frame_one_round_torch(
            physical_carry.clone(),
            controls_t,
            targets_t,
        )
        predecoder_frame_full, out_frame_full = _torch_run_batched_shots_with_injected_errors(
            sampled[round_idx],
            frame,
            keep,
            buffer_size=buffer_size,
        )

        predecoder_frame_total = predecoder_frame_full ^ accumulated_frame
        out_frame_total = out_frame_full ^ accumulated_frame

        out_m = _torch_measure(out_frame_total, meas_q_np, meas_b_np)

        x_flip_sampled = torch.zeros_like(rep_carry_data_x, dtype=torch.uint8)
        if has_feedforward and num_z > 0 and ff_parity_data is not None:
            out_m_z = _torch_measure(out_frame_total, meas_q_np[:num_z], meas_b_np[:num_z])
            x_flip_counts = (out_m_z.to(torch.float32) @ ff_parity_data.to(torch.float32))
            x_flip_data = torch.remainder(x_flip_counts.to(torch.int64), 2).to(torch.uint8)

            if apply_data_x_override:
                out_m_z_sampled = _torch_measure(
                    predecoder_frame_full,
                    meas_q_np[:num_z],
                    meas_b_np[:num_z],
                )
                x_flip_sampled_counts = (
                    out_m_z_sampled.to(torch.float32) @ ff_parity_data.to(torch.float32)
                )
                x_flip_sampled = torch.remainder(x_flip_sampled_counts.to(torch.int64),
                                                 2).to(torch.uint8)

            x_flip_full = torch.zeros((num_shots, num_qubits), dtype=torch.uint8, device=dev)
            x_flip_full[:, data_q] = x_flip_data
            predecoder_frame_total = predecoder_frame_total.clone()
            out_frame_total = out_frame_total.clone()
            predecoder_frame_total[:, :, 0] ^= x_flip_full
            out_frame_total[:, :, 0] ^= x_flip_full

        out_frame_total_physical = out_frame_total

        out_m_kept = _torch_measure(predecoder_frame_total, meas_q_np, meas_b_np)
        predecoder_carry = _torch_zero_qubits(predecoder_frame_total, meas_q_np)
        predecoder_carry_propagated = propagate_frame_one_round_torch(
            predecoder_carry,
            controls_t,
            targets_t,
        )
        new_out_m_kept = _torch_measure(predecoder_carry_propagated, meas_q_np, meas_b_np)
        new_out_m = new_out_m_kept ^ out_m_kept

        if apply_data_x_override:
            predecoder_frame_total = _torch_apply_data_x_override(
                predecoder_frame_total,
                predecoder_frame_full,
                rep_carry_data_x,
                x_flip_sampled,
                data_q,
            )
            out_frame_total = _torch_apply_data_x_override(
                out_frame_total,
                out_frame_full,
                rep_carry_data_x,
                x_flip_sampled,
                data_q,
            )

        frame_outputs.append(predecoder_frame_total.index_select(1, data_q))
        meas_old_outputs.append(out_m)
        meas_new_outputs.append(new_out_m)

        if apply_data_x_override:
            physical_carry = _torch_zero_qubits(out_frame_total_physical, meas_q_np)
            rep_carry_data_x = out_frame_total.index_select(1, data_q)[:, :, 0].to(torch.uint8)
        else:
            physical_carry = _torch_zero_qubits(out_frame_total, meas_q_np)
            rep_carry_data_x = torch.zeros_like(rep_carry_data_x, dtype=torch.uint8)

    return (
        torch.stack(frame_outputs, dim=1).to(torch.uint8),
        torch.stack(meas_old_outputs, dim=1).to(torch.uint8),
        torch.stack(meas_new_outputs, dim=1).to(torch.uint8),
    )


def _build_color_sampled_indices_for_columns(
    *,
    columns: list[tuple[int, int]],
    n_rounds: int,
    buffer_size: int,
) -> np.ndarray:
    sampled = np.zeros(
        (int(n_rounds), len(columns), max(1, int(buffer_size))),
        dtype=np.int32,
    )
    for shot_idx, (round_idx, local_error_idx) in enumerate(columns):
        if int(local_error_idx) > 0:
            sampled[int(round_idx), shot_idx, 0] = int(local_error_idx)
    return sampled


def _build_color_memory_circuit(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
    p_scalar: float,
):
    from qec.color_code.memory_circuit import MemoryCircuit

    return MemoryCircuit(
        distance=int(distance),
        idle_error=float(p_scalar),
        sqgate_error=float(p_scalar),
        tqgate_error=float(p_scalar),
        spam_error=(2.0 / 3.0) * float(p_scalar),
        n_rounds=int(n_rounds),
        basis=str(basis).upper(),
        add_tick=True,
        add_detectors=False,
        noise_model=None,
        schedule=str(schedule),
    )


def _extract_color_round_layout(
    *,
    circ,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str,
) -> dict[str, Any]:
    circuit_text = str(circ.circuit)
    if "REPEAT" not in circuit_text:
        tmp_circ = _build_color_memory_circuit(
            distance=distance,
            n_rounds=3,
            basis=basis,
            schedule=schedule,
            p_scalar=0.0,
        )
        circuit_text = str(tmp_circ.circuit)

    cnot_circuit, cx_times = extract_cnot_structure_from_stim_text(circuit_text)
    t_total = int(len(cx_times) + 2)
    nq = int(np.asarray(circ.code.all_qubits).size)

    data_qubits = np.array(circ.code.data_qubits, dtype=np.int32).reshape(-1)
    xcheck_qubits = np.array(circ.code.xcheck_qubits, dtype=np.int32).reshape(-1)
    zcheck_qubits = np.array(circ.code.zcheck_qubits, dtype=np.int32).reshape(-1)
    meas_qubits = np.concatenate([zcheck_qubits, xcheck_qubits]).astype(np.int32)
    meas_bases = np.concatenate(
        [
            np.ones(len(zcheck_qubits), dtype=np.int32),
            np.zeros(len(xcheck_qubits), dtype=np.int32),
        ]
    ).astype(np.int32)

    return {
        "cnot_circuit": cnot_circuit,
        "cx_times": cx_times,
        "t_total": t_total,
        "nq": nq,
        "data_qubits": data_qubits,
        "xcheck_qubits": xcheck_qubits,
        "zcheck_qubits": zcheck_qubits,
        "meas_qubits": meas_qubits,
        "meas_bases": meas_bases,
        "meas_qubits_per_round": np.tile(meas_qubits, (int(n_rounds), 1)).astype(np.int32),
        "meas_bases_per_round": np.tile(meas_bases, (int(n_rounds), 1)).astype(np.int32),
    }


def build_single_p_marginal_color_code(
    *,
    error_metadata_global: list[tuple[int, int, int, int, str, int]],
    t_total: int,
    n_rounds: int,
    data_qubits: np.ndarray,
    xcheck_qubits: np.ndarray,
    zcheck_qubits: np.ndarray,
    basis: str,
    p_scalar: float,
    noise_model=None,
) -> np.ndarray:
    """Build color-code single-p marginals matching the injected-error sampler.

    When ``noise_model`` is provided, per-fault-type rates from the 25p model are
    assigned directly per entry (mirroring the surface-code 25p path); otherwise
    the legacy scalar-p iid path is used.
    """
    use_nm = noise_model is not None

    if use_nm:
        nm = noise_model
        nm_idle_cnot = {
            "X": float(nm.p_idle_cnot_X),
            "Y": float(nm.p_idle_cnot_Y),
            "Z": float(nm.p_idle_cnot_Z),
        }
        nm_idle_spam = {
            "X": float(nm.p_idle_spam_X),
            "Y": float(nm.p_idle_spam_Y),
            "Z": float(nm.p_idle_spam_Z),
        }
        nm_cnot = {
            ab: float(getattr(nm, f"p_cnot_{ab}")) for ab in [
                "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY",
                "ZZ"
            ]
        }
        p_prep_X_nm = float(nm.p_prep_X)
        p_prep_Z_nm = float(nm.p_prep_Z)
        p_meas_X_nm = float(nm.p_meas_X)
        p_meas_Z_nm = float(nm.p_meas_Z)
    else:
        p_scalar = float(p_scalar)
        spam_error = p_scalar * 2.0 / 3.0

    data_set = set(int(q) for q in np.array(data_qubits).reshape(-1).tolist())
    xcheck_set = set(int(q) for q in np.array(xcheck_qubits).reshape(-1).tolist())
    zcheck_set = set(int(q) for q in np.array(zcheck_qubits).reshape(-1).tolist())
    meas_set = zcheck_set | xcheck_set

    data_basis = 0 if str(basis).upper() == "X" else 1
    prep_basis_map: dict[tuple[int, int], int] = {}
    meas_basis_map: dict[tuple[int, int], int] = {}
    for round_idx in range(int(n_rounds)):
        if round_idx == 0 or round_idx == int(n_rounds) - 1:
            for q in data_set:
                prep_basis_map[(round_idx, q)] = int(data_basis)
        for q in zcheck_set:
            prep_basis_map[(round_idx, q)] = 1
            meas_basis_map[(round_idx, q)] = 1
        for q in xcheck_set:
            prep_basis_map[(round_idx, q)] = 0
            meas_basis_map[(round_idx, q)] = 0

    num_errors = int(max(e for (e, *_rest) in error_metadata_global)) + 1
    p_err = np.zeros((num_errors,), dtype=np.float32)

    groups: dict[tuple[int, int, int, int], list[tuple[int, str]]] = {}
    for eidx, round_idx, tt, q, err_type, q2 in error_metadata_global:
        eidx = int(eidx)
        if eidx == 0:
            continue
        groups.setdefault(
            (int(round_idx), int(tt), int(q), int(q2)),
            [],
        ).append((eidx, str(err_type)))

    for (round_idx, tt, q, _q2), entries in groups.items():
        is_final_round = round_idx == int(n_rounds) - 1
        is_prep = tt == 0 and (round_idx, q) in prep_basis_map
        is_meas = tt == int(t_total) - 1 and (round_idx, q) in meas_basis_map
        is_data = q in data_set
        is_meas_qubit = q in meas_set

        valid_entries: list[tuple[int, str]] = []
        for eidx, err_type in entries:
            valid = True
            if len(err_type) == 1:
                if is_prep:
                    prep_basis = int(prep_basis_map[(round_idx, q)])
                    valid = err_type == ("Z" if prep_basis == 0 else "X")
                elif is_meas:
                    meas_basis = int(meas_basis_map[(round_idx, q)])
                    valid = err_type == ("Z" if meas_basis == 0 else "X")
            if valid:
                valid_entries.append((eidx, err_type))

        if not valid_entries:
            continue

        is_ancilla_prep = is_prep and is_meas_qubit and not is_data
        is_ancilla_meas = is_meas and is_meas_qubit and not is_data
        is_data_prep = is_prep and is_data
        is_data_meas = is_meas and is_data
        is_data_prep_final = tt == 0 and is_data and (round_idx, q) in prep_basis_map

        if use_nm:
            for eidx, err_type in valid_entries:
                if is_final_round and not is_data_prep_final:
                    p_err[int(eidx)] = 0.0
                    continue
                if len(err_type) == 2:
                    p_err[int(eidx)] = nm_cnot.get(err_type, 0.0)
                elif len(err_type) == 1:
                    if is_ancilla_prep:
                        prep_basis = int(prep_basis_map[(round_idx, q)])
                        p_err[int(eidx)] = p_prep_X_nm if prep_basis == 0 else p_prep_Z_nm
                    elif is_ancilla_meas:
                        meas_basis = int(meas_basis_map[(round_idx, q)])
                        p_err[int(eidx)] = p_meas_X_nm if meas_basis == 0 else p_meas_Z_nm
                    elif is_data_prep:
                        prep_basis = int(prep_basis_map[(round_idx, q)])
                        p_err[int(eidx)] = p_prep_X_nm if prep_basis == 0 else p_prep_Z_nm
                    elif is_data_meas:
                        p_err[int(eidx)] = nm_idle_spam.get(err_type, 0.0)
                    else:
                        p_err[int(eidx)] = nm_idle_cnot.get(err_type, 0.0)
                else:
                    p_err[int(eidx)] = 0.0
        else:
            p_non_final = (
                spam_error if
                (is_ancilla_prep or is_ancilla_meas or is_data_prep or is_data_meas) else p_scalar
            )
            if is_final_round:
                p_adjusted = spam_error if is_data_prep_final else 0.0
            else:
                p_adjusted = p_non_final

            per_error_prob = float(p_adjusted) / float(len(valid_entries))
            for eidx, _err_type in valid_entries:
                p_err[int(eidx)] = per_error_prob

    return p_err


def build_probability_vector_color_code(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str = "nearest-neighbor",
    p_scalar: float,
    noise_model=None,
) -> np.ndarray:
    """Build the global color-code augmented DEM probability vector."""
    circ = _build_color_memory_circuit(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=p_scalar,
    )
    layout = _extract_color_round_layout(
        circ=circ,
        distance=int(distance),
        n_rounds=int(n_rounds),
        basis=str(basis).upper(),
        schedule=str(schedule),
    )
    _errors_local, metadata_local = generate_all_errors_local(
        t_total=int(layout["t_total"]),
        nq=int(layout["nq"]),
        controls_by_layer=layout["cnot_circuit"],
        cx_times=layout["cx_times"],
    )
    metadata_global = replicate_metadata_across_rounds(
        metadata_local=metadata_local,
        n_rounds=int(n_rounds),
    )
    return build_single_p_marginal_color_code(
        error_metadata_global=metadata_global,
        t_total=int(layout["t_total"]),
        n_rounds=int(n_rounds),
        data_qubits=layout["data_qubits"],
        xcheck_qubits=layout["xcheck_qubits"],
        zcheck_qubits=layout["zcheck_qubits"],
        basis=str(basis).upper(),
        p_scalar=float(p_scalar),
        noise_model=noise_model,
    ).astype(np.float32)


@torch.no_grad()
def build_color_augmented_dem_bundle_torch(
    *,
    frame: torch.Tensor,
    keep: torch.Tensor,
    n_rounds: int,
    data_qubits: np.ndarray,
    zcheck_qubits: np.ndarray,
    xcheck_qubits: np.ndarray,
    meas_qubits_per_round: np.ndarray,
    meas_bases_per_round: np.ndarray,
    controls: np.ndarray,
    targets: np.ndarray,
    feedforward_mask: np.ndarray | None,
    apply_data_x_override: bool,
    use_decomposed_errors: bool = False,
    chunk_size: int = 256,
    buffer_size: int = 1,
) -> ColorAugmentedDemBundle:
    """Build the dense color-code augmented DEM response matrix using Torch only."""
    if use_decomposed_errors:
        raise NotImplementedError("Torch color augmented DEM does not yet support Y decomposition")

    n_rounds = int(n_rounds)
    chunk_size = max(1, int(chunk_size))
    buffer_size = max(1, int(buffer_size))
    num_local_errors = int(frame.shape[0])
    num_data = int(np.array(data_qubits).reshape(-1).shape[0])
    num_z = int(np.array(zcheck_qubits).reshape(-1).shape[0])
    num_x = int(np.array(xcheck_qubits).reshape(-1).shape[0])
    num_meas = num_z + num_x
    frame_rows = n_rounds * num_data * 2
    meas_old_rows = n_rounds * num_meas
    meas_new_rows = n_rounds * num_meas
    num_rows = frame_rows + meas_old_rows + meas_new_rows
    num_cols = 1 + n_rounds * (num_local_errors - 1)
    dense_bytes = int(num_rows) * int(num_cols) * np.dtype(np.uint8).itemsize
    if dense_bytes > COLOR_AUGMENTED_DEM_MAX_DENSE_BYTES:
        max_gib = COLOR_AUGMENTED_DEM_MAX_DENSE_BYTES / float(1024**3)
        got_gib = dense_bytes / float(1024**3)
        raise MemoryError(
            "Dense color augmented DEM precompute is intended for PID-scale "
            f"configs up to about d=r=13. Requested H shape=({num_rows}, {num_cols}) "
            f"would require {got_gib:.2f} GiB, above the {max_gib:.2f} GiB guard."
        )

    H = torch.zeros((num_rows, num_cols), dtype=torch.uint8)
    columns = [
        (round_idx, local_idx)
        for round_idx in range(n_rounds)
        for local_idx in range(1, num_local_errors)
    ]

    for start in range(0, len(columns), chunk_size):
        chunk = columns[start:start + chunk_size]
        sampled = _build_color_sampled_indices_for_columns(
            columns=chunk,
            n_rounds=n_rounds,
            buffer_size=buffer_size,
        )
        frame_data, meas_old, meas_new = _torch_run_color_rounds_with_injected_errors(
            sampled,
            frame,
            keep,
            np.array(data_qubits, dtype=np.int32),
            meas_qubits_per_round,
            meas_bases_per_round,
            controls,
            targets,
            buffer_size=buffer_size,
            feedforward_mask=feedforward_mask,
            apply_data_x_override=bool(apply_data_x_override),
        )
        rows = _flatten_color_augmented_rows(frame_data, meas_old, meas_new).cpu()
        cols = torch.tensor(
            [
                _color_global_col(round_idx, local_idx, num_local_errors)
                for round_idx, local_idx in chunk
            ],
            dtype=torch.long,
        )
        H[:, cols] = rows.T.contiguous()

    return ColorAugmentedDemBundle(
        H=H,
        n_rounds=n_rounds,
        num_local_errors=num_local_errors,
        num_data=num_data,
        num_meas=num_meas,
        num_z=num_z,
        num_x=num_x,
        frame_rows=frame_rows,
        meas_old_rows=meas_old_rows,
        meas_new_rows=meas_new_rows,
        use_decomposed_errors=bool(use_decomposed_errors),
    )


@torch.no_grad()
def precompute_dem_bundle_color_code(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str = "nearest-neighbor",
    p_scalar: float,
    dem_output_dir: str | None,
    device: torch.device | None = None,
    export: bool = True,
    return_artifacts: bool = False,
    enable_z_feedforward: bool = True,
    apply_data_x_override: bool = True,
    use_decomposed_errors: bool = False,
    chunk_size: int = 256,
    buffer_size: int = 1,
    noise_model=None,
) -> Path | dict[str, torch.Tensor | int | float]:
    """Precompute a Torch-native color-code augmented DEM bundle."""
    if use_decomposed_errors:
        raise NotImplementedError("Torch color augmented DEM does not yet support Y decomposition")
    if not enable_z_feedforward or not apply_data_x_override:
        raise ValueError(
            "Color augmented DEM precompute requires enable_z_feedforward=True and "
            "apply_data_x_override=True for production training."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    distance = int(distance)
    n_rounds = int(n_rounds)
    basis = str(basis).upper()
    p_scalar = float(p_scalar)
    schedule = str(schedule)

    circ = _build_color_memory_circuit(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=p_scalar,
    )
    layout = _extract_color_round_layout(
        circ=circ,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )

    cnot_circuit = layout["cnot_circuit"]
    cx_times = layout["cx_times"]
    errors_local_np, metadata_local = generate_all_errors_local(
        t_total=int(layout["t_total"]),
        nq=int(layout["nq"]),
        controls_by_layer=cnot_circuit,
        cx_times=cx_times,
    )
    errors_local = torch.from_numpy(errors_local_np).to(device=device, dtype=torch.int8)

    frame_single = presample_frame_single_round_torch(
        t_total=int(layout["t_total"]),
        nq=int(layout["nq"]),
        controls_by_layer=cnot_circuit,
        cx_times=cx_times,
        errors=errors_local,
    )
    m_local = _torch_measure(frame_single, layout["meas_qubits"], layout["meas_bases"])
    keep_local = _torch_keep_idx(m_local, frame_single, layout["data_qubits"])

    feedforward_mask = None
    if enable_z_feedforward:
        feedforward_mask = build_circuit_z_ancilla_connectivity_matrix(
            controls=cnot_circuit[:, :, 0],
            targets=cnot_circuit[:, :, 1],
            data_qubits=layout["data_qubits"],
            zcheck_qubits=layout["zcheck_qubits"],
            nq=int(layout["nq"]),
        )

    bundle = build_color_augmented_dem_bundle_torch(
        frame=frame_single,
        keep=keep_local,
        n_rounds=n_rounds,
        data_qubits=layout["data_qubits"],
        zcheck_qubits=layout["zcheck_qubits"],
        xcheck_qubits=layout["xcheck_qubits"],
        meas_qubits_per_round=layout["meas_qubits_per_round"],
        meas_bases_per_round=layout["meas_bases_per_round"],
        controls=cnot_circuit[:, :, 0],
        targets=cnot_circuit[:, :, 1],
        feedforward_mask=feedforward_mask,
        apply_data_x_override=bool(apply_data_x_override),
        use_decomposed_errors=bool(use_decomposed_errors),
        chunk_size=int(chunk_size),
        buffer_size=int(buffer_size),
    )

    metadata_global = replicate_metadata_across_rounds(
        metadata_local=metadata_local,
        n_rounds=n_rounds,
    )
    p_vec_np = build_single_p_marginal_color_code(
        error_metadata_global=metadata_global,
        t_total=int(layout["t_total"]),
        n_rounds=n_rounds,
        data_qubits=layout["data_qubits"],
        xcheck_qubits=layout["xcheck_qubits"],
        zcheck_qubits=layout["zcheck_qubits"],
        basis=basis,
        p_scalar=p_scalar,
        noise_model=noise_model,
    ).astype(np.float32)
    p_vec = torch.from_numpy(p_vec_np)

    metadata = build_color_augmented_dem_metadata(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
        p_scalar=p_scalar,
        enable_z_feedforward=bool(enable_z_feedforward),
        apply_data_x_override=bool(apply_data_x_override),
        use_decomposed_errors=bool(use_decomposed_errors),
        noise_model=noise_model,
    )

    if return_artifacts:
        return {
            "H": bundle.H.to(device=device, dtype=torch.uint8),
            "p": p_vec.to(device=device, dtype=torch.float32),
            "p_nominal": float(p_scalar),
            "n_rounds": int(bundle.n_rounds),
            "num_local_errors": int(bundle.num_local_errors),
            "num_data": int(bundle.num_data),
            "num_meas": int(bundle.num_meas),
            "num_z": int(bundle.num_z),
            "num_x": int(bundle.num_x),
            "frame_rows": int(bundle.frame_rows),
            "meas_old_rows": int(bundle.meas_old_rows),
            "meas_new_rows": int(bundle.meas_new_rows),
        }

    if export:
        if dem_output_dir is None:
            raise ValueError("dem_output_dir must be provided when export=True")
        dem_dir = Path(dem_output_dir)
        dem_dir.mkdir(parents=True, exist_ok=True)
        paths = get_color_augmented_dem_paths(
            dem_dir,
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            schedule=schedule,
        )
        metadata_json = np.array(encode_dem_artifact_metadata(metadata))
        np.savez_compressed(
            paths["H"],
            H=bundle.H.cpu().numpy().astype(np.uint8),
            n_rounds=np.array(bundle.n_rounds, dtype=np.int64),
            num_local_errors=np.array(bundle.num_local_errors, dtype=np.int64),
            num_data=np.array(bundle.num_data, dtype=np.int64),
            num_meas=np.array(bundle.num_meas, dtype=np.int64),
            num_z=np.array(bundle.num_z, dtype=np.int64),
            num_x=np.array(bundle.num_x, dtype=np.int64),
            frame_rows=np.array(bundle.frame_rows, dtype=np.int64),
            meas_old_rows=np.array(bundle.meas_old_rows, dtype=np.int64),
            meas_new_rows=np.array(bundle.meas_new_rows, dtype=np.int64),
            use_decomposed_errors=np.array(bool(use_decomposed_errors), dtype=np.bool_),
            **{DEM_ARTIFACT_METADATA_KEY: metadata_json},
        )
        np.savez_compressed(
            paths["p"],
            p=p_vec_np.astype(np.float32),
            p_nominal=np.array(p_scalar, dtype=np.float32),
            **{DEM_ARTIFACT_METADATA_KEY: metadata_json},
        )
        return dem_dir

    return Path(".")


def load_color_augmented_dem_bundle(
    dem_dir: str | Path,
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    schedule: str = "nearest-neighbor",
    device: torch.device | None = None,
    enable_z_feedforward: bool = True,
    apply_data_x_override: bool = True,
    use_decomposed_errors: bool = False,
    strict_metadata: bool = True,
) -> dict[str, Any]:
    """Load a precomputed color-code augmented DEM bundle."""
    paths = get_color_augmented_dem_paths(
        dem_dir,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        schedule=schedule,
    )
    if not paths["H"].exists() or not paths["p"].exists():
        raise FileNotFoundError(
            f"Missing color augmented DEM artifacts: {paths['H']} and/or {paths['p']}"
        )

    with np.load(paths["H"], allow_pickle=False) as z:
        metadata = (
            decode_dem_artifact_metadata(z[DEM_ARTIFACT_METADATA_KEY])
            if DEM_ARTIFACT_METADATA_KEY in z.files else None
        )
        ok, reason = _color_metadata_matches(
            metadata,
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            schedule=schedule,
            enable_z_feedforward=enable_z_feedforward,
            apply_data_x_override=apply_data_x_override,
            use_decomposed_errors=use_decomposed_errors,
        )
        if strict_metadata and not ok:
            raise ValueError(f"Color augmented DEM metadata mismatch: {reason}")
        bundle = ColorAugmentedDemBundle(
            H=torch.from_numpy(np.asarray(z["H"], dtype=np.uint8)),
            n_rounds=int(np.asarray(z["n_rounds"]).reshape(())),
            num_local_errors=int(np.asarray(z["num_local_errors"]).reshape(())),
            num_data=int(np.asarray(z["num_data"]).reshape(())),
            num_meas=int(np.asarray(z["num_meas"]).reshape(())),
            num_z=int(np.asarray(z["num_z"]).reshape(())),
            num_x=int(np.asarray(z["num_x"]).reshape(())),
            frame_rows=int(np.asarray(z["frame_rows"]).reshape(())),
            meas_old_rows=int(np.asarray(z["meas_old_rows"]).reshape(())),
            meas_new_rows=int(np.asarray(z["meas_new_rows"]).reshape(())),
            use_decomposed_errors=bool(np.asarray(z["use_decomposed_errors"]).reshape(())),
        )

    with np.load(paths["p"], allow_pickle=False) as z:
        p = torch.from_numpy(np.asarray(z["p"], dtype=np.float32))
        p_nominal = float(np.asarray(z["p_nominal"]).reshape(()))

    if device is not None:
        bundle = ColorAugmentedDemBundle(
            H=bundle.H.to(device=device, dtype=torch.uint8),
            n_rounds=bundle.n_rounds,
            num_local_errors=bundle.num_local_errors,
            num_data=bundle.num_data,
            num_meas=bundle.num_meas,
            num_z=bundle.num_z,
            num_x=bundle.num_x,
            frame_rows=bundle.frame_rows,
            meas_old_rows=bundle.meas_old_rows,
            meas_new_rows=bundle.meas_new_rows,
            use_decomposed_errors=bundle.use_decomposed_errors,
        )
        p = p.to(device=device, dtype=torch.float32)

    return {
        "bundle": bundle,
        "p": p,
        "p_nominal": p_nominal,
        "metadata": metadata,
        "paths": paths,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", type=str, default="surface", choices=["surface", "color"])
    ap.add_argument("--distance", "-d", type=int, required=True)
    ap.add_argument("--n_rounds", "-r", type=int, default=None)
    ap.add_argument("--basis", "-b", type=str, choices=["X", "Z"], required=True)
    ap.add_argument(
        "--rotation",
        "--rot",
        type=str,
        default=None,
        choices=["XV", "XH", "ZV", "ZH"],
        help="(--code surface) Rotation orientation; not supported with --code color.",
    )
    ap.add_argument("--schedule", type=str, default="nearest-neighbor")
    ap.add_argument(
        "--enable_z_feedforward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(--code color) Apply Z-ancilla feedforward to data qubits.",
    )
    ap.add_argument(
        "--apply_data_x_override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(--code color) Override data-qubit X frame with per-shot sampled X frame.",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="(--code color) Column chunk size when building the augmented DEM.",
    )
    ap.add_argument(
        "--buffer_size",
        type=int,
        default=1,
        help="(--code color) Per-shot slot count for injected errors during DEM build.",
    )
    ap.add_argument(
        "--p", type=float, default=0.01, help="Scalar p for exporting single-p marginals"
    )
    ap.add_argument(
        "--noise_model_config",
        type=str,
        default=None,
        help="YAML/JSON config containing data.noise_model, noise_model, or a direct 25p mapping",
    )
    ap.add_argument(
        "--noise_model_json",
        type=str,
        default=None,
        help="JSON file containing data.noise_model, noise_model, or a direct 25p mapping",
    )
    ap.add_argument("--dem_output_dir", type=str, default=None)
    ap.add_argument(
        "--no_save", action="store_true", help="Run precompute but do not write any files"
    )
    ap.add_argument(
        "--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu (default: auto)"
    )
    args = ap.parse_args()

    if args.noise_model_config is not None and args.noise_model_json is not None:
        ap.error("Use only one of --noise_model_config or --noise_model_json")
    noise_model = None
    noise_model_path = args.noise_model_config or args.noise_model_json
    if noise_model_path is not None:
        # Defer the import to avoid a circular dependency with data.precompute_frames
        # at module-import time (precompute_frames.py imports from this module).
        from data.precompute_frames import _load_noise_model
        noise_model = _load_noise_model(noise_model_path)

    d = int(args.distance)
    r = int(args.n_rounds) if args.n_rounds is not None else d
    dev = (
        torch.device(args.device) if args.device is not None else
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if str(args.code).lower() == "color":
        if args.rotation is not None:
            ap.error("--rotation is not supported with --code color")
        if not bool(args.enable_z_feedforward) or not bool(args.apply_data_x_override):
            ap.error(
                "--code color requires --enable_z_feedforward and --apply_data_x_override "
                "for production DEM precompute"
            )
        precompute_dem_bundle_color_code(
            distance=d,
            n_rounds=r,
            basis=str(args.basis),
            schedule=str(args.schedule),
            p_scalar=float(args.p),
            dem_output_dir=(str(args.dem_output_dir) if args.dem_output_dir is not None else None),
            device=dev,
            export=(not bool(args.no_save)),
            enable_z_feedforward=bool(args.enable_z_feedforward),
            apply_data_x_override=bool(args.apply_data_x_override),
            use_decomposed_errors=False,
            chunk_size=int(args.chunk_size),
            buffer_size=int(args.buffer_size),
            noise_model=noise_model,
        )
    else:
        precompute_dem_bundle_surface_code(
            distance=d,
            n_rounds=r,
            basis=str(args.basis),
            code_rotation=str(args.rotation) if args.rotation is not None else "XV",
            p_scalar=float(args.p),
            dem_output_dir=(str(args.dem_output_dir) if args.dem_output_dir is not None else None),
            device=dev,
            export=(not bool(args.no_save)),
            noise_model=noise_model,
        )


if __name__ == "__main__":
    main()
