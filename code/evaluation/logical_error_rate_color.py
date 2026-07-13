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
Logical Error Rate computation for Color Code using Stim + Chromobius.

This module provides Stim-based data generation and Chromobius decoding for
computing logical error rates with trained color code pre-decoder models.

Key Features:
1. Uses Stim's MemoryCircuit for color code sample generation
2. Uses Chromobius decoder (designed for color codes with proper detector coordinates)
3. Computes residual syndromes from model predictions
4. Supports X, Z, and mixed basis validation
5. Supports Gidney's exact noise model for fair benchmark comparison

Usage:
    from evaluation.logical_error_rate_color import count_logical_errors_color
    
    result = count_logical_errors_color(model, device, dist, cfg)
    # Returns dict with LER for X and/or Z basis

    # For benchmark comparison with Gidney's paper:
    result = compute_ler_with_gidney_noise(distance=5, p=0.001, n_rounds=20, basis='X', num_samples=100000)

Author: AI Assistant (based on surface code logical_error_rate.py)
"""

import chromobius
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from copy import deepcopy
import random
from typing import Optional

from torch.utils.data import DataLoader
from training.utils import dict_to_device

from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
from qec.color_code.color_code import ColorCode
from qec.color_code.data_mapping import (
    get_stab_to_grid_flat_index,
    get_data_to_grid_flat_index,
    get_parity_matrix_data_only,
)
from qec.noise_model import (
    normalize_noise_instruction_semantics,
    normalize_noise_model_family,
    resolve_test_noise_model,
)


def _packbits_gpu(t: torch.Tensor) -> torch.Tensor:
    """Pack (B, N) uint8 bit tensor into (B, ceil(N/8)) uint8, little-endian (LSB first).

    Equivalent to np.packbits(x, axis=1, bitorder='little') but runs on the same
    device as the input, avoiding a large GPU→CPU transfer before Chromobius.
    """
    B, N = t.shape
    pad = (8 - N % 8) % 8
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int32, device=t.device)
    return (t.view(B, -1, 8).to(torch.int32) * powers).sum(dim=2).to(torch.uint8)


def _sync_for_timing(device):
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _start_timing(device):
    _sync_for_timing(device)
    return time.perf_counter()


def _elapsed_since(t0, device):
    _sync_for_timing(device)
    return time.perf_counter() - t0


def sample_predictions(
    logits: torch.Tensor,
    threshold: float = 0.0,
    sampling_mode: str = "threshold",
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Convert logits to binary predictions using either thresholding or temperature sampling.
    
    Args:
        logits: Raw model outputs (before sigmoid)
        threshold: Decision threshold for deterministic mode (default 0.0 for logits)
        sampling_mode: "threshold" for deterministic, "temperature" for stochastic
        temperature: Temperature parameter for softmax sampling
    
    Returns:
        Binary predictions as int32 tensor
    """
    if sampling_mode == "temperature":
        scaled_logits = logits / max(temperature, 1e-8)
        probs = torch.sigmoid(scaled_logits)
        return torch.bernoulli(probs).to(torch.int32)
    else:
        return (logits >= threshold).to(torch.int32)


def _resolve_color_noise_settings(cfg):
    """
    Resolve high-level color-code noise settings for Stim-based evaluation.

    Preferred config axes:
    - ``test.noise_model_family``: ``legacy`` | ``si1000``
    - ``test.noise_instruction_semantics``: ``current`` | ``reference``

    Backward compatibility:
    - ``test.noise_mode`` still maps onto ``noise_model_family``
    - ``test.noise_model`` remains the lower-level escape hatch for explicit
      dicts or ``train``/``none`` when ``noise_instruction_semantics=current``
    """
    test_cfg = getattr(cfg, "test", None)
    noise_model_family = normalize_noise_model_family(
        getattr(test_cfg, "noise_model_family", None),
        fallback_noise_mode=getattr(test_cfg, "noise_mode", None),
    )
    noise_instruction_semantics = normalize_noise_instruction_semantics(
        getattr(test_cfg, "noise_instruction_semantics", None)
    )
    gidney_style_noise = bool(getattr(test_cfg, "gidney_style_noise", False))
    if noise_instruction_semantics == "reference":
        if noise_model_family != "si1000":
            raise ValueError(
                "reference noise_instruction_semantics currently requires "
                "test.noise_model_family='si1000'."
            )
        noise_model_obj = None
        test_nm_mode = "reference"
    else:
        noise_model_obj, test_nm_mode = resolve_test_noise_model(cfg)

    return (
        noise_model_obj,
        test_nm_mode,
        noise_model_family,
        noise_instruction_semantics,
        gidney_style_noise,
    )


def _align_delta_s2_for_predecoder_mode(
    delta_s2: torch.Tensor,
    apply_feedforward_to_predecoder: bool,
) -> torch.Tensor:
    """
    Align FF cascade correction with the predecoder-label timing mode.

    - apply_feedforward_to_predecoder=True:
        FF is visible in round r labels, so use per-round delta_s2 as-is.
    - apply_feedforward_to_predecoder=False:
        FF is deferred to round r+1 labels (except final round, which is applied
        in-place because there is no next round). Shift delta by +1 round, with
        an explicit final-round override.
    """
    if apply_feedforward_to_predecoder:
        return delta_s2

    if delta_s2.ndim != 3:
        raise ValueError(f"delta_s2 must have shape (B, num_plaq, T), got {tuple(delta_s2.shape)}")

    T = int(delta_s2.shape[2])
    if T <= 1:
        return delta_s2

    aligned = torch.zeros_like(delta_s2)
    aligned[:, :, 1:] = delta_s2[:, :, :-1]
    # Final round is never deferred by the simulator; keep current-round correction.
    aligned[:, :, -1] = delta_s2[:, :, -1]
    return aligned


def _build_ff_cascade_tensors(circ_obj, num_plaq: int, num_data: int, device):
    """
    Build tensors needed to convert predecoder-frame syn_z predictions to inlined frame.
    """
    z_to_data = circ_obj._z_connected_data_by_z_ancilla()
    z_list = [int(z) for z in circ_obj.code.zcheck_qubits]
    ff_np = np.zeros((num_plaq, num_data), dtype=np.float32)
    for j, z in enumerate(z_list):
        for q in z_to_data.get(z, []):
            q = int(q)
            if 0 <= q < num_data:
                ff_np[j, q] = 1.0
    ff_mask_tensor = torch.from_numpy(ff_np).to(device)

    cx_controls = []
    cx_targets = []
    cur_c, cur_t = [], []
    hit_cx = False
    for op_name, op_tgts, _ in circ_obj.stim_circuit.flattened_operations():
        if op_name == "TICK":
            if cur_c:
                cx_controls.append(torch.tensor(cur_c, dtype=torch.long, device=device))
                cx_targets.append(torch.tensor(cur_t, dtype=torch.long, device=device))
                cur_c, cur_t = [], []
        elif op_name in ("CX", "CNOT"):
            hit_cx = True
            for ii in range(0, len(op_tgts), 2):
                cur_c.append(int(op_tgts[ii]))
                cur_t.append(int(op_tgts[ii + 1]))
        elif hit_cx and op_name in ("M", "MR", "MX", "MZ", "MRX", "MRZ"):
            if cur_c:
                cx_controls.append(torch.tensor(cur_c, dtype=torch.long, device=device))
                cx_targets.append(torch.tensor(cur_t, dtype=torch.long, device=device))
            break

    z_check_offset = int(circ_obj.code.num_data + circ_obj.code.num_plaquettes)
    num_total_qubits = int(circ_obj.code.num_data + 2 * circ_obj.code.num_plaquettes)
    return ff_mask_tensor, cx_controls, cx_targets, z_check_offset, num_total_qubits


def _compute_delta_s2_from_meas_flat(
    meas_flat: torch.Tensor,
    T: int,
    num_plaq: int,
    ff_mask_tensor: torch.Tensor,
    cx_controls: list,
    cx_targets: list,
    z_check_offset: int,
    num_total_qubits: int,
) -> torch.Tensor:
    """
    Compute FF cascade contribution in Z-check space from raw per-round measurements.
    Returns delta_s2 with shape (B, num_plaq, T), int32.
    """
    B = int(meas_flat.shape[0])
    num_data = int(ff_mask_tensor.shape[1])

    meas_3d = meas_flat.view(B, T, 2 * num_plaq)  # (B, T, 2*num_plaq)
    z_meas_raw = meas_3d[:, :, :num_plaq].float()  # (B, T, num_plaq)
    ff_data = (z_meas_raw @ ff_mask_tensor).remainder_(2)  # (B, T, num_data)

    x_frame = torch.zeros(B, T, num_total_qubits, dtype=torch.int32, device=meas_flat.device)
    x_frame[:, :, :num_data] = ff_data.to(torch.int32)
    for lc, lt in zip(cx_controls, cx_targets):
        x_frame[:, :, lt] = (x_frame[:, :, lt] + x_frame[:, :, lc]).remainder(2)

    delta_s2 = x_frame[:, :, z_check_offset:z_check_offset + num_plaq]
    return delta_s2.permute(0, 2, 1).to(torch.int32)


class CUDAPrefetcher:
    """CUDA stream-based data prefetcher for efficient GPU loading."""

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._preload()

    def _to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._to_device(x) for x in obj]
            return type(obj)(t)
        return obj

    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._to_device(batch)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        self._preload()
        return batch


def _iter_batches_with_decode_executor(loader, time_batches: bool = False):
    """Yield batches while scoping the background decoder executor to the loop."""
    with ThreadPoolExecutor(max_workers=1) as decode_executor:
        if not time_batches:
            for batch_idx, batch in enumerate(loader):
                yield batch_idx, batch, decode_executor, 0.0
            return

        loader_iter = iter(loader)
        batch_idx = 0
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            yield batch_idx, batch, decode_executor, time.perf_counter() - t0
            batch_idx += 1


@lru_cache(maxsize=16)
def _build_color_code_parity_maps(distance: int):
    """
    Build parity check matrix components for color code.
    
    Color code has SAME X and Z stabilizer support (plaquettes), so we only need
    one parity matrix. Returns components needed for syndrome computation.
    
    Args:
        distance: Color code distance
        
    Returns:
        dict with parity matrix components
    """
    code = ColorCode(distance)

    # Parity matrix: (num_plaquettes, num_data) - same for X and Z
    H = get_parity_matrix_data_only(code)  # torch.Tensor
    H_i32 = H.to(torch.int32)

    num_plaq, num_data = H_i32.shape

    # Build sparse index representation for gather-based syndrome computation
    nz = H_i32.nonzero(as_tuple=False)  # (nnz, 2) [row, col]
    rows = nz[:, 0]
    cols = nz[:, 1]

    # Degree per stabilizer and max degree (6 for bulk, 4 for boundary)
    deg = torch.bincount(rows, minlength=num_plaq)
    K = int(deg.max().item())  # Should be 6

    # Build index and mask arrays
    idx = torch.full((num_plaq, K), -1, dtype=torch.long)
    msk = torch.zeros((num_plaq, K), dtype=torch.bool)

    if K > 0:
        row_offsets = torch.zeros(num_plaq + 1, dtype=torch.long)
        row_offsets[1:] = deg.cumsum(0)
        pos = torch.arange(nz.size(0), dtype=torch.long) - row_offsets[rows]
        idx[rows, pos] = cols
        ar = torch.arange(K, dtype=torch.long).unsqueeze(0).expand(num_plaq, K)
        msk = ar < deg.unsqueeze(1)

    # Grid mapping indices
    stab_to_grid = get_stab_to_grid_flat_index(code)
    data_to_grid = get_data_to_grid_flat_index(code)

    return {
        "H_i32": H_i32,
        "H_idx": idx,
        "H_mask": msk,
        "H_deg": deg,
        "K": K,
        "stab_to_grid": stab_to_grid,
        "data_to_grid": data_to_grid,
        "num_plaq": num_plaq,
        "num_data": num_data,
        "n_rows": code.n_rows,
        "n_cols": code.n_cols,
    }


def map_grid_to_stab(grid_tensor: torch.Tensor, stab_indices: torch.Tensor) -> torch.Tensor:
    """
    Map grid tensor back to stabilizer order.
    
    Args:
        grid_tensor: (B, T, n_rows, n_cols) or (B, n_rows, n_cols) tensor
        stab_indices: (num_plaq,) flat grid indices for each stabilizer
        
    Returns:
        (B, num_plaq, T) or (B, num_plaq) tensor in stabilizer order
    """
    if grid_tensor.dim() == 4:
        B, T, n_rows, n_cols = grid_tensor.shape
        flat = grid_tensor.permute(0, 2, 3, 1).reshape(B, n_rows * n_cols, T)
        return flat.index_select(dim=1, index=stab_indices)  # (B, num_plaq, T)
    elif grid_tensor.dim() == 3:
        B, n_rows, n_cols = grid_tensor.shape
        flat = grid_tensor.reshape(B, n_rows * n_cols)
        return flat.index_select(dim=1, index=stab_indices)  # (B, num_plaq)
    else:
        raise ValueError(f"Unexpected grid_tensor dim: {grid_tensor.dim()}")


class PreDecoderColorEvalModule(torch.nn.Module):
    """
    Color-code eval pipeline from model input through residual detector assembly.

    Chromobius itself remains outside this module because it currently consumes
    CPU bit-packed detector arrays. This class owns the tensorized GPU portion:
    model forward, prediction sampling, parity reconstruction, logical-frame
    extraction, and residual detector layout. Keeping this boundary tensor-only
    is the prerequisite for a follow-up combined ONNX export and trtexec
    benchmark.
    """

    def __init__(
        self,
        model,
        cfg,
        maps: dict,
        *,
        basis: str,
        obs_support: torch.Tensor,
        num_boundary_dets: int,
        enable_delta_s2_correction: bool = False,
        enable_z_ff: bool = True,
        ff_mask_tensor: Optional[torch.Tensor] = None,
        cx_controls: Optional[list] = None,
        cx_targets: Optional[list] = None,
        z_check_offset: Optional[int] = None,
        num_total_qubits: Optional[int] = None,
        use_physical_frame_observable: bool = True,
        obs_ancilla_meas_indices: Optional[list] = None,
    ):
        super().__init__()
        self.model = model
        self.basis = str(basis).upper()
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

        self.th_data = float(getattr(cfg.test, "th_data", 0.0))
        self.th_syn = float(getattr(cfg.test, "th_syn", 0.0))
        self.sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
        temperature = float(getattr(cfg.test, "temperature", 1.0))
        temperature_data = getattr(cfg.test, "temperature_data", None)
        temperature_syn = getattr(cfg.test, "temperature_syn", None)
        temperature_data = float(temperature_data) if temperature_data is not None else temperature
        temperature_syn = float(temperature_syn) if temperature_syn is not None else temperature
        self.temperature_data = temperature_data
        self.temperature_syn = temperature_syn
        self.enable_fp16 = bool(getattr(cfg, "enable_fp16", False))

        self.num_plaq = int(maps["num_plaq"])
        self.num_data = int(maps["num_data"])
        self.n_rows = int(maps["n_rows"])
        self.n_cols = int(maps["n_cols"])
        self.K = int(maps["K"])
        self.num_boundary_dets = int(num_boundary_dets)
        self.enable_delta_s2_correction = bool(enable_delta_s2_correction)
        self.enable_z_ff = bool(enable_z_ff)
        self.use_physical_frame_observable = bool(use_physical_frame_observable)
        self.z_check_offset = z_check_offset
        self.num_total_qubits = num_total_qubits
        self.cx_controls = tuple(cx_controls or ())
        self.cx_targets = tuple(cx_targets or ())
        self.obs_ancilla_meas_indices = tuple(int(x) for x in (obs_ancilla_meas_indices or ()))

        self.register_buffer("H_idx", maps["H_idx"].to(dtype=torch.long), persistent=False)
        self.register_buffer("H_mask", maps["H_mask"].to(dtype=torch.bool), persistent=False)
        self.register_buffer(
            "stab_to_grid", maps["stab_to_grid"].to(dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "data_to_grid", maps["data_to_grid"].to(dtype=torch.long), persistent=False
        )
        self.register_buffer("obs_support", obs_support.to(dtype=torch.float32), persistent=False)
        if ff_mask_tensor is not None:
            self.register_buffer(
                "ff_mask_tensor", ff_mask_tensor.to(dtype=torch.float32), persistent=False
            )
        else:
            self.ff_mask_tensor = None

    def model_forward(self, trainX: torch.Tensor) -> torch.Tensor:
        device_type = trainX.device.type if trainX.device.type in ("cuda", "cpu") else "cpu"
        # Match the eval input layout to a channels_last_3d model so half-precision
        # Conv3D stays on the fast Tensor-Core kernel (no-op for contiguous models).
        from training.precision import match_input_to_model_memory_format
        trainX = match_input_to_model_memory_format(trainX, self.model)
        with torch.amp.autocast(device_type=device_type, enabled=self.enable_fp16):
            return self.model(trainX)

    def sample_logits(self, logits: torch.Tensor):
        return (
            sample_predictions(
                logits[:, 0], self.th_data, self.sampling_mode, self.temperature_data
            ),
            sample_predictions(
                logits[:, 1], self.th_data, self.sampling_mode, self.temperature_data
            ),
            sample_predictions(logits[:, 2], self.th_syn, self.sampling_mode, self.temperature_syn),
            sample_predictions(logits[:, 3], self.th_syn, self.sampling_mode, self.temperature_syn),
        )

    def reconstruct_syndromes(self, predictions, meas_flat: Optional[torch.Tensor] = None):
        z_data_corr, x_data_corr, syn_x_grid, syn_z_grid = predictions
        B, T, _, _ = z_data_corr.shape

        z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, self.n_rows * self.n_cols, T)
        z_data = z_flat[:, self.data_to_grid, :]
        x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, self.n_rows * self.n_cols, T)
        x_data = x_flat[:, self.data_to_grid, :]

        h_idx_e = self.H_idx.clamp_min(0).view(1, self.num_plaq, self.K, 1).expand(B, -1, -1, T)
        m_h = self.H_mask.view(1, self.num_plaq, self.K, 1)

        z_data_exp = z_data.unsqueeze(2).expand(B, self.num_data, self.K, T)
        g_z = z_data_exp.gather(1, h_idx_e)
        S_from_z = g_z.masked_fill(~m_h.expand_as(g_z), 0).sum(dim=2).remainder(2).to(torch.int32)

        x_data_exp = x_data.unsqueeze(2).expand(B, self.num_data, self.K, T)
        g_x = x_data_exp.gather(1, h_idx_e)
        S_from_x = g_x.masked_fill(~m_h.expand_as(g_x), 0).sum(dim=2).remainder(2).to(torch.int32)

        syn_x_flat = map_grid_to_stab(syn_x_grid, self.stab_to_grid).to(torch.int32)
        syn_z_flat = map_grid_to_stab(syn_z_grid, self.stab_to_grid).to(torch.int32)

        if (
            self.enable_delta_s2_correction and self.enable_z_ff and
            self.ff_mask_tensor is not None and self.cx_controls and meas_flat is not None
        ):
            delta_s2 = _compute_delta_s2_from_meas_flat(
                meas_flat=meas_flat,
                T=T,
                num_plaq=self.num_plaq,
                ff_mask_tensor=self.ff_mask_tensor,
                cx_controls=list(self.cx_controls),
                cx_targets=list(self.cx_targets),
                z_check_offset=self.z_check_offset,
                num_total_qubits=self.num_total_qubits,
            )
            delta_s2 = _align_delta_s2_for_predecoder_mode(
                delta_s2, apply_feedforward_to_predecoder=True
            )
            syn_z_flat = (syn_z_flat + delta_s2).remainder(2)

        return {
            "z_data": z_data,
            "x_data": x_data,
            "syn_x_flat": syn_x_flat,
            "syn_z_flat": syn_z_flat,
            "S_from_z": S_from_z,
            "S_from_x": S_from_x,
        }

    def assemble_residual_and_logical(
        self,
        x_syn_diff: torch.Tensor,
        z_syn_diff: torch.Tensor,
        components: dict,
        boundary_dets_batch: torch.Tensor,
    ):
        B, _, T = x_syn_diff.shape
        syn_x_flat = components["syn_x_flat"]
        syn_z_flat = components["syn_z_flat"]
        S_from_z = components["S_from_z"]
        S_from_x = components["S_from_x"]

        R_X_first = (x_syn_diff[:, :, 0] + syn_x_flat[:, :, 0] + S_from_z[:, :, 0]).remainder(2)
        if T > 1:
            R_X_rest = (
                x_syn_diff[:, :, 1:] + syn_x_flat[:, :, 1:] + syn_x_flat[:, :, :-1] +
                S_from_z[:, :, 1:]
            ).remainder(2)
        else:
            R_X_rest = x_syn_diff[:, :, 1:]

        R_Z_first = (z_syn_diff[:, :, 0] + syn_z_flat[:, :, 0] + S_from_x[:, :, 0]).remainder(2)
        if T > 1:
            R_Z_rest = (
                z_syn_diff[:, :, 1:] + syn_z_flat[:, :, 1:] + syn_z_flat[:, :, :-1] +
                S_from_x[:, :, 1:]
            ).remainder(2)
        else:
            R_Z_rest = z_syn_diff[:, :, 1:]

        if self.basis == "X":
            initial_detectors = R_X_first
            if T > 1:
                R_Z_rest = torch.cat(
                    [R_Z_rest[:, :, :-1],
                     torch.zeros_like(R_Z_rest[:, :, -1:])], dim=2
                )
            data_corr = components["z_data"].to(torch.float32)
        else:
            initial_detectors = R_Z_first
            if T > 1:
                R_X_rest = torch.cat(
                    [R_X_rest[:, :, :-1],
                     torch.zeros_like(R_X_rest[:, :, -1:])], dim=2
                )
            data_corr = components["x_data"].to(torch.float32)

        R_cat_rest = torch.cat([R_X_rest, R_Z_rest], dim=1)
        rest_flat = R_cat_rest.permute(0, 2, 1).contiguous().view(B, -1)
        residual = torch.cat([initial_detectors, rest_flat], dim=1)
        residual = torch.cat([residual, boundary_dets_batch.to(dtype=residual.dtype)], dim=1)

        pre_L_t = torch.einsum("d,bdt->bt", self.obs_support,
                               data_corr).remainder(2).to(torch.int32)
        pre_L = pre_L_t.sum(dim=1).remainder(2).view(-1)

        if not self.use_physical_frame_observable and self.obs_ancilla_meas_indices:
            meas_per_round = 2 * self.num_plaq
            anc_corr_parity = torch.zeros(B, dtype=torch.long, device=x_syn_diff.device)
            for mi in self.obs_ancilla_meas_indices:
                if mi < 0:
                    continue
                t = int(mi // meas_per_round)
                if t < 0 or t >= T:
                    continue
                j = int(mi - t * meas_per_round)
                if j < 0:
                    continue
                if j < self.num_plaq:
                    anc_corr_parity = (anc_corr_parity + syn_z_flat[:, j, t].long()).remainder(2)
                else:
                    anc_corr_parity = (
                        anc_corr_parity + syn_x_flat[:, j - self.num_plaq, t].long()
                    ).remainder(2)
            pre_L = (pre_L + anc_corr_parity).remainder(2)

        return pre_L, residual

    def forward_parts(
        self,
        trainX: torch.Tensor,
        x_syn_diff: torch.Tensor,
        z_syn_diff: torch.Tensor,
        boundary_dets_batch: torch.Tensor,
        meas_flat: Optional[torch.Tensor] = None,
    ):
        logits = self.model_forward(trainX)
        predictions = self.sample_logits(logits)
        components = self.reconstruct_syndromes(predictions, meas_flat=meas_flat)
        return self.assemble_residual_and_logical(
            x_syn_diff,
            z_syn_diff,
            components,
            boundary_dets_batch,
        )

    def forward(
        self,
        trainX: torch.Tensor,
        x_syn_diff: torch.Tensor,
        z_syn_diff: torch.Tensor,
        boundary_dets_batch: torch.Tensor,
        meas_flat: Optional[torch.Tensor] = None,
    ):
        pre_L, residual = self.forward_parts(
            trainX,
            x_syn_diff,
            z_syn_diff,
            boundary_dets_batch,
            meas_flat=meas_flat,
        )
        return torch.cat([pre_L.view(pre_L.shape[0], 1), residual], dim=1).to(torch.float32)


def _timing_counter_from_values(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "shots": 0,
            "sum_us_per_round": 0.0,
            "sum_sq_us_per_round": 0.0,
            "avg_us_per_round": None,
            "variance_us_per_round_sq": None,
            "min_us_per_round": None,
            "max_us_per_round": None,
        }

    sum_value = float(arr.sum())
    sum_sq = float(np.square(arr).sum())
    shots = int(arr.size)
    variance = float(
        max((sum_sq - (sum_value * sum_value) / shots) / (shots - 1), 0.0)
    ) if shots > 1 else None
    return {
        "shots": shots,
        "sum_us_per_round": sum_value,
        "sum_sq_us_per_round": sum_sq,
        "avg_us_per_round": float(sum_value / shots),
        "variance_us_per_round_sq": variance,
        "min_us_per_round": float(arr.min()),
        "max_us_per_round": float(arr.max()),
    }


def _reduce_timing_counter(counter: dict, device) -> dict:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return counter

    shots = torch.tensor(counter["shots"], dtype=torch.long, device=device)
    sums = torch.tensor(
        [counter["sum_us_per_round"], counter["sum_sq_us_per_round"]],
        dtype=torch.float64,
        device=device,
    )
    min_value = float("inf") if counter["min_us_per_round"] is None else float(
        counter["min_us_per_round"]
    )
    max_value = float("-inf") if counter["max_us_per_round"] is None else float(
        counter["max_us_per_round"]
    )
    mins = torch.tensor(min_value, dtype=torch.float64, device=device)
    maxes = torch.tensor(max_value, dtype=torch.float64, device=device)

    torch.distributed.all_reduce(shots, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(mins, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maxes, op=torch.distributed.ReduceOp.MAX)

    shots_value = int(shots.item())
    if shots_value == 0:
        return _timing_counter_from_values([])
    return _timing_counter_from_values_from_stats(
        shots_value,
        float(sums[0].item()),
        float(sums[1].item()),
        float(mins.item()),
        float(maxes.item()),
    )


def _timing_counter_from_values_from_stats(
    shots: int, sum_value: float, sum_sq: float, min_value: float, max_value: float
) -> dict:
    variance = float(
        max((sum_sq - (sum_value * sum_value) / shots) / (shots - 1), 0.0)
    ) if shots > 1 else None
    return {
        "shots": int(shots),
        "sum_us_per_round": float(sum_value),
        "sum_sq_us_per_round": float(sum_sq),
        "avg_us_per_round": float(sum_value / shots) if shots > 0 else None,
        "variance_us_per_round_sq": variance,
        "min_us_per_round": float(min_value) if shots > 0 else None,
        "max_us_per_round": float(max_value) if shots > 0 else None,
    }


def _time_single_shot_latency_chromobius(
    decoder,
    baseline_packed_samples,
    residual_packed_samples,
    n_rounds: int,
    warmup_iterations: int = 50,
):
    """
    Time single-shot Chromobius decode latency using batch_size=1 packed inputs.
    """
    n_samples = min(len(baseline_packed_samples), len(residual_packed_samples))
    if n_samples == 0:
        return None

    warmup_n = min(warmup_iterations, n_samples)
    for i in range(warmup_n):
        _ = decoder.predict_obs_flips_from_dets_bit_packed(
            baseline_packed_samples[i % n_samples][np.newaxis, :]
        )
        _ = decoder.predict_obs_flips_from_dets_bit_packed(
            residual_packed_samples[i % n_samples][np.newaxis, :]
        )

    baseline_us_per_round = []
    predecoder_us_per_round = []
    for i in range(n_samples):
        t_start = time.perf_counter()
        _ = decoder.predict_obs_flips_from_dets_bit_packed(
            baseline_packed_samples[i][np.newaxis, :]
        )
        baseline_us_per_round.append((time.perf_counter() - t_start) / n_rounds * 1e6)

    for i in range(n_samples):
        t_start = time.perf_counter()
        _ = decoder.predict_obs_flips_from_dets_bit_packed(
            residual_packed_samples[i][np.newaxis, :]
        )
        predecoder_us_per_round.append((time.perf_counter() - t_start) / n_rounds * 1e6)

    original = _timing_counter_from_values(baseline_us_per_round)
    residual = _timing_counter_from_values(predecoder_us_per_round)

    result = {
        "samples_timed": int(n_samples),
        "warmup_iterations": int(warmup_n),
        "original_syndromes": original,
        "residual_syndromes": residual,
        "baseline_mean_us_per_round": original["avg_us_per_round"],
        "baseline_std_us_per_round":
            (
                float(np.sqrt(original["variance_us_per_round_sq"]))
                if original["variance_us_per_round_sq"] is not None else 0.0
            ),
        "baseline_min_us_per_round": original["min_us_per_round"],
        "baseline_max_us_per_round": original["max_us_per_round"],
        "predecoder_mean_us_per_round": residual["avg_us_per_round"],
        "predecoder_std_us_per_round":
            (
                float(np.sqrt(residual["variance_us_per_round_sq"]))
                if residual["variance_us_per_round_sq"] is not None else 0.0
            ),
        "predecoder_min_us_per_round": residual["min_us_per_round"],
        "predecoder_max_us_per_round": residual["max_us_per_round"],
    }
    if residual["avg_us_per_round"] is not None and residual["avg_us_per_round"] > 0:
        result["speedup"] = float(original["avg_us_per_round"] / residual["avg_us_per_round"])
    return result


def _reduce_single_shot_latency(single_shot_latency: Optional[dict], device) -> Optional[dict]:
    if single_shot_latency is None:
        original = _timing_counter_from_values([])
        residual = _timing_counter_from_values([])
        warmup_iterations = 0
    else:
        original = single_shot_latency["original_syndromes"]
        residual = single_shot_latency["residual_syndromes"]
        warmup_iterations = int(single_shot_latency.get("warmup_iterations", 0))

    original = _reduce_timing_counter(original, device)
    residual = _reduce_timing_counter(residual, device)
    samples_timed = min(original["shots"], residual["shots"])
    if samples_timed <= 0:
        return None

    result = {
        "samples_timed": int(samples_timed),
        "warmup_iterations": warmup_iterations,
        "original_syndromes": original,
        "residual_syndromes": residual,
        "baseline_mean_us_per_round": original["avg_us_per_round"],
        "baseline_std_us_per_round":
            (
                float(np.sqrt(original["variance_us_per_round_sq"]))
                if original["variance_us_per_round_sq"] is not None else 0.0
            ),
        "baseline_min_us_per_round": original["min_us_per_round"],
        "baseline_max_us_per_round": original["max_us_per_round"],
        "predecoder_mean_us_per_round": residual["avg_us_per_round"],
        "predecoder_std_us_per_round":
            (
                float(np.sqrt(residual["variance_us_per_round_sq"]))
                if residual["variance_us_per_round_sq"] is not None else 0.0
            ),
        "predecoder_min_us_per_round": residual["min_us_per_round"],
        "predecoder_max_us_per_round": residual["max_us_per_round"],
    }
    if residual["avg_us_per_round"] is not None and residual["avg_us_per_round"] > 0:
        result["speedup"] = float(original["avg_us_per_round"] / residual["avg_us_per_round"])
    return result


def _build_chromobius_timing_summary(
    detector_shape,
    packed_detector_shape,
    total_samples: int,
    num_batches_processed: int,
    n_rounds: int,
    baseline_decode_time: float,
    predecoder_decode_time: float,
    baseline_density_sum: float,
    residual_density_sum: float,
    floor_time_per_round: Optional[float],
    baseline_batch_us_per_round,
    predecoder_batch_us_per_round,
    single_shot_latency,
    inclusive_timing_totals=None,
):
    """
    Build a JSON-serializable summary of Chromobius runtime diagnostics.

    If ``inclusive_timing_totals`` is provided, the resulting ``inclusive_timing``
    block is a stage-attribution diagnostic rather than a substitute for
    production wall-clock latency. Diagnostics mode runs the baseline decode
    synchronously and brackets GPU stages with synchronization; production mode
    overlaps baseline decode with model forward using the decode executor.
    """
    total_rounds = int(total_samples * n_rounds)
    avg_baseline_density = (
        baseline_density_sum / num_batches_processed if num_batches_processed > 0 else 0.0
    )
    avg_residual_density = (
        residual_density_sum / num_batches_processed if num_batches_processed > 0 else 0.0
    )
    density_reduction_pct = (
        (avg_baseline_density - avg_residual_density) / avg_baseline_density *
        100.0 if avg_baseline_density > 0 else 0.0
    )

    baseline_time_per_round = (
        baseline_decode_time / total_rounds * 1e6 if total_rounds > 0 else float("nan")
    )
    predecoder_time_per_round = (
        predecoder_decode_time / total_rounds * 1e6 if total_rounds > 0 else float("nan")
    )
    floor_us = floor_time_per_round * 1e6 if floor_time_per_round is not None else 0.0
    baseline_above_floor = baseline_time_per_round - floor_us
    predecoder_above_floor = predecoder_time_per_round - floor_us

    summary = {
        "detector_array_shape":
            list(detector_shape) if detector_shape is not None else None,
        "packed_detector_array_shape":
            (list(packed_detector_shape) if packed_detector_shape is not None else None),
        "total_samples_decoded":
            int(total_samples),
        "num_batches":
            int(num_batches_processed),
        "n_rounds_per_sample":
            int(n_rounds),
        "total_rounds_decoded":
            int(total_rounds),
        "baseline_detector_density_mean":
            float(avg_baseline_density),
        "residual_detector_density_mean":
            float(avg_residual_density),
        "density_reduction_pct":
            float(density_reduction_pct),
        "floor_us_per_round":
            float(floor_us),
        "baseline_decode_time_total_ms":
            float(baseline_decode_time * 1000.0),
        "predecoder_decode_time_total_ms":
            float(predecoder_decode_time * 1000.0),
        "baseline_decode_time_us_per_round":
            float(baseline_time_per_round),
        "predecoder_decode_time_us_per_round":
            float(predecoder_time_per_round),
        "baseline_above_floor_us_per_round":
            float(baseline_above_floor),
        "predecoder_above_floor_us_per_round":
            float(predecoder_above_floor),
        "time_saved_pct":
            float((baseline_decode_time - predecoder_decode_time) / baseline_decode_time *
                  100.0) if baseline_decode_time > 0 else 0.0,
    }
    if predecoder_decode_time > 0:
        summary["total_speedup"] = float(baseline_decode_time / predecoder_decode_time)
    if baseline_above_floor > 0 and predecoder_above_floor > 0:
        summary["density_dependent_speedup"] = float(baseline_above_floor / predecoder_above_floor)

    if baseline_batch_us_per_round and predecoder_batch_us_per_round:
        baseline_arr = np.array(baseline_batch_us_per_round, dtype=np.float64)
        predecoder_arr = np.array(predecoder_batch_us_per_round, dtype=np.float64)
        summary["batch_time_variability"] = {
            "baseline":
                {
                    "min_us_per_round": float(baseline_arr.min()),
                    "max_us_per_round": float(baseline_arr.max()),
                    "std_us_per_round": float(baseline_arr.std()),
                    "range_us_per_round": float(baseline_arr.max() - baseline_arr.min()),
                },
            "predecoder":
                {
                    "min_us_per_round": float(predecoder_arr.min()),
                    "max_us_per_round": float(predecoder_arr.max()),
                    "std_us_per_round": float(predecoder_arr.std()),
                    "range_us_per_round": float(predecoder_arr.max() - predecoder_arr.min()),
                },
        }

    if single_shot_latency is not None:
        summary["single_shot_latency"] = single_shot_latency

    if inclusive_timing_totals:

        def per_round(seconds):
            return seconds / total_rounds * 1e6 if total_rounds > 0 else float("nan")

        input_preprocess = (
            inclusive_timing_totals.get("dataloader_batch_time", 0.0) +
            inclusive_timing_totals.get("batch_to_device_time", 0.0)
        )
        output_postprocess = (
            inclusive_timing_totals.get("prediction_sampling_time", 0.0) +
            inclusive_timing_totals.get("syndrome_reconstruction_time", 0.0) +
            inclusive_timing_totals.get("residual_assembly_time", 0.0) +
            inclusive_timing_totals.get("residual_pack_time", 0.0)
        )
        baseline_inclusive = (
            inclusive_timing_totals.get("baseline_pack_time", 0.0) + baseline_decode_time
        )
        predecoder_inclusive = (
            inclusive_timing_totals.get("model_forward_time", 0.0) + output_postprocess +
            predecoder_decode_time
        )
        diagnostics_total = input_preprocess + baseline_inclusive + predecoder_inclusive

        breakdown_us = {
            name: float(per_round(value)) for name, value in inclusive_timing_totals.items()
        }
        breakdown_ms = {
            name: float(value * 1000.0) for name, value in inclusive_timing_totals.items()
        }
        summary["inclusive_timing"] = {
            "breakdown_total_ms": breakdown_ms,
            "breakdown_us_per_round": breakdown_us,
            "measurement_note":
                (
                    "Diagnostics attribution only: measured with synchronous baseline decode "
                    "and per-stage GPU synchronization, unlike production where baseline "
                    "decode overlaps model forward."
                ),
            "dataloader_batch_time_note":
                (
                    "dataloader_batch_time includes the observed time for next(loader_iter); "
                    "with worker prefetching, batch 0 can include one-time worker startup "
                    "and dataset materialization."
                ),
            "input_preprocess_time_total_ms": float(input_preprocess * 1000.0),
            "input_preprocess_time_us_per_round": float(per_round(input_preprocess)),
            "output_postprocess_time_total_ms": float(output_postprocess * 1000.0),
            "output_postprocess_time_us_per_round": float(per_round(output_postprocess)),
            "baseline_inclusive_time_total_ms": float(baseline_inclusive * 1000.0),
            "baseline_inclusive_time_us_per_round": float(per_round(baseline_inclusive)),
            "predecoder_inclusive_time_total_ms": float(predecoder_inclusive * 1000.0),
            "predecoder_inclusive_time_us_per_round": float(per_round(predecoder_inclusive)),
            "diagnostics_total_time_total_ms": float(diagnostics_total * 1000.0),
            "diagnostics_total_time_us_per_round": float(per_round(diagnostics_total)),
        }
        if predecoder_inclusive > 0:
            summary["inclusive_timing"]["inclusive_total_speedup"] = float(
                baseline_inclusive / predecoder_inclusive
            )

    return summary


def _print_chromobius_timing_summary(summary: dict):
    """
    Emit a human-readable timing summary mirroring the surface-code diagnostics.
    """
    detector_shape = summary.get("detector_array_shape")
    print(f"\n[Chromobius Timing] Decoder Input Info:")
    print(
        f"  Detector array shape: {tuple(detector_shape) if detector_shape is not None else None} (batch_size, num_detectors)"
    )
    print(f"  Total samples decoded: {summary['total_samples_decoded']}")
    print(f"  Number of batches: {summary['num_batches']}")

    print(f"\n[Chromobius Timing] Detector Density:")
    print(
        f"  Baseline (no pre-decoder): {summary['baseline_detector_density_mean']:.6f} "
        f"({summary['baseline_detector_density_mean'] * 100:.4f}% non-zero)"
    )
    print(
        f"  After pre-decoder:         {summary['residual_detector_density_mean']:.6f} "
        f"({summary['residual_detector_density_mean'] * 100:.4f}% non-zero)"
    )
    print(f"  Density reduction:         {summary['density_reduction_pct']:.2f}%")

    print(
        f"\n[Chromobius Timing] Decode Time (ONLY decoder.predict_obs_flips_from_dets_bit_packed, "
        f"excludes GPU->CPU transfer and bit-packing):"
    )
    print(f"  n_rounds per sample: {summary['n_rounds_per_sample']}")
    print(f"  Total rounds decoded: {summary['total_rounds_decoded']:,}")
    print(
        f"  Floor (zero density):      {summary['floor_us_per_round']:.3f} us/round "
        f"(fixed overhead)"
    )
    print(
        f"  Baseline (no pre-decoder): {summary['baseline_decode_time_total_ms']:.2f} ms total, "
        f"{summary['baseline_decode_time_us_per_round']:.3f} us/round"
    )
    print(
        f"  After pre-decoder:         {summary['predecoder_decode_time_total_ms']:.2f} ms total, "
        f"{summary['predecoder_decode_time_us_per_round']:.3f} us/round"
    )

    print(f"\n[Chromobius Timing] Breakdown (time above floor = density-dependent decoder work):")
    print(
        f"  Baseline above floor:      {summary['baseline_above_floor_us_per_round']:.3f} us/round"
    )
    print(
        f"  Pre-decoder above floor:   {summary['predecoder_above_floor_us_per_round']:.3f} us/round"
    )
    if "density_dependent_speedup" in summary:
        print(f"  Density-dependent speedup: {summary['density_dependent_speedup']:.2f}x")
    if "total_speedup" in summary:
        print(
            f"\n  Total speedup:             {summary['total_speedup']:.4f}x "
            f"({summary['time_saved_pct']:.2f}% faster)"
        )

    inclusive = summary.get("inclusive_timing")
    if inclusive is not None:
        breakdown = inclusive["breakdown_us_per_round"]
        print(f"\n[Chromobius Timing] Inclusive Pipeline Timing:")
        print(
            "  Note: attribution diagnostic only; diagnostics mode serializes baseline "
            "decode and synchronizes timed GPU stages."
        )
        print(
            "  Note: DataLoader timing includes the observed next(loader_iter) cost, "
            "including first-batch worker/prefetch warmup."
        )
        print(
            f"  Input preprocess:          {inclusive['input_preprocess_time_us_per_round']:.3f} us/round "
            f"(DataLoader={breakdown.get('dataloader_batch_time', 0.0):.3f}, "
            f"to_device={breakdown.get('batch_to_device_time', 0.0):.3f})"
        )
        print(
            f"  Model forward:             {breakdown.get('model_forward_time', 0.0):.3f} us/round"
        )
        print(
            f"  Output postprocess:        {inclusive['output_postprocess_time_us_per_round']:.3f} us/round "
            f"(sample={breakdown.get('prediction_sampling_time', 0.0):.3f}, "
            f"syndrome={breakdown.get('syndrome_reconstruction_time', 0.0):.3f}, "
            f"residual={breakdown.get('residual_assembly_time', 0.0):.3f}, "
            f"pack={breakdown.get('residual_pack_time', 0.0):.3f})"
        )
        print(
            f"  Baseline inclusive:        {inclusive['baseline_inclusive_time_us_per_round']:.3f} us/round "
            f"(pack={breakdown.get('baseline_pack_time', 0.0):.3f} + decode)"
        )
        print(
            f"  Pre-decoder inclusive:     {inclusive['predecoder_inclusive_time_us_per_round']:.3f} us/round "
            f"(model + postprocess + decode)"
        )

    batch_variability = summary.get("batch_time_variability")
    if batch_variability is not None:
        baseline = batch_variability["baseline"]
        predecoder = batch_variability["predecoder"]
        print(f"\n[Chromobius Timing] Per-Batch Variability (us/round):")
        print(
            f"  Baseline:     min={baseline['min_us_per_round']:.3f}, "
            f"max={baseline['max_us_per_round']:.3f}, "
            f"std={baseline['std_us_per_round']:.3f}, "
            f"range={baseline['range_us_per_round']:.3f}"
        )
        print(
            f"  Pre-decoder:  min={predecoder['min_us_per_round']:.3f}, "
            f"max={predecoder['max_us_per_round']:.3f}, "
            f"std={predecoder['std_us_per_round']:.3f}, "
            f"range={predecoder['range_us_per_round']:.3f}"
        )

    single_shot = summary.get("single_shot_latency")
    if single_shot is not None:
        print(f"\n[Chromobius Timing] Single-Shot Latency (batch_size=1):")
        print(
            f"  Samples timed: {single_shot['samples_timed']} "
            f"(after {single_shot['warmup_iterations']} warm-up iterations)"
        )
        print(
            f"  Baseline:     {single_shot['baseline_mean_us_per_round']:.3f} ± "
            f"{single_shot['baseline_std_us_per_round']:.3f} us/round "
            f"(min={single_shot['baseline_min_us_per_round']:.3f}, "
            f"max={single_shot['baseline_max_us_per_round']:.3f})"
        )
        print(
            f"  Pre-decoder:  {single_shot['predecoder_mean_us_per_round']:.3f} ± "
            f"{single_shot['predecoder_std_us_per_round']:.3f} us/round "
            f"(min={single_shot['predecoder_min_us_per_round']:.3f}, "
            f"max={single_shot['predecoder_max_us_per_round']:.3f})"
        )
        if "speedup" in single_shot:
            print(f"  Speedup:      {single_shot['speedup']:.2f}x")


def count_logical_errors_color(
    model, device, dist, cfg, include_diagnostics: bool = False, log_summary: bool = True
):
    """
    Main entry point for color code LER computation.
    
    Supports 'X', 'Z', or 'both' measurement basis testing.
    Reports LER **per round** to match the convention in the color code
    literature (e.g. Gidney et al.).
    
    Args:
        model: Trained PyTorch model
        device: torch device
        dist: Distributed training context (has rank, world_size)
        cfg: Hydra config with test parameters
        
    Returns:
        dict with LER-per-round results per basis
    """
    result = {}
    n_rounds = int(getattr(cfg.test, 'n_rounds', cfg.n_rounds))

    if cfg.test.meas_basis_test.lower() in ("both", "mixed"):
        orig = cfg.test.meas_basis_test
        for basis in ["X", "Z"]:
            cfg.test.meas_basis_test = basis
            t0 = time.time()
            diagnostics = None
            if include_diagnostics:
                num_errors, num_shots, chromobius_errors, diagnostics = run_inference_and_decode_color(
                    model, device, dist, cfg, return_diagnostics=True, log_summary=log_summary
                )
            else:
                num_errors, num_shots, chromobius_errors = run_inference_and_decode_color(
                    model, device, dist, cfg, log_summary=log_summary
                )
            tf = time.time()
            result[basis] = _build_ler_per_round_dict(
                num_errors, num_shots, chromobius_errors, n_rounds
            )
            if diagnostics is not None:
                result[basis]["chromobius_timing"] = diagnostics
            if log_summary and dist.rank == 0:
                ler = result[basis]["logical_error_rate (mean)"]
                ler_se = result[basis]["logical_error_rate (stderr)"]
                chrom = result[basis]["chromobius_error_rate (mean)"]
                chrom_se = result[basis]["chromobius_error_rate (stderr)"]
                print(
                    f"[Color Code LER] Time taken for {basis}: {tf - t0:.3f}s  |  "
                    f"PD+Chromobius={ler:.4e} ± {ler_se:.1e} ({int(num_errors)}/{int(num_shots)})  |  "
                    f"Chromobius={chrom:.4e} ± {chrom_se:.1e} ({int(chromobius_errors)}/{int(num_shots)})"
                )
        cfg.test.meas_basis_test = orig
    else:
        t0 = time.time()
        diagnostics = None
        if include_diagnostics:
            num_errors, num_shots, chromobius_errors, diagnostics = run_inference_and_decode_color(
                model, device, dist, cfg, return_diagnostics=True, log_summary=log_summary
            )
        else:
            num_errors, num_shots, chromobius_errors = run_inference_and_decode_color(
                model, device, dist, cfg, log_summary=log_summary
            )
        tf = time.time()
        result[cfg.test.meas_basis_test
              ] = _build_ler_per_round_dict(num_errors, num_shots, chromobius_errors, n_rounds)
        if diagnostics is not None:
            result[cfg.test.meas_basis_test]["chromobius_timing"] = diagnostics
        if log_summary and dist.rank == 0:
            basis = cfg.test.meas_basis_test
            ler = result[basis]["logical_error_rate (mean)"]
            ler_se = result[basis]["logical_error_rate (stderr)"]
            chrom = result[basis]["chromobius_error_rate (mean)"]
            chrom_se = result[basis]["chromobius_error_rate (stderr)"]
            print(
                f"[Color Code LER] Time taken: {tf - t0:.3f}s  |  "
                f"PD+Chromobius={ler:.4e} ± {ler_se:.1e} ({int(num_errors)}/{int(num_shots)})  |  "
                f"Chromobius={chrom:.4e} ± {chrom_se:.1e} ({int(chromobius_errors)}/{int(num_shots)})"
            )

    return result


def _build_ler_per_round_dict(num_errors, num_shots, chromobius_errors, n_rounds):
    """Build result dict with LER per round for color code.
    
    LER per round = (num_errors / num_shots) / n_rounds
    """
    ler = float(num_errors / num_shots) / n_rounds
    chromobius_ler = float(chromobius_errors / num_shots) / n_rounds

    # Binomial standard deviation, then divide by n_rounds
    var = (num_errors - num_errors * num_errors / float(num_shots)) / num_shots
    stddev = np.sqrt(var)
    chromobius_var = (
        chromobius_errors - chromobius_errors * chromobius_errors / float(num_shots)
    ) / num_shots
    chromobius_stddev = np.sqrt(chromobius_var)

    return {
        "num_shots": int(num_shots),
        "n_rounds": int(n_rounds),
        "logical_errors": int(num_errors),
        "chromobius_errors": int(chromobius_errors),
        "logical_error_rate (mean)": ler,
        "logical_error_rate (stderr)": float(stddev / np.sqrt(num_shots)) / n_rounds,
        "chromobius_error_rate (mean)": chromobius_ler,
        "chromobius_error_rate (stderr)": float(chromobius_stddev / np.sqrt(num_shots)) / n_rounds,
    }


@torch.inference_mode()
def run_inference_and_decode_color(
    model, device, dist, cfg, return_diagnostics: bool = False, log_summary: bool = True
):
    """
    Run inference with trained model and decode with Chromobius.
    
    Pipeline:
    1. Generate samples using Stim-based color code circuit
    2. Run model forward pass to get predictions
    3. Compute residual syndromes from predictions
    4. Decode residuals with Chromobius
    5. Combine with pre-decoder logical frame to get final prediction
    6. Count logical errors
    
    Returns:
        (num_logic_errors_after_predecoder, num_samples, num_chromobius_errors_baseline)
        When return_diagnostics=True, appends a JSON-serializable Chromobius
        timing/density summary as a fourth element.
    """
    # Config extraction
    _data_cfg = getattr(cfg, "data", None)
    _enable_z_ff = bool(
        getattr(_data_cfg, "enable_z_feedforward", True)
    ) if _data_cfg is not None else True
    _enable_delta_s2_correction = bool(getattr(cfg.test, "enable_delta_s2_correction", False))
    _use_physical_frame_observable = _enable_z_ff

    collect_diagnostics = bool(return_diagnostics)

    # Distributed setup
    total_samples = int(cfg.test.num_samples)
    samples_per_gpu = total_samples // dist.world_size
    n_rounds_used = int(getattr(cfg.test, "n_rounds", cfg.n_rounds))

    if log_summary and dist.rank == 0:
        if dist.world_size > 1:
            print(f"[Color Code LER] Distributed: {dist.world_size} GPUs")
            print(f"[Color Code LER] Total samples: {total_samples}, per GPU: {samples_per_gpu}")
        print(f"[Color Code LER] Basis: {cfg.test.meas_basis_test}, p_error: {cfg.test.p_error}")
        (
            noise_model_obj,
            test_nm_mode,
            noise_model_family,
            noise_instruction_semantics,
            gidney_style_noise,
        ) = _resolve_color_noise_settings(cfg)
        print(
            f"[Color Code LER] noise_model_family: {noise_model_family}, "
            f"noise_instruction_semantics: {noise_instruction_semantics}, "
            f"test.noise_model: {test_nm_mode}, "
            f"gidney_style_noise: {gidney_style_noise}"
        )
        if noise_model_obj is not None:
            print(f"[Color Code LER] Using explicit noise_model: {noise_model_obj!r}")
        if _enable_z_ff:
            print("[Color Code LER] Using physical-frame observable (data-only parity)")
            if _enable_delta_s2_correction:
                print("[Color Code LER] delta_s2 conversion: enabled")
            else:
                print("[Color Code LER] delta_s2 conversion: disabled (default)")
    else:
        (
            noise_model_obj,
            test_nm_mode,
            noise_model_family,
            noise_instruction_semantics,
            gidney_style_noise,
        ) = _resolve_color_noise_settings(cfg)

    model.eval()

    # Save and set rank-specific random state
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        rank_seed = 12345 + dist.rank * 1000
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        np.random.seed(rank_seed)
        random.seed(rank_seed)

        # Create datapipe
        cfg_copy = deepcopy(cfg)
        cfg_copy.test.num_samples = samples_per_gpu

        test_distance = int(getattr(cfg.test, 'distance', cfg.distance))
        test_n_rounds = int(getattr(cfg.test, 'n_rounds', cfg.n_rounds))
        test_dataset = QCDataPipePreDecoder_ColorCode_inference(
            distance=test_distance,
            n_rounds=test_n_rounds,
            num_samples=samples_per_gpu,
            error_mode="circuit_level_color_code",
            p_error=cfg.test.p_error,
            measure_basis=cfg.test.meas_basis_test,
            noise_model=noise_model_obj,
            gidney_style_noise=gidney_style_noise,
            noise_model_family=noise_model_family,
            noise_instruction_semantics=noise_instruction_semantics,
            schedule=str(getattr(getattr(cfg, "data", None), "schedule", "nearest-neighbor")),
        )
    finally:
        # Restore random states
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

    # Get circuit and build Chromobius decoder
    basis = str(cfg.test.meas_basis_test).upper()
    if basis not in ("X", "Z"):
        raise ValueError(f"Invalid meas_basis_test='{basis}'. Use 'X' or 'Z'.")

    if hasattr(test_dataset, 'circ'):
        circ_obj = test_dataset.circ
        circuit = circ_obj.stim_circuit
    elif hasattr(test_dataset, 'circ_X') and basis == 'X':
        circ_obj = test_dataset.circ_X
        circuit = circ_obj.stim_circuit
    elif hasattr(test_dataset, 'circ_Z') and basis == 'Z':
        circ_obj = test_dataset.circ_Z
        circuit = circ_obj.stim_circuit
    else:
        raise RuntimeError("Cannot find circuit in datapipe")

    # Extract the OBSERVABLE_INCLUDE definition from the inlined circuit.
    # with_inlined_feedback() may modify the observable vs the raw lx/lz logical
    # operators (e.g., X-basis uses ALL data qubits, Z-basis adds ancilla refs).
    # We split references into data-qubit indices (final MX/MZ block) and
    # ancilla-measurement indices (within the per-round ancilla stream).
    #
    # When the model is trained with feedforward (_use_physical_frame_observable),
    # its data corrections are in the predecoder frame.  The physical observable
    # identity (data-only parity == data+ancilla parity) means we only need the
    # data-qubit support and can ignore ancilla refs.  The ancilla list is still
    # extracted for the non-feedforward fallback path.
    D = test_distance
    _code_tmp = ColorCode(D)
    _num_data_tmp = _code_tmp.num_data
    _total_meas = int(circuit.num_measurements)
    _data_meas_start = _total_meas - _num_data_tmp
    _total_ancilla_meas = _data_meas_start  # all measurements before final data block are ancilla
    _obs_data_qubit_indices: list = []
    _obs_ancilla_meas_indices: list = []
    _obs_m_count = 0
    for _name, _targets, _arg in circuit.flattened_operations():
        if _name in ("M", "MR", "MX", "MY", "MZ", "MRX", "MRY", "MRZ"):
            _obs_m_count += sum(isinstance(_t, int) for _t in _targets)
            continue
        if _name == "OBSERVABLE_INCLUDE":
            for _t in _targets:
                if isinstance(_t, tuple) and _t[0] == "rec":
                    _abs = _obs_m_count + int(_t[1])
                    if _abs >= _data_meas_start:
                        _obs_data_qubit_indices.append(_abs - _data_meas_start)
                    elif 0 <= _abs < _total_ancilla_meas:
                        _obs_ancilla_meas_indices.append(_abs)
    # Build a (num_data,) binary vector for the inlined observable data-qubit support.
    obs_support = torch.zeros(_num_data_tmp, dtype=torch.float32, device=device)
    for _qi in _obs_data_qubit_indices:
        obs_support[_qi] = 1.0

    # Build Chromobius decoder from DEM
    # decompose_errors=False: Chromobius natively handles hyperedges, no decomposition needed
    # approximate_disjoint_errors=True: Required for PAULI_CHANNEL_2 (multi-param noise model with overlapping errors)
    # ignore_decomposition_failures=True: Ignore high weight error decomposition issues
    det_model = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
        ignore_decomposition_failures=True,
    )
    decoder = chromobius.compile_decoder_for_dem(det_model)

    # Get baseline detector data for Chromobius-only baseline
    if hasattr(test_dataset, 'dets_and_obs'):
        stim_dets_and_obs = test_dataset.dets_and_obs.numpy()
    elif hasattr(test_dataset, 'dets_and_obs_X') and basis == 'X':
        stim_dets_and_obs = test_dataset.dets_and_obs_X.numpy()
    elif hasattr(test_dataset, 'dets_and_obs_Z') and basis == 'Z':
        stim_dets_and_obs = test_dataset.dets_and_obs_Z.numpy()

    num_obs = circuit.num_observables
    assert num_obs == 1, f"Expected 1 observable, got {num_obs}"

    stim_dets = stim_dets_and_obs[:, :-num_obs].astype(np.uint8)
    stim_obs = stim_dets_and_obs[:, -num_obs:].astype(np.uint8)
    stim_dets_gpu = torch.from_numpy(stim_dets).to(device)  # pre-loaded for GPU slicing/packing

    # Number of boundary detectors (added by add_boundary_detectors=True in MemoryCircuit)
    # For color code: one boundary detector per plaquette (for the measurement basis stabilizers)
    # These are the LAST detectors in the circuit, comparing final data measurements to last ancilla
    num_boundary_dets = (3 * (D * D - 1)) // 8  # num_plaquettes for color code

    # Build parity maps
    maps = _build_color_code_parity_maps(D)
    num_plaq = maps["num_plaq"]
    num_data = maps["num_data"]
    # --- Optional feedforward cascade correction ---
    # Disabled by default to preserve no-op invariance:
    # with zero model corrections, residual decoding must match Chromobius baseline.
    _ff_mask_tensor = None  # (num_plaq, num_data) float on device
    _cx_controls = None  # list of 1-D int tensors, one per CX layer
    _cx_targets = None  # matching targets
    _z_check_offset = None  # start of Z-check qubits in the full qubit vector
    _num_total_qubits = None
    if _enable_z_ff and _enable_delta_s2_correction:
        (
            _ff_mask_tensor,
            _cx_controls,
            _cx_targets,
            _z_check_offset,
            _num_total_qubits,
        ) = _build_ff_cascade_tensors(circ_obj, num_plaq, num_data, device)
        if log_summary and dist.rank == 0 and _cx_controls:
            _total_cx = sum(len(c) for c in _cx_controls)
            print(
                f"[Color Code LER] FF cascade correction: {len(_cx_controls)} CX layers, {_total_cx} gates"
            )

    eval_module = PreDecoderColorEvalModule(
        model,
        cfg,
        maps,
        basis=basis,
        obs_support=obs_support,
        num_boundary_dets=num_boundary_dets,
        enable_delta_s2_correction=_enable_delta_s2_correction,
        enable_z_ff=_enable_z_ff,
        ff_mask_tensor=_ff_mask_tensor,
        cx_controls=_cx_controls,
        cx_targets=_cx_targets,
        z_check_offset=_z_check_offset,
        num_total_qubits=_num_total_qubits,
        use_physical_frame_observable=_use_physical_frame_observable,
        obs_ancilla_meas_indices=_obs_ancilla_meas_indices,
    ).to(device)
    eval_module.eval()

    # DataLoader
    test_loader_kwargs = dict(cfg.test.dataloader)
    if test_loader_kwargs.get('num_workers', 0) == 0:
        test_loader_kwargs.pop('prefetch_factor', None)
        if test_loader_kwargs.get('persistent_workers', False):
            test_loader_kwargs['persistent_workers'] = False

    test_dataloader = DataLoader(test_dataset, shuffle=False, **test_loader_kwargs)

    # Counters
    logical_errors = 0
    total_samples_processed = 0
    num_chromobius_baseline_errors = 0
    baseline_sample_offset = 0
    baseline_decode_time = 0.0
    predecoder_decode_time = 0.0
    baseline_detector_density_sum = 0.0
    residual_detector_density_sum = 0.0
    num_batches_processed = 0
    detector_shape = None
    packed_detector_shape = None
    floor_time_per_round = None
    baseline_batch_us_per_round = []
    predecoder_batch_us_per_round = []
    stored_baseline_packed = []
    stored_residual_packed = []
    singleshot_storage_size = 4096
    inclusive_timing_keys = [
        "dataloader_batch_time",
        "batch_to_device_time",
        "baseline_pack_time",
        "model_forward_time",
        "prediction_sampling_time",
        "syndrome_reconstruction_time",
        "residual_assembly_time",
        "residual_pack_time",
    ]
    # dataloader_batch_time intentionally records observed next(loader_iter)
    # latency. With worker prefetching, batch 0 can include one-time worker
    # startup and dataset materialization; keep it visible rather than silently
    # skipping it from the attribution totals.
    inclusive_timing_totals = {name: 0.0 for name in inclusive_timing_keys}

    # The iterator owns the executor context so worker cleanup still happens if
    # an exception exits the batch loop.
    for batch_idx, batch, _decode_executor, batch_load_time in _iter_batches_with_decode_executor(
        test_dataloader, time_batches=collect_diagnostics
    ):
        if collect_diagnostics:
            inclusive_timing_totals["dataloader_batch_time"] += batch_load_time
            t_batch_to_device = _start_timing(device)
        batch = dict_to_device(batch, device)
        if collect_diagnostics:
            inclusive_timing_totals["batch_to_device_time"] += _elapsed_since(
                t_batch_to_device, device
            )

        # Inputs
        x_syn_diff = batch["x_syn_diff"].to(torch.int32)  # (B, num_plaq, T)
        z_syn_diff = batch["z_syn_diff"].to(torch.int32)  # (B, num_plaq, T)
        trainX = batch["trainX"]
        dets_and_obs = batch["dets_and_obs"]

        B = x_syn_diff.shape[0]
        x_syn_diff.shape[2]

        # Baseline Chromobius decode (without pre-decoder)
        baseline_dets_batch = stim_dets_gpu[baseline_sample_offset:baseline_sample_offset + B]
        baseline_obs_batch = stim_obs[baseline_sample_offset:baseline_sample_offset + B]

        if collect_diagnostics:
            t_baseline_pack = _start_timing(device)
        baseline_packed = _packbits_gpu(baseline_dets_batch).cpu().numpy()
        if collect_diagnostics:
            inclusive_timing_totals["baseline_pack_time"] += _elapsed_since(t_baseline_pack, device)
        if collect_diagnostics:
            if detector_shape is None:
                detector_shape = (B, stim_dets.shape[1])
                packed_detector_shape = baseline_packed.shape
                zero_detectors = np.zeros_like(baseline_packed)
                _ = decoder.predict_obs_flips_from_dets_bit_packed(zero_detectors)
                t_floor_start = time.perf_counter()
                for _ in range(10):
                    _ = decoder.predict_obs_flips_from_dets_bit_packed(zero_detectors)
                t_floor_end = time.perf_counter()
                floor_time_per_round = (
                    (t_floor_end - t_floor_start) / 10.0 / baseline_packed.shape[0] / n_rounds_used
                )

            t_baseline_decode_start = time.perf_counter()
            baseline_pred = decoder.predict_obs_flips_from_dets_bit_packed(baseline_packed)
            t_baseline_decode_end = time.perf_counter()
            batch_baseline_time = t_baseline_decode_end - t_baseline_decode_start
            baseline_decode_time += batch_baseline_time
            baseline_batch_us_per_round.append(batch_baseline_time / (B * n_rounds_used) * 1e6)
            baseline_detector_density_sum += baseline_dets_batch.float().mean().item()
            if len(stored_baseline_packed) < singleshot_storage_size:
                n_store = min(
                    singleshot_storage_size - len(stored_baseline_packed), baseline_packed.shape[0]
                )
                stored_baseline_packed.extend(
                    [np.ascontiguousarray(baseline_packed[i].copy()) for i in range(n_store)]
                )
            baseline_pred_unpacked = np.unpackbits(baseline_pred, axis=1,
                                                   bitorder='little')[:, :num_obs]
            num_chromobius_baseline_errors += int(
                (baseline_pred_unpacked != baseline_obs_batch).sum()
            )
        else:
            # Submit to background thread; overlaps with model forward below.
            _baseline_fut = _decode_executor.submit(
                decoder.predict_obs_flips_from_dets_bit_packed, baseline_packed
            )
        baseline_sample_offset += B

        boundary_dets_batch = baseline_dets_batch[:, -num_boundary_dets:]
        meas_flat = (
            batch["meas_flat"].to(device) if _enable_delta_s2_correction and _enable_z_ff and
            _ff_mask_tensor is not None and _cx_controls else None
        )

        # Model forward pass
        if collect_diagnostics:
            t_model_forward = _start_timing(device)
        logits = eval_module.model_forward(trainX)
        if collect_diagnostics:
            inclusive_timing_totals["model_forward_time"] += _elapsed_since(t_model_forward, device)

        # Model predictions: [z_data_corr, x_data_corr, syn_x_corr, syn_z_corr]
        # Note: In color code, both X and Z errors affect SAME stabilizers
        if collect_diagnostics:
            t_prediction_sampling = _start_timing(device)
        predictions = eval_module.sample_logits(logits)
        if collect_diagnostics:
            inclusive_timing_totals["prediction_sampling_time"] += _elapsed_since(
                t_prediction_sampling, device
            )

        # Compute syndrome induced by data corrections
        # For color code: SAME stabilizers detect both X and Z errors
        # Z errors -> syndrome from Hz (= H for color code)
        # X errors -> syndrome from Hx (= H for color code)

        if collect_diagnostics:
            t_syndrome_reconstruction = _start_timing(device)
        components = eval_module.reconstruct_syndromes(predictions, meas_flat=meas_flat)
        if collect_diagnostics:
            inclusive_timing_totals["syndrome_reconstruction_time"] += _elapsed_since(
                t_syndrome_reconstruction, device
            )

        # Compute residuals
        # For color code: both X and Z syndromes come from same stabilizers
        # R_X = x_syn_diff XOR syn_x_pred XOR syn_x_prev XOR S_from_z
        # R_Z = z_syn_diff XOR syn_z_pred XOR syn_z_prev XOR S_from_x

        if collect_diagnostics:
            t_residual_assembly = _start_timing(device)
        pre_L, residual = eval_module.assemble_residual_and_logical(
            x_syn_diff,
            z_syn_diff,
            components,
            boundary_dets_batch,
        )

        # Sanity check
        if residual.shape[1] != det_model.num_detectors:
            raise ValueError(
                f"Residual shape {residual.shape} != DEM detectors {det_model.num_detectors}. "
                f"Check detector ordering for color code."
            )
        if collect_diagnostics:
            inclusive_timing_totals["residual_assembly_time"] += _elapsed_since(
                t_residual_assembly, device
            )

        # Decode with Chromobius — pack bits on GPU, transfer the 8x-smaller packed tensor
        if collect_diagnostics:
            t_residual_pack = _start_timing(device)
        residual_packed = _packbits_gpu(residual).cpu().numpy()
        if collect_diagnostics:
            inclusive_timing_totals["residual_pack_time"] += _elapsed_since(t_residual_pack, device)

        # Production mode: baseline decode ran in background; collect result now.
        # The model forward + post-processing above served as the overlap window.
        if not collect_diagnostics:
            baseline_pred_unpacked = np.unpackbits(
                _baseline_fut.result(), axis=1, bitorder="little"
            )[:, :num_obs]
            num_chromobius_baseline_errors += int(
                (baseline_pred_unpacked != baseline_obs_batch).sum()
            )

        if collect_diagnostics:
            t_predecoder_decode_start = time.perf_counter()
            pred_obs = decoder.predict_obs_flips_from_dets_bit_packed(residual_packed)
            t_predecoder_decode_end = time.perf_counter()
            batch_predecoder_time = t_predecoder_decode_end - t_predecoder_decode_start
            predecoder_decode_time += batch_predecoder_time
            predecoder_batch_us_per_round.append(batch_predecoder_time / (B * n_rounds_used) * 1e6)
            residual_detector_density_sum += residual.float().mean().item()
            num_batches_processed += 1
            if len(stored_residual_packed) < singleshot_storage_size:
                n_store = min(
                    singleshot_storage_size - len(stored_residual_packed), residual_packed.shape[0]
                )
                stored_residual_packed.extend(
                    [np.ascontiguousarray(residual_packed[i].copy()) for i in range(n_store)]
                )
        else:
            pred_obs = decoder.predict_obs_flips_from_dets_bit_packed(residual_packed)
        pred_obs_unpacked = np.unpackbits(pred_obs, axis=1, bitorder='little')[:, :num_obs]

        pred_obs_tensor = torch.as_tensor(pred_obs_unpacked, dtype=torch.long,
                                          device=device).view(-1)
        final_L = (pre_L + pred_obs_tensor).remainder_(2)

        # Ground truth
        gt_obs = dets_and_obs[:, -num_obs:].view(-1)

        logical_errors += int((final_L != gt_obs).sum().item())
        total_samples_processed += B

    # All-reduce across ranks
    if dist.world_size > 1:
        t_log = torch.tensor(logical_errors, device=device, dtype=torch.long)
        t_n = torch.tensor(total_samples_processed, device=device, dtype=torch.long)
        t_chromobius = torch.tensor(num_chromobius_baseline_errors, device=device, dtype=torch.long)
        torch.distributed.all_reduce(t_log, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_n, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_chromobius, op=torch.distributed.ReduceOp.SUM)
        logical_errors = int(t_log.item())
        total_samples_processed = int(t_n.item())
        num_chromobius_baseline_errors = int(t_chromobius.item())

    if collect_diagnostics:
        td_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        if dist.world_size > 1 and td_ready:
            t_base_time = torch.tensor(baseline_decode_time, device=device, dtype=torch.float64)
            t_pred_time = torch.tensor(predecoder_decode_time, device=device, dtype=torch.float64)
            t_base_density = torch.tensor(
                baseline_detector_density_sum, device=device, dtype=torch.float64
            )
            t_pred_density = torch.tensor(
                residual_detector_density_sum, device=device, dtype=torch.float64
            )
            t_batches = torch.tensor(num_batches_processed, device=device, dtype=torch.long)
            t_floor = torch.tensor(
                floor_time_per_round if floor_time_per_round is not None else 0.0,
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(t_base_time, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(t_pred_time, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(t_base_density, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(t_pred_density, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(t_batches, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(t_floor, op=torch.distributed.ReduceOp.SUM)
            t_inclusive = torch.tensor(
                [inclusive_timing_totals[name] for name in inclusive_timing_keys],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(t_inclusive, op=torch.distributed.ReduceOp.SUM)
            baseline_decode_time = float(t_base_time.item())
            predecoder_decode_time = float(t_pred_time.item())
            baseline_detector_density_sum = float(t_base_density.item())
            residual_detector_density_sum = float(t_pred_density.item())
            num_batches_processed = int(t_batches.item())
            floor_time_per_round = float(t_floor.item()) / dist.world_size
            inclusive_timing_totals = {
                name: float(t_inclusive[i].item()) for i, name in enumerate(inclusive_timing_keys)
            }

            gathered_baseline_batch = [None for _ in range(dist.world_size)]
            gathered_predecoder_batch = [None for _ in range(dist.world_size)]
            torch.distributed.all_gather_object(
                gathered_baseline_batch, baseline_batch_us_per_round
            )
            torch.distributed.all_gather_object(
                gathered_predecoder_batch, predecoder_batch_us_per_round
            )
            baseline_batch_us_per_round = [
                float(value) for values in gathered_baseline_batch if values is not None
                for value in values
            ]
            predecoder_batch_us_per_round = [
                float(value) for values in gathered_predecoder_batch if values is not None
                for value in values
            ]

        single_shot_latency = None
        if stored_baseline_packed and stored_residual_packed:
            single_shot_latency = _time_single_shot_latency_chromobius(
                decoder=decoder,
                baseline_packed_samples=stored_baseline_packed,
                residual_packed_samples=stored_residual_packed,
                n_rounds=n_rounds_used,
            )
        if dist.world_size > 1 and td_ready:
            single_shot_latency = _reduce_single_shot_latency(single_shot_latency, device)

        diagnostics = _build_chromobius_timing_summary(
            detector_shape=detector_shape,
            packed_detector_shape=packed_detector_shape,
            total_samples=total_samples_processed,
            num_batches_processed=num_batches_processed,
            n_rounds=n_rounds_used,
            baseline_decode_time=baseline_decode_time,
            predecoder_decode_time=predecoder_decode_time,
            baseline_density_sum=baseline_detector_density_sum,
            residual_density_sum=residual_detector_density_sum,
            floor_time_per_round=floor_time_per_round,
            baseline_batch_us_per_round=baseline_batch_us_per_round,
            predecoder_batch_us_per_round=predecoder_batch_us_per_round,
            single_shot_latency=single_shot_latency,
            inclusive_timing_totals=inclusive_timing_totals,
        )
        if log_summary and dist.rank == 0:
            _print_chromobius_timing_summary(diagnostics)
        return (
            logical_errors,
            total_samples_processed,
            num_chromobius_baseline_errors,
            diagnostics,
        )

    return logical_errors, total_samples_processed, num_chromobius_baseline_errors


@torch.inference_mode()
def compute_syndrome_density_reduction_color(model, device, dist, cfg) -> dict:
    """
    Compute syndrome density reduction for color code.
    
    Returns input and residual syndrome densities to measure pre-decoder effectiveness.
    Handles 'both'/'mixed' basis by running X and Z separately (matching LER function).
    """

    basis = str(cfg.test.meas_basis_test).upper()

    # Handle 'both' by running X and Z separately and averaging
    if basis in ("BOTH", "MIXED"):
        orig = cfg.test.meas_basis_test
        results = {}
        for b in ("X", "Z"):
            cfg.test.meas_basis_test = b
            results[b] = compute_syndrome_density_reduction_color(model, device, dist, cfg)
        cfg.test.meas_basis_test = orig

        input_ones = int(results["X"]["input_syndrome_ones"]
                        ) + int(results["Z"]["input_syndrome_ones"])
        residual_ones = int(results["X"]["residual_syndrome_ones"]
                           ) + int(results["Z"]["residual_syndrome_ones"])
        syndrome_elements = int(results["X"]["syndrome_elements"]
                               ) + int(results["Z"]["syndrome_elements"])
        input_density = input_ones / syndrome_elements if syndrome_elements > 0 else float("nan")
        residual_density = residual_ones / syndrome_elements if syndrome_elements > 0 else float(
            "nan"
        )
        reduction = input_density / residual_density if residual_density > 0 else float("inf")

        # Keep per-basis reductions for callers comparing X and Z behavior.
        rx = results["X"].get("reduction_factor", float('nan'))
        rz = results["Z"].get("reduction_factor", float('nan'))
        import math
        if math.isfinite(rx) and math.isfinite(rz):
            avg_reduction = (rx + rz) / 2.0
        elif math.isfinite(rx):
            avg_reduction = rx
        elif math.isfinite(rz):
            avg_reduction = rz
        else:
            avg_reduction = float('nan')

        return {
            "input_syndrome_ones": input_ones,
            "residual_syndrome_ones": residual_ones,
            "syndrome_elements": syndrome_elements,
            "input_syndrome_density": input_density,
            "residual_syndrome_density": residual_density,
            "reduction_factor": reduction,
            "reduction_X": rx,
            "reduction_Z": rz,
            "mean_per_basis_reduction_factor": avg_reduction,
            "basis": "BOTH",
        }

    th_data = float(getattr(cfg.test, "th_data", 0.0))
    th_syn = float(getattr(cfg.test, "th_syn", 0.0))
    sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
    temperature = float(getattr(cfg.test, "temperature", 1.0))
    temperature_data = getattr(cfg.test, "temperature_data", None)
    temperature_syn = getattr(cfg.test, "temperature_syn", None)
    temperature_data = float(temperature_data) if temperature_data is not None else temperature
    temperature_syn = float(temperature_syn) if temperature_syn is not None else temperature
    (
        noise_model_obj,
        _test_nm_mode,
        noise_model_family,
        noise_instruction_semantics,
        gidney_style_noise,
    ) = _resolve_color_noise_settings(cfg)
    _data_cfg = getattr(cfg, "data", None)
    _enable_z_ff = bool(
        getattr(_data_cfg, "enable_z_feedforward", True)
    ) if _data_cfg is not None else True
    _enable_delta_s2_correction = bool(getattr(cfg.test, "enable_delta_s2_correction", False))

    total_samples = int(cfg.test.num_samples)
    samples_per_gpu = total_samples // dist.world_size

    # Create datapipe (single basis)
    test_distance = int(getattr(cfg.test, 'distance', cfg.distance))
    test_n_rounds = int(getattr(cfg.test, 'n_rounds', cfg.n_rounds))
    test_dataset = QCDataPipePreDecoder_ColorCode_inference(
        distance=test_distance,
        n_rounds=test_n_rounds,
        num_samples=samples_per_gpu,
        error_mode="circuit_level_color_code",
        p_error=cfg.test.p_error,
        measure_basis=basis,
        noise_model=noise_model_obj,
        gidney_style_noise=gidney_style_noise,
        noise_model_family=noise_model_family,
        noise_instruction_semantics=noise_instruction_semantics,
        schedule=str(getattr(getattr(cfg, "data", None), "schedule", "nearest-neighbor")),
    )

    D = test_distance
    maps = _build_color_code_parity_maps(D)
    H_idx = maps["H_idx"].to(device)
    H_mask = maps["H_mask"].to(device)
    stab_to_grid = maps["stab_to_grid"].to(device)
    data_to_grid = maps["data_to_grid"].to(device)
    num_plaq = maps["num_plaq"]
    num_data = maps["num_data"]
    n_rows = maps["n_rows"]
    n_cols = maps["n_cols"]
    K = maps["K"]
    _ff_mask_tensor = None
    _cx_controls = None
    _cx_targets = None
    _z_check_offset = None
    _num_total_qubits = None
    if _enable_delta_s2_correction and _enable_z_ff and hasattr(test_dataset, "circ"):
        (
            _ff_mask_tensor,
            _cx_controls,
            _cx_targets,
            _z_check_offset,
            _num_total_qubits,
        ) = _build_ff_cascade_tensors(test_dataset.circ, num_plaq, num_data, device)

    test_loader_kwargs = dict(cfg.test.dataloader)
    if test_loader_kwargs.get('num_workers', 0) == 0:
        test_loader_kwargs.pop('prefetch_factor', None)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **test_loader_kwargs)

    model.eval()

    # Accumulators
    in_ones = torch.tensor(0, dtype=torch.int64, device=device)
    in_elems = torch.tensor(0, dtype=torch.int64, device=device)
    res_ones = torch.tensor(0, dtype=torch.int64, device=device)

    for batch in test_dataloader:
        batch = dict_to_device(batch, device)

        x_syn_diff = batch["x_syn_diff"].to(torch.int32)
        z_syn_diff = batch["z_syn_diff"].to(torch.int32)
        trainX = batch["trainX"]

        B, num_plaq_b, T = x_syn_diff.shape

        # Count input syndromes (basis-matched)
        if basis == "X":
            in_ones += x_syn_diff.sum(dtype=torch.int64)
            in_elems += torch.tensor(x_syn_diff.numel(), device=device, dtype=torch.int64)
        else:
            in_ones += z_syn_diff.sum(dtype=torch.int64)
            in_elems += torch.tensor(z_syn_diff.numel(), device=device, dtype=torch.int64)

        # Model forward
        # Match the eval input layout to a channels_last_3d model so half-precision
        # Conv3D stays on the fast Tensor-Core kernel (no-op for contiguous models).
        from training.precision import match_input_to_model_memory_format
        trainX = match_input_to_model_memory_format(trainX, model)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.enable_fp16):
            logits = model(trainX)

        z_data_corr = sample_predictions(logits[:, 0], th_data, sampling_mode, temperature_data)
        x_data_corr = sample_predictions(logits[:, 1], th_data, sampling_mode, temperature_data)
        syn_x_grid = sample_predictions(logits[:, 2], th_syn, sampling_mode, temperature_syn)
        syn_z_grid = sample_predictions(logits[:, 3], th_syn, sampling_mode, temperature_syn)

        # Compute syndromes from data corrections
        z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, n_rows * n_cols, T)
        z_data = z_flat[:, data_to_grid, :]  # (B, num_data, T)

        z_data_exp = z_data.unsqueeze(2).expand(B, num_data, K, T)
        h_idx_e = H_idx.clamp_min(0).view(1, num_plaq, K, 1).expand(B, -1, -1, T)
        g_z = z_data_exp.gather(1, h_idx_e)
        m_h = H_mask.view(1, num_plaq, K, 1).expand_as(g_z)
        S_from_z = (g_z.masked_fill(~m_h, 0).sum(dim=2) & 1)

        x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, n_rows * n_cols, T)
        x_data = x_flat[:, data_to_grid, :]  # (B, num_data, T)

        x_data_exp = x_data.unsqueeze(2).expand(B, num_data, K, T)
        g_x = x_data_exp.gather(1, h_idx_e)
        S_from_x = (g_x.masked_fill(~m_h, 0).sum(dim=2) & 1)

        syn_x_flat = map_grid_to_stab(syn_x_grid, stab_to_grid).to(torch.int32)
        syn_z_flat = map_grid_to_stab(syn_z_grid, stab_to_grid).to(torch.int32)

        if _enable_delta_s2_correction and _enable_z_ff and _ff_mask_tensor is not None and _cx_controls:
            meas_flat = batch["meas_flat"].to(device)
            delta_s2 = _compute_delta_s2_from_meas_flat(
                meas_flat=meas_flat,
                T=T,
                num_plaq=num_plaq,
                ff_mask_tensor=_ff_mask_tensor,
                cx_controls=_cx_controls,
                cx_targets=_cx_targets,
                z_check_offset=_z_check_offset,
                num_total_qubits=_num_total_qubits,
            )
            delta_s2 = _align_delta_s2_for_predecoder_mode(
                delta_s2, apply_feedforward_to_predecoder=True
            )
            syn_z_flat = (syn_z_flat + delta_s2) & 1

        # Compute residuals (basis-matched)
        if basis == "X":
            R = torch.empty_like(x_syn_diff, dtype=torch.int32)
            R[:, :, 0] = (x_syn_diff[:, :, 0] + syn_x_flat[:, :, 0] + S_from_z[:, :, 0]) & 1
            if T > 1:
                R[:, :, 1:] = (
                    x_syn_diff[:, :, 1:] + syn_x_flat[:, :, 1:] + syn_x_flat[:, :, :-1] +
                    S_from_z[:, :, 1:]
                ) & 1
        else:
            R = torch.empty_like(z_syn_diff, dtype=torch.int32)
            R[:, :, 0] = (z_syn_diff[:, :, 0] + syn_z_flat[:, :, 0] + S_from_x[:, :, 0]) & 1
            if T > 1:
                R[:, :, 1:] = (
                    z_syn_diff[:, :, 1:] + syn_z_flat[:, :, 1:] + syn_z_flat[:, :, :-1] +
                    S_from_x[:, :, 1:]
                ) & 1

        res_ones += R.sum(dtype=torch.int64)

    # All-reduce
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in (in_ones, in_elems, res_ones):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    input_density = (in_ones.float() /
                     in_elems.float()).item() if in_elems.item() > 0 else float('nan')
    residual_density = (res_ones.float() /
                        in_elems.float()).item() if in_elems.item() > 0 else float('nan')
    reduction = input_density / residual_density if residual_density > 0 else float('inf')

    return {
        "input_syndrome_ones": int(in_ones.item()),
        "residual_syndrome_ones": int(res_ones.item()),
        "syndrome_elements": int(in_elems.item()),
        "input_syndrome_density": input_density,
        "residual_syndrome_density": residual_density,
        "reduction_factor": reduction,
        "basis": basis,
    }


@torch.inference_mode()
def compute_chromobius_single_shot_timing_color(model, device, dist, cfg) -> dict:
    """Compute single-shot Chromobius decoder timing stats for one color-code basis."""
    basis = str(cfg.test.meas_basis_test).upper()
    if basis in ("BOTH", "MIXED"):
        raise ValueError(
            "compute_chromobius_single_shot_timing_color expects one basis at a time; "
            "run the X and Z sweeps separately."
        )

    result = count_logical_errors_color(
        model,
        device,
        dist,
        cfg,
        include_diagnostics=True,
        log_summary=False,
    )
    timing = result[basis].get("chromobius_timing", {}).get("single_shot_latency")
    if timing is None:
        raise RuntimeError(f"No single-shot Chromobius timing was produced for basis={basis}")

    return {
        "basis": basis,
        "n_rounds": int(getattr(cfg.test, "n_rounds", cfg.n_rounds)),
        "samples_timed": int(timing["samples_timed"]),
        "warmup_iterations": int(timing.get("warmup_iterations", 0)),
        "original_syndromes": timing["original_syndromes"],
        "residual_syndromes": timing["residual_syndromes"],
    }


# Expose main entry points
__all__ = [
    'PreDecoderColorEvalModule',
    'count_logical_errors_color',
    'run_inference_and_decode_color',
    'compute_syndrome_density_reduction_color',
    'compute_chromobius_single_shot_timing_color',
]
