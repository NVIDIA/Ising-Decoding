# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pymatching
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from enum import IntEnum
from pathlib import Path
from typing import Optional


class OnnxWorkflow(IntEnum):
    """ONNX_WORKFLOW env: 0=torch only, 1=export ONNX only, 2=export ONNX and use TensorRT, 3=use engine file only."""

    TORCH_ONLY = 0
    EXPORT_ONNX_ONLY = 1
    EXPORT_AND_USE_TRT = 2
    USE_ENGINE_ONLY = 3


from data.factory import DatapipeFactory
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from training.utils import *
from torch import amp
import time
import warnings

from qec.surface_code.data_mapping import (
    normalized_weight_mapping_Xstab_memory, normalized_weight_mapping_Zstab_memory,
    compute_stabX_to_data_index_map, compute_stabZ_to_data_index_map, map_grid_to_stabilizer_tensor,
    construct_X_stab_Parity_check_Mat, construct_Z_stab_Parity_check_Mat
)


def _detect_shm_bytes() -> Optional[int]:
    try:
        st = os.statvfs("/dev/shm")
        return int(st.f_frsize * st.f_blocks)
    except Exception:
        return None


def _parse_quant_format(rank: int = 0) -> str:
    """Read and validate the QUANT_FORMAT environment variable.

    Returns the validated format string ('int8' or 'fp8'), or '' if unset or invalid.
    Prints a warning on rank 0 when the value is set but not recognised.
    """
    quant_format = os.environ.get("QUANT_FORMAT", "").strip().lower()
    if quant_format and quant_format not in ("int8", "fp8"):
        if rank == 0:
            print(f"[LER] Invalid QUANT_FORMAT='{quant_format}', ignoring. Supported: int8, fp8")
        quant_format = ""
    return quant_format


def _collect_calibration_dets(
    test_dataloader,
    num_obs: int,
    target_samples: int,
    expected_width: int,
) -> "np.ndarray":
    """Collect representative detector inputs from a dataloader for ONNX calibration.

    Args:
        test_dataloader: DataLoader yielding batches with a "dets_and_obs" key.
        num_obs: Number of observable columns at the end of dets_and_obs to strip.
        target_samples: Desired number of calibration rows.
        expected_width: Expected number of detector columns after stripping observables.

    Returns:
        np.ndarray of shape (target_samples, expected_width), dtype uint8.
    """
    if num_obs < 1:
        raise ValueError(
            f"num_obs must be >= 1, got {num_obs}. "
            "dets_and_obs[:, :-0] would silently return an empty tensor."
        )
    target_samples = max(int(target_samples), 1)
    chunks = []
    collected = 0
    for calib_batch in test_dataloader:
        dets_and_obs_batch = calib_batch["dets_and_obs"]
        dets_only_batch = dets_and_obs_batch[:, :-num_obs].to(torch.uint8).contiguous()
        if int(dets_only_batch.shape[1]) != int(expected_width):
            raise RuntimeError(
                f"Calibration det width {dets_only_batch.shape[1]} != expected {expected_width}"
            )
        if dets_only_batch.numel() == 0:
            continue
        take = min(target_samples - collected, int(dets_only_batch.shape[0]))
        if take > 0:
            chunks.append(dets_only_batch[:take].cpu().numpy())
            collected += take
        if collected >= target_samples:
            break
    if not chunks:
        raise RuntimeError("No calibration samples could be collected from test_dataloader.")
    calib = np.concatenate(chunks, axis=0)
    if calib.shape[0] < target_samples:
        reps = int(np.ceil(target_samples / float(calib.shape[0])))
        calib = np.tile(calib, (reps, 1))[:target_samples]
    return np.ascontiguousarray(calib, dtype=np.uint8)


def _ort_quantize_int8(fp32_onnx_path: str, output_path: str, calib_dets: "np.ndarray") -> None:
    """INT8 static quantization via onnxruntime.quantization (Python 3.13+ fallback).

    Used when nvidia-modelopt is unavailable (it does not support Python 3.13+).
    Quantises all Conv and Gemm nodes with QInt8 weights and activations using
    QDQ format, which is compatible with TensorRT INT8 parsing.

    Args:
        fp32_onnx_path: Path to the source FP32 ONNX model.
        output_path: Destination path for the quantized ONNX model.
        calib_dets: Calibration data array of shape (N, det_cols), dtype uint8.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _DetCalibReader(CalibrationDataReader):

        def __init__(self, data):
            self._rows = [{"dets": data[i:i + 1].astype("float32")} for i in range(len(data))]
            self._iter = iter(self._rows)

        def get_next(self):
            return next(self._iter, None)

        def rewind(self):
            self._iter = iter(self._rows)

    quantize_static(
        fp32_onnx_path,
        output_path,
        _DetCalibReader(calib_dets),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )


def _time_single_shot_latency_stim(
    matcher,
    baseline_syndromes: np.ndarray,
    residual_syndromes: np.ndarray,
    *,
    n_rounds: int,
    warmup_iterations: int = 50,
) -> tuple[float, float]:
    """
    Time single-shot PyMatching decode latency (batch_size=1) with a "clean" CPU state.

    This is the most informative latency metric for PyMatching:
    - Uses `matcher.decode()` (not `decode_batch`)
    - Timed after the main evaluation loop to avoid CPU contention from GPU work

    Returns:
        (baseline_us_per_round_mean, predecoder_us_per_round_mean)
    """
    n_rounds = max(int(n_rounds), 1)
    if baseline_syndromes is None or residual_syndromes is None:
        return (float("nan"), float("nan"))
    n_samples = int(min(len(baseline_syndromes), len(residual_syndromes)))
    if n_samples <= 0:
        return (float("nan"), float("nan"))

    # Best-effort "clean CPU state": finish GPU work + reclaim Python garbage.
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    warmup_n = min(int(warmup_iterations), n_samples)
    for i in range(warmup_n):
        _ = matcher.decode(np.asarray(baseline_syndromes[i % n_samples], dtype=np.uint8))
        _ = matcher.decode(np.asarray(residual_syndromes[i % n_samples], dtype=np.uint8))

    baseline_times = []
    for i in range(n_samples):
        t_start = time.perf_counter()
        _ = matcher.decode(np.asarray(baseline_syndromes[i], dtype=np.uint8))
        baseline_times.append(time.perf_counter() - t_start)

    predecoder_times = []
    for i in range(n_samples):
        t_start = time.perf_counter()
        _ = matcher.decode(np.asarray(residual_syndromes[i], dtype=np.uint8))
        predecoder_times.append(time.perf_counter() - t_start)

    baseline_mean_us_per_round = float(np.mean(baseline_times) / n_rounds * 1e6)
    predecoder_mean_us_per_round = float(np.mean(predecoder_times) / n_rounds * 1e6)
    return (baseline_mean_us_per_round, predecoder_mean_us_per_round)


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
        temperature: Temperature parameter for softmax sampling (ignored if mode="threshold")
                    - Lower temperature (< 1.0): sharper decisions, closer to deterministic
                    - Higher temperature (> 1.0): more randomness, predictions drift toward 50/50
    
    Returns:
        Binary predictions as int32 tensor
    
    Note:
        When using temperature sampling with multi-GPU (distributed), each GPU samples
        independently with different random states, which is beneficial for statistical
        diversity. The aggregated statistics across GPUs will reflect this diversity.
    """
    if sampling_mode == "temperature":
        # Apply temperature scaling and convert to probabilities
        # P(prediction=1) = sigmoid(logit/T) = 1 / (1 + exp(-logit/T))
        scaled_logits = logits / temperature
        probs = torch.sigmoid(scaled_logits)
        # Sample from Bernoulli distribution (each GPU samples independently)
        return torch.bernoulli(probs).to(torch.int32)
    else:
        # Deterministic thresholding (default)
        return (logits >= threshold).to(torch.int32)


class CUDAPrefetcher:

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


from functools import lru_cache

# Last DEM timing snapshot from Stim-based validation (seconds).
LAST_DEM_TIMING = {"dem_build_s": 0.0, "dem_decode_s": 0.0}


@lru_cache(maxsize=64)
def _build_stab_maps(distance: int, rotation: str = 'XV'):
    """
    Build stabilizer maps for a given distance and rotation.
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    """
    # Build parity matrices on CPU as int32 - MUST be orientation-aware!
    rotation = rotation.upper() if rotation else 'XV'
    if rotation == 'XV':
        # Use hardcoded XV matrices (original behavior)
        Hx_i32 = construct_X_stab_Parity_check_Mat(distance).to(torch.int32)
        Hz_i32 = construct_Z_stab_Parity_check_Mat(distance).to(torch.int32)
    else:
        # For other orientations, get parity matrices from SurfaceCode
        from qec.surface_code.memory_circuit import SurfaceCode
        first_bulk = rotation[0]  # 'X' or 'Z'
        rotated = rotation[1]  # 'V' or 'H'
        code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rotated)
        Hx_i32 = torch.tensor(code.hx, dtype=torch.int32)
        Hz_i32 = torch.tensor(code.hz, dtype=torch.int32)

    def _row_indices_and_mask(H_i32: torch.Tensor):
        # H: (S, D2), entries are 0/1 (int32)
        S, D2 = H_i32.shape
        nz = H_i32.nonzero(as_tuple=False)  # (nnz, 2) [row, col]
        # Ensure row-major order (usually already true)
        # nz = nz[nz[:, 0].argsort(stable=True)]

        rows = nz[:, 0]
        cols = nz[:, 1]

        # Degree per row and max degree
        deg = torch.bincount(rows, minlength=S)
        K = int(deg.max().item())

        # Allocate outputs
        idx = torch.full((S, K), -1, dtype=torch.long)  # columns or -1
        msk = torch.zeros((S, K), dtype=torch.bool)  # valid flags

        if K == 0:
            return idx, msk, deg, K  # empty H (edge case)

        # Compute position-within-row for each NZ without a Python loop
        # Row offsets: [0, cumsum(deg)[:-1]]
        row_offsets = torch.zeros(S + 1, dtype=torch.long)
        row_offsets[1:] = deg.cumsum(0)
        # Count appearances per row: position = running count within that row
        # We can get a stable per-row index by taking arange over nnz and subtracting row_offsets[rows].
        # But that requires nz to be grouped by row (usually is). If unsure, do a stable sort by rows above.
        pos = torch.arange(nz.size(0), dtype=torch.long) - row_offsets[rows]

        # Place columns
        idx[rows, pos] = cols
        # Mark masks up to degree per row
        # Build a per-row [0..K-1] matrix and compare to deg
        ar = torch.arange(K, dtype=torch.long).unsqueeze(0).expand(S, K)
        msk = ar < deg.unsqueeze(1)

        return idx, msk, deg, K

    Hx_idx_cpu, Hx_mask_cpu, Hx_deg_cpu, Kx = _row_indices_and_mask(Hx_i32)
    Hz_idx_cpu, Hz_mask_cpu, Hz_deg_cpu, Kz = _row_indices_and_mask(Hz_i32)

    # grid→stab index lists (CPU long) - now rotation-aware
    rotation = rotation.upper() if rotation else 'XV'
    x_idx_list = compute_stabX_to_data_index_map(distance, rotation)
    z_idx_list = compute_stabZ_to_data_index_map(distance, rotation)
    stab_x_cpu = x_idx_list if torch.is_tensor(x_idx_list
                                              ) else torch.tensor(x_idx_list, dtype=torch.long)
    stab_z_cpu = z_idx_list if torch.is_tensor(z_idx_list
                                              ) else torch.tensor(z_idx_list, dtype=torch.long)

    # (Optional) sanity: surface code typical degree ≤4
    # if Kx > 4 or Kz > 4:
    #     raise ValueError(f"Unexpected stabilizer degree: Kx={Kx}, Kz={Kz}")

    return {
        "Hx_i32": Hx_i32,
        "Hz_i32": Hz_i32,
        "Hx_idx": Hx_idx_cpu,
        "Hx_mask": Hx_mask_cpu,
        "Hx_deg": Hx_deg_cpu,
        "Kx": Kx,
        "Hz_idx": Hz_idx_cpu,
        "Hz_mask": Hz_mask_cpu,
        "Hz_deg": Hz_deg_cpu,
        "Kz": Kz,
        "stab_x": stab_x_cpu,
        "stab_z": stab_z_cpu,
    }


def move_to_device(x, device, non_blocking=False):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [move_to_device(v, device, non_blocking) for v in x]
        return type(x)(t)  # preserve tuple/list
    return x


def interleave_XZ_residuals(R_X: torch.Tensor, R_Z: torch.Tensor) -> torch.Tensor:
    """
    R_X: (B, num_X_stabs, T)
    R_Z: (B, num_Z_stabs, T)
    Returns:
        interleaved residuals: (B, T * (num_X_stabs + num_Z_stabs))
    """
    B, nX, T = R_X.shape
    _, nZ, _ = R_Z.shape

    # Transpose to shape (B, T, nX/nZ)
    R_X_t = R_X.permute(0, 2, 1).contiguous()  # (B, T, nX)
    R_Z_t = R_Z.permute(0, 2, 1).contiguous()  # (B, T, nZ)

    # Concatenate along stabilizer dimension: (B, T, nX + nZ)
    R_cat = torch.cat([R_X_t, R_Z_t], dim=2)  # (B, T, nX + nZ)

    # Reshape to flat per time step: (B, T*(nX + nZ))
    return R_cat.reshape(B, T * (nX + nZ))


def map_grid_to_stabilizer_tensor(grid_btdd, stab_indices_1d):
    """
    grid_btdd: (B, T, D, D) float/bool/uint8
    stab_indices_1d: (num_stabs,) long indices into flattened D*D in row-major
    returns: (B, num_stabs, T)
    """
    B, T, D, _ = grid_btdd.shape
    # (B, D*D, T)
    flat_bdt = grid_btdd.permute(0, 2, 3, 1).contiguous().reshape(B, D * D, T)
    # advanced indexing (fast + simple)
    return flat_bdt.index_select(dim=1, index=stab_indices_1d)


class PreDecoderMemoryEvalModule(nn.Module):
    """
    nn.Module that encapsulates the full pre-decoder eval path: batch input -> trainX,
    model forward -> logits -> sampling -> induced syndromes -> residuals -> logical frame
    and residual detector layout. Used for evaluation and (later) ONNX export.
    """

    def __init__(self, model: nn.Module, cfg, maps: dict, device):
        super().__init__()
        self.model = model
        self.D = cfg.distance
        self.D2 = self.D * self.D
        self.half = (self.D * self.D - 1) // 2
        self.basis = str(getattr(cfg.test, "meas_basis_test", "X")).upper()
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {self.basis}")

        self.th_data = float(getattr(cfg.test, "th_data", 0.0))
        self.th_syn = float(getattr(cfg.test, "th_syn", 0.0))
        self.sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
        self.temperature_data = float(
            getattr(cfg.test, "temperature_data", None) or getattr(cfg.test, "temperature", 1.0)
        )
        self.temperature_syn = float(
            getattr(cfg.test, "temperature_syn", None) or getattr(cfg.test, "temperature", 1.0)
        )
        self.enable_fp16 = bool(getattr(cfg, "enable_fp16", False))
        self.num_obs = 1

        # Dense matrices for post-processing matmuls (gather, parity check).
        stab_x = maps["stab_x"].to(device=device, dtype=torch.long)
        stab_z = maps["stab_z"].to(device=device, dtype=torch.long)
        col_idx = torch.arange(self.half, device=device)
        scatter_x = torch.zeros(self.D2, self.half, dtype=torch.float32, device=device)
        scatter_x[stab_x, col_idx] = 1.0
        scatter_z = torch.zeros(self.D2, self.half, dtype=torch.float32, device=device)
        scatter_z[stab_z, col_idx] = 1.0

        gather_x = scatter_x.t().contiguous()  # (half, D²)
        gather_z = scatter_z.t().contiguous()  # (half, D²)

        # Scatter-via-gather permutation for preprocessing (myelin-fusible).
        scatter_perm_x = torch.full((self.D2,), self.half, dtype=torch.long, device=device)
        scatter_perm_x[stab_x] = col_idx
        scatter_perm_z = torch.full((self.D2,), self.half, dtype=torch.long, device=device)
        scatter_perm_z[stab_z] = col_idx
        self.register_buffer("scatter_perm_x", scatter_perm_x)
        self.register_buffer("scatter_perm_z", scatter_perm_z)

        # Zero-padding row for scatter-via-gather: (1, 1, 1) expanded to (B, 1, T).
        self.register_buffer(
            "zero_pad_row", torch.zeros(1, 1, 1, dtype=torch.float32, device=device)
        )

        # Logical strings - orientation-aware, float32.
        code_rotation = getattr(cfg.data, "code_rotation", "XV").upper()
        Lx = torch.zeros((1, self.D2), dtype=torch.float32, device=device)
        Lz = torch.zeros((1, self.D2), dtype=torch.float32, device=device)
        if code_rotation in ("XV", "ZH"):
            Lx[0, :self.D] = 1
            Lz[0, ::self.D] = 1
        else:
            Lx[0, ::self.D] = 1
            Lz[0, :self.D] = 1

        # Pad the basis-dependent L vector to (half, D²) so it fits as the 5th GEMM row.
        L_padded = torch.zeros(self.half, self.D2, dtype=torch.float32, device=device)
        L_padded[0] = Lx[0] if self.basis == "X" else Lz[0]

        # Stacked post-processing matrices: (5, half, D²) — one batched GEMM.
        # [0] Hx_dense @ z_flat → S_X,  [1] Hz_dense @ x_flat → S_Z,
        # [2] gather_x @ syn_x → syn_x_flat,  [3] gather_z @ syn_z → syn_z_flat,
        # [4] L_padded @ preL_input → pre_L (row 0 only).
        Hx_dense = maps["Hx_i32"].to(dtype=torch.float32, device=device)
        Hz_dense = maps["Hz_i32"].to(dtype=torch.float32, device=device)
        self.register_buffer(
            "post_matrices",
            torch.stack([
                Hx_dense,
                Hz_dense,
                gather_x,
                gather_z,
                L_padded,
            ], dim=0)
        )

        # Presence grids: (1, 1, D, D); expanded and masked per batch in forward
        w_mapX = normalized_weight_mapping_Xstab_memory(
            self.D, getattr(cfg.data, "code_rotation", "XV")
        ).reshape(self.D, self.D).unsqueeze(0).unsqueeze(0)
        w_mapZ = normalized_weight_mapping_Zstab_memory(
            self.D, getattr(cfg.data, "code_rotation", "XV")
        ).reshape(self.D, self.D).unsqueeze(0).unsqueeze(0)
        self.register_buffer(
            "w_mapXgrid", torch.as_tensor(w_mapX, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "w_mapZgrid", torch.as_tensor(w_mapZ, dtype=torch.float32, device=device)
        )

        # Pin residual output size for ONNX so the second dimension is a constant (not Concatresidual_dim_1).
        n_rounds = getattr(getattr(cfg, "test", None), "n_rounds", None)
        self.num_residual_dets = (int(2 * n_rounds * self.half) if n_rounds is not None else None)

    def _batch_to_trainx_and_syndromes(self, dets: torch.Tensor):
        """From dets (B, 2*T*half) build trainX, x_syn_diff, z_syn_diff, baseline_detectors_batch."""
        B, num_dets = dets.shape
        half = self.half
        # num_dets = 2*T*half => T = num_dets // (2*half)
        T = num_dets // (2 * half)
        if not getattr(torch.onnx, "is_in_onnx_export", lambda: False)():
            assert T >= 2, f"T={T} is too small for DEM (need T>=2)."

        timeline_len = 2 * T

        # ── trt_L1: preprocessor (cast, deinterleave, index_select, boundary handling) ──
        # (B, 2*T*half) -> (B, half, 2*T) float32.
        dets_timeline = dets.to(torch.float32).view(B, timeline_len, half).permute(0, 2,
                                                                                   1).contiguous()
        zero_col = self.zero_pad_row.expand(B, half, 1)  # (B, half, 1)
        dets_timeline_padded = torch.cat([dets_timeline, zero_col], dim=2)  # (B, half, 2*T+1)

        # Build deinterleave indices dynamically for this T.
        sentinel_idx = timeline_len  # points to appended all-zero column
        dev = dets.device
        x_bulk_idx = torch.arange(1, timeline_len - 1, 2, dtype=torch.long, device=dev)  # T-1
        z_bulk_idx = torch.arange(2, timeline_len, 2, dtype=torch.long, device=dev)  # T-1

        _zero = torch.zeros(1, dtype=torch.long, device=dev)
        _sentinel = torch.full((1,), sentinel_idx, dtype=torch.long, device=dev)

        if self.basis == "X":
            idx_x = torch.cat([_zero, x_bulk_idx])
            idx_z = torch.cat([_sentinel, z_bulk_idx[:-1], _sentinel])
            x_syn_diff = torch.index_select(dets_timeline_padded, 2, idx_x)  # (B, half, T)
            z_syn_diff = torch.index_select(dets_timeline_padded, 2, idx_z)  # (B, half, T)
        else:
            idx_z = torch.cat([_zero, z_bulk_idx])
            idx_x = torch.cat([_sentinel, x_bulk_idx[:-1], _sentinel])
            z_syn_diff = torch.index_select(dets_timeline_padded, 2, idx_z)  # (B, half, T)
            x_syn_diff = torch.index_select(dets_timeline_padded, 2, idx_x)  # (B, half, T)

        # Presence: broadcast-multiply by round mask to zero boundary rounds (no clone/in-place)
        boundary_mask = torch.cat(
            [
                torch.zeros(1, device=dev, dtype=torch.float32),
                torch.ones(T - 2, device=dev, dtype=torch.float32),
                torch.zeros(1, device=dev, dtype=torch.float32),
            ]
        ).view(1, T, 1, 1)
        if self.basis == "X":
            x_present = self.w_mapXgrid.expand(B, T, self.D, self.D)
            z_present = (self.w_mapZgrid * boundary_mask).expand(B, T, self.D, self.D)
        else:
            x_present = (self.w_mapXgrid * boundary_mask).expand(B, T, self.D, self.D)
            z_present = self.w_mapZgrid.expand(B, T, self.D, self.D)

        # ── trt_L2: trainX assembly (scatter-via-gather → grid reshape → cat) ──
        zero_pad = self.zero_pad_row.expand(B, 1, T)
        x_grid = torch.index_select(
            torch.cat([x_syn_diff, zero_pad], dim=1), 1, self.scatter_perm_x
        )  # (B, D², T)
        z_grid = torch.index_select(
            torch.cat([z_syn_diff, zero_pad], dim=1), 1, self.scatter_perm_z
        )  # (B, D², T)
        x_type = x_grid.reshape(B, self.D, self.D, T).permute(0, 3, 1,
                                                              2).contiguous()  # (B, T, D, D)
        z_type = z_grid.reshape(B, self.D, self.D, T).permute(0, 3, 1, 2).contiguous()
        trainX = torch.cat(
            [
                x_type.unsqueeze(1),
                z_type.unsqueeze(1),
                x_present.unsqueeze(1),
                z_present.unsqueeze(1),
            ],
            dim=1
        ).contiguous()

        n_x = half
        n_z = z_syn_diff.shape[1]
        num_boundary_dets = n_x if self.basis == "X" else n_z
        baseline_detectors_batch = dets_timeline.permute(0, 2, 1).contiguous().view(B, -1)

        return trainX, x_syn_diff, z_syn_diff, baseline_detectors_batch, num_boundary_dets, B, T, n_x, n_z

    def forward(self, dets: torch.Tensor):
        """
        Args:
            dets: (B, 2*T*half) detector bits only (no observable), dtype uint8.
        Returns:
            out: (B, 1 + num_detectors) uint8, where out[:, 0] is pre_L and
                 out[:, 1:] is residual (for global decoder).
        """
        trainX, x_syn_diff, z_syn_diff, baseline_detectors_batch, num_boundary_dets, B, T, n_x, n_z = self._batch_to_trainx_and_syndromes(
            dets
        )

        device_type = next(self.parameters()).device.type if next(
            self.parameters()
        ).device.type in ("cuda", "cpu") else "cpu"
        # ── trt_L3–L6: four Conv3D blocks (inside self.model) ──
        with torch.amp.autocast(device_type=device_type, enabled=self.enable_fp16):
            logits = self.model(trainX)

        # ── trt_L7: threshold/cast/reshape/transpose (sample_predictions → flatten → stack) ──
        z_data_corr = sample_predictions(
            logits[:, 0], self.th_data, self.sampling_mode, self.temperature_data
        )
        x_data_corr = sample_predictions(
            logits[:, 1], self.th_data, self.sampling_mode, self.temperature_data
        )
        syn_x_grid = sample_predictions(
            logits[:, 2], self.th_syn, self.sampling_mode, self.temperature_syn
        )
        syn_z_grid = sample_predictions(
            logits[:, 3], self.th_syn, self.sampling_mode, self.temperature_syn
        )

        # Flatten each grid to (B, D^2, T), then stack for one fused batched GEMM.
        # post_matrices: (5, half, D^2)
        # all_flat:      (B, 5, D^2, T)
        # matmul result: (B, 5, half, T)
        z_corr_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, self.D2, T)
        x_corr_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, self.D2, T)
        syn_x_flat_in = syn_x_grid.permute(0, 2, 3, 1).contiguous().view(B, self.D2, T)
        syn_z_flat_in = syn_z_grid.permute(0, 2, 3, 1).contiguous().view(B, self.D2, T)
        preL_input = z_corr_flat if self.basis == "X" else x_corr_flat
        all_flat = torch.stack(
            [
                z_corr_flat,
                x_corr_flat,
                syn_x_flat_in,
                syn_z_flat_in,
                preL_input,
            ], dim=1
        ).to(torch.float32)  # (B, 5, D², T)

        # ── trt_L8: batched GEMM (5-matrix post_matrices @ all_flat) ──
        all_results = torch.matmul(self.post_matrices, all_flat)  # (B, 5, half, T)

        # ── trt_L9: pre_L slice (extract pre_L row from GEMM result) ──
        # ── trt_L10: pre_L reduction (mod2 → sum over time → mod2 → cast int32) ──
        pre_L = all_results[:, 4,
                            0:1, :].remainder(2).sum(dim=-1).remainder(2).view(-1).to(torch.int32)

        # ── trt_L11: residual assembly + interleave (stacked R_X/R_Z → permute → cat → cast int32) ──
        S_xz = all_results[:, :2].remainder(2)  # (B, 2, half, T)
        syn_xz = all_results[:, 2:4]  # (B, 2, half, T)
        syn_diff_xz = torch.stack([x_syn_diff, z_syn_diff], dim=1)  # (B, 2, half, T)

        # Time-recurrent residual equation:
        # R_0    = (input + pred + induced) mod 2
        # R_rest = (input[t] + pred[t] + pred[t-1] + induced[t]) mod 2
        R_0 = (syn_diff_xz[..., 0:1] + syn_xz[..., 0:1] + S_xz[..., 0:1]).remainder(2)
        R_rest = (syn_diff_xz[..., 1:] + syn_xz[..., 1:] + syn_xz[..., :-1] +
                  S_xz[..., 1:]).remainder(2)
        R_xz = torch.cat([R_0, R_rest], dim=-1)  # (B, 2, half, T)

        if self.basis == "X":
            initial = R_xz[:, 0, :, 0:1].reshape(B, -1)  # (B, half)
        else:
            initial = R_xz[:, 1, :, 0:1].reshape(B, -1)  # (B, half)
        # (B, 2, half, T-1) → (B, T-1, 2, half) → (B, (T-1)*2*half)
        rest = R_xz[:, :, :, 1:].permute(0, 3, 1, 2).contiguous().reshape(B, -1)
        residual = torch.cat(
            [initial, rest, baseline_detectors_batch[:, -num_boundary_dets:]], dim=1
        ).to(torch.int32)

        # Reshape with constant second dim so ONNX exports shape [batch, num_residual_dets] (no symbolic Concatresidual_dim_1).
        if self.num_residual_dets is not None:
            residual = residual.reshape(B, self.num_residual_dets)

        # ── trt_L12: final output assembly (transpose, reshape, concat, cast → uint8) ──
        out = torch.cat([pre_L.view(B, 1), residual], dim=1).to(torch.uint8)
        return out


def count_logical_errors_with_errorbar(model, device, dist, cfg):
    #logical_errors.item(), total_samples, num_pymatch_errors
    result = {}

    if cfg.test.meas_basis_test.lower() in ("both", "mixed"):
        orig = cfg.test.meas_basis_test
        # First run X, then Z
        for i in range(2):
            if i == 0:
                cfg.test.meas_basis_test = "X"
            else:
                cfg.test.meas_basis_test = "Z"
            verbose = bool(getattr(cfg.test, "verbose_inference", False)
                          ) or bool(getattr(cfg.test, "verbose", False))
            t0 = time.time()
            num_errors, num_shots, pymatch_predictions, baseline_us_per_round, predecoder_us_per_round = run_inference_and_decode_pre_decoder_memory(
                model, device, dist, cfg
            )
            tf = time.time()
            if verbose and dist.rank == 0:
                print(f"Time taken for {cfg.test.meas_basis_test}: {tf - t0:.3f}s")

            # Because each element is either 1 or 0, the sum_i of x_i == the sum_i of x_i^2.
            var = (num_errors - num_errors * num_errors / float(num_shots)) / num_shots
            stddev = np.sqrt(var)
            pymatch_var = (
                pymatch_predictions - pymatch_predictions * pymatch_predictions / float(num_shots)
            ) / num_shots
            pymatch_stddev = np.sqrt(pymatch_var)

            result[cfg.test.meas_basis_test] = {
                "num shots":
                    int(num_shots),
                "logical errors":
                    int(num_errors),
                "pymatch flips":
                    int(pymatch_predictions),
                "logical error ratio (mean)":
                    float(num_errors / num_shots),
                "logical error ratio (standard error)":
                    float(stddev / np.sqrt(num_shots)),
                "logical error ratio (pymatch mean)":
                    float(pymatch_predictions / float(num_shots)),
                "logical error ratio (pymatch standard error)":
                    float(pymatch_stddev / np.sqrt(num_shots)),
                "pymatch latency (baseline µs/round)":
                    float(baseline_us_per_round),
                "pymatch latency (after predecoder µs/round)":
                    float(predecoder_us_per_round),
            }
        cfg.test.meas_basis_test = orig
    else:
        num_errors, num_shots, pymatch_predictions, baseline_us_per_round, predecoder_us_per_round = run_inference_and_decode_pre_decoder_memory(
            model, device, dist, cfg
        )

        # Because each element is either 1 or 0, the sum_i of x_i == the sum_i of x_i^2.
        var = (num_errors - num_errors * num_errors / float(num_shots)) / num_shots
        stddev = np.sqrt(var)
        pymatch_var = (
            pymatch_predictions - pymatch_predictions * pymatch_predictions / float(num_shots)
        ) / num_shots
        pymatch_stddev = np.sqrt(pymatch_var)

        result[cfg.test.meas_basis_test] = {
            "num shots":
                int(num_shots),
            "logical errors":
                int(num_errors),
            "pymatch flips":
                int(pymatch_predictions),
            "logical error ratio (mean)":
                float(num_errors / num_shots),
            "logical error ratio (standard error)":
                float(stddev / np.sqrt(num_shots)),
            "logical error ratio (pymatch mean)":
                float(pymatch_predictions / float(num_shots)),
            "logical error ratio (pymatch standard error)":
                float(pymatch_stddev / np.sqrt(num_shots)),
            "pymatch latency (baseline µs/round)":
                float(baseline_us_per_round),
            "pymatch latency (after predecoder µs/round)":
                float(predecoder_us_per_round),
        }
    return result


@torch.inference_mode()
def run_inference_and_decode_pre_decoder_memory(model, device, dist, cfg) -> dict:
    """
    Runs inference with the trained model, forms residual syndromes consistent with the DEM,
    and computes the final logical error rate with PyMatching.
    Returns: (num_logic_errors_after_predecoder+pymatching, num_samples, num_pymatch_errors_baseline)
    """

    th_data = float(getattr(cfg.test, "th_data", 0.0))
    th_syn = float(getattr(cfg.test, "th_syn", 0.0))

    # Sampling mode configuration
    sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
    temperature = float(getattr(cfg.test, "temperature", 1.0))
    temperature_data = getattr(cfg.test, "temperature_data", None)
    temperature_syn = getattr(cfg.test, "temperature_syn", None)
    temperature_data = float(temperature_data) if temperature_data is not None else temperature
    temperature_syn = float(temperature_syn) if temperature_syn is not None else temperature

    # Calculate samples per GPU (divide total across GPUs)
    total_samples = int(cfg.test.num_samples)
    samples_per_gpu = total_samples // dist.world_size

    verbose = bool(getattr(cfg.test, "verbose_inference", False)
                  ) or bool(getattr(cfg.test, "verbose", False))
    # Log distributed and sampling configuration (only on rank 0, verbose only)
    if verbose and dist.rank == 0:
        if dist.world_size > 1:
            print(f"[LER] Distributed inference: {dist.world_size} GPUs")
            print(f"[LER] Total samples (cfg): {total_samples}")
            print(f"[LER] Samples per GPU: {samples_per_gpu}")
        if sampling_mode == "temperature":
            print(
                f"[LER] Sampling mode: temperature (T_data={temperature_data}, T_syn={temperature_syn})"
            )
        else:
            print(f"[LER] Sampling mode: threshold (th_data={th_data}, th_syn={th_syn})")

    model.eval()

    # Latency measurement: sample a small subset and time decode at batch_size=1.
    # Default: time on 10k single-shot decodes (batch_size=1) for a stable, informative metric.
    latency_num_samples = int(getattr(cfg.test, "latency_num_samples", 10_000))
    latency_num_samples = max(latency_num_samples, 0)
    latency_baseline_rows = [] if dist.rank == 0 and latency_num_samples > 0 else None
    latency_predecoder_rows = [] if dist.rank == 0 and latency_num_samples > 0 else None

    # --- Distributed: Each GPU generates its share of samples with rank-specific seed ---
    import random
    from copy import deepcopy

    # Save random states
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # Set rank-specific seed so each GPU generates DIFFERENT samples
        rank_seed = 12345 + dist.rank * 1000
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        np.random.seed(rank_seed)
        random.seed(rank_seed)

        # Create datapipe with reduced sample count per GPU
        cfg_copy = deepcopy(cfg)
        cfg_copy.test.num_samples = samples_per_gpu
        test_dataset = DatapipeFactory.create_datapipe_inference(cfg_copy)
    finally:
        # Restore random states
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

    if cfg.test.meas_basis_test.upper() in ["X", "Z"]:
        circuit = test_dataset.circ.stim_circuit

    dem_build_s = 0.0
    dem_decode_s = 0.0

    # --- Build PyMatching from the circuit's exact detector ordering ---
    # NOTE: With explicit noise models we use PAULI_CHANNEL_2, which requires
    # approximate_disjoint_errors=True when extracting a detector error model.
    t_dem_build_start = time.perf_counter()
    det_model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    matcher = pymatching.Matching.from_detector_error_model(det_model)
    dem_build_s += time.perf_counter() - t_dem_build_start

    # Baseline: PyMatching directly on Stim detectors (no pre-decoder), for reference
    # Each GPU computes baseline on its OWN samples
    stim_dets = np.asarray(test_dataset.dets_and_obs[:, :-circuit.num_observables], dtype=np.uint8)
    assert stim_dets.shape[1] == det_model.num_detectors, \
        f"Stim dets width {stim_dets.shape[1]} != DEM {det_model.num_detectors}"
    stim_obs = np.asarray(test_dataset.dets_and_obs[:, -circuit.num_observables:], dtype=np.uint8)

    # Each GPU computes baseline on its own samples
    # (Boundary detectors are included in stim_dets via add_boundary_detectors=True)

    if dist.rank == 0 and latency_baseline_rows is not None and len(
        latency_baseline_rows
    ) < latency_num_samples:
        remaining = latency_num_samples - len(latency_baseline_rows)
        take = min(int(stim_dets.shape[0]), remaining)
        for i in range(take):
            latency_baseline_rows.append(stim_dets[i].copy())

    t_dem_decode = time.perf_counter()
    baseline_pred = matcher.decode_batch(stim_dets)
    dem_decode_s += time.perf_counter() - t_dem_decode

    baseline_pred = np.asarray(baseline_pred, dtype=np.uint8).reshape(-1, circuit.num_observables)
    num_pymatch_errors = int((baseline_pred != stim_obs).sum())

    # --- DataLoader: NO DistributedSampler - each GPU processes ALL of its own samples ---
    test_loader_kwargs = dict(cfg.test.dataloader)

    batch_size_original = test_loader_kwargs.get("batch_size", 1)
    T_original = cfg.test.n_rounds

    # Allow env override for SDR DataLoader workers (container/shm safety).
    try:
        override_workers = (
            os.environ.get("PREDECODER_SDR_NUM_WORKERS") or
            os.environ.get("PREDECODER_EVAL_NUM_WORKERS") or
            os.environ.get("PREDECODER_INFERENCE_NUM_WORKERS")
        )
        if override_workers is not None:
            test_loader_kwargs["num_workers"] = int(override_workers)
        else:
            is_container = os.path.exists("/.dockerenv")
            if is_container and int(test_loader_kwargs.get("num_workers", 0)) > 0:
                shm_bytes = _detect_shm_bytes()
                if shm_bytes is not None and shm_bytes < 1_000_000_000:
                    test_loader_kwargs["num_workers"] = 0
                    try:
                        if dist.rank == 0:
                            print(
                                f"[Evaluation] Detected small /dev/shm "
                                f"({shm_bytes / (1024 ** 2):.1f} MiB); "
                                "setting num_workers=0. "
                                "Override with PREDECODER_INFERENCE_NUM_WORKERS."
                            )
                    except Exception:
                        pass
    except Exception:
        pass
    # Handle prefetch_factor when num_workers=0
    if test_loader_kwargs.get('num_workers', 0) == 0:
        test_loader_kwargs.pop('prefetch_factor', None)
        if test_loader_kwargs.get('persistent_workers', False):
            test_loader_kwargs['persistent_workers'] = False
    else:
        test_loader_kwargs.setdefault("multiprocessing_context", "spawn")
    # Use regular DataLoader - no sampler partitioning
    test_dataloader = DataLoader(test_dataset, shuffle=False, **test_loader_kwargs)

    # --- Precompute parity structs and build eval pipeline (nn.Module for inference + post-process) ---
    D = cfg.distance
    code_rotation = getattr(cfg.data, 'code_rotation', 'XV')
    maps = _build_stab_maps(D, code_rotation)
    pipeline_module = PreDecoderMemoryEvalModule(model, cfg, maps, device).to(device)
    pipeline_module.eval()

    basis = str(getattr(cfg.test, "meas_basis_test", "X")).upper()
    if basis not in ("X", "Z"):
        raise AssertionError(f"Invalid meas_basis_test='{basis}'. Use 'X' or 'Z'.")

    # --- ONNX_WORKFLOW: 0=torch only, 1=export ONNX only, 2=export ONNX and use TensorRT, 3=use engine file only ---
    _workflow_raw = os.environ.get("ONNX_WORKFLOW", "0").strip()
    try:
        onnx_workflow = OnnxWorkflow(int(_workflow_raw))
    except ValueError:
        onnx_workflow = OnnxWorkflow.TORCH_ONLY
        if dist.rank == 0:
            print(f"[LER] Invalid ONNX_WORKFLOW='{_workflow_raw}', using 0 (torch only).")
    trt_context = None  # (context, engine, device_id) when using TensorRT
    # --- QUANT_FORMAT: optional quantization (int8, fp8) applied to ONNX after FP32 export ---
    quant_format = _parse_quant_format(rank=dist.rank)
    quant_suffix = f"_{quant_format}" if quant_format else ""
    onnx_path = os.path.join(
        os.getcwd(), f"predecoder_memory_d{D}_T{T_original}_{basis}{quant_suffix}.onnx"
    )
    engine_path = os.path.join(
        os.getcwd(), f"predecoder_memory_d{D}_T{T_original}_{basis}{quant_suffix}.engine"
    )
    half = (D * D - 1) // 2
    example_shape = (batch_size_original, 2 * T_original * half)

    if onnx_workflow == OnnxWorkflow.USE_ENGINE_ONLY and device.type == "cuda":
        if os.path.isfile(engine_path):
            try:
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                t_load_start = time.perf_counter()
                with open(engine_path, "rb") as f:
                    serialized = f.read()
                engine = runtime.deserialize_cuda_engine(serialized)
                t_load_end = time.perf_counter()
                if engine is None:
                    raise RuntimeError("TensorRT engine deserialize from file failed")
                trt_context = (engine.create_execution_context(), engine)
                if dist.rank == 0:
                    print(
                        f"[LER] TensorRT engine loaded from {engine_path} in {t_load_end - t_load_start:.3f}s"
                    )
            except ImportError as e:
                raise RuntimeError(
                    "[LER] ONNX_WORKFLOW=3 (USE_ENGINE_ONLY) requires tensorrt to be installed. "
                    "Install with: pip install tensorrt"
                ) from e
            except Exception as e:
                if dist.rank == 0:
                    print(f"[LER] TensorRT engine load failed: {e}; falling back to PyTorch.")
                trt_context = None
        else:
            if dist.rank == 0:
                print(
                    f"[LER] ONNX_WORKFLOW=3 but engine file not found: {engine_path}; falling back to PyTorch."
                )

    elif onnx_workflow in (OnnxWorkflow.EXPORT_ONNX_ONLY, OnnxWorkflow.EXPORT_AND_USE_TRT):
        if dist.rank == 0:
            try:
                example_dets = torch.randint(0, 2, example_shape, dtype=torch.uint8, device=device)

                # Step 1: Always export FP32 ONNX first
                fp32_onnx_path = (
                    onnx_path
                    if not quant_format else onnx_path.replace(f"_{quant_format}.onnx", ".onnx")
                )
                torch.onnx.export(
                    pipeline_module,
                    example_dets,
                    fp32_onnx_path,
                    opset_version=18,
                    external_data=False,
                    input_names=["dets"],
                    output_names=["L_and_residual_dets"],
                    dynamic_axes={
                        "dets": {
                            0: "batch"
                        },
                        "L_and_residual_dets": {
                            0: "batch"
                        },
                    },
                    do_constant_folding=True,
                    dynamo=False,
                )
                print(f"[LER] Exported FP32 ONNX: {fp32_onnx_path}")

                # Step 2: If QUANT_FORMAT is set, apply ONNX-level quantization.
                # Backend: nvidia-modelopt on Python <3.13; onnxruntime on Python 3.13+
                # (nvidia-modelopt does not support Python 3.13+).
                if quant_format:
                    try:
                        num_obs_for_calib = circuit.num_observables
                        calib_num_samples = int(os.environ.get("QUANT_CALIB_SAMPLES", "256"))
                        print(
                            f"[LER] Collecting {calib_num_samples} calibration samples "
                            "from inference dataloader..."
                        )
                        calib_dets = _collect_calibration_dets(
                            test_dataloader, num_obs_for_calib, calib_num_samples, example_shape[1]
                        )

                        print(
                            f"[LER] Applying {quant_format.upper()} quantization to ONNX model..."
                        )
                        # Prefer modelopt (INT8+FP8); fall back to onnxruntime (INT8 only)
                        # when modelopt is not installed.  On Python 3.13+ modelopt can
                        # be installed with: pip install nvidia-modelopt[onnx]
                        #                                  --ignore-requires-python
                        try:
                            import modelopt.onnx.quantization as mq
                            quant_kwargs = {}
                            if quant_format == "fp8":
                                quant_kwargs["op_types_to_quantize"] = ["Conv"]
                                quant_kwargs["high_precision_dtype"] = "fp16"
                            mq.quantize(
                                onnx_path=fp32_onnx_path,
                                quantize_mode=quant_format,
                                calibration_data={"dets": calib_dets.astype("float32")},
                                output_path=onnx_path,
                                **quant_kwargs,
                            )
                        except ImportError:
                            if quant_format == "fp8":
                                raise RuntimeError(
                                    "[LER] FP8 quantization requires nvidia-modelopt. "
                                    "Install with: pip install 'nvidia-modelopt[onnx]'"
                                    " --ignore-requires-python"
                                )
                            _ort_quantize_int8(fp32_onnx_path, onnx_path, calib_dets)
                        print(f"[LER] Exported quantized ONNX: {onnx_path}")
                    except Exception as e:
                        if quant_format == "fp8":
                            raise RuntimeError(
                                f"[LER] FP8 ONNX quantization failed (fail-fast): {e}"
                            ) from e
                        print(f"[LER] ONNX quantization failed: {e}; using FP32 ONNX.")
                        onnx_path = fp32_onnx_path
            except Exception as e:
                if dist.rank == 0:
                    print(f"[LER] ONNX export failed: {e}; falling back to PyTorch.")
                onnx_workflow = OnnxWorkflow.TORCH_ONLY
        if dist.world_size > 1:
            torch.distributed.barrier()
        # Re-derive engine_path from the final onnx_path (may have changed on quant fallback)
        engine_path = str(Path(onnx_path).with_suffix(".engine"))
        if onnx_workflow == OnnxWorkflow.EXPORT_AND_USE_TRT and device.type == "cuda":
            try:
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                builder = trt.Builder(logger)
                net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                if quant_format in ("fp8", "int8"):
                    net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
                network = builder.create_network(net_flags)
                parser = trt.OnnxParser(network, logger)
                with open(onnx_path, "rb") as f:
                    if not parser.parse(f.read()):
                        raise RuntimeError("TensorRT ONNX parse failed")
                config = builder.create_builder_config()
                if not quant_format:
                    config.set_flag(trt.BuilderFlag.FP16)
                # Uncomment this out to speedup engine build time.
                # config.builder_optimization_level = 0
                in_name = "dets"
                in_cols_trt = 2 * T_original * half
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    in_name,
                    (1, in_cols_trt),
                    (batch_size_original, in_cols_trt),
                    (batch_size_original, in_cols_trt),
                )
                config.add_optimization_profile(profile)
                # TensorRT 8.5+ removed builder.build_engine(); use build_serialized_network + deserialize.
                t_build_start = time.perf_counter()
                serialized = builder.build_serialized_network(network, config)
                t_build_end = time.perf_counter()
                if serialized is None:
                    raise RuntimeError("TensorRT build failed")
                print(f"[LER] TensorRT engine built in {t_build_end - t_build_start:.3f}s")
                engine = runtime.deserialize_cuda_engine(serialized)
                if engine is None:
                    raise RuntimeError("TensorRT deserialize failed")
                if dist.rank == 0:
                    with open(engine_path, "wb") as f:
                        f.write(engine.serialize())
                    print(f"[LER] TensorRT engine saved to {engine_path}")
                t_create_context_start = time.perf_counter()
                trt_context = (engine.create_execution_context(), engine)
                t_create_context_end = time.perf_counter()
                print(
                    f"[LER] TensorRT execution context created in {t_create_context_end - t_create_context_start:.3f}s"
                )
                if dist.rank == 0:
                    print(f"[LER] TensorRT engine built from {onnx_path}")
                    inspector = engine.create_engine_inspector()
                    if inspector is not None:
                        layer_info = inspector.get_engine_information(
                            trt.LayerInformationFormat.JSON
                        )
                        import json as _json
                        try:
                            info = _json.loads(layer_info)
                            layers = info.get("Layers", [])
                            precision_counts: dict = {}
                            for layer in layers:
                                prec = layer.get(
                                    "LayerPrecision", layer.get("Precision", "unknown")
                                )
                                precision_counts[prec] = precision_counts.get(prec, 0) + 1
                            print(f"[LER] TensorRT engine layer precisions: {precision_counts}")
                        except Exception:
                            pass
            except ImportError as e:
                raise RuntimeError(
                    "[LER] ONNX_WORKFLOW=2 (EXPORT_AND_USE_TRT) requires tensorrt to be installed. "
                    "Install with: pip install tensorrt"
                ) from e
            except Exception as e:
                if dist.rank == 0:
                    print(f"[LER] TensorRT build/load failed: {e}; falling back to PyTorch.")
                trt_context = None

    num_obs = circuit.num_observables
    assert num_obs == 1, f"Expected 1 observable, got {num_obs}"

    logical_errors = 0
    total_samples = 0

    t_start = time.perf_counter()
    t_model_time = 0.0
    t_pm_time = 0.0
    t_dataloader_s = 0.0
    t_to_device_s = 0.0
    t_postmodel_s = 0.0  # sampling + syndrome + residuals + residual build
    t_cpu_copy_s = 0.0  # .cpu().numpy() + latency_predecoder_rows
    t_post_decode_s = 0.0  # as_tensor, final_L, gt_obs, logical_errors
    for batch_idx, batch in enumerate(test_dataloader):
        # print(f"Batch {batch_idx} of {len(test_dataloader)}")
        _t0 = time.perf_counter()
        batch = dict_to_device(batch, device)
        t_to_device_s += time.perf_counter() - _t0

        dets_and_obs = batch["dets_and_obs"]
        dets_only = dets_and_obs[:, :-num_obs].contiguous()
        B = dets_and_obs.shape[0]

        t_model_start = time.perf_counter()
        if trt_context is not None:
            context, engine = trt_context
            dets = dets_only.to(torch.uint8).contiguous()
            B, in_cols = dets.shape
            context.set_input_shape("dets", (B, in_cols))
            out_shape = tuple(context.get_tensor_shape("L_and_residual_dets"))
            L_and_residual_dets = torch.empty(out_shape, device=device, dtype=torch.uint8)
            bindings = [
                int(dets.data_ptr()),
                int(L_and_residual_dets.data_ptr()),
            ]
            t_execute_start = time.perf_counter()
            context.execute_v2(bindings=bindings)
            t_execute_end = time.perf_counter()
            if batch_idx == 0 and dist.rank == 0:
                print(
                    f"[LER] TensorRT first batch executed in {t_execute_end - t_execute_start:.3f}s"
                )
        else:
            L_and_residual_dets = pipeline_module(dets_only)
        pre_L = L_and_residual_dets[:, 0].to(torch.int32)
        residual = L_and_residual_dets[:, 1:].to(torch.int32)
        t_model_time += time.perf_counter() - t_model_start
        if residual.shape[1] != det_model.num_detectors:
            raise ValueError(
                f"Residual shape {residual.shape} != DEM detectors {det_model.num_detectors}. "
                f"Check interleave order for basis '{basis}' and time slicing."
            )
        _t_cpu = time.perf_counter()
        residual_np = residual.cpu().numpy()
        if dist.rank == 0 and latency_predecoder_rows is not None and len(
            latency_predecoder_rows
        ) < latency_num_samples:
            remaining = latency_num_samples - len(latency_predecoder_rows)
            take = min(int(residual_np.shape[0]), remaining)
            for i in range(take):
                latency_predecoder_rows.append(residual_np[i].copy())
        t_cpu_copy_s += time.perf_counter() - _t_cpu
        t_dem_decode = time.perf_counter()
        pred_obs = matcher.decode_batch(residual_np)
        t_pm_time += time.perf_counter() - t_dem_decode
        dem_decode_s += time.perf_counter() - t_dem_decode
        _t_postdecode = time.perf_counter()
        pred_obs = torch.as_tensor(pred_obs, dtype=torch.long, device=device)
        pred_obs = pred_obs.view(-1).contiguous()
        final_L = (pre_L + pred_obs).remainder_(2)
        gt_obs = dets_and_obs[:, -num_obs:].view(-1).contiguous()
        logical_errors += int((final_L != gt_obs).sum().item())
        total_samples += B
        t_post_decode_s += time.perf_counter() - _t_postdecode

    # Dataloader time = time spent waiting for next batch (iteration overhead)
    num_batches = batch_idx + 1 if "batch_idx" in locals() else 0
    t_end = time.perf_counter()
    t_dataloader_s = (t_end - t_start) - (
        t_to_device_s + t_model_time + t_postmodel_s + t_cpu_copy_s + t_pm_time + t_post_decode_s
    )
    # Disable detailed printing for now
    if False:
        print(f"Time taken for total_samples={total_samples}: {t_end - t_start:.3f}s")
        print(f"  Model: {t_model_time:.3f}s  PyMatching: {t_pm_time:.3f}s")
        print(
            f"  ToDevice: {t_to_device_s:.3f}s  PostModel (syn+residual): {t_postmodel_s:.3f}s  CPU copy: {t_cpu_copy_s:.3f}s  PostDecode: {t_post_decode_s:.3f}s  Dataloader (est.): {t_dataloader_s:.3f}s"
        )
    if dist.world_size > 1:
        t_log = torch.tensor(logical_errors, device=device, dtype=torch.long)
        t_n = torch.tensor(total_samples, device=device, dtype=torch.long)
        t_pymatch = torch.tensor(num_pymatch_errors, device=device, dtype=torch.long)
        torch.distributed.all_reduce(t_log, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_n, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_pymatch, op=torch.distributed.ReduceOp.SUM)
        logical_errors = int(t_log.item())
        total_samples = int(t_n.item())
        num_pymatch_errors = int(t_pymatch.item())

    # Latency: single-shot (batch_size=1, matcher.decode) on a small subset on rank 0 only,
    # timed after the main loop for a clean CPU state.
    baseline_us_per_round = float("nan")
    predecoder_us_per_round = float("nan")
    if dist.rank == 0 and latency_baseline_rows is not None and latency_predecoder_rows is not None:
        baseline_us_per_round, predecoder_us_per_round = _time_single_shot_latency_stim(
            matcher=matcher,
            baseline_syndromes=np.asarray(latency_baseline_rows, dtype=np.uint8),
            residual_syndromes=np.asarray(latency_predecoder_rows, dtype=np.uint8),
            n_rounds=int(cfg.n_rounds),
            warmup_iterations=50,
        )

    if dist.rank == 0:
        LAST_DEM_TIMING.update(
            {
                "dem_build_s": float(dem_build_s),
                "dem_decode_s": float(dem_decode_s),
                "basis": basis,
                "num_batches": int(num_batches),
            }
        )
        print(
            f"[DEM Timing] build={dem_build_s:.2f}s decode={dem_decode_s:.2f}s "
            f"(basis={basis}, batches={num_batches})"
        )

    return logical_errors, total_samples, num_pymatch_errors, baseline_us_per_round, predecoder_us_per_round


@torch.inference_mode()
def compute_syndrome_density_reduction(model, device, dist, cfg) -> dict:
    """
    Applies a trained model to compute the reduction in syndrome density.
    Aggregates properly across batches and ranks.
    """

    import random
    import numpy as np

    th_data = float(getattr(cfg.test, "th_data", 0.0))
    th_syn = float(getattr(cfg.test, "th_syn", 0.0))

    # Sampling mode configuration
    sampling_mode = str(getattr(cfg.test, "sampling_mode", "threshold")).lower()
    temperature = float(getattr(cfg.test, "temperature", 1.0))
    temperature_data = getattr(cfg.test, "temperature_data", None)
    temperature_syn = getattr(cfg.test, "temperature_syn", None)
    temperature_data = float(temperature_data) if temperature_data is not None else temperature
    temperature_syn = float(temperature_syn) if temperature_syn is not None else temperature

    rank = dist.rank if dist else 0
    world_size = dist.world_size if dist else 1

    # ===== NEW: choose ablation mode =====
    mode = str(getattr(cfg.test, "syn_red", "full")).lower().replace("-", "_")
    # Quick-validation escape hatch (CI/fast runs only). Full validation should keep SDR enabled.
    if mode == "none":
        if rank == 0:
            print("[Syndrome Density] syn_red=none; skipping computation.")
        return None
    if mode not in {"full", "syn_only", "s_only"}:
        print(f"[warn] Unknown cfg.test.syn_red={mode!r}; using 'full'.")
        mode = "full"

    # Calculate samples per GPU (divide total across GPUs)
    total_samples = int(cfg.test.num_samples)
    samples_per_gpu = total_samples // world_size

    # ----- Distributed: Each GPU generates its share of samples with rank-specific seed -----
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # Set rank-specific seed so each GPU generates DIFFERENT samples
        rank_seed = 54321 + rank * 1000
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        np.random.seed(rank_seed)
        random.seed(rank_seed)

        if rank == 0:
            if world_size > 1:
                print(
                    f"[Syndrome Density] Distributed: {world_size} GPUs, {total_samples} total samples"
                )
                print(f"[Syndrome Density] Samples per GPU: {samples_per_gpu}")

        # ----- Data: Create datapipe with reduced sample count per GPU -----
        from copy import deepcopy
        cfg_copy = deepcopy(cfg)
        cfg_copy.test.num_samples = samples_per_gpu
        test_dataset = DatapipeFactory.create_datapipe_inference(cfg_copy)

        if rank == 0:
            print(f"[Syndrome Density] Each GPU processes {len(test_dataset)} samples")
    finally:
        # Restore original random state (including CUDA)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

    # --- DataLoader: NO DistributedSampler - each GPU processes ALL of its own samples ---
    test_loader_kwargs = dict(cfg.test.dataloader)
    # Allow env override for SDR DataLoader workers (container/shm safety).
    try:
        override_workers = (
            os.environ.get("PREDECODER_SDR_NUM_WORKERS") or
            os.environ.get("PREDECODER_EVAL_NUM_WORKERS") or
            os.environ.get("PREDECODER_INFERENCE_NUM_WORKERS")
        )
        if override_workers is not None:
            test_loader_kwargs["num_workers"] = int(override_workers)
        else:
            is_container = os.path.exists("/.dockerenv")
            if is_container and int(test_loader_kwargs.get("num_workers", 0)) > 0:
                shm_bytes = _detect_shm_bytes()
                if shm_bytes is not None and shm_bytes < 1_000_000_000:
                    test_loader_kwargs["num_workers"] = 0
                    try:
                        if rank == 0:
                            print(
                                f"[Syndrome Density] Detected small /dev/shm "
                                f"({shm_bytes / (1024 ** 2):.1f} MiB); "
                                "setting num_workers=0. "
                                "Override with PREDECODER_INFERENCE_NUM_WORKERS."
                            )
                    except Exception:
                        pass
    except Exception:
        pass
    if int(test_loader_kwargs.get("num_workers", 0)) == 0:
        test_loader_kwargs.pop("prefetch_factor", None)
        if test_loader_kwargs.get("persistent_workers", False):
            test_loader_kwargs["persistent_workers"] = False
    else:
        test_loader_kwargs.setdefault("multiprocessing_context", "spawn")

    if torch.cuda.is_available():
        test_loader_kwargs["pin_memory"] = True

    # No DistributedSampler - each GPU processes ALL of its own samples
    test_dataloader = DataLoader(test_dataset, shuffle=False, **test_loader_kwargs)

    # ----- Parity check matrices (build once) -----
    code_rotation = getattr(cfg.data, 'code_rotation', 'XV')
    maps = _build_stab_maps(cfg.distance, code_rotation)
    Hx_idx = maps["Hx_idx"].to(device)
    Hx_mask = maps["Hx_mask"].to(device)
    Hz_idx = maps["Hz_idx"].to(device)
    Hz_mask = maps["Hz_mask"].to(device)
    stab_indices_x = maps["stab_x"].to(device=device, dtype=torch.long)
    stab_indices_z = maps["stab_z"].to(device=device, dtype=torch.long)

    # ----- Accumulators -----
    in_ones_X = torch.tensor(0, dtype=torch.int64, device=device)
    in_elems_X = torch.tensor(0, dtype=torch.int64, device=device)
    res_ones_X = torch.tensor(0, dtype=torch.int64, device=device)

    in_ones_Z = torch.tensor(0, dtype=torch.int64, device=device)
    in_elems_Z = torch.tensor(0, dtype=torch.int64, device=device)
    res_ones_Z = torch.tensor(0, dtype=torch.int64, device=device)

    model.eval()
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    try:
        model = model.to(memory_format=torch.channels_last_3d)
    except Exception:
        pass

    if device.type == "cuda":
        data_iter = CUDAPrefetcher(test_dataloader, device)
    else:
        data_iter = iter(test_dataloader)

    use_autocast = (device.type == "cuda") and bool(cfg.enable_fp16)

    batch_count = 0
    t_start = time.perf_counter()
    for sample_batched in data_iter:
        batch_count += 1
        sample_batched = move_to_device(sample_batched, device, non_blocking=True)

        # === Inputs ===
        x_syn_diff = sample_batched["x_syn_diff"].to(dtype=torch.int32)  # (B,Sx,T)
        z_syn_diff = sample_batched["z_syn_diff"].to(dtype=torch.int32)  # (B,Sz,T)
        trainX = sample_batched["trainX"]  # (B,4,T,D,D)

        # Basis masks (from first-round presence maps)
        x_present_first = trainX[:, 2, 0]
        z_present_first = trainX[:, 3, 0]
        mask_X = (z_present_first == 0).view(z_present_first.size(0), -1).all(dim=1)  # (B,)
        mask_Z = (x_present_first == 0).view(x_present_first.size(0), -1).all(dim=1)  # (B,)

        try:
            trainX = trainX.contiguous(memory_format=torch.channels_last_3d)
        except Exception:
            pass

        # --- count input density ---
        # IMPORTANT: Use test.meas_basis_test, not meas_basis (training config)
        test_meas_basis = str(getattr(cfg.test, "meas_basis_test", cfg.meas_basis)).lower()

        # (Keep logs minimal: no first-batch debug prints here.)

        if test_meas_basis == 'x':
            in_ones_X += x_syn_diff.sum(dtype=torch.int64)
            in_elems_X += torch.tensor(x_syn_diff.numel(), device=device, dtype=torch.int64)
        elif test_meas_basis == 'z':
            in_ones_Z += z_syn_diff.sum(dtype=torch.int64)
            in_elems_Z += torch.tensor(z_syn_diff.numel(), device=device, dtype=torch.int64)
        elif test_meas_basis in ('both', 'mixed'):
            if mask_X.any():
                x_in = x_syn_diff[mask_X]
                in_ones_X += x_in.sum(dtype=torch.int64)
                in_elems_X += torch.tensor(x_in.numel(), device=device, dtype=torch.int64)
            if mask_Z.any():
                z_in = z_syn_diff[mask_Z]
                in_ones_Z += z_in.sum(dtype=torch.int64)
                in_elems_Z += torch.tensor(z_in.numel(), device=device, dtype=torch.int64)
        else:
            raise ValueError(f"Unsupported measurement basis: {cfg.meas_basis}")

        # === Model forward ===
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
            logits = model(trainX)

        B, _, T, D, _ = logits.shape
        D2 = D * D

        if test_meas_basis == 'x':
            # channels: 0=z_data_corr, 2=syn_x_grid
            z_data_corr = sample_predictions(logits[:, 0], th_data, sampling_mode, temperature_data)
            syn_x_grid = sample_predictions(logits[:, 2], th_syn, sampling_mode, temperature_syn)

            # S_X from z_data_corr
            z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T)
            Kx = Hx_idx.size(1)
            z_exp = z_flat.unsqueeze(2).expand(B, D2, Kx, T)
            hx_idx_e = Hx_idx.clamp_min(0).view(1, -1, Kx, 1).expand(B, -1, -1, T)
            g_x = z_exp.gather(dim=1, index=hx_idx_e)
            m_x = Hx_mask.view(1, -1, Kx, 1).expand_as(g_x)
            S_X = (g_x.masked_fill(~m_x, 0).sum(dim=2) & 1)  # (B,Sx,T)

            syn_x_flat = map_grid_to_stabilizer_tensor(syn_x_grid,
                                                       stab_indices_x).to(torch.int32)  # (B,Sx,T)

            # ===== NEW: ablation =====
            if mode == "syn_only":
                S_X = torch.zeros_like(S_X)
            elif mode == "s_only":
                syn_x_flat = torch.zeros_like(syn_x_flat)

            # Residuals
            R_X = torch.empty_like(x_syn_diff, dtype=torch.int32)
            R_X[:, :, 0] = (x_syn_diff[:, :, 0] + syn_x_flat[:, :, 0] + S_X[:, :, 0]) & 1
            if T > 1:
                R_X[:, :, 1:] = (
                    x_syn_diff[:, :, 1:] + syn_x_flat[:, :, 1:] + syn_x_flat[:, :, :-1] +
                    S_X[:, :, 1:]
                ) & 1

            res_ones_X += R_X.sum(dtype=torch.int64)

        elif test_meas_basis == 'z':
            # channels: 1=x_data_corr, 3=syn_z_grid
            x_data_corr = sample_predictions(logits[:, 1], th_data, sampling_mode, temperature_data)
            syn_z_grid = sample_predictions(logits[:, 3], th_syn, sampling_mode, temperature_syn)

            # S_Z from x_data_corr
            x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T)
            Kz = Hz_idx.size(1)
            x_exp = x_flat.unsqueeze(2).expand(B, D2, Kz, T)
            hz_idx_e = Hz_idx.clamp_min(0).view(1, -1, Kz, 1).expand(B, -1, -1, T)
            g_z = x_exp.gather(dim=1, index=hz_idx_e)
            m_z = Hz_mask.view(1, -1, Kz, 1).expand_as(g_z)
            S_Z = (g_z.masked_fill(~m_z, 0).sum(dim=2) & 1)  # (B,Sz,T)

            syn_z_flat = map_grid_to_stabilizer_tensor(syn_z_grid, stab_indices_z).to(torch.int32)

            # ===== NEW: ablation =====
            if mode == "syn_only":
                S_Z = torch.zeros_like(S_Z)
            elif mode == "s_only":
                syn_z_flat = torch.zeros_like(syn_z_flat)

            R_Z = torch.empty_like(z_syn_diff, dtype=torch.int32)
            R_Z[:, :, 0] = (z_syn_diff[:, :, 0] + syn_z_flat[:, :, 0] + S_Z[:, :, 0]) & 1
            if T > 1:
                R_Z[:, :, 1:] = (
                    z_syn_diff[:, :, 1:] + syn_z_flat[:, :, 1:] + syn_z_flat[:, :, :-1] +
                    S_Z[:, :, 1:]
                ) & 1

            res_ones_Z += R_Z.sum(dtype=torch.int64)

        elif test_meas_basis in ('both', 'mixed'):
            # channels: 0=z_data_corr, 1=x_data_corr, 2=syn_x_grid, 3=syn_z_grid
            z_data_corr = sample_predictions(logits[:, 0], th_data, sampling_mode, temperature_data)
            x_data_corr = sample_predictions(logits[:, 1], th_data, sampling_mode, temperature_data)
            syn_x_grid = sample_predictions(logits[:, 2], th_syn, sampling_mode, temperature_syn)
            syn_z_grid = sample_predictions(logits[:, 3], th_syn, sampling_mode, temperature_syn)

            # S_X
            z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T)
            Kx = Hx_idx.size(1)
            z_exp = z_flat.unsqueeze(2).expand(B, D2, Kx, T)
            hx_idx_e = Hx_idx.clamp_min(0).view(1, -1, Kx, 1).expand(B, -1, -1, T)
            g_x = z_exp.gather(dim=1, index=hx_idx_e)
            m_x = Hx_mask.view(1, -1, Kx, 1).expand_as(g_x)
            S_X = (g_x.masked_fill(~m_x, 0).sum(dim=2) & 1)

            # S_Z
            x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T)
            Kz = Hz_idx.size(1)
            x_exp = x_flat.unsqueeze(2).expand(B, D2, Kz, T)
            hz_idx_e = Hz_idx.clamp_min(0).view(1, -1, Kz, 1).expand(B, -1, -1, T)
            g_z = x_exp.gather(dim=1, index=hz_idx_e)
            m_z = Hz_mask.view(1, -1, Kz, 1).expand_as(g_z)
            S_Z = (g_z.masked_fill(~m_z, 0).sum(dim=2) & 1)

            # syn maps
            syn_x_flat = map_grid_to_stabilizer_tensor(syn_x_grid, stab_indices_x).to(torch.int32)
            syn_z_flat = map_grid_to_stabilizer_tensor(syn_z_grid, stab_indices_z).to(torch.int32)

            # ===== NEW: ablation =====
            if mode == "syn_only":
                S_X = torch.zeros_like(S_X)
                S_Z = torch.zeros_like(S_Z)
            elif mode == "s_only":
                syn_x_flat = torch.zeros_like(syn_x_flat)
                syn_z_flat = torch.zeros_like(syn_z_flat)

            # Residuals
            R_X = torch.empty_like(x_syn_diff, dtype=torch.int32)
            R_X[:, :, 0] = (x_syn_diff[:, :, 0] + syn_x_flat[:, :, 0] + S_X[:, :, 0]) & 1
            if T > 1:
                R_X[:, :, 1:] = (
                    x_syn_diff[:, :, 1:] + syn_x_flat[:, :, 1:] + syn_x_flat[:, :, :-1] +
                    S_X[:, :, 1:]
                ) & 1

            R_Z = torch.empty_like(z_syn_diff, dtype=torch.int32)
            R_Z[:, :, 0] = (z_syn_diff[:, :, 0] + syn_z_flat[:, :, 0] + S_Z[:, :, 0]) & 1
            if T > 1:
                R_Z[:, :, 1:] = (
                    z_syn_diff[:, :, 1:] + syn_z_flat[:, :, 1:] + syn_z_flat[:, :, :-1] +
                    S_Z[:, :, 1:]
                ) & 1

            if mask_X.any():
                res_ones_X += R_X[mask_X].sum(dtype=torch.int64)
            if mask_Z.any():
                res_ones_Z += R_Z[mask_Z].sum(dtype=torch.int64)

    t_end = time.perf_counter()
    # Disable detailed printing for now
    if False:
        print(f"Time taken for batch_count={batch_count}: {t_end - t_start:.3f}s")

    # ----- All-reduce across ranks -----
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in (in_ones_X, in_elems_X, res_ones_X, in_ones_Z, in_elems_Z, res_ones_Z):
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)

    # Debug: Print raw counts (only on rank 0)
    rank = dist.rank if dist else 0
    if rank == 0:
        print(f"\n[Syndrome Density Debug]")
        print(f"  Processed {batch_count} batches")
        print(
            f"  X-basis: in_ones={in_ones_X.item()}, in_elems={in_elems_X.item()}, res_ones={res_ones_X.item()}"
        )
        print(
            f"  Z-basis: in_ones={in_ones_Z.item()}, in_elems={in_elems_Z.item()}, res_ones={res_ones_Z.item()}"
        )

    def safe_ratio(num, den):
        return (num.float() /
                den.float()) if den.item() > 0 else torch.tensor(float('nan'), device=device)

    input_density_X = safe_ratio(in_ones_X, in_elems_X)
    residual_density_X = safe_ratio(res_ones_X, in_elems_X)
    input_density_Z = safe_ratio(in_ones_Z, in_elems_Z)
    residual_density_Z = safe_ratio(res_ones_Z, in_elems_Z)

    as_float = lambda t: float(t.item())

    def finite_pos(x: torch.Tensor) -> bool:
        return torch.isfinite(x).item() and (x > 0).item()

    if test_meas_basis in ('both', 'mixed'):
        # Calculate reduction factors, checking both numerator and denominator
        reduction_x = float('nan')
        if finite_pos(input_density_X) and finite_pos(residual_density_X):
            reduction_x = as_float(input_density_X / residual_density_X)
        elif finite_pos(input_density_X) and residual_density_X.item() == 0:
            # Perfect correction: input had syndromes, output has none
            reduction_x = float('inf')

        reduction_z = float('nan')
        if finite_pos(input_density_Z) and finite_pos(residual_density_Z):
            reduction_z = as_float(input_density_Z / residual_density_Z)
        elif finite_pos(input_density_Z) and residual_density_Z.item() == 0:
            # Perfect correction: input had syndromes, output has none
            reduction_z = float('inf')

        result = {
            "ablation": mode,
            "input syndrome density (X stabs)": as_float(input_density_X),
            "residual syndrome density (X stabs)": as_float(residual_density_X),
            "input syndrome density (Z stabs)": as_float(input_density_Z),
            "residual syndrome density (Z stabs)": as_float(residual_density_Z),
            "reduction factor (X)": reduction_x,
            "reduction factor (Z)": reduction_z,
        }
    else:
        if test_meas_basis == 'x':
            reduction_x = float('nan')
            if finite_pos(input_density_X) and finite_pos(residual_density_X):
                reduction_x = as_float(input_density_X / residual_density_X)
            elif finite_pos(input_density_X) and residual_density_X.item() == 0:
                reduction_x = float('inf')

            result = {
                "ablation": mode,
                "input syndrome density (meas_basis=X)": as_float(input_density_X),
                "residual syndrome density (meas_basis=X)": as_float(residual_density_X),
                "reduction factor (X)": reduction_x,
            }
        else:  # test_meas_basis == 'z'
            reduction_z = float('nan')
            if finite_pos(input_density_Z) and finite_pos(residual_density_Z):
                reduction_z = as_float(input_density_Z / residual_density_Z)
            elif finite_pos(input_density_Z) and residual_density_Z.item() == 0:
                reduction_z = float('inf')

            result = {
                "ablation": mode,
                "input syndrome density (meas_basis=Z)": as_float(input_density_Z),
                "residual syndrome density (meas_basis=Z)": as_float(residual_density_Z),
                "reduction factor (Z)": reduction_z,
            }

    return result
