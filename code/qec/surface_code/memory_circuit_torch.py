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

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import time

import numpy as np
import torch

from qec.dem_sampling import dem_sampling, measure_from_stacked_frames, timelike_syndromes
from qec.surface_code.data_mapping import (
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
    reshape_Xstabilizers_to_grid_vectorized,
    reshape_Zstabilizers_to_grid_vectorized,
)
from qec.surface_code.homological_equivalence_torch import (
    apply_weight1_timelike_homological_equivalence_torch,
    build_spacelike_he_cache,
    build_timelike_he_cache,
    build_weight2_timelike_cache,
    warmup_he_compile,
)
from qec.surface_code.memory_circuit import SurfaceCode


def _npz1(path: Path):
    z = np.load(path)
    # Prefer explicit payload keys; fall back to first non-scalar array.
    for k in ("p", "arr_0"):
        if k in z.files:
            return z[k]
    for k in z.files:
        a = z[k]
        if getattr(a, "ndim", 0) > 0 and a.size > 1:
            return a
    return z[z.files[0]]


class MemoryCircuitTorch:
    """
    Torch-only data generator using precomputed error→frame matrices (H) and per-error probabilities (p).

    Required precomputed artifacts in `precomputed_frames_dir` (basis-specific):
    - `surface_d{d}_r{r}_{basis}_frame_predecoder.X.npz`  : Hx (num_detectors, num_errors)
    - `surface_d{d}_r{r}_{basis}_frame_predecoder.Z.npz`  : Hz (num_detectors, num_errors)
    - `surface_d{d}_r{r}_{basis}_frame_predecoder.p.npz`  : p  (num_errors,)
    - optional `surface_d{d}_r{r}_{basis}_frame_predecoder.A.npz` : timelike map A (n_rounds*num_meas, 2*num_detectors)
    """

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        basis: str,
        precomputed_frames_dir: str | None = None,
        code_rotation: str = "XV",
        timelike_he: bool = True,
        num_he_cycles: int = 1,
        max_passes_w1: int = 32,
        device: Optional[torch.device] = None,
        use_compile: bool = False,
        compile_chunk_size: int = 2,
        compute_dtype: Optional[torch.dtype] = None,
        use_weight2: bool = False,
        max_passes_w2: int = 4,
        use_coset_search: bool = False,
        coset_max_generators: int = 20,
        use_dense_overlap: bool = False,
        # Optional in-memory DEM artifacts (to avoid writing/loading files).
        H: torch.Tensor | None = None,  # (2*num_detectors, num_errors) uint8
        p: torch.Tensor | None = None,  # (num_errors,) float32
        A: torch.Tensor | None = None,  # (n_rounds*num_meas, 2*num_detectors) uint8
    ):
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.basis = str(basis).upper()
        self.code_rotation = str(code_rotation).upper()
        self.timelike_he = bool(timelike_he)
        self.num_he_cycles = int(num_he_cycles)
        self.max_passes_w1 = int(max_passes_w1)
        self.use_compile = bool(use_compile)
        self.compile_chunk_size = int(compile_chunk_size)
        self.compute_dtype = compute_dtype
        self.use_weight2 = bool(use_weight2)
        self.max_passes_w2 = int(max_passes_w2)
        self.use_coset_search = bool(use_coset_search)
        self.coset_max_generators = int(coset_max_generators)
        self.use_dense_overlap = bool(use_dense_overlap)
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        import threading
        self._compile_thread: threading.Thread | None = None
        if self.timelike_he and self.device.type == "cuda" and self.use_compile:
            self._compile_thread = threading.Thread(
                target=warmup_he_compile,
                kwargs=dict(
                    distance=self.distance,
                    n_rounds=self.n_rounds,
                    basis=self.basis,
                    max_passes_w1=self.max_passes_w1,
                    use_weight2=self.use_weight2,
                    max_passes_w2=self.max_passes_w2,
                ),
                daemon=True,
            )
            self._compile_thread.start()

        # Circuit metadata.
        first_bulk, rot = self.code_rotation[0], self.code_rotation[1]
        self.code = SurfaceCode(
            self.distance,
            first_bulk_syndrome_type=first_bulk,
            rotated_type=("V" if rot == "V" else "H")
        )
        self.data_qubits = torch.as_tensor(
            self.code.data_qubits, dtype=torch.long, device=self.device
        )
        self.xcheck_qubits = torch.as_tensor(
            self.code.xcheck_qubits, dtype=torch.long, device=self.device
        )
        self.zcheck_qubits = torch.as_tensor(
            self.code.zcheck_qubits, dtype=torch.long, device=self.device
        )
        self.nq = int(len(self.code.all_qubits))

        self.meas_qubits = torch.cat([self.xcheck_qubits, self.zcheck_qubits], dim=0)
        self.meas_bases = torch.cat(
            [
                torch.zeros(len(self.xcheck_qubits), dtype=torch.long, device=self.device),
                torch.ones(len(self.zcheck_qubits), dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        # Load DEM artifacts (either from disk or in-memory).
        if H is not None and p is not None:
            self.H = H.to(device=self.device, dtype=torch.uint8)
            self.p = p.to(device=self.device, dtype=torch.float32)
            self.A = (A.to(device=self.device, dtype=torch.uint8) if A is not None else None)
        else:
            if precomputed_frames_dir is None:
                raise ValueError(
                    "Provide either precomputed_frames_dir or in-memory H/p (and optional A)."
                )
            d = Path(precomputed_frames_dir)
            prefix = f"surface_d{self.distance}_r{self.n_rounds}_{self.basis}_frame_predecoder"
            hx_path = d / f"{prefix}.X.npz"
            hz_path = d / f"{prefix}.Z.npz"
            p_path = d / f"{prefix}.p.npz"
            A_path = d / f"{prefix}.A.npz"
            if not (hx_path.exists() and hz_path.exists() and p_path.exists()):
                raise FileNotFoundError(
                    f"Missing DEM artifacts for this basis in {str(d)!r}. Expected:\n"
                    f"  - {hx_path.name}\n  - {hz_path.name}\n  - {p_path.name}\n"
                    f"(Generate via precompute_frames.py with --dem_output_dir {str(d)!r}.)"
                )
            if not A_path.exists():
                A_path = None

            hx = _npz1(hx_path)
            hz = _npz1(hz_path)
            p_arr = _npz1(p_path)
            errors = int(np.asarray(p_arr).reshape(-1).shape[0])
            hx = np.asarray(hx, dtype=np.uint8)
            hz = np.asarray(hz, dtype=np.uint8)
            p_arr = np.asarray(p_arr).reshape(-1)
            hx = hx if hx.shape[1] == errors else hx.T
            hz = hz if hz.shape[1] == errors else hz.T
            self.H = torch.from_numpy(np.concatenate([hx, hz],
                                                     axis=0)).to(self.device, dtype=torch.uint8)
            self.p = torch.from_numpy(p_arr).to(self.device, dtype=torch.float32)
            self.A = (
                torch.from_numpy(np.asarray(_npz1(A_path), dtype=np.uint8)
                                ).to(self.device, dtype=torch.uint8) if A_path else None
            )

        # HE caches (built once).
        self.parity_X = torch.tensor(self.code.hx, dtype=torch.uint8, device=self.device)
        self.parity_Z = torch.tensor(self.code.hz, dtype=torch.uint8, device=self.device)
        self.cache_X_sp = build_spacelike_he_cache(
            self.parity_X, distance=self.distance, basis="X", device=self.device
        )
        self.cache_Z_sp = build_spacelike_he_cache(
            self.parity_Z, distance=self.distance, basis="Z", device=self.device
        )
        self.cache_X_tl = build_timelike_he_cache(self.parity_X)
        self.cache_Z_tl = build_timelike_he_cache(self.parity_Z)

        self.cache_X_w2 = None
        self.cache_Z_w2 = None
        if self.use_weight2:
            self.cache_X_w2 = build_weight2_timelike_cache(
                self.parity_Z, self.parity_Z, self.distance, "X", self.device
            )
            self.cache_Z_w2 = build_weight2_timelike_cache(
                self.parity_X, self.parity_X, self.distance, "Z", self.device
            )

        # Weight maps for trainX presence channels.
        self.w_mapXgrid = (
            normalized_weight_mapping_Xstab_memory(self.distance, rotation=self.code_rotation
                                                  ).reshape(self.distance,
                                                            self.distance).to(self.device)
        )
        self.w_mapZgrid = (
            normalized_weight_mapping_Zstab_memory(self.distance, rotation=self.code_rotation
                                                  ).reshape(self.distance,
                                                            self.distance).to(self.device)
        )

    def generate_batch(
        self,
        *,
        batch_size: int,
        return_aux: bool = False,
        collect_timing: bool = False,
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, dict[str, float]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str,
                                                                                         float]],
    ]:
        """
        Generate a batch of (trainX, trainY).

        - If return_aux=True, also return (meas_old, x_cum, z_cum) for tests that
          build Stim dets_and_obs from circuit-order measurements.
        - If collect_timing=True, also return timing breakdown in milliseconds:
          data generation, HE, format, and total.
        """
        if self._compile_thread is not None:
            # torch.compile warmup can be slow; 20 min cap prevents silent hangs.
            self._compile_thread.join(timeout=1200)
            if self._compile_thread.is_alive():
                raise RuntimeError("warmup_he_compile thread did not finish within 20 min")
            self._compile_thread = None

        if collect_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
        frames_xz = dem_sampling(self.H, self.p, int(batch_size))  # (B, 2*num_detectors)
        meas_old = measure_from_stacked_frames(
            frames_xz, self.meas_qubits, self.meas_bases, nq=self.nq
        )  # (B, R, m)
        meas_new = timelike_syndromes(frames_xz, self.A,
                                      meas_old) if self.A is not None else meas_old.clone()

        # Cumulative data-qubit frames (B, R, D2) from stacked [X|Z] blocks.
        D = frames_xz.shape[1] // 2
        idx_data = (
            torch.arange(self.n_rounds, device=self.device)[:, None] * self.nq +
            self.data_qubits[None, :]
        ).reshape(-1)
        x_cum = frames_xz[:, :D].index_select(1, idx_data).reshape(batch_size, self.n_rounds, -1)
        z_cum = frames_xz[:, D:].index_select(1, idx_data).reshape(batch_size, self.n_rounds, -1)

        if collect_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter()

        if self.timelike_he:
            num_x = int(self.xcheck_qubits.numel())
            s1s2x, s1s2z = meas_new[:, :, :num_x], meas_new[:, :, num_x:]
            mx, mz = meas_old[:, :, :num_x], meas_old[:, :, num_x:]
            mxp = torch.cat([torch.zeros_like(mx[:, :1, :]), mx], dim=1)
            mzp = torch.cat([torch.zeros_like(mz[:, :1, :]), mz], dim=1)
            trainX_x, trainX_z = mxp[:, :-1, :] ^ mxp[:, 1:, :], mzp[:, :-1, :] ^ mzp[:, 1:, :]
            z_diff, x_diff, s1s2x, s1s2z = apply_weight1_timelike_homological_equivalence_torch(
                z_cum,
                x_cum,
                s1s2x,
                s1s2z,
                self.parity_Z,
                self.parity_X,
                self.distance,
                self.num_he_cycles,
                self.max_passes_w1,
                self.basis,
                True,
                trainX_x=trainX_x,
                trainX_z=trainX_z,
                cache_Z_spacelike=self.cache_Z_sp,
                cache_X_spacelike=self.cache_X_sp,
                use_compile=self.use_compile,
                compile_chunk_size=self.compile_chunk_size,
                compute_dtype=self.compute_dtype,
                use_weight2=self.use_weight2,
                max_passes_w2=self.max_passes_w2,
                cache_Z_w2=self.cache_Z_w2,
                cache_X_w2=self.cache_X_w2,
                use_coset_search=self.use_coset_search,
                coset_max_generators=self.coset_max_generators,
                use_dense_overlap=self.use_dense_overlap,
            )
            meas_new = torch.cat([s1s2x, s1s2z], dim=2)
        else:
            # diffs from cumulative
            xpad = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
            zpad = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
            x_diff, z_diff = xpad[:, :-1, :] ^ xpad[:, 1:, :], zpad[:, :-1, :] ^ zpad[:, 1:, :]

        if collect_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = time.perf_counter()

        trainX, trainY = self._format_for_model(x_diff, z_diff, meas_old, meas_new)
        timing_dict = None
        if collect_timing:
            t3 = time.perf_counter()
            timing_dict = {
                "data_gen_ms": (t1 - t0) * 1000,
                "he_ms": (t2 - t1) * 1000,
                "format_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000,
            }

        if return_aux:
            if timing_dict is not None:
                return (trainX, trainY, meas_old, x_cum, z_cum, timing_dict)
            return (trainX, trainY, meas_old, x_cum, z_cum)
        if timing_dict is not None:
            return (trainX, trainY, timing_dict)
        return (trainX, trainY)

    def _format_for_model(self, x_diff, z_diff, meas_old,
                          meas_new) -> tuple[torch.Tensor, torch.Tensor]:
        B, R, D2 = x_diff.shape
        D = self.distance
        num_x = int(self.xcheck_qubits.numel())
        x_raw, z_raw = meas_old[:, :, :num_x], meas_old[:, :, num_x:]
        s1x, s1z = meas_new[:, :, :num_x], meas_new[:, :, num_x:]

        xp = torch.cat([torch.zeros_like(x_raw[:, :1, :]), x_raw], dim=1)
        zp = torch.cat([torch.zeros_like(z_raw[:, :1, :]), z_raw], dim=1)
        x_syn = (xp[:, :-1, :] ^ xp[:, 1:, :]).transpose(1, 2)
        z_syn = (zp[:, :-1, :] ^ zp[:, 1:, :]).transpose(1, 2)
        s1x = s1x.transpose(1, 2)
        s1z = s1z.transpose(1, 2)

        x_syn_g = (
            reshape_Xstabilizers_to_grid_vectorized(x_syn, D, rotation=self.code_rotation
                                                   ).reshape(B, D, D, R).permute(0, 3, 1,
                                                                                 2).contiguous()
        )
        z_syn_g = (
            reshape_Zstabilizers_to_grid_vectorized(z_syn, D, rotation=self.code_rotation
                                                   ).reshape(B, D, D, R).permute(0, 3, 1,
                                                                                 2).contiguous()
        )
        s1x_g = (
            reshape_Xstabilizers_to_grid_vectorized(s1x, D, rotation=self.code_rotation
                                                   ).reshape(B, D, D, R).permute(0, 3, 1,
                                                                                 2).contiguous()
        )
        s1z_g = (
            reshape_Zstabilizers_to_grid_vectorized(s1z, D, rotation=self.code_rotation
                                                   ).reshape(B, D, D, R).permute(0, 3, 1,
                                                                                 2).contiguous()
        )

        x_err = x_diff.reshape(B, R, D, D)
        z_err = z_diff.reshape(B, R, D, D)

        x_pres = self.w_mapXgrid.unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()
        z_pres = self.w_mapZgrid.unsqueeze(0).unsqueeze(0).expand(B, R, D, D).clone()
        if self.basis == "X":
            z_pres[:, 0] = 0
            z_syn_g[:, 0] = 0
            z_pres[:, -1] = 0
            z_syn_g[:, -1] = 0
        else:
            x_pres[:, 0] = 0
            x_syn_g[:, 0] = 0
            x_pres[:, -1] = 0
            x_syn_g[:, -1] = 0

        trainX = torch.stack(
            [x_syn_g.float(), z_syn_g.float(),
             x_pres.float(), z_pres.float()], dim=1
        ).contiguous()
        trainY = torch.stack([z_err.float(),
                              x_err.float(),
                              s1x_g.float(),
                              s1z_g.float()], dim=1).contiguous()
        return trainX, trainY


__all__ = ["MemoryCircuitTorch"]
