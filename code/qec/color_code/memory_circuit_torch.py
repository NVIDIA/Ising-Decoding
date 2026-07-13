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
Torch-only color-code memory-circuit consumer of precomputed augmented DEM bundles.

This module replaces the legacy color-code memory-circuit simulator. It
consumes the `color_d{d}_r{r}_{basis}_{schedule}_augmented_dem.{H,p}.npz` bundle
produced by `qec.precompute_dem.precompute_dem_bundle_color_code` and produces
training batches in the same `(trainX, trainY)` contract as the legacy path:

    trainX: (B, 4, n_rounds, n_rows, n_cols) uint8 / float32
            channels = [x_syn, z_syn, x_present, z_present]
    trainY: (B, 4, n_rounds, n_rows, n_cols) uint8 / float32
            channels = [z_err, x_err, s1s2_x, s1s2_z]

Sampling runs on the GPU via `qec.dem_sampling.dem_sampling` (cuStabilizer
BitMatrixSampler), then error labels are canonicalized with the Torch spacelike
color-code HE implementation. Timelike color-code HE is deliberately out of
scope for this path.

Limitations (v1):
    - The augmented DEM *structure* is precomputed at a single nominal `p`, but the
      per-error probabilities follow the *configured* noise level (a scalar `p` or a
      `noise_model`) rather than the bundle's baked-in p: the bundle's structure is
      reused and the probability vector is rebuilt at the configured p. Each circuit
      instance still has a single fixed p, so batch-wise p sweeps need multiple
      instances.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from qec.color_code.color_code import ColorCode
from qec.color_code.homological_equivalence_torch import (
    apply_homological_equivalence_color_torch,
    build_color_spacelike_he_cache,
)
from qec.dem_sampling import dem_sampling
from qec.precompute_dem import (
    ColorAugmentedDemBundle,
    build_probability_vector_color_code,
    load_color_augmented_dem_bundle,
)
from qec.noise_model import get_grouped_totals


class ColorMemoryCircuitTorch:
    """Torch+cuStabilizer consumer of color-code augmented DEM bundles.

    See module docstring for the (trainX, trainY) contract this class produces.

    ``p_scalar`` is the configured nominal physical error rate: the bundle supplies
    the DEM *structure*, but the per-error probabilities are rebuilt at this p so the
    configured noise wins over the bundle's baked-in p. ``None`` => legacy behaviour
    (trust the bundle's ``p_nominal``). When a ``noise_model`` is given the per-fault
    probabilities come from the model and ``active_p_nominal`` follows its grouped
    totals, but the configured ``p_scalar`` still drives the nominal fields
    (``p_nominal``/``p_min``/``p_max``).
    """

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        basis: str,
        schedule: str = "nearest-neighbor",
        precomputed_frames_dir: str,
        enable_z_feedforward: bool = True,
        apply_data_x_override: bool = True,
        apply_spacelike_he: bool = True,
        he_max_iterations: int = 16,
        use_coset_search: bool = False,
        device: Optional[torch.device] = None,
        strict_metadata: bool = True,
        noise_model=None,
        p_scalar: Optional[float] = None,
    ) -> None:
        if device is None:
            # Concrete device index (see ColorQCDataGeneratorTorch): avoids index-less
            # "cuda" placing tensors on cuda:0 in background data-gen threads under DDP.
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device("cpu")
            )

        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.basis = str(basis).upper()
        if self.basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
        self.schedule = str(schedule)
        self.enable_z_feedforward = bool(enable_z_feedforward)
        self.apply_data_x_override = bool(apply_data_x_override)
        self.apply_spacelike_he = bool(apply_spacelike_he)
        self.he_max_iterations = int(he_max_iterations)
        self.use_coset_search = bool(use_coset_search)
        self.device = device

        loaded = load_color_augmented_dem_bundle(
            precomputed_frames_dir,
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=self.basis,
            schedule=self.schedule,
            device=device,
            enable_z_feedforward=self.enable_z_feedforward,
            apply_data_x_override=self.apply_data_x_override,
            use_decomposed_errors=False,
            strict_metadata=strict_metadata,
        )
        self.bundle: ColorAugmentedDemBundle = loaded["bundle"]
        self.p: torch.Tensor = loaded["p"]
        self.bundle_p_nominal: float = float(loaded["p_nominal"])
        # Config p wins, mirroring the surface path (see QCDataGeneratorTorch): a
        # precomputed bundle supplies the expensive cached DEM *structure*, but the
        # per-error probabilities are cheap and must follow the *configured* noise
        # level, not whatever p the bundle happened to be built at. ``p_scalar`` is
        # that configured nominal p (None => legacy: trust the bundle's p_nominal).
        # Without this, pointing training at a bundle built at a different p would
        # silently train at the bundle's p and ignore the configured p_min/p_max.
        cfg_p: Optional[float] = float(p_scalar) if p_scalar is not None else None
        effective_p_nominal: float = cfg_p if cfg_p is not None else self.bundle_p_nominal
        self.p_nominal: float = effective_p_nominal
        self.active_p_nominal: float = effective_p_nominal
        self.p_min: float = effective_p_nominal
        self.p_max: float = effective_p_nominal
        self.fixed_p: bool = True
        self.metadata = loaded["metadata"]
        self.noise_model = noise_model
        # Set True iff the configured scalar p overrode the bundle's baked-in p (the
        # probability vector was rebuilt at the configured p). The override is
        # surfaced -- without per-rank/per-basis print spam -- by
        # ColorQCDataGeneratorTorch, which announces it once under verbose+rank0.
        self.p_rebuilt_from_config_p: bool = False

        if noise_model is not None:
            p_np = build_probability_vector_color_code(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=self.basis,
                schedule=self.schedule,
                p_scalar=effective_p_nominal,
                noise_model=noise_model,
            )
            self.p = torch.from_numpy(p_np).to(device=device, dtype=torch.float32)
            try:
                self.active_p_nominal = float(get_grouped_totals(noise_model)["max_group"])
            except Exception:
                self.active_p_nominal = float(self.p.max().item()) if self.p.numel() else 0.0
        elif cfg_p is not None and abs(cfg_p - self.bundle_p_nominal) > 1e-12:
            # Scalar path: the configured p differs from the bundle's baked-in p.
            # Rebuild the probability vector at the configured p (reusing the bundle's
            # structure) so the config wins. The override is flagged here and announced
            # once by ColorQCDataGeneratorTorch (verbose+rank0) so it is never silent
            # (this was previously a silent train/eval noise mismatch) without spamming
            # the log once per rank / per basis.
            p_np = build_probability_vector_color_code(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=self.basis,
                schedule=self.schedule,
                p_scalar=cfg_p,
                noise_model=None,
            )
            self.p = torch.from_numpy(p_np).to(device=device, dtype=torch.float32)
            self.p_rebuilt_from_config_p = True

        # Cache geometry for grid scattering and presence masks.
        cc = ColorCode(self.distance)
        self._he_cache = (
            build_color_spacelike_he_cache(cc, device=device) if self.apply_spacelike_he else None
        )
        self.n_rows = int(cc.n_rows)
        self.n_cols = int(cc.n_cols)
        self.num_data = int(self.bundle.num_data)
        self.num_z = int(self.bundle.num_z)
        self.num_x = int(self.bundle.num_x)
        self.num_meas = int(self.bundle.num_meas)
        if self.num_meas != self.num_z + self.num_x:
            raise RuntimeError(
                f"Inconsistent bundle: num_meas={self.num_meas} != num_z+num_x="
                f"{self.num_z + self.num_x}"
            )

        stab_flat = cc.get_syndrome_grid_indices().astype(np.int64)
        self._stab_flat_idx = torch.tensor(stab_flat, dtype=torch.long, device=device)

        data_flat = np.array(
            [
                int(cc.qubit_to_grid[int(q)][0]) * self.n_cols + int(cc.qubit_to_grid[int(q)][1])
                for q in cc.data_qubits
            ],
            dtype=np.int64,
        )
        self._data_flat_idx = torch.tensor(data_flat, dtype=torch.long, device=device)

        pres = torch.zeros(self.n_rows * self.n_cols, dtype=torch.float32, device=device)
        pres[self._stab_flat_idx] = 1.0
        self._stab_present_flat = pres

    def generate_batch(
        self,
        batch_size: int,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one training batch from the augmented DEM bundle.

        Returns (trainX, trainY), both `(B, 4, n_rounds, n_rows, n_cols)`
        float32 on `self.device`.
        """
        B = int(batch_size)
        H = self.bundle.H
        p = self.p

        device_id = None
        if self.device.type == "cuda":
            device_index = self.device.index
            device_id = int(torch.cuda.current_device() if device_index is None else device_index)
        outcomes = dem_sampling(H, p, B, device_id=device_id, seed=seed)
        outcomes = outcomes.to(self.device)

        R = self.n_rounds
        Nd = self.num_data
        Nm = self.num_meas
        f_end = self.bundle.frame_rows
        mo_end = f_end + self.bundle.meas_old_rows

        frame_part = outcomes[:, :f_end].reshape(B, R, Nd, 2)
        meas_old = outcomes[:, f_end:mo_end].reshape(B, R, Nm)
        meas_new = outcomes[:, mo_end:].reshape(B, R, Nm)

        return self._format_for_model(frame_part, meas_old, meas_new)

    def _format_for_model(
        self,
        predecoder: torch.Tensor,
        meas_old: torch.Tensor,
        meas_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match the legacy `(trainX, trainY)` contract.

        `predecoder`: (B, R, num_data, 2) — cumulative X/Z frames on data qubits
        `meas_old`  : (B, R, num_meas)   — [Z checks..., X checks...] per round
        `meas_new`  : (B, R, num_meas)   — same ordering; s2 - s1 deltas
        """
        B, R, Nd, _ = predecoder.shape
        num_z = self.num_z

        mz = meas_old[:, :, :num_z].to(torch.uint8)
        mx = meas_old[:, :, num_z:].to(torch.uint8)
        mz_pad = torch.cat([torch.zeros_like(mz[:, :1, :]), mz], dim=1)
        mx_pad = torch.cat([torch.zeros_like(mx[:, :1, :]), mx], dim=1)
        z_det = (mz_pad[:, 1:, :] ^ mz_pad[:, :-1, :]).to(torch.uint8)
        x_det = (mx_pad[:, 1:, :] ^ mx_pad[:, :-1, :]).to(torch.uint8)

        if self.basis == "X":
            z_det[:, 0, :] = 0
            z_det[:, -1, :] = 0
        else:
            x_det[:, 0, :] = 0
            x_det[:, -1, :] = 0

        s1s2_z = meas_new[:, :, :num_z].to(torch.uint8)
        s1s2_x = meas_new[:, :, num_z:].to(torch.uint8)

        x_cum = predecoder[..., 0].to(torch.uint8)
        z_cum = predecoder[..., 1].to(torch.uint8)
        x_pad_e = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum], dim=1)
        z_pad_e = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum], dim=1)
        x_diff = (x_pad_e[:, 1:, :] ^ x_pad_e[:, :-1, :]).to(torch.uint8)
        z_diff = (z_pad_e[:, 1:, :] ^ z_pad_e[:, :-1, :]).to(torch.uint8)

        if self._he_cache is not None:
            z_diff, x_diff = apply_homological_equivalence_color_torch(
                z_diff,
                x_diff,
                self._he_cache,
                max_iterations=self.he_max_iterations,
                use_coset_search=self.use_coset_search,
            )

        x_syn_grid = self._scatter_stabs(x_det)
        z_syn_grid = self._scatter_stabs(z_det)
        s1s2_x_grid = self._scatter_stabs(s1s2_x)
        s1s2_z_grid = self._scatter_stabs(s1s2_z)
        x_err_grid = self._scatter_data(x_diff)
        z_err_grid = self._scatter_data(z_diff)

        x_pres = self._stab_present_flat.view(1, 1, self.n_rows, self.n_cols).expand(B, R, -1,
                                                                                     -1).clone()
        z_pres = self._stab_present_flat.view(1, 1, self.n_rows, self.n_cols).expand(B, R, -1,
                                                                                     -1).clone()
        if self.basis == "X":
            z_pres[:, 0] = 0
            z_pres[:, -1] = 0
        else:
            x_pres[:, 0] = 0
            x_pres[:, -1] = 0

        trainX = torch.stack([x_syn_grid, z_syn_grid, x_pres, z_pres], dim=1).contiguous()
        trainY = torch.stack([z_err_grid, x_err_grid, s1s2_x_grid, s1s2_z_grid], dim=1).contiguous()
        return trainX, trainY

    def _scatter_stabs(self, vals_brt: torch.Tensor) -> torch.Tensor:
        B, R, _S = vals_brt.shape
        flat = torch.zeros(
            (B, R, self.n_rows * self.n_cols), dtype=torch.float32, device=vals_brt.device
        )
        idx = self._stab_flat_idx.to(vals_brt.device).view(1, 1, -1).expand(B, R, -1)
        flat.scatter_(2, idx, vals_brt.to(torch.float32))
        return flat.view(B, R, self.n_rows, self.n_cols)

    def _scatter_data(self, vals_brd: torch.Tensor) -> torch.Tensor:
        B, R, _D = vals_brd.shape
        flat = torch.zeros(
            (B, R, self.n_rows * self.n_cols), dtype=torch.float32, device=vals_brd.device
        )
        idx = self._data_flat_idx.to(vals_brd.device).view(1, 1, -1).expand(B, R, -1)
        flat.scatter_(2, idx, vals_brd.to(torch.float32))
        return flat.view(B, R, self.n_rows, self.n_cols)
