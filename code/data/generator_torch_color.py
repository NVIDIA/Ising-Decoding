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
Torch + cuStabilizer training data generator for color codes.

Drop-in replacement for the legacy on-the-fly generator on the color-code path. The actual
sampling lives in `qec.color_code.memory_circuit_torch.ColorMemoryCircuitTorch`,
which consumes the augmented DEM bundle produced by
`qec.precompute_dem.precompute_dem_bundle_color_code`.

Selected by the color-code training path in `train.py`.

Generator contract:
    generator.generate_batch(step, batch_size, return_timing=False, ...)
        -> (trainX, trainY) or (trainX, trainY, timing_dict)
    trainX / trainY: (B, 4, n_rounds, n_rows, n_cols) on CUDA.

v1 limitation: the augmented DEM is fixed-p, so batch-wise p sweeps require
multiple bundles.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Union

import torch

from qec.color_code.memory_circuit_torch import ColorMemoryCircuitTorch


class ColorQCDataGeneratorTorch:
    """Torch + cuStabilizer color-code training data generator."""

    def __init__(
        self,
        *,
        distance: int,
        n_rounds: int,
        schedule: str = "nearest-neighbor",
        measure_basis: str = "both",
        precomputed_frames_dir: str,
        enable_z_feedforward: bool = True,
        apply_data_x_override: bool = True,
        apply_spacelike_he: bool = True,
        he_max_iterations: int = 16,
        use_coset_search: bool = False,
        device: Optional[torch.device] = None,
        rank: int = 0,
        global_rank: Optional[int] = None,
        base_seed: int = 42,
        verbose: bool = False,
        strict_metadata: bool = True,
        noise_model=None,
        p_error: Optional[float] = None,
        p_min: Optional[float] = None,
        p_max: Optional[float] = None,
    ) -> None:
        if device is None:
            # Resolve to a concrete device index (not the index-less "cuda"): data
            # generation runs in a background thread whose per-thread CUDA current
            # device defaults to 0, so an index-less device would place freshly
            # created tensors on cuda:0 while cached tensors sit on the rank's GPU
            # (breaks DDP). Pinning the current device index keeps everything aligned.
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device("cpu")
            )
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.schedule = str(schedule)
        self.measure_basis = str(measure_basis).upper()
        self._mixed = self.measure_basis in ("BOTH", "MIXED")
        self.precomputed_frames_dir = str(precomputed_frames_dir)
        self.enable_z_feedforward = bool(enable_z_feedforward)
        self.apply_data_x_override = bool(apply_data_x_override)
        self.apply_spacelike_he = bool(apply_spacelike_he)
        self.he_max_iterations = int(he_max_iterations)
        self.use_coset_search = bool(use_coset_search)
        self.device = device
        self.rank = int(rank)
        self.global_rank = int(global_rank if global_rank is not None else rank)
        self.base_seed = int(base_seed)
        self._step_seed_offset = (self.global_rank + 1) * 1_000_003
        self.noise_model = noise_model
        # Configured nominal p (mirrors the surface QCDataGeneratorTorch convention:
        # p_error if set, else p_max). Passed to the circuit so the precomputed
        # bundle's probabilities follow the configured noise level rather than the
        # bundle's baked-in p. None => legacy behaviour (trust the bundle's p).
        self._p_scalar: Optional[float] = (
            float(p_error) if p_error is not None else float(p_max) if p_max is not None else None
        )

        def _make(basis: str) -> ColorMemoryCircuitTorch:
            return ColorMemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=basis,
                schedule=self.schedule,
                precomputed_frames_dir=self.precomputed_frames_dir,
                enable_z_feedforward=self.enable_z_feedforward,
                apply_data_x_override=self.apply_data_x_override,
                apply_spacelike_he=self.apply_spacelike_he,
                he_max_iterations=self.he_max_iterations,
                use_coset_search=self.use_coset_search,
                device=self.device,
                strict_metadata=strict_metadata,
                noise_model=noise_model,
                p_scalar=self._p_scalar,
            )

        if self._mixed:
            self.sim_X = _make("X")
            self.sim_Z = _make("Z")
            self.sim = None
        else:
            self.sim = _make(self.measure_basis)
            self.sim_X = self.sim if self.measure_basis == "X" else None
            self.sim_Z = self.sim if self.measure_basis == "Z" else None

        if verbose and self.rank == 0:
            ref = self.sim_X if self._mixed else self.sim
            nm_tag = (f" noise_model={noise_model.sha256()[:8]}" if noise_model is not None else "")
            print(
                f"[ColorQCDataGeneratorTorch] distance={self.distance} n_rounds={self.n_rounds} "
                f"schedule={self.schedule} basis={self.measure_basis} "
                f"bundle_p_nominal={ref.bundle_p_nominal:.6g} "
                f"active_p_nominal={ref.active_p_nominal:.6g} device={self.device}{nm_tag}"
            )
            # Announce a config-p override once (rank0), instead of having each
            # ColorMemoryCircuitTorch print per rank / per basis. Flagged by the sim
            # when the configured scalar p was rebuilt over the bundle's baked-in p.
            if ref.p_rebuilt_from_config_p:
                print(
                    f"[ColorQCDataGeneratorTorch] precomputed bundle was built at "
                    f"p={ref.bundle_p_nominal:.6g}, but config requests p={ref.p_nominal:.6g}; "
                    f"rebuilt the probability vector at the configured p (bundle structure reused)."
                )

    @property
    def p_nominal(self) -> float:
        ref = self.sim_X if self._mixed else self.sim
        return ref.p_nominal

    @property
    def bundle_p_nominal(self) -> float:
        ref = self.sim_X if self._mixed else self.sim
        return ref.bundle_p_nominal

    @property
    def active_p_nominal(self) -> float:
        ref = self.sim_X if self._mixed else self.sim
        return ref.active_p_nominal

    def generate_batch(
        self,
        step: int,
        batch_size: int,
        return_timing: bool = False,
        profile_generator_subphases: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]:
        del profile_generator_subphases
        t0 = time.perf_counter() if return_timing else None

        seed = (self.base_seed + self._step_seed_offset + int(step)) & 0x7FFFFFFF
        if self._mixed:
            sim = self.sim_X if (int(step) % 2 == 0) else self.sim_Z
        else:
            sim = self.sim
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                trainX, trainY = sim.generate_batch(int(batch_size), seed=seed)
        else:
            trainX, trainY = sim.generate_batch(int(batch_size), seed=seed)

        if return_timing:
            timing = {"generator_total_s": time.perf_counter() - t0}
            return trainX, trainY, timing
        return trainX, trainY
