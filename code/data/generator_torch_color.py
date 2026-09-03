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

from data.superbatch import SuperbatchBuffer
from qec.color_code.memory_circuit_torch import ColorMemoryCircuitTorch

# Distance between the seed ranges of consecutive global ranks. Rank r draws
# its step seeds from [base + (r+1)*STRIDE, base + (r+2)*STRIDE); a run whose
# generator step count reaches the stride would silently continue into rank
# r+1's range. ``check_step_budget`` guards that boundary at start-up.
RANK_STREAM_STRIDE = 1_000_003


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
        superbatch_size: int = 0,
        superbatch_prefetch: bool = False,
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
        self._step_seed_offset = (self.global_rank + 1) * RANK_STREAM_STRIDE
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

        self._superbatch_buffer = SuperbatchBuffer(
            superbatch_size=int(superbatch_size),
            prefetch=bool(superbatch_prefetch),
            device=self.device,
            build_fn=self._build_superbatch,
        )

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

    # Mixed-basis superbatches draw two seeds per build (one per basis). They
    # live in their own seed range so that no two builds — and no build and a
    # direct-generation step — ever hand the same seed to the same simulator.
    _SUPERBATCH_SEED_SALT = 0x2545F491

    def _seed_for_step(self, step: int) -> int:
        return (self.base_seed + self._step_seed_offset + int(step)) & 0x7FFFFFFF

    def _seed_for_superbatch_basis(self, start_step: int, basis_index: int) -> int:
        """Seed for one basis of the superbatch that starts at ``start_step``.

        Keyed by (start_step, basis) with stride 2, so consecutive superbatches
        never reuse a seed even when a superbatch holds a single minibatch
        (superbatch_size == batch_size), where ``start_step`` advances by 1.
        """
        return (
            self.base_seed + self._step_seed_offset + self._SUPERBATCH_SEED_SALT +
            2 * int(start_step) + int(basis_index)
        ) & 0x7FFFFFFF

    def max_generator_steps(self) -> int:
        """Largest generator step count that keeps this rank inside its seed range.

        Direct generation uses one seed per step; mixed-basis superbatch builds
        use two seeds per start step (stride 2), which halves the budget.
        """
        if self._mixed and self.superbatch_active():
            return RANK_STREAM_STRIDE // 2
        return RANK_STREAM_STRIDE

    def check_step_budget(self, total_generator_steps: int) -> None:
        """Fail fast if a run would push this rank's seeds into the next rank's range.

        ``total_generator_steps`` is the number of minibatches the run will
        request from ``generate_batch`` (microbatch iterations, not optimizer
        updates), i.e. the largest step index plus one.
        """
        limit = self.max_generator_steps()
        if int(total_generator_steps) > limit:
            mode = "mixed-basis superbatching (two seeds per build)" if (
                self._mixed and self.superbatch_active()
            ) else "direct generation (one seed per step)"
            raise ValueError(
                f"Training would take {int(total_generator_steps):,} generator steps, but with "
                f"{mode} a rank's seed range holds only {limit:,}; beyond that its stream "
                f"overlaps the next global rank's (stride {RANK_STREAM_STRIDE:,}). Reduce "
                "epochs or samples per epoch, or enlarge the per-rank batch size."
            )

    def _generate_with_sim(self, sim, step: int, batch_size: int, *, seed: int | None = None):
        if seed is None:
            seed = self._seed_for_step(step)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                return sim.generate_batch(int(batch_size), seed=seed)
        return sim.generate_batch(int(batch_size), seed=seed)

    def _generate_direct(self, step: int, batch_size: int,
                         collect_timing: bool) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        t0 = time.perf_counter()
        if self._mixed:
            sim = self.sim_X if int(step) % 2 == 0 else self.sim_Z
        else:
            sim = self.sim
        train_x, train_y = self._generate_with_sim(sim, int(step), int(batch_size))
        timing = {}
        if collect_timing:
            timing["generator_total_s"] = time.perf_counter() - t0
        return train_x, train_y, timing

    def _build_superbatch(self, start_step: int, sample_count: int,
                          collect_timing: bool) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Build one superbatch, interleaving bases so every slice stays balanced.

        Direct generation alternates X and Z per step. A superbatch covers many
        steps at once, so the two bases are generated in halves and interleaved
        row-by-row; any contiguous even-sized slice then holds equal X and Z.
        """
        if not self._mixed:
            return self._generate_direct(int(start_step), int(sample_count), collect_timing)
        if sample_count % 2:
            raise ValueError("mixed-basis color superbatches require an even size")

        t0 = time.perf_counter()
        first_sim, second_sim = (
            (self.sim_X, self.sim_Z) if int(start_step) % 2 == 0 else (self.sim_Z, self.sim_X)
        )
        half = int(sample_count) // 2
        first_x, first_y = self._generate_with_sim(
            first_sim,
            int(start_step),
            half,
            seed=self._seed_for_superbatch_basis(int(start_step), 0),
        )
        second_x, second_y = self._generate_with_sim(
            second_sim,
            int(start_step),
            half,
            seed=self._seed_for_superbatch_basis(int(start_step), 1),
        )
        train_x = torch.empty(
            (sample_count, *first_x.shape[1:]), dtype=first_x.dtype, device=first_x.device
        )
        train_y = torch.empty(
            (sample_count, *first_y.shape[1:]), dtype=first_y.dtype, device=first_y.device
        )
        train_x[0::2].copy_(first_x)
        train_x[1::2].copy_(second_x)
        train_y[0::2].copy_(first_y)
        train_y[1::2].copy_(second_y)
        timing = {}
        if collect_timing:
            timing["generator_total_s"] = time.perf_counter() - t0
        return train_x, train_y, timing

    def generate_direct_batch(
        self,
        step: int,
        batch_size: int,
        return_timing: bool = False,
        profile_generator_subphases: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]:
        """Generate one batch, bypassing the superbatch buffer."""
        del profile_generator_subphases
        train_x, train_y, timing = self._generate_direct(
            int(step), int(batch_size), bool(return_timing)
        )
        if return_timing:
            return train_x, train_y, timing
        return train_x, train_y

    def generate_batch(
        self,
        step: int,
        batch_size: int,
        return_timing: bool = False,
        profile_generator_subphases: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]:
        del profile_generator_subphases
        return self._superbatch_buffer.generate(
            step=int(step),
            batch_size=int(batch_size),
            return_timing=bool(return_timing),
            direct_fn=self._generate_direct,
        )

    def superbatch_active(self) -> bool:
        return self._superbatch_buffer.active

    def release_superbatch(self, *, wait: bool = True, keep_pending: bool = False) -> None:
        self._superbatch_buffer.release(wait=wait, keep_pending=keep_pending)

    def close(self) -> None:
        self._superbatch_buffer.close()

    def __del__(self):
        buffer = getattr(self, "_superbatch_buffer", None)
        if buffer is not None:
            try:
                buffer.close()
            except Exception:
                pass
