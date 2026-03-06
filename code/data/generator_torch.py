# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch


class QCDataGeneratorTorch:
    """Torch-only on-the-fly generator using precomputed H/p/A."""

    def __init__(
        self,
        *,
        distance,
        n_rounds,
        p_error=None,
        p_min=None,
        p_max=None,
        measure_basis="both",
        rank=0,
        global_rank=None,
        mode="train",
        verbose=False,
        timelike_he=True,
        num_he_cycles=1,
        use_weight2_timelike=False,
        max_passes_w1=32,
        max_passes_w2=32,
        decompose_y=False,
        precomputed_frames_dir=None,
        code_rotation="XV",
        noise_model=None,
        use_multiround_frames=True,
        use_torch=False,
        base_seed=42,
        seed_offset=0,
        device=None,
        **_ignored,
    ):
        if global_rank is None:
            global_rank = rank
        self.distance = int(distance)
        self.n_rounds = int(n_rounds)
        self.rank = int(rank)
        self.global_rank = int(global_rank)
        self.mode = str(mode).lower()
        self.verbose = bool(verbose)
        self.code_rotation = str(code_rotation).upper()

        self._mixed = str(measure_basis).lower() in ("both", "mixed")
        self._single_basis = None if self._mixed else str(measure_basis).upper()

        if device is None:
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Torch-only constraints (keep the API surface compatible with old configs).
        if bool(decompose_y):
            raise ValueError(
                "decompose_y is not supported in the Torch-only generator (set decompose_y=false)."
            )
        if noise_model is not None:
            # TODO: wire noise_model through to precompute_dem_bundle_surface_code()
            # so train/eval use the same 25-parameter noise distribution.
            # build_single_p_marginal() already supports noise_model; only this
            # generator-level plumbing is missing.
            raise ValueError(
                "noise_model is not supported in the Torch-only generator (simple single-p only)."
            )

        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch
        from qec.precompute_dem import precompute_dem_bundle_surface_code

        # If no DEM dir is provided, precompute DEM artifacts in-memory once.
        # This is slower at startup, but avoids any file I/O and supports "train without loading DEMs".
        dem_cache = {}
        if precomputed_frames_dir is None:
            # Pick a nominal p for building the single-p marginal vector.
            # (This matches existing behavior when using a precomputed directory.)
            p_nom = float(
                p_error if p_error is not None else (p_max if p_max is not None else 0.004)
            )
            if self.verbose:
                print(
                    f"[QCDataGeneratorTorch] precomputed_frames_dir=None -> building in-memory DEM bundle at p={p_nom}"
                )
            bases_needed = ["X", "Z"] if self._mixed else [self._single_basis]
            for b in bases_needed:
                dem_cache[b] = precompute_dem_bundle_surface_code(
                    distance=self.distance,
                    n_rounds=self.n_rounds,
                    basis=b,
                    code_rotation=self.code_rotation,
                    p_scalar=p_nom,
                    dem_output_dir=None,
                    device=self.device,
                    export=False,
                    return_artifacts=True,
                    # TODO: pass noise_model=noise_model here once the guard above is removed
                )

        if self._mixed:
            self.sim_X = MemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis="X",
                precomputed_frames_dir=precomputed_frames_dir,
                code_rotation=self.code_rotation,
                timelike_he=timelike_he,
                num_he_cycles=num_he_cycles,
                max_passes_w1=max_passes_w1,
                device=self.device,
                H=(dem_cache.get("X", {}).get("H") if dem_cache else None),
                p=(dem_cache.get("X", {}).get("p") if dem_cache else None),
                A=(dem_cache.get("X", {}).get("A") if dem_cache else None),
            )
            self.sim_Z = MemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis="Z",
                precomputed_frames_dir=precomputed_frames_dir,
                code_rotation=self.code_rotation,
                timelike_he=timelike_he,
                num_he_cycles=num_he_cycles,
                max_passes_w1=max_passes_w1,
                device=self.device,
                H=(dem_cache.get("Z", {}).get("H") if dem_cache else None),
                p=(dem_cache.get("Z", {}).get("p") if dem_cache else None),
                A=(dem_cache.get("Z", {}).get("A") if dem_cache else None),
            )
        else:
            self.sim = MemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=self._single_basis,
                precomputed_frames_dir=precomputed_frames_dir,
                code_rotation=self.code_rotation,
                timelike_he=timelike_he,
                num_he_cycles=num_he_cycles,
                max_passes_w1=max_passes_w1,
                device=self.device,
                H=(dem_cache.get(self._single_basis, {}).get("H") if dem_cache else None),
                p=(dem_cache.get(self._single_basis, {}).get("p") if dem_cache else None),
                A=(dem_cache.get(self._single_basis, {}).get("A") if dem_cache else None),
            )

        seed = int(base_seed) + int(self.global_rank) * 1_000_000 + int(seed_offset)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.verbose:
            b = "both" if self._mixed else self._single_basis
            print(
                f"[QCDataGeneratorTorch] Initialized (d={self.distance}, r={self.n_rounds}, basis={b}, device={self.device})"
            )

    def generate_batch(self, step, batch_size):
        # Basis alternation: even=X, odd=Z.
        if self._mixed:
            sim = self.sim_X if (int(step) % 2 == 0) else self.sim_Z
        else:
            sim = self.sim
        return sim.generate_batch(batch_size=int(batch_size))


__all__ = ["QCDataGeneratorTorch"]
