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

import torch
from pathlib import Path


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
        use_weight2=False,
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
        use_compile=False,
        compile_chunk_size=2,
        compute_dtype=None,
        use_coset_search=False,
        coset_max_generators=20,
        use_dense_overlap=False,
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
        self.noise_model = noise_model

        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch
        from qec.precompute_dem import (
            build_probability_vector_surface_code,
            dem_artifact_metadata_matches,
            load_dem_artifact_metadata,
            precompute_dem_bundle_surface_code,
        )

        import threading
        self._early_compile_threads: list[threading.Thread] = []
        if bool(use_compile) and bool(timelike_he) and self.device.type == "cuda":
            from qec.surface_code.homological_equivalence_torch import warmup_he_compile
            bases_to_warm = ["X", "Z"] if self._mixed else [self._single_basis]
            for b in bases_to_warm:
                t = threading.Thread(
                    target=warmup_he_compile,
                    kwargs=dict(
                        distance=self.distance,
                        n_rounds=self.n_rounds,
                        basis=b,
                        max_passes_w1=max_passes_w1,
                        use_weight2=use_weight2,
                        max_passes_w2=max_passes_w2,
                    ),
                    daemon=True,
                )
                t.start()
                self._early_compile_threads.append(t)

        dem_cache = {}
        p_overrides = {}
        bases_needed = ["X", "Z"] if self._mixed else [self._single_basis]
        p_nom = float(p_error if p_error is not None else (p_max if p_max is not None else 0.004))
        effective_precomputed_frames_dir = precomputed_frames_dir
        metadata_reason = None
        if precomputed_frames_dir is not None:
            precomputed_dir = Path(precomputed_frames_dir)
            for b in bases_needed:
                p_path = (
                    precomputed_dir /
                    f"surface_d{self.distance}_r{self.n_rounds}_{b}_frame_predecoder.p.npz"
                )
                if not p_path.exists():
                    # Asymmetric handling on purpose:
                    # - Scalar mode preserves the legacy behaviour of letting
                    #   MemoryCircuitTorch raise its own clear FileNotFoundError
                    #   downstream (so `continue` here keeps the dir intact).
                    # - 25p mode self-heals by falling back to an in-memory
                    #   build, since the p artifact is needed to determine the
                    #   cached error-column count.
                    if noise_model is None:
                        continue
                    effective_precomputed_frames_dir = None
                    metadata_reason = f"missing p artifact for basis {b}"
                    break
                metadata = load_dem_artifact_metadata(p_path)
                ok, reason = dem_artifact_metadata_matches(
                    metadata,
                    distance=self.distance,
                    n_rounds=self.n_rounds,
                    basis=b,
                    code_rotation=self.code_rotation,
                    p_scalar=p_nom,
                    noise_model=noise_model,
                )
                if not ok:
                    effective_precomputed_frames_dir = None
                    metadata_reason = f"basis {b}: {reason}"
                    break
                p_overrides[b] = torch.from_numpy(
                    build_probability_vector_surface_code(
                        distance=self.distance,
                        n_rounds=self.n_rounds,
                        basis=b,
                        code_rotation=self.code_rotation,
                        p_scalar=p_nom,
                        noise_model=noise_model,
                    )
                )

        if effective_precomputed_frames_dir is None:
            nm_tag = ", noise_model=25p" if noise_model is not None else ""
            if precomputed_frames_dir is None:
                source = "precomputed_frames_dir=None"
            else:
                source = f"precomputed DEM metadata mismatch ({metadata_reason})"
            # Always announce a metadata-driven rebuild on rank 0 so silent
            # rebuilds are visible in non-verbose distributed runs. The
            # precomputed_frames_dir=None path is only logged when verbose.
            should_log = self.verbose or (
                precomputed_frames_dir is not None and int(self.rank) == 0
            )
            if should_log:
                print(
                    f"[QCDataGeneratorTorch] {source} -> building in-memory DEM bundle "
                    f"at p={p_nom}{nm_tag}"
                )
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
                    noise_model=noise_model,
                )
        elif self.verbose:
            nm_tag = (
                f", refreshed_p_from_noise_model={noise_model.sha256()}"
                if noise_model is not None else f", refreshed_p_from_scalar={p_nom:g}"
            )
            print(
                f"[QCDataGeneratorTorch] using disk DEM structure from "
                f"{effective_precomputed_frames_dir}{nm_tag}"
            )

        _he_kwargs = dict(
            timelike_he=timelike_he,
            num_he_cycles=num_he_cycles,
            max_passes_w1=max_passes_w1,
            use_compile=use_compile,
            compile_chunk_size=compile_chunk_size,
            compute_dtype=compute_dtype,
            use_weight2=use_weight2,
            max_passes_w2=max_passes_w2,
            use_coset_search=use_coset_search,
            coset_max_generators=coset_max_generators,
            use_dense_overlap=use_dense_overlap,
        )

        if self._mixed:
            self.sim_X = MemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis="X",
                precomputed_frames_dir=effective_precomputed_frames_dir,
                code_rotation=self.code_rotation,
                device=self.device,
                H=(dem_cache.get("X", {}).get("H") if dem_cache else None),
                p=(dem_cache.get("X", {}).get("p") if dem_cache else None),
                A=(dem_cache.get("X", {}).get("A") if dem_cache else None),
                p_override=p_overrides.get("X"),
                **_he_kwargs,
            )
            self.sim_Z = MemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis="Z",
                precomputed_frames_dir=effective_precomputed_frames_dir,
                code_rotation=self.code_rotation,
                device=self.device,
                H=(dem_cache.get("Z", {}).get("H") if dem_cache else None),
                p=(dem_cache.get("Z", {}).get("p") if dem_cache else None),
                A=(dem_cache.get("Z", {}).get("A") if dem_cache else None),
                p_override=p_overrides.get("Z"),
                **_he_kwargs,
            )
        else:
            self.sim = MemoryCircuitTorch(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=self._single_basis,
                precomputed_frames_dir=effective_precomputed_frames_dir,
                code_rotation=self.code_rotation,
                device=self.device,
                H=(dem_cache.get(self._single_basis, {}).get("H") if dem_cache else None),
                p=(dem_cache.get(self._single_basis, {}).get("p") if dem_cache else None),
                A=(dem_cache.get(self._single_basis, {}).get("A") if dem_cache else None),
                p_override=p_overrides.get(self._single_basis),
                **_he_kwargs,
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
        if self._early_compile_threads:
            for t in self._early_compile_threads:
                # torch.compile warmup can be slow; 20 min cap prevents silent hangs.
                t.join(timeout=1200)
                if t.is_alive():
                    raise RuntimeError("warmup_he_compile thread did not finish within 20 min")
            self._early_compile_threads.clear()

        if self._mixed:
            sim = self.sim_X if (int(step) % 2 == 0) else self.sim_Z
        else:
            sim = self.sim
        return sim.generate_batch(batch_size=int(batch_size))


__all__ = ["QCDataGeneratorTorch"]
