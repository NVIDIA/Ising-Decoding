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
Torch-only multi-pair on-the-fly training data generator.

Drop-in replacement for the legacy multi-pair generator. Holds one per-pair Torch
generator (`QCDataGeneratorTorch` for surface, `ColorQCDataGeneratorTorch`
for color), selects one per training step via round-robin, and forwards
generate_batch so that batches never mix shapes.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch


class MultiQCDataGeneratorTorch:
    """Round-robin manager over per-(distance, n_rounds) Torch generators."""

    def __init__(
        self,
        distances: Sequence[int],
        rounds: Sequence[int],
        *,
        code: str = "surface",
        p_error=None,
        p_min: float = 0.001,
        p_max: float = 0.008,
        measure_basis: str = "both",
        rank: int = 0,
        global_rank: Optional[int] = None,
        mode: str = "train",
        verbose: bool = False,
        timelike_he: bool = True,
        num_he_cycles: int = 1,
        use_weight2: bool = False,
        use_weight2_timelike: bool = False,
        max_passes_w1: int = 32,
        max_passes_w2: int = 32,
        base_seed: int = 42,
        decompose_y: bool = False,
        precomputed_frames_dir: Optional[str] = None,
        code_rotation: str = "XV",
        noise_model=None,
        device: Optional[torch.device] = None,
        # Color-only knobs:
        schedule: str = "nearest-neighbor",
        enable_z_feedforward: bool = True,
        apply_data_x_override: bool = True,
        apply_spacelike_he: bool = True,
        he_max_iterations: int = 16,
        use_coset_search: bool = False,
        coset_max_generators: int = 20,
        # Surface-only knobs:
        use_compile: bool = False,
        compile_chunk_size: int = 2,
        compute_dtype=None,
        use_dense_overlap: bool = False,
        use_parallel_spacelike: bool = False,
        **_ignored,
    ) -> None:
        if not isinstance(distances, (list, tuple)) or not isinstance(rounds, (list, tuple)):
            raise TypeError("distances and rounds must be lists or tuples")
        if len(distances) != len(rounds) or len(distances) == 0:
            raise ValueError("distances and rounds must have the same non-zero length")

        self._pairs: List[Tuple[int, int]] = [(int(d), int(r)) for d, r in zip(distances, rounds)]
        self._mode = str(mode)
        self._verbose = bool(verbose)
        self._gens: list = []
        self._local_steps: List[int] = []
        self.code_rotation = str(code_rotation).upper() if code_rotation else "XV"

        code_lower = str(code).lower()
        is_color = code_lower.startswith("color")

        if verbose and rank == 0:
            summary = ", ".join(f"(d={d}, r={r})" for d, r in self._pairs)
            print(
                f"[MultiQCDataGeneratorTorch] code={code_lower} pairs={summary} "
                f"precomputed_frames_dir={precomputed_frames_dir}"
            )

        for idx, (d, r) in enumerate(self._pairs):
            seed_offset = idx * 10_000_000
            if is_color:
                if precomputed_frames_dir is None:
                    raise ValueError(
                        "MultiQCDataGeneratorTorch with code=color requires "
                        "precomputed_frames_dir (color augmented DEM bundle path)."
                    )
                from data.generator_torch_color import ColorQCDataGeneratorTorch
                gen = ColorQCDataGeneratorTorch(
                    distance=d,
                    n_rounds=r,
                    schedule=schedule,
                    measure_basis=measure_basis,
                    precomputed_frames_dir=precomputed_frames_dir,
                    enable_z_feedforward=enable_z_feedforward,
                    apply_data_x_override=apply_data_x_override,
                    apply_spacelike_he=apply_spacelike_he,
                    he_max_iterations=he_max_iterations,
                    use_coset_search=use_coset_search,
                    device=device,
                    rank=rank,
                    global_rank=global_rank,
                    base_seed=base_seed + seed_offset,
                    verbose=verbose,
                    noise_model=noise_model,
                    p_error=p_error,
                    p_min=p_min,
                    p_max=p_max,
                )
            else:
                from data.generator_torch import QCDataGeneratorTorch
                gen = QCDataGeneratorTorch(
                    distance=d,
                    n_rounds=r,
                    p_error=p_error,
                    p_min=p_min,
                    p_max=p_max,
                    measure_basis=measure_basis,
                    rank=rank,
                    global_rank=global_rank,
                    mode=mode,
                    verbose=verbose,
                    timelike_he=timelike_he,
                    num_he_cycles=num_he_cycles,
                    use_weight2=bool(use_weight2 or use_weight2_timelike),
                    max_passes_w1=max_passes_w1,
                    max_passes_w2=max_passes_w2,
                    decompose_y=decompose_y,
                    precomputed_frames_dir=precomputed_frames_dir,
                    code_rotation=self.code_rotation,
                    noise_model=noise_model,
                    base_seed=base_seed,
                    seed_offset=seed_offset,
                    device=device,
                    use_compile=use_compile,
                    compile_chunk_size=compile_chunk_size,
                    compute_dtype=compute_dtype,
                    use_coset_search=use_coset_search,
                    coset_max_generators=coset_max_generators,
                    use_dense_overlap=use_dense_overlap,
                    use_parallel_spacelike=use_parallel_spacelike,
                )
            self._gens.append(gen)
            self._local_steps.append(0)

        if verbose and rank == 0:
            print(f"[MultiQCDataGeneratorTorch] All {len(self._pairs)} generators initialized.")

    def _index_for_step(self, step: int) -> int:
        # Spend 2 consecutive steps per pair so an X-Z basis pair lives on the
        # same shape. Matches the legacy multi-pair generator semantics.
        return (int(step) // 2) % len(self._gens)

    def generate_batch(
        self,
        step: int,
        batch_size: Union[int, Sequence[int]],
        return_timing: bool = False,
        profile_generator_subphases: bool = False,
    ):
        idx = self._index_for_step(step)
        local_step = self._local_steps[idx]
        self._local_steps[idx] += 1
        if isinstance(batch_size, (list, tuple)):
            effective_batch = batch_size[idx]
        else:
            effective_batch = batch_size
        return self._gens[idx].generate_batch(
            local_step,
            effective_batch,
            return_timing=return_timing,
            profile_generator_subphases=profile_generator_subphases,
        )

    def get_current_pair(self, step: int) -> Tuple[int, int]:
        return self._pairs[self._index_for_step(step)]

    def get_info(self) -> dict:
        return {
            "mode": self._mode,
            "num_pairs": len(self._pairs),
            "pairs": [{
                "distance": d,
                "n_rounds": r
            } for d, r in self._pairs],
        }

    def get_generator_for_pair(self, distance: int, n_rounds: int):
        for idx, (d, r) in enumerate(self._pairs):
            if int(d) == int(distance) and int(r) == int(n_rounds):
                return self._gens[idx]
        raise ValueError(f"No generator for (d={distance}, r={n_rounds})")

    def get_all_generators(self) -> List[Tuple[Tuple[int, int], object]]:
        return list(zip(self._pairs, self._gens))

    def is_multi_pair(self) -> bool:
        return True


__all__ = ["MultiQCDataGeneratorTorch"]
