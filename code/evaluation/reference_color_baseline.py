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
Reference-noise Chromobius baselines for the fixed superdense/CX color code.
"""

from __future__ import annotations

import chromobius
import numpy as np

from qec.color_code.reference_superdense_noise import (
    PAPER_SUPERDENSE_SI1000_ORACLE,
    build_color_memory_circuit,
)
from qec.noise_model import (
    normalize_noise_instruction_semantics,
    normalize_noise_model_family,
)


def compute_chromobius_ler(
    *,
    distance: int,
    p: float,
    n_rounds: int,
    basis: str,
    num_shots: int,
    batch_size: int = 100_000,
    noise_model_family: str = "legacy",
    noise_instruction_semantics: str = "current",
    gidney_style_noise: bool = False,
) -> dict:
    """
    Compute logical error rate / round for one color-code memory point.
    """
    circuit_obj = build_color_memory_circuit(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        p_error=p,
        noise_model_family=normalize_noise_model_family(noise_model_family),
        noise_instruction_semantics=normalize_noise_instruction_semantics(
            noise_instruction_semantics
        ),
        noise_model=None,
        gidney_style_noise=gidney_style_noise,
        add_boundary_detectors=True,
    )
    circuit = circuit_obj.stim_circuit

    dem = circuit.detector_error_model(
        decompose_errors=False,
        approximate_disjoint_errors=True,
        ignore_decomposition_failures=True,
    )
    decoder = chromobius.compile_decoder_for_dem(dem)
    sampler = circuit.compile_detector_sampler()

    total_errors = 0
    total_sampled = 0
    remaining = int(num_shots)
    num_obs = int(circuit.num_observables)

    while remaining > 0:
        batch = min(remaining, int(batch_size))
        dets, obs = sampler.sample(batch, separate_observables=True)
        packed = np.packbits(dets.astype(np.uint8), axis=1, bitorder="little")
        predictions = decoder.predict_obs_flips_from_dets_bit_packed(packed)
        pred = np.unpackbits(predictions, axis=1, bitorder="little")[:, :num_obs]
        total_errors += int(np.sum(pred != obs.astype(np.uint8)))
        total_sampled += batch
        remaining -= batch

    ler_total = total_errors / total_sampled
    ler_per_round = ler_total / n_rounds
    stderr = np.sqrt(ler_total * (1.0 - ler_total) / total_sampled) / n_rounds

    return {
        "distance":
            int(distance),
        "n_rounds":
            int(n_rounds),
        "p":
            float(p),
        "basis":
            str(basis).upper(),
        "noise_model_family":
            normalize_noise_model_family(noise_model_family),
        "noise_instruction_semantics":
            normalize_noise_instruction_semantics(noise_instruction_semantics),
        "gidney_style_noise":
            bool(gidney_style_noise),
        "ler_per_round":
            float(ler_per_round),
        "ler_total":
            float(ler_total),
        "stderr":
            float(stderr),
        "num_errors":
            int(total_errors),
        "num_shots":
            int(total_sampled),
    }


def sweep_chromobius_baseline(
    *,
    distances,
    p_values,
    bases,
    num_shots: int,
    n_rounds_list=None,
    batch_size: int = 100_000,
    noise_model_family: str = "legacy",
    noise_instruction_semantics: str = "current",
    gidney_style_noise: bool = False,
) -> list[dict]:
    if n_rounds_list is None:
        n_rounds_list = [4 * int(d) for d in distances]

    rows = []
    for distance, n_rounds in zip(distances, n_rounds_list):
        for p in p_values:
            for basis in bases:
                rows.append(
                    compute_chromobius_ler(
                        distance=int(distance),
                        p=float(p),
                        n_rounds=int(n_rounds),
                        basis=str(basis),
                        num_shots=num_shots,
                        batch_size=batch_size,
                        noise_model_family=noise_model_family,
                        noise_instruction_semantics=noise_instruction_semantics,
                        gidney_style_noise=gidney_style_noise,
                    )
                )
    return rows


def compare_results_to_paper(
    results: list[dict],
    *,
    series: str = "chromobius",
) -> dict:
    oracle = PAPER_SUPERDENSE_SI1000_ORACLE[series]
    comparisons = []
    for row in results:
        if str(row["basis"]).upper() != "X":
            continue
        paper_ler = oracle.get(int(row["distance"]), {}).get(float(row["p"]))
        if paper_ler is None:
            continue
        comparisons.append(
            {
                "distance":
                    int(row["distance"]),
                "n_rounds":
                    int(row["n_rounds"]),
                "p":
                    float(row["p"]),
                "basis":
                    str(row["basis"]).upper(),
                "noise_model_family":
                    row["noise_model_family"],
                "noise_instruction_semantics":
                    row["noise_instruction_semantics"],
                "paper_ler_per_round":
                    float(paper_ler),
                "ours_ler_per_round":
                    float(row["ler_per_round"]),
                "ratio_ours_over_paper":
                    float(row["ler_per_round"] / paper_ler) if paper_ler > 0 else None,
                "abs_diff":
                    float(row["ler_per_round"] - paper_ler),
            }
        )
    return {
        "oracle_source": "local PGF extraction",
        "oracle_series": series,
        "comparisons": comparisons,
    }


def compare_current_vs_reference(
    *,
    distance: int,
    p: float,
    n_rounds: int,
    basis: str,
    num_shots: int,
    batch_size: int = 100_000,
    gidney_style_noise: bool = False,
) -> dict:
    current_row = compute_chromobius_ler(
        distance=distance,
        p=p,
        n_rounds=n_rounds,
        basis=basis,
        num_shots=num_shots,
        batch_size=batch_size,
        noise_model_family="si1000",
        noise_instruction_semantics="current",
        gidney_style_noise=gidney_style_noise,
    )
    reference_row = compute_chromobius_ler(
        distance=distance,
        p=p,
        n_rounds=n_rounds,
        basis=basis,
        num_shots=num_shots,
        batch_size=batch_size,
        noise_model_family="si1000",
        noise_instruction_semantics="reference",
        gidney_style_noise=False,
    )
    paper_ler = PAPER_SUPERDENSE_SI1000_ORACLE["chromobius"].get(distance, {}).get(p)
    return {
        "current": current_row,
        "reference": reference_row,
        "paper_ler_per_round": paper_ler,
        "reference_over_current":
            (
                reference_row["ler_per_round"] /
                current_row["ler_per_round"] if current_row["ler_per_round"] > 0 else None
            ),
        "current_over_paper": (current_row["ler_per_round"] / paper_ler if paper_ler else None),
        "reference_over_paper": (reference_row["ler_per_round"] / paper_ler if paper_ler else None),
    }
