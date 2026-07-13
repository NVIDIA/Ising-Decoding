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

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.reference_color_baseline import compare_results_to_paper
from qec.color_code.reference_superdense_noise import (
    build_color_memory_circuit,
    summarize_reference_noise_semantics,
)


def test_reference_noise_backend_splits_measure_reset_windows():
    p = 1e-3
    n_rounds = 20
    current = build_color_memory_circuit(
        distance=5,
        n_rounds=n_rounds,
        basis="X",
        p_error=p,
        noise_model_family="si1000",
        noise_instruction_semantics="current",
        gidney_style_noise=False,
        add_boundary_detectors=True,
    )
    reference = build_color_memory_circuit(
        distance=5,
        n_rounds=n_rounds,
        basis="X",
        p_error=p,
        noise_model_family="si1000",
        noise_instruction_semantics="reference",
        gidney_style_noise=False,
        add_boundary_detectors=True,
    )

    current_summary = summarize_reference_noise_semantics(current.stim_circuit_raw, p=p)
    reference_summary = summarize_reference_noise_semantics(reference.stim_circuit_raw, p=p)

    assert reference_summary["measure_reset_idle_noise_ops"] == 2 * (n_rounds - 1)
    assert reference_summary["other_pauli1_noise_ops"] == 0
    assert current_summary["measure_reset_idle_noise_ops"] == 0
    assert current_summary["other_pauli1_noise_ops"] > 0


def test_compare_results_to_paper_emits_residual_ratios():
    results = [
        {
            "distance": 5,
            "n_rounds": 20,
            "p": 1e-3,
            "basis": "X",
            "noise_model_family": "si1000",
            "noise_instruction_semantics": "reference",
            "gidney_style_noise": False,
            "ler_per_round": 1e-3,
            "ler_total": 2e-2,
            "stderr": 1e-5,
            "num_errors": 10,
            "num_shots": 10000,
        }
    ]
    payload = compare_results_to_paper(results, series="chromobius")
    assert payload["oracle_series"] == "chromobius"
    assert len(payload["comparisons"]) == 1
    row = payload["comparisons"][0]
    assert row["distance"] == 5
    assert row["p"] == 1e-3
    assert row["paper_ler_per_round"] > 0
    assert row["ratio_ours_over_paper"] > 0
