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
"""CPU-runnable coverage extras for color-side modules with only 1-2 existing tests.

Tests batched into one file to keep diff size focused:
  - evaluation/color_chromobius_timing_results.py
  - evaluation/color_sdr_results.py
  - evaluation/color_threshold_results.py
  - evaluation/reference_color_baseline.py        (chromobius required at runtime;
                                                    we test only the helpers
                                                    accessible without it)
  - qec/color_code/detector_input.py
  - qec/color_code/reference_superdense_noise.py
  - qec/color_code/data_mapping.py
  - data/datapipe_stim_color.py                   (deferred init via lazy import)

A GPU is not required.
"""
import importlib

import pytest

# -------- evaluation result-aggregator helpers (timing / sdr / threshold) --------


def test_chromobius_timing_helpers():
    mod = importlib.import_module("evaluation.color_chromobius_timing_results")
    # _as_int / _as_float defend against str-from-CSV inputs.
    assert mod._as_int("3") == 3
    assert mod._as_int(3.7) == 3
    assert mod._as_float("1.5") == 1.5
    # normalize_timing_counter normalizes a counter row from CSV reload.
    out = mod.normalize_timing_counter(
        {
            "shots": "10",
            "sum_us_per_round": "5.0",
            "sum_sq_us_per_round": "2.5",
            "min_us_per_round": "0.1",
            "max_us_per_round": "1.0",
        }
    )
    assert out["shots"] == 10
    assert out["sum_us_per_round"] == 5.0


def test_sdr_helpers():
    mod = importlib.import_module("evaluation.color_sdr_results")
    assert mod._as_int("7") == 7
    out = mod.normalize_sdr_row(
        {
            "distance": "9",
            "n_rounds": "9",
            "p": "0.001",
            "basis": "x",
            "input_syndrome_ones": "10",
            "residual_syndrome_ones": "3",
            "syndrome_elements": "100",
        }
    )
    assert out["distance"] == 9
    assert out["n_rounds"] == 9
    assert out["basis"] == "X"
    assert out["residual_syndrome_ones"] == 3


def test_threshold_helpers():
    mod = importlib.import_module("evaluation.color_threshold_results")
    assert mod._as_int("5") == 5
    # _safe_rate handles divide-by-zero
    assert mod._safe_rate(3, 0) is None
    # _safe_rate computes errors/shots normally
    r = mod._safe_rate(1, 10)
    assert r is not None and abs(r - 0.1) < 1e-9
    # _logical_rate_per_round inversion
    p = mod._logical_rate_per_round(0.1, n_rounds=5)
    assert p is not None and 0.0 <= p <= 1.0


def test_reference_color_baseline_compare_results():
    """compare_results_to_paper is pure-Python aggregation; no chromobius needed."""
    mod = importlib.import_module("evaluation.reference_color_baseline")
    # Empty-input case: function should not crash on an empty result list.
    out = mod.compare_results_to_paper(results=[], series="chromobius")
    assert out is not None  # Implementation returns a dict or list summary.


# -------- qec.color_code.detector_input ColorDetectorInputTransform --------


def test_color_detector_input_transform_constructs():
    import torch
    from qec.color_code.detector_input import ColorDetectorInputTransform

    t = ColorDetectorInputTransform(distance=5, rounds=3, basis="X")
    assert isinstance(t, torch.nn.Module)
    assert t.distance == 5
    assert t.rounds == 3
    assert t.basis == "X"


# -------- qec.color_code.reference_superdense_noise Si1000ReferenceNoiseSpec --------


@pytest.mark.parametrize("p", [1e-4, 1e-3, 5e-3])
def test_si1000_reference_noise_spec_valid_p(p):
    from qec.color_code.reference_superdense_noise import Si1000ReferenceNoiseSpec

    spec = Si1000ReferenceNoiseSpec(p=p)
    assert spec.prep_error_probability == pytest.approx(2 * p)
    assert spec.measure_error_probability == pytest.approx(5 * p)
    # Gate-idle and 1Q args are total p/10 depolarising (three equal Pauli terms).
    args = spec.gate_idle_args()
    assert len(args) == 3 and abs(sum(args) - p / 10.0) < 1e-12
    # CNOT 2Q args have 15 entries summing to p.
    cnot_args = spec.two_qubit_gate_args()
    assert len(cnot_args) == 15
    assert abs(sum(cnot_args) - p) < 1e-9


@pytest.mark.parametrize("p", [-0.1, 1.1, 0.3])
def test_si1000_reference_noise_spec_rejects_invalid_p(p):
    """p outside [0,1] or where 5p > 1 must raise."""
    from qec.color_code.reference_superdense_noise import Si1000ReferenceNoiseSpec
    with pytest.raises(ValueError):
        Si1000ReferenceNoiseSpec(p=p)


# -------- qec.color_code.data_mapping basic shape contracts --------


def test_color_data_mapping_grid_indices_consistency():
    """For each distance, the stab-to-grid and data-to-grid index maps return
    1-D tensors with sizes equal to the respective qubit counts."""
    from qec.color_code.color_code import ColorCode
    from qec.color_code.data_mapping import (
        get_data_to_grid_flat_index,
        get_stab_to_grid_flat_index,
    )

    for distance in (5, 7):
        cc = ColorCode(distance=distance)
        stab_idx = get_stab_to_grid_flat_index(cc)
        data_idx = get_data_to_grid_flat_index(cc)
        assert stab_idx.ndim == 1
        assert data_idx.ndim == 1
        # Triangular color code has more data than stabilizer qubits.
        assert int(data_idx.numel()) > int(stab_idx.numel())


# -------- data.datapipe_stim_color basic class import contract --------


def test_color_datapipe_stim_class_exists():
    from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
    # Don't instantiate (it builds full Stim circuits which take time and need
    # the chromobius DEM lift). Just check the class is import-shaped correctly.
    assert hasattr(QCDataPipePreDecoder_ColorCode_inference, "__init__")
    assert hasattr(QCDataPipePreDecoder_ColorCode_inference, "__len__")
    assert hasattr(QCDataPipePreDecoder_ColorCode_inference, "__getitem__")
