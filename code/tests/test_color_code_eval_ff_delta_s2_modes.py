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
Unit tests for color-code evaluation FF cascade correction modes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.logical_error_rate_color import (
    _align_delta_s2_for_predecoder_mode,
    _build_ff_cascade_tensors,
)
from qec.color_code.memory_circuit import MemoryCircuit
from qec.precompute_dem import (
    build_circuit_z_ancilla_connectivity_matrix,
    extract_cnot_structure_from_stim_text,
)


def test_align_delta_s2_round_r_mode_is_identity():
    delta = torch.tensor(
        [[[0, 1, 0, 1, 1], [1, 1, 0, 0, 1]]],
        dtype=torch.int32,
    )  # (B=1, num_plaq=2, T=5)
    aligned = _align_delta_s2_for_predecoder_mode(delta, apply_feedforward_to_predecoder=True)
    assert torch.equal(aligned, delta)


def test_align_delta_s2_deferred_mode_shifts_with_last_round_override():
    delta = torch.tensor(
        [[[0, 1, 0, 1, 1]]],
        dtype=torch.int32,
    )  # (B=1, num_plaq=1, T=5)
    aligned = _align_delta_s2_for_predecoder_mode(delta, apply_feedforward_to_predecoder=False)
    expected = torch.tensor(
        [[[0, 0, 1, 0, 1]]],  # shift by +1, but keep final round from current
        dtype=torch.int32,
    )
    assert torch.equal(aligned, expected)


def test_align_delta_s2_deferred_mode_single_round_is_identity():
    delta = torch.tensor([[[1]]], dtype=torch.int32)  # (B=1, num_plaq=1, T=1)
    aligned = _align_delta_s2_for_predecoder_mode(delta, apply_feedforward_to_predecoder=False)
    assert torch.equal(aligned, delta)


def test_eval_ff_mask_matches_runtime_circuit_connectivity():
    # Small circuit for fast validation.
    mc = MemoryCircuit(
        distance=3,
        idle_error=0.0,
        sqgate_error=0.0,
        tqgate_error=0.0,
        spam_error=0.0,
        n_rounds=5,
        basis="X",
        schedule="nearest-neighbor",
        add_detectors=False,
    )

    ff_mask_tensor, _cx_c, _cx_t, _z_off, _nq = _build_ff_cascade_tensors(
        mc, mc.code.num_plaquettes, mc.code.num_data, torch.device("cpu")
    )
    ff_eval = ff_mask_tensor.cpu().numpy().astype(np.uint8)

    circuit, _ = extract_cnot_structure_from_stim_text(str(mc.circuit))
    controls = circuit[..., 0]
    targets = circuit[..., 1]
    ff_true_full = build_circuit_z_ancilla_connectivity_matrix(
        controls=np.array(controls),
        targets=np.array(targets),
        data_qubits=np.array(mc.code.data_qubits, dtype=np.int32),
        zcheck_qubits=np.array(mc.code.zcheck_qubits, dtype=np.int32),
        nq=int(mc.code.all_qubits.size),
    )
    ff_true = ff_true_full[:, np.array(mc.code.data_qubits, dtype=np.int32)]

    assert np.array_equal(ff_eval, ff_true)
