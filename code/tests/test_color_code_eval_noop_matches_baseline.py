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
Regression test: color-code evaluation must not break Z basis.

If the model predicts *no* corrections (all logits < 0 with threshold=0),
then the evaluation pipeline should reduce to plain Chromobius decoding.

In particular, for both X and Z bases:
  logical_errors == chromobius_errors

This used to fail for Z because the evaluation code incorrectly XOR'ed
*measured* ancilla bits (from the inlined observable definition) into the
pre-decoder logical frame, cancelling the observable's ancilla contribution
and producing ~50% errors in Z.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

import torch

from evaluation.logical_error_rate_color import count_logical_errors_color


class _Dist:
    rank = 0
    world_size = 1


class _NoOpModel(torch.nn.Module):
    """Always predicts zero corrections with threshold=0.0."""

    def forward(self, x):
        # x: (B, 4, T, n_rows, n_cols)
        b, _, t, n_rows, n_cols = x.shape
        return torch.full((b, 4, t, n_rows, n_cols), -1.0, device=x.device, dtype=x.dtype)


def test_eval_noop_model_matches_chromobius_baseline_both_bases():
    cfg = SimpleNamespace(
        code="color",
        distance=9,
        n_rounds=9,
        enable_fp16=False,
    )

    cfg.test = SimpleNamespace(
        num_samples=4096,
        trials=1,
        distance=5,
        n_rounds=20,
        noise_model="none",
        p_error=0.001,
        meas_basis_test="both",
        use_model_checkpoint=0,
        th_data=0.0,
        th_syn=0.0,
        sampling_mode="threshold",
        temperature=1.0,
        dataloader={
            "batch_size": 1024,
            "num_workers": 0,
            "pin_memory": False,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _NoOpModel().to(device)

    res = count_logical_errors_color(model, device, _Dist(), cfg)

    assert res["X"]["logical_errors"] == res["X"]["chromobius_errors"]
    assert res["Z"]["logical_errors"] == res["Z"]["chromobius_errors"]


def test_eval_noop_model_matches_baseline_with_feedforward():
    """
    No-op invariance must hold with feedforward enabled.
    """
    cfg = SimpleNamespace(
        code="color",
        distance=9,
        n_rounds=9,
        enable_fp16=False,
    )
    cfg.test = SimpleNamespace(
        num_samples=4096,
        trials=1,
        distance=5,
        n_rounds=20,
        noise_model="none",
        p_error=0.001,
        meas_basis_test="both",
        use_model_checkpoint=0,
        th_data=0.0,
        th_syn=0.0,
        sampling_mode="threshold",
        temperature=1.0,
        dataloader={
            "batch_size": 1024,
            "num_workers": 0,
            "pin_memory": False,
        },
        enable_delta_s2_correction=False,
    )
    cfg.data = SimpleNamespace(enable_z_feedforward=True,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _NoOpModel().to(device)

    res = count_logical_errors_color(model, device, _Dist(), cfg)
    assert res["X"]["logical_errors"] == res["X"]["chromobius_errors"]
    assert res["Z"]["logical_errors"] == res["Z"]["chromobius_errors"]


def test_eval_noop_model_matches_chromobius_baseline_si1000():
    cfg = SimpleNamespace(
        code="color",
        distance=9,
        n_rounds=9,
        enable_fp16=False,
    )

    cfg.test = SimpleNamespace(
        num_samples=2048,
        trials=1,
        distance=5,
        n_rounds=20,
        noise_model_family="si1000",
        noise_instruction_semantics="current",
        noise_mode="Si1000",
        gidney_style_noise=False,
        noise_model="none",
        p_error=0.001,
        meas_basis_test="both",
        use_model_checkpoint=0,
        th_data=0.0,
        th_syn=0.0,
        sampling_mode="threshold",
        temperature=1.0,
        dataloader={
            "batch_size": 512,
            "num_workers": 0,
            "pin_memory": False,
        },
    )
    cfg.data = SimpleNamespace(enable_z_feedforward=True,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _NoOpModel().to(device)

    res = count_logical_errors_color(model, device, _Dist(), cfg)
    assert res["X"]["logical_errors"] == res["X"]["chromobius_errors"]
    assert res["Z"]["logical_errors"] == res["Z"]["chromobius_errors"]


def test_eval_noop_model_matches_chromobius_baseline_si1000_reference():
    cfg = SimpleNamespace(
        code="color",
        distance=9,
        n_rounds=9,
        enable_fp16=False,
    )

    cfg.test = SimpleNamespace(
        num_samples=2048,
        trials=1,
        distance=5,
        n_rounds=20,
        noise_model_family="si1000",
        noise_instruction_semantics="reference",
        noise_model="none",
        p_error=0.001,
        meas_basis_test="both",
        use_model_checkpoint=0,
        th_data=0.0,
        th_syn=0.0,
        sampling_mode="threshold",
        temperature=1.0,
        dataloader={
            "batch_size": 512,
            "num_workers": 0,
            "pin_memory": False,
        },
    )
    cfg.data = SimpleNamespace(enable_z_feedforward=True,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _NoOpModel().to(device)

    res = count_logical_errors_color(model, device, _Dist(), cfg)
    assert res["X"]["logical_errors"] == res["X"]["chromobius_errors"]
    assert res["Z"]["logical_errors"] == res["Z"]["chromobius_errors"]
