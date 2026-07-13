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
"""Smoke tests for qec.surface_code.detector_input (CPU/Torch).

Constructor signature: SurfaceDetectorInputTransform(*, distance, rounds, basis,
rotation='XV', preprocess_strategy='gather'). The forward path takes a flat
detector tensor of shape (B, detector_width) and emits the per-round grid
tensor consumed by the pre-decoder.

Previously zero unit-test coverage; tested only indirectly through the GPU
smoke run.
"""
import pytest
import torch

from qec.surface_code.detector_input import SurfaceDetectorInputTransform


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_module_constructs(basis):
    t = SurfaceDetectorInputTransform(distance=3, rounds=3, basis=basis)
    assert isinstance(t, torch.nn.Module)
    assert t.distance == 3
    assert t.rounds == 3
    assert t.basis == basis
    # d=3 surface has 4 X-stabilizers and 4 Z-stabilizers.
    assert t.num_stabs == 4


def test_invalid_basis_rejected():
    with pytest.raises(ValueError, match="basis must be"):
        SurfaceDetectorInputTransform(distance=3, rounds=3, basis="Y")


def test_invalid_preprocess_strategy_rejected():
    with pytest.raises(ValueError, match="Unsupported preprocess"):
        SurfaceDetectorInputTransform(distance=3, rounds=3, basis="X", preprocess_strategy="nope")


@pytest.mark.parametrize("distance,rounds", [(3, 3), (5, 5)])
def test_forward_shape(distance, rounds):
    t = SurfaceDetectorInputTransform(distance=distance, rounds=rounds, basis="X")
    batch = 2
    dets = torch.zeros(batch, t.detector_width, dtype=torch.float32)
    out = t.forward(dets)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == batch


def test_two_preprocess_strategies_match():
    """Both 'gather' and 'dense_matmul' strategies should produce identical
    outputs for a given input."""
    dets = torch.randn(
        2,
        SurfaceDetectorInputTransform(distance=3, rounds=3, basis="X").detector_width
    )
    t_gather = SurfaceDetectorInputTransform(
        distance=3, rounds=3, basis="X", preprocess_strategy="gather"
    )
    t_dense = SurfaceDetectorInputTransform(
        distance=3, rounds=3, basis="X", preprocess_strategy="dense_matmul"
    )
    out_g = t_gather.forward(dets)
    out_d = t_dense.forward(dets)
    assert torch.allclose(out_g, out_d, atol=1e-5)
