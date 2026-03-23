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
Public model registry for the early-access public release.

External users choose `model_id` in {1..5}. This registry maps model_id to:
- the underlying architecture parameters (num_filters, kernel_size)
- the model receptive field R (in rounds / distance units)

Receptive field convention matches `compare_receptive_field_with_window_data`
in `code/training/utils.py`:
  R = 1 + sum_i (k_i - 1)   for kernel sizes k_i (assumed odd, with same-padding)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def compute_receptive_field(kernel_sizes: List[int]) -> int:
    """Compute receptive field R from a list of kernel sizes."""
    if not kernel_sizes:
        raise ValueError("kernel_sizes must be non-empty")
    if any(not isinstance(k, int) for k in kernel_sizes):
        raise ValueError(f"kernel_sizes must be ints, got: {kernel_sizes!r}")
    if any(k <= 0 for k in kernel_sizes):
        raise ValueError(f"kernel_sizes must be positive, got: {kernel_sizes!r}")
    # Match training/utils.py: R = 1 + sum(k) - len(k) == 1 + sum(k-1)
    return 1 + sum(kernel_sizes) - len(kernel_sizes)


@dataclass(frozen=True)
class PublicModelSpec:
    model_id: int
    num_filters: List[int]
    kernel_size: List[int]
    receptive_field: int
    model_version: str = "predecoder_memory_v1"


_MODEL_SPECS: Dict[int, PublicModelSpec] = {
    # Model 1: 4 conv layers, k=3
    1:
        PublicModelSpec(
            model_id=1,
            num_filters=[128, 128, 128, 4],
            kernel_size=[3, 3, 3, 3],
            receptive_field=compute_receptive_field([3, 3, 3, 3]),
        ),
    # Model 2: 4 conv layers, k=3, wider
    2:
        PublicModelSpec(
            model_id=2,
            num_filters=[256, 256, 256, 4],
            kernel_size=[3, 3, 3, 3],
            receptive_field=compute_receptive_field([3, 3, 3, 3]),
        ),
    # Model 3: 4 conv layers, k=5
    3:
        PublicModelSpec(
            model_id=3,
            num_filters=[128, 128, 128, 4],
            kernel_size=[5, 5, 5, 5],
            receptive_field=compute_receptive_field([5, 5, 5, 5]),
        ),
    # Model 4: 6 conv layers, k=3
    4:
        PublicModelSpec(
            model_id=4,
            num_filters=[128, 128, 128, 128, 128, 4],
            kernel_size=[3, 3, 3, 3, 3, 3],
            receptive_field=compute_receptive_field([3, 3, 3, 3, 3, 3]),
        ),
    # Model 5: 6 conv layers, k=3, wider
    5:
        PublicModelSpec(
            model_id=5,
            num_filters=[256, 256, 256, 256, 256, 4],
            kernel_size=[3, 3, 3, 3, 3, 3],
            receptive_field=compute_receptive_field([3, 3, 3, 3, 3, 3]),
        ),
}


def get_model_spec(model_id: int) -> PublicModelSpec:
    """Return the public model spec for a given model_id (1..5)."""
    try:
        mid = int(model_id)
    except Exception as e:
        raise ValueError(f"model_id must be an int in [1..5], got: {model_id!r}") from e
    if mid == 0:
        raise ValueError("model_id=0 is not supported in the public release")
    if mid not in _MODEL_SPECS:
        raise ValueError(f"model_id must be in [1..5], got: {mid}")
    return _MODEL_SPECS[mid]
