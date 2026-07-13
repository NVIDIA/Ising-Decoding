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
Public model registry for the public release.

External users choose `model_id` in {1..5} (surface/color) or "B" (color only).
This registry maps model_id to:
- the underlying architecture parameters (num_filters, kernel_size for the
  convolutional models; a full `model_overrides` block for non-convolutional
  models such as the cascade/bottleneck model "B")
- the model receptive field R (in rounds / distance units)

Receptive field convention matches `compare_receptive_field_with_window_data`
in `code/training/utils.py`:
  R = 1 + sum_i (k_i - 1)   for kernel sizes k_i (assumed odd, with same-padding)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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
    model_id: Union[int, str]
    num_filters: List[int]
    kernel_size: List[int]
    receptive_field: int
    model_version: str = "predecoder_memory_v1"
    # Non-convolutional models (e.g. the cascade/bottleneck model "B") are not
    # described by num_filters/kernel_size. For those, `model_overrides` carries
    # the full `model.*` block that should be written into the merged config.
    # When set, it takes precedence over num_filters/kernel_size.
    model_overrides: Optional[Dict[str, Any]] = None


# Full model.* block for the cascade/bottleneck model "B" (color code only).
# Kept as a module constant so the spec below stays a flat list of kwargs.
# This matches the paper's Model B: a plain Conv3d stem + 5 bottleneck blocks
# (~2.94M params). `activation` is informational — the cascade model uses SiLU
# internally regardless of this field.
_MODEL_B_OVERRIDES = {
    "version": "predecoder_memory_cascade",
    "plain_stem": True,
    "dropout_p": 0.01,
    "activation": "silu",
    "embed_dim": 512,
    "num_blocks": 6,
    "bottleneck_ratio": 4,
    "kernel_size": 3,
    "input_channels": 4,
    "out_channels": 4,
}

_MODEL_SPECS: Dict[Union[int, str], PublicModelSpec] = {
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
    # Model B: cascade architecture (predecoder_memory_cascade), matching the
    # paper's Model B — plain Conv3d stem + 5 bottleneck blocks, ~2.94M params.
    # Color code only; carries a full model override block instead of
    # num_filters/kernel_size. R=13 keeps it within the color receptive-field
    # limit. LR is fixed to the color default (1e-5)
    # in config_validator.py.
    "B":
        PublicModelSpec(
            model_id="B",
            num_filters=[],
            kernel_size=[],
            receptive_field=13,
            model_version="predecoder_memory_cascade",
            model_overrides=_MODEL_B_OVERRIDES,
        ),
}


def _normalize_model_id(model_id: Union[int, str]) -> Union[int, str]:
    """Normalize a public model_id to its registry key.

    Numeric ids (and numeric-looking strings, e.g. "1") map to ints; the
    non-numeric alias for the cascade model normalizes to upper-case "B".
    """
    if isinstance(model_id, str):
        key = model_id.strip()
        try:
            return int(key)
        except ValueError:
            return key.upper()
    return int(model_id)


def get_model_spec(model_id: Union[int, str]) -> PublicModelSpec:
    """Return the public model spec for a given model_id (1..5 or "B")."""
    try:
        key = _normalize_model_id(model_id)
    except Exception as e:
        raise ValueError(f"model_id must be one of [1..5] or 'B', got: {model_id!r}") from e
    if key == 0:
        raise ValueError("model_id=0 is not supported in the public release")
    if key not in _MODEL_SPECS:
        raise ValueError(f"model_id must be one of [1..5] or 'B', got: {model_id!r}")
    return _MODEL_SPECS[key]
