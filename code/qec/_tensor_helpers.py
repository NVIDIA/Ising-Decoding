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
"""Tensor-side helpers shared between color-code and surface-code detector
input modules. Pure Torch, no code-family specialization."""

import torch


def _one_hot_map(indices: torch.Tensor, output_width: int) -> torch.Tensor:
    out = torch.zeros((int(indices.numel()), int(output_width)), dtype=torch.float32)
    out[torch.arange(indices.numel(), dtype=torch.long), indices.to(dtype=torch.long)] = 1.0
    return out


def _grid_to_padded_stab(indices: torch.Tensor, output_width: int) -> torch.Tensor:
    out = torch.zeros((int(output_width),), dtype=torch.long)
    out[indices.to(dtype=torch.long)] = torch.arange(1, int(indices.numel()) + 1)
    return out
