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

from .color_code import ColorCode

from .data_mapping import (
    get_stab_to_grid_flat_index,
    get_data_to_grid_flat_index,
    normalized_weight_mapping_stab,
    reshape_stabilizers_to_grid,
    reshape_stabilizers_to_grid_2d,
    reshape_data_to_grid,
    reshape_data_to_grid_2d,
    reshape_stabilizers_to_grid_vectorized,
    reshape_data_to_grid_vectorized,
    map_grid_to_stabilizer_tensor,
    map_grid_to_data_tensor,
    get_stabilizer_presence_mask,
    get_data_presence_mask,
    compute_stab_to_data_index_map,
    compute_data_to_stab_index_map,
    get_parity_matrix_data_only,
)

__all__ = [
    'ColorCode',
    # Data mapping functions
    'get_stab_to_grid_flat_index',
    'get_data_to_grid_flat_index',
    'normalized_weight_mapping_stab',
    'reshape_stabilizers_to_grid',
    'reshape_stabilizers_to_grid_2d',
    'reshape_data_to_grid',
    'reshape_data_to_grid_2d',
    'reshape_stabilizers_to_grid_vectorized',
    'reshape_data_to_grid_vectorized',
    'map_grid_to_stabilizer_tensor',
    'map_grid_to_data_tensor',
    'get_stabilizer_presence_mask',
    'get_data_presence_mask',
    'compute_stab_to_data_index_map',
    'compute_data_to_stab_index_map',
    'get_parity_matrix_data_only',
]
