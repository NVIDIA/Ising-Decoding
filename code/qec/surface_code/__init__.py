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
Surface code module.

Contains:
- Circuit generation: MemoryCircuit, SurfaceCode (Stim-based)
- Data mappings: stabilizer-to-data qubit mappings and related functions
- Stim utilities: stim_utils
"""

# Circuit generation and simulation
from qec.surface_code.memory_circuit import MemoryCircuit, SurfaceCode

# Data mappings
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
    reshape_Xstabilizers_to_grid_vectorized,
    reshape_Zstabilizers_to_grid_vectorized,
    compute_data_to_stabX_index_map,
    compute_data_to_stabZ_index_map,
    map_grid_to_stabilizer_tensor,
    construct_X_stab_Parity_check_Mat,
    construct_Z_stab_Parity_check_Mat,
)
