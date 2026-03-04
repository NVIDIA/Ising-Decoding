# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
