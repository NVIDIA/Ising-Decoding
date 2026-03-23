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
Surface code stabilizer-to-data qubit mappings and weight functions.

This module contains functions for mapping stabilizer syndromes to data qubit 
indices on a DxD grid, supporting all four surface code orientations:
XV, XH, ZV, ZH.

Key functions:
- compute_stabX_to_data_index_map: Map X stabilizers to data qubits
- compute_stabZ_to_data_index_map: Map Z stabilizers to data qubits
- normalized_weight_mapping_Xstab_memory: Normalized weights for X stabs
- normalized_weight_mapping_Zstab_memory: Normalized weights for Z stabs
- reshape_Xstabilizers_to_grid_vectorized: Reshape X stabs to DxD grid
- reshape_Zstabilizers_to_grid_vectorized: Reshape Z stabs to DxD grid
"""

import torch


def _compute_stabX_to_data_XV(distance):
    """
    X stabilizer mapping for XV orientation (backward compatible with original).
    Pattern: boundary stabilizers at left/right edges, bulk in checkerboard.
    """
    cols = (distance + 1) // 2
    rows = distance - 1

    total_stabs = cols * rows
    stab_to_data_index_map = torch.empty(total_stabs, dtype=torch.int32)

    data_qubit_index = 0
    idx = 0

    for rr in range(rows):
        for cc in range(cols):
            stab_to_data_index_map[idx] = data_qubit_index
            idx += 1

            # Update data_qubit_index based on location
            if rr % 2 == 0:
                if cc == cols - 1:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2
            else:
                if cc == 0:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2

    return stab_to_data_index_map


def _compute_stab_to_data_from_parity_X_boundary_aware(
    parity_matrix: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    Compute X stabilizer-to-data mapping from parity matrix with BOUNDARY-AWARE selection.
    
    Boundary convention (applies to weight-2 stabilizers, per orientations_mappings.txt):
    - Horizontal pairs (top/bottom boundaries) → pick LEFT qubit (smallest col)
    - Vertical pairs (left/right boundaries) → pick TOP qubit (smallest row)
    - Bulk (weight-4) → pick top-left (smallest row, then smallest col)
    
    Args:
        parity_matrix: Parity check matrix (num_stabilizers, num_data_qubits)
        distance: Code distance for row/col computation
        
    Returns:
        Tensor of shape (num_stabilizers,) mapping stab_idx -> data_qubit_idx
    """
    num_stabs = parity_matrix.shape[0]
    stab_to_data = torch.empty(num_stabs, dtype=torch.int32)

    for stab_idx in range(num_stabs):
        support = torch.nonzero(parity_matrix[stab_idx], as_tuple=True)[0].tolist()
        positions = [(idx // distance, idx % distance, idx) for idx in support]

        if len(support) == 2:  # Boundary stabilizer
            rows = [p[0] for p in positions]
            if rows[0] == rows[1]:  # Horizontal pair (top or bottom boundary)
                # Both top and bottom boundaries use LEFT (smallest col) per orientations_mappings.txt
                # top-boundary: bottom-left, bottom-boundary: top-left → both mean LEFT
                positions.sort(key=lambda x: x[1])  # Sort by col ascending (LEFT)
            else:  # Vertical pair (left or right boundary for X in XV/ZH)
                # Both left and right boundaries use TOP (smallest row) per orientations_mappings.txt
                # left-boundary: top-right, right-boundary: top-left → both mean TOP
                positions.sort(key=lambda x: x[0])  # Sort by row ascending (TOP)
        else:  # Bulk stabilizer → top-left
            positions.sort(key=lambda x: (x[0], x[1]))

        stab_to_data[stab_idx] = positions[0][2]

    return stab_to_data


def _compute_stab_to_data_from_parity_Z_boundary_aware(
    parity_matrix: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    Compute Z stabilizer-to-data mapping from parity matrix with BOUNDARY-AWARE selection.
    
    Boundary convention (applies to weight-2 stabilizers, per orientations_mappings.txt):
    - Vertical pairs (left/right boundaries) → pick TOP qubit (smallest row)
    - Horizontal pairs (top/bottom boundaries) → pick RIGHT qubit (largest col)
    - Bulk (weight-4) → pick top-right (smallest row, then largest col)
    
    Args:
        parity_matrix: Parity check matrix (num_stabilizers, num_data_qubits)
        distance: Code distance for row/col computation
        
    Returns:
        Tensor of shape (num_stabilizers,) mapping stab_idx -> data_qubit_idx
    """
    num_stabs = parity_matrix.shape[0]
    stab_to_data = torch.empty(num_stabs, dtype=torch.int32)

    for stab_idx in range(num_stabs):
        support = torch.nonzero(parity_matrix[stab_idx], as_tuple=True)[0].tolist()
        positions = [(idx // distance, idx % distance, idx) for idx in support]

        if len(support) == 2:  # Boundary stabilizer
            cols = [p[1] for p in positions]
            if cols[0] == cols[1]:  # Vertical pair (left or right boundary)
                # Both left and right boundaries use TOP (smallest row) per orientations_mappings.txt
                # left-boundary: top-right, right-boundary: top-left → both mean TOP
                positions.sort(key=lambda x: x[0])  # Sort by row ascending (TOP)
            else:  # Horizontal pair (top or bottom boundary for Z in XV/ZH)
                # Both top and bottom boundaries use RIGHT (largest col) per orientations_mappings.txt
                # top-boundary: bottom-right, bottom-boundary: top-right → both mean RIGHT
                positions.sort(key=lambda x: -x[1])  # Sort by col descending (RIGHT)
        else:  # Bulk stabilizer → top-right
            positions.sort(key=lambda x: (x[0], -x[1]))

        stab_to_data[stab_idx] = positions[0][2]

    return stab_to_data


def _compute_stab_to_data_from_parity_topleft(
    parity_matrix: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    DEPRECATED: Use _compute_stab_to_data_from_parity_X_boundary_aware instead.
    Kept for backward compatibility reference only.
    """
    return _compute_stab_to_data_from_parity_X_boundary_aware(parity_matrix, distance)


def _compute_stab_to_data_from_parity_topright(
    parity_matrix: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    DEPRECATED: Use _compute_stab_to_data_from_parity_Z_boundary_aware instead.
    Kept for backward compatibility reference only.
    """
    return _compute_stab_to_data_from_parity_Z_boundary_aware(parity_matrix, distance)


def compute_stabX_to_data_index_map(distance, rotation='XV'):
    """
    Returns a torch tensor mapping stab_tensor_index → data_qubit_index
    for X stabilizers of a rotated surface code of given distance.
    
    Boundary convention:
    - Top boundary → RIGHT qubit, Bottom boundary → LEFT qubit
    - Left boundary → TOP qubit, Right boundary → BOTTOM qubit
    - Bulk → top-left (smallest row, then smallest col)
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    
    Returns:
        Torch tensor of shape (num_X_stabs,) mapping stabilizer index to data qubit index
    """
    rotation = rotation.upper()
    if rotation == 'XV':
        # Use optimized hardcoded pattern for XV (backward compatible)
        return _compute_stabX_to_data_XV(distance)
    elif rotation in ('XH', 'ZV', 'ZH'):
        # For other orientations, compute from parity matrix with boundary-aware selection
        from qec.surface_code.memory_circuit import SurfaceCode
        first_bulk = rotation[0]  # 'X' or 'Z'
        rotated = rotation[1]  # 'V' or 'H'
        code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rotated)
        hx = torch.tensor(code.hx, dtype=torch.int32)
        return _compute_stab_to_data_from_parity_X_boundary_aware(hx, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


def _normalized_weight_mapping_Xstab_XV(distance):
    """
    Normalized weights for X stabilizers in XV/ZH orientation.
    """
    cols = (distance + 1) // 2
    rows = distance - 1

    data_qubit_index = 0
    out = torch.zeros(distance * distance)

    for rr in range(rows):
        for cc in range(cols):
            if rr % 2 == 0:
                if cc == cols - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2
            else:
                if cc == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2

    return out


def _normalized_weight_mapping_Xstab_XH(distance):
    """
    Normalized weights for X stabilizers in XH/ZV orientation.
    """
    cols = distance - 1
    rows = (distance + 1) // 2

    data_qubit_index = 0
    out = torch.zeros(distance * distance)

    for cc in range(cols):
        data_qubit_top = data_qubit_index
        for rr in range(rows):
            if cc % 2 == 0:
                if rr == rows - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2 * distance
            else:
                if rr == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += distance
                elif rr == rows - 1:
                    out[data_qubit_index] = 1
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2 * distance

    return out


def _compute_normalized_weight_from_parity(
    parity_matrix: torch.Tensor, stab_to_data: torch.Tensor, distance: int
) -> torch.Tensor:
    """
    Compute normalized weights from parity matrix and stab-to-data mapping.
    
    Boundary stabilizers (2-qubit support) get weight 0.5, bulk (4-qubit) get 1.0.
    
    Args:
        parity_matrix: Parity check matrix (num_stabilizers, num_data_qubits)
        stab_to_data: Stabilizer to data qubit index mapping
        distance: Code distance
        
    Returns:
        Tensor of shape (distance²,) with normalized weights at mapped positions
    """
    num_stabs = parity_matrix.shape[0]
    out = torch.zeros(distance * distance)

    for stab_idx in range(num_stabs):
        support_size = parity_matrix[stab_idx].sum().item()
        data_idx = int(stab_to_data[stab_idx])
        # Boundary stabilizers have 2 qubits, bulk have 4
        out[data_idx] = 0.5 if support_size == 2 else 1.0

    return out


def normalized_weight_mapping_Xstab_memory(distance, rotation='XV'):
    """
    Returns normalized weights of X-type stabilizers mapped to data qubits.
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    
    Returns:
        Torch tensor of shape (distance²,) with normalized weights
    """
    rotation = rotation.upper()
    if rotation == 'XV':
        # Use optimized hardcoded pattern for XV (backward compatible)
        return _normalized_weight_mapping_Xstab_XV(distance)
    elif rotation in ('XH', 'ZV', 'ZH'):
        # For other orientations, compute from parity matrix using TOP-LEFT selection
        from qec.surface_code.memory_circuit import SurfaceCode
        first_bulk = rotation[0]  # 'X' or 'Z'
        rotated = rotation[1]  # 'V' or 'H'
        code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rotated)
        hx = torch.tensor(code.hx, dtype=torch.int32)
        stab_to_data = _compute_stab_to_data_from_parity_topleft(hx, distance)
        return _compute_normalized_weight_from_parity(hx, stab_to_data, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


def compute_data_to_stabX_index_map(distance, rotation='XV'):
    """
    Returns a list `data_to_stab_index_map` such that:
    if stab_to_data_index_map[stab_idx] = data_idx,
    then data_to_stab_index_map[data_idx] = stab_idx
    (for data_idx values that are touched by an X stabilizer)
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    """
    stab_to_data_index_map = compute_stabX_to_data_index_map(distance, rotation)

    data_to_stab_index_map = [-1] * (distance * distance)  # Default -1 for positions not used
    for stab_idx, data_idx in enumerate(stab_to_data_index_map):
        data_to_stab_index_map[data_idx] = stab_idx

    return data_to_stab_index_map


def _compute_stabZ_to_data_XV(distance):
    """
    Z stabilizer mapping for XV and ZH orientations.
    Pattern: boundary stabilizers at top/bottom edges, bulk in checkerboard.
    """
    cols = distance - 1
    rows = (distance + 1) // 2

    total_stabs = cols * rows
    stab_to_data_index_map = torch.empty(total_stabs, dtype=torch.int32)

    data_qubit_index = 1
    stab_idx = 0

    for cc in range(cols):
        data_qubit_top = data_qubit_index
        for rr in range(rows):
            stab_to_data_index_map[stab_idx] = data_qubit_index
            stab_idx += 1

            # Update data_qubit_index based on location
            if cc % 2 == 0:
                if rr == 0:
                    data_qubit_index += distance
                elif rr == rows - 1:
                    data_qubit_index = data_qubit_top + 1
                else:
                    data_qubit_index += 2 * distance
            else:
                if rr == rows - 1:
                    data_qubit_index = data_qubit_top + 1
                else:
                    data_qubit_index += 2 * distance

    return stab_to_data_index_map


def _compute_stabZ_to_data_XH(distance):
    """
    Z stabilizer mapping for XH and ZV orientations.
    Pattern: boundary stabilizers at left/right edges.
    """
    cols = (distance + 1) // 2
    rows = distance - 1

    total_stabs = cols * rows
    stab_to_data_index_map = torch.empty(total_stabs, dtype=torch.int32)

    data_qubit_index = 1
    idx = 0

    for rr in range(rows):
        for cc in range(cols):
            stab_to_data_index_map[idx] = data_qubit_index
            idx += 1

            # Update data_qubit_index based on location
            if rr % 2 == 0:
                if cc == 0:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2
            else:
                if cc == cols - 1:
                    data_qubit_index += 1
                else:
                    data_qubit_index += 2

    return stab_to_data_index_map


def compute_stabZ_to_data_index_map(distance, rotation='XV'):
    """
    Returns a torch tensor mapping stab_tensor_index → data_qubit_index
    for Z stabilizers of a rotated surface code of given distance.
    
    Boundary convention:
    - Left boundary → TOP qubit, Right boundary → BOTTOM qubit
    - Top boundary → RIGHT qubit, Bottom boundary → LEFT qubit
    - Bulk → top-right (smallest row, then largest col)
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    
    Returns:
        Torch tensor of shape (num_Z_stabs,) mapping stabilizer index to data qubit index
    """
    rotation = rotation.upper()
    if rotation == 'XV':
        # Use optimized hardcoded pattern for XV (backward compatible)
        return _compute_stabZ_to_data_XV(distance)
    elif rotation in ('XH', 'ZV', 'ZH'):
        # For other orientations, compute from parity matrix with boundary-aware selection
        from qec.surface_code.memory_circuit import SurfaceCode
        first_bulk = rotation[0]  # 'X' or 'Z'
        rotated = rotation[1]  # 'V' or 'H'
        code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rotated)
        hz = torch.tensor(code.hz, dtype=torch.int32)
        return _compute_stab_to_data_from_parity_Z_boundary_aware(hz, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


def _normalized_weight_mapping_Zstab_XV(distance):
    """
    Normalized weights for Z stabilizers in XV/ZH orientation.
    """
    cols = distance - 1
    rows = (distance + 1) // 2

    data_qubit_index = 1
    out = torch.zeros(distance * distance)

    for cc in range(cols):
        data_qubit_top = data_qubit_index
        for rr in range(rows):
            if (cc % 2) == 0:
                if (rr == 0):
                    out[data_qubit_index] = 0.5
                    data_qubit_index = data_qubit_index + distance
                elif (rr == rows - 1):
                    out[data_qubit_index] = 1
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index = data_qubit_index + 2 * distance
            else:
                if (rr == rows - 1):
                    out[data_qubit_index] = 0.5
                    data_qubit_index = data_qubit_top + 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index = data_qubit_index + 2 * distance

    return out


def _normalized_weight_mapping_Zstab_XH(distance):
    """
    Normalized weights for Z stabilizers in XH/ZV orientation.
    """
    cols = (distance + 1) // 2
    rows = distance - 1

    data_qubit_index = 1
    out = torch.zeros(distance * distance)

    for rr in range(rows):
        for cc in range(cols):
            if rr % 2 == 0:
                if cc == 0:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2
            else:
                if cc == cols - 1:
                    out[data_qubit_index] = 0.5
                    data_qubit_index += 1
                else:
                    out[data_qubit_index] = 1
                    data_qubit_index += 2

    return out


def normalized_weight_mapping_Zstab_memory(distance, rotation='XV'):
    """
    Returns normalized weights of Z-type stabilizers mapped to data qubits.
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    
    Returns:
        Torch tensor of shape (distance²,) with normalized weights
    """
    rotation = rotation.upper()
    if rotation == 'XV':
        # Use optimized hardcoded pattern for XV (backward compatible)
        return _normalized_weight_mapping_Zstab_XV(distance)
    elif rotation in ('XH', 'ZV', 'ZH'):
        # For other orientations, compute from parity matrix using TOP-RIGHT selection
        from qec.surface_code.memory_circuit import SurfaceCode
        first_bulk = rotation[0]  # 'X' or 'Z'
        rotated = rotation[1]  # 'V' or 'H'
        code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rotated)
        hz = torch.tensor(code.hz, dtype=torch.int32)
        stab_to_data = _compute_stab_to_data_from_parity_topright(hz, distance)
        return _compute_normalized_weight_from_parity(hz, stab_to_data, distance)
    else:
        raise ValueError(f"Invalid rotation '{rotation}'. Must be one of: XV, XH, ZV, ZH")


def compute_data_to_stabZ_index_map(distance, rotation='XV'):
    """
    Returns a list `data_to_stab_index_map` such that:
    if stab_to_data_index_map[stab_idx] = data_idx,
    then data_to_stab_index_map[data_idx] = stab_idx
    (for data_idx values that are touched by a Z stabilizer)
    
    Args:
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    """
    stab_to_data_index_map = compute_stabZ_to_data_index_map(distance, rotation)

    data_to_stab_index_map = [-1] * (distance * distance)  # Default -1 for positions not used
    for stab_idx, data_idx in enumerate(stab_to_data_index_map):
        data_to_stab_index_map[data_idx] = stab_idx

    return data_to_stab_index_map


def map_grid_to_stabilizer_tensor(
    grid_tensor: torch.Tensor, stab_indices: torch.Tensor
) -> torch.Tensor:
    """
    Maps grid-shaped data (B, T, D, D) → (B, num_stabs, T)
    """
    B, T, D, _ = grid_tensor.shape
    flat_grid = grid_tensor.reshape(B, T, D * D)  # Shape: (B, T, D²)
    stab_tensor = torch.index_select(
        flat_grid, dim=2, index=stab_indices
    )  # Shape: (B, T, num_stabs)
    return stab_tensor.permute(0, 2, 1).contiguous()  # Shape: (B, num_stabs, T)


def reshape_Xstabilizers_to_grid_vectorized(stab_tensor, distance, rotation='XV'):
    """
    Vectorized remapping of X stabilizers to grid without explicit loops.
    
    Args:
        stab_tensor: Tensor of shape (B, num_stabs, T) or (num_stabs, T)
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    
    Returns:
        Tensor of shape (B, distance², T) or (distance², T) with stabilizers mapped to grid positions
    """
    if stab_tensor.ndim == 2:  # No batch
        stab_tensor = stab_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, num_stabs, T = stab_tensor.shape
    idx_map = torch.as_tensor(
        compute_stabX_to_data_index_map(distance, rotation),
        dtype=torch.long,
        device=stab_tensor.device
    )

    out = torch.zeros(B, distance * distance, T, device=stab_tensor.device, dtype=stab_tensor.dtype)
    out[:, idx_map, :] = stab_tensor
    return out[0] if squeeze_output else out


def reshape_Zstabilizers_to_grid_vectorized(stab_tensor, distance, rotation='XV'):
    """
    Vectorized remapping of Z stabilizers to grid without explicit loops.
    
    Args:
        stab_tensor: Tensor of shape (B, num_stabs, T) or (num_stabs, T)
        distance: Code distance (odd integer)
        rotation: Code rotation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV' for backward compatibility.
    
    Returns:
        Tensor of shape (B, distance², T) or (distance², T) with stabilizers mapped to grid positions
    """
    if stab_tensor.ndim == 2:  # No batch
        stab_tensor = stab_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    B, num_stabs, T = stab_tensor.shape
    idx_map = torch.as_tensor(
        compute_stabZ_to_data_index_map(distance, rotation),
        dtype=torch.long,
        device=stab_tensor.device
    )

    out = torch.zeros(B, distance * distance, T, device=stab_tensor.device, dtype=stab_tensor.dtype)
    out[:, idx_map, :] = stab_tensor
    return out[0] if squeeze_output else out


def construct_X_stab_Parity_check_Mat(distance):
    """
    Constructs the H_X stabilizer parity check matrix for the surface code.
    """
    m = (distance**2 - 1) // 2  # number of stabilizers
    n = distance**2  # number of qubits
    colLen = (distance + 1) // 2
    H = torch.zeros((m, n))

    q1 = 0
    q2 = distance
    stabCount = 0
    for rows in range(distance - 1):
        for cols in range(colLen):
            if rows % 2 == 0:
                if cols == colLen - 1:
                    # Weight-2 X stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    q1 += 1
                    q2 += 1
                else:
                    # Weight-4 X stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    H[stabCount, q1 + 1] = 1
                    H[stabCount, q2 + 1] = 1
                    q1 += 2
                    q2 += 2
            else:
                if cols == 0:
                    # Weight-2 X stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    q1 += 1
                    q2 += 1
                else:
                    # Weight-4 X stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    H[stabCount, q1 + 1] = 1
                    H[stabCount, q2 + 1] = 1
                    q1 += 2
                    q2 += 2
            stabCount += 1
    return H


def construct_Z_stab_Parity_check_Mat(distance):
    """
    Constructs the H_Z stabilizer parity check matrix for the surface code.
    """
    m = (distance**2 - 1) // 2  # number of stabilizers
    n = distance**2  # number of qubits

    colLen = distance - 1
    rowLen = (distance + 1) // 2
    H = torch.zeros((m, n))

    q1 = 0
    q2 = 1
    stabCount = 0
    for cols in range(colLen):
        q1Top = q1
        for rows in range(rowLen):
            if (cols % 2 == 0):
                if (rows == 0):
                    # Weight-2 Z stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    q1 += distance
                    q2 += distance
                elif (rows == rowLen - 1):
                    # Weight-4 Z stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    H[stabCount, q1 + distance] = 1
                    H[stabCount, q2 + distance] = 1

                    q1 = q1Top + 1
                    q2 = q1 + 1
                else:
                    # Weight-4 Z stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    H[stabCount, q1 + distance] = 1
                    H[stabCount, q2 + distance] = 1
                    q1 += 2 * distance
                    q2 += 2 * distance
            else:
                if (rows == rowLen - 1):
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1

                    q1 = q1Top + 1
                    q2 = q1 + 1
                else:
                    # Weight-4 Z stabilizer
                    H[stabCount, q1] = 1
                    H[stabCount, q2] = 1
                    H[stabCount, q1 + distance] = 1
                    H[stabCount, q2 + distance] = 1
                    q1 += 2 * distance
                    q2 += 2 * distance

            stabCount += 1

    return H
