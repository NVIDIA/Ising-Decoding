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
Color Code data mapping functions for CNN embedding.

This module provides functions for mapping stabilizer syndromes and data qubits
to a rectangular grid suitable for CNN pre-decoder models.

Grid Embedding:
- Grid size: (n_rows, n_cols) where n_rows = d + (d-1)//2, n_cols = d
- Data qubits are embedded in a triangular pattern within this rectangle
- Stabilizer syndromes are mapped to their associated data qubit positions

Key differences from surface code:
- Color code has SAME X and Z stabilizers (they share plaquettes)
- Grid is (n_rows, n_cols) not (D, D)
- Mapping is pre-computed in ColorCode class

Key functions:
- get_stab_to_grid_flat_index: Get flat grid indices for stabilizers
- get_data_to_grid_flat_index: Get flat grid indices for data qubits
- normalized_weight_mapping_stab: Normalized weights for stabilizers
- reshape_stabilizers_to_grid: Reshape stabilizers to grid
- reshape_data_to_grid: Reshape data qubits to grid
- map_grid_to_stabilizer_tensor: Map grid back to stabilizer tensor
- map_grid_to_data_tensor: Map grid back to data tensor
"""

import torch
import numpy as np
from typing import Union, Optional

# Type alias for ColorCode to avoid circular import
ColorCodeType = "ColorCode"


def get_stab_to_grid_flat_index(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get flat grid indices for stabilizer-to-grid mapping.
    
    Each stabilizer maps to a single grid position based on its associated
    data qubit (determined by ColorCode's syndrome mapping rules).
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (num_plaquettes,) with flat grid indices
    """
    indices = color_code.get_syndrome_grid_indices()  # (num_plaquettes,)
    return torch.tensor(indices, dtype=torch.long)


def get_data_to_grid_flat_index(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get flat grid indices for data qubit-to-grid mapping.
    
    Each data qubit maps to its position in the rectangular grid.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (num_data,) with flat grid indices
    """
    n_cols = color_code.n_cols
    data_flat = []
    for q in range(color_code.num_data):
        grid_row, grid_col = color_code.qubit_to_grid[q]
        data_flat.append(grid_row * n_cols + grid_col)
    return torch.tensor(data_flat, dtype=torch.long)


def normalized_weight_mapping_stab(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get normalized weights for stabilizers mapped to grid.
    
    Weight-6 bulk plaquettes get weight 1.0, weight-4 boundary plaquettes get 0.5.
    This is analogous to surface code's boundary vs bulk normalization.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (n_rows * n_cols,) with normalized weights at mapped positions
    """
    n_rows, n_cols = color_code.n_rows, color_code.n_cols
    out = torch.zeros(n_rows * n_cols, dtype=torch.float32)

    stab_indices = get_stab_to_grid_flat_index(color_code)

    for plaq_idx, plaq in enumerate(color_code.plaquettes):
        weight = plaq['weight']
        flat_idx = int(stab_indices[plaq_idx])
        # Weight-6 bulk → 1.0, weight-4 boundary → 0.5
        out[flat_idx] = 1.0 if weight == 6 else 0.5

    return out


def reshape_stabilizers_to_grid(
    stab_tensor: torch.Tensor, n_rows: int, n_cols: int, stab_indices: torch.Tensor
) -> torch.Tensor:
    """
    Reshape stabilizer tensor from flat stabilizer order to grid.
    
    Args:
        stab_tensor: Tensor of shape (B, num_stabs, T) or (num_stabs, T)
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        stab_indices: Flat grid indices of shape (num_stabs,)
        
    Returns:
        Tensor of shape (B, n_rows*n_cols, T) or (n_rows*n_cols, T)
    """
    squeeze_output = False
    if stab_tensor.ndim == 2:  # (num_stabs, T)
        stab_tensor = stab_tensor.unsqueeze(0)
        squeeze_output = True

    B, num_stabs, T = stab_tensor.shape
    device = stab_tensor.device
    dtype = stab_tensor.dtype

    out = torch.zeros(B, n_rows * n_cols, T, device=device, dtype=dtype)
    idx = stab_indices.to(device)

    # Scatter stabilizers to grid positions
    idx_expanded = idx.view(1, -1, 1).expand(B, -1, T)
    out.scatter_(1, idx_expanded, stab_tensor)

    return out.squeeze(0) if squeeze_output else out


def reshape_stabilizers_to_grid_2d(
    stab_tensor: torch.Tensor, n_rows: int, n_cols: int, stab_indices: torch.Tensor
) -> torch.Tensor:
    """
    Reshape stabilizer tensor to 2D grid format.
    
    Args:
        stab_tensor: Tensor of shape (B, num_stabs, T) or (num_stabs, T)
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        stab_indices: Flat grid indices of shape (num_stabs,)
        
    Returns:
        Tensor of shape (B, T, n_rows, n_cols) or (T, n_rows, n_cols)
    """
    flat = reshape_stabilizers_to_grid(stab_tensor, n_rows, n_cols, stab_indices)

    if flat.ndim == 2:  # (n_rows*n_cols, T)
        return flat.view(n_rows, n_cols, -1).permute(2, 0, 1).contiguous()
    else:  # (B, n_rows*n_cols, T)
        B, _, T = flat.shape
        return flat.view(B, n_rows, n_cols, T).permute(0, 3, 1, 2).contiguous()


def reshape_data_to_grid(
    data_tensor: torch.Tensor, n_rows: int, n_cols: int, data_indices: torch.Tensor
) -> torch.Tensor:
    """
    Reshape data qubit tensor from flat data order to grid.
    
    Args:
        data_tensor: Tensor of shape (B, num_data, T) or (num_data, T)
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        data_indices: Flat grid indices of shape (num_data,)
        
    Returns:
        Tensor of shape (B, n_rows*n_cols, T) or (n_rows*n_cols, T)
    """
    squeeze_output = False
    if data_tensor.ndim == 2:  # (num_data, T)
        data_tensor = data_tensor.unsqueeze(0)
        squeeze_output = True

    B, num_data, T = data_tensor.shape
    device = data_tensor.device
    dtype = data_tensor.dtype

    out = torch.zeros(B, n_rows * n_cols, T, device=device, dtype=dtype)
    idx = data_indices.to(device)

    # Scatter data to grid positions
    idx_expanded = idx.view(1, -1, 1).expand(B, -1, T)
    out.scatter_(1, idx_expanded, data_tensor)

    return out.squeeze(0) if squeeze_output else out


def reshape_data_to_grid_2d(
    data_tensor: torch.Tensor, n_rows: int, n_cols: int, data_indices: torch.Tensor
) -> torch.Tensor:
    """
    Reshape data qubit tensor to 2D grid format.
    
    Args:
        data_tensor: Tensor of shape (B, num_data, T) or (num_data, T)
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        data_indices: Flat grid indices of shape (num_data,)
        
    Returns:
        Tensor of shape (B, T, n_rows, n_cols) or (T, n_rows, n_cols)
    """
    flat = reshape_data_to_grid(data_tensor, n_rows, n_cols, data_indices)

    if flat.ndim == 2:  # (n_rows*n_cols, T)
        return flat.view(n_rows, n_cols, -1).permute(2, 0, 1).contiguous()
    else:  # (B, n_rows*n_cols, T)
        B, _, T = flat.shape
        return flat.view(B, n_rows, n_cols, T).permute(0, 3, 1, 2).contiguous()


def map_grid_to_stabilizer_tensor(
    grid_tensor: torch.Tensor, stab_indices: torch.Tensor
) -> torch.Tensor:
    """
    Map grid-shaped data back to stabilizer tensor.
    
    Args:
        grid_tensor: Tensor of shape (B, T, n_rows, n_cols)
        stab_indices: Flat grid indices of shape (num_stabs,)
        
    Returns:
        Tensor of shape (B, num_stabs, T)
    """
    B, T, n_rows, n_cols = grid_tensor.shape
    device = grid_tensor.device

    # Flatten spatial dimensions
    flat_grid = grid_tensor.view(B, T, n_rows * n_cols)  # (B, T, n_rows*n_cols)

    # Index select stabilizer positions
    idx = stab_indices.to(device)
    stab_tensor = torch.index_select(flat_grid, dim=2, index=idx)  # (B, T, num_stabs)

    return stab_tensor.permute(0, 2, 1).contiguous()  # (B, num_stabs, T)


def map_grid_to_data_tensor(grid_tensor: torch.Tensor, data_indices: torch.Tensor) -> torch.Tensor:
    """
    Map grid-shaped data back to data qubit tensor.
    
    Args:
        grid_tensor: Tensor of shape (B, T, n_rows, n_cols)
        data_indices: Flat grid indices of shape (num_data,)
        
    Returns:
        Tensor of shape (B, num_data, T)
    """
    B, T, n_rows, n_cols = grid_tensor.shape
    device = grid_tensor.device

    # Flatten spatial dimensions
    flat_grid = grid_tensor.view(B, T, n_rows * n_cols)  # (B, T, n_rows*n_cols)

    # Index select data qubit positions
    idx = data_indices.to(device)
    data_tensor = torch.index_select(flat_grid, dim=2, index=idx)  # (B, T, num_data)

    return data_tensor.permute(0, 2, 1).contiguous()  # (B, num_data, T)


# ============================================================================
# Convenience functions using ColorCode directly
# ============================================================================


def reshape_stabilizers_to_grid_vectorized(
    stab_tensor: torch.Tensor, color_code: ColorCodeType
) -> torch.Tensor:
    """
    Vectorized reshaping of stabilizers to grid using ColorCode.
    
    Args:
        stab_tensor: Tensor of shape (B, num_stabs, T) or (num_stabs, T)
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (B, n_rows*n_cols, T) or (n_rows*n_cols, T)
    """
    stab_indices = get_stab_to_grid_flat_index(color_code)
    return reshape_stabilizers_to_grid(
        stab_tensor, color_code.n_rows, color_code.n_cols, stab_indices
    )


def reshape_data_to_grid_vectorized(
    data_tensor: torch.Tensor, color_code: ColorCodeType
) -> torch.Tensor:
    """
    Vectorized reshaping of data qubits to grid using ColorCode.
    
    Args:
        data_tensor: Tensor of shape (B, num_data, T) or (num_data, T)
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (B, n_rows*n_cols, T) or (n_rows*n_cols, T)
    """
    data_indices = get_data_to_grid_flat_index(color_code)
    return reshape_data_to_grid(data_tensor, color_code.n_rows, color_code.n_cols, data_indices)


# ============================================================================
# Grid presence masks
# ============================================================================


def get_stabilizer_presence_mask(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get a binary mask indicating which grid positions contain stabilizers.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (n_rows, n_cols) with 1s at stabilizer positions
    """
    n_rows, n_cols = color_code.n_rows, color_code.n_cols
    mask = torch.zeros(n_rows, n_cols, dtype=torch.float32)

    stab_indices = get_stab_to_grid_flat_index(color_code)
    flat_mask = torch.zeros(n_rows * n_cols, dtype=torch.float32)
    flat_mask[stab_indices] = 1.0

    return flat_mask.view(n_rows, n_cols)


def get_data_presence_mask(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get a binary mask indicating which grid positions contain data qubits.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (n_rows, n_cols) with 1s at data qubit positions
    """
    n_rows, n_cols = color_code.n_rows, color_code.n_cols
    mask = torch.zeros(n_rows, n_cols, dtype=torch.float32)

    data_indices = get_data_to_grid_flat_index(color_code)
    flat_mask = torch.zeros(n_rows * n_cols, dtype=torch.float32)
    flat_mask[data_indices] = 1.0

    return flat_mask.view(n_rows, n_cols)


# ============================================================================
# Compatibility functions mirroring surface code API
# ============================================================================


def compute_stab_to_data_index_map(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get mapping from stabilizer index to associated data qubit index.
    
    This is the color code equivalent of surface code's
    compute_stabX_to_data_index_map / compute_stabZ_to_data_index_map.
    
    Note: For color code, X and Z stabilizers share the same plaquettes,
    so there is only one mapping.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (num_plaquettes,) mapping stab_idx -> data_qubit_idx
    """
    return torch.tensor(color_code.stab_to_data_idx, dtype=torch.int32)


def compute_data_to_stab_index_map(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get reverse mapping from data qubit index to stabilizer index.
    
    For data qubits not associated with any stabilizer, returns -1.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (num_data,) mapping data_idx -> stab_idx (or -1)
    """
    data_to_stab = torch.full((color_code.num_data,), -1, dtype=torch.int32)

    for stab_idx, data_idx in enumerate(color_code.stab_to_data_idx):
        data_to_stab[data_idx] = stab_idx

    return data_to_stab


# ============================================================================
# Parity check matrix construction (for completeness)
# ============================================================================


def get_parity_matrix_data_only(color_code: ColorCodeType) -> torch.Tensor:
    """
    Get the parity check matrix with only data qubit columns.
    
    For color code, hx and hz are identical (CSS code with same support),
    so we only need one matrix.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Tensor of shape (num_plaquettes, num_data)
    """
    return torch.tensor(color_code.hx[:, :color_code.num_data], dtype=torch.float32)


__all__ = [
    # Core mapping functions
    "get_stab_to_grid_flat_index",
    "get_data_to_grid_flat_index",
    "normalized_weight_mapping_stab",

    # Reshape functions
    "reshape_stabilizers_to_grid",
    "reshape_stabilizers_to_grid_2d",
    "reshape_data_to_grid",
    "reshape_data_to_grid_2d",
    "reshape_stabilizers_to_grid_vectorized",
    "reshape_data_to_grid_vectorized",

    # Inverse mapping functions
    "map_grid_to_stabilizer_tensor",
    "map_grid_to_data_tensor",

    # Presence masks
    "get_stabilizer_presence_mask",
    "get_data_presence_mask",

    # Compatibility functions
    "compute_stab_to_data_index_map",
    "compute_data_to_stab_index_map",
    "get_parity_matrix_data_only",
]
