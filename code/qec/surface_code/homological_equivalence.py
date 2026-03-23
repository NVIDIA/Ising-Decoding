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
Homological Equivalence Functions for Surface Code Pre-decoder

This module implements the homological equivalence transformations described in the
pre-decoder paper (Section II.D) to reduce training data complexity by fixing canonical
representatives of homologically equivalent errors.

Key Functions:
- apply_homological_equivalence: Main interface function
- weight_reduction_X/Z: Reduce weight-3/4 errors using stabilizers  
- fix_equivalence_X/Z: Fix canonical forms for weight-2 error patterns
- simplify_X/Z: Iterative application until steady state

Author: Muyuan Li
"""

import torch
import numpy as np
from typing import Tuple, List, Dict


def linear_index_to_coordinates(index: int, distance: int) -> Tuple[int, int]:
    """
    Convert linear data qubit index to 2D coordinates (α, β).
    
    Args:
        index: Linear index of data qubit (0 to distance²-1)
        distance: Surface code distance D
        
    Returns:
        Tuple (α, β) where α=row, β=column
    """
    alpha = index // distance  # row
    beta = index % distance  # column
    return alpha, beta


def coordinates_to_linear_index(alpha: int, beta: int, distance: int) -> int:
    """
    Convert 2D coordinates (α, β) to linear data qubit index.
    
    Args:
        alpha: Row coordinate
        beta: Column coordinate 
        distance: Surface code distance D
        
    Returns:
        Linear index of data qubit
    """
    return alpha * distance + beta


def get_stabilizer_support_from_parity_matrix(stab_idx: int,
                                              parity_matrix: torch.Tensor) -> List[int]:
    """
    Get the data qubit indices that form the support of a given stabilizer from parity check matrix.
    
    Args:
        stab_idx: Index of the stabilizer
        parity_matrix: Parity check matrix (num_stabilizers, num_data_qubits)
        
    Returns:
        List of data qubit indices in the stabilizer's support
    """
    if stab_idx >= parity_matrix.shape[0]:
        return []

    # Find all data qubits where this stabilizer has support (matrix entry = 1)
    support_mask = parity_matrix[stab_idx, :] == 1
    support_indices = torch.nonzero(support_mask, as_tuple=True)[0].tolist()

    return support_indices


def apply_fix_equivalence_X_local(
    error_config: torch.Tensor, support: List[int], distance: int
) -> torch.Tensor:
    """
    Apply fixEquivalenceX transformations to a single stabilizer's support.
    
    Transforms weight-2 X error patterns within the support to canonical forms:
    - Vertical: (α,β) + (α+1,β) → (α,β+1) + (α+1,β+1) [Left -> Right]
    - Horizontal: (α+1,β) + (α+1,β+1) → (α,β) + (α,β+1) [Bottom -> Top]
    - Diagonal: (α,β) + (α+1,β+1) → (α,β+1) + (α+1,β)
    
    Args:
        error_config: Binary tensor (D²,) representing X errors on data qubits
        support: List of 4 data qubit indices forming the stabilizer's support
        distance: Surface code distance D
        
    Returns:
        Transformed error configuration tensor
    """
    error_config = error_config.clone()

    if len(support) != 4:
        return error_config

    # Find which qubits have errors
    error_positions = [idx for idx in support if error_config[idx] == 1]

    if len(error_positions) != 2:
        return error_config

    # Get coordinates of error positions
    error_coords = [linear_index_to_coordinates(idx, distance) for idx in error_positions]
    (alpha1, beta1), (alpha2, beta2) = error_coords[0], error_coords[1]

    # Sort coordinates to have consistent ordering
    if alpha1 > alpha2 or (alpha1 == alpha2 and beta1 > beta2):
        (alpha1, beta1), (alpha2, beta2) = (alpha2, beta2), (alpha1, beta1)
        error_positions[0], error_positions[1] = error_positions[1], error_positions[0]

    # Determine transformation
    new_pos1, new_pos2 = -1, -1

    # Vertical: (α,β) + (α+1,β) -> (α,β+1) + (α+1,β+1)
    if alpha2 == alpha1 + 1 and beta2 == beta1:
        new_pos1 = coordinates_to_linear_index(alpha1, beta1 + 1, distance)
        new_pos2 = coordinates_to_linear_index(alpha2, beta2 + 1, distance)
    # Horizontal: (α+1,β) + (α+1,β+1) -> (α,β) + (α,β+1) [Bottom -> Top]
    elif alpha2 == alpha1 and beta2 == beta1 + 1:
        new_pos1 = coordinates_to_linear_index(alpha1 - 1, beta1, distance)
        new_pos2 = coordinates_to_linear_index(alpha2 - 1, beta2, distance)
    # Diagonal X: (α,β) + (α+1,β+1) -> (α,β+1) + (α+1,β)
    elif alpha2 == alpha1 + 1 and beta2 == beta1 + 1:
        new_pos1 = coordinates_to_linear_index(alpha1, beta1 + 1, distance)
        new_pos2 = coordinates_to_linear_index(alpha1 + 1, beta1, distance)

    # Apply transformation if valid
    if new_pos1 >= 0 and new_pos1 in support and new_pos2 in support:
        error_config[error_positions[0]] = 0
        error_config[error_positions[1]] = 0
        error_config[new_pos1] = 1
        error_config[new_pos2] = 1

    return error_config


def apply_fix_equivalence_Z_local(
    error_config: torch.Tensor, support: List[int], distance: int
) -> torch.Tensor:
    """
    Apply fixEquivalenceZ transformations to a single stabilizer's support.
    
    Transforms weight-2 Z error patterns within the support to canonical forms:
    - Vertical: (α,β) + (α+1,β) → (α,β+1) + (α+1,β+1) [Left -> Right]
    - Horizontal: (α+1,β) + (α+1,β+1) → (α,β) + (α,β+1) [Bottom -> Top]
    - Diagonal: (α,β+1) + (α+1,β) → (α,β) + (α+1,β+1)
    
    Args:
        error_config: Binary tensor (D²,) representing Z errors on data qubits
        support: List of 4 data qubit indices forming the stabilizer's support
        distance: Surface code distance D
        
    Returns:
        Transformed error configuration tensor
    """
    error_config = error_config.clone()

    if len(support) != 4:
        return error_config

    # Find which qubits have errors
    error_positions = [idx for idx in support if error_config[idx] == 1]

    if len(error_positions) != 2:
        return error_config

    # Get coordinates of error positions
    error_coords = [linear_index_to_coordinates(idx, distance) for idx in error_positions]
    (alpha1, beta1), (alpha2, beta2) = error_coords[0], error_coords[1]

    # Sort coordinates to have consistent ordering
    if alpha1 > alpha2 or (alpha1 == alpha2 and beta1 > beta2):
        (alpha1, beta1), (alpha2, beta2) = (alpha2, beta2), (alpha1, beta1)
        error_positions[0], error_positions[1] = error_positions[1], error_positions[0]

    # Determine transformation
    new_pos1, new_pos2 = -1, -1

    # Vertical: (α,β) + (α+1,β) -> (α,β+1) + (α+1,β+1)
    if alpha2 == alpha1 + 1 and beta2 == beta1:
        new_pos1 = coordinates_to_linear_index(alpha1, beta1 + 1, distance)
        new_pos2 = coordinates_to_linear_index(alpha2, beta2 + 1, distance)
    # Horizontal: (α+1,β) + (α+1,β+1) -> (α,β) + (α,β+1) [Bottom -> Top]
    elif alpha2 == alpha1 and beta2 == beta1 + 1:
        new_pos1 = coordinates_to_linear_index(alpha1 - 1, beta1, distance)
        new_pos2 = coordinates_to_linear_index(alpha2 - 1, beta2, distance)
    # Diagonal Z: (α,β+1) + (α+1,β) -> (α,β) + (α+1,β+1)
    elif alpha2 == alpha1 + 1 and beta2 == beta1 - 1:
        new_pos1 = coordinates_to_linear_index(alpha1, beta2, distance)
        new_pos2 = coordinates_to_linear_index(alpha2, beta1, distance)

    # Apply transformation if valid
    if new_pos1 >= 0 and new_pos1 in support and new_pos2 in support:
        error_config[error_positions[0]] = 0
        error_config[error_positions[1]] = 0
        error_config[new_pos1] = 1
        error_config[new_pos2] = 1

    return error_config


def weight_reduction_X(
    error_config: torch.Tensor, distance: int, parity_matrix_X: torch.Tensor
) -> torch.Tensor:
    """
    Apply weight reduction transformations for X errors.
    
    Reduces weight-3 X errors to weight-1 and removes weight-4 X errors entirely
    by multiplying with appropriate X stabilizers.
    
    Args:
        error_config: Binary tensor (D²,) representing X errors on data qubits
        distance: Surface code distance D
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        
    Returns:
        Reduced error configuration tensor
    """
    error_config = error_config.clone()

    # Iterate through all X stabilizers
    for stab_idx in range(parity_matrix_X.shape[0]):
        support = get_stabilizer_support_from_parity_matrix(stab_idx, parity_matrix_X)

        if len(support) == 0:
            continue

        # Count errors in this stabilizer's support
        error_count = sum(
            error_config[qubit_idx].item()
            for qubit_idx in support
            if qubit_idx < len(error_config)
        )

        # Apply weight reduction rules
        if error_count == 4:
            # Weight-4 error: equivalent to stabilizer, remove entirely
            for qubit_idx in support:
                if qubit_idx < len(error_config):
                    error_config[qubit_idx] = 0

        elif error_count == 3:
            # Weight-3 error: reduce to weight-1 by multiplying with stabilizer
            for qubit_idx in support:
                if qubit_idx < len(error_config):
                    error_config[qubit_idx] = error_config[qubit_idx] ^ 1

        elif error_count == 2 and len(support) == 2:
            # Weight-2 error on boundary stabilizer: remove
            for qubit_idx in support:
                if qubit_idx < len(error_config):
                    error_config[qubit_idx] = 0

    return error_config


def weight_reduction_Z(
    error_config: torch.Tensor, distance: int, parity_matrix_Z: torch.Tensor
) -> torch.Tensor:
    """
    Apply weight reduction transformations for Z errors.
    
    Same logic as weight_reduction_X but for Z stabilizers.
    
    Args:
        error_config: Binary tensor (D²,) representing Z errors on data qubits
        distance: Surface code distance D
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        
    Returns:
        Reduced error configuration tensor
    """
    error_config = error_config.clone()

    # Iterate through all Z stabilizers
    for stab_idx in range(parity_matrix_Z.shape[0]):
        support = get_stabilizer_support_from_parity_matrix(stab_idx, parity_matrix_Z)

        if len(support) == 0:
            continue

        # Count errors in this stabilizer's support
        error_count = sum(
            error_config[qubit_idx].item()
            for qubit_idx in support
            if qubit_idx < len(error_config)
        )

        # Apply weight reduction rules
        if error_count == 4:
            # Weight-4 error: equivalent to stabilizer, remove entirely
            for qubit_idx in support:
                if qubit_idx < len(error_config):
                    error_config[qubit_idx] = 0

        elif error_count == 3:
            # Weight-3 error: reduce to weight-1 by multiplying with stabilizer
            for qubit_idx in support:
                if qubit_idx < len(error_config):
                    error_config[qubit_idx] = error_config[qubit_idx] ^ 1

        elif error_count == 2 and len(support) == 2:
            # Weight-2 error on boundary stabilizer: remove
            for qubit_idx in support:
                if qubit_idx < len(error_config):
                    error_config[qubit_idx] = 0

    return error_config


def fix_equivalence_X(
    error_config: torch.Tensor, distance: int, parity_matrix_X: torch.Tensor
) -> torch.Tensor:
    """
    Apply equivalence fixing transformations for X errors.
    
    Transforms weight-2 X error patterns within weight-4 stabilizers to canonical forms:
    - Vertical chain: (α,β) + (α+1,β) → (α,β+1) + (α+1,β+1) [Left -> Right]
    - Horizontal chain: (α+1,β) + (α+1,β+1) → (α,β) + (α,β+1) [Bottom -> Top]
    - Diagonal chain: (α,β) + (α+1,β+1) → (α,β+1) + (α+1,β)
    
    Args:
        error_config: Binary tensor (D²,) representing X errors on data qubits
        distance: Surface code distance D
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        
    Returns:
        Canonicalized error configuration tensor
    """
    error_config = error_config.clone()

    # Iterate through all X stabilizers
    skipped_w2_stabs = []
    for stab_idx in range(parity_matrix_X.shape[0]):
        support = get_stabilizer_support_from_parity_matrix(stab_idx, parity_matrix_X)

        if len(support) != 4:
            if len(support) == 2:
                skipped_w2_stabs.append((stab_idx, support))
            continue

        # Count errors in this stabilizer's support
        error_count = sum(error_config[qubit_idx].item() for qubit_idx in support)

        if error_count != 2:
            continue

        # Apply local transformation
        error_config = apply_fix_equivalence_X_local(error_config, support, distance)

    return error_config


def fix_equivalence_Z(
    error_config: torch.Tensor, distance: int, parity_matrix_Z: torch.Tensor
) -> torch.Tensor:
    """
    Apply equivalence fixing transformations for Z errors.
    
    Similar to fix_equivalence_X but with different diagonal transformation:
    - Vertical chain: (α,β) + (α+1,β) → (α,β+1) + (α+1,β+1) [Left -> Right]
    - Horizontal chain: (α+1,β) + (α+1,β+1) → (α,β) + (α,β+1) [Bottom -> Top]
    - Diagonal chain: (α,β+1) + (α+1,β) → (α,β) + (α+1,β+1) [DIFFERENT from X!]
    
    Args:
        error_config: Binary tensor (D²,) representing Z errors on data qubits
        distance: Surface code distance D
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        
    Returns:
        Canonicalized error configuration tensor
    """
    error_config = error_config.clone()

    # Iterate through all Z stabilizers
    for stab_idx in range(parity_matrix_Z.shape[0]):
        support = get_stabilizer_support_from_parity_matrix(stab_idx, parity_matrix_Z)

        if len(support) != 4:
            continue

        # Count errors in this stabilizer's support
        error_count = sum(error_config[qubit_idx].item() for qubit_idx in support)

        if error_count != 2:
            continue

        # Apply local transformation using helper
        error_config = apply_fix_equivalence_Z_local(error_config, support, distance)

    return error_config


def simplify_X(
    error_config: torch.Tensor,
    distance: int,
    parity_matrix_X: torch.Tensor,
    max_iterations: int = 100
) -> torch.Tensor:
    """
    Iteratively apply weight reduction and equivalence fixing for X errors until steady state.
    
    Args:
        error_config: Binary tensor (D²,) representing X errors on data qubits
        distance: Surface code distance D
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        Steady-state canonical error configuration tensor
    """
    current_config = error_config.clone()

    for iteration in range(max_iterations):
        # Store previous configuration to check for convergence
        previous_config = current_config.clone()

        # Apply weight reduction followed by equivalence fixing
        current_config = weight_reduction_X(current_config, distance, parity_matrix_X)
        current_config = fix_equivalence_X(current_config, distance, parity_matrix_X)

        # Check for convergence (steady state)
        if torch.equal(current_config, previous_config):
            break

    return current_config


def simplify_Z(
    error_config: torch.Tensor,
    distance: int,
    parity_matrix_Z: torch.Tensor,
    max_iterations: int = 100
) -> torch.Tensor:
    """
    Iteratively apply weight reduction and equivalence fixing for Z errors until steady state.
    
    Args:
        error_config: Binary tensor (D²,) representing Z errors on data qubits
        distance: Surface code distance D
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        Steady-state canonical error configuration tensor
    """
    current_config = error_config.clone()

    for iteration in range(max_iterations):
        # Store previous configuration to check for convergence
        previous_config = current_config.clone()

        # Apply weight reduction followed by equivalence fixing
        current_config = weight_reduction_Z(current_config, distance, parity_matrix_Z)
        current_config = fix_equivalence_Z(current_config, distance, parity_matrix_Z)

        # Check for convergence (steady state)
        if torch.equal(current_config, previous_config):
            break

    return current_config


def simplify_X_with_count(
    error_config: torch.Tensor,
    distance: int,
    parity_matrix_X: torch.Tensor,
    max_iterations: int = 100
) -> tuple:
    """
    Iteratively apply weight reduction and equivalence fixing for X errors until steady state.
    Returns both the result and iteration count.
    
    Args:
        error_config: Binary tensor (D²,) representing X errors on data qubits
        distance: Surface code distance D
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        (canonical_config, iterations_used): Tuple of result and iteration count
    """
    current_config = error_config.clone()

    for iteration in range(max_iterations):
        # Store previous configuration to check for convergence
        previous_config = current_config.clone()

        # Apply weight reduction followed by equivalence fixing
        current_config = weight_reduction_X(current_config, distance, parity_matrix_X)
        current_config = fix_equivalence_X(current_config, distance, parity_matrix_X)

        # Check for convergence (steady state)
        if torch.equal(current_config, previous_config):
            return current_config, iteration + 1

    return current_config, max_iterations


def simplify_Z_with_count(
    error_config: torch.Tensor,
    distance: int,
    parity_matrix_Z: torch.Tensor,
    max_iterations: int = 100
) -> tuple:
    """
    Iteratively apply weight reduction and equivalence fixing for Z errors until steady state.
    Returns both the result and iteration count.
    
    Args:
        error_config: Binary tensor (D²,) representing Z errors on data qubits
        distance: Surface code distance D
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        (canonical_config, iterations_used): Tuple of result and iteration count
    """
    current_config = error_config.clone()

    for iteration in range(max_iterations):
        # Store previous configuration to check for convergence
        previous_config = current_config.clone()

        # Apply weight reduction followed by equivalence fixing
        current_config = weight_reduction_Z(current_config, distance, parity_matrix_Z)
        current_config = fix_equivalence_Z(current_config, distance, parity_matrix_Z)

        # Check for convergence (steady state)
        if torch.equal(current_config, previous_config):
            return current_config, iteration + 1

    return current_config, max_iterations


def apply_spacelike_homological_equivalence(
    x_error_diff: torch.Tensor, z_error_diff: torch.Tensor, distance: int,
    parity_matrix_X: torch.Tensor, parity_matrix_Z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply spacelike homological equivalence transformations to error difference tensors.
    
    Main interface function that:
    1. Reconstructs full error configurations from differences
    2. Applies weight reduction and equivalence fixing per round
    3. Converts back to error differences for training
    
    Args:
        x_error_diff: X error differences tensor (D², T)
        z_error_diff: Z error differences tensor (D², T)  
        distance: Surface code distance D
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        
    Returns:
        Tuple of canonicalized (x_error_diff, z_error_diff) tensors
    """
    num_qubits, n_rounds = x_error_diff.shape

    # Step 1: Reconstruct full error configurations from differences
    x_error_config = torch.zeros_like(x_error_diff)
    z_error_config = torch.zeros_like(z_error_diff)

    # Cumulative XOR to get full error configuration from differences
    for t in range(n_rounds):
        if t == 0:
            x_error_config[..., t] = x_error_diff[..., t]
            z_error_config[..., t] = z_error_diff[..., t]
        else:
            x_error_config[..., t] = x_error_config[..., t - 1] ^ x_error_diff[..., t]
            z_error_config[..., t] = z_error_config[..., t - 1] ^ z_error_diff[..., t]

    # Step 2: Apply homological equivalence per round and batch
    for t in range(n_rounds):
        # Extract current round error configuration
        x_errors_round = x_error_config[..., t]  # (D²,)
        z_errors_round = z_error_config[..., t]  # (D²,)

        # Apply simplifyX and simplifyZ until steady state
        x_simplified = simplify_X(x_errors_round, distance, parity_matrix_X)
        z_simplified = simplify_Z(z_errors_round, distance, parity_matrix_Z)

        # Update the error configuration
        x_error_config[..., t] = x_simplified
        z_error_config[..., t] = z_simplified

    # Step 3: Convert back to error differences
    x_error_diff_new = torch.zeros_like(x_error_diff)
    z_error_diff_new = torch.zeros_like(z_error_diff)

    for t in range(n_rounds):
        if t == 0:
            x_error_diff_new[..., t] = x_error_config[..., t]
            z_error_diff_new[..., t] = z_error_config[..., t]
        else:
            x_error_diff_new[..., t] = x_error_config[..., t] ^ x_error_config[..., t - 1]
            z_error_diff_new[..., t] = z_error_config[..., t] ^ z_error_config[..., t - 1]

    return x_error_diff_new, z_error_diff_new


# Backward compatibility alias
apply_homological_equivalence = apply_spacelike_homological_equivalence

### New: timelike homological equivalence


def simplifytimeX(
    x_error_diff_round_round_plus_1: torch.Tensor, s1s2z_round_round_plus_1: torch.Tensor,
    parity_matrix_Z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply timelike homological equivalence for X errors (detected by Z measurements and mapped with parity_matrix_Z).
    
    Args:
        x_error_diff_round_round_plus_1: X error diff configuration tensor (B, D², 2)
        s1s2z_round_round_plus_1: s1s2z measurement tensor (B, num_Z_stabs, 2)
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        
    Returns:
        Tuple of (simplified X error tensor (B, D², 2), simplified s1s2z tensor (B, num_Z_stabs, 2), num_accepted)
    """

    # Clone once for all data qubits
    new_s1s2z = s1s2z_round_round_plus_1.clone()
    new_x_error_diff = x_error_diff_round_round_plus_1.clone()

    # Flip all data qubits and stabilizers at once (only flip first round for stabilizers): (B, D², 2) and (B, num_Z_stabs, 2)
    new_x_error_diff = (new_x_error_diff + 1) % 2
    new_s1s2z[..., 0] = (new_s1s2z[..., 0] + 1) % 2

    # Compute old timelike loop densities and new ones
    # x_error_diff: (B, D², 2), s1s2z: (B, num_Z_stabs, 2), parity: (num_Z_stabs, D²)
    # einsum: batch, stabs, time × stabs, dataqubits -> batch, dataqubits, time
    old_density_per_time = x_error_diff_round_round_plus_1 + torch.einsum(
        'bst,sd->bdt', s1s2z_round_round_plus_1.float(), parity_matrix_Z.float()
    )
    old_density = old_density_per_time.sum(dim=2)  # (B, D²)

    new_density_per_time = new_x_error_diff + torch.einsum(
        'bst,sd->bdt', new_s1s2z.float(), parity_matrix_Z.float()
    )
    new_density = new_density_per_time.sum(dim=2)  # (B, D²)

    # Accept mask: (B, D²) - True where new density is lower
    accept_mask = new_density < old_density  # (B, D²)

    # Tie-breaker: when densities are equal, prefer more 1's in first round (t)
    # This pushes errors later in time (gives better density reduction)
    density_equal = (new_density == old_density)  # (B, D²)
    old_round0_density = old_density_per_time[:, :, 0]  # (B, D²)
    new_round0_density = new_density_per_time[:, :, 0]  # (B, D²)
    tie_breaker = density_equal & (new_round0_density > old_round0_density)  # (B, D²)

    # Combine: accept if density improves OR (equal density AND more in round k+1)
    accept_mask = accept_mask | tie_breaker  # (B, D²)

    # Count number of acceptances
    num_accepted = int(accept_mask.sum().item())

    # Apply changes selectively per (batch, data_qubit)
    # For x_error_diff: straightforward - accept_mask is (B, D²), expand to (B, D², 1) for time dim
    x_error_diff_round_round_plus_1 = torch.where(
        accept_mask.unsqueeze(2),  # (B, D², 1)
        new_x_error_diff,
        x_error_diff_round_round_plus_1
    )

    # For s1s2z: need to determine which stabs to flip based on which data qubits were accepted
    # flip_count[b, s] = sum over q of (accept_mask[b, q] * parity_matrix_Z[s, q])
    # (B, D²) @ (D², num_Z_stabs) -> (B, num_Z_stabs)
    flip_count = torch.matmul(accept_mask.float(), parity_matrix_Z.T.float())  # (B, num_Z_stabs)
    should_flip = (flip_count % 2).bool()  # (B, num_Z_stabs)

    # Apply flips to s1s2z
    s1s2z_round_round_plus_1 = torch.where(
        should_flip.unsqueeze(2),  # (B, num_Z_stabs, 1)
        (s1s2z_round_round_plus_1 + 1) % 2,
        s1s2z_round_round_plus_1
    )

    return x_error_diff_round_round_plus_1, s1s2z_round_round_plus_1, num_accepted


def simplifytimeZ(
    z_error_diff_round_round_plus_1: torch.Tensor, s1s2x_round_round_plus_1: torch.Tensor,
    parity_matrix_X: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply timelike homological equivalence for Z errors (detected by X measurements and mapped with parity_matrix_X).
    
    Args:
        z_error_diff_round_round_plus_1: Z error diff configuration tensor (B, D², 2)
        s1s2x_round_round_plus_1: s1s2x measurement tensor (B, num_X_stabs, 2)
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)

    Returns:
        Tuple of (simplified Z error tensor (B, D², 2), simplified s1s2x tensor (B, num_X_stabs, 2), num_accepted)
    """

    new_s1s2x = s1s2x_round_round_plus_1.clone()
    new_z_error_diff = z_error_diff_round_round_plus_1.clone()

    # Flip all data qubits and stabilizers at once (only flip first round for stabilizers): (B, D², 2) and (B, num_X_stabs, 2)
    new_z_error_diff = (new_z_error_diff + 1) % 2
    new_s1s2x[..., 0] = (new_s1s2x[..., 0] + 1) % 2

    # Compute old timelike loop densities and new ones
    # z_error_diff: (B, D², 2), s1s2x: (B, num_X_stabs, 2), parity: (num_X_stabs, D²)
    # einsum: batch, stabs, time × stabs, dataqubits -> batch, dataqubits, time
    old_density_per_time = z_error_diff_round_round_plus_1 + torch.einsum(
        'bst,sd->bdt', s1s2x_round_round_plus_1.float(), parity_matrix_X.float()
    )
    old_density = old_density_per_time.sum(dim=2)  # (B, D²)

    new_density_per_time = new_z_error_diff + torch.einsum(
        'bst,sd->bdt', new_s1s2x.float(), parity_matrix_X.float()
    )
    new_density = new_density_per_time.sum(dim=2)  # (B, D²)

    # Accept mask: (B, D²) - True where new density is lower
    accept_mask = new_density < old_density  # (B, D²)

    # Tie-breaker: when densities are equal, prefer more 1's in first round (t)
    # This pushes errors earlier in time (gives better density reduction)
    density_equal = (new_density == old_density)  # (B, D²)
    old_round0_density = old_density_per_time[:, :, 0]  # (B, D²)
    new_round0_density = new_density_per_time[:, :, 0]  # (B, D²)
    tie_breaker = density_equal & (new_round0_density > old_round0_density)  # (B, D²)

    # Combine: accept if density improves OR (equal density AND more in round k+1)
    accept_mask = accept_mask | tie_breaker  # (B, D²)

    # Count number of acceptances
    num_accepted = int(accept_mask.sum().item())

    # Apply changes selectively per (batch, data_qubit)
    # For z_error_diff: straightforward - accept_mask is (B, D²), expand to (B, D², 1) for time dim
    z_error_diff_round_round_plus_1 = torch.where(
        accept_mask.unsqueeze(2),  # (B, D², 1)
        new_z_error_diff,
        z_error_diff_round_round_plus_1
    )

    # For s1s2x: need to determine which stabs to flip based on which data qubits were accepted
    # flip_count[b, s] = sum over q of (accept_mask[b, q] * parity_matrix_X[s, q])
    # (B, D²) @ (D², num_X_stabs) -> (B, num_X_stabs)
    flip_count = torch.matmul(accept_mask.float(), parity_matrix_X.T.float())  # (B, num_X_stabs)
    should_flip = (flip_count % 2).bool()  # (B, num_X_stabs)

    # Apply flips to s1s2x
    s1s2x_round_round_plus_1 = torch.where(
        should_flip.unsqueeze(2),  # (B, num_X_stabs, 1)
        (s1s2x_round_round_plus_1 + 1) % 2,
        s1s2x_round_round_plus_1
    )

    return z_error_diff_round_round_plus_1, s1s2x_round_round_plus_1, num_accepted


def get_anticommuting_stabilizers(qubit_indices: List[int],
                                  parity_matrix: torch.Tensor) -> List[int]:
    """
    Find stabilizers that anticommute with errors on the given qubits.
    A stabilizer anticommutes if it shares an odd number of qubits with the error set.
    
    Args:
        qubit_indices: List of qubit indices with errors
        parity_matrix: Parity check matrix
        
    Returns:
        List of stabilizer indices
    """
    if not qubit_indices:
        return []

    # Sum columns corresponding to qubits (modulo 2)
    relevant_cols = parity_matrix[:, qubit_indices]
    syndrome = relevant_cols.sum(dim=1) % 2

    # Indices where syndrome is 1
    return torch.nonzero(syndrome, as_tuple=True)[0].tolist()


def simplifytimeZ_weight2(
    z_error_diff: torch.Tensor, s1s2x: torch.Tensor, parity_matrix_X: torch.Tensor,
    parity_matrix_Z: torch.Tensor, distance: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply timelike homological equivalence for weight-2 Z errors.
    Tries all 3 weight-2 patterns (horizontal, vertical, diagonal) for each stabilizer.
    Sequential implementation as fixEquivalenceZ must be applied iteratively.
    
    Args:
        z_error_diff: Z error diff configuration tensor (B, D², 2)
        s1s2x: s1s2x measurement tensor (B, num_X_stabs, 2)
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        distance: Surface code distance D
        
    Returns:
        (simplified Z errors, simplified s1s2x, num_accepted)
    """
    B = z_error_diff.shape[0]
    num_accepted = 0

    # Iterate through all weight-4 Z-stabilizers
    for stab_idx in range(parity_matrix_Z.shape[0]):
        support = get_stabilizer_support_from_parity_matrix(stab_idx, parity_matrix_Z)

        if len(support) != 4:
            continue

        # Get qubit coordinates for this stabilizer
        coords = [linear_index_to_coordinates(idx, distance) for idx in support]
        coords_with_idx = [(alpha, beta, idx) for (alpha, beta), idx in zip(coords, support)]
        coords_with_idx.sort()  # Sort by (alpha, beta)

        # Identify all 3 possible pairs
        # Horizontal: Top row (canonical for fixEquivalenceZ horizontal)
        top_row = coords_with_idx[:2]
        top_row.sort(key=lambda x: x[1])
        horizontal_pair = (top_row[0][2], top_row[1][2])  # (top-left, top-right)

        # Vertical: Right column (canonical for fixEquivalenceZ vertical)
        coords_sorted_by_beta = sorted(coords_with_idx, key=lambda x: x[1])
        right_col = coords_sorted_by_beta[2:]
        right_col.sort(key=lambda x: x[0])
        vertical_pair = (right_col[0][2], right_col[1][2])  # (top-right, bottom-right)

        # Diagonal: Top-left to bottom-right (canonical for fixEquivalenceZ diagonal)
        diagonal_pair = (coords_with_idx[0][2], coords_with_idx[3][2])  # (top-left, bottom-right)

        # Try patterns in order: Horizontal (single fault), Vertical, Diagonal
        pairs_to_try = [
            ('horizontal', horizontal_pair),
            # ('vertical', vertical_pair),
            # ('diagonal', diagonal_pair)
        ]

        had_acceptance = False  # Track if any pattern was accepted for this stabilizer

        for pattern_name, (q1_idx, q2_idx) in pairs_to_try:
            # STEP 1: Find anticommuting X-stabilizers for this pair
            anticommuting_stabs = get_anticommuting_stabilizers([q1_idx, q2_idx], parity_matrix_X)

            # STEP 2: Try transformation per batch
            old_density_per_time = z_error_diff + torch.einsum(
                'bst,sd->bdt', s1s2x.float(), parity_matrix_X.float()
            )
            old_density_total = old_density_per_time.sum(dim=2).sum(dim=1)  # (B,)
            old_density_k_plus_1 = old_density_per_time[:, :, 1].sum(dim=1)  # (B,)

            new_z_error = z_error_diff.clone()
            new_s1s2x = s1s2x.clone()

            new_z_error[:, q1_idx, :] = (new_z_error[:, q1_idx, :] + 1) % 2
            new_z_error[:, q2_idx, :] = (new_z_error[:, q2_idx, :] + 1) % 2

            # CRITICAL: Only flip stabilizer measurements in round k (index 0), NOT round k+1
            # Physical reason: The weight-2 error pattern anticommutes with these stabilizers
            # only in round k (when the error is first introduced). In round k+1, the
            # propagated frame already accounts for this, so flipping measurements in
            # both rounds would be incorrect. This matches Algorithm 4 in the paper.
            if anticommuting_stabs:
                new_s1s2x[:, anticommuting_stabs,
                          0] = (new_s1s2x[:, anticommuting_stabs, 0] + 1) % 2

            new_density_per_time = new_z_error + torch.einsum(
                'bst,sd->bdt', new_s1s2x.float(), parity_matrix_X.float()
            )
            new_density_total = new_density_per_time.sum(dim=2).sum(dim=1)  # (B,)
            new_density_k_plus_1 = new_density_per_time[:, :, 1].sum(dim=1)  # (B,)

            accept_mask = (new_density_total < old_density_total) | \
                          ((new_density_total == old_density_total) & (new_density_k_plus_1 > old_density_k_plus_1))

            # If accepted for any batch, update state and count
            if accept_mask.any():
                num_accepted += int(accept_mask.sum().item())
                z_error_diff = torch.where(
                    accept_mask.unsqueeze(1).unsqueeze(2), new_z_error, z_error_diff
                )
                s1s2x = torch.where(accept_mask.unsqueeze(1).unsqueeze(2), new_s1s2x, s1s2x)
                had_acceptance = True

        # CLEANUP: If this stabilizer had acceptances, apply full spacelike HE to clean up
        # This handles cross-stabilizer effects (weight-3/4 in neighbors) and re-canonicalizes
        if had_acceptance:
            for t in range(2):  # Both times k and k+1
                for b in range(B):
                    # Apply simplify_Z (weight reduction + fixEquivalence until convergence)
                    z_error_diff[b, :, t] = simplify_Z(
                        z_error_diff[b, :, t].to(torch.long), distance, parity_matrix_Z
                    ).float()

    return z_error_diff, s1s2x, num_accepted


def simplifytimeX_weight2(
    x_error_diff: torch.Tensor, s1s2z: torch.Tensor, parity_matrix_Z: torch.Tensor,
    parity_matrix_X: torch.Tensor, distance: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply timelike homological equivalence for weight-2 X errors.
    Tries all 3 weight-2 patterns (vertical, horizontal, diagonal) for each stabilizer.
    
    Args:
        x_error_diff: X error diff configuration tensor (B, D², 2)
        s1s2z: s1s2z measurement tensor (B, num_Z_stabs, 2)
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        distance: Surface code distance D
        
    Returns:
        (simplified X errors, simplified s1s2z, num_accepted)
    """
    B = x_error_diff.shape[0]
    num_accepted = 0

    # Iterate through all weight-4 X-stabilizers
    for stab_idx in range(parity_matrix_X.shape[0]):
        support = get_stabilizer_support_from_parity_matrix(stab_idx, parity_matrix_X)

        if len(support) != 4:
            continue

        # Get qubit coordinates for this stabilizer
        coords = [linear_index_to_coordinates(idx, distance) for idx in support]
        coords_with_idx = [(alpha, beta, idx) for (alpha, beta), idx in zip(coords, support)]
        coords_with_idx.sort()  # Sort by (alpha, beta)

        # Identify all 3 possible pairs
        # Vertical: Right column (canonical for fixEquivalenceX vertical)
        coords_sorted_by_beta = sorted(coords_with_idx, key=lambda x: x[1])
        right_col = coords_sorted_by_beta[2:]
        right_col.sort(key=lambda x: x[0])
        vertical_pair = (right_col[0][2], right_col[1][2])  # (top-right, bottom-right)

        # Horizontal: Top row (canonical for fixEquivalenceX horizontal)
        top_row = coords_with_idx[:2]
        top_row.sort(key=lambda x: x[1])
        horizontal_pair = (top_row[0][2], top_row[1][2])  # (top-left, top-right)

        # Diagonal: Top-right to bottom-left (canonical for fixEquivalenceX diagonal)
        # fixEquivalenceX diagonal: (α,β) + (α+1,β+1) -> (α,β+1) + (α+1,β)
        # Canonical is (α,β+1) + (α+1,β) = top-right + bottom-left
        top_right = coords_sorted_by_beta[2:][0][2]  # Right col, top
        bottom_left = coords_sorted_by_beta[:2][1][2]  # Left col, bottom
        diagonal_pair = (top_right, bottom_left)

        # Try patterns in order: Vertical (single fault), Horizontal, Diagonal
        pairs_to_try = [
            ('vertical', vertical_pair),
            # ('horizontal', horizontal_pair),
            # ('diagonal', diagonal_pair)
        ]

        had_acceptance = False  # Track if any pattern was accepted for this stabilizer

        for pattern_name, (q1_idx, q2_idx) in pairs_to_try:
            # STEP 1: Find anticommuting Z-stabilizers for this pair
            anticommuting_stabs = get_anticommuting_stabilizers([q1_idx, q2_idx], parity_matrix_Z)

            # STEP 2: Try transformation per batch
            old_density_per_time = x_error_diff + torch.einsum(
                'bst,sd->bdt', s1s2z.float(), parity_matrix_Z.float()
            )
            old_density_total = old_density_per_time.sum(dim=2).sum(dim=1)
            old_density_k_plus_1 = old_density_per_time[:, :, 1].sum(dim=1)

            new_x_error = x_error_diff.clone()
            new_s1s2z = s1s2z.clone()

            new_x_error[:, q1_idx, :] = (new_x_error[:, q1_idx, :] + 1) % 2
            new_x_error[:, q2_idx, :] = (new_x_error[:, q2_idx, :] + 1) % 2

            # CRITICAL: Only flip stabilizer measurements in round k (index 0), NOT round k+1
            # Physical reason: The weight-2 error pattern anticommutes with these stabilizers
            # only in round k (when the error is first introduced). In round k+1, the
            # propagated frame already accounts for this, so flipping measurements in
            # both rounds would be incorrect. This matches Algorithm 5 in the paper.
            if anticommuting_stabs:
                new_s1s2z[:, anticommuting_stabs,
                          0] = (new_s1s2z[:, anticommuting_stabs, 0] + 1) % 2

            new_density_per_time = new_x_error + torch.einsum(
                'bst,sd->bdt', new_s1s2z.float(), parity_matrix_Z.float()
            )
            new_density_total = new_density_per_time.sum(dim=2).sum(dim=1)
            new_density_k_plus_1 = new_density_per_time[:, :, 1].sum(dim=1)

            accept_mask = (new_density_total < old_density_total) | \
                          ((new_density_total == old_density_total) & (new_density_k_plus_1 > old_density_k_plus_1))

            # If accepted for any batch, update state and count
            if accept_mask.any():
                num_accepted += int(accept_mask.sum().item())
                x_error_diff = torch.where(
                    accept_mask.unsqueeze(1).unsqueeze(2), new_x_error, x_error_diff
                )
                s1s2z = torch.where(accept_mask.unsqueeze(1).unsqueeze(2), new_s1s2z, s1s2z)
                had_acceptance = True

        # CLEANUP: If this stabilizer had acceptances, apply full spacelike HE to clean up
        # This handles cross-stabilizer effects (weight-3/4 in neighbors) and re-canonicalizes
        if had_acceptance:
            for t in range(2):  # Both times k and k+1
                for b in range(B):
                    x_error_diff[b, :, t] = simplify_X(
                        x_error_diff[b, :, t].to(torch.long), distance, parity_matrix_X
                    ).float()

    return x_error_diff, s1s2z, num_accepted


def apply_timelike_homological_equivalence(
    trainY: torch.Tensor,
    parity_matrix_X: torch.Tensor,
    parity_matrix_Z: torch.Tensor,
    max_iterations: int = 32,
    basis: str = 'X',
    final_cleanup: bool = False,
    code_rotation: str = 'XV'
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Apply timelike homological equivalence transformations to an element of trainY
    
    Args:
        trainY: TrainY tensor (4, n_rounds, D, D)
        parity_matrix_X: X stabilizer parity check matrix (num_X_stabs, D²)
        parity_matrix_Z: Z stabilizer parity check matrix (num_Z_stabs, D²)
        max_iterations: Number of passes to run (default 4, convergence typically in 1-2)
        basis: 'X' or 'Z' - memory circuit basis (for excluding round 0 of opposite basis)
        code_rotation: Surface code orientation ('XV', 'XH', 'ZV', 'ZH'). Default 'XV'.
        
    Returns:
        Tuple of (Simplified trainY tensor, acceptance_counts dict)
    """
    from qec.surface_code.data_mapping import (
        map_grid_to_stabilizer_tensor,
        compute_stabX_to_data_index_map,
        compute_stabZ_to_data_index_map,
        reshape_Xstabilizers_to_grid_vectorized,
        reshape_Zstabilizers_to_grid_vectorized,
    )

    _, D2 = parity_matrix_X.shape
    _, D2 = parity_matrix_Z.shape
    D = int(D2**0.5)
    B = trainY.shape[0]
    n_rounds = trainY.shape[2]  # trainY shape: (B, 4, n_rounds, D, D)

    # Get the index maps and move to the same device as trainY (now rotation-aware)
    code_rotation = code_rotation.upper() if code_rotation else 'XV'
    x_stab_indices = torch.as_tensor(
        compute_stabX_to_data_index_map(D, code_rotation), dtype=torch.long, device=trainY.device
    )
    z_stab_indices = torch.as_tensor(
        compute_stabZ_to_data_index_map(D, code_rotation), dtype=torch.long, device=trainY.device
    )

    # Convert grid back to flattened stabilizer form for both timelike and spacelike errors
    # trainY[:, 2, :, :, :] has shape (B, n_rounds, D, D)
    s1s2_x_flat = map_grid_to_stabilizer_tensor(
        trainY[:, 2, :, :, :], x_stab_indices
    )  # (B, num_X_stabs, n_rounds)
    s1s2_z_flat = map_grid_to_stabilizer_tensor(
        trainY[:, 3, :, :, :], z_stab_indices
    )  # (B, num_Z_stabs, n_rounds)

    z_error_diff_flat = trainY[:, 0, :, :, :].reshape(B, n_rounds, D2)  # (B, n_rounds, D²)
    x_error_diff_flat = trainY[:, 1, :, :, :].reshape(B, n_rounds, D2)  # (B, n_rounds, D²)

    z_error_diff_flat = z_error_diff_flat.transpose(1, 2)  # (B, D², n_rounds)
    x_error_diff_flat = x_error_diff_flat.transpose(1, 2)  # (B, D², n_rounds)

    # Track acceptance counts
    total_accepted_x = 0
    total_accepted_z = 0
    total_accepted_weight2 = 0

    # Stop before last round (when n_rounds > 2) since data there is unreliable
    max_t = n_rounds - 2 if n_rounds > 2 else n_rounds - 1
    # Also skip round 0 pairs for opposite basis (measurements are random)
    # For basis 'X': skip round 0 for X-errors
    # For basis 'Z': skip round 0 for Z-errors
    min_t_x = 1 if basis == 'X' else 0
    min_t_z = 1 if basis == 'Z' else 0

    # ========================================================================
    # PHASE 1: Single-qubit timelike HE until convergence
    # ========================================================================
    phase1_iterations = 0
    for iteration in range(max_iterations):
        phase1_iterations = iteration + 1
        phase1_accepted_this_iter = 0

        for t in range(max(0, max_t)):
            # Extract current data errors in the round and in the next round
            x_error_diff_round_round_plus_1 = x_error_diff_flat[:, :, t:t + 2]
            z_error_diff_round_round_plus_1 = z_error_diff_flat[:, :, t:t + 2]

            # Extract current measurements in the round and in the next round
            s1s2_x_round_round_plus_1 = s1s2_x_flat[..., t:t + 2]
            s1s2_z_round_round_plus_1 = s1s2_z_flat[..., t:t + 2]

            # Apply simplifytimeX (skip if t < min_t_x)
            if t >= min_t_x:
                x_error_diff_round_round_plus_1, s1s2_z_round_round_plus_1, num_accepted_x = simplifytimeX(
                    x_error_diff_round_round_plus_1, s1s2_z_round_round_plus_1, parity_matrix_Z
                )
                phase1_accepted_this_iter += num_accepted_x
                total_accepted_x += num_accepted_x
                x_error_diff_flat[..., t:t + 2] = x_error_diff_round_round_plus_1
                s1s2_z_flat[..., t:t + 2] = s1s2_z_round_round_plus_1

            # Apply simplifytimeZ (skip if t < min_t_z)
            if t >= min_t_z:
                z_error_diff_round_round_plus_1, s1s2_x_round_round_plus_1, num_accepted_z = simplifytimeZ(
                    z_error_diff_round_round_plus_1, s1s2_x_round_round_plus_1, parity_matrix_X
                )
                phase1_accepted_this_iter += num_accepted_z
                total_accepted_z += num_accepted_z
                z_error_diff_flat[..., t:t + 2] = z_error_diff_round_round_plus_1
                s1s2_x_flat[..., t:t + 2] = s1s2_x_round_round_plus_1

        # Check convergence: stop if no acceptances this iteration
        if phase1_accepted_this_iter == 0:
            break

    # ========================================================================
    # PHASE 2: Spacelike HE until convergence
    # ========================================================================
    # phase2_iterations = 0
    # for iteration in range(max_iterations):
    #     phase2_iterations = iteration + 1
    #     prev_x_phase2 = x_error_diff_flat.clone()
    #     prev_z_phase2 = z_error_diff_flat.clone()

    #     for b in range(B):
    #         # Store before state for comparison
    #         x_before = x_error_diff_flat[b].clone()
    #         z_before = z_error_diff_flat[b].clone()

    #         # Extract this batch's data: x_error_diff_flat[b] is (D², n_rounds)
    #         x_batch = x_error_diff_flat[b].to(torch.uint8)  # (D², n_rounds)
    #         z_batch = z_error_diff_flat[b].to(torch.uint8)  # (D², n_rounds)

    #         x_batch, z_batch = apply_spacelike_homological_equivalence(
    #             x_batch, z_batch, D, parity_matrix_X, parity_matrix_Z
    #         )

    #         x_error_diff_flat[b] = x_batch
    #         z_error_diff_flat[b] = z_batch

    #     # Check convergence
    #     if torch.equal(x_error_diff_flat, prev_x_phase2) and torch.equal(z_error_diff_flat, prev_z_phase2):
    #         break

    # ========================================================================
    # PHASE 3: Two-qubit timelike HE until convergence
    # ========================================================================
    # phase3_iterations = 0
    # for iteration in range(max_iterations):
    #     phase3_iterations = iteration + 1
    #     phase3_accepted_this_iter = 0

    #     for t in range(max(0, max_t)):
    #         # Extract pairs
    #         x_error_diff_round_round_plus_1 = x_error_diff_flat[:, :, t: t+2]
    #         z_error_diff_round_round_plus_1 = z_error_diff_flat[:, :, t: t+2]
    #         s1s2_x_round_round_plus_1 = s1s2_x_flat[..., t: t+2]
    #         s1s2_z_round_round_plus_1 = s1s2_z_flat[..., t: t+2]

    #         # Apply weight-2 timelike HE (skip if t < min_t)
    #         if t >= min_t_x:
    #             x_error_diff_round_round_plus_1, s1s2_z_round_round_plus_1, num_accepted_x = simplifytimeX_weight2(
    #                 x_error_diff_round_round_plus_1, s1s2_z_round_round_plus_1,
    #                 parity_matrix_Z, parity_matrix_X, D
    #             )
    #             phase3_accepted_this_iter += num_accepted_x
    #             total_accepted_weight2 += num_accepted_x
    #             x_error_diff_flat[..., t: t+2] = x_error_diff_round_round_plus_1
    #             s1s2_z_flat[..., t: t+2] = s1s2_z_round_round_plus_1

    #         if t >= min_t_z:
    #             z_error_diff_round_round_plus_1, s1s2_x_round_round_plus_1, num_accepted_z = simplifytimeZ_weight2(
    #                 z_error_diff_round_round_plus_1, s1s2_x_round_round_plus_1,
    #                 parity_matrix_X, parity_matrix_Z, D
    #             )
    #             phase3_accepted_this_iter += num_accepted_z
    #             total_accepted_weight2 += num_accepted_z
    #             z_error_diff_flat[..., t: t+2] = z_error_diff_round_round_plus_1
    #             s1s2_x_flat[..., t: t+2] = s1s2_x_round_round_plus_1

    #     # Check convergence: stop if no acceptances this iteration
    #     if phase3_accepted_this_iter == 0:
    #         break

    # if final_cleanup:
    #     # Apply spacelike HE to each batch sample iteratively until convergence
    #     # apply_spacelike_homological_equivalence expects (D², T) uint8 tensors
    #     max_cleanup_iterations = 100  # Limit iterations for safety

    #     for cleanup_iter in range(max_cleanup_iterations):
    #         prev_x_cleanup = x_error_diff_flat.clone()
    #         prev_z_cleanup = z_error_diff_flat.clone()

    #         for b in range(B):
    #             # Extract this batch's data: x_error_diff_flat[b] is (D², n_rounds)
    #             x_batch = x_error_diff_flat[b].to(torch.uint8)  # (D², n_rounds)
    #             z_batch = z_error_diff_flat[b].to(torch.uint8)  # (D², n_rounds)

    #             x_batch, z_batch = apply_spacelike_homological_equivalence(
    #                 x_batch, z_batch, D, parity_matrix_X, parity_matrix_Z
    #             )

    #             x_error_diff_flat[b] = x_batch
    #             z_error_diff_flat[b] = z_batch

    #         # Check convergence
    #         if torch.equal(x_error_diff_flat, prev_x_cleanup) and torch.equal(z_error_diff_flat, prev_z_cleanup):
    #             break

    # Step 4: Convert back to grid format for trainY
    trainY_new = torch.zeros_like(trainY)

    s1s2x_new = reshape_Xstabilizers_to_grid_vectorized(s1s2_x_flat, D, code_rotation)
    s1s2z_new = reshape_Zstabilizers_to_grid_vectorized(s1s2_z_flat, D, code_rotation)

    x_error_diff_flat = x_error_diff_flat.transpose(1, 2)
    z_error_diff_flat = z_error_diff_flat.transpose(1, 2)
    x_error_diff_new = x_error_diff_flat.reshape(B, n_rounds, D, D)
    z_error_diff_new = z_error_diff_flat.reshape(B, n_rounds, D, D)

    # s1s2x_new and s1s2z_new have shape (B, D*D, n_rounds), need to reshape to (B, n_rounds, D, D)
    s1s2x_new = s1s2x_new.reshape(B, D, D, n_rounds).permute(0, 3, 1, 2)  # (B, n_rounds, D, D)
    s1s2z_new = s1s2z_new.reshape(B, D, D, n_rounds).permute(0, 3, 1, 2)  # (B, n_rounds, D, D)

    trainY_new[:, 0, :, :, :] = z_error_diff_new
    trainY_new[:, 1, :, :, :] = x_error_diff_new
    trainY_new[:, 2, :, :, :] = s1s2x_new
    trainY_new[:, 3, :, :, :] = s1s2z_new

    # Return trainY and acceptance counts with iteration statistics
    acceptance_counts = {
        'total_accepted_x': total_accepted_x,
        'total_accepted_z': total_accepted_z,
        'total_accepted_weight2': total_accepted_weight2,
        'total_accepted': total_accepted_x + total_accepted_z + total_accepted_weight2,
        'phase1_iterations': phase1_iterations,
        # 'phase2_iterations': phase2_iterations,
        # 'phase3_iterations': phase3_iterations
    }

    return trainY_new, acceptance_counts
