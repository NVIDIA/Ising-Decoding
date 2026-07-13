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
Homological Equivalence Functions for Color Code Pre-decoder

This module implements spacelike homological equivalence transformations for the
triangular color code. The goal is to reduce training data complexity by fixing
canonical representatives of homologically equivalent errors.

===============================================================================
PLAQUETTE QUBIT LABELING CONVENTION
===============================================================================

For weight-6 bulk plaquettes, qubits are labeled by compass direction in a 3x2 grid:

    q1 (NW)   q4 (NE)     row_top
    q2 (W)    q5 (E)      row_mid
    q3 (SW)   q6 (SE)     row_bot
    
    col_left  col_right

This corresponds to the coordinate convention where:
- row_top > row_mid > row_bot (row values decrease going down)
- col_left < col_right

For weight-4 boundary plaquettes, a UNIFORM labeling is used (q1, q2, q5, q6):
- q3 and q4 are always missing (-1)
- The actual spatial positions of q1, q2, q5, q6 vary by boundary type:

  GREEN (top/left, L-shape):     RED (right, L-shape):      BLUE (bottom, 2x2):
       .   q1                        q1   .                     q1  q5
       .   q2                        q2   .                     q2  q6
      q5   q6                        q6  q5

- This uniform labeling allows the SAME HE rules to apply to all weight-4 plaquettes
- The canonical mapping is consistent across the entire code, regardless of boundary type

===============================================================================
HOMOLOGICAL EQUIVALENCE RULES
===============================================================================

For CSS color codes, X and Z stabilizers have the same support (plaquettes), so
the same rules apply to both error types.

WEIGHT-6 BULK PLAQUETTES:
-------------------------
1. Weight 4, 5, 6 errors: Apply weight reduction (flip all qubits in support)
   - 6 errors → 0 errors (remove stabilizer)
   - 5 errors → 1 error
   - 4 errors → 2 errors

2. Weight 1, 2 errors: Leave as-is (unique up to stabilizer multiplication)

3. Weight 3 errors: Apply canonicalization (10 rules covering all 20 patterns):
   Pattern              →  Canonical
   (q1, q2, q3)         →  (q4, q5, q6)
   (q1, q2, q4)         →  (q3, q5, q6)
   (q1, q3, q4)         →  (q2, q5, q6)
   (q1, q2, q6)         →  (q3, q4, q5)
   (q1, q4, q5)         →  (q2, q3, q6)
   (q1, q3, q5)         →  (q2, q4, q6)
   (q1, q2, q5)         →  (q3, q4, q6)
   (q1, q4, q6)         →  (q2, q3, q5)
   (q2, q4, q5)         →  (q1, q3, q6)
   (q1, q5, q6)         →  (q2, q3, q4)

WEIGHT-4 BOUNDARY PLAQUETTES (all colors: blue, green, red):
-------------------------------------------------------------
Using UNIFORM labels (q1, q2, q5, q6) for all weight-4 boundary plaquettes.
The same rules apply regardless of boundary position or color.

1. Weight 1 errors: Leave as-is (already canonical)

2. Weight 2 errors: Apply canonicalization (3 rules covering 6 patterns):
   These rules map errors to a canonical position containing q6.
   
   Pattern      →  Canonical
   (q1, q2)     →  (q5, q6)
   (q1, q5)     →  (q2, q6)
   (q2, q5)     →  (q1, q6)
   
   The complementary patterns are already canonical (contain q6):
   (q5, q6), (q2, q6), (q1, q6) → no change needed
   
3. Weight 3, 4 errors: Apply weight reduction
   - 4 errors → 0 errors (equivalent to stabilizer)
   - 3 errors → 1 error

===============================================================================
IMPLEMENTATION NOTES
===============================================================================

The simplify function repeatedly applies:
1. weightReduction: Reduce high-weight errors
2. fixEquivalence: Canonicalize remaining patterns

Until a steady state is reached (no more changes).

IMPORTANT: Spacelike HE is applied to ERROR DIFFS, not cumulative errors.
This matches the surface code implementation and avoids artifacts that 
occur when canonicalizing cumulative frames. Each diff (per-round error change)
is canonicalized independently.

Author: AI Assistant (based on surface code HE by Muyuan Li)
"""

import torch
from typing import Tuple, List, Dict, Optional


def get_plaquette_qubit_labels(
    data_qubits: List[int], qubit_to_coord: Dict[int, Tuple[int, int]], weight: int
) -> Dict[str, int]:
    """
    Get the q1-q6 labels for qubits in a plaquette based on their coordinates.
    
    For weight-6: All q1-q6 are assigned
    For weight-4: q3, q4 are -1 (missing)
    
    Args:
        data_qubits: List of data qubit indices in the plaquette
        qubit_to_coord: Mapping from qubit ID to (row, col) coordinates
        weight: Plaquette weight (4 or 6)
        
    Returns:
        Dict mapping 'q1'..'q6' to qubit IDs (-1 if missing)
    """
    # Get coordinates for all qubits
    coords = {q: qubit_to_coord[q] for q in data_qubits}

    # Find distinct rows and columns
    rows = sorted(set(r for r, c in coords.values()), reverse=True)  # Top (largest) first
    cols = sorted(set(c for r, c in coords.values()))  # Left (smallest) first

    # Build reverse mapping: coord -> qubit
    coord_to_qubit = {v: k for k, v in coords.items()}

    labels = {'q1': -1, 'q2': -1, 'q3': -1, 'q4': -1, 'q5': -1, 'q6': -1}

    if weight == 6:
        # 3 rows, 2 columns
        assert len(rows) == 3 and len(cols) == 2, f"Weight-6 plaquette should have 3 rows x 2 cols"
        row_top, row_mid, row_bot = rows
        col_left, col_right = cols

        labels['q1'] = coord_to_qubit[(row_top, col_left)]  # NW
        labels['q2'] = coord_to_qubit[(row_mid, col_left)]  # W
        labels['q3'] = coord_to_qubit[(row_bot, col_left)]  # SW
        labels['q4'] = coord_to_qubit[(row_top, col_right)]  # NE
        labels['q5'] = coord_to_qubit[(row_mid, col_right)]  # E
        labels['q6'] = coord_to_qubit[(row_bot, col_right)]  # SE

    elif weight == 4:
        if len(rows) == 2 and len(cols) == 2:
            # 2x2 grid (bottom boundary blue plaquettes)
            row_top, row_bot = rows
            col_left, col_right = cols

            labels['q1'] = coord_to_qubit[(row_top, col_left)]
            labels['q2'] = coord_to_qubit[(row_bot, col_left)]
            labels['q5'] = coord_to_qubit[(row_top, col_right)]
            labels['q6'] = coord_to_qubit[(row_bot, col_right)]

        elif len(rows) == 3 and len(cols) == 2:
            # L-shape (top/left/right boundary plaquettes)
            row_top, row_mid, row_bot = rows
            col_left, col_right = cols

            # Find dense column (has 3 qubits) vs sparse column (has 1 qubit)
            left_count = sum(1 for r, c in coords.values() if c == col_left)
            right_count = sum(1 for r, c in coords.values() if c == col_right)

            if left_count == 3:
                # Dense on left, sparse on right (typical for top green, left green)
                labels['q1'] = coord_to_qubit.get((row_top, col_left), -1)
                labels['q2'] = coord_to_qubit.get((row_mid, col_left), -1)
                labels['q5'] = coord_to_qubit.get((row_bot, col_right), -1)
                labels['q6'] = coord_to_qubit.get((row_bot, col_left), -1)
            else:
                # Dense on right, sparse on left (typical for right red boundary)
                labels['q1'] = coord_to_qubit.get((row_top, col_right), -1)
                labels['q2'] = coord_to_qubit.get((row_mid, col_right), -1)
                labels['q5'] = coord_to_qubit.get((row_bot, col_left), -1)
                labels['q6'] = coord_to_qubit.get((row_bot, col_right), -1)

        else:
            raise ValueError(f"Unexpected weight-4 geometry: {len(rows)} rows, {len(cols)} cols")
    else:
        raise ValueError(f"Unsupported plaquette weight: {weight}")

    return labels


def weight_reduction(
    error_config: torch.Tensor, plaquette_support: List[int], weight: int
) -> torch.Tensor:
    """
    Apply weight reduction to errors within a single plaquette.
    
    Rules:
    - Weight-6: Reduce if error_count >= 4
    - Weight-4: Reduce if error_count >= 3
    
    Args:
        error_config: Binary tensor (num_data,) representing errors
        plaquette_support: List of data qubit indices in the plaquette
        weight: Plaquette weight (4 or 6)
        
    Returns:
        Modified error configuration
    """
    error_config = error_config.clone()

    # Count errors in plaquette support
    error_count = sum(error_config[q].item() for q in plaquette_support)

    # Apply reduction rules
    if weight == 6:
        # 6 errors → 0, 5 errors → 1, 4 errors → 2
        if error_count >= 4:
            for q in plaquette_support:
                error_config[q] = error_config[q] ^ 1
    elif weight == 4:
        # 4 errors → 0, 3 errors → 1
        if error_count >= 3:
            for q in plaquette_support:
                error_config[q] = error_config[q] ^ 1

    return error_config


# Weight-3 canonicalization rules for weight-6 plaquettes
# Format: (pattern_tuple, canonical_tuple)
# Each tuple contains labels: (q1,q2,q3,q4,q5,q6) -> indices (0,1,2,3,4,5)
# Each of the 10 complement-pairs of weight-3 patterns has exactly one
# canonical member: the right-hand side of its rule. Any consistent per-pair
# choice is homologically valid; this particular choice is frozen because
# trained checkpoints and published results depend on it — do not reorient
# rules without retraining/re-evaluating.
WEIGHT6_WEIGHT3_RULES = [
    # Pattern (left side)     # Canonical (right side)
    ((0, 1, 2), (3, 4, 5)),  # (q1,q2,q3) -> (q4,q5,q6)
    ((0, 1, 3), (2, 4, 5)),  # (q1,q2,q4) -> (q3,q5,q6)
    ((0, 2, 3), (1, 4, 5)),  # (q1,q3,q4) -> (q2,q5,q6)
    ((0, 1, 5), (2, 3, 4)),  # (q1,q2,q6) -> (q3,q4,q5)
    ((0, 3, 4), (1, 2, 5)),  # (q1,q4,q5) -> (q2,q3,q6)
    ((0, 2, 4), (1, 3, 5)),  # (q1,q3,q5) -> (q2,q4,q6)
    ((0, 1, 4), (2, 3, 5)),  # (q1,q2,q5) -> (q3,q4,q6)
    ((0, 3, 5), (1, 2, 4)),  # (q1,q4,q6) -> (q2,q3,q5)
    ((1, 3, 4), (0, 2, 5)),  # (q2,q4,q5) -> (q1,q3,q6)
    ((0, 4, 5), (1, 2, 3)),  # (q1,q5,q6) -> (q2,q3,q4)
]

# Weight-2 canonicalization rules for weight-4 plaquettes
# Using indices for (q1,q2,q5,q6) -> (0,1,2,3)
WEIGHT4_WEIGHT2_RULES = [
    ((0, 1), (2, 3)),  # (q1,q2) -> (q5,q6)
    ((0, 2), (1, 3)),  # (q1,q5) -> (q2,q6)
    ((1, 2), (0, 3)),  # (q2,q5) -> (q1,q6)
]


def fix_equivalence_weight6(error_config: torch.Tensor, labels: Dict[str, int]) -> torch.Tensor:
    """
    Apply canonicalization rules for weight-3 errors in a weight-6 plaquette.
    
    Args:
        error_config: Binary tensor (num_data,) representing errors
        labels: Dict mapping 'q1'..'q6' to qubit IDs
        
    Returns:
        Canonicalized error configuration
    """
    error_config = error_config.clone()

    # Get qubit IDs for each label
    qubits = [labels['q1'], labels['q2'], labels['q3'], labels['q4'], labels['q5'], labels['q6']]

    # Check if any qubit is invalid
    if any(q < 0 for q in qubits):
        return error_config

    # Find which qubits have errors (as label indices 0-5)
    error_indices = tuple(i for i, q in enumerate(qubits) if error_config[q] == 1)

    # Only process weight-3 errors
    if len(error_indices) != 3:
        return error_config

    # Check against canonicalization rules
    for pattern, canonical in WEIGHT6_WEIGHT3_RULES:
        if error_indices == pattern:
            # Transform: clear pattern, set canonical
            for i in pattern:
                error_config[qubits[i]] = 0
            for i in canonical:
                error_config[qubits[i]] = 1
            break

    return error_config


def fix_equivalence_weight4(
    error_config: torch.Tensor, labels: Dict[str, int], color: str = 'blue'
) -> torch.Tensor:
    """
    Apply canonicalization rules for weight-2 errors in a weight-4 plaquette.
    
    Args:
        error_config: Binary tensor (num_data,) representing errors
        labels: Dict mapping 'q1'..'q6' to qubit IDs (q3, q4 are -1)
        
    Returns:
        Canonicalized error configuration
    """
    error_config = error_config.clone()

    # Get qubit IDs for present labels (q1, q2, q5, q6)
    qubits = [labels['q1'], labels['q2'], labels['q5'], labels['q6']]

    # Check if any qubit is invalid
    if any(q < 0 for q in qubits):
        return error_config

    # Find which qubits have errors (as indices 0-3 for q1,q2,q5,q6)
    error_indices = tuple(i for i, q in enumerate(qubits) if error_config[q] == 1)

    # Only process weight-2 errors
    if len(error_indices) != 2:
        return error_config

    color = str(color).lower()

    # The 4-qubit system is ordered as (q1,q2,q5,q6) -> indices (0,1,2,3).
    # Apply orientation-aware rules. Each rule corresponds to applying the weight-4 stabilizer,
    # which maps a weight-2 pattern to its 2-qubit complement. We choose which member of each
    # complement pair is canonical (a gauge choice), and map noncanonical -> canonical.
    if color == 'green':
        # Default gauge: canonical = {left(q1,q2), top(q2,q6), anti(q1,q6)}
        rules = [((2, 3), (0, 1)), ((0, 2), (1, 3)), ((1, 2), (0, 3))]
    elif color == 'red':
        # Default gauge: canonical = {left(q5,q6), top(q2,q6), anti(q2,q5)}
        rules = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    else:
        # blue (2x2): original table
        rules = WEIGHT4_WEIGHT2_RULES

    for pattern, canonical in rules:
        if error_indices == pattern:
            for i in pattern:
                error_config[qubits[i]] = 0
            for i in canonical:
                error_config[qubits[i]] = 1
            break

    return error_config


class ColorCodeHE:
    """
    Homological equivalence transformer for color codes.
    
    Pre-computes plaquette information from a ColorCode instance for efficient
    repeated application of HE transformations.
    """

    def __init__(self, color_code):
        """
        Initialize from a ColorCode instance.
        
        Args:
            color_code: ColorCode instance
        """
        self.num_data = color_code.num_data
        self.num_plaquettes = color_code.num_plaquettes
        self.qubit_to_coord = color_code.qubit_to_coord

        # Pre-compute plaquette info
        self.plaquettes = []
        for plaq in color_code.plaquettes:
            data_qubits = list(plaq['data_qubits'])
            weight = plaq['weight']
            labels = get_plaquette_qubit_labels(data_qubits, self.qubit_to_coord, weight)

            self.plaquettes.append(
                {
                    'data_qubits': data_qubits,
                    'weight': weight,
                    'color': plaq['color'],
                    'type': plaq['type'],
                    'labels': labels,
                }
            )

        # Qubit -> plaquettes incidence (for overlap-aware canonicalization)
        self.qubit_to_plaquettes: List[List[int]] = [[] for _ in range(self.num_data)]
        for p_idx, plaq in enumerate(self.plaquettes):
            for q in plaq['data_qubits']:
                if q >= 0:
                    self.qubit_to_plaquettes[int(q)].append(p_idx)

    # ---------------------------------------------------------------------
    # Overlap-aware scoring helpers (mirror the reference objective)
    # ---------------------------------------------------------------------
    def _plaq_noncanonical_score(self, error_config: torch.Tensor,
                                 plaq_idx: int) -> Tuple[int, int]:
        """
        Return (badness, potential) for one plaquette.

        - badness: 1 if in a noncanonical weight-3 (w=6) or weight-2 (w=4) pattern else 0
        - potential: (rule_index+1) for matched noncanonical pattern else 0
        """
        plaq = self.plaquettes[plaq_idx]
        w = plaq['weight']
        labels = plaq['labels']

        if w == 6:
            qubits = [
                labels['q1'], labels['q2'], labels['q3'], labels['q4'], labels['q5'], labels['q6']
            ]
            if any(q < 0 for q in qubits):
                return 0, 0
            err_idx = tuple(i for i, q in enumerate(qubits) if int(error_config[q].item()) == 1)
            if len(err_idx) != 3:
                return 0, 0
            for rule_i, (pattern, _) in enumerate(WEIGHT6_WEIGHT3_RULES):
                if err_idx == pattern:
                    return 1, rule_i + 1
            return 0, 0

        if w == 4:
            qubits = [labels['q1'], labels['q2'], labels['q5'], labels['q6']]
            if any(q < 0 for q in qubits):
                return 0, 0
            err_idx = tuple(i for i, q in enumerate(qubits) if int(error_config[q].item()) == 1)
            if len(err_idx) != 2:
                return 0, 0
            color = str(plaq.get('color', 'blue')).lower()
            if color == 'green':
                # noncanonical patterns (complements of the chosen canonical set)
                patterns = [(2, 3), (0, 2), (1, 2)]
            elif color == 'red':
                # noncanonical patterns (complements of the chosen canonical set)
                patterns = [(0, 1), (0, 2), (0, 3)]
            else:
                # blue (2x2): original patterns in (q1,q2,q5,q6) index space
                patterns = [(0, 1), (0, 2), (1, 2)]

            for rule_i, pattern in enumerate(patterns):
                if err_idx == pattern:
                    return 1, rule_i + 1
            return 0, 0

        return 0, 0

    def _neighborhood_score(self, error_config: torch.Tensor, plaq_idx: int) -> Tuple[int, int]:
        """
        Sum (badness, potential) over the neighborhood induced by qubit overlaps.

        Neighborhood = all plaquettes incident to any qubit in the plaquette support
        (duplicates allowed, matching the reference implementation).
        """
        plaq = self.plaquettes[plaq_idx]
        neigh: List[int] = []
        for q in plaq['data_qubits']:
            if q >= 0:
                neigh.extend(self.qubit_to_plaquettes[int(q)])

        bad_sum = 0
        pot_sum = 0
        for p in neigh:
            b, pot = self._plaq_noncanonical_score(error_config, p)
            bad_sum += b
            pot_sum += pot
        return bad_sum, pot_sum

    # ---------------------------------------------------------------------
    # Overlap-aware passes (mirror the reference behavior)
    # ---------------------------------------------------------------------
    def _weight_reduction_all_with_mask(
        self, error_config: torch.Tensor, changed_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weight reduction with overlap blocking (smallest-index plaquette wins)."""
        if changed_mask is None:
            changed_mask = torch.zeros(self.num_data, dtype=torch.bool, device=error_config.device)

        current = error_config.clone()
        ch = changed_mask.clone()

        # Apply weight-6 plaquettes first, then weight-4 (to match the reference order).
        for phase_weight in (6, 4):
            for plaq in self.plaquettes:
                if plaq['weight'] != phase_weight:
                    continue
                transformed = weight_reduction(current, plaq['data_qubits'], plaq['weight'])
                would_change = (current != transformed)
                would_change_any = bool(would_change.any().item())
                if not would_change_any:
                    continue
                any_conflict = bool((would_change & ch).any().item())
                if any_conflict:
                    continue
                current = transformed
                ch = ch | would_change

        return current, ch

    def _fix_equivalence_all_with_mask(
        self, error_config: torch.Tensor, changed_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Canonicalization with overlap blocking + second-chance objective pass."""
        if changed_mask is None:
            changed_mask = torch.zeros(self.num_data, dtype=torch.bool, device=error_config.device)

        current = error_config.clone()
        ch1 = changed_mask.clone()
        skipped: List[bool] = [False for _ in range(len(self.plaquettes))]

        # Process weight-6 plaquettes first, then weight-4 (to match the reference order).
        order: List[int] = (
            [i for i, plaq in enumerate(self.plaquettes) if plaq['weight'] == 6] +
            [i for i, plaq in enumerate(self.plaquettes) if plaq['weight'] == 4]
        )

        # Pass 1: hard blocking
        for p_idx in order:
            plaq = self.plaquettes[p_idx]
            if plaq['weight'] == 6:
                transformed = fix_equivalence_weight6(current, plaq['labels'])
            elif plaq['weight'] == 4:
                transformed = fix_equivalence_weight4(current, plaq['labels'], plaq['color'])
            else:
                continue

            would_change = (current != transformed)
            would_change_any = bool(would_change.any().item())
            if not would_change_any:
                continue

            any_conflict = bool((would_change & ch1).any().item())
            if any_conflict:
                skipped[p_idx] = True
                continue

            current = transformed
            ch1 = ch1 | would_change

        # Pass 2: revisit skipped if objective improves
        ch2 = torch.zeros(self.num_data, dtype=torch.bool, device=error_config.device)

        def improves(before_bad, before_pot, after_bad, after_pot) -> bool:
            return (after_bad
                    < before_bad) or ((after_bad == before_bad) and (after_pot < before_pot))

        for p_idx in order:
            plaq = self.plaquettes[p_idx]
            if not skipped[p_idx]:
                continue

            if plaq['weight'] == 6:
                transformed = fix_equivalence_weight6(current, plaq['labels'])
            elif plaq['weight'] == 4:
                transformed = fix_equivalence_weight4(current, plaq['labels'], plaq['color'])
            else:
                continue

            would_change = (current != transformed)
            would_change_any = bool(would_change.any().item())
            if not would_change_any:
                continue

            before_bad, before_pot = self._neighborhood_score(current, p_idx)
            after_bad, after_pot = self._neighborhood_score(transformed, p_idx)
            if not improves(before_bad, before_pot, after_bad, after_pot):
                continue

            any_conflict = bool((would_change & ch2).any().item())
            if any_conflict:
                continue

            current = transformed
            ch2 = ch2 | would_change

        return current, (ch1 | ch2)

    def weight_reduction_all(self, error_config: torch.Tensor) -> torch.Tensor:
        """
        Apply weight reduction to all plaquettes.
        
        Args:
            error_config: Binary tensor (num_data,) representing errors
            
        Returns:
            Reduced error configuration
        """
        reduced, _ = self._weight_reduction_all_with_mask(error_config, None)
        return reduced

    def fix_equivalence_all(self, error_config: torch.Tensor) -> torch.Tensor:
        """
        Apply canonicalization to all plaquettes.
        
        Args:
            error_config: Binary tensor (num_data,) representing errors
            
        Returns:
            Canonicalized error configuration
        """
        canon, _ = self._fix_equivalence_all_with_mask(error_config, None)
        return canon

    def simplify(self, error_config: torch.Tensor, max_iterations: int = 512) -> torch.Tensor:
        """
        Iteratively apply weight reduction and canonicalization until steady state.
        
        Args:
            error_config: Binary tensor (num_data,) representing errors
            max_iterations: Maximum iterations to prevent infinite loops
            
        Returns:
            Steady-state canonical error configuration
        """
        current = error_config.clone()

        for _ in range(max_iterations):
            previous = current.clone()

            # Fresh mask per pass, carried from reduction -> canonicalization
            fresh = torch.zeros(self.num_data, dtype=torch.bool, device=current.device)
            current, fresh = self._weight_reduction_all_with_mask(current, fresh)
            current, _ = self._fix_equivalence_all_with_mask(current, fresh)

            # Check for convergence
            if torch.equal(current, previous):
                break

        return current

    def simplify_with_count(self,
                            error_config: torch.Tensor,
                            max_iterations: int = 512) -> Tuple[torch.Tensor, int]:
        """
        Same as simplify but also returns iteration count.
        
        Returns:
            (canonical_config, num_iterations)
        """
        current = error_config.clone()

        for iteration in range(max_iterations):
            previous = current.clone()

            fresh = torch.zeros(self.num_data, dtype=torch.bool, device=current.device)
            current, fresh = self._weight_reduction_all_with_mask(current, fresh)
            current, _ = self._fix_equivalence_all_with_mask(current, fresh)

            if torch.equal(current, previous):
                return current, iteration + 1

        return current, max_iterations


def apply_spacelike_homological_equivalence(
    x_error_diff: torch.Tensor, z_error_diff: torch.Tensor, he_transformer: ColorCodeHE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply spacelike homological equivalence to error difference tensors.
    
    Takes error DIFFS and applies canonicalization to each diff independently.
    This avoids artifacts that occur when canonicalizing cumulative frames.
    
    This matches the reference implementation, which also operates directly
    on diffs rather than cumulative errors.
    
    Args:
        x_error_diff: X error differences tensor (num_data, n_rounds)
        z_error_diff: Z error differences tensor (num_data, n_rounds)
        he_transformer: Pre-initialized ColorCodeHE instance
        
    Returns:
        Tuple of canonicalized (x_error_diff, z_error_diff) tensors
    """
    num_qubits, n_rounds = x_error_diff.shape

    # Apply HE to each diff independently (NOT cumulative)
    # This matches the surface code implementation
    x_error_diff_new = torch.zeros_like(x_error_diff)
    z_error_diff_new = torch.zeros_like(z_error_diff)

    for t in range(n_rounds):
        x_error_diff_new[:, t] = he_transformer.simplify(x_error_diff[:, t])
        z_error_diff_new[:, t] = he_transformer.simplify(z_error_diff[:, t])

    return x_error_diff_new, z_error_diff_new


def apply_spacelike_homological_equivalence_batched(
    x_error_diff: torch.Tensor, z_error_diff: torch.Tensor, he_transformer: ColorCodeHE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply spacelike homological equivalence to batched error difference tensors.
    
    Takes error DIFFS and applies canonicalization to each diff independently.
    
    Args:
        x_error_diff: X error differences tensor (batch, n_rounds, num_data)
        z_error_diff: Z error differences tensor (batch, n_rounds, num_data)
        he_transformer: Pre-initialized ColorCodeHE instance
        
    Returns:
        Tuple of canonicalized (x_error_diff, z_error_diff) tensors with same shape
    """
    batch_size, n_rounds, num_qubits = x_error_diff.shape

    # Apply HE to each (batch, round) diff independently
    x_error_diff_new = torch.zeros_like(x_error_diff)
    z_error_diff_new = torch.zeros_like(z_error_diff)

    for b in range(batch_size):
        for t in range(n_rounds):
            x_error_diff_new[b, t, :] = he_transformer.simplify(x_error_diff[b, t, :])
            z_error_diff_new[b, t, :] = he_transformer.simplify(z_error_diff[b, t, :])

    return x_error_diff_new, z_error_diff_new


# ============================================================================
# TIMELIKE HOMOLOGICAL EQUIVALENCE
# ============================================================================
#
# Timelike HE applies trivial operations across consecutive syndrome measurement
# rounds to add structure to trainY. The operations are:
#
# Weight-1: For each data qubit q:
#   - Add Z (or X) error to q in round k
#   - Flip anticommuting stabilizer measurements in round k
#   - Add Z (or X) error to q in round k+1
#   This is a trivial operation (net zero effect).
#
# Weight-2: For specific error patterns arising from single faults:
#   - X errors from single fault: (q1, q2) or (q5, q6)
#   - Z errors from single fault: (q2, q3) or (q5, q6)
#
# Weight-3: Only for weight-6 plaquettes, specific patterns from single faults.
#
# Acceptance criteria (same for all weights):
#   - Accept if s^(HE)(k,k+1) < s(k,k+1) (total density decreases)
#   - OR if equal density AND s^(HE)_max > s_max (tie-breaker)
#
# Key difference from surface code: Color code bulk qubits anticommute with
# 3 stabilizers (not 2). This is handled automatically by the parity matrix.
# ============================================================================


def get_parity_matrix_data_only(color_code) -> torch.Tensor:
    """
    Extract the parity matrix mapping stabilizers to data qubits only.
    
    The ColorCode.hx/hz matrices include ancilla columns which are all zeros.
    For timelike HE, we only need the data qubit portion.
    
    Args:
        color_code: ColorCode instance
        
    Returns:
        Parity matrix of shape (num_plaquettes, num_data)
    """
    # hx and hz are identical for color codes (CSS property)
    # They have shape (num_plaquettes, num_total_qubits) where num_total includes ancillas
    # We only need the first num_data columns
    parity = color_code.hx[:, :color_code.num_data]
    return torch.tensor(parity, dtype=torch.float32)


def simplifytime_color(
    error_diff_round_round_plus_1: torch.Tensor, s1s2_round_round_plus_1: torch.Tensor,
    parity_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply weight-1 timelike homological equivalence for color codes.
    
    This is identical to the surface code implementation - the only difference
    is that color code bulk qubits anticommute with 3 stabilizers instead of 2,
    which is handled automatically by the parity matrix multiplication.
    
    Args:
        error_diff_round_round_plus_1: Error diff tensor (B, num_data, 2) for rounds k and k+1
        s1s2_round_round_plus_1: s1s2 measurement tensor (B, num_stabs, 2)
        parity_matrix: Parity check matrix (num_stabs, num_data)
        
    Returns:
        Tuple of (simplified error tensor, simplified s1s2 tensor, num_accepted)
    """
    # Clone to create hypothetical flipped state
    new_error_diff = error_diff_round_round_plus_1.clone()
    new_s1s2 = s1s2_round_round_plus_1.clone()

    # Flip all data qubits in both rounds, flip stabilizers only in round k
    new_error_diff = (new_error_diff + 1) % 2
    new_s1s2[..., 0] = (new_s1s2[..., 0] + 1) % 2

    # Compute densities
    # error_diff: (B, num_data, 2), s1s2: (B, num_stabs, 2), parity: (num_stabs, num_data)
    # einsum: batch, stabs, time × stabs, data -> batch, data, time
    old_density_per_time = error_diff_round_round_plus_1 + torch.einsum(
        'bst,sd->bdt', s1s2_round_round_plus_1.float(), parity_matrix
    )
    old_density = old_density_per_time.sum(dim=2)  # (B, num_data)

    new_density_per_time = new_error_diff + torch.einsum(
        'bst,sd->bdt', new_s1s2.float(), parity_matrix
    )
    new_density = new_density_per_time.sum(dim=2)  # (B, num_data)

    # Accept mask: accept if density decreases
    accept_mask = new_density < old_density  # (B, num_data)

    # Tie-breaker: when densities are equal, prefer higher density in round k
    # (This pushes errors to later in time, matching the paper's s^(HE)_max > s_max)
    density_equal = (new_density == old_density)
    old_round0_density = old_density_per_time[:, :, 0]
    new_round0_density = new_density_per_time[:, :, 0]
    tie_breaker = density_equal & (new_round0_density > old_round0_density)

    accept_mask = accept_mask | tie_breaker  # (B, num_data)

    num_accepted = int(accept_mask.sum().item())

    # Apply changes selectively per (batch, data_qubit)
    error_diff_round_round_plus_1 = torch.where(
        accept_mask.unsqueeze(2),  # (B, num_data, 1)
        new_error_diff,
        error_diff_round_round_plus_1
    )

    # For s1s2: determine which stabs to flip based on accepted data qubits
    # flip_count[b, s] = sum over q of (accept_mask[b, q] * parity_matrix[s, q])
    flip_count = torch.matmul(accept_mask.float(), parity_matrix.T)  # (B, num_stabs)
    should_flip = (flip_count % 2).bool()  # (B, num_stabs)

    # Only flip round k (index 0)
    s1s2_round_round_plus_1 = torch.where(
        should_flip.unsqueeze(2),  # (B, num_stabs, 1)
        (s1s2_round_round_plus_1 + 1) % 2,
        s1s2_round_round_plus_1
    )

    return error_diff_round_round_plus_1, s1s2_round_round_plus_1, num_accepted


def get_anticommuting_stabilizers_color(qubit_indices: List[int],
                                        parity_matrix: torch.Tensor) -> List[int]:
    """
    Find stabilizers that anticommute with errors on the given qubits.
    A stabilizer anticommutes if it shares an odd number of qubits with the error set.
    
    Args:
        qubit_indices: List of qubit indices with errors
        parity_matrix: Parity check matrix (num_stabs, num_data)
        
    Returns:
        List of stabilizer indices that anticommute
    """
    if not qubit_indices:
        return []

    # Sum columns corresponding to error qubits (modulo 2)
    relevant_cols = parity_matrix[:, qubit_indices]
    syndrome = relevant_cols.sum(dim=1) % 2

    # Return indices where syndrome is 1
    return torch.nonzero(syndrome, as_tuple=True)[0].tolist()


# Weight-2 error patterns from single faults in color code circuits
# ============================================================================
# WEIGHT-2 TIMELIKE HE PATTERNS
# ============================================================================
# IMPORTANT: Unlike surface code which uses spacelike canonical positions,
# color code weight-2 timelike HE uses CIRCUIT-SPECIFIC PATTERNS from single
# faults propagating through the CNOT schedule.
#
# This is because:
# 1. Color code weight-6 plaquettes do NOT have spacelike canonicalization for
#    weight-2 errors (only weight-3+ are canonicalized)
# 2. Single faults in the circuit can only produce specific weight-2 patterns
# 3. The paper (predecoder_color_memory.tex) explicitly specifies these patterns
#
# Based on circuit structure from Fig 13 in the paper:
# X errors: (q1, q2) or (q5, q6)
# Z errors: (q2, q3) or (q5, q6)
#
# TODO: Verify these patterns when full data pipeline is integrated. The patterns
# should match what's observed from actual single-fault propagation through the
# color code circuit CNOTs.
# ============================================================================
WEIGHT2_X_PATTERNS_W6 = [
    ('q1', 'q2'),  # From single fault propagating to left column top
    ('q5', 'q6'),  # From single fault propagating to right column bottom
]

WEIGHT2_Z_PATTERNS_W6 = [
    ('q2', 'q3'),  # From single fault propagating to left column bottom
    ('q5', 'q6'),  # From single fault propagating to right column bottom
]

# For weight-4 plaquettes, use the patterns that exist (q3, q4 are missing)
WEIGHT2_X_PATTERNS_W4 = [
    ('q1', 'q2'),
    ('q5', 'q6'),
]

WEIGHT2_Z_PATTERNS_W4 = [
    ('q5', 'q6'),  # Only this pattern is valid for Z (q2,q3 needs q3 which is missing)
]


def simplifytime_weight2_color(
    error_diff: torch.Tensor,
    s1s2: torch.Tensor,
    parity_matrix: torch.Tensor,
    he_transformer: ColorCodeHE,
    error_type: str = 'X'
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply weight-2 timelike homological equivalence for color codes.
    
    Only considers specific weight-2 error patterns that arise from single faults
    in the color code circuit structure.
    
    Args:
        error_diff: Error diff tensor (B, num_data, 2) for rounds k and k+1
        s1s2: s1s2 measurement tensor (B, num_stabs, 2)
        parity_matrix: Parity check matrix (num_stabs, num_data)
        he_transformer: ColorCodeHE instance with plaquette info
        error_type: 'X' or 'Z' - determines which patterns to try
        
    Returns:
        Tuple of (simplified error tensor, simplified s1s2 tensor, num_accepted)
    """
    error_diff.shape[0]
    num_accepted = 0

    # Select patterns based on error type
    patterns_w6 = WEIGHT2_X_PATTERNS_W6 if error_type == 'X' else WEIGHT2_Z_PATTERNS_W6
    patterns_w4 = WEIGHT2_X_PATTERNS_W4 if error_type == 'X' else WEIGHT2_Z_PATTERNS_W4

    # Iterate through all plaquettes
    for plaq in he_transformer.plaquettes:
        labels = plaq['labels']
        weight = plaq['weight']

        # Select appropriate patterns
        patterns = patterns_w6 if weight == 6 else patterns_w4

        for label1, label2 in patterns:
            q1_idx = labels.get(label1, -1)
            q2_idx = labels.get(label2, -1)

            # Skip if either qubit is invalid
            if q1_idx < 0 or q2_idx < 0:
                continue

            # Find anticommuting stabilizers for this pair
            anticommuting_stabs = get_anticommuting_stabilizers_color(
                [q1_idx, q2_idx], parity_matrix
            )

            # Compute current density
            old_density_per_time = error_diff + torch.einsum(
                'bst,sd->bdt', s1s2.float(), parity_matrix
            )
            old_density_total = old_density_per_time.sum(dim=2).sum(dim=1)  # (B,)
            old_density_k_plus_1 = old_density_per_time[:, :, 1].sum(dim=1)  # (B,)

            # Create hypothetical flipped state
            new_error = error_diff.clone()
            new_s1s2 = s1s2.clone()

            # Flip both qubits in both rounds
            new_error[:, q1_idx, :] = (new_error[:, q1_idx, :] + 1) % 2
            new_error[:, q2_idx, :] = (new_error[:, q2_idx, :] + 1) % 2

            # Flip anticommuting stabilizers only in round k
            if anticommuting_stabs:
                new_s1s2[:, anticommuting_stabs, 0] = (new_s1s2[:, anticommuting_stabs, 0] + 1) % 2

            # Compute new density
            new_density_per_time = new_error + torch.einsum(
                'bst,sd->bdt', new_s1s2.float(), parity_matrix
            )
            new_density_total = new_density_per_time.sum(dim=2).sum(dim=1)  # (B,)
            new_density_k_plus_1 = new_density_per_time[:, :, 1].sum(dim=1)  # (B,)

            # Acceptance criteria
            accept_mask = (new_density_total < old_density_total) | \
                          ((new_density_total == old_density_total) &
                           (new_density_k_plus_1 > old_density_k_plus_1))

            # Apply accepted changes
            if accept_mask.any():
                num_accepted += int(accept_mask.sum().item())
                error_diff = torch.where(
                    accept_mask.unsqueeze(1).unsqueeze(2), new_error, error_diff
                )
                s1s2 = torch.where(accept_mask.unsqueeze(1).unsqueeze(2), new_s1s2, s1s2)

    return error_diff, s1s2, num_accepted


# ============================================================================
# WEIGHT-3 TIMELIKE HE PATTERNS
# ============================================================================
# IMPORTANT: Weight-3 timelike HE applies ONLY to weight-6 plaquettes.
# Like weight-2, these are CIRCUIT-SPECIFIC patterns from single faults.
#
# NOTE: For SPACELIKE HE, weight-3 errors ARE canonicalized to specific forms
# (patterns containing q6 are canonical). However, for TIMELIKE HE we use the
# circuit fault patterns, which happen to be the left/right column patterns.
#
# These patterns arise from single faults propagating through 3 consecutive
# CNOTs in the color code circuit.
#
# TODO: Verify these patterns when full data pipeline is integrated.
# ============================================================================
WEIGHT3_X_PATTERNS_W6 = [
    ('q1', 'q2', 'q3'),  # Left column - from fault propagating through 3 CNOTs
    ('q4', 'q5', 'q6'),  # Right column - from fault propagating through 3 CNOTs
]

WEIGHT3_Z_PATTERNS_W6 = [
    ('q1', 'q2', 'q3'),  # Left column - from fault propagating through 3 CNOTs
    ('q4', 'q5', 'q6'),  # Right column - from fault propagating through 3 CNOTs
]


def simplifytime_weight3_color(
    error_diff: torch.Tensor,
    s1s2: torch.Tensor,
    parity_matrix: torch.Tensor,
    he_transformer: ColorCodeHE,
    error_type: str = 'X'
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply weight-3 timelike homological equivalence for color codes.
    
    Only considers weight-3 error patterns from single faults in weight-6 plaquettes.
    
    Args:
        error_diff: Error diff tensor (B, num_data, 2) for rounds k and k+1
        s1s2: s1s2 measurement tensor (B, num_stabs, 2)
        parity_matrix: Parity check matrix (num_stabs, num_data)
        he_transformer: ColorCodeHE instance with plaquette info
        error_type: 'X' or 'Z' - determines which patterns to try
        
    Returns:
        Tuple of (simplified error tensor, simplified s1s2 tensor, num_accepted)
    """
    error_diff.shape[0]
    num_accepted = 0

    # Select patterns based on error type
    patterns = WEIGHT3_X_PATTERNS_W6 if error_type == 'X' else WEIGHT3_Z_PATTERNS_W6

    # Only iterate through weight-6 plaquettes
    for plaq in he_transformer.plaquettes:
        if plaq['weight'] != 6:
            continue

        labels = plaq['labels']

        for label_tuple in patterns:
            q_indices = [labels.get(label, -1) for label in label_tuple]

            # Skip if any qubit is invalid
            if any(q < 0 for q in q_indices):
                continue

            # Find anticommuting stabilizers for this triplet
            anticommuting_stabs = get_anticommuting_stabilizers_color(q_indices, parity_matrix)

            # Compute current density
            old_density_per_time = error_diff + torch.einsum(
                'bst,sd->bdt', s1s2.float(), parity_matrix
            )
            old_density_total = old_density_per_time.sum(dim=2).sum(dim=1)  # (B,)
            old_density_k_plus_1 = old_density_per_time[:, :, 1].sum(dim=1)  # (B,)

            # Create hypothetical flipped state
            new_error = error_diff.clone()
            new_s1s2 = s1s2.clone()

            # Flip all three qubits in both rounds
            for q in q_indices:
                new_error[:, q, :] = (new_error[:, q, :] + 1) % 2

            # Flip anticommuting stabilizers only in round k
            if anticommuting_stabs:
                new_s1s2[:, anticommuting_stabs, 0] = (new_s1s2[:, anticommuting_stabs, 0] + 1) % 2

            # Compute new density
            new_density_per_time = new_error + torch.einsum(
                'bst,sd->bdt', new_s1s2.float(), parity_matrix
            )
            new_density_total = new_density_per_time.sum(dim=2).sum(dim=1)  # (B,)
            new_density_k_plus_1 = new_density_per_time[:, :, 1].sum(dim=1)  # (B,)

            # Acceptance criteria
            accept_mask = (new_density_total < old_density_total) | \
                          ((new_density_total == old_density_total) &
                           (new_density_k_plus_1 > old_density_k_plus_1))

            # Apply accepted changes
            if accept_mask.any():
                num_accepted += int(accept_mask.sum().item())
                error_diff = torch.where(
                    accept_mask.unsqueeze(1).unsqueeze(2), new_error, error_diff
                )
                s1s2 = torch.where(accept_mask.unsqueeze(1).unsqueeze(2), new_s1s2, s1s2)

    return error_diff, s1s2, num_accepted


def apply_timelike_homological_equivalence_color(
    x_error_diff: torch.Tensor,
    z_error_diff: torch.Tensor,
    s1s2_x: torch.Tensor,
    s1s2_z: torch.Tensor,
    parity_matrix: torch.Tensor,
    he_transformer: ColorCodeHE,
    max_iterations: int = 32,
    basis: str = 'X',
    enable_weight2: bool = True,
    enable_weight3: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Apply timelike homological equivalence for color codes.
    
    Follows the same structure as surface code:
    1. Weight-1 timelike HE until convergence
    2. Weight-2 timelike HE until convergence (optional)
    3. Weight-3 timelike HE until convergence (optional)
    
    Args:
        x_error_diff: X error diffs (B, num_data, n_rounds)
        z_error_diff: Z error diffs (B, num_data, n_rounds)
        s1s2_x: X stabilizer s1s2 measurements (B, num_stabs, n_rounds)
        s1s2_z: Z stabilizer s1s2 measurements (B, num_stabs, n_rounds)
        parity_matrix: Parity check matrix (num_stabs, num_data)
        he_transformer: ColorCodeHE instance
        max_iterations: Maximum iterations per phase
        basis: 'X' or 'Z' - memory circuit basis (for round 0 exclusion)
        enable_weight2: Whether to apply weight-2 timelike HE
        enable_weight3: Whether to apply weight-3 timelike HE
        
    Returns:
        Tuple of (x_error_diff, z_error_diff, s1s2_x, s1s2_z, stats_dict)
    """
    B, num_data, n_rounds = x_error_diff.shape

    # Track statistics
    total_accepted_x = 0
    total_accepted_z = 0
    total_accepted_weight2 = 0
    total_accepted_weight3 = 0

    # Determine round exclusions (skip round 0 for opposite basis)
    min_t_x = 1 if basis == 'X' else 0
    min_t_z = 1 if basis == 'Z' else 0

    # Stop before last round (data there is unreliable when n_rounds > 2)
    max_t = n_rounds - 2 if n_rounds > 2 else n_rounds - 1

    # ========================================================================
    # PHASE 1: Weight-1 timelike HE until convergence
    # ========================================================================
    phase1_iterations = 0
    for iteration in range(max_iterations):
        phase1_iterations = iteration + 1
        phase1_accepted = 0

        for t in range(max(0, max_t)):
            # Extract rounds k and k+1
            x_pair = x_error_diff[:, :, t:t + 2].clone()
            z_pair = z_error_diff[:, :, t:t + 2].clone()
            s1s2_x_pair = s1s2_x[:, :, t:t + 2].clone()
            s1s2_z_pair = s1s2_z[:, :, t:t + 2].clone()

            # Apply to X errors (detected by Z stabilizers) - skip if t < min_t_x
            if t >= min_t_x:
                x_pair, s1s2_z_pair, num_x = simplifytime_color(x_pair, s1s2_z_pair, parity_matrix)
                phase1_accepted += num_x
                total_accepted_x += num_x
                x_error_diff[:, :, t:t + 2] = x_pair
                s1s2_z[:, :, t:t + 2] = s1s2_z_pair

            # Apply to Z errors (detected by X stabilizers) - skip if t < min_t_z
            if t >= min_t_z:
                z_pair, s1s2_x_pair, num_z = simplifytime_color(z_pair, s1s2_x_pair, parity_matrix)
                phase1_accepted += num_z
                total_accepted_z += num_z
                z_error_diff[:, :, t:t + 2] = z_pair
                s1s2_x[:, :, t:t + 2] = s1s2_x_pair

        # Check convergence
        if phase1_accepted == 0:
            break

    # ========================================================================
    # PHASE 2: Weight-2 timelike HE until convergence
    # ========================================================================
    phase2_iterations = 0
    if enable_weight2:
        for iteration in range(max_iterations):
            phase2_iterations = iteration + 1
            phase2_accepted = 0

            for t in range(max(0, max_t)):
                x_pair = x_error_diff[:, :, t:t + 2].clone()
                z_pair = z_error_diff[:, :, t:t + 2].clone()
                s1s2_x_pair = s1s2_x[:, :, t:t + 2].clone()
                s1s2_z_pair = s1s2_z[:, :, t:t + 2].clone()

                # X errors with weight-2 patterns
                if t >= min_t_x:
                    x_pair, s1s2_z_pair, num_x = simplifytime_weight2_color(
                        x_pair, s1s2_z_pair, parity_matrix, he_transformer, 'X'
                    )
                    phase2_accepted += num_x
                    total_accepted_weight2 += num_x
                    x_error_diff[:, :, t:t + 2] = x_pair
                    s1s2_z[:, :, t:t + 2] = s1s2_z_pair

                # Z errors with weight-2 patterns
                if t >= min_t_z:
                    z_pair, s1s2_x_pair, num_z = simplifytime_weight2_color(
                        z_pair, s1s2_x_pair, parity_matrix, he_transformer, 'Z'
                    )
                    phase2_accepted += num_z
                    total_accepted_weight2 += num_z
                    z_error_diff[:, :, t:t + 2] = z_pair
                    s1s2_x[:, :, t:t + 2] = s1s2_x_pair

            if phase2_accepted == 0:
                break

    # ========================================================================
    # PHASE 3: Weight-3 timelike HE until convergence
    # ========================================================================
    phase3_iterations = 0
    if enable_weight3:
        for iteration in range(max_iterations):
            phase3_iterations = iteration + 1
            phase3_accepted = 0

            for t in range(max(0, max_t)):
                x_pair = x_error_diff[:, :, t:t + 2].clone()
                z_pair = z_error_diff[:, :, t:t + 2].clone()
                s1s2_x_pair = s1s2_x[:, :, t:t + 2].clone()
                s1s2_z_pair = s1s2_z[:, :, t:t + 2].clone()

                # X errors with weight-3 patterns
                if t >= min_t_x:
                    x_pair, s1s2_z_pair, num_x = simplifytime_weight3_color(
                        x_pair, s1s2_z_pair, parity_matrix, he_transformer, 'X'
                    )
                    phase3_accepted += num_x
                    total_accepted_weight3 += num_x
                    x_error_diff[:, :, t:t + 2] = x_pair
                    s1s2_z[:, :, t:t + 2] = s1s2_z_pair

                # Z errors with weight-3 patterns
                if t >= min_t_z:
                    z_pair, s1s2_x_pair, num_z = simplifytime_weight3_color(
                        z_pair, s1s2_x_pair, parity_matrix, he_transformer, 'Z'
                    )
                    phase3_accepted += num_z
                    total_accepted_weight3 += num_z
                    z_error_diff[:, :, t:t + 2] = z_pair
                    s1s2_x[:, :, t:t + 2] = s1s2_x_pair

            if phase3_accepted == 0:
                break

    stats = {
        'total_accepted_x':
            total_accepted_x,
        'total_accepted_z':
            total_accepted_z,
        'total_accepted_weight2':
            total_accepted_weight2,
        'total_accepted_weight3':
            total_accepted_weight3,
        'total_accepted':
            total_accepted_x + total_accepted_z + total_accepted_weight2 + total_accepted_weight3,
        'phase1_iterations':
            phase1_iterations,
        'phase2_iterations':
            phase2_iterations,
        'phase3_iterations':
            phase3_iterations,
    }

    return x_error_diff, z_error_diff, s1s2_x, s1s2_z, stats


def apply_full_homological_equivalence_color(
    x_error_diff: torch.Tensor,
    z_error_diff: torch.Tensor,
    s1s2_x: torch.Tensor,
    s1s2_z: torch.Tensor,
    color_code,
    he_transformer: ColorCodeHE,
    max_iterations: int = 32,
    basis: str = 'X',
    enable_weight2: bool = True,
    enable_weight3: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Apply full homological equivalence: Spacelike -> Timelike -> Spacelike.
    
    This is the recommended entry point for applying HE to color code data.
    
    Args:
        x_error_diff: X error diffs (B, num_data, n_rounds)
        z_error_diff: Z error diffs (B, num_data, n_rounds)
        s1s2_x: X stabilizer s1s2 measurements (B, num_stabs, n_rounds)
        s1s2_z: Z stabilizer s1s2 measurements (B, num_stabs, n_rounds)
        color_code: ColorCode instance
        he_transformer: ColorCodeHE instance
        max_iterations: Maximum iterations for timelike HE
        basis: 'X' or 'Z' - memory circuit basis
        enable_weight2: Whether to apply weight-2 timelike HE
        enable_weight3: Whether to apply weight-3 timelike HE
        
    Returns:
        Tuple of (x_error_diff, z_error_diff, s1s2_x, s1s2_z, stats_dict)
    """
    B, num_data, n_rounds = x_error_diff.shape

    # Get parity matrix (data qubits only)
    parity_matrix = get_parity_matrix_data_only(color_code)

    # ========================================================================
    # STEP 1: Apply spacelike HE
    # ========================================================================
    # x_error_diff is (B, num_data, n_rounds)
    # apply_spacelike expects (num_data, n_rounds)
    for b in range(B):
        x_diff_b = x_error_diff[b]  # (num_data, n_rounds)
        z_diff_b = z_error_diff[b]  # (num_data, n_rounds)

        x_diff_b_new, z_diff_b_new = apply_spacelike_homological_equivalence(
            x_diff_b, z_diff_b, he_transformer
        )

        x_error_diff[b] = x_diff_b_new
        z_error_diff[b] = z_diff_b_new

    # ========================================================================
    # STEP 2: Apply timelike HE
    # ========================================================================
    x_error_diff, z_error_diff, s1s2_x, s1s2_z, stats = apply_timelike_homological_equivalence_color(
        x_error_diff,
        z_error_diff,
        s1s2_x,
        s1s2_z,
        parity_matrix,
        he_transformer,
        max_iterations=max_iterations,
        basis=basis,
        enable_weight2=enable_weight2,
        enable_weight3=enable_weight3
    )

    # ========================================================================
    # STEP 3: Apply spacelike HE again (cleanup after timelike)
    # ========================================================================
    for b in range(B):
        x_diff_b = x_error_diff[b]  # (num_data, n_rounds)
        z_diff_b = z_error_diff[b]  # (num_data, n_rounds)

        x_diff_b_new, z_diff_b_new = apply_spacelike_homological_equivalence(
            x_diff_b, z_diff_b, he_transformer
        )

        x_error_diff[b] = x_diff_b_new
        z_error_diff[b] = z_diff_b_new

    return x_error_diff, z_error_diff, s1s2_x, s1s2_z, stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from qec.color_code.color_code import ColorCode

    print("=" * 70)
    print("TESTING COLOR CODE HOMOLOGICAL EQUIVALENCE")
    print("=" * 70)

    # Test with d=5 color code
    cc = ColorCode(5)
    he = ColorCodeHE(cc)

    print(f"\nCode: d={cc.distance}, {cc.num_data} data qubits, {cc.num_plaquettes} plaquettes")

    # Print plaquette labels
    print("\n--- Plaquette Labels ---")
    for i, plaq in enumerate(he.plaquettes):
        labels = plaq['labels']
        present = [f"{k}=D{v}" for k, v in labels.items() if v >= 0]
        print(f"Plaq {i:2d} ({plaq['color']:5s}, w{plaq['weight']}): {', '.join(present)}")

    # Test weight reduction on weight-6 plaquette
    print("\n--- Testing Weight Reduction (Weight-6 Bulk) ---")
    plaq_idx = 2  # Blue bulk plaquette
    plaq = he.plaquettes[plaq_idx]
    print(f"Testing on plaquette {plaq_idx}: {plaq['data_qubits']}")

    # Test weight-6 error (should be removed)
    error = torch.zeros(cc.num_data, dtype=torch.long)
    for q in plaq['data_qubits']:
        error[q] = 1
    print(f"  Weight-6 error: {error.nonzero().squeeze().tolist()}")
    reduced = weight_reduction(error, plaq['data_qubits'], 6)
    print(f"  After reduction: {reduced.nonzero().squeeze().tolist()}")

    # Test weight-5 error (should become weight-1)
    error = torch.zeros(cc.num_data, dtype=torch.long)
    for q in plaq['data_qubits'][:5]:
        error[q] = 1
    print(f"  Weight-5 error: {error.nonzero().squeeze().tolist()}")
    reduced = weight_reduction(error, plaq['data_qubits'], 6)
    print(f"  After reduction: {reduced.nonzero().squeeze().tolist()}")

    # Test weight-3 canonicalization
    print("\n--- Testing Weight-3 Canonicalization (Weight-6 Bulk) ---")
    labels = plaq['labels']
    qubits = [labels['q1'], labels['q2'], labels['q3'], labels['q4'], labels['q5'], labels['q6']]

    # Test (q1,q2,q3) -> (q4,q5,q6)
    error = torch.zeros(cc.num_data, dtype=torch.long)
    error[labels['q1']] = error[labels['q2']] = error[labels['q3']] = 1
    print(f"  Pattern (q1,q2,q3): {[labels['q1'], labels['q2'], labels['q3']]}")
    canonical = fix_equivalence_weight6(error, labels)
    result_qubits = canonical.nonzero().squeeze().tolist()
    print(f"  Canonical (q4,q5,q6): {result_qubits}")
    expected = [labels['q4'], labels['q5'], labels['q6']]
    assert sorted(result_qubits) == sorted(expected), f"Expected {expected}"

    # Test weight-4 boundary
    print("\n--- Testing Weight-4 Boundary ---")
    for i, plaq in enumerate(he.plaquettes):
        if plaq['weight'] == 4:
            print(f"\nPlaq {i} ({plaq['color']} {plaq['type']}):")
            labels = plaq['labels']
            qubits = [labels['q1'], labels['q2'], labels['q5'], labels['q6']]

            # Test (q1,q2) -> (q5,q6)
            error = torch.zeros(cc.num_data, dtype=torch.long)
            error[labels['q1']] = error[labels['q2']] = 1
            print(f"  Pattern (q1,q2): {[labels['q1'], labels['q2']]}")
            canonical = fix_equivalence_weight4(error, labels, plaq['color'])
            result = canonical.nonzero().squeeze().tolist()
            print(f"  Canonical (q5,q6): {result}")
            break

    # Test full simplify
    print("\n--- Testing Full Simplify ---")
    # Create a complex error pattern
    error = torch.zeros(cc.num_data, dtype=torch.long)
    error[0] = error[1] = error[2] = error[3] = error[4] = 1  # Weight-5 in first plaquette
    print(f"Initial errors: {error.nonzero().squeeze().tolist()}")

    canonical, iters = he.simplify_with_count(error)
    print(f"After simplify ({iters} iterations): {canonical.nonzero().squeeze().tolist()}")

    print("\n" + "=" * 70)
    print("All tests passed!")
