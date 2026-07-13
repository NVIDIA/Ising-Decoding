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
Triangular Color Code with rectangular grid embedding for CNN pre-decoder.

The data qubits are embedded in a (n_rows x n_cols) rectangular grid where:
- n_rows = d + (d-1)//2
- n_cols = d

Coordinate system:
- Top qubit (qubit 0) is always at (row=0, col=0)
- Row index decreases (more negative) going down the triangle
- Column is centered around 0, expanding symmetrically

Syndrome-to-grid mapping:
- For all stabilizers EXCEPT right boundary: map to top-right data qubit
- For right boundary (red weight-4): map to top-left data qubit

Reference plaquettes for verification:

d=3 (7 data qubits, 3 plaquettes, 4x3 grid):
  [0,1,2,3]: green (boundary)
  [2,3,4,5]: blue (boundary)
  [1,3,5,6]: red (boundary)

d=5 (19 data qubits, 9 plaquettes, 7x5 grid):
  [0,1,2,3]: green (boundary)
  [2,3,4,5,7,8]: blue (bulk)
  [1,3,5,6]: red (boundary)
  [5,6,8,9,12,13]: green (bulk)
  [7,8,11,12,15,16]: red (bulk)
  [4,7,10,11]: green (boundary)
  [10,11,14,15]: blue (boundary)
  [12,13,16,17]: blue (boundary)
  [9,13,17,18]: red (boundary)

d=7 (37 data qubits, 18 plaquettes, 10x7 grid):
  [0,1,2,3]: green (boundary)
  [2,3,4,5,7,8]: blue (bulk)
  [1,3,5,6]: red (boundary)
  [5,6,8,9,12,13]: green (bulk)
  [7,8,11,12,15,16]: red (bulk)
  [4,7,10,11]: green (boundary)
  [10,11,14,15,19,20]: blue (bulk)
  [12,13,16,17,21,22]: blue (bulk)
  [9,13,17,18]: red (boundary)
  [14,19,24,25]: green (boundary)
  [15,16,20,21,26,27]: green (bulk)
  [17,18,22,23,28,29]: green (bulk)
  [19,20,25,26,31,32]: red (bulk)
  [21,22,27,28,33,34]: red (bulk)
  [23,29,35,36]: red (boundary)
  [24,25,30,31]: blue (boundary)
  [26,27,32,33]: blue (boundary)
  [28,29,34,35]: blue (boundary)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ColorCode:
    """
    Triangular color code with rectangular grid embedding.
    
    Qubit numbering (for distance d):
    - Data qubits: [0, num_data)
    - X-check ancillas: [num_data, num_data + num_plaquettes)
    - Z-check ancillas: [num_data + num_plaquettes, num_data + 2*num_plaquettes)
    
    Data qubits are numbered top-to-bottom, left-to-right.
    Ancillas are numbered by their mapped grid position (top-to-bottom, left-to-right).
    
    Args:
        distance: Code distance (odd integer >= 3)
    
    Attributes:
        data_qubits: Array of data qubit indices [0, num_data)
        xcheck_qubits: Array of X-check ancilla indices
        zcheck_qubits: Array of Z-check ancilla indices
        stab_to_data_idx: Maps stabilizer index to data qubit grid position
    """

    def __init__(self, distance: int):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and >= 3")

        self.distance = distance
        self.n_rows = distance + (distance - 1) // 2
        self.n_cols = distance
        self.num_data = (3 * distance * distance + 1) // 4
        self.num_plaquettes = (3 * (distance * distance - 1)) // 8

        # Generate data qubit grid layout
        self._generate_data_qubit_grid()

        # Generate plaquettes (stabilizers) algorithmically
        self._generate_plaquettes()

        # Compute syndrome-to-data mapping and sort plaquettes by grid position
        self._compute_syndrome_mapping_and_sort()

        # Create qubit index arrays (after sorting)
        self.data_qubits = np.arange(self.num_data)
        self.xcheck_qubits = np.arange(self.num_data, self.num_data + self.num_plaquettes)
        self.zcheck_qubits = np.arange(
            self.num_data + self.num_plaquettes, self.num_data + 2 * self.num_plaquettes
        )
        self.all_qubits = np.arange(self.num_data + 2 * self.num_plaquettes)

        # Build parity check matrices
        self._build_parity_matrices()

        # Generate logical operators
        self._generate_logical_operators()

    def _get_row_width(self, row_idx: int) -> int:
        """Get number of qubits in a given row (0-indexed from top)."""
        group = row_idx // 3
        pos = row_idx % 3
        return 2 * group + 1 if pos < 2 else 2 * group + 2

    def _get_row_start_col(self, width: int) -> int:
        """Get starting column for a row of given width (centered around 0)."""
        return -(width // 2)

    def _generate_data_qubit_grid(self):
        """Generate data qubit positions on rectangular grid."""
        self.qubit_to_coord = {}  # qubit_id -> (row, col)
        self.coord_to_qubit = {}  # (row, col) -> qubit_id
        self.grid_to_qubit = {}  # (grid_row, grid_col) -> qubit_id (0-indexed grid)
        self.qubit_to_grid = {}  # qubit_id -> (grid_row, grid_col)

        qubit_id = 0
        for row_idx in range(self.n_rows):
            width = self._get_row_width(row_idx)
            start_col = self._get_row_start_col(width)
            row = -row_idx  # User's coordinate: row 0 at top, negative going down

            for i in range(width):
                col = start_col + i

                self.qubit_to_coord[qubit_id] = (row, col)
                self.coord_to_qubit[(row, col)] = qubit_id

                # Also store 0-indexed grid position for CNN
                grid_row = row_idx
                grid_col = col + (self.n_cols // 2)
                self.grid_to_qubit[(grid_row, grid_col)] = qubit_id
                self.qubit_to_grid[qubit_id] = (grid_row, grid_col)

                qubit_id += 1

        assert qubit_id == self.num_data, f"Expected {self.num_data} qubits, got {qubit_id}"

    def _try_pattern(self, row: int, col: int, pattern: List[Tuple[int,
                                                                   int]]) -> Optional[List[int]]:
        """Try to form a plaquette with given pattern from anchor position."""
        qubits = []
        for dr, dc in pattern:
            pos = (row + dr, col + dc)
            if pos in self.coord_to_qubit:
                qubits.append(self.coord_to_qubit[pos])
            else:
                return None
        return sorted(qubits)

    def _generate_plaquettes(self):
        """Generate plaquette connectivity algorithmically."""
        colors = ['green', 'blue', 'red']
        plaquettes = []
        added_plaqs = set()
        used_bulk = {c: set() for c in colors}

        # Plaquette patterns (relative to anchor position)
        pattern_w6 = [(0, 0), (0, 1), (-1, 0), (-1, 1), (-2, 0), (-2, 1)]  # 3x2 bulk
        pattern_top = [(0, 0), (-1, 0), (-2, -1), (-2, 0)]  # top cap
        pattern_left = [(-2, 0), (-2, 1), (-1, 1), (0, 1)]  # left boundary
        pattern_right = [(-2, 0), (-2, 1), (-1, 0), (0, 0)]  # right boundary
        pattern_bottom = [(-1, 0), (-1, 1), (0, 0), (0, 1)]  # bottom 2x2

        def add_plaq(qubits, color, ptype, check_bulk_overlap=True):
            key = tuple(sorted(qubits))
            if key in added_plaqs:
                return False
            if check_bulk_overlap and ptype == 'bulk':
                if any(q in used_bulk[color] for q in qubits):
                    return False
                used_bulk[color].update(qubits)
            plaquettes.append((qubits, color, ptype))
            added_plaqs.add(key)
            return True

        # 1. Top plaquette (green) - always qubits 0,1,2,3
        add_plaq([0, 1, 2, 3], 'green', 'boundary', False)

        # 2. Weight-6 bulk plaquettes
        color_occurrence = {'green': 0, 'blue': 0, 'red': 0}

        for row_idx in range(2, self.n_rows - 2):
            row = -row_idx
            color = colors[row % 3]
            occ = color_occurrence[color]
            color_occurrence[color] += 1

            # Compute valid column positions for this color
            if color == 'green':
                # Green: alternates between center+even offsets and odd offsets
                if occ == 0:
                    valid_cols = [0]
                elif occ % 2 == 1:  # Odd occurrence: use odd offsets
                    valid_cols = sorted([c for c in range(-occ, occ + 1) if c % 2 != 0])
                else:  # Even occurrence: use even offsets (including 0)
                    valid_cols = sorted([c for c in range(-occ, occ + 1) if c % 2 == 0])
            else:
                # Blue/Red: start from -(occ+1), step by 2
                start = -(occ + 1)
                num_plaqs = occ + 1
                valid_cols = [start + 2 * i for i in range(num_plaqs)]

            for col in valid_cols:
                qubits = self._try_pattern(row, col, pattern_w6)
                if qubits:
                    add_plaq(qubits, color, 'bulk', True)

        # 3. Left boundary (green) - every 3 rows starting from row_idx=3
        for row_idx in range(3, self.n_rows - 2, 3):
            row = -row_idx
            width = self._get_row_width(row_idx)
            anchor_col = self._get_row_start_col(width) - 1
            qubits = self._try_pattern(row, anchor_col, pattern_left)
            if qubits:
                add_plaq(qubits, 'green', 'boundary', False)

        # 4. Right boundary (red) - every 3 rows starting from row_idx=1
        for row_idx in range(1, self.n_rows - 2, 3):
            row = -row_idx
            width = self._get_row_width(row_idx)
            anchor_col = self._get_row_start_col(width) + width - 1
            qubits = self._try_pattern(row, anchor_col, pattern_right)
            if qubits:
                add_plaq(qubits, 'red', 'boundary', False)

        # 5. Bottom boundary (blue) - 2x2 blocks on second-to-last row, step by 2
        second_last_row_idx = self.n_rows - 2
        row = -second_last_row_idx
        width = self._get_row_width(second_last_row_idx)
        start_col = self._get_row_start_col(width)
        for col in range(start_col, start_col + width - 1, 2):
            qubits = self._try_pattern(row, col, pattern_bottom)
            if qubits:
                add_plaq(qubits, 'blue', 'boundary', False)

        # Store raw plaquettes temporarily (will be sorted later)
        self._raw_plaquettes = plaquettes

        assert len(plaquettes) == self.num_plaquettes, \
            f"Expected {self.num_plaquettes} plaquettes, got {len(plaquettes)}"

    def _get_mapped_data_qubit(self, data_qubits: List[int], color: str, ptype: str) -> int:
        """
        Get the data qubit that a plaquette's syndrome maps to.
        
        Rules:
        - Right boundary (red weight-4): top-left data qubit
        - All others: top-right data qubit
        """
        # Get coordinates for all data qubits in plaquette
        coords = [(q, self.qubit_to_coord[q]) for q in data_qubits]

        # Find top row (highest, i.e., least negative)
        top_row = max(c[0] for _, c in coords)
        top_qubits = [(q, c) for q, c in coords if c[0] == top_row]

        # Determine if this is right boundary (red weight-4)
        is_right_boundary = (color == 'red' and ptype == 'boundary')

        if is_right_boundary:
            # Top-left: minimum column
            return min(top_qubits, key=lambda x: x[1][1])[0]
        else:
            # Top-right: maximum column
            return max(top_qubits, key=lambda x: x[1][1])[0]

    def _compute_syndrome_mapping_and_sort(self):
        """Compute syndrome-to-data mapping and sort plaquettes by grid position."""
        # Compute mapped data qubit for each plaquette
        plaq_with_mapping = []
        for qubits, color, ptype in self._raw_plaquettes:
            mapped_qubit = self._get_mapped_data_qubit(qubits, color, ptype)
            grid_pos = self.qubit_to_grid[mapped_qubit]
            plaq_with_mapping.append(
                {
                    'data_qubits': qubits,
                    'color': color,
                    'type': ptype,
                    'weight': len(qubits),
                    'mapped_qubit': mapped_qubit,
                    'grid_pos': grid_pos,
                }
            )

        # Sort by grid position: top-to-bottom (row), left-to-right (col)
        plaq_with_mapping.sort(key=lambda p: (p['grid_pos'][0], p['grid_pos'][1]))

        # Assign ancilla IDs based on sorted order
        # X-ancilla at index i: num_data + i
        # Z-ancilla at index i: num_data + num_plaquettes + i
        self.plaquettes = []
        self.stab_to_data_idx = np.zeros(self.num_plaquettes, dtype=np.int32)

        for plaq_idx, plaq in enumerate(plaq_with_mapping):
            x_ancilla_id = self.num_data + plaq_idx
            z_ancilla_id = self.num_data + self.num_plaquettes + plaq_idx

            self.plaquettes.append(
                {
                    'x_ancilla': x_ancilla_id,
                    'z_ancilla': z_ancilla_id,
                    'data_qubits': plaq['data_qubits'],
                    'weight': plaq['weight'],
                    'type': plaq['type'],
                    'color': plaq['color'],
                    'mapped_qubit': plaq['mapped_qubit'],
                    'grid_pos': plaq['grid_pos'],
                }
            )

            self.stab_to_data_idx[plaq_idx] = plaq['mapped_qubit']

        # Clean up temporary storage
        del self._raw_plaquettes

    def _build_parity_matrices(self):
        """Build Hx and Hz parity check matrices."""
        num_total = len(self.all_qubits)
        self.hx = np.zeros((self.num_plaquettes, num_total))
        self.hz = np.zeros((self.num_plaquettes, num_total))

        for i, plaq in enumerate(self.plaquettes):
            for data_qubit in plaq['data_qubits']:
                self.hx[i, data_qubit] = 1
                self.hz[i, data_qubit] = 1

    def _generate_logical_operators(self):
        """Generate logical X and Z operators.
        
        For the triangular color code, the minimal logical operator is the
        bottom edge (blue boundary), which has exactly d qubits.
        This is preferred over using all data qubits because:
        - Measurement errors scale as O(d) instead of O(d²)
        - No need for boundary detectors in memory experiments
        """
        num_total = len(self.all_qubits)
        self.lx = np.zeros((1, num_total))
        self.lz = np.zeros((1, num_total))

        # Find bottom edge qubits (the row with minimum row coordinate)
        bottom_row = min(self.qubit_to_coord[q][0] for q in range(self.num_data))
        self.logical_qubits = sorted(
            [q for q in range(self.num_data) if self.qubit_to_coord[q][0] == bottom_row]
        )

        # Set logical operators on bottom edge
        for qid in self.logical_qubits:
            self.lx[0, qid] = 1
            self.lz[0, qid] = 1

    def get_grid_array(self) -> np.ndarray:
        """Return 2D array of qubit IDs on the grid (-1 for padding)."""
        grid = np.full((self.n_rows, self.n_cols), -1, dtype=np.int32)
        for qid, (grid_row, grid_col) in self.qubit_to_grid.items():
            grid[grid_row, grid_col] = qid
        return grid

    def get_syndrome_grid_indices(self) -> np.ndarray:
        """
        Return array mapping stabilizer index to flat grid index.
        
        For use with reshape_stabilizers_to_grid functions.
        Returns array of shape (num_plaquettes,) where each entry is the
        flat index (row * n_cols + col) into the n_rows x n_cols grid.
        """
        indices = np.zeros(self.num_plaquettes, dtype=np.int32)
        for i, plaq in enumerate(self.plaquettes):
            grid_row, grid_col = plaq['grid_pos']
            indices[i] = grid_row * self.n_cols + grid_col
        return indices

    def print_structure(self):
        """Print code structure summary."""
        print(f"Triangular Color Code - Distance {self.distance}")
        print(f"  Grid size: {self.n_rows} x {self.n_cols}")
        print(f"  Data qubits: {self.num_data} (IDs: 0-{self.num_data-1})")
        print(
            f"  X-check ancillas: {self.num_plaquettes} (IDs: {self.xcheck_qubits[0]}-{self.xcheck_qubits[-1]})"
        )
        print(
            f"  Z-check ancillas: {self.num_plaquettes} (IDs: {self.zcheck_qubits[0]}-{self.zcheck_qubits[-1]})"
        )
        print(f"  Total qubits: {len(self.all_qubits)}")
        print(f"  Plaquettes: {len(self.plaquettes)}")
        print()

        # Print grid layout
        print("Data qubit grid (user coordinates):")
        for row_idx in range(self.n_rows):
            row = -row_idx
            width = self._get_row_width(row_idx)
            start_col = self._get_row_start_col(width)

            qubits_str = []
            for i in range(width):
                col = start_col + i
                if (row, col) in self.coord_to_qubit:
                    qid = self.coord_to_qubit[(row, col)]
                    qubits_str.append(f"D{qid:2d}")

            indent = "  " * (self.n_cols // 2 - (-start_col))
            print(f"  row {row:3d}: {indent}{' '.join(qubits_str)}")
        print()

        # Print CNN grid
        print("CNN grid layout (0-indexed, -1 = padding):")
        grid = self.get_grid_array()
        print("         " + "  ".join(f"c{c}" for c in range(self.n_cols)))
        for r in range(self.n_rows):
            row_str = " ".join(
                f"{grid[r,c]:3d}" if grid[r, c] >= 0 else "  ." for c in range(self.n_cols)
            )
            print(f"  row {r}: {row_str}")
        print()

        # Print plaquettes with syndrome mapping
        print("Plaquettes (sorted by grid position, top-to-bottom, left-to-right):")
        boundary_count = bulk_count = 0
        for i, plaq in enumerate(self.plaquettes):
            if plaq['type'] == 'boundary':
                boundary_count += 1
            else:
                bulk_count += 1
            grid_pos = plaq['grid_pos']
            print(
                f"  Plaq {i:2d} (X:{plaq['x_ancilla']:2d}, Z:{plaq['z_ancilla']:2d}, {plaq['color']:5s}, "
                f"{plaq['type']:8s}, w{plaq['weight']}): "
                f"{plaq['data_qubits']} -> D{plaq['mapped_qubit']} @ grid({grid_pos[0]},{grid_pos[1]})"
            )
        print(
            f"\nSummary: {boundary_count} boundary + {bulk_count} bulk = {len(self.plaquettes)} plaquettes"
        )
        print()

    # ----------------------------------------------------------------------------------
    # Circuit-only physical layout (rectangular grid with blanks, including ancillas)
    # ----------------------------------------------------------------------------------
    def get_circuit_physical_layout(self,
                                    *,
                                    id_order: str = "rtl",
                                    flip_rows: bool = False) -> Dict[int, Tuple[int, int]]:
        """
        Return a *circuit-only* physical layout mapping qubit_id -> (r, c) on a rectangular grid.

        This layout is meant ONLY for reasoning about 2D nearest-neighbor connectivity constraints during
        circuit construction. It does not affect the existing coordinate systems used elsewhere:
          - `qubit_to_coord` / `coord_to_qubit` (triangular user coords)
          - `qubit_to_grid` / `grid_to_qubit` (CNN rectangular embedding)

        Layout rules (as provided by user, generalized for odd distance d):
          - Number of rows equals `self.n_rows = d + (d-1)//2`.
          - Row lengths follow: 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, ... (increments 2,1,1 repeating).
          - Within each row, tokens alternate between data sites (D) and ancilla pairs (X,Z), and the X/Z
            ancillas are adjacent horizontally with Z immediately to the left of X (left-to-right: Z then X).
          - Row start offsets create a triangular envelope; the resulting rectangular width is (2*d - 1).

        Mapping convention:
          - Data sites are assigned ids 0..num_data-1 using a deterministic per-row order controlled by `id_order`.
          - Ancillas are assigned ids in two global blocks (all X first, then all Z), and within each block we also
            use `id_order` per row.
          - NOTE: Even though physically we place Z before X in each row, we still assign X ids first globally.

        Args:
          id_order:
            - "rtl": assign ids within each row from right-to-left. This matches the legacy triangle's upside-down
              convention and makes the legacy schedule nearest-neighbor on this grid for odd distances tested.
            - "ltr": assign ids within each row from left-to-right (natural scan order).
          flip_rows:
            - If True, return coordinates after reflecting the row index: (r, c) -> (n_rows - 1 - r, c).
              This is useful when you want the same schedule embedded onto an up-facing vs down-facing triangle.
        """
        d = int(self.distance)
        width = 2 * d - 1
        n_rows = int(self.n_rows)
        if id_order not in ("rtl", "ltr"):
            raise ValueError("id_order must be 'rtl' or 'ltr'")

        # Row lengths: 1, 3, 4, 5, 7, 8, 9, 11, ...
        def row_len(row_idx: int) -> int:
            if row_idx == 0:
                return 1
            g = row_idx // 3
            pos = row_idx % 3
            # pos 0: +1 from previous end of triple => 2g+5 for g>=1 (works out from pattern)
            # easiest: build from recurrence: L0=1; delta pattern [2,1,1] repeating.
            # but closed form below matches the observed sequence:
            if pos == 0:
                return 2 * g + 5
            if pos == 1:
                return 2 * g + 3
            # pos == 2
            return 2 * g + 4

        # More robust: generate lengths by recurrence to avoid mistakes.
        lengths = [1]
        deltas = [2, 1, 1]
        for i in range(1, n_rows):
            lengths.append(lengths[-1] + deltas[(i - 1) % 3])

        # Row start offsets (column indices in [-d+1, d-1])
        starts = [0]
        for i in range(1, n_rows):
            # first delta=2 and first delta=1 both shift start left by 1; second delta=1 keeps start
            starts.append(starts[-1] + (-1 if (i - 1) % 3 in (0, 1) else 0))

        # Token generators per row type (Z then X)
        def tokens_for_row(row_idx: int, L: int) -> List[str]:
            if row_idx == 0:
                return ["D"]
            pos = row_idx % 3
            out: List[str] = []
            if pos == 0:
                # start with "DD"
                out.extend(["D", "D"])
            elif pos == 1:
                # start with "D"
                out.append("D")
            else:  # pos == 2
                # start with "ZX"
                out.extend(["Z", "X"])

            # Alternate between ZX and DD blocks.
            # Determine next block type based on what we ended with.
            next_block = "ZX" if (len(out) > 0 and out[-1] == "D") else "DD"
            while len(out) < L:
                if next_block == "ZX":
                    out.extend(["Z", "X"])
                    next_block = "DD"
                else:
                    out.extend(["D", "D"])
                    next_block = "ZX"
            return out[:L]

        # Collect token coordinates; assign ids afterward using `id_order`.
        layout: Dict[int, Tuple[int, int]] = {}
        d_positions: List[Tuple[int, int]] = []
        x_positions: List[Tuple[int, int]] = []
        z_positions: List[Tuple[int, int]] = []

        for r in range(n_rows):
            L = lengths[r]
            start = starts[r]
            toks = tokens_for_row(r, L)
            for j, tok in enumerate(toks):
                col = start + j
                # shift to [0..width-1]
                c = col + (d - 1)
                if not (0 <= c < width):
                    raise AssertionError(
                        f"Physical layout column out of range: row={r} col={col} width={width}"
                    )
                if tok == "D":
                    d_positions.append((r, c))
                elif tok == "X":
                    x_positions.append((r, c))
                elif tok == "Z":
                    z_positions.append((r, c))
                else:
                    raise AssertionError(f"Unknown token {tok}")

        if len(d_positions) != int(self.num_data):
            raise AssertionError(
                f"Expected exactly num_data={self.num_data} data sites in physical layout, got {len(d_positions)}"
            )

        def _row_sort_key(rc: Tuple[int, int]) -> Tuple[int, int]:
            rr, cc = rc
            return (rr, -cc) if id_order == "rtl" else (rr, cc)

        # Assign data ids
        data_id = 0
        for rc in sorted(d_positions, key=_row_sort_key):
            layout[data_id] = rc
            data_id += 1

        # Assign ancilla ids: X first globally, then Z (regardless of left-to-right placement).
        if len(x_positions) != int(self.num_plaquettes
                                  ) or len(z_positions) != int(self.num_plaquettes):
            raise AssertionError(
                f"Expected exactly num_plaquettes={self.num_plaquettes} X and Z ancillas in physical layout, "
                f"got X={len(x_positions)} Z={len(z_positions)}"
            )

        x_id = int(self.num_data)
        for (r, c) in sorted(x_positions, key=_row_sort_key):
            layout[x_id] = (r, c)
            x_id += 1

        z_id = int(self.num_data + self.num_plaquettes)
        for (r, c) in sorted(z_positions, key=_row_sort_key):
            layout[z_id] = (r, c)
            z_id += 1

        if data_id != int(self.num_data
                         ) or x_id != int(self.num_data + self.num_plaquettes
                                         ) or z_id != int(self.num_data + 2 * self.num_plaquettes):
            raise AssertionError(
                "Physical layout did not assign all qubits: "
                f"data={data_id}/{self.num_data} x={x_id - self.num_data}/{self.num_plaquettes} z={z_id - (self.num_data + self.num_plaquettes)}/{self.num_plaquettes}"
            )

        if flip_rows:
            H = n_rows
            return {q: (H - 1 - rc[0], rc[1]) for q, rc in layout.items()}
        return layout

    def superdense_plaquette(self, plaq_idx: int) -> Dict[str, int]:
        """
        Return a canonical labeling for a plaquette for the superdense circuit.

        Labels follow the convention discussed in chat:
        - a1: X-ancilla (prepared in |+>, measured in X)
        - a2: Z-ancilla (prepared in |0>, measured in Z)
        - q1..q6: data qubits ordered by compass position around the plaquette (for weight-6):
            q1 = NW, q2 = W, q3 = SW, q4 = NE, q5 = E, q6 = SE

        Weight-6 plaquettes occupy a 3x2 block in (row, col) coordinates.

        Weight-4 plaquettes are embedded into the same frame by populating only:
          - q1, q2 (feed into a1 via the q*->a1 half)
          - q5, q6 (feed into a2 via the q*->a2 half)
        and setting q3 and q4 to -1 (missing). This matches using the same global 8-step schedule,
        while weight-4 plaquettes naturally skip the third pair steps.
        """
        if plaq_idx < 0 or plaq_idx >= len(self.plaquettes):
            raise IndexError(f"plaq_idx out of range: {plaq_idx}")

        plaq = self.plaquettes[plaq_idx]
        w = int(plaq["weight"])
        if w not in (4, 6):
            raise ValueError(f"Unsupported plaquette weight={w} for plaq_idx={plaq_idx}")

        data = list(plaq["data_qubits"])
        coords = {q: self.qubit_to_coord[q] for q in data}  # (row, col)

        rows = sorted(
            {r for r, _ in coords.values()}, reverse=True
        )  # top (largest) -> bottom (smallest)
        cols = sorted({c for _, c in coords.values()})  # left -> right

        out: Dict[str, int] = {
            "a1": int(plaq["x_ancilla"]),
            "a2": int(plaq["z_ancilla"]),
        }

        coord_to_qid = {v: k for k, v in coords.items()}

        if w == 6:
            if len(rows) != 3 or len(cols) != 2:
                raise ValueError(
                    "Expected weight-6 plaquette data qubits to occupy exactly 3 distinct rows and 2 distinct cols "
                    f"but got rows={rows}, cols={cols} for plaq_idx={plaq_idx}, data_qubits={data}, coords={coords}"
                )

            row_top, row_mid, row_bot = rows
            col_left, col_right = cols

            # West column (left): q1 (NW), q2 (W), q3 (SW)
            # East column (right): q4 (NE), q5 (E), q6 (SE)
            expected_positions = {
                "q1": (row_top, col_left),
                "q2": (row_mid, col_left),
                "q3": (row_bot, col_left),
                "q4": (row_top, col_right),
                "q5": (row_mid, col_right),
                "q6": (row_bot, col_right),
            }

            missing = []
            for label, pos in expected_positions.items():
                qid = coord_to_qid.get(pos)
                if qid is None:
                    missing.append((label, pos))
                else:
                    out[label] = int(qid)

            if missing:
                raise ValueError(
                    f"Could not assign all q1..q6 labels for plaq_idx={plaq_idx}; missing={missing}; "
                    f"data_qubits={data}; coords={coords}"
                )
            return out

        # --- weight-4 embedding into q1/q2/q5/q6; q3/q4 are missing ---
        out["q3"] = -1
        out["q4"] = -1

        # Case A: South boundary 2x2 block (two rows, two cols)
        if len(rows) == 2 and len(cols) == 2:
            row_top, row_bot = rows  # already sorted top->bottom
            col_left, col_right = cols

            # User-confirmed for south boundary:
            # q1 = NW (top-left), q2 = W (bottom-left), q3 = NE (top-right), q4 = E (bottom-right).
            # We embed into the w6 schedule by mapping:
            #   q1 -> q1, q2 -> q2, q3 -> q5, q4 -> q6  (and q3/q4 are the unused w6 labels).
            pos_q1 = (row_top, col_left)
            pos_q2 = (row_bot, col_left)
            pos_q5 = (row_top, col_right)  # NE
            pos_q6 = (row_bot, col_right)  # E / SE

            for key, pos in [("q1", pos_q1), ("q2", pos_q2), ("q5", pos_q5), ("q6", pos_q6)]:
                qid = coord_to_qid.get(pos)
                if qid is None:
                    raise ValueError(
                        f"Missing expected {key} position {pos} for w4 2x2 plaq_idx={plaq_idx}, coords={coords}"
                    )
                out[key] = int(qid)
            return out

        # Case B: L-shape boundary (three rows, two cols, four points).
        # Empirically in this construction, one column has 3 qubits (dense) and the other has 1 (sparse),
        # with the sparse qubit on the bottom row.
        if len(rows) == 3 and len(cols) == 2:
            col_a, col_b = cols
            pts_a = [q for q, (r, c) in coords.items() if c == col_a]
            pts_b = [q for q, (r, c) in coords.items() if c == col_b]

            if len(pts_a) == 3 and len(pts_b) == 1:
                dense_col, sparse_col = col_a, col_b
            elif len(pts_b) == 3 and len(pts_a) == 1:
                dense_col, sparse_col = col_b, col_a
            else:
                raise ValueError(
                    f"Unexpected w4 L-shape column counts for plaq_idx={plaq_idx}: cols={cols}, counts={[len(pts_a), len(pts_b)]}, coords={coords}"
                )

            row_top, row_mid, row_bot = rows

            # Use dense column for q1/q2 (feeding a1), and use (sparse bottom, dense bottom)
            # for q5/q6 (feeding a2). This uses all 4 data qubits and is consistent across boundary types.
            pos_q1 = (row_top, dense_col)
            pos_q2 = (row_mid, dense_col)
            pos_q5 = (row_bot, sparse_col)
            pos_q6 = (row_bot, dense_col)

            for key, pos in [("q1", pos_q1), ("q2", pos_q2), ("q5", pos_q5), ("q6", pos_q6)]:
                qid = coord_to_qid.get(pos)
                if qid is None:
                    raise ValueError(
                        f"Missing expected {key} position {pos} for w4 L-shape plaq_idx={plaq_idx}, coords={coords}"
                    )
                out[key] = int(qid)
            return out

        raise ValueError(
            f"Unsupported w4 geometry for plaq_idx={plaq_idx}: distinct_rows={len(rows)}, distinct_cols={len(cols)}, coords={coords}"
        )

        return out


if __name__ == "__main__":
    for d in [3, 5, 7]:
        print("=" * 60)
        c = ColorCode(d)
        c.print_structure()
