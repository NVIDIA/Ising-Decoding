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
Tests for Color Code data mapping functions.

Tests verify:
- Index mapping correctness for stabilizers and data qubits
- Reshape operations (stabilizer → grid, data → grid)
- Inverse operations (grid → stabilizer, grid → data)
- Round-trip consistency
- Edge cases and different distances
"""

import pytest
import sys

sys.path.insert(0, 'code')

import torch
import numpy as np


class TestIndexMappings:
    """Test index mapping functions."""

    @pytest.fixture
    def color_code_d3(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(3)

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    @pytest.fixture
    def color_code_d7(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(7)

    def test_stab_indices_shape_d3(self, color_code_d3):
        """Test stabilizer indices have correct shape for d=3."""
        from qec.color_code.data_mapping import get_stab_to_grid_flat_index

        cc = color_code_d3
        indices = get_stab_to_grid_flat_index(cc)

        assert indices.shape == (cc.num_plaquettes,)
        assert indices.dtype == torch.long

    def test_stab_indices_shape_d5(self, color_code_d5):
        """Test stabilizer indices have correct shape for d=5."""
        from qec.color_code.data_mapping import get_stab_to_grid_flat_index

        cc = color_code_d5
        indices = get_stab_to_grid_flat_index(cc)

        assert indices.shape == (cc.num_plaquettes,)
        assert indices.dtype == torch.long

    def test_stab_indices_in_bounds(self, color_code_d5):
        """Test all stabilizer indices are within grid bounds."""
        from qec.color_code.data_mapping import get_stab_to_grid_flat_index

        cc = color_code_d5
        indices = get_stab_to_grid_flat_index(cc)
        grid_size = cc.n_rows * cc.n_cols

        assert (indices >= 0).all()
        assert (indices < grid_size).all()

    def test_stab_indices_unique(self, color_code_d5):
        """Test all stabilizer indices are unique."""
        from qec.color_code.data_mapping import get_stab_to_grid_flat_index

        cc = color_code_d5
        indices = get_stab_to_grid_flat_index(cc)

        # All indices should be unique
        assert len(indices.unique()) == len(indices)

    def test_data_indices_shape_d3(self, color_code_d3):
        """Test data indices have correct shape for d=3."""
        from qec.color_code.data_mapping import get_data_to_grid_flat_index

        cc = color_code_d3
        indices = get_data_to_grid_flat_index(cc)

        assert indices.shape == (cc.num_data,)
        assert indices.dtype == torch.long

    def test_data_indices_in_bounds(self, color_code_d5):
        """Test all data indices are within grid bounds."""
        from qec.color_code.data_mapping import get_data_to_grid_flat_index

        cc = color_code_d5
        indices = get_data_to_grid_flat_index(cc)
        grid_size = cc.n_rows * cc.n_cols

        assert (indices >= 0).all()
        assert (indices < grid_size).all()

    def test_data_indices_unique(self, color_code_d5):
        """Test all data indices are unique."""
        from qec.color_code.data_mapping import get_data_to_grid_flat_index

        cc = color_code_d5
        indices = get_data_to_grid_flat_index(cc)

        # All indices should be unique
        assert len(indices.unique()) == len(indices)

    def test_stab_indices_match_colorcode(self, color_code_d5):
        """Test our indices match ColorCode's built-in method."""
        from qec.color_code.data_mapping import get_stab_to_grid_flat_index

        cc = color_code_d5
        our_indices = get_stab_to_grid_flat_index(cc)
        cc_indices = torch.tensor(cc.get_syndrome_grid_indices(), dtype=torch.long)

        assert torch.equal(our_indices, cc_indices)


class TestReshapeOperations:
    """Test reshape operations."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_reshape_stabilizers_to_grid_basic(self, color_code_d5):
        """Test basic stabilizer reshape."""
        from qec.color_code.data_mapping import (
            reshape_stabilizers_to_grid, get_stab_to_grid_flat_index
        )

        cc = color_code_d5
        stab_indices = get_stab_to_grid_flat_index(cc)

        # Create simple test tensor
        B, T = 4, 3
        stab_tensor = torch.randn(B, cc.num_plaquettes, T)

        grid_flat = reshape_stabilizers_to_grid(stab_tensor, cc.n_rows, cc.n_cols, stab_indices)

        # Check shape
        assert grid_flat.shape == (B, cc.n_rows * cc.n_cols, T)

    def test_reshape_stabilizers_to_grid_2d(self, color_code_d5):
        """Test 2D stabilizer reshape."""
        from qec.color_code.data_mapping import (
            reshape_stabilizers_to_grid_2d, get_stab_to_grid_flat_index
        )

        cc = color_code_d5
        stab_indices = get_stab_to_grid_flat_index(cc)

        B, T = 4, 3
        stab_tensor = torch.randn(B, cc.num_plaquettes, T)

        grid_2d = reshape_stabilizers_to_grid_2d(stab_tensor, cc.n_rows, cc.n_cols, stab_indices)

        # Check shape (B, T, n_rows, n_cols)
        assert grid_2d.shape == (B, T, cc.n_rows, cc.n_cols)

    def test_reshape_stabilizers_no_batch(self, color_code_d5):
        """Test stabilizer reshape without batch dimension."""
        from qec.color_code.data_mapping import (
            reshape_stabilizers_to_grid, get_stab_to_grid_flat_index
        )

        cc = color_code_d5
        stab_indices = get_stab_to_grid_flat_index(cc)

        T = 3
        stab_tensor = torch.randn(cc.num_plaquettes, T)

        grid_flat = reshape_stabilizers_to_grid(stab_tensor, cc.n_rows, cc.n_cols, stab_indices)

        # Should squeeze batch dimension
        assert grid_flat.shape == (cc.n_rows * cc.n_cols, T)

    def test_reshape_data_to_grid_basic(self, color_code_d5):
        """Test basic data reshape."""
        from qec.color_code.data_mapping import (reshape_data_to_grid, get_data_to_grid_flat_index)

        cc = color_code_d5
        data_indices = get_data_to_grid_flat_index(cc)

        B, T = 4, 3
        data_tensor = torch.randn(B, cc.num_data, T)

        grid_flat = reshape_data_to_grid(data_tensor, cc.n_rows, cc.n_cols, data_indices)

        assert grid_flat.shape == (B, cc.n_rows * cc.n_cols, T)

    def test_reshape_data_preserves_values(self, color_code_d5):
        """Test that reshape preserves values at correct positions."""
        from qec.color_code.data_mapping import (reshape_data_to_grid, get_data_to_grid_flat_index)

        cc = color_code_d5
        data_indices = get_data_to_grid_flat_index(cc)

        B, T = 2, 1
        data_tensor = torch.arange(cc.num_data, dtype=torch.float32)
        data_tensor = data_tensor.view(1, cc.num_data, 1).expand(B, -1, T)

        grid_flat = reshape_data_to_grid(data_tensor, cc.n_rows, cc.n_cols, data_indices)

        # Check values at known positions
        for i in range(cc.num_data):
            flat_idx = data_indices[i].item()
            expected = float(i)
            assert grid_flat[0, flat_idx, 0].item() == expected


class TestInverseOperations:
    """Test inverse mapping operations."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_map_grid_to_stabilizer(self, color_code_d5):
        """Test grid to stabilizer mapping."""
        from qec.color_code.data_mapping import (
            map_grid_to_stabilizer_tensor, get_stab_to_grid_flat_index
        )

        cc = color_code_d5
        stab_indices = get_stab_to_grid_flat_index(cc)

        B, T = 4, 3
        grid_tensor = torch.randn(B, T, cc.n_rows, cc.n_cols)

        stab_tensor = map_grid_to_stabilizer_tensor(grid_tensor, stab_indices)

        assert stab_tensor.shape == (B, cc.num_plaquettes, T)

    def test_map_grid_to_data(self, color_code_d5):
        """Test grid to data mapping."""
        from qec.color_code.data_mapping import (
            map_grid_to_data_tensor, get_data_to_grid_flat_index
        )

        cc = color_code_d5
        data_indices = get_data_to_grid_flat_index(cc)

        B, T = 4, 3
        grid_tensor = torch.randn(B, T, cc.n_rows, cc.n_cols)

        data_tensor = map_grid_to_data_tensor(grid_tensor, data_indices)

        assert data_tensor.shape == (B, cc.num_data, T)


class TestRoundTrip:
    """Test round-trip consistency."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_stab_round_trip(self, color_code_d5):
        """Test stabilizer → grid → stabilizer round trip."""
        from qec.color_code.data_mapping import (
            reshape_stabilizers_to_grid_2d, map_grid_to_stabilizer_tensor,
            get_stab_to_grid_flat_index
        )

        cc = color_code_d5
        stab_indices = get_stab_to_grid_flat_index(cc)

        B, T = 4, 3
        original = torch.randn(B, cc.num_plaquettes, T)

        # Forward: stab → grid
        grid = reshape_stabilizers_to_grid_2d(original, cc.n_rows, cc.n_cols, stab_indices)

        # Backward: grid → stab
        recovered = map_grid_to_stabilizer_tensor(grid, stab_indices)

        # Should be identical
        assert torch.allclose(original, recovered)

    def test_data_round_trip(self, color_code_d5):
        """Test data → grid → data round trip."""
        from qec.color_code.data_mapping import (
            reshape_data_to_grid_2d, map_grid_to_data_tensor, get_data_to_grid_flat_index
        )

        cc = color_code_d5
        data_indices = get_data_to_grid_flat_index(cc)

        B, T = 4, 3
        original = torch.randn(B, cc.num_data, T)

        # Forward: data → grid
        grid = reshape_data_to_grid_2d(original, cc.n_rows, cc.n_cols, data_indices)

        # Backward: grid → data
        recovered = map_grid_to_data_tensor(grid, data_indices)

        # Should be identical
        assert torch.allclose(original, recovered)


class TestConvenienceFunctions:
    """Test convenience functions that take ColorCode directly."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_reshape_stabilizers_vectorized(self, color_code_d5):
        """Test vectorized stabilizer reshape."""
        from qec.color_code.data_mapping import reshape_stabilizers_to_grid_vectorized

        cc = color_code_d5
        B, T = 4, 3
        stab_tensor = torch.randn(B, cc.num_plaquettes, T)

        result = reshape_stabilizers_to_grid_vectorized(stab_tensor, cc)

        assert result.shape == (B, cc.n_rows * cc.n_cols, T)

    def test_reshape_data_vectorized(self, color_code_d5):
        """Test vectorized data reshape."""
        from qec.color_code.data_mapping import reshape_data_to_grid_vectorized

        cc = color_code_d5
        B, T = 4, 3
        data_tensor = torch.randn(B, cc.num_data, T)

        result = reshape_data_to_grid_vectorized(data_tensor, cc)

        assert result.shape == (B, cc.n_rows * cc.n_cols, T)


class TestPresenceMasks:
    """Test presence mask generation."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_stabilizer_presence_mask_shape(self, color_code_d5):
        """Test stabilizer presence mask shape."""
        from qec.color_code.data_mapping import get_stabilizer_presence_mask

        cc = color_code_d5
        mask = get_stabilizer_presence_mask(cc)

        assert mask.shape == (cc.n_rows, cc.n_cols)

    def test_stabilizer_presence_mask_count(self, color_code_d5):
        """Test stabilizer presence mask has correct number of 1s."""
        from qec.color_code.data_mapping import get_stabilizer_presence_mask

        cc = color_code_d5
        mask = get_stabilizer_presence_mask(cc)

        assert mask.sum().item() == cc.num_plaquettes

    def test_data_presence_mask_shape(self, color_code_d5):
        """Test data presence mask shape."""
        from qec.color_code.data_mapping import get_data_presence_mask

        cc = color_code_d5
        mask = get_data_presence_mask(cc)

        assert mask.shape == (cc.n_rows, cc.n_cols)

    def test_data_presence_mask_count(self, color_code_d5):
        """Test data presence mask has correct number of 1s."""
        from qec.color_code.data_mapping import get_data_presence_mask

        cc = color_code_d5
        mask = get_data_presence_mask(cc)

        assert mask.sum().item() == cc.num_data


class TestNormalizedWeights:
    """Test normalized weight mapping."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_normalized_weights_shape(self, color_code_d5):
        """Test normalized weights have correct shape."""
        from qec.color_code.data_mapping import normalized_weight_mapping_stab

        cc = color_code_d5
        weights = normalized_weight_mapping_stab(cc)

        assert weights.shape == (cc.n_rows * cc.n_cols,)

    def test_normalized_weights_values(self, color_code_d5):
        """Test normalized weights have correct values."""
        from qec.color_code.data_mapping import normalized_weight_mapping_stab

        cc = color_code_d5
        weights = normalized_weight_mapping_stab(cc)

        # Count weight-6 (bulk) and weight-4 (boundary) plaquettes
        n_bulk = sum(1 for p in cc.plaquettes if p['weight'] == 6)
        n_boundary = sum(1 for p in cc.plaquettes if p['weight'] == 4)

        # Check values
        expected_sum = n_bulk * 1.0 + n_boundary * 0.5
        actual_sum = weights.sum().item()

        assert abs(expected_sum - actual_sum) < 1e-6


class TestMultipleDistances:
    """Test functions work for multiple distances."""

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_stab_indices_all_distances(self, distance):
        """Test stabilizer indices work for all distances."""
        from qec.color_code.color_code import ColorCode
        from qec.color_code.data_mapping import get_stab_to_grid_flat_index

        cc = ColorCode(distance)
        indices = get_stab_to_grid_flat_index(cc)

        assert indices.shape == (cc.num_plaquettes,)
        assert (indices >= 0).all()
        assert (indices < cc.n_rows * cc.n_cols).all()

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_data_indices_all_distances(self, distance):
        """Test data indices work for all distances."""
        from qec.color_code.color_code import ColorCode
        from qec.color_code.data_mapping import get_data_to_grid_flat_index

        cc = ColorCode(distance)
        indices = get_data_to_grid_flat_index(cc)

        assert indices.shape == (cc.num_data,)
        assert (indices >= 0).all()
        assert (indices < cc.n_rows * cc.n_cols).all()

    @pytest.mark.parametrize("distance", [3, 5, 7])
    def test_round_trip_all_distances(self, distance):
        """Test round trip works for all distances."""
        from qec.color_code.color_code import ColorCode
        from qec.color_code.data_mapping import (
            reshape_stabilizers_to_grid_2d, map_grid_to_stabilizer_tensor,
            get_stab_to_grid_flat_index
        )

        cc = ColorCode(distance)
        stab_indices = get_stab_to_grid_flat_index(cc)

        B, T = 2, 2
        original = torch.randn(B, cc.num_plaquettes, T)

        grid = reshape_stabilizers_to_grid_2d(original, cc.n_rows, cc.n_cols, stab_indices)
        recovered = map_grid_to_stabilizer_tensor(grid, stab_indices)

        assert torch.allclose(original, recovered)


class TestCompatibilityFunctions:
    """Test compatibility functions."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_compute_stab_to_data_index_map(self, color_code_d5):
        """Test stab to data index map."""
        from qec.color_code.data_mapping import compute_stab_to_data_index_map

        cc = color_code_d5
        mapping = compute_stab_to_data_index_map(cc)

        assert mapping.shape == (cc.num_plaquettes,)
        assert (mapping >= 0).all()
        assert (mapping < cc.num_data).all()

    def test_compute_data_to_stab_index_map(self, color_code_d5):
        """Test data to stab index map."""
        from qec.color_code.data_mapping import compute_data_to_stab_index_map

        cc = color_code_d5
        mapping = compute_data_to_stab_index_map(cc)

        assert mapping.shape == (cc.num_data,)

        # Count non -1 entries (should match num_plaquettes)
        non_sentinel = (mapping >= 0).sum().item()
        assert non_sentinel == cc.num_plaquettes

    def test_stab_data_maps_consistent(self, color_code_d5):
        """Test forward and reverse maps are consistent."""
        from qec.color_code.data_mapping import (
            compute_stab_to_data_index_map, compute_data_to_stab_index_map
        )

        cc = color_code_d5
        stab_to_data = compute_stab_to_data_index_map(cc)
        data_to_stab = compute_data_to_stab_index_map(cc)

        # For each stabilizer, check reverse map
        for stab_idx in range(cc.num_plaquettes):
            data_idx = stab_to_data[stab_idx].item()
            assert data_to_stab[data_idx].item() == stab_idx


class TestParityMatrix:
    """Test parity matrix extraction."""

    @pytest.fixture
    def color_code_d5(self):
        from qec.color_code.color_code import ColorCode
        return ColorCode(5)

    def test_parity_matrix_shape(self, color_code_d5):
        """Test parity matrix has correct shape."""
        from qec.color_code.data_mapping import get_parity_matrix_data_only

        cc = color_code_d5
        parity = get_parity_matrix_data_only(cc)

        assert parity.shape == (cc.num_plaquettes, cc.num_data)

    def test_parity_matrix_weights(self, color_code_d5):
        """Test parity matrix row sums match plaquette weights."""
        from qec.color_code.data_mapping import get_parity_matrix_data_only

        cc = color_code_d5
        parity = get_parity_matrix_data_only(cc)

        for i, plaq in enumerate(cc.plaquettes):
            row_sum = parity[i].sum().item()
            assert row_sum == plaq['weight']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
