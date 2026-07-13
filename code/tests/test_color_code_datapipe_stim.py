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
Tests for Color Code Stim-based datapipe for inference.

Tests verify:
- Initialization with different parameters
- Correct output shapes and types
- Measurement parsing and grid embedding
- X and Z basis handling
- Mixed basis mode
"""

import pytest
import sys

sys.path.insert(0, 'code')

import torch


class TestColorCodeDatapipeStimBasic:
    """Basic tests for color code datapipe."""

    def test_import(self):
        """Test that the datapipe can be imported."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
        assert QCDataPipePreDecoder_ColorCode_inference is not None

    def test_init_x_basis(self):
        """Test initialization with X basis."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=3,
            num_samples=10,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='X',
        )

        assert dp.distance == 3
        assert dp.n_rounds == 3
        assert dp.num_samples == 10
        assert dp.measure_basis == 'X'
        assert len(dp) == 10

    def test_init_z_basis(self):
        """Test initialization with Z basis."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=3,
            num_samples=10,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='Z',
        )

        assert dp.measure_basis == 'Z'
        assert len(dp) == 10

    def test_init_mixed_basis(self):
        """Test initialization with mixed basis."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=3,
            num_samples=10,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='both',
        )

        assert dp._mixed
        assert len(dp) == 10

    def test_invalid_error_mode(self):
        """Test that invalid error mode raises error."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        with pytest.raises(ValueError):
            QCDataPipePreDecoder_ColorCode_inference(
                distance=3,
                n_rounds=3,
                num_samples=10,
                error_mode="invalid",
                p_error=0.001,
            )


class TestColorCodeDatapipeStimOutput:
    """Test output shapes and types."""

    @pytest.fixture
    def dp_d3(self):
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
        return QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=3,
            num_samples=5,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='X',
        )

    @pytest.fixture
    def dp_d5(self):
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
        return QCDataPipePreDecoder_ColorCode_inference(
            distance=5,
            n_rounds=5,
            num_samples=5,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='X',
        )

    def test_getitem_output_keys(self, dp_d3):
        """Test that getitem returns correct keys."""
        sample = dp_d3[0]

        assert 'trainX' in sample
        assert 'x_syn_diff' in sample
        assert 'z_syn_diff' in sample
        assert 'dets_and_obs' in sample

    def test_trainX_shape_d3(self, dp_d3):
        """Test trainX shape for d=3."""
        sample = dp_d3[0]
        trainX = sample['trainX']

        # d=3: n_rows = 3 + (3-1)//2 = 4, n_cols = 3
        # Shape: (4, n_rounds, n_rows, n_cols) = (4, 3, 4, 3)
        assert trainX.shape == (4, 3, 4, 3), f"Expected (4, 3, 4, 3), got {trainX.shape}"
        assert trainX.dtype == torch.float32

    def test_trainX_shape_d5(self, dp_d5):
        """Test trainX shape for d=5."""
        sample = dp_d5[0]
        trainX = sample['trainX']

        # d=5: n_rows = 5 + (5-1)//2 = 7, n_cols = 5
        # Shape: (4, n_rounds, n_rows, n_cols) = (4, 5, 7, 5)
        assert trainX.shape == (4, 5, 7, 5), f"Expected (4, 5, 7, 5), got {trainX.shape}"

    def test_syn_diff_shape_d3(self, dp_d3):
        """Test syndrome diff shapes for d=3."""
        sample = dp_d3[0]

        # d=3: num_plaquettes = (3 * (9-1)) // 8 = 3
        num_plaq = 3
        n_rounds = 3

        assert sample['x_syn_diff'].shape == (num_plaq, n_rounds)
        assert sample['z_syn_diff'].shape == (num_plaq, n_rounds)
        assert sample['x_syn_diff'].dtype == torch.int32
        assert sample['z_syn_diff'].dtype == torch.int32

    def test_syn_diff_shape_d5(self, dp_d5):
        """Test syndrome diff shapes for d=5."""
        sample = dp_d5[0]

        # d=5: num_plaquettes = (3 * (25-1)) // 8 = 9
        num_plaq = 9
        n_rounds = 5

        assert sample['x_syn_diff'].shape == (num_plaq, n_rounds)
        assert sample['z_syn_diff'].shape == (num_plaq, n_rounds)

    def test_trainX_contiguous(self, dp_d3):
        """Test that trainX is contiguous."""
        sample = dp_d3[0]
        assert sample['trainX'].is_contiguous()

    def test_dets_and_obs_type(self, dp_d3):
        """Test dets_and_obs type."""
        sample = dp_d3[0]
        assert sample['dets_and_obs'].dtype == torch.uint8


class TestColorCodeDatapipeStimBasisMasking:
    """Test basis-specific masking."""

    def test_x_basis_z_masking(self):
        """Test that Z syndromes are masked in first/last round for X basis."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=5,
            num_samples=5,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='X',
        )

        sample = dp[0]
        z_syn_diff = sample['z_syn_diff']

        # First and last round should be all zeros for Z
        assert torch.all(z_syn_diff[:, 0] == 0), "Z first round should be masked"
        assert torch.all(z_syn_diff[:, -1] == 0), "Z last round should be masked"

    def test_z_basis_x_masking(self):
        """Test that X syndromes are masked in first/last round for Z basis."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=5,
            num_samples=5,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='Z',
        )

        sample = dp[0]
        x_syn_diff = sample['x_syn_diff']

        # First and last round should be all zeros for X
        assert torch.all(x_syn_diff[:, 0] == 0), "X first round should be masked"
        assert torch.all(x_syn_diff[:, -1] == 0), "X last round should be masked"


class TestColorCodeDatapipeStimMixedBasis:
    """Test mixed basis mode."""

    def test_mixed_alternates_basis(self):
        """Test that mixed mode alternates between X and Z."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=5,
            num_samples=10,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='both',
        )

        # Even indices should be X (Z masked in first/last)
        sample_0 = dp[0]
        assert torch.all(sample_0['z_syn_diff'][:, 0] == 0), "Even idx should be X basis"

        # Odd indices should be Z (X masked in first/last)
        sample_1 = dp[1]
        assert torch.all(sample_1['x_syn_diff'][:, 0] == 0), "Odd idx should be Z basis"


class TestColorCodeDatapipeStimDistances:
    """Test different code distances."""

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_various_distances(self, d):
        """Test datapipe works for various distances."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=d,
            n_rounds=d,
            num_samples=3,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='X',
        )

        sample = dp[0]

        # Check grid dimensions
        n_rows = d + (d - 1) // 2
        n_cols = d

        assert sample['trainX'].shape == (4, d, n_rows, n_cols)


class TestColorCodeDatapipeStimRounds:
    """Test different numbers of rounds."""

    @pytest.mark.parametrize("n_rounds", [1, 3, 5, 7])
    def test_various_rounds(self, n_rounds):
        """Test datapipe works for various round counts."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=n_rounds,
            num_samples=3,
            error_mode="circuit_level_color_code",
            p_error=0.001,
            measure_basis='X',
        )

        sample = dp[0]

        # Check time dimension
        assert sample['trainX'].shape[1] == n_rounds
        assert sample['x_syn_diff'].shape[1] == n_rounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
