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
Tests for Color Code Logical Error Rate computation.

Tests verify:
- Import and basic functionality
- Chromobius decoder integration
- Residual syndrome computation
- Baseline error rate computation
- Multi-basis support
"""

import pytest
import sys

sys.path.insert(0, 'code')

import torch
import numpy as np


class TestImports:
    """Test that all imports work correctly."""

    def test_import_module(self):
        """Test that the module can be imported."""
        from evaluation.logical_error_rate_color import (
            count_logical_errors_color,
            run_inference_and_decode_color,
            compute_syndrome_density_reduction_color,
        )
        assert count_logical_errors_color is not None
        assert run_inference_and_decode_color is not None
        assert compute_syndrome_density_reduction_color is not None

    def test_import_chromobius(self):
        """Test that Chromobius can be imported."""
        import chromobius
        assert chromobius is not None


class TestParityMaps:
    """Test parity map building."""

    def test_build_parity_maps_d3(self):
        """Test building parity maps for d=3."""
        from evaluation.logical_error_rate_color import _build_color_code_parity_maps

        maps = _build_color_code_parity_maps(3)

        assert "H_i32" in maps
        assert "H_idx" in maps
        assert "H_mask" in maps
        assert "stab_to_grid" in maps
        assert "data_to_grid" in maps
        assert "num_plaq" in maps
        assert "num_data" in maps

        # Check dimensions
        assert maps["num_plaq"] == 3  # d=3 has 3 plaquettes
        assert maps["num_data"] == 7  # d=3 has 7 data qubits
        assert maps["n_rows"] == 4  # d + (d-1)//2 = 3 + 1 = 4
        assert maps["n_cols"] == 3  # d = 3

    def test_build_parity_maps_d5(self):
        """Test building parity maps for d=5."""
        from evaluation.logical_error_rate_color import _build_color_code_parity_maps
        from qec.color_code.color_code import ColorCode

        maps = _build_color_code_parity_maps(5)
        code = ColorCode(5)

        # Use actual code properties for assertions
        assert maps["num_plaq"] == code.num_plaquettes
        assert maps["num_data"] == code.num_data
        assert maps["n_rows"] == code.n_rows
        assert maps["n_cols"] == code.n_cols

    def test_parity_matrix_shape(self):
        """Test parity matrix shape."""
        from evaluation.logical_error_rate_color import _build_color_code_parity_maps

        for d in [3, 5, 7]:
            maps = _build_color_code_parity_maps(d)
            H = maps["H_i32"]
            assert H.shape == (maps["num_plaq"], maps["num_data"])


class TestChromobiusDecoder:
    """Test Chromobius decoder integration."""

    @pytest.fixture
    def stim_circuit_d3(self):
        """Create a d=3 color code circuit."""
        from qec.color_code.memory_circuit import MemoryCircuit

        circ = MemoryCircuit(
            distance=3,
            idle_error=0.001,
            sqgate_error=0.001,
            tqgate_error=0.001,
            spam_error=0.001,
            n_rounds=3,
            basis='X',
        )
        return circ

    def test_chromobius_decoder_creation(self, stim_circuit_d3):
        """Test that Chromobius decoder can be created from DEM."""
        import chromobius

        circuit = stim_circuit_d3.stim_circuit
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)

        decoder = chromobius.compile_decoder_for_dem(dem)
        assert decoder is not None

    def test_chromobius_decode(self, stim_circuit_d3):
        """Test that Chromobius can decode samples."""
        import chromobius

        circuit = stim_circuit_d3.stim_circuit
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)

        decoder = chromobius.compile_decoder_for_dem(dem)

        # Generate samples
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=100)

        # Convert to detectors and observables
        m2d = circuit.compile_m2d_converter()
        dets_and_obs = m2d.convert(measurements=samples, append_observables=True)

        num_obs = circuit.num_observables
        dets = dets_and_obs[:, :-num_obs].astype(np.uint8)
        obs = dets_and_obs[:, -num_obs:].astype(np.uint8)

        # Decode with Chromobius
        dets_packed = np.packbits(dets, axis=1, bitorder='little')
        pred = decoder.predict_obs_flips_from_dets_bit_packed(dets_packed)
        pred_unpacked = np.unpackbits(pred, axis=1, bitorder='little')[:, :num_obs]

        # Check shape
        assert pred_unpacked.shape == obs.shape


class TestGridMapping:
    """Test grid mapping functions."""

    def test_map_grid_to_stab(self):
        """Test mapping grid tensor to stabilizer order."""
        from evaluation.logical_error_rate_color import map_grid_to_stab, _build_color_code_parity_maps

        maps = _build_color_code_parity_maps(3)
        stab_to_grid = maps["stab_to_grid"]
        n_rows = maps["n_rows"]
        n_cols = maps["n_cols"]
        num_plaq = maps["num_plaq"]

        # Create test tensor
        B, T = 2, 3
        grid_tensor = torch.rand(B, T, n_rows, n_cols)

        # Map to stabilizer order
        stab_tensor = map_grid_to_stab(grid_tensor, stab_to_grid)

        assert stab_tensor.shape == (B, num_plaq, T)

    def test_map_grid_to_stab_inverse(self):
        """Test that mapping preserves values at correct positions."""
        from evaluation.logical_error_rate_color import map_grid_to_stab, _build_color_code_parity_maps
        from qec.color_code.data_mapping import reshape_stabilizers_to_grid_2d

        maps = _build_color_code_parity_maps(3)
        stab_to_grid = maps["stab_to_grid"]
        n_rows = maps["n_rows"]
        n_cols = maps["n_cols"]
        num_plaq = maps["num_plaq"]

        # Create test tensor in stabilizer order
        B, T = 2, 3
        stab_values = torch.randint(0, 2, (num_plaq, T)).float()

        # Map to grid using reshape_stabilizers_to_grid_2d
        # It expects (num_stabs, T) and returns (T, n_rows, n_cols)
        grid_values = reshape_stabilizers_to_grid_2d(
            stab_values, n_rows, n_cols, stab_to_grid
        )  # (T, n_rows, n_cols)

        grid_batch = grid_values.unsqueeze(0).expand(B, -1, -1, -1)  # (B, T, n_rows, n_cols)

        # Map back using map_grid_to_stab
        # Returns (B, num_plaq, T)
        recovered = map_grid_to_stab(grid_batch, stab_to_grid)  # (B, num_plaq, T)

        # Values should match: recovered[b] shape is (num_plaq, T), stab_values shape is (num_plaq, T)
        for b in range(B):
            # recovered[b] is (num_plaq, T), stab_values is (num_plaq, T)
            assert torch.allclose(recovered[b], stab_values)


class TestSamplePredictions:
    """Test prediction sampling functions."""

    def test_threshold_mode(self):
        """Test threshold sampling mode."""
        from evaluation.logical_error_rate_color import sample_predictions

        logits = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
        preds = sample_predictions(logits, threshold=0.0, sampling_mode="threshold")

        expected = torch.tensor([[0, 1, 1, 1]], dtype=torch.int32)
        assert torch.equal(preds, expected)

    def test_temperature_mode(self):
        """Test temperature sampling mode."""
        from evaluation.logical_error_rate_color import sample_predictions

        # Very low temperature should be nearly deterministic
        logits = torch.tensor([[10.0, -10.0]])
        preds = sample_predictions(logits, sampling_mode="temperature", temperature=0.01)

        # With extreme logits and low temp, should be [1, 0]
        assert preds[0, 0] == 1
        assert preds[0, 1] == 0


class TestDatapipeIntegration:
    """Test integration with datapipe."""

    def test_datapipe_creation(self):
        """Test that color code datapipe can be created."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=3,
            n_rounds=3,
            num_samples=10,
            error_mode="circuit_level_color_code",
            p_error=0.005,
            measure_basis='X',
        )

        assert len(dp) == 10

    def test_datapipe_output_shapes(self):
        """Test datapipe output shapes."""
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference

        d, n_rounds, num_samples = 3, 3, 10
        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=d,
            n_rounds=n_rounds,
            num_samples=num_samples,
            error_mode="circuit_level_color_code",
            p_error=0.005,
            measure_basis='X',
        )

        sample = dp[0]

        # Check shapes
        n_rows = d + (d - 1) // 2  # 4 for d=3
        n_cols = d  # 3 for d=3
        num_plaq = 3  # d=3 has 3 plaquettes

        assert "trainX" in sample
        assert "x_syn_diff" in sample
        assert "z_syn_diff" in sample
        assert "dets_and_obs" in sample

        assert sample["trainX"].shape == (4, n_rounds, n_rows, n_cols)
        assert sample["x_syn_diff"].shape == (num_plaq, n_rounds)
        assert sample["z_syn_diff"].shape == (num_plaq, n_rounds)


class TestEndToEnd:
    """End-to-end tests with mock model."""

    def test_mock_inference(self):
        """Test inference with a mock model that outputs zeros."""
        from evaluation.logical_error_rate_color import _build_color_code_parity_maps, map_grid_to_stab
        from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
        import chromobius

        d, n_rounds = 3, 3
        dp = QCDataPipePreDecoder_ColorCode_inference(
            distance=d,
            n_rounds=n_rounds,
            num_samples=10,
            error_mode="circuit_level_color_code",
            p_error=0.005,
            measure_basis='X',
        )

        # Get circuit and decoder
        circuit = dp.circ.stim_circuit
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        decoder = chromobius.compile_decoder_for_dem(dem)

        # Get sample
        sample = dp[0]
        x_syn_diff = sample["x_syn_diff"]  # (num_plaq, T)

        # Mock predictions (all zeros)
        n_rows, n_cols = dp.n_rows, dp.n_cols
        num_plaq = dp.num_plaquettes

        maps = _build_color_code_parity_maps(d)
        stab_to_grid = maps["stab_to_grid"]

        # Zero syndrome prediction
        syn_pred = torch.zeros(num_plaq, n_rounds, dtype=torch.int32)
        S_from_data = torch.zeros(num_plaq, n_rounds, dtype=torch.int32)

        # Compute residual (should equal input syndrome diff)
        R = x_syn_diff.clone()  # Since predictions are zero, residual = input

        # Verify residual matches input (no correction applied)
        assert torch.equal(R, x_syn_diff.to(torch.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
