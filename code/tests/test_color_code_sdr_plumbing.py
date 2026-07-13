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
Test SDR (Syndrome Density Reduction) plumbing for color code training/inference pipeline.

Verifies:
1. configure_metrics correctly selects color code SDR function
2. SDR function is importable and callable
3. SDR computation produces well-structured results
4. SDR integrates with the validation loop interface
5. SDR result extraction helper handles color code format

Speed: FAST (small samples, mock model or identity)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from types import SimpleNamespace


# -----------------------------------------------------------------------------
# Test 1: configure_metrics selects correct color code SDR function
# -----------------------------------------------------------------------------
def test_configure_metrics_selects_color_sdr():
    """Verify configure_metrics picks color-specific SDR when code='color'."""
    from evaluation.metrics import configure_metrics, HAS_LER_MODULE_COLOR
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color

    if not HAS_LER_MODULE_COLOR:
        pytest.skip("Color code LER module not available")

    ler_fn, sdr_fn = configure_metrics(rank=0, code="color")

    assert sdr_fn is compute_syndrome_density_reduction_color, (
        f"Expected compute_syndrome_density_reduction_color, got {sdr_fn}"
    )


def test_configure_metrics_color_variants():
    """Test that various 'color' code string variants are recognized."""
    from evaluation.metrics import configure_metrics, HAS_LER_MODULE_COLOR
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color

    if not HAS_LER_MODULE_COLOR:
        pytest.skip("Color code LER module not available")

    # Test different case variants
    for code_name in ["color", "Color", "COLOR", "color_code", "ColorCode"]:
        _, sdr_fn = configure_metrics(rank=0, code=code_name)
        assert sdr_fn is compute_syndrome_density_reduction_color, (
            f"code='{code_name}' should select color SDR"
        )


# -----------------------------------------------------------------------------
# Test 2: SDR function is importable and has correct signature
# -----------------------------------------------------------------------------
def test_sdr_function_importable():
    """Verify SDR function can be imported from logical_error_rate_color."""
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color

    assert callable(compute_syndrome_density_reduction_color)


def test_sdr_function_signature():
    """Verify SDR function has the expected signature."""
    import inspect
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color

    sig = inspect.signature(compute_syndrome_density_reduction_color)
    params = list(sig.parameters.keys())

    # Expected parameters: model, device, dist, cfg
    assert "model" in params
    assert "device" in params
    assert "dist" in params
    assert "cfg" in params


# -----------------------------------------------------------------------------
# Test 3: SDR result structure is correct
# -----------------------------------------------------------------------------
def test_sdr_result_has_required_keys():
    """Test that SDR result dict has all required keys for downstream processing."""
    # Simulate what a real SDR result should look like
    mock_result = {
        "input_syndrome_density": 0.05,
        "residual_syndrome_density": 0.01,
        "reduction_factor": 5.0,
        "basis": "X",
    }

    # Verify keys that metrics.py expects
    required_keys = ["input_syndrome_density", "residual_syndrome_density", "reduction_factor"]
    for key in required_keys:
        assert key in mock_result, f"Missing required key: {key}"


def test_extract_reduction_factor_from_color_sdr():
    """Test that _extract_reduction_factor correctly handles color code SDR format."""
    from evaluation.metrics import _extract_reduction_factor

    # Color code SDR returns 'reduction_factor' key
    color_result = {
        "input_syndrome_density": 0.05,
        "residual_syndrome_density": 0.01,
        "reduction_factor": 5.0,
        "basis": "X",
    }

    factor = _extract_reduction_factor(color_result)
    assert factor == 5.0, f"Expected 5.0, got {factor}"


def test_extract_reduction_factor_handles_nested_stim_format():
    """Test _extract_reduction_factor handles stim-style nested format."""
    from evaluation.metrics import _extract_reduction_factor

    # Surface code format with nested 'stim' key
    nested_result = {
        "stim": {
            "reduction factor (X)": 3.0,
            "reduction factor (Z)": 4.0,
        }
    }

    factor = _extract_reduction_factor(nested_result)
    assert factor == 3.5, f"Expected average 3.5, got {factor}"


# -----------------------------------------------------------------------------
# Test 4: SDR computation with mock model (end-to-end plumbing)
# -----------------------------------------------------------------------------
class MockColorCodeModel(torch.nn.Module):
    """Trivial model that outputs large negative logits (predicts no corrections).
    
    Note: sample_predictions uses threshold=0.0, so we need logits < 0 to get
    predictions of 0. Returning exactly 0 would give predictions of 1 since 0 >= 0.
    """

    def __init__(self, n_rows, n_cols, T):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.T = T
        # Minimal parameter to make it a valid module
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.shape[0]
        # Output 4 channels: [z_data, x_data, syn_x, syn_z]
        # Large negative logits -> sample_predictions returns 0 (no corrections)
        return torch.full(
            (B, 4, self.T, self.n_rows, self.n_cols),
            fill_value=-10.0,  # Large negative so threshold=0.0 gives 0
            device=x.device
        )


@pytest.fixture
def mock_dist():
    """Create mock distributed context."""
    return SimpleNamespace(
        rank=0,
        world_size=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


@pytest.fixture
def mock_cfg():
    """Create mock config for SDR computation."""
    return SimpleNamespace(
        distance=3,
        n_rounds=4,
        enable_fp16=False,
        test=SimpleNamespace(
            num_samples=64,
            p_error=0.01,
            meas_basis_test="X",
            th_data=0.0,
            th_syn=0.0,
            sampling_mode="threshold",
            temperature=1.0,
            dataloader={
                "batch_size": 32,
                "num_workers": 0
            },
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sdr_computation_end_to_end(mock_dist, mock_cfg):
    """Test full SDR computation pipeline with mock model."""
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color
    from qec.color_code.color_code import ColorCode

    # Setup
    D = mock_cfg.distance
    T = mock_cfg.n_rounds
    code = ColorCode(D)
    n_rows, n_cols = code.n_rows, code.n_cols

    # Create mock model
    model = MockColorCodeModel(n_rows, n_cols, T).to(mock_dist.device)

    # Run SDR computation
    result = compute_syndrome_density_reduction_color(
        model=model,
        device=mock_dist.device,
        dist=mock_dist,
        cfg=mock_cfg,
    )

    # Verify result structure
    assert isinstance(result, dict)
    assert "input_syndrome_density" in result
    assert "residual_syndrome_density" in result
    assert "reduction_factor" in result
    assert "basis" in result

    # Verify values are reasonable
    assert result["input_syndrome_density"] >= 0.0
    assert result["residual_syndrome_density"] >= 0.0
    # With mock model (no corrections), residual should equal input
    # so reduction factor should be ~1.0
    assert 0.5 <= result["reduction_factor"] <= 2.0, (
        f"With no-correction model, reduction should be ~1.0, got {result['reduction_factor']}"
    )
    assert result["basis"] == "X"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sdr_computation_both_bases(mock_dist, mock_cfg):
    """Test SDR computation works for both X and Z bases."""
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color
    from qec.color_code.color_code import ColorCode

    D = mock_cfg.distance
    T = mock_cfg.n_rounds
    code = ColorCode(D)
    n_rows, n_cols = code.n_rows, code.n_cols

    model = MockColorCodeModel(n_rows, n_cols, T).to(mock_dist.device)

    for basis in ["X", "Z"]:
        mock_cfg.test.meas_basis_test = basis

        result = compute_syndrome_density_reduction_color(
            model=model,
            device=mock_dist.device,
            dist=mock_dist,
            cfg=mock_cfg,
        )

        assert result["basis"] == basis
        assert result["input_syndrome_density"] >= 0.0


# -----------------------------------------------------------------------------
# Test 5: SDR integrates with compute_syndrome_density wrapper
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sdr_via_metrics_wrapper(mock_dist, mock_cfg):
    """Test SDR through the metrics.compute_syndrome_density wrapper."""
    from evaluation.metrics import configure_metrics, compute_syndrome_density, HAS_LER_MODULE_COLOR
    from qec.color_code.color_code import ColorCode

    if not HAS_LER_MODULE_COLOR:
        pytest.skip("Color code LER module not available")

    # Configure for color code
    configure_metrics(rank=0, code="color")

    D = mock_cfg.distance
    T = mock_cfg.n_rounds
    code = ColorCode(D)
    n_rows, n_cols = code.n_rows, code.n_cols

    model = MockColorCodeModel(n_rows, n_cols, T).to(mock_dist.device)

    # Call through wrapper
    result = compute_syndrome_density(
        model=model,
        device=mock_dist.device,
        dist=mock_dist,
        cfg=mock_cfg,
        generator=None,  # No on-the-fly generator for this test
        rank=0,
    )

    # Wrapper extracts reduction factor
    assert result is not None
    assert isinstance(result, float)
    assert result > 0


# -----------------------------------------------------------------------------
# Test 6: SDR handles edge cases
# -----------------------------------------------------------------------------
def test_sdr_handles_zero_syndrome_density():
    """Test that SDR computation handles zero syndrome case gracefully."""
    from evaluation.metrics import _extract_reduction_factor

    # Edge case: what if residual is 0?
    result = {
        "input_syndrome_density": 0.05,
        "residual_syndrome_density": 0.0,  # Perfect model
        "reduction_factor": float('inf'),
    }

    # _extract_reduction_factor should handle inf
    factor = _extract_reduction_factor(result)
    assert factor == float('inf')


def test_sdr_handles_nan():
    """Test SDR result extraction handles NaN values."""
    from evaluation.metrics import _extract_reduction_factor

    result = {
        "reduction_factor": float('nan'),
    }

    factor = _extract_reduction_factor(result)
    assert np.isnan(factor)


# -----------------------------------------------------------------------------
# Test 7: SDR parity map building
# -----------------------------------------------------------------------------
def test_sdr_parity_maps_correct_shape():
    """Verify parity maps have correct shape for color code."""
    from evaluation.logical_error_rate_color import _build_color_code_parity_maps
    from qec.color_code.color_code import ColorCode

    for D in [3, 5, 7]:
        code = ColorCode(D)
        maps = _build_color_code_parity_maps(D)

        # Color code formulas:
        # num_plaquettes = (3 * (d^2 - 1)) // 8
        # num_data = (3 * d^2 + 1) // 4
        expected_num_plaq = (3 * (D * D - 1)) // 8
        expected_num_data = (3 * D * D + 1) // 4

        assert maps["num_plaq"] == expected_num_plaq, f"d={D}: wrong num_plaq"
        assert maps["num_data"] == expected_num_data, f"d={D}: wrong num_data"
        assert maps["H_i32"].shape == (expected_num_plaq, expected_num_data)
        assert maps["H_idx"].shape[0] == expected_num_plaq
        assert maps["H_mask"].shape[0] == expected_num_plaq


def test_sdr_parity_maps_cached():
    """Verify parity maps are cached (same object returned for same distance)."""
    from evaluation.logical_error_rate_color import _build_color_code_parity_maps

    maps1 = _build_color_code_parity_maps(5)
    maps2 = _build_color_code_parity_maps(5)

    # Should be same cached object
    assert maps1 is maps2


# -----------------------------------------------------------------------------
# Test 8: SDR sample_predictions helper
# -----------------------------------------------------------------------------
def test_sample_predictions_threshold_mode():
    """Test sample_predictions in threshold mode."""
    from evaluation.logical_error_rate_color import sample_predictions

    logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    # Default threshold=0.0
    preds = sample_predictions(logits, threshold=0.0, sampling_mode="threshold")

    expected = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
    assert torch.equal(preds, expected)


def test_sample_predictions_temperature_mode():
    """Test sample_predictions in temperature mode produces valid binary output."""
    from evaluation.logical_error_rate_color import sample_predictions

    torch.manual_seed(42)
    logits = torch.randn(10, 100)

    preds = sample_predictions(logits, sampling_mode="temperature", temperature=1.0)

    # Should be binary
    assert torch.all((preds == 0) | (preds == 1))
    # Should be int32
    assert preds.dtype == torch.int32


# -----------------------------------------------------------------------------
# Test 9: Verify SDR formula correctness
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sdr_formula_input_equals_residual_for_identity_model(mock_dist, mock_cfg):
    """With identity model (no corrections), residual should equal input syndrome."""
    from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color
    from qec.color_code.color_code import ColorCode

    D = mock_cfg.distance
    T = mock_cfg.n_rounds
    code = ColorCode(D)
    n_rows, n_cols = code.n_rows, code.n_cols

    # Identity model: outputs zeros -> no corrections
    model = MockColorCodeModel(n_rows, n_cols, T).to(mock_dist.device)

    result = compute_syndrome_density_reduction_color(
        model=model,
        device=mock_dist.device,
        dist=mock_dist,
        cfg=mock_cfg,
    )

    # With no corrections: residual = input XOR 0 = input
    # So densities should be nearly equal (within sampling noise)
    input_d = result["input_syndrome_density"]
    residual_d = result["residual_syndrome_density"]

    # Allow small tolerance for any XOR with previous rounds
    assert abs(input_d - residual_d) < 0.1, (
        f"Expected input ≈ residual for identity model, got {input_d} vs {residual_d}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
