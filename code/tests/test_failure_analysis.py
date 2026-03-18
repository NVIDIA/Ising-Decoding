# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

try:
    import ldpc
    import beliefmatching
    import scipy
    _HAS_LDPC_DEPS = True
except ImportError:
    _HAS_LDPC_DEPS = False

_skip_ldpc = unittest.skipUnless(_HAS_LDPC_DEPS, "ldpc/beliefmatching/scipy not installed")


def _make_tiny_dem(distance=3, n_rounds=3, basis="X", code_rotation="XV"):
    """Build a minimal surface-code DEM (with boundary detectors) for testing."""
    from qec.surface_code.memory_circuit import MemoryCircuit
    mc = MemoryCircuit(
        distance=distance,
        idle_error=0.01,
        sqgate_error=0.01,
        tqgate_error=0.01,
        spam_error=0.007,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=code_rotation,
        add_boundary_detectors=True,
    )
    mc.set_error_rates()
    return mc.stim_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )


def _make_cfg(output_dir, distance=3, n_rounds=3, basis="X", n_samples=8):
    """Build a minimal cfg SimpleNamespace for decoder_ablation_study."""
    test_ns = types.SimpleNamespace(
        th_data=0.0,
        th_syn=0.0,
        sampling_mode="threshold",
        temperature=1.0,
        temperature_data=None,
        temperature_syn=None,
        meas_basis_test=basis,
        num_samples=n_samples,
        p_error=0.01,
        dataloader=types.SimpleNamespace(batch_size=n_samples),
        use_model_checkpoint=-1,
    )
    data_ns = types.SimpleNamespace(
        enable_correlated_pymatching=False,
        code_rotation="XV",
    )
    return types.SimpleNamespace(
        test=test_ns,
        data=data_ns,
        distance=distance,
        n_rounds=n_rounds,
        enable_fp16=False,
        output=output_dir,
    )


class _ZeroModel(torch.nn.Module):
    """Model that always returns zero logits (same shape as input)."""

    def forward(self, x):
        return torch.zeros_like(x)


class _FakeDist:
    rank = 0
    world_size = 1
    local_rank = 0
    device = torch.device("cpu")


@_skip_ldpc
class TestBuildLdpcDecoders(unittest.TestCase):
    """_build_ldpc_decoders must return correctly keyed decoder objects with consistent shapes."""

    def setUp(self):
        from evaluation.failure_analysis import _build_ldpc_decoders
        self.det_model = _make_tiny_dem()
        self.decoders = _build_ldpc_decoders(self.det_model)

    def test_expected_decoder_names_present(self):
        from evaluation.failure_analysis import LDPC_DECODER_NAMES
        for name in LDPC_DECODER_NAMES:
            self.assertIn(name, self.decoders)

    def test_each_entry_is_decoder_and_l_dense_pair(self):
        for name, (dec, L_dense) in self.decoders.items():
            with self.subTest(decoder=name):
                self.assertIsInstance(L_dense, np.ndarray)
                self.assertEqual(L_dense.dtype, np.uint8)
                # rows = num_observables (1 for surface code), cols = num error mechanisms
                self.assertEqual(L_dense.shape[0], self.det_model.num_observables)
                self.assertGreater(L_dense.shape[1], 0)
                self.assertTrue(hasattr(dec, "decode"), f"{name} decoder has no .decode()")

    def test_l_dense_columns_consistent_across_decoders(self):
        widths = [v[1].shape[1] for v in self.decoders.values()]
        self.assertEqual(len(set(widths)), 1, "All L_dense must have the same column count")


@_skip_ldpc
class TestDecodeLdpcBatch(unittest.TestCase):
    """_decode_ldpc_batch must return correct shape/dtype; zero syndrome decodes to 0."""

    def setUp(self):
        from evaluation.failure_analysis import _build_ldpc_decoders, _decode_ldpc_batch
        self._fn = _decode_ldpc_batch
        det_model = _make_tiny_dem()
        self.decoders = _build_ldpc_decoders(det_model)
        self.num_detectors = det_model.num_detectors

    def test_zero_syndrome_gives_zero_observable(self):
        B = 4
        syndromes = np.zeros((B, self.num_detectors), dtype=np.uint8)
        for name, (dec, L_dense) in self.decoders.items():
            with self.subTest(decoder=name):
                obs = self._fn(dec, L_dense, syndromes)
                np.testing.assert_array_equal(
                    obs,
                    np.zeros(B, dtype=np.uint8),
                    err_msg=f"{name}: zero syndrome should give zero observable",
                )

    def test_output_shape_is_batch_size(self):
        for B in (1, 6):
            syndromes = np.zeros((B, self.num_detectors), dtype=np.uint8)
            for name, (dec, L_dense) in self.decoders.items():
                with self.subTest(decoder=name, B=B):
                    obs = self._fn(dec, L_dense, syndromes)
                    self.assertEqual(obs.shape, (B,))
                    self.assertEqual(obs.dtype, np.uint8)

    def test_output_values_are_binary(self):
        """Observable must be 0 or 1; use sparse single-bit syndromes (fast for all decoders)."""
        B = min(4, self.num_detectors)
        syndromes = np.zeros((B, self.num_detectors), dtype=np.uint8)
        for i in range(B):
            syndromes[i, i] = 1  # one detector fired per sample
        for name, (dec, L_dense) in self.decoders.items():
            with self.subTest(decoder=name):
                obs = self._fn(dec, L_dense, syndromes)
                self.assertTrue(
                    np.all((obs == 0) | (obs == 1)),
                    f"{name}: output contains values other than 0/1",
                )


@_skip_ldpc
class TestDecoderAblationStudy(unittest.TestCase):
    """
    Smoke test: decoder_ablation_study must complete, return expected keys,
    and report the correct sample count.
    """

    _D = 3
    _T = 3
    _N = 8

    def _build_datapipe(self, basis):
        from data.datapipe_stim import QCDataPipePreDecoder_Memory_inference
        return QCDataPipePreDecoder_Memory_inference(
            distance=self._D,
            n_rounds=self._T,
            num_samples=self._N,
            error_mode="circuit_level_surface_custom",
            p_error=0.01,
            measure_basis=basis,
            code_rotation="XV",
        )

    def _run(self, basis):
        from evaluation.failure_analysis import decoder_ablation_study
        real_ds = self._build_datapipe(basis)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_cfg(
                tmpdir, distance=self._D, n_rounds=self._T, basis=basis, n_samples=self._N
            )
            with patch("data.factory.DatapipeFactory") as mock_factory:
                mock_factory.create_datapipe_inference.return_value = real_ds
                result = decoder_ablation_study(_ZeroModel(), _FakeDist.device, _FakeDist(), cfg)
        return result

    def test_return_keys_present(self):
        result = self._run("X")
        for key in (
            "total_samples",
            "baseline_errors",
            "decoder_errors",
            "residual_weights",
            "weight_bucket_stats",
            "agreement_count",
        ):
            self.assertIn(key, result, f"Missing key in result: {key}")

    def test_total_samples_matches_dataset_size(self):
        result = self._run("X")
        self.assertEqual(result["total_samples"], self._N)

    def test_decoder_errors_has_all_six_decoders(self):
        from evaluation.failure_analysis import DECODER_NAMES
        result = self._run("X")
        self.assertEqual(set(result["decoder_errors"].keys()), set(DECODER_NAMES))

    def test_residual_weights_length_matches_total_samples(self):
        result = self._run("X")
        self.assertEqual(len(result["residual_weights"]), result["total_samples"])

    def test_agreement_count_within_bounds(self):
        result = self._run("X")
        self.assertGreaterEqual(result["agreement_count"], 0)
        self.assertLessEqual(result["agreement_count"], result["total_samples"])

    def test_z_basis_runs_and_returns_correct_structure(self):
        result = self._run("Z")
        self.assertEqual(result["total_samples"], self._N)
        self.assertIn("decoder_errors", result)


if __name__ == "__main__":
    unittest.main()
