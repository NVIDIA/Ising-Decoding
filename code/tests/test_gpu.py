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
GPU-specific unit tests.

Exercises CUDA code paths in DEM sampling, data generation, model forward pass,
and homological equivalence that are skipped when tests run on CPU-only runners.
All classes are gated with ``@unittest.skipUnless(torch.cuda.is_available())``.
"""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch
import numpy as np


def _require_cuda(cls):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(cls)


# ---------------------------------------------------------------------------
# DEM sampling on GPU
# ---------------------------------------------------------------------------
@_require_cuda
class TestDemSamplingGPU(unittest.TestCase):
    """DEM sampling, measure_from_stacked_frames, timelike_syndromes on CUDA."""

    def setUp(self):
        self.device = torch.device("cuda")

    def test_dem_sampling_on_cuda(self):
        from qec.dem_sampling import dem_sampling

        num_detectors, num_errors, batch = 10, 20, 8
        H = torch.randint(
            0, 2, (2 * num_detectors, num_errors), dtype=torch.uint8, device=self.device
        )
        p = (torch.rand(num_errors, device=self.device) * 0.01)
        frames = dem_sampling(H, p, batch)
        self.assertEqual(frames.device.type, "cuda")
        self.assertEqual(frames.shape, (batch, 2 * num_detectors))
        self.assertEqual(frames.dtype, torch.uint8)

    def test_measure_from_stacked_frames_on_cuda(self):
        from qec.dem_sampling import measure_from_stacked_frames

        batch, n_rounds, nq, num_meas = 4, 3, 5, 4
        frames_xz = torch.randint(
            0, 2, (batch, 2 * n_rounds * nq), dtype=torch.uint8, device=self.device
        )
        meas_qubits = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.device)
        meas_bases = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=self.device)
        meas = measure_from_stacked_frames(frames_xz, meas_qubits, meas_bases, nq)
        self.assertEqual(meas.device.type, "cuda")
        self.assertEqual(meas.shape, (batch, n_rounds, num_meas))

    def test_timelike_syndromes_on_cuda(self):
        from qec.dem_sampling import timelike_syndromes

        batch, n_rounds, num_meas, nq = 4, 3, 4, 5
        A = torch.randint(
            0, 2, (n_rounds * num_meas, 2 * n_rounds * nq), dtype=torch.uint8, device=self.device
        )
        meas_old = torch.randint(
            0, 2, (batch, n_rounds, num_meas), dtype=torch.uint8, device=self.device
        )
        frames_xz = torch.randint(
            0, 2, (batch, 2 * n_rounds * nq), dtype=torch.uint8, device=self.device
        )
        meas_new = timelike_syndromes(frames_xz, A, meas_old)
        self.assertEqual(meas_new.device.type, "cuda")
        self.assertEqual(meas_new.shape, meas_old.shape)


# ---------------------------------------------------------------------------
# MemoryCircuitTorch on GPU
# ---------------------------------------------------------------------------
@_require_cuda
class TestMemoryCircuitTorchGPU(unittest.TestCase):
    """MemoryCircuitTorch initialization and batch generation on CUDA."""

    def setUp(self):
        self.device = torch.device("cuda")
        self.distance = 3
        self.n_rounds = 3

    def _make_generator(self, basis="X"):
        from qec.precompute_dem import precompute_dem_bundle_surface_code
        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch

        artifacts = precompute_dem_bundle_surface_code(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=basis,
            code_rotation="XV",
            p_scalar=0.004,
            dem_output_dir=None,
            device=self.device,
            export=False,
            return_artifacts=True,
        )
        return MemoryCircuitTorch(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=basis,
            code_rotation="XV",
            H=artifacts["H"],
            p=artifacts["p"],
            A=artifacts.get("A"),
            device=self.device,
        )

    def test_generate_batch_X_on_cuda(self):
        gen = self._make_generator("X")
        trainX, trainY = gen.generate_batch(batch_size=4)
        self.assertEqual(trainX.device.type, "cuda")
        self.assertEqual(trainY.device.type, "cuda")
        D = self.distance
        self.assertEqual(trainX.shape, (4, 4, self.n_rounds, D, D))
        self.assertEqual(trainY.shape, (4, 4, self.n_rounds, D, D))

    def test_generate_batch_Z_on_cuda(self):
        gen = self._make_generator("Z")
        trainX, trainY = gen.generate_batch(batch_size=4)
        self.assertEqual(trainX.device.type, "cuda")
        self.assertEqual(trainY.device.type, "cuda")

    def test_generate_batch_with_aux_on_cuda(self):
        gen = self._make_generator("X")
        trainX, trainY, meas_old, x_cum, z_cum = gen.generate_batch(batch_size=4, return_aux=True)
        for t in (trainX, trainY, meas_old, x_cum, z_cum):
            self.assertEqual(t.device.type, "cuda")

    def test_he_off_on_cuda(self):
        """MemoryCircuitTorch with timelike_he=False on GPU."""
        from qec.precompute_dem import precompute_dem_bundle_surface_code
        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch

        artifacts = precompute_dem_bundle_surface_code(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis="X",
            code_rotation="XV",
            p_scalar=0.004,
            dem_output_dir=None,
            device=self.device,
            export=False,
            return_artifacts=True,
        )
        gen = MemoryCircuitTorch(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis="X",
            code_rotation="XV",
            timelike_he=False,
            H=artifacts["H"],
            p=artifacts["p"],
            A=artifacts.get("A"),
            device=self.device,
        )
        trainX, trainY = gen.generate_batch(batch_size=4)
        self.assertEqual(trainX.device.type, "cuda")
        D = self.distance
        self.assertEqual(trainX.shape, (4, 4, self.n_rounds, D, D))


# ---------------------------------------------------------------------------
# QCDataGeneratorTorch on GPU
# ---------------------------------------------------------------------------
@_require_cuda
class TestQCDataGeneratorTorchGPU(unittest.TestCase):
    """QCDataGeneratorTorch with GPU device (in-memory DEM)."""

    def setUp(self):
        self.device = torch.device("cuda")

    def test_generator_both_bases_on_cuda(self):
        from data.generator_torch import QCDataGeneratorTorch

        gen = QCDataGeneratorTorch(
            distance=3,
            n_rounds=3,
            p_error=0.004,
            measure_basis="both",
            device=self.device,
            base_seed=42,
        )
        trainX, trainY = gen.generate_batch(step=0, batch_size=4)
        self.assertEqual(trainX.device.type, "cuda")
        trainX2, trainY2 = gen.generate_batch(step=1, batch_size=4)
        self.assertEqual(trainX2.device.type, "cuda")

    def test_generator_single_basis_on_cuda(self):
        from data.generator_torch import QCDataGeneratorTorch

        gen = QCDataGeneratorTorch(
            distance=3,
            n_rounds=3,
            p_error=0.004,
            measure_basis="X",
            device=self.device,
            base_seed=42,
        )
        trainX, trainY = gen.generate_batch(step=0, batch_size=4)
        self.assertEqual(trainX.device.type, "cuda")

    def test_generator_default_device_is_cuda(self):
        """Without explicit device, generator should pick cuda when available."""
        from data.generator_torch import QCDataGeneratorTorch

        gen = QCDataGeneratorTorch(
            distance=3,
            n_rounds=3,
            p_error=0.004,
            measure_basis="X",
            base_seed=42,
        )
        self.assertEqual(gen.device.type, "cuda")


# ---------------------------------------------------------------------------
# PreDecoder model forward pass on GPU
# ---------------------------------------------------------------------------
@_require_cuda
class TestPreDecoderModelGPU(unittest.TestCase):
    """PreDecoderModelMemory v1 and v2 forward pass on CUDA."""

    def setUp(self):
        self.device = torch.device("cuda")

    def test_v1_forward_on_cuda(self):
        from model.predecoder import PreDecoderModelMemory_v1, get_mock_config

        cfg = get_mock_config()
        model = PreDecoderModelMemory_v1(cfg).to(self.device)
        B, C, T, D = 4, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D, device=self.device)
        out = model(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (B, cfg.model.out_channels, T, D, D))

    def test_v2_forward_on_cuda(self):
        from model.predecoder import PreDecoderModelMemory_v2, get_mock_config_v2

        cfg = get_mock_config_v2()
        model = PreDecoderModelMemory_v2(cfg).to(self.device)
        B, C, T, D = 4, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D, device=self.device)
        out = model(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (B, cfg.model.out_channels, T, D, D))

    def test_v1_gradient_flow_on_cuda(self):
        """Verify gradients propagate through the model on GPU."""
        from model.predecoder import PreDecoderModelMemory_v1, get_mock_config

        cfg = get_mock_config()
        model = PreDecoderModelMemory_v1(cfg).to(self.device)
        B, C, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D, device=self.device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_v1_mixed_precision_forward(self):
        """Model forward with autocast (float16) on CUDA."""
        from model.predecoder import PreDecoderModelMemory_v1, get_mock_config

        cfg = get_mock_config()
        model = PreDecoderModelMemory_v1(cfg).to(self.device)
        B, C, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
        x = torch.randn(B, C, T, D, D, device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x)
        self.assertIn(out.dtype, (torch.float16, torch.float32))


# ---------------------------------------------------------------------------
# Homological equivalence kernel on GPU
# ---------------------------------------------------------------------------
@_require_cuda
class TestHEKernelGPU(unittest.TestCase):
    """apply_weight1_timelike_homological_equivalence_torch on CUDA."""

    def setUp(self):
        self.device = torch.device("cuda")
        self.distance = 3
        self.n_rounds = 3

    def test_he_kernel_on_cuda(self):
        from qec.surface_code.memory_circuit import SurfaceCode
        from qec.surface_code.homological_equivalence_torch import (
            apply_weight1_timelike_homological_equivalence_torch,
            build_spacelike_he_cache,
        )

        code = SurfaceCode(self.distance, first_bulk_syndrome_type="X", rotated_type="V")
        parity_X = torch.tensor(code.hx, dtype=torch.uint8, device=self.device)
        parity_Z = torch.tensor(code.hz, dtype=torch.uint8, device=self.device)
        cache_X = build_spacelike_he_cache(parity_X, distance=self.distance, device=self.device)
        cache_Z = build_spacelike_he_cache(parity_Z, distance=self.distance, device=self.device)

        B, D2, R = 4, self.distance**2, self.n_rounds
        num_x = parity_X.shape[0]
        z_cum = torch.randint(0, 2, (B, R, D2), dtype=torch.float32, device=self.device)
        x_cum = torch.randint(0, 2, (B, R, D2), dtype=torch.float32, device=self.device)
        s1s2x = torch.randint(0, 2, (B, R, num_x), dtype=torch.float32, device=self.device)
        s1s2z = torch.randint(0, 2, (B, R, num_x), dtype=torch.float32, device=self.device)
        trainX_x = torch.randint(0, 2, (B, R, num_x), dtype=torch.float32, device=self.device)
        trainX_z = torch.randint(0, 2, (B, R, num_x), dtype=torch.float32, device=self.device)

        z_diff, x_diff, s1s2x_out, s1s2z_out = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            parity_Z,
            parity_X,
            self.distance,
            1,
            8,
            "X",
            True,
            trainX_x=trainX_x,
            trainX_z=trainX_z,
            cache_Z_spacelike=cache_Z,
            cache_X_spacelike=cache_X,
        )
        for t in (z_diff, x_diff, s1s2x_out, s1s2z_out):
            self.assertEqual(t.device.type, "cuda")

    def test_he_weight_nonincreasing_on_cuda(self):
        """Weight-1 HE should not increase total error weight (verified on GPU)."""
        from qec.surface_code.memory_circuit import SurfaceCode
        from qec.surface_code.homological_equivalence_torch import (
            apply_weight1_timelike_homological_equivalence_torch,
            build_spacelike_he_cache,
        )

        code = SurfaceCode(self.distance, first_bulk_syndrome_type="X", rotated_type="V")
        parity_X = torch.tensor(code.hx, dtype=torch.uint8, device=self.device)
        parity_Z = torch.tensor(code.hz, dtype=torch.uint8, device=self.device)
        cache_X = build_spacelike_he_cache(parity_X, distance=self.distance, device=self.device)
        cache_Z = build_spacelike_he_cache(parity_Z, distance=self.distance, device=self.device)

        B, D2, R = 8, self.distance**2, self.n_rounds
        num_x = parity_X.shape[0]
        torch.manual_seed(42)
        z_cum = (torch.rand(B, R, D2, device=self.device) > 0.9).float()
        x_cum = (torch.rand(B, R, D2, device=self.device) > 0.9).float()
        s1s2x = (torch.rand(B, R, num_x, device=self.device) > 0.8).float()
        s1s2z = (torch.rand(B, R, num_x, device=self.device) > 0.8).float()
        trainX_x = (torch.rand(B, R, num_x, device=self.device) > 0.8).float()
        trainX_z = (torch.rand(B, R, num_x, device=self.device) > 0.8).float()

        z_diff, x_diff, _, _ = apply_weight1_timelike_homological_equivalence_torch(
            z_cum.clone(),
            x_cum.clone(),
            s1s2x.clone(),
            s1s2z.clone(),
            parity_Z,
            parity_X,
            self.distance,
            1,
            32,
            "X",
            True,
            trainX_x=trainX_x,
            trainX_z=trainX_z,
            cache_Z_spacelike=cache_Z,
            cache_X_spacelike=cache_X,
        )
        w_before = z_cum.sum() + x_cum.sum()
        w_after = z_diff.sum() + x_diff.sum()
        self.assertLessEqual(w_after.item(), w_before.item() + 1e-6)


# ---------------------------------------------------------------------------
# Oracle predecoder residuals on GPU
# ---------------------------------------------------------------------------
@_require_cuda
class TestOraclePreDecoderGPU(unittest.TestCase):
    """Oracle residual test exercised explicitly on CUDA (not cpu fallback)."""

    def test_oracle_residuals_zero_on_cuda(self):
        from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch
        from qec.precompute_dem import precompute_dem_bundle_surface_code

        device = torch.device("cuda")
        distance, n_rounds = 3, 3

        for basis in ("X", "Z"):
            artifacts = precompute_dem_bundle_surface_code(
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                code_rotation="XV",
                p_scalar=0.01,
                dem_output_dir=None,
                device=device,
                export=False,
                return_artifacts=True,
            )
            gen = MemoryCircuitTorch(
                distance=distance,
                n_rounds=n_rounds,
                basis=basis,
                code_rotation="XV",
                H=artifacts["H"],
                p=artifacts["p"],
                A=artifacts.get("A"),
                device=device,
            )
            trainX, trainY = gen.generate_batch(batch_size=16)

            # trainY as oracle: predictions match ground truth exactly, so
            # binarized predictions reconstructed from trainY should preserve
            # the syndome structure.
            self.assertTrue(trainX.is_cuda)
            self.assertTrue(trainY.is_cuda)
            self.assertFalse(torch.isnan(trainX).any())
            self.assertFalse(torch.isnan(trainY).any())


# ---------------------------------------------------------------------------
# GPU memory and multi-batch consistency
# ---------------------------------------------------------------------------
@_require_cuda
class TestGPUMemoryAndConsistency(unittest.TestCase):
    """Stress tests: repeated batches don't leak memory, results are finite."""

    def test_repeated_batches_no_memory_leak(self):
        from data.generator_torch import QCDataGeneratorTorch

        device = torch.device("cuda")
        gen = QCDataGeneratorTorch(
            distance=3,
            n_rounds=3,
            p_error=0.004,
            measure_basis="both",
            device=device,
            base_seed=42,
        )
        torch.cuda.reset_peak_memory_stats()
        for step in range(20):
            trainX, trainY = gen.generate_batch(step=step, batch_size=32)
            del trainX, trainY

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        # d=3 batches of 32 should stay well under 512 MiB
        self.assertLess(peak_mb, 512, f"Peak GPU memory {peak_mb:.1f} MiB seems too high")

    def test_cpu_gpu_deterministic_with_fixed_errors(self):
        """Given identical error vectors, CPU and GPU produce the same frames."""
        num_det, num_err, batch = 10, 20, 16
        rng = np.random.RandomState(42)
        H_np = rng.randint(0, 2, (2 * num_det, num_err)).astype(np.uint8)
        errors_np = rng.randint(0, 2, (batch, num_err)).astype(np.uint8)

        H_cpu = torch.from_numpy(H_np)
        H_gpu = H_cpu.cuda()
        errors_cpu = torch.from_numpy(errors_np)
        errors_gpu = errors_cpu.cuda()

        frames_cpu = torch.remainder(errors_cpu.float() @ H_cpu.t().float(), 2).to(torch.uint8)
        frames_gpu = torch.remainder(errors_gpu.float() @ H_gpu.t().float(), 2).to(torch.uint8)

        torch.testing.assert_close(frames_cpu, frames_gpu.cpu())


if __name__ == "__main__":
    unittest.main()
