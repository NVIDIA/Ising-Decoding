#!/usr/bin/env python3
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
Comprehensive tests for surface code boundary detectors.

Tests:
1. Basic functionality (detector counts, DEM building)
2. Both X and Z basis
3. Multiple distances
4. Different noise models (simple DEPOLARIZE vs NoiseModel/PAULI_CHANNEL_2)
5. LER comparison with and without boundary detectors
6. Deterministic detector verification
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np


# In CI (e.g. GitLab) use fewer samples so unit-tests job stays ~5 min. Local/full runs keep full counts.
def _ler_test_samples(full_count: int, ci_count: int = 2000) -> int:
    return ci_count if os.environ.get("CI") == "true" else full_count


_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import stim
import pymatching
from qec.surface_code.memory_circuit import MemoryCircuit, SurfaceCode
from qec.noise_model import NoiseModel


class TestBoundaryDetectorBasics(unittest.TestCase):
    """Test basic boundary detector functionality."""

    def test_detector_count_d3(self):
        """Test that boundary detectors add correct number of detectors for d=3."""
        # d=3 has 4 X-stabilizers and 4 Z-stabilizers
        mc_no_bd = MemoryCircuit(
            distance=3,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=3,
            basis='X',
            add_boundary_detectors=False
        )
        mc_with_bd = MemoryCircuit(
            distance=3,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=3,
            basis='X',
            add_boundary_detectors=True
        )

        # For d=3, X-basis: should add 4 boundary detectors (one per X-stabilizer)
        added = mc_with_bd.stim_circuit.num_detectors - mc_no_bd.stim_circuit.num_detectors
        self.assertEqual(added, 4, f"Expected 4 boundary detectors for d=3 X-basis, got {added}")

    def test_detector_count_d5(self):
        """Test that boundary detectors add correct number of detectors for d=5."""
        # d=5 has 12 X-stabilizers and 12 Z-stabilizers
        mc_no_bd = MemoryCircuit(
            distance=5,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=5,
            basis='X',
            add_boundary_detectors=False
        )
        mc_with_bd = MemoryCircuit(
            distance=5,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=5,
            basis='X',
            add_boundary_detectors=True
        )

        # For d=5, X-basis: should add 12 boundary detectors (one per X-stabilizer)
        added = mc_with_bd.stim_circuit.num_detectors - mc_no_bd.stim_circuit.num_detectors
        self.assertEqual(added, 12, f"Expected 12 boundary detectors for d=5 X-basis, got {added}")

    def test_detector_count_z_basis(self):
        """Test boundary detectors for Z-basis memory."""
        mc_no_bd = MemoryCircuit(
            distance=3,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=3,
            basis='Z',
            add_boundary_detectors=False
        )
        mc_with_bd = MemoryCircuit(
            distance=3,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=3,
            basis='Z',
            add_boundary_detectors=True
        )

        # For d=3, Z-basis: should add 4 boundary detectors (one per Z-stabilizer)
        added = mc_with_bd.stim_circuit.num_detectors - mc_no_bd.stim_circuit.num_detectors
        self.assertEqual(added, 4, f"Expected 4 boundary detectors for d=3 Z-basis, got {added}")


class TestDEMBuilding(unittest.TestCase):
    """Test that DEM builds correctly with boundary detectors."""

    def test_dem_builds_simple_noise(self):
        """Test DEM builds with simple DEPOLARIZE noise."""
        mc = MemoryCircuit(
            distance=3,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=3,
            basis='X',
            add_boundary_detectors=True
        )

        # Should not raise
        dem = mc.stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )

        self.assertEqual(dem.num_detectors, mc.stim_circuit.num_detectors)
        self.assertGreater(dem.num_errors, 0)

    def test_dem_builds_noise_model(self):
        """Test DEM builds with NoiseModel (PAULI_CHANNEL_2)."""
        noise_model = NoiseModel.from_single_p(0.01)

        mc = MemoryCircuit(
            distance=3,
            idle_error=0.01,
            sqgate_error=0.01,
            tqgate_error=0.01,
            spam_error=0.01,
            n_rounds=3,
            basis='X',
            add_boundary_detectors=True,
            noise_model=noise_model
        )

        # Should not raise
        dem = mc.stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )

        self.assertEqual(dem.num_detectors, mc.stim_circuit.num_detectors)
        self.assertGreater(dem.num_errors, 0)

    def test_no_nondeterministic_detectors(self):
        """Test that boundary detectors don't create non-deterministic detectors."""
        for d in [3, 5, 7]:
            for basis in ['X', 'Z']:
                mc = MemoryCircuit(
                    distance=d,
                    idle_error=0.01,
                    sqgate_error=0.01,
                    tqgate_error=0.01,
                    spam_error=0.01,
                    n_rounds=d,
                    basis=basis,
                    add_boundary_detectors=True
                )

                try:
                    dem = mc.stim_circuit.detector_error_model(
                        decompose_errors=True, approximate_disjoint_errors=True
                    )
                except ValueError as e:
                    if "non-deterministic" in str(e):
                        self.fail(f"Non-deterministic detector for d={d}, basis={basis}: {e}")
                    raise


class TestDecodingCorrectness(unittest.TestCase):
    """Test that decoding works correctly with boundary detectors."""

    def test_pymatching_decodes_simple_noise(self):
        """Test PyMatching can decode with boundary detectors (simple noise)."""
        mc = MemoryCircuit(
            distance=5,
            idle_error=0.001,
            sqgate_error=0.001,
            tqgate_error=0.001,
            spam_error=0.001,
            n_rounds=5,
            basis='X',
            add_boundary_detectors=True
        )

        dem = mc.stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = mc.stim_circuit.compile_detector_sampler()

        # Sample and decode
        samples, obs = sampler.sample(1000, separate_observables=True)
        predictions = matcher.decode_batch(samples)

        # Should have some successful decodes (not all errors)
        errors = np.sum(predictions != obs)
        self.assertLess(errors, 1000, "All samples had errors - something is wrong")

    def test_pymatching_decodes_noise_model(self):
        """Test PyMatching can decode with boundary detectors (NoiseModel)."""
        noise_model = NoiseModel.from_single_p(0.001)

        mc = MemoryCircuit(
            distance=5,
            idle_error=0.001,
            sqgate_error=0.001,
            tqgate_error=0.001,
            spam_error=0.001,
            n_rounds=5,
            basis='X',
            add_boundary_detectors=True,
            noise_model=noise_model
        )

        dem = mc.stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = mc.stim_circuit.compile_detector_sampler()

        # Sample and decode
        samples, obs = sampler.sample(1000, separate_observables=True)
        predictions = matcher.decode_batch(samples)

        # Should have some successful decodes
        errors = np.sum(predictions != obs)
        self.assertLess(errors, 1000, "All samples had errors - something is wrong")


class TestLERComparison(unittest.TestCase):
    """Test LER behavior with and without boundary detectors."""

    def test_ler_improves_with_bd_noise_model(self):
        """Test that LER improves with boundary detectors when using NoiseModel."""
        noise_model = NoiseModel.from_single_p(0.002)
        num_samples = _ler_test_samples(50000, 20000)

        # Circuit WITHOUT boundary detectors
        mc_no_bd = MemoryCircuit(
            distance=5,
            idle_error=0.002,
            sqgate_error=0.002,
            tqgate_error=0.002,
            spam_error=0.002,
            n_rounds=5,
            basis='X',
            add_boundary_detectors=False,
            noise_model=noise_model
        )

        dem_no_bd = mc_no_bd.stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        matcher_no_bd = pymatching.Matching.from_detector_error_model(dem_no_bd)
        sampler_no_bd = mc_no_bd.stim_circuit.compile_detector_sampler()
        samples_no_bd, obs_no_bd = sampler_no_bd.sample(num_samples, separate_observables=True)
        pred_no_bd = matcher_no_bd.decode_batch(samples_no_bd)
        ler_no_bd = np.sum(pred_no_bd != obs_no_bd) / num_samples

        # Circuit WITH boundary detectors
        mc_with_bd = MemoryCircuit(
            distance=5,
            idle_error=0.002,
            sqgate_error=0.002,
            tqgate_error=0.002,
            spam_error=0.002,
            n_rounds=5,
            basis='X',
            add_boundary_detectors=True,
            noise_model=noise_model
        )

        dem_with_bd = mc_with_bd.stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        matcher_with_bd = pymatching.Matching.from_detector_error_model(dem_with_bd)
        sampler_with_bd = mc_with_bd.stim_circuit.compile_detector_sampler()
        samples_with_bd, obs_with_bd = sampler_with_bd.sample(
            num_samples, separate_observables=True
        )
        pred_with_bd = matcher_with_bd.decode_batch(samples_with_bd)
        ler_with_bd = np.sum(pred_with_bd != obs_with_bd) / num_samples

        print(f"\nLER with NoiseModel (d=5, p=0.002, {num_samples} samples):")
        print(f"  Without BD: {ler_no_bd:.4e}")
        print(f"  With BD:    {ler_with_bd:.4e}")
        ratio = (ler_no_bd / ler_with_bd) if ler_with_bd > 0 else float("inf")
        print(f"  Improvement: {ratio:.2f}x")

        # With NoiseModel, boundary detectors should improve LER
        self.assertLess(
            ler_with_bd, ler_no_bd,
            f"Expected LER to improve with BD: {ler_with_bd:.4e} >= {ler_no_bd:.4e}"
        )

    def test_ler_improves_with_bd_all_orientations(self):
        """Test LER improves with boundary detectors for all four orientations (short run)."""
        noise_model = NoiseModel.from_single_p(0.005)
        num_samples = _ler_test_samples(10000, 10000)
        d = 5
        for rotation in TestCodeRotations.ROTATIONS:
            with self.subTest(rotation=rotation):
                mc_no_bd = MemoryCircuit(
                    distance=d,
                    idle_error=0.005,
                    sqgate_error=0.005,
                    tqgate_error=0.005,
                    spam_error=0.005,
                    n_rounds=d,
                    basis='X',
                    add_boundary_detectors=False,
                    noise_model=noise_model,
                    code_rotation=rotation
                )
                dem_no_bd = mc_no_bd.stim_circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
                matcher_no_bd = pymatching.Matching.from_detector_error_model(dem_no_bd)
                sampler_no_bd = mc_no_bd.stim_circuit.compile_detector_sampler()
                samples_no_bd, obs_no_bd = sampler_no_bd.sample(
                    num_samples, separate_observables=True
                )
                pred_no_bd = matcher_no_bd.decode_batch(samples_no_bd)
                ler_no_bd = np.sum(pred_no_bd != obs_no_bd) / num_samples
                mc_with_bd = MemoryCircuit(
                    distance=d,
                    idle_error=0.005,
                    sqgate_error=0.005,
                    tqgate_error=0.005,
                    spam_error=0.005,
                    n_rounds=d,
                    basis='X',
                    add_boundary_detectors=True,
                    noise_model=noise_model,
                    code_rotation=rotation
                )
                dem_with_bd = mc_with_bd.stim_circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
                matcher_with_bd = pymatching.Matching.from_detector_error_model(dem_with_bd)
                sampler_with_bd = mc_with_bd.stim_circuit.compile_detector_sampler()
                samples_with_bd, obs_with_bd = sampler_with_bd.sample(
                    num_samples, separate_observables=True
                )
                pred_with_bd = matcher_with_bd.decode_batch(samples_with_bd)
                ler_with_bd = np.sum(pred_with_bd != obs_with_bd) / num_samples
                self.assertLessEqual(
                    ler_with_bd, ler_no_bd,
                    f"rotation={rotation}: expected LER with BD <= without BD; got {ler_with_bd:.4e} > {ler_no_bd:.4e}"
                )


class TestMultipleDistances(unittest.TestCase):
    """Test boundary detectors work correctly across different distances."""

    def test_various_distances(self):
        """Test boundary detectors for d=3,5,7,9."""
        for d in [3, 5, 7, 9]:
            with self.subTest(distance=d):
                mc = MemoryCircuit(
                    distance=d,
                    idle_error=0.01,
                    sqgate_error=0.01,
                    tqgate_error=0.01,
                    spam_error=0.01,
                    n_rounds=d,
                    basis='X',
                    add_boundary_detectors=True
                )

                # Verify DEM builds
                dem = mc.stim_circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )

                # Verify correct number of boundary detectors added
                num_x_stab = (d * d - 1) // 2

                mc_no_bd = MemoryCircuit(
                    distance=d,
                    idle_error=0.01,
                    sqgate_error=0.01,
                    tqgate_error=0.01,
                    spam_error=0.01,
                    n_rounds=d,
                    basis='X',
                    add_boundary_detectors=False
                )

                added = mc.stim_circuit.num_detectors - mc_no_bd.stim_circuit.num_detectors
                self.assertEqual(
                    added, num_x_stab,
                    f"Expected {num_x_stab} boundary detectors for d={d}, got {added}"
                )


class TestCodeRotations(unittest.TestCase):
    """Test boundary detectors work with all four code orientations (O1=XV, O2=XH, O3=ZV, O4=ZH)."""

    ROTATIONS = ('XV', 'XH', 'ZV', 'ZH')

    def test_all_rotations_dem_builds(self):
        """Test DEM builds for all four code rotations."""
        for rotation in self.ROTATIONS:
            with self.subTest(rotation=rotation):
                mc = MemoryCircuit(
                    distance=3,
                    idle_error=0.01,
                    sqgate_error=0.01,
                    tqgate_error=0.01,
                    spam_error=0.01,
                    n_rounds=3,
                    basis='X',
                    add_boundary_detectors=True,
                    code_rotation=rotation
                )
                dem = mc.stim_circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
                self.assertGreater(dem.num_detectors, 0)

    def test_all_rotations_detector_count_consistent(self):
        """Test boundary detector count is consistent across all four orientations (d=3, X and Z basis)."""
        d = 3
        num_x_stab = (d * d - 1) // 2
        num_z_stab = (d * d - 1) // 2
        for rotation in self.ROTATIONS:
            for basis in ('X', 'Z'):
                with self.subTest(rotation=rotation, basis=basis):
                    mc_no_bd = MemoryCircuit(
                        distance=d,
                        idle_error=0.01,
                        sqgate_error=0.01,
                        tqgate_error=0.01,
                        spam_error=0.01,
                        n_rounds=d,
                        basis=basis,
                        add_boundary_detectors=False,
                        code_rotation=rotation
                    )
                    mc_bd = MemoryCircuit(
                        distance=d,
                        idle_error=0.01,
                        sqgate_error=0.01,
                        tqgate_error=0.01,
                        spam_error=0.01,
                        n_rounds=d,
                        basis=basis,
                        add_boundary_detectors=True,
                        code_rotation=rotation
                    )
                    added = mc_bd.stim_circuit.num_detectors - mc_no_bd.stim_circuit.num_detectors
                    expected = num_x_stab if basis == 'X' else num_z_stab
                    self.assertEqual(
                        added, expected,
                        f"rotation={rotation} basis={basis}: added {added} != {expected}"
                    )


def run_quick_tests():
    """Run a quick subset of tests for development."""
    suite = unittest.TestSuite()

    # Add quick tests
    suite.addTest(TestBoundaryDetectorBasics('test_detector_count_d3'))
    suite.addTest(TestBoundaryDetectorBasics('test_detector_count_d5'))
    suite.addTest(TestDEMBuilding('test_dem_builds_simple_noise'))
    suite.addTest(TestDEMBuilding('test_no_nondeterministic_detectors'))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryDetectorBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestDEMBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestDecodingCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestLERComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestMultipleDistances))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeRotations))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    args = parser.parse_args()

    if args.quick:
        result = run_quick_tests()
    else:
        result = run_all_tests()

    sys.exit(0 if result.wasSuccessful() else 1)
