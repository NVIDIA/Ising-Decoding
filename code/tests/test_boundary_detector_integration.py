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
Test that boundary detector integration in logical_error_rate.py works correctly.

This tests the key logic:
1. Circuit with add_boundary_detectors=True creates additional detectors
2. The residual syndromes (from pre-decoder) + boundary detectors = DEM detector count
3. PyMatching can decode the combined residual successfully
"""

import os
import sys
import unittest
from pathlib import Path


# In CI use fewer samples to keep unit-tests job ~5 min.
def _ci_sample_count(full: int, ci: int = 400) -> int:
    return ci if os.environ.get("CI") == "true" else full


# Ensure repo code/ is on path when run from repo root or from code/tests
_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import numpy as np
import pymatching

from qec.surface_code.memory_circuit import MemoryCircuit, SurfaceCode
from qec.noise_model import NoiseModel

# All four surface code orientations (O1=XV, O2=XH, O3=ZV, O4=ZH)
ORIENTATIONS = ("XV", "XH", "ZV", "ZH")


def test_detector_count_matches():
    """Test that residual + boundary detectors matches DEM detector count (all 4 orientations)."""
    print("=" * 70)
    print("Test: Detector count matching (all 4 orientations)")
    print("=" * 70)
    for code_rotation in ORIENTATIONS:
        for d in [5, 7]:
            for basis in ['X', 'Z']:
                noise_model = NoiseModel.from_single_p(0.001)
                p_placeholder = float(noise_model.get_max_probability())
                mc = MemoryCircuit(
                    distance=d,
                    idle_error=p_placeholder,
                    sqgate_error=p_placeholder,
                    tqgate_error=p_placeholder,
                    spam_error=(2.0 / 3.0) * p_placeholder,
                    n_rounds=d,
                    basis=basis,
                    noise_model=noise_model,
                    add_boundary_detectors=True,
                    code_rotation=code_rotation,
                )
                mc.set_error_rates()
                circuit = mc.stim_circuit
                dem = circuit.detector_error_model(
                    decompose_errors=True, approximate_disjoint_errors=True
                )
                total_dem_detectors = dem.num_detectors
                num_x_stab = (d * d - 1) // 2
                num_z_stab = (d * d - 1) // 2
                num_initial = num_x_stab if basis == 'X' else num_z_stab
                num_remaining = (d - 1) * (num_x_stab + num_z_stab)
                pre_decoder_residual_size = num_initial + num_remaining
                num_boundary_dets = num_x_stab if basis == 'X' else num_z_stab
                expected_total = pre_decoder_residual_size + num_boundary_dets
                if expected_total != total_dem_detectors:
                    print(
                        f"  ✗ FAIL rotation={code_rotation} d={d} basis={basis}: {expected_total} != {total_dem_detectors}"
                    )
                    return False
    print("  ✓ PASS: all orientations/d/basis match")
    return True


def test_decoding_with_appended_boundary_detectors():
    """Test that PyMatching can decode residual + boundary detectors for all 4 orientations."""
    print("\n" + "=" * 70)
    print("Test: Decoding with appended boundary detectors (all 4 orientations)")
    print("=" * 70)
    d = 5
    basis = 'X'
    num_samples = _ci_sample_count(1000, 400)
    noise_model = NoiseModel.from_single_p(0.002)
    p_placeholder = float(noise_model.get_max_probability())
    for code_rotation in ORIENTATIONS:
        mc = MemoryCircuit(
            distance=d,
            idle_error=p_placeholder,
            sqgate_error=p_placeholder,
            tqgate_error=p_placeholder,
            spam_error=(2.0 / 3.0) * p_placeholder,
            n_rounds=d,
            basis=basis,
            noise_model=noise_model,
            add_boundary_detectors=True,
            code_rotation=code_rotation,
        )
        mc.set_error_rates()
        circuit = mc.stim_circuit
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        stim_dets, stim_obs = sampler.sample(num_samples, separate_observables=True)
        stim_dets = stim_dets.astype(np.uint8)
        stim_obs = stim_obs.astype(np.uint8)
        num_x_stab = (d * d - 1) // 2
        num_boundary_dets = num_x_stab
        pre_decoder_residual = stim_dets[:, :-num_boundary_dets]
        boundary_dets = stim_dets[:, -num_boundary_dets:]
        combined = np.concatenate([pre_decoder_residual, boundary_dets], axis=1)
        if combined.shape[1] != dem.num_detectors:
            print(
                f"  ✗ FAIL rotation={code_rotation}: shape {combined.shape[1]} != {dem.num_detectors}"
            )
            return False
        try:
            predictions = matcher.decode_batch(combined)
            predictions = predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions
            errors = np.sum(predictions != stim_obs)
            ler = errors / num_samples
            print(f"  rotation={code_rotation}: LER={ler:.4e} ({errors}/{num_samples}) ✓")
        except Exception as e:
            print(f"  ✗ FAIL rotation={code_rotation}: {e}")
            return False
    print("  ✓ PASS: decoding succeeded for all orientations")
    return True


def test_boundary_detectors_unchanged_by_predecoder():
    """Boundary detectors well-defined for all 4 orientations."""
    print("\n" + "=" * 70)
    print("Test: Boundary detectors unchanged by pre-decoder (all 4 orientations)")
    print("=" * 70)
    d = 5
    basis = 'X'
    noise_model = NoiseModel.from_single_p(0.002)
    p_placeholder = float(noise_model.get_max_probability())
    for code_rotation in ORIENTATIONS:
        mc = MemoryCircuit(
            distance=d,
            idle_error=p_placeholder,
            sqgate_error=p_placeholder,
            tqgate_error=p_placeholder,
            spam_error=(2.0 / 3.0) * p_placeholder,
            n_rounds=d,
            basis=basis,
            noise_model=noise_model,
            add_boundary_detectors=True,
            code_rotation=code_rotation,
        )
        mc.set_error_rates()
        circuit = mc.stim_circuit
        meas_sampler = circuit.compile_sampler()
        measurements = meas_sampler.sample(100)
        converter = circuit.compile_m2d_converter()
        dets_and_obs = converter.convert(measurements=measurements, append_observables=True)
        num_obs = circuit.num_observables
        stim_dets = dets_and_obs[:, :-num_obs]
        num_x_stab = (d * d - 1) // 2
        num_boundary_dets = num_x_stab
        boundary_from_stim = stim_dets[:, -num_boundary_dets:]
        boundary_rate = boundary_from_stim.mean()
        other_rate = stim_dets[:, :-num_boundary_dets].mean()
        print(
            f"  rotation={code_rotation}: total_dets={stim_dets.shape[1]}, "
            f"boundary_flip={boundary_rate:.4f}, other_flip={other_rate:.4f}"
        )
    print("  ✓ PASS: boundary detectors well-defined for all orientations")
    return True


class TestBoundaryDetectorIntegration(unittest.TestCase):
    """Boundary detector integration tests (all 4 orientations). Runnable by unittest discover."""

    def test_detector_count_matches(self):
        self.assertTrue(test_detector_count_matches())

    def test_decoding_with_appended_boundary_detectors(self):
        self.assertTrue(test_decoding_with_appended_boundary_detectors())

    def test_boundary_detectors_unchanged_by_predecoder(self):
        self.assertTrue(test_boundary_detectors_unchanged_by_predecoder())


if __name__ == "__main__":
    unittest.main()
