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
Comprehensive tests for Color Code Homological Equivalence

Tests the following functionality:
1. Weight reduction rules for weight-6 (bulk) and weight-4 (boundary) plaquettes
2. Fix equivalence rules for weight-3 (bulk) and weight-2 (boundary) errors
3. Simplify convergence
4. apply_spacelike_homological_equivalence interface

Author: AI Assistant
"""

import unittest
import sys
from pathlib import Path
import torch

# Ensure imports work when running via unittest discovery
sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.color_code import ColorCode
from qec.color_code.homological_equivalence import (
    ColorCodeHE,
    weight_reduction,
    fix_equivalence_weight6,
    fix_equivalence_weight4,
    apply_spacelike_homological_equivalence,
    apply_spacelike_homological_equivalence_batched,
)


class TestColorCodeHESetup(unittest.TestCase):
    """Test that ColorCodeHE initializes correctly for different distances."""

    def test_initialization_d3(self):
        """Test HE initialization for d=3."""
        cc = ColorCode(3)
        he = ColorCodeHE(cc)

        # d=3: 7 data qubits, 3 plaquettes
        self.assertEqual(cc.num_data, 7)
        self.assertEqual(cc.num_plaquettes, 3)
        self.assertEqual(len(he.plaquettes), 3)

        # Check plaquette info is populated correctly
        for i, plaq in enumerate(he.plaquettes):
            self.assertIn('weight', plaq)
            self.assertIn('labels', plaq)
            self.assertIn('data_qubits', plaq)
            self.assertIn(plaq['weight'], [4, 6])

    def test_initialization_d5(self):
        """Test HE initialization for d=5."""
        cc = ColorCode(5)
        he = ColorCodeHE(cc)

        # d=5: 19 data qubits, 9 plaquettes
        self.assertEqual(cc.num_data, 19)
        self.assertEqual(cc.num_plaquettes, 9)
        self.assertEqual(len(he.plaquettes), 9)

    def test_initialization_d7(self):
        """Test HE initialization for d=7."""
        cc = ColorCode(7)
        he = ColorCodeHE(cc)

        # d=7: 37 data qubits, 18 plaquettes
        self.assertEqual(cc.num_data, 37)
        self.assertEqual(cc.num_plaquettes, 18)
        self.assertEqual(len(he.plaquettes), 18)


class TestWeightReductionWeight6(unittest.TestCase):
    """Test weight reduction for weight-6 (bulk) plaquettes."""

    def setUp(self):
        self.cc = ColorCode(5)  # d=5 has both weight-4 and weight-6 plaquettes
        self.he = ColorCodeHE(self.cc)

        # Find a weight-6 plaquette
        self.weight6_plaq = None
        for i, plaq in enumerate(self.he.plaquettes):
            if plaq['weight'] == 6:
                self.weight6_plaq = plaq
                self.weight6_idx = i
                break

        self.assertIsNotNone(self.weight6_plaq, "No weight-6 plaquette found in d=5")

    def test_weight6_to_weight0(self):
        """Weight-6 error on weight-6 plaquette should be removed (stabilizer)."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in self.weight6_plaq['data_qubits']:
            error[q] = 1

        self.assertEqual(error.sum().item(), 6)

        # Use standalone function with plaquette support and weight
        result = weight_reduction(error, self.weight6_plaq['data_qubits'], 6)

        self.assertEqual(result.sum().item(), 0, "Weight-6 error should be reduced to 0")

    def test_weight5_to_weight1(self):
        """Weight-5 error on weight-6 plaquette should be reduced to weight-1."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        # Put 5 errors on the plaquette (all except one)
        for q in self.weight6_plaq['data_qubits'][:5]:
            error[q] = 1

        self.assertEqual(error.sum().item(), 5)

        result = weight_reduction(error, self.weight6_plaq['data_qubits'], 6)

        self.assertEqual(result.sum().item(), 1, "Weight-5 error should be reduced to 1")

    def test_weight4_to_weight2(self):
        """Weight-4 error on weight-6 plaquette should be reduced to weight-2."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        # Put 4 errors on the plaquette
        for q in self.weight6_plaq['data_qubits'][:4]:
            error[q] = 1

        self.assertEqual(error.sum().item(), 4)

        result = weight_reduction(error, self.weight6_plaq['data_qubits'], 6)

        self.assertEqual(result.sum().item(), 2, "Weight-4 error should be reduced to 2")

    def test_weight3_unchanged(self):
        """Weight-3 error should NOT be reduced by weight_reduction (handled by fix_equivalence)."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        # Put 3 errors on the plaquette
        labels = self.weight6_plaq['labels']
        error[labels['q1']] = 1
        error[labels['q2']] = 1
        error[labels['q3']] = 1

        self.assertEqual(error.sum().item(), 3)

        result = weight_reduction(error, self.weight6_plaq['data_qubits'], 6)

        # Weight reduction does NOT change weight-3 errors
        self.assertEqual(
            result.sum().item(), 3, "Weight-3 error should NOT be reduced by weight_reduction"
        )


class TestWeightReductionWeight4(unittest.TestCase):
    """Test weight reduction for weight-4 (boundary) plaquettes."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

        # Find a weight-4 plaquette
        self.weight4_plaq = None
        for i, plaq in enumerate(self.he.plaquettes):
            if plaq['weight'] == 4:
                self.weight4_plaq = plaq
                self.weight4_idx = i
                break

        self.assertIsNotNone(self.weight4_plaq, "No weight-4 plaquette found in d=5")

    def test_weight4_to_weight0(self):
        """Weight-4 error on weight-4 plaquette should be removed."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in self.weight4_plaq['data_qubits']:
            error[q] = 1

        self.assertEqual(error.sum().item(), 4)

        result = weight_reduction(error, self.weight4_plaq['data_qubits'], 4)

        self.assertEqual(
            result.sum().item(), 0, "Weight-4 error on weight-4 plaquette should be reduced to 0"
        )

    def test_weight3_to_weight1(self):
        """Weight-3 error on weight-4 plaquette should be reduced to weight-1."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        # Put 3 errors on the plaquette
        for q in self.weight4_plaq['data_qubits'][:3]:
            error[q] = 1

        self.assertEqual(error.sum().item(), 3)

        result = weight_reduction(error, self.weight4_plaq['data_qubits'], 4)

        self.assertEqual(
            result.sum().item(), 1, "Weight-3 error on weight-4 plaquette should be reduced to 1"
        )


class TestFixEquivalenceWeight6(unittest.TestCase):
    """Test fix_equivalence for weight-3 errors on weight-6 plaquettes."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

        # Find a weight-6 plaquette
        self.weight6_plaq = None
        for i, plaq in enumerate(self.he.plaquettes):
            if plaq['weight'] == 6:
                self.weight6_plaq = plaq
                break

        self.assertIsNotNone(self.weight6_plaq)
        self.labels = self.weight6_plaq['labels']

    def _create_error(self, qubits):
        """Helper to create an error on specified qubit labels."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in qubits:
            error[self.labels[q]] = 1
        return error

    def _get_error_qubits(self, error):
        """Helper to get label names for qubits with errors."""
        result = []
        for name, idx in self.labels.items():
            if error[idx] == 1:
                result.append(name)
        return sorted(result)

    def test_q1_q2_q3_maps_to_q4_q5_q6(self):
        """Rule 1: (q1, q2, q3) -> (q4, q5, q6)"""
        error = self._create_error(['q1', 'q2', 'q3'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q4', 'q5', 'q6'])

    def test_q1_q2_q4_maps_to_q3_q5_q6(self):
        """Rule 2: (q1, q2, q4) -> (q3, q5, q6)"""
        error = self._create_error(['q1', 'q2', 'q4'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q3', 'q5', 'q6'])

    def test_q1_q3_q4_maps_to_q2_q5_q6(self):
        """Rule 3: (q1, q3, q4) -> (q2, q5, q6)"""
        error = self._create_error(['q1', 'q3', 'q4'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q2', 'q5', 'q6'])

    def test_q1_q5_q6_maps_to_q2_q3_q4(self):
        """Rule 10: (q1, q5, q6) -> (q2, q3, q4)"""
        error = self._create_error(['q1', 'q5', 'q6'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q2', 'q3', 'q4'])

    def test_q1_q2_q5_maps_to_q3_q4_q6(self):
        """Rule 5: (q1, q2, q5) -> (q3, q4, q6)"""
        error = self._create_error(['q1', 'q2', 'q5'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q3', 'q4', 'q6'])

    def test_q1_q3_q5_maps_to_q2_q4_q6(self):
        """Rule 6: (q1, q3, q5) -> (q2, q4, q6)"""
        error = self._create_error(['q1', 'q3', 'q5'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q2', 'q4', 'q6'])

    def test_q1_q4_q6_maps_to_q2_q3_q5(self):
        """Rule 8: (q1, q4, q6) -> (q2, q3, q5)"""
        error = self._create_error(['q1', 'q4', 'q6'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q2', 'q3', 'q5'])

    def test_q1_q4_q5_maps_to_q2_q3_q6(self):
        """Rule 8: (q1, q4, q5) -> (q2, q3, q6)"""
        error = self._create_error(['q1', 'q4', 'q5'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q2', 'q3', 'q6'])

    def test_q2_q4_q5_maps_to_q1_q3_q6(self):
        """Rule 9: (q2, q4, q5) -> (q1, q3, q6)"""
        error = self._create_error(['q2', 'q4', 'q5'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q1', 'q3', 'q6'])

    def test_q1_q2_q6_maps_to_q3_q4_q5(self):
        """Rule 4: (q1, q2, q6) -> (q3, q4, q5)"""
        error = self._create_error(['q1', 'q2', 'q6'])
        result = fix_equivalence_weight6(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q3', 'q4', 'q5'])

    def test_canonical_patterns_unchanged(self):
        """Canonical patterns (the rules' right-hand sides) should remain unchanged."""
        canonical_patterns = [
            ['q4', 'q5', 'q6'],
            ['q3', 'q5', 'q6'],
            ['q2', 'q5', 'q6'],
            ['q3', 'q4', 'q5'],
            ['q2', 'q3', 'q6'],
            ['q2', 'q4', 'q6'],
            ['q3', 'q4', 'q6'],
            ['q2', 'q3', 'q5'],
            ['q1', 'q3', 'q6'],
            ['q2', 'q3', 'q4'],
        ]

        for pattern in canonical_patterns:
            error = self._create_error(pattern)
            result = fix_equivalence_weight6(error, self.labels)
            result_qubits = self._get_error_qubits(result)
            self.assertEqual(
                result_qubits, sorted(pattern), f"Canonical pattern {pattern} should be unchanged"
            )


class TestFixEquivalenceWeight4(unittest.TestCase):
    """Test fix_equivalence for weight-2 errors on weight-4 plaquettes."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

        # Find a weight-4 plaquette
        self.weight4_plaq = None
        for i, plaq in enumerate(self.he.plaquettes):
            if plaq['weight'] == 4:
                self.weight4_plaq = plaq
                break

        self.assertIsNotNone(self.weight4_plaq)
        self.labels = self.weight4_plaq['labels']

    def _create_error(self, qubits):
        """Helper to create an error on specified qubit labels."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in qubits:
            error[self.labels[q]] = 1
        return error

    def _get_error_qubits(self, error):
        """Helper to get label names for qubits with errors."""
        result = []
        for name, idx in self.labels.items():
            if error[idx] == 1:
                result.append(name)
        return sorted(result)

    def test_q1_q2_maps_to_q5_q6(self):
        """Rule 1: (q1, q2) -> (q5, q6)"""
        error = self._create_error(['q1', 'q2'])
        result = fix_equivalence_weight4(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q5', 'q6'])

    def test_q1_q5_maps_to_q2_q6(self):
        """Rule 2: (q1, q5) -> (q2, q6)"""
        error = self._create_error(['q1', 'q5'])
        result = fix_equivalence_weight4(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q2', 'q6'])

    def test_q2_q5_maps_to_q1_q6(self):
        """Rule 3: (q2, q5) -> (q1, q6)"""
        error = self._create_error(['q2', 'q5'])
        result = fix_equivalence_weight4(error, self.labels)

        result_qubits = self._get_error_qubits(result)
        self.assertEqual(result_qubits, ['q1', 'q6'])

    def test_canonical_patterns_unchanged(self):
        """Canonical patterns (containing q6) should remain unchanged."""
        canonical_patterns = [
            ['q5', 'q6'],
            ['q2', 'q6'],
            ['q1', 'q6'],
        ]

        for pattern in canonical_patterns:
            error = self._create_error(pattern)
            result = fix_equivalence_weight4(error, self.labels)
            result_qubits = self._get_error_qubits(result)
            self.assertEqual(
                result_qubits, sorted(pattern), f"Canonical pattern {pattern} should be unchanged"
            )


class TestSimplifyConvergence(unittest.TestCase):
    """Test that simplify converges and produces canonical forms."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

    def test_simplify_weight6_to_0(self):
        """Weight-6 error should simplify to 0."""
        # Find weight-6 plaquette
        for plaq in self.he.plaquettes:
            if plaq['weight'] == 6:
                break

        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in plaq['data_qubits']:
            error[q] = 1

        result = self.he.simplify(error)
        self.assertEqual(result.sum().item(), 0)

    def test_simplify_weight5_to_1(self):
        """Weight-5 error should simplify to weight-1."""
        for plaq in self.he.plaquettes:
            if plaq['weight'] == 6:
                break

        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in plaq['data_qubits'][:5]:
            error[q] = 1

        result = self.he.simplify(error)
        self.assertEqual(result.sum().item(), 1)

    def test_simplify_idempotent(self):
        """Applying simplify twice should give the same result."""
        for plaq in self.he.plaquettes:
            if plaq['weight'] == 6:
                break

        labels = plaq['labels']
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        error[labels['q1']] = 1
        error[labels['q2']] = 1
        error[labels['q3']] = 1

        result1 = self.he.simplify(error)
        result2 = self.he.simplify(result1)

        self.assertTrue(torch.equal(result1, result2), "simplify should be idempotent")

    def test_simplify_with_count(self):
        """Test simplify_with_count returns correct iteration count."""
        for plaq in self.he.plaquettes:
            if plaq['weight'] == 6:
                break

        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        for q in plaq['data_qubits']:
            error[q] = 1

        result, iters = self.he.simplify_with_count(error)

        self.assertEqual(result.sum().item(), 0)
        self.assertGreaterEqual(iters, 1)


class TestApplySpacelikeHE(unittest.TestCase):
    """Test the main apply_spacelike_homological_equivalence interface."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

        # Find plaquettes
        self.weight6_plaq = None
        self.weight4_plaq = None
        for plaq in self.he.plaquettes:
            if plaq['weight'] == 6 and self.weight6_plaq is None:
                self.weight6_plaq = plaq
            if plaq['weight'] == 4 and self.weight4_plaq is None:
                self.weight4_plaq = plaq

    def test_basic_diff_processing(self):
        """Test that diffs are processed independently."""
        n_rounds = 5
        x_diff = torch.zeros(self.cc.num_data, n_rounds, dtype=torch.long)
        z_diff = torch.zeros(self.cc.num_data, n_rounds, dtype=torch.long)

        # Round 0: weight-6 error (should become 0)
        for q in self.weight6_plaq['data_qubits']:
            x_diff[q, 0] = 1

        # Round 2: different error
        labels = self.weight6_plaq['labels']
        x_diff[labels['q1'], 2] = 1
        x_diff[labels['q2'], 2] = 1
        x_diff[labels['q3'], 2] = 1

        x_new, z_new = apply_spacelike_homological_equivalence(x_diff, z_diff, self.he)

        # Round 0 should be empty
        self.assertEqual(x_new[:, 0].sum().item(), 0)

        # Round 2 should have canonical form
        self.assertEqual(x_new[:, 2].sum().item(), 3)  # Still 3 errors, just canonical

        # Rounds 1, 3, 4 should be unchanged (empty)
        self.assertEqual(x_new[:, 1].sum().item(), 0)
        self.assertEqual(x_new[:, 3].sum().item(), 0)
        self.assertEqual(x_new[:, 4].sum().item(), 0)

    def test_z_errors_processed_same_as_x(self):
        """Test that Z errors are processed identically to X (CSS symmetry)."""
        n_rounds = 3
        x_diff = torch.zeros(self.cc.num_data, n_rounds, dtype=torch.long)
        z_diff = torch.zeros(self.cc.num_data, n_rounds, dtype=torch.long)

        # Put same error pattern in both
        for q in self.weight6_plaq['data_qubits']:
            x_diff[q, 0] = 1
            z_diff[q, 0] = 1

        x_new, z_new = apply_spacelike_homological_equivalence(x_diff, z_diff, self.he)

        # Both should be reduced to 0
        self.assertEqual(x_new[:, 0].sum().item(), 0)
        self.assertEqual(z_new[:, 0].sum().item(), 0)


class TestApplySpacelikeHEBatched(unittest.TestCase):
    """Test the batched apply_spacelike_homological_equivalence interface."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

        for plaq in self.he.plaquettes:
            if plaq['weight'] == 6:
                self.weight6_plaq = plaq
                break

    def test_batched_shape(self):
        """Test that batched function preserves shape."""
        batch_size = 4
        n_rounds = 5

        x_diff = torch.zeros(batch_size, n_rounds, self.cc.num_data, dtype=torch.long)
        z_diff = torch.zeros(batch_size, n_rounds, self.cc.num_data, dtype=torch.long)

        x_new, z_new = apply_spacelike_homological_equivalence_batched(x_diff, z_diff, self.he)

        self.assertEqual(x_new.shape, (batch_size, n_rounds, self.cc.num_data))
        self.assertEqual(z_new.shape, (batch_size, n_rounds, self.cc.num_data))

    def test_batched_independent(self):
        """Test that each batch element is processed independently."""
        batch_size = 3
        n_rounds = 2

        x_diff = torch.zeros(batch_size, n_rounds, self.cc.num_data, dtype=torch.long)
        z_diff = torch.zeros(batch_size, n_rounds, self.cc.num_data, dtype=torch.long)

        # Different patterns in each batch
        # Batch 0: weight-6 error
        for q in self.weight6_plaq['data_qubits']:
            x_diff[0, 0, q] = 1

        # Batch 1: weight-5 error
        for q in self.weight6_plaq['data_qubits'][:5]:
            x_diff[1, 0, q] = 1

        # Batch 2: empty

        x_new, z_new = apply_spacelike_homological_equivalence_batched(x_diff, z_diff, self.he)

        # Check each batch independently
        self.assertEqual(x_new[0, 0].sum().item(), 0)  # weight-6 -> 0
        self.assertEqual(x_new[1, 0].sum().item(), 1)  # weight-5 -> 1
        self.assertEqual(x_new[2, 0].sum().item(), 0)  # empty -> empty


class TestAllDistances(unittest.TestCase):
    """Test HE works correctly for all supported distances."""

    def test_all_plaquettes_d3(self):
        """Test HE on all plaquettes for d=3."""
        self._test_all_plaquettes(3)

    def test_all_plaquettes_d5(self):
        """Test HE on all plaquettes for d=5."""
        self._test_all_plaquettes(5)

    def test_all_plaquettes_d7(self):
        """Test HE on all plaquettes for d=7."""
        self._test_all_plaquettes(7)

    def _test_all_plaquettes(self, distance):
        """Helper to test all plaquettes for a given distance."""
        cc = ColorCode(distance)
        he = ColorCodeHE(cc)

        for i, plaq in enumerate(he.plaquettes):
            weight = plaq['weight']

            # Test full weight error -> 0
            error = torch.zeros(cc.num_data, dtype=torch.long)
            for q in plaq['data_qubits']:
                error[q] = 1

            result = he.simplify(error)
            self.assertEqual(
                result.sum().item(), 0,
                f"d={distance}, plaq {i} (weight-{weight}): full weight should reduce to 0"
            )

            # Test weight-1 less than full
            if weight == 6:
                # Weight-5 -> 1
                error = torch.zeros(cc.num_data, dtype=torch.long)
                for q in plaq['data_qubits'][:5]:
                    error[q] = 1
                result = he.simplify(error)
                self.assertEqual(
                    result.sum().item(), 1, f"d={distance}, plaq {i}: weight-5 should reduce to 1"
                )
            elif weight == 4:
                # Weight-3 -> 1
                error = torch.zeros(cc.num_data, dtype=torch.long)
                for q in plaq['data_qubits'][:3]:
                    error[q] = 1
                result = he.simplify(error)
                self.assertEqual(
                    result.sum().item(), 1, f"d={distance}, plaq {i}: weight-3 should reduce to 1"
                )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.cc = ColorCode(5)
        self.he = ColorCodeHE(self.cc)

    def test_empty_error(self):
        """Empty error should remain empty."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        result = self.he.simplify(error)
        self.assertEqual(result.sum().item(), 0)

    def test_single_qubit_error(self):
        """Single qubit error should remain unchanged."""
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        error[0] = 1
        result = self.he.simplify(error)
        self.assertEqual(result.sum().item(), 1)

    def test_error_outside_plaquettes(self):
        """Errors outside any plaquette support should be unchanged."""
        # This tests that we don't accidentally modify unrelated qubits
        error = torch.zeros(self.cc.num_data, dtype=torch.long)
        error[0] = 1  # Single qubit

        original_pos = error.clone()
        result = self.he.simplify(error)

        # Should still have weight 1
        self.assertEqual(result.sum().item(), 1)


if __name__ == '__main__':
    unittest.main()
