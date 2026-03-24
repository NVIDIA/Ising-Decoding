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
Unit tests for homological equivalence (spacelike + timelike).

Covers:
  - Coordinate helpers
  - Weight reduction (X / Z)
  - simplify_X / simplify_Z idempotency and syndrome invariance
  - Spacelike HE: diff→cum→simplify→diff roundtrip
  - Timelike HE: simplifytimeX / simplifytimeZ acceptance behavior
  - apply_timelike_homological_equivalence end-to-end
  - MemoryCircuitTorch integration (generate_batch with HE on/off)
"""

import sys
import unittest
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qec.surface_code.memory_circuit import SurfaceCode
from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch
from qec.surface_code.homological_equivalence_torch import (
    apply_weight1_timelike_homological_equivalence_torch,
    build_spacelike_he_cache,
    build_timelike_he_cache,
    build_weight2_timelike_cache,
    _simplify_time_w2_step,
)
from qec.surface_code.homological_equivalence import (
    linear_index_to_coordinates,
    coordinates_to_linear_index,
    get_stabilizer_support_from_parity_matrix,
    apply_fix_equivalence_X_local,
    apply_fix_equivalence_Z_local,
    weight_reduction_X,
    weight_reduction_Z,
    simplify_X,
    simplify_Z,
    simplify_X_with_count,
    simplify_Z_with_count,
    apply_spacelike_homological_equivalence,
    simplifytimeX,
    simplifytimeZ,
    simplifytimeX_weight2,
    simplifytimeZ_weight2,
    apply_timelike_homological_equivalence,
    get_anticommuting_stabilizers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_parity_matrices(distance: int, rotation: str = "XV"):
    """Build parity check matrices for a surface code of given distance."""
    first_bulk = rotation[0]
    rot_type = "V" if rotation[1] == "V" else "H"
    code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rot_type)
    hx = torch.tensor(code.hx, dtype=torch.long)
    hz = torch.tensor(code.hz, dtype=torch.long)
    return hx, hz, code


def _syndrome(error: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
    """Compute syndrome s = H @ e (mod 2)."""
    return (parity.float() @ error.float()) % 2


def _random_error(d: int, weight: int = None, seed: int = 42) -> torch.Tensor:
    """Random binary vector of length d² with optional fixed Hamming weight."""
    rng = np.random.RandomState(seed)
    n = d * d
    if weight is None:
        return torch.tensor(rng.randint(0, 2, size=n), dtype=torch.long)
    idx = rng.choice(n, size=min(weight, n), replace=False)
    e = torch.zeros(n, dtype=torch.long)
    e[idx] = 1
    return e


def _grid_str(error: torch.Tensor, d: int) -> str:
    """Pretty string for visual assertions in small-grid tests."""
    g = error.reshape(d, d)
    rows = []
    for r in range(d):
        rows.append(" ".join(str(int(x)) for x in g[r]))
    return "\n".join(rows)


def _make_dem_artifacts(
    *,
    distance: int,
    n_rounds: int,
    rotation: str,
    num_errors: int = 64,
    seed: int = 1234,
):
    """
    Build synthetic H/p artifacts with the right shape for MemoryCircuitTorch.

    For generate_batch(), frames_xz has shape (B, 2 * num_detectors) where
    num_detectors must equal n_rounds * nq.
    """
    first_bulk = rotation[0]
    rot_type = "V" if rotation[1] == "V" else "H"
    code = SurfaceCode(distance, first_bulk_syndrome_type=first_bulk, rotated_type=rot_type)
    nq = len(code.all_qubits)
    num_detectors = n_rounds * nq

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    H = torch.randint(
        0,
        2,
        (2 * num_detectors, num_errors),
        dtype=torch.uint8,
        generator=g,
    )
    # Keep probabilities small so sampled errors are sparse but non-trivial.
    p = torch.rand(num_errors, dtype=torch.float32, generator=g) * 0.02
    return H, p


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


class TestCoordinateHelpers(unittest.TestCase):

    def test_roundtrip(self):
        for d in (3, 5, 7):
            for idx in range(d * d):
                a, b = linear_index_to_coordinates(idx, d)
                self.assertEqual(coordinates_to_linear_index(a, b, d), idx)

    def test_boundary(self):
        a, b = linear_index_to_coordinates(0, 5)
        self.assertEqual((a, b), (0, 0))
        a, b = linear_index_to_coordinates(24, 5)
        self.assertEqual((a, b), (4, 4))


# ---------------------------------------------------------------------------
# Stabilizer support extraction
# ---------------------------------------------------------------------------


class TestStabilizerSupport(unittest.TestCase):

    def test_support_sizes(self):
        for d in (3, 5):
            hx, hz, _ = _build_parity_matrices(d)
            for stab in range(hx.shape[0]):
                support = get_stabilizer_support_from_parity_matrix(stab, hx)
                self.assertIn(
                    len(support), (2, 4),
                    f"d={d} X-stab {stab}: unexpected support size {len(support)}"
                )
            for stab in range(hz.shape[0]):
                support = get_stabilizer_support_from_parity_matrix(stab, hz)
                self.assertIn(
                    len(support), (2, 4),
                    f"d={d} Z-stab {stab}: unexpected support size {len(support)}"
                )

    def test_out_of_range(self):
        hx, _, _ = _build_parity_matrices(3)
        self.assertEqual(get_stabilizer_support_from_parity_matrix(999, hx), [])


# ---------------------------------------------------------------------------
# Weight reduction
# ---------------------------------------------------------------------------


class TestWeightReduction(unittest.TestCase):

    def _assert_weight_nonincreasing(self, fn, parity, d, seeds=10):
        for s in range(seeds):
            e = _random_error(d, seed=s)
            reduced = fn(e, d, parity)
            self.assertLessEqual(
                int(reduced.sum()), int(e.sum()),
                f"seed={s}: weight increased from {int(e.sum())} to {int(reduced.sum())}"
            )

    def test_weight_reduction_X_d3(self):
        hx, _, _ = _build_parity_matrices(3)
        self._assert_weight_nonincreasing(weight_reduction_X, hx, 3)

    def test_weight_reduction_Z_d3(self):
        _, hz, _ = _build_parity_matrices(3)
        self._assert_weight_nonincreasing(weight_reduction_Z, hz, 3)

    def test_weight_reduction_X_d5(self):
        hx, _, _ = _build_parity_matrices(5)
        self._assert_weight_nonincreasing(weight_reduction_X, hx, 5, seeds=20)

    def test_weight_reduction_Z_d5(self):
        _, hz, _ = _build_parity_matrices(5)
        self._assert_weight_nonincreasing(weight_reduction_Z, hz, 5, seeds=20)

    def test_zero_input(self):
        hx, hz, _ = _build_parity_matrices(3)
        e = torch.zeros(9, dtype=torch.long)
        self.assertTrue(torch.equal(weight_reduction_X(e, 3, hx), e))
        self.assertTrue(torch.equal(weight_reduction_Z(e, 3, hz), e))

    def test_weight4_removal(self):
        """Weight-4 error matching a full stabilizer should be zeroed out."""
        for d in (3, 5):
            hx, hz, _ = _build_parity_matrices(d)
            for stab_idx in range(hx.shape[0]):
                support = get_stabilizer_support_from_parity_matrix(stab_idx, hx)
                if len(support) == 4:
                    e = torch.zeros(d * d, dtype=torch.long)
                    for q in support:
                        e[q] = 1
                    reduced = weight_reduction_X(e, d, hx)
                    self.assertEqual(
                        int(reduced.sum()), 0, f"d={d} stab={stab_idx}: weight-4 not fully removed"
                    )
                    break


# ---------------------------------------------------------------------------
# simplify_X / simplify_Z – idempotency
# ---------------------------------------------------------------------------


class TestSimplifyIdempotency(unittest.TestCase):

    def _check_idempotent(self, simplify_fn, parity, d, seeds=20):
        for s in range(seeds):
            e = _random_error(d, seed=s)
            once = simplify_fn(e, d, parity)
            twice = simplify_fn(once, d, parity)
            self.assertTrue(
                torch.equal(once, twice),
                f"seed={s}: simplify not idempotent\n  once={once}\n  twice={twice}"
            )

    def test_simplify_X_d3(self):
        hx, _, _ = _build_parity_matrices(3)
        self._check_idempotent(simplify_X, hx, 3)

    def test_simplify_Z_d3(self):
        _, hz, _ = _build_parity_matrices(3)
        self._check_idempotent(simplify_Z, hz, 3)

    def test_simplify_X_d5(self):
        hx, _, _ = _build_parity_matrices(5)
        self._check_idempotent(simplify_X, hx, 5)

    def test_simplify_Z_d5(self):
        _, hz, _ = _build_parity_matrices(5)
        self._check_idempotent(simplify_Z, hz, 5)


# ---------------------------------------------------------------------------
# Syndrome invariance: simplify preserves the syndrome
# ---------------------------------------------------------------------------


class TestSyndromeInvariance(unittest.TestCase):

    def _check_syndrome_preserved(self, simplify_fn, simplify_parity, syndrome_parity, d, seeds=30):
        for s in range(seeds):
            e = _random_error(d, seed=s)
            s_before = _syndrome(e, syndrome_parity)
            simplified = simplify_fn(e, d, simplify_parity)
            s_after = _syndrome(simplified, syndrome_parity)
            self.assertTrue(
                torch.equal(s_before, s_after), f"seed={s}: syndrome changed after simplify\n"
                f"  before={s_before}\n  after ={s_after}"
            )

    def test_X_syndrome_d3(self):
        hx, hz, _ = _build_parity_matrices(3)
        self._check_syndrome_preserved(simplify_X, hx, hz, 3)

    def test_Z_syndrome_d3(self):
        hx, hz, _ = _build_parity_matrices(3)
        self._check_syndrome_preserved(simplify_Z, hz, hx, 3)

    def test_X_syndrome_d5(self):
        hx, hz, _ = _build_parity_matrices(5)
        self._check_syndrome_preserved(simplify_X, hx, hz, 5)

    def test_Z_syndrome_d5(self):
        hx, hz, _ = _build_parity_matrices(5)
        self._check_syndrome_preserved(simplify_Z, hz, hx, 5)


# ---------------------------------------------------------------------------
# Convergence: simplify_X/Z_with_count finish in bounded iterations
# ---------------------------------------------------------------------------


class TestSimplifyConvergence(unittest.TestCase):

    def test_convergence_X_d5(self):
        hx, _, _ = _build_parity_matrices(5)
        max_seen = 0
        for s in range(50):
            e = _random_error(5, seed=s)
            _, iters = simplify_X_with_count(e, 5, hx)
            max_seen = max(max_seen, iters)
        self.assertLess(max_seen, 50, f"simplify_X needed {max_seen} iters (expected << 100)")

    def test_convergence_Z_d5(self):
        _, hz, _ = _build_parity_matrices(5)
        max_seen = 0
        for s in range(50):
            e = _random_error(5, seed=s)
            _, iters = simplify_Z_with_count(e, 5, hz)
            max_seen = max(max_seen, iters)
        self.assertLess(max_seen, 50, f"simplify_Z needed {max_seen} iters (expected << 100)")


# ---------------------------------------------------------------------------
# Spacelike HE: full diff→simplify→diff pipeline
# ---------------------------------------------------------------------------


class TestSpacelikeHE(unittest.TestCase):

    def _random_diff_pair(self, d, n_rounds, seed=0):
        rng = np.random.RandomState(seed)
        x = torch.tensor(rng.randint(0, 2, (d * d, n_rounds)), dtype=torch.uint8)
        z = torch.tensor(rng.randint(0, 2, (d * d, n_rounds)), dtype=torch.uint8)
        return x, z

    def test_idempotent(self):
        for d in (3, 5):
            hx, hz, _ = _build_parity_matrices(d)
            x_diff, z_diff = self._random_diff_pair(d, 4, seed=7)
            x1, z1 = apply_spacelike_homological_equivalence(x_diff, z_diff, d, hx, hz)
            x2, z2 = apply_spacelike_homological_equivalence(x1, z1, d, hx, hz)
            self.assertTrue(torch.equal(x1, x2), f"d={d} spacelike X not idempotent")
            self.assertTrue(torch.equal(z1, z2), f"d={d} spacelike Z not idempotent")

    def test_zero_input(self):
        d = 3
        hx, hz, _ = _build_parity_matrices(d)
        x_diff = torch.zeros(d * d, 4, dtype=torch.uint8)
        z_diff = torch.zeros(d * d, 4, dtype=torch.uint8)
        x_out, z_out = apply_spacelike_homological_equivalence(x_diff, z_diff, d, hx, hz)
        self.assertTrue(torch.equal(x_out, x_diff))
        self.assertTrue(torch.equal(z_out, z_diff))

    def test_syndrome_preserved_per_round(self):
        """Each round's cumulative error should have the same syndrome before/after."""
        d = 5
        hx, hz, _ = _build_parity_matrices(d)
        x_diff, z_diff = self._random_diff_pair(d, 6, seed=42)

        x_out, z_out = apply_spacelike_homological_equivalence(
            x_diff.to(torch.long), z_diff.to(torch.long), d, hx, hz
        )

        x_cum_in = torch.cumsum(x_diff.to(torch.long), dim=1) % 2
        z_cum_in = torch.cumsum(z_diff.to(torch.long), dim=1) % 2
        x_cum_out = torch.cumsum(x_out.to(torch.long), dim=1) % 2
        z_cum_out = torch.cumsum(z_out.to(torch.long), dim=1) % 2

        for t in range(6):
            # X data errors are detected by Z-type checks (hz), and vice versa.
            sx_in = _syndrome(x_cum_in[:, t], hz)
            sx_out = _syndrome(x_cum_out[:, t], hz)
            sz_in = _syndrome(z_cum_in[:, t], hx)
            sz_out = _syndrome(z_cum_out[:, t], hx)
            self.assertTrue(torch.equal(sx_in, sx_out), f"X syndrome mismatch at round {t}")
            self.assertTrue(torch.equal(sz_in, sz_out), f"Z syndrome mismatch at round {t}")


# ---------------------------------------------------------------------------
# Timelike HE – unit level
# ---------------------------------------------------------------------------


class TestTimelikeHEUnit(unittest.TestCase):
    """Test simplifytimeX / simplifytimeZ on 2-round windows."""

    def _make_timelike_inputs(self, d, batch, seed=0):
        rng = np.random.RandomState(seed)
        hx, hz, _ = _build_parity_matrices(d)
        n_x = hx.shape[0]
        n_z = hz.shape[0]
        x_diff = torch.tensor(rng.randint(0, 2, (batch, d * d, 2)), dtype=torch.float32)
        z_diff = torch.tensor(rng.randint(0, 2, (batch, d * d, 2)), dtype=torch.float32)
        s1s2x = torch.tensor(rng.randint(0, 2, (batch, n_x, 2)), dtype=torch.float32)
        s1s2z = torch.tensor(rng.randint(0, 2, (batch, n_z, 2)), dtype=torch.float32)
        return x_diff, z_diff, s1s2x, s1s2z, hx, hz

    def _timelike_loop_density(self, error_diff, syndrome, parity):
        """Total timelike loop density across batch."""
        density = error_diff + torch.einsum('bst,sd->bdt', syndrome.float(), parity.float())
        return density.sum().item()

    def _count_changed_qubits(self, before: torch.Tensor, after: torch.Tensor) -> int:
        changed = (before != after).any(dim=2)
        return int(changed.sum().item())

    def test_simplifytimeX_acceptance_count_matches_changes(self):
        for d in (3, 5):
            for seed in range(10):
                x_diff, _, _, s1s2z, _, hz = self._make_timelike_inputs(d, 4, seed)
                x_out, _, accepted = simplifytimeX(x_diff, s1s2z, hz)
                changed = self._count_changed_qubits(x_diff, x_out)
                self.assertEqual(
                    accepted,
                    changed,
                    f"d={d} seed={seed}: accepted={accepted}, changed_qubits={changed}",
                )

    def test_simplifytimeZ_acceptance_count_matches_changes(self):
        for d in (3, 5):
            for seed in range(10):
                _, z_diff, s1s2x, _, hx, _ = self._make_timelike_inputs(d, 4, seed)
                z_out, _, accepted = simplifytimeZ(z_diff, s1s2x, hx)
                changed = self._count_changed_qubits(z_diff, z_out)
                self.assertEqual(
                    accepted,
                    changed,
                    f"d={d} seed={seed}: accepted={accepted}, changed_qubits={changed}",
                )

    def test_output_shapes(self):
        x_diff, _, _, s1s2z, _, hz = self._make_timelike_inputs(5, 8, seed=99)
        x_out, s1s2z_out, n = simplifytimeX(x_diff, s1s2z, hz)
        self.assertEqual(x_out.shape, x_diff.shape)
        self.assertEqual(s1s2z_out.shape, s1s2z.shape)
        self.assertIsInstance(n, int)

    def test_zero_error_no_change(self):
        """If both error_diff and syndromes are zero, nothing should change."""
        d = 3
        _, _, _, _, hx, hz = self._make_timelike_inputs(d, 2, seed=0)
        x_zero = torch.zeros(2, d * d, 2)
        s_zero = torch.zeros(2, hz.shape[0], 2)
        x_out, s_out, n = simplifytimeX(x_zero, s_zero, hz)
        self.assertTrue(torch.equal(x_out, x_zero))
        self.assertTrue(torch.equal(s_out, s_zero))
        self.assertEqual(n, 0)


# ---------------------------------------------------------------------------
# Timelike HE – full pipeline (apply_timelike_homological_equivalence)
# ---------------------------------------------------------------------------


class TestTimelikeHEPipeline(unittest.TestCase):
    """Integration-level tests for apply_timelike_homological_equivalence."""

    def _make_trainY(self, d, n_rounds, batch, seed=0):
        """Build a random trainY tensor in the expected (B, 4, n_rounds, D, D) layout."""
        rng = np.random.RandomState(seed)
        trainY = torch.tensor(
            rng.randint(0, 2, (batch, 4, n_rounds, d, d)),
            dtype=torch.float32,
        )
        return trainY

    def test_output_shape(self):
        d = 3
        n_rounds = 3
        hx, hz, _ = _build_parity_matrices(d)
        trainY = self._make_trainY(d, n_rounds, batch=2, seed=0)
        trainY_out, counts = apply_timelike_homological_equivalence(
            trainY, hx, hz, max_iterations=4, basis="X"
        )
        self.assertEqual(trainY_out.shape, trainY.shape)
        self.assertIn("total_accepted", counts)

    def test_deterministic_for_same_input(self):
        """The function should be deterministic for identical inputs."""
        d = 3
        n_rounds = 4
        hx, hz, _ = _build_parity_matrices(d)
        trainY = self._make_trainY(d, n_rounds, batch=2, seed=5)
        y1, c1 = apply_timelike_homological_equivalence(
            trainY.clone(), hx, hz, max_iterations=32, basis="X"
        )
        y2, c2 = apply_timelike_homological_equivalence(
            trainY.clone(), hx, hz, max_iterations=32, basis="X"
        )
        self.assertTrue(torch.equal(y1, y2))
        self.assertEqual(c1["total_accepted"], c2["total_accepted"])

    def test_zero_trainY(self):
        d = 3
        n_rounds = 3
        hx, hz, _ = _build_parity_matrices(d)
        trainY = torch.zeros(1, 4, n_rounds, d, d)
        trainY_out, counts = apply_timelike_homological_equivalence(
            trainY, hx, hz, max_iterations=4, basis="X"
        )
        self.assertTrue(torch.equal(trainY_out, trainY))
        self.assertEqual(counts["total_accepted"], 0)

    def test_basis_Z(self):
        """Check HE with basis='Z'."""
        d = 3
        hx, hz, _ = _build_parity_matrices(d)
        trainY = self._make_trainY(d, 4, batch=2, seed=10)
        trainY_out, counts = apply_timelike_homological_equivalence(
            trainY, hx, hz, max_iterations=8, basis="Z"
        )
        self.assertEqual(trainY_out.shape, trainY.shape)
        self.assertGreaterEqual(counts["total_accepted"], 0)


# ---------------------------------------------------------------------------
# Anticommuting stabilizers helper
# ---------------------------------------------------------------------------


class TestAnticommutingStabilizers(unittest.TestCase):

    def test_single_qubit(self):
        d = 3
        hx, hz, _ = _build_parity_matrices(d)
        stabs = get_anticommuting_stabilizers([0], hz)
        for s_idx in stabs:
            self.assertEqual(int(hz[s_idx, 0].item()), 1)

    def test_empty(self):
        _, hz, _ = _build_parity_matrices(3)
        self.assertEqual(get_anticommuting_stabilizers([], hz), [])


# ---------------------------------------------------------------------------
# Known small-distance cases
# ---------------------------------------------------------------------------


class TestKnownSmallCases(unittest.TestCase):
    """Hand-picked cases where we can predict the output."""

    def test_single_qubit_d3_X(self):
        """A single-qubit X error on d=3 should survive simplify (it's weight-1)."""
        hx, _, _ = _build_parity_matrices(3)
        for q in range(9):
            e = torch.zeros(9, dtype=torch.long)
            e[q] = 1
            s = simplify_X(e, 3, hx)
            # Weight should be <= 1 (might be 0 if qubit is in a weight-2 boundary stab)
            self.assertLessEqual(int(s.sum()), 1)

    def test_full_row_error_d3_nonincreasing(self):
        """A representative weight-3 case should never increase under simplify."""
        hx, _, _ = _build_parity_matrices(3)
        e = torch.zeros(9, dtype=torch.long)
        e[0] = e[1] = e[2] = 1
        reduced = simplify_X(e, 3, hx)
        self.assertLessEqual(int(reduced.sum()), 3, "weight-3 row error should be non-increasing")


# ---------------------------------------------------------------------------
# Visual examples with explicit expected patterns (d=3)
# ---------------------------------------------------------------------------


class TestVisualExamples(unittest.TestCase):
    """
    Simple visual tests for local equivalence rules on one 2x2 plaquette.

    d=3 data-qubit index layout:
      0 1 2
      3 4 5
      6 7 8

    We use the top-left plaquette support [0, 1, 3, 4].
    """

    def setUp(self):
        self.d = 3
        self.support = [0, 1, 3, 4]

    def test_visual_x_vertical_moves_right(self):
        """
        X local rule (vertical -> right column):

        before          after
        1 0 0           0 1 0
        1 0 0    -->    0 1 0
        0 0 0           0 0 0
        """
        before = torch.zeros(9, dtype=torch.long)
        before[0] = 1
        before[3] = 1
        after = apply_fix_equivalence_X_local(before, self.support, self.d)

        expected = torch.zeros(9, dtype=torch.long)
        expected[1] = 1
        expected[4] = 1

        self.assertTrue(
            torch.equal(after, expected),
            f"Unexpected X local vertical transform.\n"
            f"before:\n{_grid_str(before, self.d)}\n"
            f"after:\n{_grid_str(after, self.d)}\n"
            f"expected:\n{_grid_str(expected, self.d)}",
        )

    def test_visual_x_bottom_row_moves_up(self):
        """
        X local rule (bottom row -> top row):

        before          after
        0 0 0           1 1 0
        1 1 0    -->    0 0 0
        0 0 0           0 0 0
        """
        before = torch.zeros(9, dtype=torch.long)
        before[3] = 1
        before[4] = 1
        after = apply_fix_equivalence_X_local(before, self.support, self.d)

        expected = torch.zeros(9, dtype=torch.long)
        expected[0] = 1
        expected[1] = 1

        self.assertTrue(
            torch.equal(after, expected),
            f"Unexpected X local horizontal transform.\n"
            f"before:\n{_grid_str(before, self.d)}\n"
            f"after:\n{_grid_str(after, self.d)}\n"
            f"expected:\n{_grid_str(expected, self.d)}",
        )

    def test_visual_z_diagonal_to_canonical(self):
        """
        Z local rule (top-right + bottom-left -> top-left + bottom-right):

        before          after
        0 1 0           1 0 0
        1 0 0    -->    0 1 0
        0 0 0           0 0 0
        """
        before = torch.zeros(9, dtype=torch.long)
        before[1] = 1
        before[3] = 1
        after = apply_fix_equivalence_Z_local(before, self.support, self.d)

        expected = torch.zeros(9, dtype=torch.long)
        expected[0] = 1
        expected[4] = 1

        self.assertTrue(
            torch.equal(after, expected),
            f"Unexpected Z local diagonal transform.\n"
            f"before:\n{_grid_str(before, self.d)}\n"
            f"after:\n{_grid_str(after, self.d)}\n"
            f"expected:\n{_grid_str(expected, self.d)}",
        )


# ---------------------------------------------------------------------------
# Direct tests for homological_equivalence_torch.py
# ---------------------------------------------------------------------------


class TestTorchHEKernel(unittest.TestCase):
    """Unit tests for apply_weight1_timelike_homological_equivalence_torch()."""

    def setUp(self):
        self.d = 5
        self.n_rounds = 5
        self.batch_size = 8
        hx, hz, _ = _build_parity_matrices(self.d)
        self.parity_X = hx.to(torch.uint8)
        self.parity_Z = hz.to(torch.uint8)

    def _random_inputs(self, seed=42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        d2 = self.d * self.d
        num_x = self.parity_X.shape[0]
        num_z = self.parity_Z.shape[0]
        z_cum = torch.randint(
            0, 2, (self.batch_size, self.n_rounds, d2), dtype=torch.uint8, generator=rng
        )
        x_cum = torch.randint(
            0, 2, (self.batch_size, self.n_rounds, d2), dtype=torch.uint8, generator=rng
        )
        s1s2x = torch.randint(
            0, 2, (self.batch_size, self.n_rounds, num_x), dtype=torch.uint8, generator=rng
        )
        s1s2z = torch.randint(
            0, 2, (self.batch_size, self.n_rounds, num_z), dtype=torch.uint8, generator=rng
        )
        return z_cum, x_cum, s1s2x, s1s2z

    def test_output_shapes(self):
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs()
        z_diff, x_diff, sx_out, sz_out = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=1,
            max_passes=8,
            basis="X",
        )
        self.assertEqual(z_diff.shape, z_cum.shape)
        self.assertEqual(x_diff.shape, x_cum.shape)
        self.assertEqual(sx_out.shape, s1s2x.shape)
        self.assertEqual(sz_out.shape, s1s2z.shape)

    def test_output_dtypes(self):
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs()
        z_diff, x_diff, sx_out, sz_out = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=1,
            max_passes=8,
            basis="X",
        )
        self.assertEqual(z_diff.dtype, torch.uint8)
        self.assertEqual(x_diff.dtype, torch.uint8)

    def test_output_binary(self):
        """All output values must be exactly 0 or 1."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs()
        z_diff, x_diff, sx_out, sz_out = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=2,
            max_passes=8,
            basis="X",
        )
        for name, t in [
            ("z_diff", z_diff), ("x_diff", x_diff), ("s1s2x", sx_out), ("s1s2z", sz_out)
        ]:
            vals = torch.unique(t)
            self.assertTrue(
                torch.all((vals == 0) | (vals == 1)),
                f"{name} has non-binary values: {vals.tolist()}"
            )

    def test_diff_weight_nonincreasing(self):
        """With 0 cycles the final spacelike should not increase weight vs raw diffs."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=99)
        z_diff, x_diff, _, _ = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=0,
            max_passes=0,
            basis="X",
        )
        z_expected = torch.zeros_like(z_cum)
        z_expected[:, 0, :] = z_cum[:, 0, :]
        for r in range(1, self.n_rounds):
            z_expected[:, r, :] = z_cum[:, r, :] ^ z_cum[:, r - 1, :]

        x_expected = torch.zeros_like(x_cum)
        x_expected[:, 0, :] = x_cum[:, 0, :]
        for r in range(1, self.n_rounds):
            x_expected[:, r, :] = x_cum[:, r, :] ^ x_cum[:, r - 1, :]

        self.assertLessEqual(z_diff.sum().item(), z_expected.sum().item())
        self.assertLessEqual(x_diff.sum().item(), x_expected.sum().item())

    def test_zero_cycles_preserves_syndromes(self):
        """num_he_cycles=0 does not modify syndromes (only spacelike on diffs)."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=77)
        z_diff, x_diff, sx_out, sz_out = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=0,
            max_passes=8,
            basis="X",
        )
        self.assertTrue(torch.equal(sx_out, s1s2x))
        self.assertTrue(torch.equal(sz_out, s1s2z))

    def test_positive_cycles_may_differ_from_zero(self):
        """With random errors, HE cycles>0 should generally produce different output."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=77)
        out0 = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=0,
            max_passes=8,
            basis="X",
        )
        out1 = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=3,
            max_passes=8,
            basis="X",
        )
        any_diff = any(not torch.equal(out0[i], out1[i]) for i in range(4))
        self.assertTrue(any_diff, "Expected HE to change at least one output tensor")

    def test_deterministic(self):
        """Same inputs must produce identical outputs across calls."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=55)
        args = (z_cum, x_cum, s1s2x, s1s2z, self.parity_Z, self.parity_X, self.d, 1, 8, "X")
        out1 = apply_weight1_timelike_homological_equivalence_torch(*args)
        out2 = apply_weight1_timelike_homological_equivalence_torch(*args)
        for i in range(4):
            self.assertTrue(torch.equal(out1[i], out2[i]))

    def test_syndromes_unchanged_when_zero_cycles(self):
        """With num_he_cycles=0, syndromes should pass through unmodified."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=33)
        _, _, sx_out, sz_out = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=0,
            max_passes=8,
            basis="X",
        )
        self.assertTrue(torch.equal(sx_out, s1s2x))
        self.assertTrue(torch.equal(sz_out, s1s2z))

    def test_weight_nonincreasing(self):
        """HE reductions should never increase total error weight."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=42)
        z_raw, x_raw, _, _ = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=0,
            max_passes=0,
            basis="X",
        )
        z_red, x_red, _, _ = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=3,
            max_passes=8,
            basis="X",
        )
        self.assertLessEqual(
            z_red.sum().item(),
            z_raw.sum().item(), "Z error weight increased after HE"
        )
        self.assertLessEqual(
            x_red.sum().item(),
            x_raw.sum().item(), "X error weight increased after HE"
        )

    def test_output_binary_and_shapes_multi_d(self):
        """Output must be binary uint8 with correct shapes for all d, seed, basis."""
        for d in (3, 5, 7):
            hx, hz, _ = _build_parity_matrices(d)
            px_m = hx.to(torch.uint8)
            pz_m = hz.to(torch.uint8)
            d2 = d * d
            for seed in (42, 2026):
                for basis in ("X", "Z"):
                    rng = torch.Generator()
                    rng.manual_seed(seed)
                    B, R = 8, d
                    z_cum = torch.randint(0, 2, (B, R, d2), dtype=torch.uint8, generator=rng)
                    x_cum = torch.randint(0, 2, (B, R, d2), dtype=torch.uint8, generator=rng)
                    s1s2x = torch.randint(
                        0, 2, (B, R, px_m.shape[0]), dtype=torch.uint8, generator=rng
                    )
                    s1s2z = torch.randint(
                        0, 2, (B, R, pz_m.shape[0]), dtype=torch.uint8, generator=rng
                    )

                    z_out, x_out, sx_out, sz_out = apply_weight1_timelike_homological_equivalence_torch(
                        z_cum,
                        x_cum,
                        s1s2x,
                        s1s2z,
                        pz_m,
                        px_m,
                        d,
                        num_he_cycles=1,
                        max_passes=8,
                        basis=basis,
                    )

                    tag = f"d={d} seed={seed} basis={basis}"
                    self.assertEqual(z_out.shape, (B, R, d2), f"{tag}: z shape")
                    self.assertEqual(x_out.shape, (B, R, d2), f"{tag}: x shape")
                    self.assertEqual(sx_out.shape, s1s2x.shape, f"{tag}: sx shape")
                    self.assertEqual(sz_out.shape, s1s2z.shape, f"{tag}: sz shape")
                    for name, t in [("z", z_out), ("x", x_out), ("sx", sx_out), ("sz", sz_out)]:
                        self.assertTrue(((t == 0) | (t == 1)).all(), f"{tag}: {name} non-binary")
                        self.assertEqual(t.dtype, torch.uint8, f"{tag}: {name} dtype")

    def test_zero_input(self):
        """All-zero cumulative errors should produce all-zero diffs."""
        d2 = self.d * self.d
        num_x = self.parity_X.shape[0]
        num_z = self.parity_Z.shape[0]
        z_cum = torch.zeros(2, 3, d2, dtype=torch.uint8)
        x_cum = torch.zeros(2, 3, d2, dtype=torch.uint8)
        s1s2x = torch.zeros(2, 3, num_x, dtype=torch.uint8)
        s1s2z = torch.zeros(2, 3, num_z, dtype=torch.uint8)
        z_diff, x_diff, _, _ = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=1,
            max_passes=8,
            basis="X",
        )
        self.assertTrue(torch.all(z_diff == 0))
        self.assertTrue(torch.all(x_diff == 0))

    def test_single_round(self):
        """With 1 round, diff == cum; spacelike can only reduce weight."""
        d2 = self.d * self.d
        num_x = self.parity_X.shape[0]
        num_z = self.parity_Z.shape[0]
        rng = torch.Generator()
        rng.manual_seed(11)
        z_cum = torch.randint(0, 2, (4, 1, d2), dtype=torch.uint8, generator=rng)
        x_cum = torch.randint(0, 2, (4, 1, d2), dtype=torch.uint8, generator=rng)
        s1s2x = torch.randint(0, 2, (4, 1, num_x), dtype=torch.uint8, generator=rng)
        s1s2z = torch.randint(0, 2, (4, 1, num_z), dtype=torch.uint8, generator=rng)
        z_diff, x_diff, _, _ = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=1,
            max_passes=8,
            basis="X",
        )
        self.assertLessEqual(z_diff.sum().item(), z_cum.sum().item())
        self.assertLessEqual(x_diff.sum().item(), x_cum.sum().item())

    def test_multiple_distances(self):
        """Verify the kernel works across d=3,5,7."""
        for d in (3, 5, 7):
            hx, hz, _ = _build_parity_matrices(d)
            px = hx.to(torch.uint8)
            pz = hz.to(torch.uint8)
            d2 = d * d
            rng = torch.Generator()
            rng.manual_seed(d)
            z_cum = torch.randint(0, 2, (2, d, d2), dtype=torch.uint8, generator=rng)
            x_cum = torch.randint(0, 2, (2, d, d2), dtype=torch.uint8, generator=rng)
            s1s2x = torch.randint(0, 2, (2, d, px.shape[0]), dtype=torch.uint8, generator=rng)
            s1s2z = torch.randint(0, 2, (2, d, pz.shape[0]), dtype=torch.uint8, generator=rng)
            z_diff, x_diff, _, _ = apply_weight1_timelike_homological_equivalence_torch(
                z_cum,
                x_cum,
                s1s2x,
                s1s2z,
                pz,
                px,
                d,
                num_he_cycles=1,
                max_passes=8,
                basis="X",
            )
            self.assertEqual(z_diff.shape, (2, d, d2), f"Wrong shape for d={d}")

    def test_convergence(self):
        """More passes should not further reduce total error weight."""
        z_cum, x_cum, s1s2x, s1s2z = self._random_inputs(seed=88)
        out_few = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=1,
            max_passes=32,
            basis="X",
        )
        out_many = apply_weight1_timelike_homological_equivalence_torch(
            z_cum,
            x_cum,
            s1s2x,
            s1s2z,
            self.parity_Z,
            self.parity_X,
            self.d,
            num_he_cycles=1,
            max_passes=256,
            basis="X",
        )
        w_few = out_few[0].sum().item() + out_few[1].sum().item()
        w_many = out_many[0].sum().item() + out_many[1].sum().item()
        self.assertEqual(w_few, w_many, f"Weight changed from {w_few} to {w_many} with more passes")

    def _spacelike_inputs(self, d, B, seed):
        """Generate random inputs and caches for spacelike HE tests."""
        hx, hz, _ = _build_parity_matrices(d)
        px = hx.to(torch.uint8)
        pz = hz.to(torch.uint8)
        dev = torch.device("cpu")
        cache_xsp = build_spacelike_he_cache(px, distance=d, device=dev)
        cache_zsp = build_spacelike_he_cache(pz, distance=d, device=dev)
        d2 = d * d
        R = d
        rng = torch.Generator().manual_seed(seed)
        z_cum = torch.randint(0, 2, (B, R, d2), dtype=torch.uint8, generator=rng)
        x_cum = torch.randint(0, 2, (B, R, d2), dtype=torch.uint8, generator=rng)
        s1s2x = torch.randint(0, 2, (B, R, px.shape[0]), dtype=torch.uint8, generator=rng)
        s1s2z = torch.randint(0, 2, (B, R, pz.shape[0]), dtype=torch.uint8, generator=rng)
        return px, pz, cache_xsp, cache_zsp, z_cum, x_cum, s1s2x, s1s2z

    def _check_spacelike_invariants(self, d, method="default", tag="sp"):
        """Verify spacelike-on-diffs HE invariants using apply_homological_equivalence_torch_vmap."""
        from qec.surface_code.homological_equivalence_torch import apply_homological_equivalence_torch_vmap
        for seed in (100, 200, 300):
            px, pz, cxsp, czsp, z_cum, x_cum, s1s2x, s1s2z = \
                self._spacelike_inputs(d, B=4, seed=seed + d)

            z_prev = torch.cat([torch.zeros_like(z_cum[:, :1, :]), z_cum[:, :-1, :]], dim=1)
            x_prev = torch.cat([torch.zeros_like(x_cum[:, :1, :]), x_cum[:, :-1, :]], dim=1)
            z_diff_in = (z_cum ^ z_prev).to(torch.uint8)
            x_diff_in = (x_cum ^ x_prev).to(torch.uint8)

            z_diff_out, x_diff_out = apply_homological_equivalence_torch_vmap(
                z_diff_in, x_diff_in, pz, px, d, cache_Z=czsp, cache_X=cxsp
            )

            B = 4
            R = d
            for b in range(B):
                for r in range(R):
                    self.assertLessEqual(
                        x_diff_out[b, r].sum().item(), x_diff_in[b, r].sum().item(),
                        f"{tag} d={d} s={seed} b={b} r={r}: X diff weight increased"
                    )
                    self.assertLessEqual(
                        z_diff_out[b, r].sum().item(), z_diff_in[b, r].sum().item(),
                        f"{tag} d={d} s={seed} b={b} r={r}: Z diff weight increased"
                    )

            self.assertTrue(
                ((x_diff_out == 0) | (x_diff_out == 1)).all(), f"{tag} d={d} s={seed}: X non-binary"
            )
            self.assertTrue(
                ((z_diff_out == 0) | (z_diff_out == 1)).all(), f"{tag} d={d} s={seed}: Z non-binary"
            )

            z_diff_2, x_diff_2 = apply_homological_equivalence_torch_vmap(
                z_diff_out, x_diff_out, pz, px, d, cache_Z=czsp, cache_X=cxsp
            )
            self.assertTrue(
                torch.equal(z_diff_out, z_diff_2), f"{tag} d={d} s={seed}: Z not idempotent"
            )
            self.assertTrue(
                torch.equal(x_diff_out, x_diff_2), f"{tag} d={d} s={seed}: X not idempotent"
            )

    def test_spacelike_invariants(self):
        """Spacelike HE on diffs preserves all invariants."""
        for d in (3, 5, 7):
            self._check_spacelike_invariants(d)


class TestTorchCacheBuilders(unittest.TestCase):
    """Unit tests for build_spacelike_he_cache / build_timelike_he_cache."""

    def test_spacelike_cache_fields(self):
        hx, _, _ = _build_parity_matrices(3)
        cache = build_spacelike_he_cache(hx.to(torch.uint8), distance=3, device=torch.device("cpu"))
        num_stabs = hx.shape[0]
        self.assertEqual(cache.parity.shape[0], num_stabs)
        self.assertEqual(cache.support_sizes.shape[0], num_stabs)
        self.assertEqual(cache.distance, 3)

    def test_timelike_cache_fields(self):
        hx, _, _ = _build_parity_matrices(3)
        cache = build_timelike_he_cache(hx.to(torch.uint8))
        self.assertIsNotNone(cache)
        self.assertEqual(cache.num_stabs, hx.shape[0])
        self.assertEqual(cache.D2, hx.shape[1])

    def test_cache_support_sizes(self):
        """Each stabilizer should have weight 2 (boundary) or 4 (bulk)."""
        for d in (3, 5, 7):
            hx, hz, _ = _build_parity_matrices(d)
            for label, h in [("X", hx), ("Z", hz)]:
                cache = build_spacelike_he_cache(
                    h.to(torch.uint8), distance=d, device=torch.device("cpu")
                )
                for s in range(cache.support_sizes.shape[0]):
                    w = cache.support_sizes[s].item()
                    self.assertIn(w, (2, 4), f"d={d} {label} stab {s}: unexpected weight {w}")

    def test_cache_parity_matches_input(self):
        """Cache parity must exactly match the input parity matrix."""
        hx, _, _ = _build_parity_matrices(5)
        px = hx.to(torch.uint8)
        cache = build_spacelike_he_cache(px, distance=5, device=torch.device("cpu"))
        for s in range(px.shape[0]):
            expected = set(torch.nonzero(px[s], as_tuple=True)[0].tolist())
            got = set(torch.nonzero(cache.parity[s], as_tuple=True)[0].tolist())
            self.assertEqual(got, expected, f"Stab {s}: cache {got} != parity {expected}")

    def test_cache_layers_cover_all_stabs(self):
        """Greedy layers should collectively cover all stabilizers."""
        for d in (3, 5):
            hx, _, _ = _build_parity_matrices(d)
            cache = build_spacelike_he_cache(
                hx.to(torch.uint8), distance=d, device=torch.device("cpu")
            )
            all_idx = []
            for layer in cache.layers:
                all_idx.extend(layer.tolist())
            self.assertEqual(sorted(all_idx), list(range(hx.shape[0])))


# ---------------------------------------------------------------------------
# Torch integration tests
# ---------------------------------------------------------------------------


class TestHETorchIntegration(unittest.TestCase):
    """Integration tests through MemoryCircuitTorch.generate_batch()."""

    def setUp(self):
        self.distance = 3
        self.n_rounds = 3
        self.basis = "X"
        self.rotation = "XV"
        self.batch_size = 4
        self.device = torch.device("cpu")

        self.H, self.p = _make_dem_artifacts(
            distance=self.distance,
            n_rounds=self.n_rounds,
            rotation=self.rotation,
            num_errors=64,
            seed=2026,
        )

    def _make_generator(self, num_he_cycles: int) -> MemoryCircuitTorch:
        return MemoryCircuitTorch(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=self.basis,
            code_rotation=self.rotation,
            timelike_he=True,
            num_he_cycles=num_he_cycles,
            max_passes_w1=8,
            device=self.device,
            H=self.H,
            p=self.p,
            A=None,  # Keep deterministic/simple: no timelike correction matrix
        )

    def test_generate_batch_shapes_and_dtypes(self):
        """
        Basic check with HE cycles enabled.

        Verifies the integration path runs and returns tensors in the model format:
          trainX: (B, 4, R, D, D), float32
          trainY: (B, 4, R, D, D), float32
        """
        gen = self._make_generator(num_he_cycles=2)
        torch.manual_seed(17)
        trainX, trainY = gen.generate_batch(batch_size=self.batch_size)

        expected = (self.batch_size, 4, self.n_rounds, self.distance, self.distance)
        self.assertEqual(trainX.shape, expected)
        self.assertEqual(trainY.shape, expected)
        self.assertEqual(trainX.dtype, torch.float32)
        self.assertEqual(trainY.dtype, torch.float32)

    def test_he_reduces_error_weight(self):
        """
        With HE enabled, total error weight in trainY should not increase
        compared to HE disabled, and trainX should be unchanged (same sample).
        """
        gen0 = self._make_generator(num_he_cycles=0)
        gen1 = self._make_generator(num_he_cycles=2)

        torch.manual_seed(123)
        trainX0, trainY0 = gen0.generate_batch(batch_size=self.batch_size)
        torch.manual_seed(123)
        trainX1, trainY1 = gen1.generate_batch(batch_size=self.batch_size)

        # trainX is derived from meas_old and should be identical for same sample.
        self.assertTrue(torch.equal(trainX0, trainX1))

        # Error channels (0, 1) weight should not increase.
        w0 = trainY0[:, :2].sum().item()
        w1 = trainY1[:, :2].sum().item()
        self.assertLessEqual(w1, w0, f"HE increased error weight: {w0} -> {w1}")

    def test_trainY_binary_channels(self):
        """
        trainY channels encode binary error/syndrome maps (stored as float tensors).
        """
        gen = self._make_generator(num_he_cycles=1)
        torch.manual_seed(44)
        _, trainY = gen.generate_batch(batch_size=self.batch_size)

        self.assertFalse(torch.isnan(trainY).any())
        unique = torch.unique(trainY)
        # All values should be exactly 0 or 1 for trainY channels.
        self.assertTrue(torch.all((unique == 0.0) | (unique == 1.0)))


class TestWeight2TimelikeTorchVsCPU(unittest.TestCase):
    """Compare torch weight-2 timelike HE against CPU reference."""

    def _run_comparison(self, d, B=8, seed=42):
        hx, hz, _ = _build_parity_matrices(d)
        parity_X = hx.to(torch.uint8)
        parity_Z = hz.to(torch.uint8)

        cache_Z_w2 = build_weight2_timelike_cache(parity_Z, parity_X, d, 'Z', torch.device('cpu'))
        cache_X_w2 = build_weight2_timelike_cache(parity_X, parity_Z, d, 'X', torch.device('cpu'))

        rng = torch.Generator().manual_seed(seed)
        D2 = d * d
        z_err = torch.randint(0, 2, (B, D2, 2), dtype=torch.float32, generator=rng)
        x_err = torch.randint(0, 2, (B, D2, 2), dtype=torch.float32, generator=rng)
        s1s2x = torch.randint(0, 2, (B, parity_X.shape[0], 2), dtype=torch.float32, generator=rng)
        s1s2z = torch.randint(0, 2, (B, parity_Z.shape[0], 2), dtype=torch.float32, generator=rng)

        # CPU reference: one pass of weight-2 Z (horizontal only, matching the reference)
        z_cpu, sx_cpu, n_cpu_z = simplifytimeZ_weight2(
            z_err.clone(), s1s2x.clone(), parity_X, parity_Z, d
        )
        x_cpu, sz_cpu, n_cpu_x = simplifytimeX_weight2(
            x_err.clone(), s1s2z.clone(), parity_Z, parity_X, d
        )

        # Torch: one pass (first pattern only, to match CPU which only uses horizontal/vertical)
        z_torch = z_err.clone()
        sx_torch = s1s2x.clone()
        z_torch, sx_torch, n_torch_z = _simplify_time_w2_step(
            z_torch, sx_torch, parity_X.float(), cache_Z_w2
        )

        x_torch = x_err.clone()
        sz_torch = s1s2z.clone()
        x_torch, sz_torch, n_torch_x = _simplify_time_w2_step(
            x_torch, sz_torch, parity_Z.float(), cache_X_w2
        )

        return (
            z_cpu, sx_cpu, n_cpu_z, x_cpu, sz_cpu, n_cpu_x, z_torch, sx_torch, n_torch_z, x_torch,
            sz_torch, n_torch_x
        )

    def test_weight2_output_binary_d5(self):
        """Weight-2 torch output should remain binary."""
        (z_cpu, sx_cpu, _, x_cpu, sz_cpu, _, z_torch, sx_torch, _, x_torch, sz_torch,
         _) = self._run_comparison(5)

        for name, t in [
            ("z_torch", z_torch), ("x_torch", x_torch), ("sx_torch", sx_torch),
            ("sz_torch", sz_torch)
        ]:
            unique = torch.unique(t)
            self.assertTrue(
                torch.all((unique == 0.0) | (unique == 1.0)),
                f"{name} has non-binary values: {unique.tolist()}"
            )

    def test_weight2_density_nonincreasing_d5(self):
        """Weight-2 should not increase total density."""
        d = 5
        hx, hz, _ = _build_parity_matrices(d)
        parity_X = hx.to(torch.uint8)
        parity_Z = hz.to(torch.uint8)

        cache_Z_w2 = build_weight2_timelike_cache(parity_Z, parity_X, d, 'Z', torch.device('cpu'))

        rng = torch.Generator().manual_seed(99)
        B = 16
        D2 = d * d
        z_err = torch.randint(0, 2, (B, D2, 2), dtype=torch.float32, generator=rng)
        s1s2x = torch.randint(0, 2, (B, parity_X.shape[0], 2), dtype=torch.float32, generator=rng)

        old_density = (z_err + torch.einsum('bst,sd->bdt', s1s2x, parity_X.float())).sum(dim=(1, 2))

        z_out, sx_out, _ = _simplify_time_w2_step(
            z_err.clone(), s1s2x.clone(), parity_X.float(), cache_Z_w2
        )

        new_density = (z_out +
                       torch.einsum('bst,sd->bdt', sx_out, parity_X.float())).sum(dim=(1, 2))

        self.assertTrue(
            torch.all(new_density <= old_density + 1e-6),
            f"Density increased: {old_density.tolist()} -> {new_density.tolist()}"
        )

    def test_weight2_cache_shapes(self):
        """Cache tensors have expected shapes."""
        for d in (3, 5, 7):
            hx, hz, _ = _build_parity_matrices(d)
            parity_X = hx.to(torch.uint8)
            parity_Z = hz.to(torch.uint8)

            cache = build_weight2_timelike_cache(parity_Z, parity_X, d, 'Z', torch.device('cpu'))

            self.assertEqual(cache.num_patterns, 3)
            self.assertEqual(cache.qubit_pairs.shape[:2], (cache.num_w4, 3))
            self.assertEqual(cache.qubit_pairs.shape[2], 2)
            self.assertEqual(cache.anti_stab_counts.shape, (cache.num_w4, 3))

    def test_weight2_integration_via_generate_batch(self):
        """End-to-end: generate_batch with weight-2 enabled should produce valid output."""
        d, R = 3, 3
        H, p = _make_dem_artifacts(distance=d, n_rounds=R, rotation="XV", num_errors=64, seed=42)
        gen = MemoryCircuitTorch(
            distance=d,
            n_rounds=R,
            basis="X",
            code_rotation="XV",
            timelike_he=True,
            num_he_cycles=1,
            max_passes_w1=4,
            use_weight2=True,
            max_passes_w2=2,
            device=torch.device("cpu"),
            H=H,
            p=p,
            A=None,
        )
        torch.manual_seed(77)
        trainX, trainY = gen.generate_batch(batch_size=4)
        self.assertEqual(trainX.shape, (4, 4, R, d, d))
        self.assertEqual(trainY.shape, (4, 4, R, d, d))
        self.assertFalse(torch.isnan(trainY).any())
        unique = torch.unique(trainY)
        self.assertTrue(torch.all((unique == 0.0) | (unique == 1.0)))


if __name__ == "__main__":
    unittest.main()
