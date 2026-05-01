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
Statistical tests for the 25-parameter NoiseModel (Stim-based).

Key conventions enforced by this test module:
- Always use d=5 and n_rounds=5 for comparisons.
- Compare syndrome *diffs* (XOR of consecutive rounds), not raw cumulative syndromes.
- Apply inference-style masking for comparisons:
  - Mask the non-basis syndrome type at round 0 and also at the last round.
- Provide two tiers:
  - Fast tier (~10k shots)
  - Slow tier (>=100k shots) gated by RUN_SLOW=1
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.noise_model import (
    NoiseModel,
    CNOT_ERROR_TYPES,
    CNOT_ERROR_INDEX,
    _single_p_mapping,
    get_grouped_totals,
    get_training_upscaled_noise_model,
    SURFACE_CODE_TRAINING_UPSCALE_TARGET,
    SURFACE_CODE_THRESHOLD_APPROX,
)
from qec.surface_code.memory_circuit import MemoryCircuit
from qec.surface_code.data_mapping import (
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
    reshape_Xstabilizers_to_grid_vectorized,
    reshape_Zstabilizers_to_grid_vectorized,
)


def _shots_fast() -> int:
    return int(os.environ.get("NOISEMODEL_FAST_SHOTS", "10000"))


def _shots_slow() -> int:
    # Requirement: >=100k
    return int(os.environ.get("NOISEMODEL_SLOW_SHOTS", "100000"))


def _run_slow() -> bool:
    return os.environ.get("RUN_SLOW", "0") == "1"


def _compute_density_from_trainX_np(trainX_np: np.ndarray) -> dict:
    """
    trainX_np: (B, 4, T, D, D) with channels [x_syn, z_syn, x_present, z_present]
    Returns:
      dict with overall/per-round densities for x and z, counting only present stabilizers.
    """
    x_syn = trainX_np[:, 0]
    z_syn = trainX_np[:, 1]
    x_pres = trainX_np[:, 2] > 0
    z_pres = trainX_np[:, 3] > 0

    # Per-round denominators
    x_den_t = x_pres.sum(axis=(0, 2, 3)).astype(np.float64)
    z_den_t = z_pres.sum(axis=(0, 2, 3)).astype(np.float64)

    x_num_t = (x_syn.astype(np.int32) * x_pres.astype(np.int32)).sum(axis=(0, 2, 3)
                                                                    ).astype(np.float64)
    z_num_t = (z_syn.astype(np.int32) * z_pres.astype(np.int32)).sum(axis=(0, 2, 3)
                                                                    ).astype(np.float64)

    # Avoid divide-by-zero (shouldn't happen, but keep robust)
    x_den_t = np.maximum(x_den_t, 1.0)
    z_den_t = np.maximum(z_den_t, 1.0)

    x_density_t = x_num_t / x_den_t
    z_density_t = z_num_t / z_den_t

    # Overall density: weighted across rounds by presence count
    x_density = x_num_t.sum() / x_den_t.sum()
    z_density = z_num_t.sum() / z_den_t.sum()

    return {
        "x_density": float(x_density),
        "z_density": float(z_density),
        "x_density_t": x_density_t,
        "z_density_t": z_density_t,
    }


def _noise_model_from_p(p: float) -> NoiseModel:
    return NoiseModel.from_config_dict(_single_p_mapping(p))


def _stim_trainX_np(
    distance: int, n_rounds: int, basis: str, noise_model: NoiseModel | None
) -> np.ndarray:
    """Build Stim circuit -> sample measurements -> compute syndrome diffs -> map to trainX grid (inference masking)."""
    basis = basis.upper()
    code_rotation = "XV"

    # For backwards compatibility, MemoryCircuit still wants the legacy scalar rates; when noise_model is used,
    # these serve primarily as placeholders/buffer defaults.
    if noise_model is None:
        p = 0.01
        spam_error = 2.0 * p / 3.0
        circ = MemoryCircuit(
            distance=distance,
            idle_error=p,
            sqgate_error=p,
            tqgate_error=p,
            spam_error=spam_error,
            n_rounds=n_rounds,
            basis=basis,
            code_rotation=code_rotation
        )
    else:
        # Use max-prob as a safe placeholder for scalar slots.
        p = float(noise_model.get_max_probability())
        circ = MemoryCircuit(
            distance=distance,
            idle_error=p,
            sqgate_error=p,
            tqgate_error=p,
            spam_error=p,
            n_rounds=n_rounds,
            basis=basis,
            code_rotation=code_rotation,
            noise_model=noise_model
        )
    circ.set_error_rates()

    meas = stim.Circuit(circ.circuit).compile_sampler().sample(shots=_shots_fast())
    # Drop final D*D data-qubit measurements, reshape to (B, T, D^2-1)
    D = distance
    B = meas.shape[0]
    meas_anc = meas[:, :-(D * D)].reshape(B, n_rounds, D * D - 1).astype(np.uint8)

    half = (D * D - 1) // 2
    x_raw = meas_anc[:, :, :half]  # (B, T, Sx)
    z_raw = meas_anc[:, :, half:]  # (B, T, Sz)

    # XOR diffs with leading zeros
    x_pad = np.concatenate([np.zeros((B, 1, half), dtype=np.uint8), x_raw], axis=1)
    z_pad = np.concatenate([np.zeros((B, 1, half), dtype=np.uint8), z_raw], axis=1)
    x_diff = (x_pad[:, 1:] ^ x_pad[:, :-1]).astype(np.uint8)  # (B, T, Sx)
    z_diff = (z_pad[:, 1:] ^ z_pad[:, :-1]).astype(np.uint8)  # (B, T, Sz)

    # Inference-style masking: mask non-basis at round 0 and last round
    if basis == "X":
        z_diff[:, 0] = 0
        z_diff[:, -1] = 0
    else:  # "Z"
        x_diff[:, 0] = 0
        x_diff[:, -1] = 0

    # Map to grid using the same helpers as training formatting
    x_syn = x_diff.transpose(0, 2, 1)  # (B, Sx, T)
    z_syn = z_diff.transpose(0, 2, 1)  # (B, Sz, T)

    # Mapping helpers are implemented in torch; use CPU tensors here.
    import torch
    x_syn_t = torch.from_numpy(x_syn)
    z_syn_t = torch.from_numpy(z_syn)

    x_syn_mapped = reshape_Xstabilizers_to_grid_vectorized(x_syn_t, D, rotation=code_rotation)
    z_syn_mapped = reshape_Zstabilizers_to_grid_vectorized(z_syn_t, D, rotation=code_rotation)

    # (B, D*D, T) -> (B, T, D, D)
    x_syn_grid = x_syn_mapped.reshape(B, D, D,
                                      n_rounds).permute(0, 3, 1,
                                                        2).contiguous().cpu().numpy().astype(
                                                            np.float32
                                                        )
    z_syn_grid = z_syn_mapped.reshape(B, D, D,
                                      n_rounds).permute(0, 3, 1,
                                                        2).contiguous().cpu().numpy().astype(
                                                            np.float32
                                                        )

    # Presence maps (weights) + masking consistent with datapipe_stim
    w_mapX = normalized_weight_mapping_Xstab_memory(D,
                                                    code_rotation).reshape(D,
                                                                           D).cpu().numpy().astype(
                                                                               np.float32
                                                                           )
    w_mapZ = normalized_weight_mapping_Zstab_memory(D,
                                                    code_rotation).reshape(D,
                                                                           D).cpu().numpy().astype(
                                                                               np.float32
                                                                           )
    x_present = np.broadcast_to(w_mapX[None, None, :, :], (B, n_rounds, D, D)).copy()
    z_present = np.broadcast_to(w_mapZ[None, None, :, :], (B, n_rounds, D, D)).copy()

    if basis == "X":
        z_present[:, 0] = 0
        z_present[:, -1] = 0
    else:
        x_present[:, 0] = 0
        x_present[:, -1] = 0

    trainX = np.stack([x_syn_grid, z_syn_grid, x_present, z_present], axis=1).astype(np.float32)
    return trainX


def _random_noise_model(seed: int, scale: float = 0.01) -> NoiseModel:
    """Generate a random (but reproducible) 22-parameter NoiseModel for stress testing bookkeeping."""
    rng = np.random.default_rng(seed)
    # Prep/meas in [0.2, 1.5] * scale
    p_prep_X = float(scale * rng.uniform(0.2, 1.5))
    p_prep_Z = float(scale * rng.uniform(0.2, 1.5))
    p_meas_X = float(scale * rng.uniform(0.2, 1.5))
    p_meas_Z = float(scale * rng.uniform(0.2, 1.5))

    # Idle components in [0.1, 1.0] * scale
    # In the 25p model we split idle into:
    # - idle_cnot_*: bulk/CNOT-layer idles
    # - idle_spam_*: data idles during ancilla prep/reset window
    #
    # For random stress tests we sample them independently (same scale) to stress classification.
    p_idle_cnot_X = float(scale * rng.uniform(0.1, 1.0))
    p_idle_cnot_Y = float(scale * rng.uniform(0.1, 1.0))
    p_idle_cnot_Z = float(scale * rng.uniform(0.1, 1.0))

    p_idle_spam_X = float(scale * rng.uniform(0.1, 1.0))
    p_idle_spam_Y = float(scale * rng.uniform(0.1, 1.0))
    p_idle_spam_Z = float(scale * rng.uniform(0.1, 1.0))

    # CNOT components around scale (each ~ scale/15 * [0.2, 3.0])
    cnot_probs = {
        f"p_cnot_{k}": float((scale / 15.0) * rng.uniform(0.2, 3.0)) for k in CNOT_ERROR_TYPES
    }

    return NoiseModel(
        p_prep_X=p_prep_X,
        p_prep_Z=p_prep_Z,
        p_meas_X=p_meas_X,
        p_meas_Z=p_meas_Z,
        p_idle_cnot_X=p_idle_cnot_X,
        p_idle_cnot_Y=p_idle_cnot_Y,
        p_idle_cnot_Z=p_idle_cnot_Z,
        p_idle_spam_X=p_idle_spam_X,
        p_idle_spam_Y=p_idle_spam_Y,
        p_idle_spam_Z=p_idle_spam_Z,
        **cnot_probs,
    )


class TestNoiseModel(unittest.TestCase):

    def test_noise_model_roundtrip_and_invariants(self):
        p = 0.01
        nm = _noise_model_from_p(p)
        # CNOT-layer idle total matches p
        self.assertAlmostEqual(nm.get_total_idle_cnot_probability(), p, places=12)
        self.assertAlmostEqual(nm.get_total_cnot_probability(), p, places=12)

        cfg = nm.to_config_dict()
        nm2 = NoiseModel.from_config_dict(cfg)
        self.assertEqual(nm, nm2)

        with self.assertRaises(ValueError):
            NoiseModel(p_prep_X=1.5)

    def test_canonical_noise_model_hash_uses_public_parameters_only(self):
        nm = _noise_model_from_p(0.006)
        nm_copy = nm.copy()
        self.assertEqual(nm.sha256(), nm_copy.sha256())
        self.assertNotIn("_reference", nm.canonical_parameters())

        nm_copy._reference = {k: float(v) * 1.7 for k, v in nm_copy._reference.items()}
        self.assertEqual(nm.sha256(), nm_copy.sha256())

        changed = nm.copy()
        changed.p_prep_X += 1e-9
        self.assertNotEqual(nm.sha256(), changed.sha256())

    def test_stim_circuit_audit_no_cnot_noise_in_logical_measurement_section(self):
        # Non-trivial noise model: ensure PAULI_CHANNEL_2 appears in repeat block but NOT after it.
        D = 5
        T = 5
        nm = NoiseModel(
            p_prep_X=0.01,
            p_prep_Z=0.02,
            p_meas_X=0.01,
            p_meas_Z=0.02,
            p_idle_cnot_X=0.003,
            p_idle_cnot_Y=0.002,
            p_idle_cnot_Z=0.004,
            p_idle_spam_X=0.003,
            p_idle_spam_Y=0.002,
            p_idle_spam_Z=0.004,
            **{f"p_cnot_{k}": (0.0005 if k != "ZZ" else 0.001) for k in CNOT_ERROR_TYPES}
        )
        circ = MemoryCircuit(
            distance=D,
            idle_error=nm.get_max_probability(),
            sqgate_error=nm.get_max_probability(),
            tqgate_error=nm.get_max_probability(),
            spam_error=nm.get_max_probability(),
            n_rounds=T,
            basis="X",
            noise_model=nm,
            code_rotation="XV"
        )
        circ.set_error_rates()

        lines = circ.circuit.split("\n")
        in_repeat = False
        after_repeat = False
        pauli2_in_repeat = 0
        pauli2_after_repeat = 0
        for line in lines:
            if line.startswith("REPEAT"):
                in_repeat = True
                continue
            if in_repeat and line.strip() == "}":
                in_repeat = False
                after_repeat = True
                continue
            if "PAULI_CHANNEL_2" in line:
                if in_repeat:
                    pauli2_in_repeat += 1
                elif after_repeat:
                    pauli2_after_repeat += 1

        self.assertGreater(pauli2_in_repeat, 0, "Expected PAULI_CHANNEL_2 inside stabilizer rounds")
        self.assertEqual(
            pauli2_after_repeat, 0,
            "Expected NO CNOT noise instructions in logical-measurement section"
        )

    def test_no_double_measurement_noise_in_final_data_qubit_readout(self):
        """
        Regression test for double measurement-noise injection on data qubits at the end of
        MemoryCircuit.__init__ when using the 25-parameter NoiseModel.

        _add_stabilizer_round(logical_measurement=True) injects a single "fake SPAM" error on
        data qubits (time-reversed p_meas) and then restores self.noise_model before returning.
        Without the fix the subsequent add_measure(data_qubits) call at the __init__ call site
        would see a non-None noise_model and inject the same p_meas channel a *second* time,
        creating phantom DEM error entries that bias LER/threshold estimates.

        The fix suppresses noise_model around that add_measure call.  This test verifies that
        the post-REPEAT circuit section contains exactly ONE measurement-error injection on data
        qubits (the legitimate fake-SPAM line), not two.
        """
        D = 3
        T = 3  # n_rounds must be >= 3 for the circuit to use a REPEAT block
        nm = NoiseModel(
            p_prep_X=0.01,
            p_prep_Z=0.02,
            p_meas_X=0.03,  # non-zero: triggers double-injection if bug is present
            p_meas_Z=0.04,
            p_idle_cnot_X=0.002,
            p_idle_cnot_Y=0.001,
            p_idle_cnot_Z=0.003,
            p_idle_spam_X=0.002,
            p_idle_spam_Y=0.001,
            p_idle_spam_Z=0.003,
            **{f"p_cnot_{k}": 0.0005 for k in CNOT_ERROR_TYPES}
        )

        for basis in ("X", "Z"):
            circ = MemoryCircuit(
                distance=D,
                idle_error=nm.get_max_probability(),
                sqgate_error=nm.get_max_probability(),
                tqgate_error=nm.get_max_probability(),
                spam_error=nm.get_max_probability(),
                n_rounds=T,
                basis=basis,
                noise_model=nm,
                code_rotation="XV",
            )
            circ.set_error_rates()

            # Isolate the circuit section that appears after the REPEAT block.
            lines = circ.circuit.split("\n")
            in_repeat = False
            after_repeat = False
            post_repeat_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("REPEAT"):
                    in_repeat = True
                    continue
                if in_repeat and stripped == "}":
                    in_repeat = False
                    after_repeat = True
                    continue
                if after_repeat:
                    post_repeat_lines.append(stripped)

            # Basis-labelled semantics for data-qubit readout failure:
            #   X-basis measurement error -> Z_ERROR(p_meas_X)
            #   Z-basis measurement error -> X_ERROR(p_meas_Z)
            # The only legitimate occurrence in the post-REPEAT section is the single fake-SPAM
            # injection inside _add_stabilizer_round(logical_measurement=True).  A second line
            # with the same instruction is the regression.
            if basis == "X":
                error_instr = "Z_ERROR"
                p_meas = float(nm.p_meas_X)
            else:
                error_instr = "X_ERROR"
                p_meas = float(nm.p_meas_Z)

            meas_error_lines = [l for l in post_repeat_lines if l.startswith(error_instr)]
            self.assertEqual(
                len(meas_error_lines), 1,
                f"basis={basis}: expected exactly 1 {error_instr} line in post-REPEAT section "
                f"(fake-SPAM only), got {len(meas_error_lines)}. "
                f"Double injection would indicate the noise_model suppression fix is missing. "
                f"Lines: {meas_error_lines}"
            )
            # Confirm the single line carries the correct probability.
            expected_prefix = f"{error_instr}({p_meas:.10f})"
            self.assertTrue(
                meas_error_lines[0].startswith(expected_prefix),
                f"basis={basis}: expected {error_instr} with p={p_meas:.10f}, "
                f"got: {meas_error_lines[0]}"
            )


class TestNoiseModelUpscaling(unittest.TestCase):
    """Tests for surface-code training noise model upscaling (get_training_upscaled_noise_model)."""

    def test_get_grouped_totals(self):
        """get_grouped_totals returns effective channels used for training scaling."""
        nm = _noise_model_from_p(0.01)
        tot = get_grouped_totals(nm)
        self.assertAlmostEqual(tot["p_prep_X"], 2.0 * 0.01 / 3.0, places=12)
        self.assertAlmostEqual(tot["p_prep_Z"], 2.0 * 0.01 / 3.0, places=12)
        self.assertAlmostEqual(tot["p_meas_X"], 2.0 * 0.01 / 3.0, places=12)
        self.assertAlmostEqual(tot["p_meas_Z"], 2.0 * 0.01 / 3.0, places=12)
        self.assertAlmostEqual(tot["p_prep_total"], 2.0 * 0.01 / 3.0 * 2, places=12)
        self.assertAlmostEqual(tot["p_meas_total"], 2.0 * 0.01 / 3.0 * 2, places=12)
        self.assertAlmostEqual(tot["p_idle_cnot"], 0.01, places=12)
        self.assertAlmostEqual(
            tot["p_idle_spam_raw"], nm.get_total_idle_spam_probability(), places=12
        )
        self.assertAlmostEqual(
            tot["p_idle_spam_effective"], nm.get_total_idle_spam_probability() / 2.0, places=12
        )
        self.assertAlmostEqual(tot["p_cnot"], 0.01, places=12)
        self.assertGreater(tot["max_group"], 0)
        self.assertEqual(
            tot["max_group"],
            max(
                tot["p_prep_X"],
                tot["p_prep_Z"],
                tot["p_meas_X"],
                tot["p_meas_Z"],
                tot["p_idle_cnot"],
                tot["p_idle_spam_effective"],
                tot["p_cnot"],
            )
        )

    def test_depolarizing_p006_has_target_effective_max_group(self):
        """The p=6e-3 config should not look above target due to channel double-counting."""
        nm = NoiseModel(
            p_prep_X=0.004,
            p_prep_Z=0.004,
            p_meas_X=0.004,
            p_meas_Z=0.004,
            p_idle_cnot_X=0.002,
            p_idle_cnot_Y=0.002,
            p_idle_cnot_Z=0.002,
            p_idle_spam_X=0.003984,
            p_idle_spam_Y=0.003984,
            p_idle_spam_Z=0.003984,
            **{f"p_cnot_{k}": 0.0004 for k in CNOT_ERROR_TYPES}
        )
        tot = get_grouped_totals(nm)
        self.assertAlmostEqual(tot["p_prep_X"], 0.004, places=12)
        self.assertAlmostEqual(tot["p_prep_Z"], 0.004, places=12)
        self.assertAlmostEqual(tot["p_meas_X"], 0.004, places=12)
        self.assertAlmostEqual(tot["p_meas_Z"], 0.004, places=12)
        self.assertAlmostEqual(tot["p_idle_cnot"], 0.006, places=12)
        self.assertAlmostEqual(tot["p_idle_spam_raw"], 0.011952, places=12)
        self.assertAlmostEqual(tot["p_idle_spam_effective"], 0.005976, places=12)
        self.assertAlmostEqual(tot["p_cnot"], 0.006, places=12)
        self.assertAlmostEqual(tot["max_group"], SURFACE_CODE_TRAINING_UPSCALE_TARGET, places=12)

        training_nm, info = get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertTrue(info["applied_upscale"])
        self.assertFalse(info["downscale_skipped"])
        self.assertFalse(info["above_target_warning"])
        self.assertAlmostEqual(info["scale_factor"], 1.0, places=12)
        self.assertEqual(training_nm.to_config_dict(), nm.to_config_dict())

    def test_precomputed_dem_probability_vector_uses_25p_values(self):
        """DEM precompute should build p from the explicit 25p model, not scalar p/3 or p/15."""
        import torch
        from qec.precompute_dem import precompute_dem_bundle_surface_code

        cnot_probs = {f"p_cnot_{k}": 0.00011 + i * 0.00001 for i, k in enumerate(CNOT_ERROR_TYPES)}
        nm = NoiseModel(
            p_prep_X=0.0011,
            p_prep_Z=0.0022,
            p_meas_X=0.0033,
            p_meas_Z=0.0044,
            p_idle_cnot_X=0.0051,
            p_idle_cnot_Y=0.0052,
            p_idle_cnot_Z=0.0053,
            p_idle_spam_X=0.0061,
            p_idle_spam_Y=0.0062,
            p_idle_spam_Z=0.0063,
            **cnot_probs
        )

        observed = []
        for basis in ("X", "Z"):
            artifacts = precompute_dem_bundle_surface_code(
                distance=3,
                n_rounds=3,
                basis=basis,
                code_rotation="XV",
                p_scalar=0.1234,
                dem_output_dir=None,
                device=torch.device("cpu"),
                export=False,
                return_artifacts=True,
                noise_model=nm,
            )
            observed.append(artifacts["p"].cpu().numpy())
        p_values = np.concatenate(observed)

        expected_values = [
            nm.p_prep_X,
            nm.p_prep_Z,
            nm.p_meas_X,
            nm.p_meas_Z,
            nm.p_idle_cnot_X,
            nm.p_idle_cnot_Y,
            nm.p_idle_cnot_Z,
            nm.p_idle_spam_X,
            nm.p_idle_spam_Y,
            nm.p_idle_spam_Z,
            nm.p_cnot_IX,
            nm.p_cnot_ZZ,
        ]
        for expected in expected_values:
            self.assertTrue(
                np.any(np.isclose(p_values, expected, rtol=0.0, atol=1e-9)),
                f"Expected 25p probability {expected} in DEM p vector",
            )

        scalar_derived_values = [0.1234 / 3.0, 0.1234 / 15.0, 2.0 * 0.1234 / 3.0]
        for scalar_value in scalar_derived_values:
            self.assertFalse(
                np.any(np.isclose(p_values, scalar_value, rtol=0.0, atol=1e-9)),
                f"Unexpected scalar-derived probability {scalar_value} in 25p DEM p vector",
            )

    def test_precompute_dem_export_writes_noise_metadata(self):
        import torch
        from qec.precompute_dem import (
            load_dem_artifact_metadata,
            precompute_dem_bundle_surface_code,
        )

        with tempfile.TemporaryDirectory() as tmp:
            precompute_dem_bundle_surface_code(
                distance=3,
                n_rounds=3,
                basis="X",
                code_rotation="XV",
                p_scalar=0.004,
                dem_output_dir=tmp,
                device=torch.device("cpu"),
                export=True,
            )
            scalar_meta = load_dem_artifact_metadata(
                Path(tmp) / "surface_d3_r3_X_frame_predecoder.p.npz"
            )
        self.assertEqual(scalar_meta["noise_mode"], "scalar")
        self.assertEqual(scalar_meta["distance"], 3)
        self.assertEqual(scalar_meta["basis"], "X")
        self.assertAlmostEqual(float(scalar_meta["p_scalar"]), 0.004, places=12)

        nm = _noise_model_from_p(0.005)
        with tempfile.TemporaryDirectory() as tmp:
            precompute_dem_bundle_surface_code(
                distance=3,
                n_rounds=3,
                basis="Z",
                code_rotation="XV",
                p_scalar=0.123,
                dem_output_dir=tmp,
                device=torch.device("cpu"),
                export=True,
                noise_model=nm,
            )
            nm_meta = load_dem_artifact_metadata(
                Path(tmp) / "surface_d3_r3_Z_frame_predecoder.p.npz"
            )
        self.assertEqual(nm_meta["noise_mode"], "noise_model")
        self.assertEqual(nm_meta["noise_model_sha256"], nm.sha256())
        self.assertEqual(nm_meta["noise_model"], nm.canonical_parameters())

    def test_upscale_small_noise(self):
        """When max_group < target, all 25 p's are scaled so that new max_group = target."""
        # Single-p 1e-4 -> max_group is around 1e-4 (order of magnitude)
        nm = _noise_model_from_p(1e-4)
        tot = get_grouped_totals(nm)
        self.assertLess(tot["max_group"], SURFACE_CODE_TRAINING_UPSCALE_TARGET)
        training_nm, info = get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertTrue(info["applied_upscale"])
        self.assertFalse(info["downscale_skipped"])
        scale = info["scale_factor"]
        self.assertGreaterEqual(scale, 1.0)
        self.assertAlmostEqual(
            scale, SURFACE_CODE_TRAINING_UPSCALE_TARGET / tot["max_group"], places=10
        )
        new_tot = get_grouped_totals(training_nm)
        self.assertAlmostEqual(
            new_tot["max_group"], SURFACE_CODE_TRAINING_UPSCALE_TARGET, places=10
        )
        # All params scaled by the same factor
        for k, v in nm.to_config_dict().items():
            self.assertAlmostEqual(training_nm.to_config_dict()[k], v * scale, places=12, msg=k)

    def test_upscale_exact_target_scale_one(self):
        """When max_group equals target, scale_factor is 1.0 and model is unchanged."""
        # Build a model with max_group = target by scaling a small model up
        nm_small = _noise_model_from_p(1e-4)
        tot_small = get_grouped_totals(nm_small)
        scale_to_target = SURFACE_CODE_TRAINING_UPSCALE_TARGET / tot_small["max_group"]
        params = {k: v * scale_to_target for k, v in nm_small.to_config_dict().items()}
        nm = NoiseModel.from_config_dict(params)
        training_nm, info = get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertTrue(info["applied_upscale"])
        self.assertAlmostEqual(info["scale_factor"], 1.0, places=10)
        self.assertAlmostEqual(
            training_nm.get_total_cnot_probability(), nm.get_total_cnot_probability(), places=12
        )

    def test_downscale_not_applied(self):
        """When max_group > target, parameters are NOT modified; downscale_skipped is True."""
        nm = _noise_model_from_p(1e-2)  # max_group well above 6e-3
        tot = get_grouped_totals(nm)
        self.assertGreater(tot["max_group"], SURFACE_CODE_TRAINING_UPSCALE_TARGET)
        training_nm, info = get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertFalse(info["applied_upscale"])
        self.assertTrue(info["downscale_skipped"])
        self.assertTrue(info["above_target_warning"])
        # Same object parameters (identity)
        self.assertEqual(nm.to_config_dict(), training_nm.to_config_dict())
        self.assertIs(training_nm, nm)

    def test_above_target_warning(self):
        """When max_group > target, above_target_warning is True."""
        nm = _noise_model_from_p(0.01)
        _, info = get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertTrue(info["above_target_warning"])
        nm_low = _noise_model_from_p(1e-4)
        _, info_low = get_training_upscaled_noise_model(nm_low, code_type="surface_code")
        self.assertFalse(info_low["above_target_warning"])

    def test_non_surface_code_no_upscaling(self):
        """For code_type != 'surface_code', no scaling is applied; original model returned."""
        nm = _noise_model_from_p(1e-4)
        training_nm, info = get_training_upscaled_noise_model(nm, code_type="color_code")
        self.assertFalse(info.get("applied_upscale", False))
        self.assertEqual(nm.to_config_dict(), training_nm.to_config_dict())
        self.assertIn("message", info)
        self.assertIn("surface_code", info["message"])

    def test_invalid_zero_totals_raises(self):
        """When all grouped totals are zero, get_training_upscaled_noise_model raises ValueError."""
        nm = NoiseModel()  # all zeros
        with self.assertRaises(ValueError) as ctx:
            get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertIn("all grouped totals are <= 0", str(ctx.exception))

    def test_upscale_preserves_reference(self):
        """Upscaled training model preserves _reference from the original."""
        nm = _noise_model_from_p(1e-4)
        ref = dict(nm._reference)
        training_nm, _ = get_training_upscaled_noise_model(nm, code_type="surface_code")
        self.assertEqual(training_nm._reference, ref)

    def test_skip_upscale_returns_original(self):
        """When skip_upscale=True, the original model is returned unchanged regardless of max_group."""
        nm = _noise_model_from_p(1e-4)
        tot = get_grouped_totals(nm)
        self.assertLess(tot["max_group"], SURFACE_CODE_TRAINING_UPSCALE_TARGET)
        training_nm, info = get_training_upscaled_noise_model(
            nm, code_type="surface_code", skip_upscale=True
        )
        self.assertIs(training_nm, nm)
        self.assertFalse(info["applied_upscale"])
        self.assertFalse(info["downscale_skipped"])
        self.assertTrue(info["skipped_by_user"])
        self.assertIn("SKIPPED", info["message"])

    def test_skip_upscale_above_target(self):
        """skip_upscale=True also works when max_group > target (no warning about downscale)."""
        nm = _noise_model_from_p(1e-2)
        training_nm, info = get_training_upscaled_noise_model(
            nm, code_type="surface_code", skip_upscale=True
        )
        self.assertIs(training_nm, nm)
        self.assertTrue(info["skipped_by_user"])
        self.assertFalse(info["applied_upscale"])
        self.assertFalse(info["downscale_skipped"])


if __name__ == "__main__":
    unittest.main()
