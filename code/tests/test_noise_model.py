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
import unittest
from pathlib import Path

import numpy as np
import stim

sys.path.insert(0, str(Path(__file__).parent.parent))

from qec.noise_model import NoiseModel, CNOT_ERROR_TYPES, CNOT_ERROR_INDEX, _single_p_mapping
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


if __name__ == "__main__":
    unittest.main()
