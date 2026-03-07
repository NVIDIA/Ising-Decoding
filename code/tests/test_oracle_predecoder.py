#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Oracle test: use trainY as the would-be predictions of a perfect pre-decoder.
# Given (trainX, trainY) from the data generator (no model), verify:
# 1. Residual syndromes R_X, R_Z are always zero when using trainY as predictions.
# 2. When building the full residual vector and passing it with pre_L to PyMatching,
#    there is no logical error (final_L == gt_obs).
# This validates both data generation and the inference residual pipeline.
"""
Oracle test: perfect pre-decoder (trainY as predictions).

Uses the same data generator (MemoryCircuitTorch) for data generation and as the
"oracle" predictor: trainY is used as the pre-decoder output. Then we compute
residual syndromes (same formulas as in logical_error_rate) and assert they are
zero, and that PyMatching on the residual + pre_L yields no logical error.
"""

import sys
import unittest
from pathlib import Path

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

import torch
import numpy as np
import pymatching

from qec.surface_code.memory_circuit_torch import MemoryCircuitTorch
from qec.surface_code.memory_circuit import MemoryCircuit
from qec.precompute_dem import precompute_dem_bundle_surface_code

# Residual computation (mirrors logical_error_rate.run_inference_...)
from evaluation.logical_error_rate import map_grid_to_stabilizer_tensor
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
)


def _build_stab_maps_from_code(code, device: torch.device):
    """Build stab maps from a SurfaceCode so S_X/S_Z use the same parity as the generator."""
    import torch
    Hx_i32 = torch.tensor(code.hx, dtype=torch.int32)
    Hz_i32 = torch.tensor(code.hz, dtype=torch.int32)
    Sx, D2 = Hx_i32.shape
    Sz, _ = Hz_i32.shape

    def _row_indices_and_mask(H_i32):
        S, D2 = H_i32.shape
        nz = H_i32.nonzero(as_tuple=False)
        rows, cols = nz[:, 0], nz[:, 1]
        deg = torch.bincount(rows, minlength=S)
        K = int(deg.max().item())
        if K == 0:
            return torch.full((S, 1), -1,
                              dtype=torch.long), torch.zeros((S, 1), dtype=torch.bool), deg, 0
        idx = torch.full((S, K), -1, dtype=torch.long)
        msk = torch.zeros((S, K), dtype=torch.bool)
        row_offsets = torch.zeros(S + 1, dtype=torch.long)
        row_offsets[1:] = deg.cumsum(0)
        pos = torch.arange(nz.size(0), dtype=torch.long) - row_offsets[rows]
        idx[rows, pos] = cols
        ar = torch.arange(K, dtype=torch.long).unsqueeze(0).expand(S, K)
        msk = ar < deg.unsqueeze(1)
        return idx, msk, deg, K

    Hx_idx, Hx_mask, _, Kx = _row_indices_and_mask(Hx_i32)
    Hz_idx, Hz_mask, _, Kz = _row_indices_and_mask(Hz_i32)
    rotation = code.first_bulk_syndrome_type + code.rotated_type  # e.g. XV, XH
    sx = compute_stabX_to_data_index_map(code.distance, rotation)
    sz = compute_stabZ_to_data_index_map(code.distance, rotation)
    stab_x = sx.clone().detach().to(torch.long
                                   ) if torch.is_tensor(sx) else torch.tensor(sx, dtype=torch.long)
    stab_z = sz.clone().detach().to(torch.long
                                   ) if torch.is_tensor(sz) else torch.tensor(sz, dtype=torch.long)
    return {
        "Hx_idx": Hx_idx.to(device),
        "Hx_mask": Hx_mask.to(device),
        "Kx": Kx,
        "Hz_idx": Hz_idx.to(device),
        "Hz_mask": Hz_mask.to(device),
        "Kz": Kz,
        "stab_x": stab_x.to(device),
        "stab_z": stab_z.to(device),
    }


def _compute_residuals_from_predictions(
    trainX: torch.Tensor,
    trainY: torch.Tensor,
    distance: int,
    code_rotation: str,
    basis: str,
    device: torch.device,
    code=None,
):
    """
    Compute residual syndromes R_X, R_Z when using the given predictions
    (same logic as logical_error_rate for model outputs).
    trainX: (B, 4, T, D, D), trainY: (B, 4, T, D, D).
    If code is provided (SurfaceCode from the generator), use its hx/hz for S_X/S_Z so parity matches.
    Returns (R_X, R_Z) as (B, n_x, T), (B, n_z, T) uint8.
    """
    B, _, T, D, _ = trainX.shape
    D2 = D * D
    if T < 2:
        raise ValueError("T must be >= 2 for residual computation")

    if code is not None:
        maps = _build_stab_maps_from_code(code, device)
    else:
        from evaluation.logical_error_rate import _build_stab_maps
        maps = _build_stab_maps(distance, code_rotation)
    Hx_idx = maps["Hx_idx"].to(device=device, dtype=torch.long)
    Hz_idx = maps["Hz_idx"].to(device=device, dtype=torch.long)
    Hx_mask = maps["Hx_mask"].to(device=device, dtype=torch.bool)
    Hz_mask = maps["Hz_mask"].to(device=device, dtype=torch.bool)
    Kx = maps["Kx"]
    Kz = maps["Kz"]
    stab_indices_x = maps["stab_x"].to(device=device, dtype=torch.long)
    stab_indices_z = maps["stab_z"].to(device=device, dtype=torch.long)

    # Syndrome diffs from trainX (grid -> stabilizer order)
    x_syn_diff = map_grid_to_stabilizer_tensor(trainX[:, 0].to(device),
                                               stab_indices_x).to(torch.int32)
    z_syn_diff = map_grid_to_stabilizer_tensor(trainX[:, 1].to(device),
                                               stab_indices_z).to(torch.int32)
    n_x, n_z = x_syn_diff.shape[1], z_syn_diff.shape[1]

    # Use trainY as predictions (binarize)
    z_data_corr = (trainY[:, 0].to(device) >= 0.5).to(torch.int32)
    x_data_corr = (trainY[:, 1].to(device) >= 0.5).to(torch.int32)
    syn_x_grid = (trainY[:, 2].to(device) >= 0.5).to(torch.int32)
    syn_z_grid = (trainY[:, 3].to(device) >= 0.5).to(torch.int32)

    # S_X from z_data_corr, S_Z from x_data_corr (same as LER)
    z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T)
    x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, T)
    z_exp = z_flat.unsqueeze(2).expand(B, D2, Kx, T)
    hx_idx_e = Hx_idx.clamp_min(0).view(1, -1, Kx, 1).expand(B, -1, -1, T)
    g_x = z_exp.gather(1, hx_idx_e)
    m_x = Hx_mask.view(1, -1, Kx, 1).expand_as(g_x)
    S_X = (g_x.masked_fill(~m_x, 0).sum(dim=2) & 1)
    x_exp = x_flat.unsqueeze(2).expand(B, D2, Kz, T)
    hz_idx_e = Hz_idx.clamp_min(0).view(1, -1, Kz, 1).expand(B, -1, -1, T)
    g_z = x_exp.gather(1, hz_idx_e)
    m_z = Hz_mask.view(1, -1, Kz, 1).expand_as(g_z)
    S_Z = (g_z.masked_fill(~m_z, 0).sum(dim=2) & 1)

    syn_x_flat = map_grid_to_stabilizer_tensor(syn_x_grid, stab_indices_x).to(torch.int32)
    syn_z_flat = map_grid_to_stabilizer_tensor(syn_z_grid, stab_indices_z).to(torch.int32)

    R_X = torch.empty_like(x_syn_diff, dtype=torch.uint8)
    R_X[:, :, 0] = (x_syn_diff[:, :, 0] + syn_x_flat[:, :, 0] + S_X[:, :, 0]) & 1
    if T > 1:
        R_X[:, :, 1:] = (
            x_syn_diff[:, :, 1:] + syn_x_flat[:, :, 1:] + syn_x_flat[:, :, :-1] + S_X[:, :, 1:]
        ) & 1

    R_Z = torch.empty_like(z_syn_diff, dtype=torch.uint8)
    R_Z[:, :, 0] = (z_syn_diff[:, :, 0] + syn_z_flat[:, :, 0] + S_Z[:, :, 0]) & 1
    if T > 1:
        R_Z[:, :, 1:] = (
            z_syn_diff[:, :, 1:] + syn_z_flat[:, :, 1:] + syn_z_flat[:, :, :-1] + S_Z[:, :, 1:]
        ) & 1

    return R_X, R_Z


def _measurement_order_from_stim_circuit(stim_circuit):
    """
    Get the ordered list of (basis, qubit) for each measurement bit from a Stim circuit.
    Uses flattened_operations() so the order matches what compile_m2d_converter() expects.
    Returns: list of (basis, qubit) where basis is 'X' or 'Z', qubit is global qubit index.
    """
    order = []
    for op in stim_circuit.flattened_operations():
        name = op[0]
        if name not in ("M", "MX", "MZ", "MR", "MRX", "MRZ"):
            continue
        targets = op[1]
        if name in ("MX", "MRX"):
            basis = "X"
        elif name in ("MZ", "MRZ"):
            basis = "Z"
        else:
            basis = "Z"  # M/MR default
        for q in targets:
            order.append((basis, int(q)))
    return order


def _stim_measurement_from_torch_frame(
    meas_old: torch.Tensor,
    x_cum: torch.Tensor,
    z_cum: torch.Tensor,
    basis: str,
) -> np.ndarray:
    """
    Build full Stim measurement record from torch generator outputs (simple layout).
    Order: ancilla (round-major, R*(D^2-1)) then data qubits at end (D^2).
    """
    B, R, num_ancilla = meas_old.shape
    D2 = x_cum.shape[2]
    ancilla_flat = meas_old.cpu().numpy().reshape(B, -1).astype(np.uint8)
    if basis == "X":
        data_final = x_cum[:, -1, :].cpu().numpy().astype(np.uint8)
    else:
        data_final = z_cum[:, -1, :].cpu().numpy().astype(np.uint8)
    return np.concatenate([ancilla_flat, data_final], axis=1)


def _stim_measurement_from_torch_frame_circuit_order(
    stim_circuit,
    code,
    meas_old: torch.Tensor,
    x_cum: torch.Tensor,
    z_cum: torch.Tensor,
) -> np.ndarray:
    """
    Build Stim measurement record in the exact order of the circuit's M/MX/MZ instructions.
    Uses _measurement_order_from_stim_circuit and looks up each (basis, qubit) in our frame.
    """
    order = _measurement_order_from_stim_circuit(stim_circuit)
    B, R, num_ancilla = meas_old.shape
    D2 = x_cum.shape[2]
    data_qubits = np.asarray(code.data_qubits).reshape(-1)
    xcheck_qubits = np.asarray(code.xcheck_qubits).reshape(-1)
    zcheck_qubits = np.asarray(code.zcheck_qubits).reshape(-1)
    num_x, num_z = len(xcheck_qubits), len(zcheck_qubits)
    # Map global qubit -> index in xcheck/zcheck/data for lookup
    xcheck_map = {int(q): i for i, q in enumerate(xcheck_qubits)}
    zcheck_map = {int(q): i for i, q in enumerate(zcheck_qubits)}
    data_map = {int(q): i for i, q in enumerate(data_qubits)}

    meas_old_np = meas_old.cpu().numpy()
    x_cum_np = x_cum.cpu().numpy()
    z_cum_np = z_cum.cpu().numpy()

    out = np.zeros((B, len(order)), dtype=np.uint8)
    num_ancilla_per_round = num_x + num_z
    for k, (b, q) in enumerate(order):
        if q in data_map:
            j = data_map[q]
            # X-basis measurement reads Z-component; Z-basis reads X-component (anti-commutation)
            if b == "X":
                out[:, k] = z_cum_np[:, -1, j]
            else:
                out[:, k] = x_cum_np[:, -1, j]
        else:
            r = k // num_ancilla_per_round
            if b == "X":
                i = xcheck_map[q]
            else:
                i = num_x + zcheck_map[q]
            out[:, k] = meas_old_np[:, r, i]
    return out


def _dets_and_obs_from_stim_circuit(
    stim_circuit,
    meas_full: np.ndarray,
) -> np.ndarray:
    """Run Stim m2d converter on measurement record; return (N, num_detectors + num_observables)."""
    converter = stim_circuit.compile_m2d_converter()
    # Pass unpacked (one byte per bit) so bit order matches circuit; packed uint8 uses a different convention
    meas_bool = np.asarray(meas_full, dtype=bool)
    out = converter.convert(measurements=meas_bool, append_observables=True)
    return np.asarray(out, dtype=np.uint8)


def _assert_residuals_zero_oracle(R_X: torch.Tensor, R_Z: torch.Tensor, basis: str):
    """
    Assert residual syndromes are zero where the input is defined.
    For X basis, Z is masked at round 0 and last so we only check R_Z on middle rounds.
    For Z basis, X is masked at round 0 and last so we only check R_X on middle rounds.
    """
    _, _, T = R_X.shape
    if basis == "X":
        assert (R_X == 0).all(), "R_X should be all zeros (X basis)"
        if T > 2:
            assert (R_Z[:, :, 1:-1] == 0).all(), "R_Z should be zero at non-masked rounds (1..T-2)"
    else:
        assert (R_Z == 0).all(), "R_Z should be all zeros (Z basis)"
        if T > 2:
            assert (R_X[:, :, 1:-1] == 0).all(), "R_X should be zero at non-masked rounds (1..T-2)"


def _compute_pre_L_and_gt_obs(
    trainY: torch.Tensor,
    x_cum: torch.Tensor,
    z_cum: torch.Tensor,
    distance: int,
    code_rotation: str,
    basis: str,
    device: torch.device,
):
    """Compute pre-decoder logical frame (pre_L) from trainY and ground-truth observable from data frames."""
    B, R, D2 = x_cum.shape
    D = distance
    code_rotation = code_rotation.upper()
    if code_rotation in ("XV", "ZH"):
        Lx = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lx[0, :D] = 1
        Lz = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lz[0, ::D] = 1
    else:
        Lx = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lx[0, ::D] = 1
        Lz = torch.zeros((1, D2), dtype=torch.int32, device=device)
        Lz[0, :D] = 1

    z_data_corr = (trainY[:, 0].to(device) >= 0.5).to(torch.int32)
    x_data_corr = (trainY[:, 1].to(device) >= 0.5).to(torch.int32)
    z_flat = z_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, R)
    x_flat = x_data_corr.permute(0, 2, 3, 1).contiguous().view(B, D2, R)

    if basis == "X":
        pre_L_t = torch.einsum(
            "ld,bdt->blt",
            Lx.to(torch.float32),
            z_flat.to(torch.float32),
        ).remainder_(2).to(torch.int32)
    else:
        pre_L_t = torch.einsum(
            "ld,bdt->blt",
            Lz.to(torch.float32),
            x_flat.to(torch.float32),
        ).remainder_(2).to(torch.int32)
    pre_L = pre_L_t.sum(dim=2).remainder_(2).view(-1)

    # Ground truth: logical of final data qubit state
    z_final = z_cum[:, -1, :]
    x_final = x_cum[:, -1, :]
    if basis == "X":
        gt_obs = (torch.einsum("ld,bd->b", Lx.to(torch.float32), z_final.to(torch.float32)) %
                  2).to(torch.int64)
    else:
        gt_obs = (torch.einsum("ld,bd->b", Lz.to(torch.float32), x_final.to(torch.float32)) %
                  2).to(torch.int64)
    return pre_L, gt_obs


class TestOraclePreDecoder(unittest.TestCase):
    """Oracle test: trainY as perfect pre-decoder -> zero residuals and no logical error."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distance = 5
        self.n_rounds = 5
        self.code_rotation = "XV"

    def _make_generator(self, basis: str):
        """Create MemoryCircuitTorch with in-memory DEM (no precomputed files)."""
        artifacts = precompute_dem_bundle_surface_code(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=basis,
            code_rotation=self.code_rotation,
            p_scalar=0.01,
            dem_output_dir=None,
            device=self.device,
            export=False,
            return_artifacts=True,
        )
        return MemoryCircuitTorch(
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=basis,
            code_rotation=self.code_rotation,
            H=artifacts["H"],
            p=artifacts["p"],
            A=artifacts.get("A"),
            device=self.device,
        )

    def test_residuals_zero_with_oracle_X_basis(self):
        """Using trainY as predictions, residual syndromes R_X and R_Z are zero (X basis)."""
        gen = self._make_generator("X")
        trainX, trainY = gen.generate_batch(batch_size=8)
        R_X, R_Z = _compute_residuals_from_predictions(
            trainX,
            trainY,
            self.distance,
            self.code_rotation,
            "X",
            self.device,
            code=gen.code,
        )
        _assert_residuals_zero_oracle(R_X, R_Z, "X")

    def test_residuals_zero_with_oracle_Z_basis(self):
        """Using trainY as predictions, residual syndromes R_X and R_Z are zero (Z basis)."""
        gen = self._make_generator("Z")
        trainX, trainY = gen.generate_batch(batch_size=8)
        R_X, R_Z = _compute_residuals_from_predictions(
            trainX,
            trainY,
            self.distance,
            self.code_rotation,
            "Z",
            self.device,
            code=gen.code,
        )
        _assert_residuals_zero_oracle(R_X, R_Z, "Z")

    def test_residuals_zero_multiple_batches(self):
        """Residuals zero over several batches (stability check)."""
        for basis in ("X", "Z"):
            gen = self._make_generator(basis)
            for _ in range(3):
                trainX, trainY = gen.generate_batch(batch_size=4)
                R_X, R_Z = _compute_residuals_from_predictions(
                    trainX,
                    trainY,
                    self.distance,
                    self.code_rotation,
                    basis,
                    self.device,
                    code=gen.code,
                )
                _assert_residuals_zero_oracle(R_X, R_Z, basis)

    def test_pymatching_no_logical_error_with_oracle(self):
        """
        With oracle predictions (trainY), residual is zero; residual + pre_L passed to
        PyMatching should yield final_L == gt_obs (no logical error).

        Uses Option A: build the Stim measurement record in the exact order of the
        circuit's M/MX/MZ instructions (via flattened_operations()), then run the
        m2d converter and PyMatching. The circuit is built with add_boundary_detectors=True
        to match the decoder used in evaluation. Measurements are passed to the converter
        unpacked (one byte per bit) to match Stim's bit order; packed uint8 would require
        a different convention. Researchers can verify alignment (measurement order,
        detector/observable indexing, X/Z readout convention) if extending this test.
        """
        basis = "X"
        gen = self._make_generator(basis)
        trainX, trainY, meas_old, x_cum, z_cum = gen.generate_batch(batch_size=8, return_aux=True)
        # Stim circuit with boundary detectors (same params as eval pipeline)
        mem_circuit = MemoryCircuit(
            self.distance,
            0.01,
            0.01,
            0.01,
            0.007,
            self.n_rounds,
            basis,
            code_rotation=self.code_rotation,
            add_boundary_detectors=True,
        )
        mem_circuit.set_error_rates()
        stim_circuit = mem_circuit.stim_circuit
        # Measurement record in circuit order (Option A)
        meas_full = _stim_measurement_from_torch_frame_circuit_order(
            stim_circuit, gen.code, meas_old, x_cum, z_cum
        )
        dets_and_obs = _dets_and_obs_from_stim_circuit(stim_circuit, meas_full)
        num_dets = stim_circuit.num_detectors
        num_obs = stim_circuit.num_observables
        dets = np.asarray(dets_and_obs[:, :num_dets], dtype=np.uint8)
        obs_from_stim = np.asarray(dets_and_obs[:, -num_obs:], dtype=np.uint8)

        pre_L, gt_obs = _compute_pre_L_and_gt_obs(
            trainY,
            x_cum,
            z_cum,
            self.distance,
            self.code_rotation,
            basis,
            self.device,
        )
        gt_obs_np = gt_obs.cpu().numpy().reshape(-1, num_obs)

        # Sanity: observables from our frame (via circuit-order meas) should match gt_obs
        np.testing.assert_array_equal(
            obs_from_stim,
            gt_obs_np,
            err_msg=
            "Observables from Stim converter should match frame-derived gt_obs (measurement order)."
        )

        det_model = stim_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
        matcher = pymatching.Matching.from_detector_error_model(det_model)
        # Oracle: residual is zero, so pass zeros to the decoder
        residual = np.zeros((dets.shape[0], num_dets), dtype=np.uint8)
        pred_obs = matcher.decode_batch(residual)
        pred_obs = np.asarray(pred_obs, dtype=np.uint8).reshape(-1, num_obs)
        pre_L_np = pre_L.cpu().numpy().reshape(-1, num_obs)
        final_L = (pre_L_np + pred_obs) % 2
        np.testing.assert_array_equal(
            final_L,
            gt_obs_np,
            err_msg="Oracle: final_L (pre_L + PyMatching on zero residual) should equal gt_obs."
        )


if __name__ == "__main__":
    unittest.main()
