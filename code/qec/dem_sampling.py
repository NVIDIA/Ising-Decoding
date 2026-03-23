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
Torch-only DEM sampling utilities for training data generation.

This module provides the sampling functions needed by MemoryCircuitTorch
to generate training batches from precomputed DEM matrices (H, p, A).
"""

from __future__ import annotations

import torch
import numpy as np


def dem_sampling(H: torch.Tensor, p: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Sample errors from a detector error model (DEM) using precomputed H and p matrices.

    Args:
        H: (2*num_detectors, num_errors) uint8 - Detector-error incidence matrix
        p: (num_errors,) float32 - Per-error probabilities
        batch_size: int - Number of samples to generate

    Returns:
        frames_xz: (batch_size, 2*num_detectors) uint8 - Detector outcomes
    """
    p = torch.as_tensor(p, device=H.device)
    errors = (torch.rand(batch_size, p.numel(), device=H.device) < p).to(torch.uint8)
    return torch.remainder(errors.float() @ H.t().float(), 2).to(torch.uint8)


def measure_from_stacked_frames(
    frames_xz: torch.Tensor, meas_qubits: torch.Tensor, meas_bases: torch.Tensor, nq: int
) -> torch.Tensor:
    """
    Extract measurement outcomes from stacked frame data.

    Convention: Z-basis measurement reads the X-component of the Pauli frame
    (anti-commutation), and X-basis measurement reads the Z-component.

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8 - Stacked [X|Z] detector frames
        meas_qubits: (num_meas,) long - Qubit indices for measurements
        meas_bases: (num_meas,) long - Basis for each measurement (0=X, 1=Z)
        nq: int - Total number of qubits

    Returns:
        meas_old: (batch_size, n_rounds, num_meas) uint8 - Measurement outcomes
    """
    meas_qubits = torch.as_tensor(meas_qubits, device=frames_xz.device,
                                  dtype=torch.long).reshape(-1)
    meas_bases = torch.as_tensor(meas_bases, device=frames_xz.device, dtype=torch.long).reshape(-1)
    D = frames_xz.shape[1] // 2
    R = D // int(nq)
    assert D == R * int(nq), f"Detector count {D} must be divisible by nq={nq}"

    idx = (torch.arange(R, device=frames_xz.device)[:, None] * int(nq) +
           meas_qubits[None, :]).reshape(-1)
    x = frames_xz[:, :D].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    z = frames_xz[:, D:].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    # Z-basis reads X-component, X-basis reads Z-component (anti-commutation)
    return torch.where(meas_bases[None, None, :] == 1, x, z).to(torch.uint8)


def timelike_syndromes(
    frames_xz: torch.Tensor, A: torch.Tensor, meas_old: torch.Tensor
) -> torch.Tensor:
    """
    Apply timelike corrections to measurements using the A matrix.

    A is a linear map over GF(2) producing s2 from frames_xz;
    meas_new = s2 ^ meas_old.

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8
        A: (n_rounds*num_meas, 2*num_detectors) uint8
        meas_old: (batch_size, n_rounds, num_meas) uint8

    Returns:
        meas_new: (batch_size, n_rounds, num_meas) uint8
    """
    s2 = torch.remainder(frames_xz.float() @ A.t().float(), 2).to(torch.uint8)
    return (s2 ^ meas_old.reshape(meas_old.shape[0], -1)).reshape_as(meas_old)
