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
Stim-based datapipe for inference.

This module provides Stim-based data generation for inference/testing.

Classes:
- QCDataPipePreDecoder_Memory_inference: Stim-based inference datapipe
"""

import torch
from torch.utils.data import Dataset

from qec.surface_code.memory_circuit import MemoryCircuit
from qec.surface_code.data_mapping import (
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
)


class QCDataPipePreDecoder_Memory_inference(Dataset):
    """
    Datapipe for generating data used during inference with stim.
    Torch-only, consistent with training datapipe. Supports 'X' | 'Z' | 'both'.
    """

    def __init__(
        self,
        distance,
        n_rounds,
        num_samples,
        error_mode,
        p_error=0.005,
        measure_basis='X',
        code_rotation='XV',  # <--- NEW: surface code orientation
        noise_model=None,  # Optional explicit NoiseModel (overrides p_error when provided)
    ):
        self.distance = int(distance)
        self.n_rounds = max(int(n_rounds), 1)
        self.num_samples = int(num_samples)
        self.measure_basis = str(measure_basis).upper()
        self.code_rotation = code_rotation.upper() if code_rotation else 'XV'

        if error_mode != "circuit_level_surface_custom":
            raise ValueError("error_mode not supported")

        D = self.distance
        # (1,1,D,D) as torch
        self.w_mapXgrid = normalized_weight_mapping_Xstab_memory(D, self.code_rotation).reshape(
            D, D
        ).unsqueeze(0).unsqueeze(0)
        self.w_mapZgrid = normalized_weight_mapping_Zstab_memory(D, self.code_rotation).reshape(
            D, D
        ).unsqueeze(0).unsqueeze(0)

        self.stabX_to_data_idx = compute_stabX_to_data_index_map(self.distance, self.code_rotation)
        self.stabZ_to_data_idx = compute_stabZ_to_data_index_map(self.distance, self.code_rotation)
        self._n_stab_x = len(self.stabX_to_data_idx)
        self._n_stab_z = len(self.stabZ_to_data_idx)

        self._mixed = self.measure_basis in ("BOTH", "MIXED")

        # Precompute constants for fast post-Stim transformation in __getitem__
        T = self.n_rounds
        self._half = (D * D - 1) // 2
        self._zero_row = torch.zeros((self._half, 1), dtype=torch.uint8)
        self._idx_map_x = torch.as_tensor(self.stabX_to_data_idx, dtype=torch.long)
        self._idx_map_z = torch.as_tensor(self.stabZ_to_data_idx, dtype=torch.long)

        # Precomputed presence maps (1, T, D, D) with masks applied; no clone/mask in hot path
        w_x = self.w_mapXgrid.expand(1, T, D, D).to(torch.float32).clone()
        w_z = self.w_mapZgrid.expand(1, T, D, D).to(torch.float32).clone()
        self._presence_x_X = w_x.clone()
        self._presence_z_X = w_z.clone()
        self._presence_x_Z = w_x.clone()
        self._presence_z_Z = w_z.clone()
        self._presence_z_X[:, 0] = 0
        self._presence_z_X[:, -1] = 0
        self._presence_x_Z[:, 0] = 0
        self._presence_x_Z[:, -1] = 0

        # If using explicit noise model, use a conservative scalar placeholder for MemoryCircuit's legacy slots.
        if noise_model is not None:
            p_placeholder = float(noise_model.get_max_probability())
        else:
            p_placeholder = float(p_error)

        if self._mixed:
            # Split shots deterministically 50/50 over samples (even idx -> X, odd idx -> Z)
            self.nX = (self.num_samples + 1) // 2
            self.nZ = self.num_samples // 2

            # X circuit
            self.circ_X = MemoryCircuit(
                distance=D,
                idle_error=p_placeholder,
                sqgate_error=p_placeholder,
                tqgate_error=p_placeholder,
                spam_error=(2.0 / 3.0) * p_placeholder,
                n_rounds=self.n_rounds,
                basis='X',
                code_rotation=self.code_rotation,
                noise_model=noise_model,
                add_boundary_detectors=True,  # Required for proper PyMatching decoding
            )
            self.circ_X.set_error_rates()
            meas_X = self.circ_X.stim_circuit.compile_sampler().sample(shots=self.nX)
            # drop final D*D data-qubit measurements and reshape to (shots, n_rounds, D^2-1)
            self.meas_X = (
                torch.from_numpy(meas_X[..., :-(D * D)]
                                ).to(torch.uint8).view(self.nX, self.n_rounds,
                                                       D * D - 1).contiguous()
            )

            converter_X = self.circ_X.stim_circuit.compile_m2d_converter()
            # We pass the FULL measurements, including the data-qubit measurements
            # The m2d converter needs the full measurement record (including the final data-qubit measurements) to compute:
            # 1. All detectors
            # 2. The observable (which depends on the final data qubit measurements)
            self.dets_and_obs_X = torch.from_numpy(
                converter_X.convert(measurements=meas_X, append_observables=True)
            ).to(torch.uint8)

            # Z circuit
            self.circ_Z = MemoryCircuit(
                distance=D,
                idle_error=p_placeholder,
                sqgate_error=p_placeholder,
                tqgate_error=p_placeholder,
                spam_error=(2.0 / 3.0) * p_placeholder,
                n_rounds=self.n_rounds,
                basis='Z',
                code_rotation=self.code_rotation,
                noise_model=noise_model,
                add_boundary_detectors=True,  # Required for proper PyMatching decoding
            )
            self.circ_Z.set_error_rates()
            meas_Z = self.circ_Z.stim_circuit.compile_sampler().sample(shots=self.nZ)
            self.meas_Z = (
                torch.from_numpy(meas_Z[..., :-(D * D)]
                                ).to(torch.uint8).view(self.nZ, self.n_rounds,
                                                       D * D - 1).contiguous()
            )
            converter_Z = self.circ_Z.stim_circuit.compile_m2d_converter()
            self.dets_and_obs_Z = torch.from_numpy(
                converter_Z.convert(measurements=meas_Z, append_observables=True)
            ).to(torch.uint8)

            # Pre-compute all transformations for X and Z batches
            self._precompute_transformations_X()
            self._precompute_transformations_Z()
        else:
            self.circ = MemoryCircuit(
                distance=D,
                idle_error=p_placeholder,
                sqgate_error=p_placeholder,
                tqgate_error=p_placeholder,
                spam_error=(2.0 / 3.0) * p_placeholder,
                n_rounds=self.n_rounds,
                basis=self.measure_basis,
                code_rotation=self.code_rotation,
                noise_model=noise_model,
                add_boundary_detectors=True,  # Required for proper PyMatching decoding
            )
            self.circ.set_error_rates()
            meas = self.circ.stim_circuit.compile_sampler().sample(shots=self.num_samples)
            self.meas = (
                torch.from_numpy(meas[..., :-(D * D)]
                                ).to(torch.uint8).view(self.num_samples, self.n_rounds,
                                                       D * D - 1).contiguous()
            )
            converter = self.circ.stim_circuit.compile_m2d_converter()
            self.dets_and_obs = torch.from_numpy(
                converter.convert(measurements=meas, append_observables=True)
            ).to(torch.uint8)

            # Pre-compute all transformations
            self._precompute_transformations()

    def _precompute_transformations(self):
        """Pre-compute all transformations for all samples (non-mixed case)."""
        D, T = self.distance, self.n_rounds
        half = self._half
        N = self.num_samples

        # Batch process all frames: (N, T, D^2-1) -> (N, half, T) for x and z
        frames = self.meas  # (N, T, D^2-1)
        x_raw = frames[:, :, :half].permute(0, 2, 1).contiguous()  # (N, half, T)
        z_raw = frames[:, :, half:].permute(0, 2, 1).contiguous()  # (N, half, T)

        # XOR diff: add zero frame and diff along time
        zero_batch = torch.zeros((N, half, 1), dtype=torch.uint8)
        x_aug = torch.cat([zero_batch, x_raw], dim=2)  # (N, half, T+1)
        z_aug = torch.cat([zero_batch, z_raw], dim=2)
        x_syn_diff = (x_aug[:, :, 1:] ^ x_aug[:, :, :-1]).to(torch.int32
                                                            ).contiguous()  # (N, half, T)
        z_syn_diff = (z_aug[:, :, 1:] ^ z_aug[:, :, :-1]).to(torch.int32).contiguous()

        # Mask based on basis
        if self.measure_basis == "X":
            z_syn_diff[:, :, 0] = 0
            z_syn_diff[:, :, -1] = 0
            x_present = self._presence_x_X  # (1, T, D, D)
            z_present = self._presence_z_X  # (1, T, D, D)
        else:  # "Z"
            x_syn_diff[:, :, 0] = 0
            x_syn_diff[:, :, -1] = 0
            x_present = self._presence_x_Z  # (1, T, D, D)
            z_present = self._presence_z_Z  # (1, T, D, D)

        # Map to grid: (N, n_stab, T) -> (N, D*D, T) -> (N, T, D, D)
        x_syn_stab = x_syn_diff[:, :self._n_stab_x, :]  # (N, n_stab_x, T)
        z_syn_stab = z_syn_diff[:, :self._n_stab_z, :]  # (N, n_stab_z, T)

        x_grid = torch.zeros(N, D * D, T, dtype=torch.float32)
        z_grid = torch.zeros(N, D * D, T, dtype=torch.float32)
        x_grid[:, self._idx_map_x, :] = x_syn_stab.to(torch.float32)
        z_grid[:, self._idx_map_z, :] = z_syn_stab.to(torch.float32)

        x_type = x_grid.reshape(N, D, D, T).permute(0, 3, 1, 2).contiguous()  # (N, T, D, D)
        z_type = z_grid.reshape(N, D, D, T).permute(0, 3, 1, 2).contiguous()

        # Stack: (N, 4, T, D, D)
        # x_present, z_present: (1, T, D, D) -> expand to (N, T, D, D) -> (N, 1, T, D, D)
        x_present_batch = x_present.expand(N, -1, -1, -1)  # (N, T, D, D)
        z_present_batch = z_present.expand(N, -1, -1, -1)  # (N, T, D, D)
        trainX = torch.cat(
            [
                x_type.unsqueeze(1),  # (N, 1, T, D, D)
                z_type.unsqueeze(1),  # (N, 1, T, D, D)
                x_present_batch.unsqueeze(1),  # (N, 1, T, D, D)
                z_present_batch.unsqueeze(1),  # (N, 1, T, D, D)
            ],
            dim=1
        ).contiguous()

        self.x_syn_diff_all = x_syn_diff  # (N, half, T)
        self.z_syn_diff_all = z_syn_diff  # (N, half, T)
        self.trainX_all = trainX  # (N, 4, T, D, D)

    def _precompute_transformations_X(self):
        """Pre-compute all transformations for X samples (mixed case)."""
        D, T = self.distance, self.n_rounds
        half = self._half
        N = self.nX

        frames = self.meas_X  # (N, T, D^2-1)
        x_raw = frames[:, :, :half].permute(0, 2, 1).contiguous()
        z_raw = frames[:, :, half:].permute(0, 2, 1).contiguous()

        zero_batch = torch.zeros((N, half, 1), dtype=torch.uint8)
        x_aug = torch.cat([zero_batch, x_raw], dim=2)
        z_aug = torch.cat([zero_batch, z_raw], dim=2)
        x_syn_diff = (x_aug[:, :, 1:] ^ x_aug[:, :, :-1]).to(torch.int32).contiguous()
        z_syn_diff = (z_aug[:, :, 1:] ^ z_aug[:, :, :-1]).to(torch.int32).contiguous()

        z_syn_diff[:, :, 0] = 0
        z_syn_diff[:, :, -1] = 0
        x_present = self._presence_x_X  # (1, T, D, D)
        z_present = self._presence_z_X  # (1, T, D, D)

        x_syn_stab = x_syn_diff[:, :self._n_stab_x, :]
        z_syn_stab = z_syn_diff[:, :self._n_stab_z, :]

        x_grid = torch.zeros(N, D * D, T, dtype=torch.float32)
        z_grid = torch.zeros(N, D * D, T, dtype=torch.float32)
        x_grid[:, self._idx_map_x, :] = x_syn_stab.to(torch.float32)
        z_grid[:, self._idx_map_z, :] = z_syn_stab.to(torch.float32)

        x_type = x_grid.reshape(N, D, D, T).permute(0, 3, 1, 2).contiguous()
        z_type = z_grid.reshape(N, D, D, T).permute(0, 3, 1, 2).contiguous()

        x_present_batch = x_present.expand(N, -1, -1, -1)
        z_present_batch = z_present.expand(N, -1, -1, -1)
        trainX = torch.cat(
            [
                x_type.unsqueeze(1),
                z_type.unsqueeze(1),
                x_present_batch.unsqueeze(1),
                z_present_batch.unsqueeze(1),
            ],
            dim=1
        ).contiguous()

        self.x_syn_diff_X = x_syn_diff
        self.z_syn_diff_X = z_syn_diff
        self.trainX_X = trainX

    def _precompute_transformations_Z(self):
        """Pre-compute all transformations for Z samples (mixed case)."""
        D, T = self.distance, self.n_rounds
        half = self._half
        N = self.nZ

        frames = self.meas_Z  # (N, T, D^2-1)
        x_raw = frames[:, :, :half].permute(0, 2, 1).contiguous()
        z_raw = frames[:, :, half:].permute(0, 2, 1).contiguous()

        zero_batch = torch.zeros((N, half, 1), dtype=torch.uint8)
        x_aug = torch.cat([zero_batch, x_raw], dim=2)
        z_aug = torch.cat([zero_batch, z_raw], dim=2)
        x_syn_diff = (x_aug[:, :, 1:] ^ x_aug[:, :, :-1]).to(torch.int32).contiguous()
        z_syn_diff = (z_aug[:, :, 1:] ^ z_aug[:, :, :-1]).to(torch.int32).contiguous()

        x_syn_diff[:, :, 0] = 0
        x_syn_diff[:, :, -1] = 0
        x_present = self._presence_x_Z  # (1, T, D, D)
        z_present = self._presence_z_Z  # (1, T, D, D)

        x_syn_stab = x_syn_diff[:, :self._n_stab_x, :]
        z_syn_stab = z_syn_diff[:, :self._n_stab_z, :]

        x_grid = torch.zeros(N, D * D, T, dtype=torch.float32)
        z_grid = torch.zeros(N, D * D, T, dtype=torch.float32)
        x_grid[:, self._idx_map_x, :] = x_syn_stab.to(torch.float32)
        z_grid[:, self._idx_map_z, :] = z_syn_stab.to(torch.float32)

        x_type = x_grid.reshape(N, D, D, T).permute(0, 3, 1, 2).contiguous()
        z_type = z_grid.reshape(N, D, D, T).permute(0, 3, 1, 2).contiguous()

        x_present_batch = x_present.expand(N, -1, -1, -1)
        z_present_batch = z_present.expand(N, -1, -1, -1)
        trainX = torch.cat(
            [
                x_type.unsqueeze(1),
                z_type.unsqueeze(1),
                x_present_batch.unsqueeze(1),
                z_present_batch.unsqueeze(1),
            ],
            dim=1
        ).contiguous()

        self.x_syn_diff_Z = x_syn_diff
        self.z_syn_diff_Z = z_syn_diff
        self.trainX_Z = trainX

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Fast indexing into pre-computed transformations."""
        if self._mixed:
            if (idx % 2) == 0:  # even -> X
                lidx = idx // 2
                return {
                    "x_syn_diff": self.x_syn_diff_X[lidx],  # (half, T)
                    "z_syn_diff": self.z_syn_diff_X[lidx],  # (half, T)
                    "trainX": self.trainX_X[lidx],  # (4, T, D, D)
                    "dets_and_obs": self.dets_and_obs_X[lidx],  # (num_detectors + num_observables,)
                }
            else:  # odd -> Z
                lidx = idx // 2
                return {
                    "x_syn_diff": self.x_syn_diff_Z[lidx],
                    "z_syn_diff": self.z_syn_diff_Z[lidx],
                    "trainX": self.trainX_Z[lidx],
                    "dets_and_obs": self.dets_and_obs_Z[lidx],
                }
        else:
            return {
                "x_syn_diff": self.x_syn_diff_all[idx],  # (half, T)
                "z_syn_diff": self.z_syn_diff_all[idx],  # (half, T)
                "trainX": self.trainX_all[idx],  # (4, T, D, D)
                "dets_and_obs": self.dets_and_obs[idx],  # (num_detectors + num_observables,)
            }


__all__ = ['QCDataPipePreDecoder_Memory_inference']
