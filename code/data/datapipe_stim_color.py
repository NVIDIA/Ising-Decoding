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
Stim-based datapipe for color code inference.

This module provides Stim-based data generation for inference/testing with color codes.

Classes:
- QCDataPipePreDecoder_ColorCode_inference: Stim-based inference datapipe for color code
"""

import torch
from torch.utils.data import Dataset

from qec.color_code.color_code import ColorCode
from qec.color_code.detector_input import ColorDetectorInputTransform
from qec.color_code.reference_superdense_noise import build_color_memory_circuit


class QCDataPipePreDecoder_ColorCode_inference(Dataset):
    """
    Datapipe for generating color code data used during inference with stim.
    Torch-only, consistent with training datapipe. Supports 'X' | 'Z' | 'both'.
    
    Key differences from surface code datapipe:
    - Grid size is (n_rows, n_cols) where n_rows = d + (d-1)//2, n_cols = d
    - X and Z stabilizers share the same plaquettes (same grid positions)
    - Number of stabilizers per type = num_plaquettes
    - Measurements per round: 2 * num_plaquettes (Z first, then X)
    """

    def __init__(
        self,
        distance,
        n_rounds,
        num_samples,
        error_mode,
        p_error=0.005,
        measure_basis='X',
        noise_model=None,  # Optional explicit NoiseModel (overrides p_error when provided)
        gidney_style_noise=False,
        noise_model_family='legacy',
        noise_instruction_semantics='current',
        schedule='nearest-neighbor',
    ):
        self.distance = int(distance)
        self.n_rounds = max(int(n_rounds), 1)
        self.num_samples = int(num_samples)
        self.measure_basis = str(measure_basis).upper()

        if error_mode != "circuit_level_color_code":
            raise ValueError(f"error_mode must be 'circuit_level_color_code', got '{error_mode}'")

        # Initialize ColorCode for grid dimensions
        self.code = ColorCode(self.distance)
        self.n_rows = self.code.n_rows
        self.n_cols = self.code.n_cols
        self.num_plaquettes = self.code.num_plaquettes
        self.num_data = self.code.num_data

        self._mixed = self.measure_basis in ("BOTH", "MIXED")
        self._input_transform_X = ColorDetectorInputTransform(
            distance=self.distance,
            rounds=self.n_rounds,
            basis="X",
        )
        self._input_transform_Z = ColorDetectorInputTransform(
            distance=self.distance,
            rounds=self.n_rounds,
            basis="Z",
        )

        # Measurements per round: 2 * num_plaquettes (Z first, then X)
        meas_per_round = 2 * self.num_plaquettes

        if self._mixed:
            # Split shots deterministically 50/50 over samples (even idx -> X, odd idx -> Z)
            self.nX = (self.num_samples + 1) // 2
            self.nZ = self.num_samples // 2

            # X circuit
            self.circ_X = build_color_memory_circuit(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis='X',
                p_error=float(p_error),
                noise_model_family=str(noise_model_family),
                noise_instruction_semantics=str(noise_instruction_semantics),
                noise_model=noise_model,
                gidney_style_noise=gidney_style_noise,
                schedule=str(schedule),
                add_boundary_detectors=True,
            )
            meas_X = self.circ_X.stim_circuit.compile_sampler().sample(shots=self.nX)

            total_ancilla_meas = self.n_rounds * meas_per_round

            # Drop final data-qubit measurements, keep only ancilla measurements
            # Measurements: [round1_Z, round1_X, round2_Z, round2_X, ..., final_data]
            self.meas_X = (
                torch.from_numpy(meas_X[..., :total_ancilla_meas]
                                ).to(torch.uint8).view(self.nX, self.n_rounds,
                                                       meas_per_round).contiguous()
            )

            converter_X = self.circ_X.stim_circuit.compile_m2d_converter()
            self.dets_and_obs_X = torch.from_numpy(
                converter_X.convert(measurements=meas_X, append_observables=True)
            ).to(torch.uint8)

            # Z circuit
            self.circ_Z = build_color_memory_circuit(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis='Z',
                p_error=float(p_error),
                noise_model_family=str(noise_model_family),
                noise_instruction_semantics=str(noise_instruction_semantics),
                noise_model=noise_model,
                gidney_style_noise=gidney_style_noise,
                schedule=str(schedule),
                add_boundary_detectors=True,
            )
            meas_Z = self.circ_Z.stim_circuit.compile_sampler().sample(shots=self.nZ)
            self.meas_Z = (
                torch.from_numpy(meas_Z[..., :total_ancilla_meas]
                                ).to(torch.uint8).view(self.nZ, self.n_rounds,
                                                       meas_per_round).contiguous()
            )
            converter_Z = self.circ_Z.stim_circuit.compile_m2d_converter()
            self.dets_and_obs_Z = torch.from_numpy(
                converter_Z.convert(measurements=meas_Z, append_observables=True)
            ).to(torch.uint8)

            # Precompute the number of main (non-boundary) detectors for each basis.
            # Main detectors = ancilla-based syndrome detectors.
            # Boundary detectors (if any) are appended at the end.
            self._num_main_dets_X = self._count_main_detectors(
                self.circ_X.stim_circuit, 'X', self.n_rounds, self.num_plaquettes
            )
            self._num_main_dets_Z = self._count_main_detectors(
                self.circ_Z.stim_circuit, 'Z', self.n_rounds, self.num_plaquettes
            )

        else:
            self.circ = build_color_memory_circuit(
                distance=self.distance,
                n_rounds=self.n_rounds,
                basis=self.measure_basis,
                p_error=float(p_error),
                noise_model_family=str(noise_model_family),
                noise_instruction_semantics=str(noise_instruction_semantics),
                noise_model=noise_model,
                gidney_style_noise=gidney_style_noise,
                schedule=str(schedule),
                add_boundary_detectors=True,
            )
            meas = self.circ.stim_circuit.compile_sampler().sample(shots=self.num_samples)

            total_ancilla_meas = self.n_rounds * meas_per_round

            self.meas = (
                torch.from_numpy(meas[..., :total_ancilla_meas]
                                ).to(torch.uint8
                                    ).view(self.num_samples, self.n_rounds,
                                           meas_per_round).contiguous()
            )
            converter = self.circ.stim_circuit.compile_m2d_converter()
            self.dets_and_obs = torch.from_numpy(
                converter.convert(measurements=meas, append_observables=True)
            ).to(torch.uint8)

            self._num_main_dets = self._count_main_detectors(
                self.circ.stim_circuit, self.measure_basis, self.n_rounds, self.num_plaquettes
            )

    @staticmethod
    def _count_main_detectors(stim_circuit, basis, n_rounds, num_plaquettes):
        """Count the number of main (non-boundary) syndrome detectors.

        Main detector ordering (matching MemoryCircuit):
        - Round 0: basis-matched only (num_stabs detectors)
        - Rounds 1..R-1: X detectors (num_stabs) then Z detectors (num_stabs)

        Total = num_stabs + (R-1) * 2 * num_stabs = num_stabs * (2*R - 1)
        """
        expected = num_plaquettes * (2 * n_rounds - 1)

        # Validate against total detectors in the (inlined) circuit.
        total_dets = stim_circuit.num_detectors
        # With boundary detectors: total = main + num_plaquettes
        # Without: total = main
        if total_dets not in (expected, expected + num_plaquettes):
            raise RuntimeError(
                f"Unexpected detector count: {total_dets} in circuit, "
                f"expected {expected} or {expected + num_plaquettes}"
            )
        return expected

    def __len__(self):
        return self.num_samples

    def _build_example_from_measurements(
        self, frame: torch.Tensor, dets_and_obs: torch.Tensor, use_basis: str, num_main_dets: int
    ):
        """
        Build a training example using detector events from compile_m2d_converter().

        This replaces the old manual XOR-diff computation, which was incorrect for
        Z-type detectors in inlined (feedforward-absorbed) circuits.

        Args:
            frame: (n_rounds, 2*num_plaquettes) torch.uint8 — raw ancilla measurements (kept for API compat)
            dets_and_obs: (num_detectors + num_observables,) torch.uint8 — from compile_m2d_converter
            use_basis: 'X' or 'Z'
            num_main_dets: number of main (non-boundary) syndrome detectors

        Returns:
            dict with trainX (float32), dets_and_obs
        """
        transform = self._input_transform_X if use_basis == "X" else self._input_transform_Z
        if int(num_main_dets) != int(transform.num_main_dets):
            raise RuntimeError(
                f"Detector unpack mismatch: circuit has {num_main_dets} main detectors, "
                f"transform expects {transform.num_main_dets}"
            )

        dets = dets_and_obs[:transform.detector_width].view(1, transform.detector_width)
        trainX, x_syn_diff, z_syn_diff, _ = transform.build_train_x(dets)

        return {
            "x_syn_diff": x_syn_diff[0].contiguous(),  # (num_stabs, R) int32
            "z_syn_diff": z_syn_diff[0].contiguous(),  # (num_stabs, R) int32
            "trainX": trainX[0].contiguous(),  # (4, T, n_rows, n_cols) float32
            "dets_and_obs": dets_and_obs,  # (num_detectors + num_observables,) uint8
            "meas_flat":
                frame.reshape(-1).contiguous(),  # (R*2*num_plaq,) uint8 — raw ancilla measurements
        }

    def __getitem__(self, idx):
        if self._mixed:
            if (idx % 2) == 0:  # even -> X
                lidx = idx // 2
                frame = self.meas_X[lidx]  # (Tr, 2*num_plaq) uint8
                dets_and_obs = self.dets_and_obs_X[lidx]
                return self._build_example_from_measurements(
                    frame, dets_and_obs, use_basis="X", num_main_dets=self._num_main_dets_X
                )
            else:  # odd -> Z
                lidx = idx // 2
                frame = self.meas_Z[lidx]
                dets_and_obs = self.dets_and_obs_Z[lidx]
                return self._build_example_from_measurements(
                    frame, dets_and_obs, use_basis="Z", num_main_dets=self._num_main_dets_Z
                )
        else:
            frame = self.meas[idx]  # (Tr, 2*num_plaq) uint8
            dets_and_obs = self.dets_and_obs[idx]
            return self._build_example_from_measurements(
                frame,
                dets_and_obs,
                use_basis=self.measure_basis,
                num_main_dets=self._num_main_dets
            )


__all__ = ['QCDataPipePreDecoder_ColorCode_inference']
