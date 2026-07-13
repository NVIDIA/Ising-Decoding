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
from qec.surface_code.detector_input import SurfaceDetectorInputTransform
from qec.surface_code.stim_sample_io import read_stim_detector_samples, resolve_stim_sample_paths
from data.predecoder_transform import dets_to_predecoder_inputs


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
        self._input_transform_X = SurfaceDetectorInputTransform(
            distance=self.distance,
            rounds=self.n_rounds,
            basis="X",
            rotation=self.code_rotation,
        )
        self._input_transform_Z = SurfaceDetectorInputTransform(
            distance=self.distance,
            rounds=self.n_rounds,
            basis="Z",
            rotation=self.code_rotation,
        )

        self._mixed = self.measure_basis in ("BOTH", "MIXED")

        # If using explicit noise model, use a conservative scalar placeholder for MemoryCircuit's legacy slots.
        # (Actual probabilities come from noise_model.)
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

    def __len__(self):
        return self.num_samples

    def _build_example_from_detector_stream(
        self,
        _frame: torch.Tensor,
        dets_and_obs: torch.Tensor,
        use_basis: str,
    ):
        """Build a model example from the detector stream consumed by the decoder."""
        transform = self._input_transform_X if use_basis == "X" else self._input_transform_Z
        if dets_and_obs.numel() < transform.detector_width:
            raise RuntimeError(
                f"Detector vector has {dets_and_obs.numel()} values, "
                f"expected at least {transform.detector_width}"
            )

        dets = dets_and_obs[:transform.detector_width].view(1, transform.detector_width)
        trainX, x_syn_diff, z_syn_diff, _ = transform.build_train_x(dets)

        return {
            "x_syn_diff": x_syn_diff[0].contiguous(),  # (Sx, T) int32
            "z_syn_diff": z_syn_diff[0].contiguous(),  # (Sz, T) int32
            "trainX": trainX[0].contiguous(),  # (4, T, D, D) float32
            "dets_and_obs": dets_and_obs,  # (num_detectors + num_observables,) uint8
        }

    def __getitem__(self, idx):
        if self._mixed:
            if (idx % 2) == 0:  # even -> X
                lidx = idx // 2
                frame = self.meas_X[lidx]  # (Tr, D^2-1) uint8
                dets_and_obs = self.dets_and_obs_X[lidx]  # (num_detectors + num_observables,) uint8
                return self._build_example_from_detector_stream(frame, dets_and_obs, use_basis="X")
            else:  # odd -> Z
                lidx = idx // 2
                frame = self.meas_Z[lidx]
                dets_and_obs = self.dets_and_obs_Z[lidx]
                return self._build_example_from_detector_stream(frame, dets_and_obs, use_basis="Z")
        else:
            frame = self.meas[idx]  # (Tr, D^2-1) uint8
            dets_and_obs = self.dets_and_obs[idx]
            return self._build_example_from_detector_stream(
                frame,
                dets_and_obs,
                use_basis=self.measure_basis,
            )


class QCDataPipePreDecoder_Memory_from_stim_file(Dataset):
    """
    Datapipe for offline inference from Stim detector-sample files.

    The file stores detector events plus appended observables. Metadata is
    validated against a freshly rebuilt MemoryCircuit before data is exposed.

    Noise-model validation: when ``noise_model`` is provided (the typical
    inference path), the datapipe computes a deterministic fingerprint of its
    25-parameter dict and asks :func:`read_stim_detector_samples` to compare it
    against the value recorded in the JSON metadata. Mismatches raise unless
    ``strict_noise`` is ``False`` (in which case a warning is emitted). When
    ``noise_model`` is ``None``, only the scalar ``p_error`` is checked.

    Args:
        distance, n_rounds, num_samples, error_mode, measure_basis,
            code_rotation: Standard circuit parameters; ``num_samples`` may
            truncate the loaded file to the first N shots when positive.
        stim_samples_dir: Directory containing ``samples_{basis}.dets`` and
            ``metadata_{basis}.json``.
        p_error: Scalar physical error rate used by the active config. Compared
            against ``metadata['p_error']`` when present.
        noise_model: Optional explicit :class:`NoiseModel`. When set, its
            ``sha256()`` is compared against ``metadata['noise_model_sha256']``.
        strict_noise: ``True`` (default) raises on noise-fingerprint drift;
            ``False`` downgrades the failure to a :class:`UserWarning`.
    """

    def __init__(
        self,
        distance,
        n_rounds,
        num_samples,
        error_mode,
        stim_samples_dir,
        p_error=0.005,
        measure_basis='X',
        code_rotation='XV',
        noise_model=None,
        strict_noise: bool = True,
    ):
        self.distance = int(distance)
        self.n_rounds = max(int(n_rounds), 1)
        self.measure_basis = str(measure_basis).upper()
        self.code_rotation = code_rotation.upper() if code_rotation else 'XV'
        self.requested_num_samples = int(num_samples) if num_samples is not None else 0

        if self.measure_basis not in ("X", "Z"):
            raise ValueError(
                "Stim file datapipe expects one basis at a time. "
                f"Got measure_basis={measure_basis!r}."
            )
        if error_mode != "circuit_level_surface_custom":
            raise ValueError("error_mode not supported")

        D = self.distance
        if noise_model is not None:
            p_placeholder = float(noise_model.get_max_probability())
            noise_sha = noise_model.sha256()
            noise_label = "25-param"
        else:
            p_placeholder = float(p_error)
            noise_sha = None
            noise_label = "simple"

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
            add_boundary_detectors=True,
        )
        self.circ.set_error_rates()

        samples_path, metadata_path = resolve_stim_sample_paths(
            stim_samples_dir, self.measure_basis
        )
        dets_and_obs, metadata = read_stim_detector_samples(
            samples_path=samples_path,
            metadata_path=metadata_path,
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=self.measure_basis,
            code_rotation=self.code_rotation,
            num_detectors=self.circ.stim_circuit.num_detectors,
            num_observables=self.circ.stim_circuit.num_observables,
            p_error=float(p_error),
            noise_model_sha256=noise_sha,
            noise_model_label=noise_label,
            strict_noise=bool(strict_noise),
        )
        if self.requested_num_samples > 0:
            dets_and_obs = dets_and_obs[:self.requested_num_samples]

        self.samples_path = samples_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.dets_and_obs = torch.from_numpy(dets_and_obs).to(torch.uint8).contiguous()
        self.num_samples = int(self.dets_and_obs.shape[0])
        self._half = (D * D - 1) // 2

        self._precompute_transformations_from_dets()

    def _precompute_transformations_from_dets(self):
        num_obs = self.circ.stim_circuit.num_observables
        dets = self.dets_and_obs[:, :-num_obs].contiguous()
        train_x, x_syn_diff, z_syn_diff = dets_to_predecoder_inputs(
            dets,
            distance=self.distance,
            n_rounds=self.n_rounds,
            basis=self.measure_basis,
            code_rotation=self.code_rotation,
        )
        self.x_syn_diff_all = x_syn_diff
        self.z_syn_diff_all = z_syn_diff
        self.trainX_all = train_x

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "x_syn_diff": self.x_syn_diff_all[idx],
            "z_syn_diff": self.z_syn_diff_all[idx],
            "trainX": self.trainX_all[idx],
            "dets_and_obs": self.dets_and_obs[idx],
        }


__all__ = [
    'QCDataPipePreDecoder_Memory_inference',
    'QCDataPipePreDecoder_Memory_from_stim_file',
]
