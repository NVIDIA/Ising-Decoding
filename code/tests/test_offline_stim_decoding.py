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
"""Tests for offline decoding from Stim detector-sample files."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymatching
import torch
import torch.nn as nn

_repo_code = Path(__file__).resolve().parent.parent
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

from data.datapipe_stim import (
    QCDataPipePreDecoder_Memory_from_stim_file,
    QCDataPipePreDecoder_Memory_inference,
)
from data.predecoder_transform import dets_to_predecoder_inputs
from evaluation.logical_error_rate import count_logical_errors_with_errorbar
from qec.noise_model import NoiseModel
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map,
    compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory,
    normalized_weight_mapping_Zstab_memory,
)
from qec.surface_code.memory_circuit import MemoryCircuit
from qec.surface_code.stim_sample_io import (
    build_stim_sample_metadata,
    read_metadata_json,
    read_stim_detector_samples,
    resolve_stim_sample_paths,
    write_metadata_json,
    write_stim_detector_samples,
)


class _DummyDist:
    rank = 0
    world_size = 1
    device = torch.device("cpu")


class _UnusedModel(nn.Module):

    def forward(self, x):
        raise AssertionError("pymatching_only mode must not run the neural predecoder")


class _ZeroCorrectionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B, _, T, D, _ = x.shape
        return torch.full((B, 4, T, D, D), -1.0, device=x.device) + self.anchor * 0.0


def _build_memory_circuit(distance=3, n_rounds=3, basis="X", rotation="XV", p_error=0.003):
    noise_model = NoiseModel.from_single_p(p_error)
    p_placeholder = float(noise_model.get_max_probability())
    circuit = MemoryCircuit(
        distance=distance,
        idle_error=p_placeholder,
        sqgate_error=p_placeholder,
        tqgate_error=p_placeholder,
        spam_error=(2.0 / 3.0) * p_placeholder,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=rotation,
        noise_model=noise_model,
        add_boundary_detectors=True,
    )
    circuit.set_error_rates()
    return circuit


def _write_artifact(
    root: Path,
    *,
    basis: str,
    shots: int = 128,
    rotation: str = "XV",
    distance: int = 3,
    n_rounds: int = 3,
    p_error: float = 0.003,
    record_noise_fingerprint: bool = False,
    samples_basename: str | None = None,
    metadata_basename: str | None = None,
):
    mem = _build_memory_circuit(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        rotation=rotation,
        p_error=p_error,
    )
    stim_circuit = mem.stim_circuit
    dets_and_obs = stim_circuit.compile_detector_sampler(seed=1234).sample(
        shots=shots,
        append_observables=True,
    )
    dets_and_obs = np.asarray(dets_and_obs, dtype=np.uint8)
    samples_path = root / (samples_basename or f"samples_{basis}.dets")
    metadata_path = root / (metadata_basename or f"metadata_{basis}.json")
    write_stim_detector_samples(
        path=samples_path,
        dets_and_obs=dets_and_obs,
        num_detectors=stim_circuit.num_detectors,
        num_observables=stim_circuit.num_observables,
    )
    noise_kwargs: dict = {}
    if record_noise_fingerprint:
        noise_model = NoiseModel.from_single_p(p_error)
        noise_kwargs.update(
            p_error=float(p_error),
            noise_model_label="25-param",
            noise_model_params=noise_model.canonical_parameters(),
            noise_model_sha256=noise_model.sha256(),
        )
    metadata = build_stim_sample_metadata(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=rotation,
        num_detectors=stim_circuit.num_detectors,
        num_observables=stim_circuit.num_observables,
        num_shots=shots,
        **noise_kwargs,
    )
    write_metadata_json(metadata_path, metadata)

    dem = stim_circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    dets = dets_and_obs[:, :-stim_circuit.num_observables]
    obs = dets_and_obs[:, -stim_circuit.num_observables:]
    predictions = np.asarray(matcher.decode_batch(dets), dtype=np.uint8).reshape(obs.shape)
    errors = int((predictions != obs).sum())
    return SimpleNamespace(
        mem=mem,
        dets_and_obs=dets_and_obs,
        predictions=predictions,
        errors=errors,
        ler=errors / float(shots),
        samples_path=samples_path,
        metadata_path=metadata_path,
    )


def _write_measurement_artifact(
    root: Path,
    *,
    basis: str,
    distance: int,
    n_rounds: int,
    shots: int,
    rotation: str = "XV",
    p_error: float = 0.02,
):
    mem = _build_memory_circuit(
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        rotation=rotation,
        p_error=p_error,
    )
    stim_circuit = mem.stim_circuit
    measurements = stim_circuit.compile_sampler(seed=1234).sample(shots=shots)
    dets_and_obs = stim_circuit.compile_m2d_converter().convert(
        measurements=measurements,
        append_observables=True,
    )
    dets_and_obs = np.asarray(dets_and_obs, dtype=np.uint8)

    samples_path = root / f"samples_{basis}.dets"
    metadata_path = root / f"metadata_{basis}.json"
    write_stim_detector_samples(
        path=samples_path,
        dets_and_obs=dets_and_obs,
        num_detectors=stim_circuit.num_detectors,
        num_observables=stim_circuit.num_observables,
    )
    write_metadata_json(
        metadata_path,
        build_stim_sample_metadata(
            distance=distance,
            n_rounds=n_rounds,
            basis=basis,
            code_rotation=rotation,
            num_detectors=stim_circuit.num_detectors,
            num_observables=stim_circuit.num_observables,
            num_shots=shots,
        ),
    )
    return SimpleNamespace(
        mem=mem,
        measurements=np.asarray(measurements, dtype=np.uint8),
        dets_and_obs=dets_and_obs,
        samples_path=samples_path,
        metadata_path=metadata_path,
    )


def _reference_tensors_from_measurements(
    *,
    measurements: np.ndarray,
    distance: int,
    n_rounds: int,
    basis: str,
    rotation: str,
):
    """Independent oracle: build (x_syn_diff, z_syn_diff, trainX) from raw
    Stim measurements via XOR-differencing.

    This is the *third* implementation by design: it is the slowest, simplest
    one written purely against the surface-code memory experiment convention,
    so it cross-checks both the production dets-based helper and Stim's m2d
    converter at once. Any drift in the production helper or Stim's detector
    emission ordering will surface here.
    """
    D = distance
    T = n_rounds
    shots = measurements.shape[0]
    half = (D * D - 1) // 2
    frames = torch.from_numpy(measurements[..., :-(D * D)]).to(torch.uint8)
    frames = frames.view(shots, T, D * D - 1).contiguous()

    x_raw = frames[:, :, :half].permute(0, 2, 1).contiguous()
    z_raw = frames[:, :, half:].permute(0, 2, 1).contiguous()
    zero_batch = torch.zeros((shots, half, 1), dtype=torch.uint8)
    x_aug = torch.cat([zero_batch, x_raw], dim=2)
    z_aug = torch.cat([zero_batch, z_raw], dim=2)
    x_syn_diff = (x_aug[:, :, 1:] ^ x_aug[:, :, :-1]).to(torch.int32).contiguous()
    z_syn_diff = (z_aug[:, :, 1:] ^ z_aug[:, :, :-1]).to(torch.int32).contiguous()

    w_map_x = normalized_weight_mapping_Xstab_memory(D, rotation).reshape(D, D)
    w_map_z = normalized_weight_mapping_Zstab_memory(D, rotation).reshape(D, D)
    x_present = w_map_x.unsqueeze(0).expand(shots, T, D, D).to(torch.float32)
    z_present = w_map_z.unsqueeze(0).expand(shots, T, D, D).to(torch.float32)
    if basis == "X":
        z_syn_diff[:, :, 0] = 0
        z_syn_diff[:, :, -1] = 0
        z_present = z_present.clone()
        z_present[:, 0] = 0
        z_present[:, -1] = 0
    else:
        x_syn_diff[:, :, 0] = 0
        x_syn_diff[:, :, -1] = 0
        x_present = x_present.clone()
        x_present[:, 0] = 0
        x_present[:, -1] = 0

    idx_map_x = torch.as_tensor(compute_stabX_to_data_index_map(D, rotation), dtype=torch.long)
    idx_map_z = torch.as_tensor(compute_stabZ_to_data_index_map(D, rotation), dtype=torch.long)
    x_grid = torch.zeros(shots, D * D, T, dtype=torch.float32)
    z_grid = torch.zeros(shots, D * D, T, dtype=torch.float32)
    x_grid[:, idx_map_x, :] = x_syn_diff[:, :len(idx_map_x), :].to(torch.float32)
    z_grid[:, idx_map_z, :] = z_syn_diff[:, :len(idx_map_z), :].to(torch.float32)

    x_type = x_grid.reshape(shots, D, D, T).permute(0, 3, 1, 2).contiguous()
    z_type = z_grid.reshape(shots, D, D, T).permute(0, 3, 1, 2).contiguous()
    train_x = torch.cat(
        [
            x_type.unsqueeze(1),
            z_type.unsqueeze(1),
            x_present.unsqueeze(1),
            z_present.unsqueeze(1),
        ],
        dim=1,
    ).contiguous()
    return x_syn_diff, z_syn_diff, train_x


class TestStimSampleFileContract(unittest.TestCase):

    def test_write_read_round_trip_for_x_and_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for basis in ("X", "Z"):
                artifact = _write_artifact(root, basis=basis, shots=64)
                read, metadata = read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis=basis,
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                )
                self.assertTrue(np.array_equal(read, artifact.dets_and_obs))
                self.assertEqual(metadata["basis"], basis)

    def test_file_datapipe_exposes_valid_predecoder_inputs_for_x_and_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for basis in ("X", "Z"):
                artifact = _write_artifact(root, basis=basis, shots=32)
                pipe = QCDataPipePreDecoder_Memory_from_stim_file(
                    distance=3,
                    n_rounds=3,
                    num_samples=32,
                    error_mode="circuit_level_surface_custom",
                    stim_samples_dir=root,
                    p_error=0.003,
                    measure_basis=basis,
                    code_rotation="XV",
                    noise_model=NoiseModel.from_single_p(0.003),
                )
                self.assertEqual(len(pipe), 32)
                self.assertTrue(
                    torch.equal(pipe.dets_and_obs, torch.from_numpy(artifact.dets_and_obs))
                )
                sample = pipe[0]
                self.assertEqual(tuple(sample["trainX"].shape), (4, 3, 3, 3))
                self.assertEqual(
                    sample["dets_and_obs"].numel(),
                    artifact.mem.stim_circuit.num_detectors +
                    artifact.mem.stim_circuit.num_observables,
                )

    def test_file_datapipe_matches_measurement_xor_reference(self):
        """File datapipe (consumes Stim dets) must agree with the independent
        measurement-XOR oracle for several (D, T, basis) triples.

        This is the strongest cross-check: it pins three things at once —
        Stim's detector emission ordering, the canonical helper, and the
        measurement-based reference implementation.
        """
        cases = (
            (3, 3, 32),
            (5, 5, 24),
            (7, 7, 8),
        )
        for distance, n_rounds, shots in cases:
            for basis in ("X", "Z"):
                with self.subTest(distance=distance, n_rounds=n_rounds, basis=basis):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        artifact = _write_measurement_artifact(
                            root,
                            basis=basis,
                            distance=distance,
                            n_rounds=n_rounds,
                            shots=shots,
                        )
                        pipe = QCDataPipePreDecoder_Memory_from_stim_file(
                            distance=distance,
                            n_rounds=n_rounds,
                            num_samples=shots,
                            error_mode="circuit_level_surface_custom",
                            stim_samples_dir=root,
                            p_error=0.02,
                            measure_basis=basis,
                            code_rotation="XV",
                            noise_model=NoiseModel.from_single_p(0.02),
                        )
                        expected_x, expected_z, expected_train_x = _reference_tensors_from_measurements(
                            measurements=artifact.measurements,
                            distance=distance,
                            n_rounds=n_rounds,
                            basis=basis,
                            rotation="XV",
                        )

                        self.assertTrue(
                            torch.equal(pipe.dets_and_obs, torch.from_numpy(artifact.dets_and_obs))
                        )
                        self.assertTrue(torch.equal(pipe.x_syn_diff_all, expected_x))
                        self.assertTrue(torch.equal(pipe.z_syn_diff_all, expected_z))
                        self.assertTrue(torch.equal(pipe.trainX_all, expected_train_x))

    def test_canonical_helper_matches_measurement_xor_reference(self):
        """`dets_to_predecoder_inputs` (the single source of truth for the
        production datapipes) must match the independent measurement-XOR
        oracle directly, without going through file I/O.
        """
        for distance, n_rounds in ((3, 3), (5, 4)):
            for basis in ("X", "Z"):
                with self.subTest(distance=distance, n_rounds=n_rounds, basis=basis):
                    with tempfile.TemporaryDirectory() as tmp:
                        artifact = _write_measurement_artifact(
                            Path(tmp),
                            basis=basis,
                            distance=distance,
                            n_rounds=n_rounds,
                            shots=12,
                        )
                        num_obs = artifact.mem.stim_circuit.num_observables
                        dets = torch.from_numpy(artifact.dets_and_obs[:, :-num_obs]).to(torch.uint8)
                        train_x, x_syn, z_syn = dets_to_predecoder_inputs(
                            dets,
                            distance=distance,
                            n_rounds=n_rounds,
                            basis=basis,
                            code_rotation="XV",
                        )
                        expected_x, expected_z, expected_train_x = _reference_tensors_from_measurements(
                            measurements=artifact.measurements,
                            distance=distance,
                            n_rounds=n_rounds,
                            basis=basis,
                            rotation="XV",
                        )
                        self.assertTrue(torch.equal(x_syn, expected_x))
                        self.assertTrue(torch.equal(z_syn, expected_z))
                        self.assertTrue(torch.equal(train_x, expected_train_x))

    def test_in_memory_datapipe_matches_canonical_helper_on_its_own_dets(self):
        """The in-memory datapipe computes its tensors from the raw
        measurement record, but Stim's m2d converter is supposed to give the
        same syndromes after XOR'ing. Round-trip its `dets_and_obs` through
        the canonical helper and assert numerical equality with the per-sample
        examples the pipe yields from `__getitem__`. Catches any drift between
        the in-memory pipe (SurfaceDetectorInputTransform) and the file pipe.
        """
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                pipe = QCDataPipePreDecoder_Memory_inference(
                    distance=3,
                    n_rounds=3,
                    num_samples=24,
                    error_mode="circuit_level_surface_custom",
                    p_error=0.02,
                    measure_basis=basis,
                    code_rotation="XV",
                    noise_model=NoiseModel.from_single_p(0.02),
                )
                num_obs = pipe.circ.stim_circuit.num_observables
                dets = pipe.dets_and_obs[:, :-num_obs]
                train_x, x_syn, z_syn = dets_to_predecoder_inputs(
                    dets,
                    distance=3,
                    n_rounds=3,
                    basis=basis,
                    code_rotation="XV",
                )
                # The refactored in-memory pipe builds examples lazily in
                # __getitem__ rather than exposing precomputed batch tensors,
                # so stack the per-sample outputs back into batched form.
                pipe_x = torch.stack([pipe[i]["x_syn_diff"] for i in range(len(pipe))])
                pipe_z = torch.stack([pipe[i]["z_syn_diff"] for i in range(len(pipe))])
                pipe_train_x = torch.stack([pipe[i]["trainX"] for i in range(len(pipe))])
                self.assertTrue(torch.equal(x_syn, pipe_x.to(x_syn.dtype)))
                self.assertTrue(torch.equal(z_syn, pipe_z.to(z_syn.dtype)))
                self.assertTrue(torch.equal(train_x, pipe_train_x.to(train_x.dtype)))

    def test_metadata_mismatches_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(root, basis="X", shots=16)
            metadata = read_metadata_json(artifact.metadata_path)
            metadata["basis"] = "Z"
            write_metadata_json(artifact.metadata_path, metadata)

            with self.assertRaisesRegex(ValueError, "basis mismatch.*file has 'Z'.*expected 'X'"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                )

    def test_wrong_orientation_detector_count_and_missing_observable_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(root, basis="X", shots=16)

            metadata = read_metadata_json(artifact.metadata_path)
            metadata["code_rotation"] = "XH"
            write_metadata_json(artifact.metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "code_rotation mismatch"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                )

            metadata["code_rotation"] = "XV"
            metadata["num_detectors"] = metadata["num_detectors"] + 1
            write_metadata_json(artifact.metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "num_detectors mismatch"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                )

            metadata["num_detectors"] = artifact.mem.stim_circuit.num_detectors
            metadata["num_observables"] = 0
            write_metadata_json(artifact.metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, "missing observables"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                )

    def test_p_error_drift_raises_under_strict_noise(self):
        """When metadata records ``p_error`` and the decoder passes a different
        ``p_error``, strict validation must surface the drift instead of
        silently building a wrong-weight matcher.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(
                root,
                basis="X",
                shots=16,
                p_error=0.003,
                record_noise_fingerprint=True,
            )
            with self.assertRaisesRegex(ValueError, "p_error mismatch"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                    p_error=0.005,
                )

    def test_noise_model_sha_drift_raises_under_strict_noise(self):
        """A different :class:`NoiseModel` (even with the same scalar
        ``p_error``) yields a different fingerprint, which must be caught."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(
                root,
                basis="X",
                shots=16,
                p_error=0.003,
                record_noise_fingerprint=True,
            )
            different = NoiseModel.from_single_p(0.003, spam_factor=0.5)
            with self.assertRaisesRegex(ValueError, "noise_model_sha256 mismatch"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                    p_error=0.003,
                    noise_model_sha256=different.sha256(),
                    noise_model_label="25-param",
                )

    def test_noise_drift_downgraded_to_warning_when_strict_disabled(self):
        """``strict_noise=False`` must downgrade the failure to a warning so
        users can override after eyeballing the metadata."""
        import warnings as _warnings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(
                root,
                basis="X",
                shots=16,
                p_error=0.003,
                record_noise_fingerprint=True,
            )
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                    p_error=0.005,
                    strict_noise=False,
                )
            warning_messages = [str(w.message) for w in caught]
            self.assertTrue(
                any("p_error mismatch" in msg for msg in warning_messages),
                f"expected p_error warning, got: {warning_messages}",
            )

    def test_legacy_metadata_without_noise_fields_still_loads(self):
        """Older files that predate the noise fingerprint (no ``p_error`` or
        ``noise_model_sha256`` in metadata) must keep loading even when the
        decoder passes its active fingerprint."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(
                root,
                basis="X",
                shots=8,
                p_error=0.003,
                record_noise_fingerprint=False,
            )
            metadata = read_metadata_json(artifact.metadata_path)
            self.assertNotIn("p_error", metadata)
            self.assertNotIn("noise_model_sha256", metadata)
            data, _ = read_stim_detector_samples(
                samples_path=artifact.samples_path,
                metadata_path=artifact.metadata_path,
                distance=3,
                n_rounds=3,
                basis="X",
                code_rotation="XV",
                num_detectors=artifact.mem.stim_circuit.num_detectors,
                num_observables=artifact.mem.stim_circuit.num_observables,
                p_error=0.005,
                noise_model_sha256=NoiseModel.from_single_p(0.001).sha256(),
            )
            self.assertEqual(data.shape[0], 8)

    def test_per_basis_subdirectory_layout_resolves(self):
        """``{root}/{basis}/samples.dets`` is a valid layout; the resolver
        must find it, and the file datapipe must consume it transparently."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basis = "X"
            (root / basis).mkdir(parents=True, exist_ok=True)
            artifact = _write_artifact(
                root,
                basis=basis,
                shots=8,
                samples_basename=f"{basis}/samples.dets",
                metadata_basename=f"{basis}/metadata.json",
                record_noise_fingerprint=True,
            )
            resolved_samples, resolved_metadata = resolve_stim_sample_paths(root, basis)
            self.assertEqual(resolved_samples, root / basis / "samples.dets")
            self.assertEqual(resolved_metadata, root / basis / "metadata.json")

            pipe = QCDataPipePreDecoder_Memory_from_stim_file(
                distance=3,
                n_rounds=3,
                num_samples=8,
                error_mode="circuit_level_surface_custom",
                stim_samples_dir=root,
                p_error=0.003,
                measure_basis=basis,
                code_rotation="XV",
                noise_model=NoiseModel.from_single_p(0.003),
            )
            self.assertEqual(len(pipe), 8)
            self.assertTrue(torch.equal(pipe.dets_and_obs, torch.from_numpy(artifact.dets_and_obs)))

    def test_truncated_dets_file_raises_at_load_time(self):
        """Truncating the .dets file must trip the num_shots/shape check in
        :func:`read_stim_detector_samples` rather than silently producing
        a partial array (which would skew LER computations)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(
                root,
                basis="X",
                shots=16,
                record_noise_fingerprint=True,
            )
            raw_bytes = artifact.samples_path.read_bytes()
            # Drop the trailing 25% of the file mid-shot. Stim's dets format is
            # newline-delimited, so we deliberately drop bytes after a newline
            # to keep the parser from raising on its own; the shot-count check
            # in read_stim_detector_samples should then notice.
            lines = raw_bytes.splitlines(keepends=True)
            truncated = b"".join(lines[:max(1, len(lines) // 2)])
            artifact.samples_path.write_bytes(truncated)
            with self.assertRaisesRegex(ValueError, "num_shots mismatch"):
                read_stim_detector_samples(
                    samples_path=artifact.samples_path,
                    metadata_path=artifact.metadata_path,
                    distance=3,
                    n_rounds=3,
                    basis="X",
                    code_rotation="XV",
                    num_detectors=artifact.mem.stim_circuit.num_detectors,
                    num_observables=artifact.mem.stim_circuit.num_observables,
                    p_error=0.003,
                    noise_model_sha256=NoiseModel.from_single_p(0.003).sha256(),
                )

    def test_write_rejects_shape_mismatch(self):
        """``write_stim_detector_samples`` must refuse arrays whose width does
        not match ``num_detectors + num_observables``; otherwise the file
        would round-trip into a malformed shape check on read."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bogus = np.zeros((4, 5), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "width mismatch"):
                write_stim_detector_samples(
                    path=root / "samples_X.dets",
                    dets_and_obs=bogus,
                    num_detectors=10,
                    num_observables=1,
                )

    def test_helper_handles_t_equals_one(self):
        """The canonical helper must produce the right tensor shapes for
        ``T = 1`` even though the ONNX-bound eval module asserts ``T >= 2``.
        The file/in-memory datapipes use the helper, so ``T = 1`` flowing
        from a one-round QPU dump must not silently corrupt the residual
        masking on the cross-basis row."""
        distance, n_rounds, shots = 3, 1, 4
        half = (distance * distance - 1) // 2
        torch.manual_seed(13)
        dets = torch.randint(0, 2, (shots, 2 * n_rounds * half), dtype=torch.uint8)
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                train_x, x_syn, z_syn = dets_to_predecoder_inputs(
                    dets,
                    distance=distance,
                    n_rounds=n_rounds,
                    basis=basis,
                    code_rotation="XV",
                )
                self.assertEqual(tuple(train_x.shape), (shots, 4, n_rounds, distance, distance))
                self.assertEqual(tuple(x_syn.shape), (shots, half, n_rounds))
                self.assertEqual(tuple(z_syn.shape), (shots, half, n_rounds))
                # Cross-basis row must be all zero because there is no interior
                # round to populate when T = 1.
                if basis == "X":
                    self.assertTrue(torch.all(z_syn == 0))
                    self.assertTrue(torch.all(train_x[:, 3] == 0))
                else:
                    self.assertTrue(torch.all(x_syn == 0))
                    self.assertTrue(torch.all(train_x[:, 2] == 0))


class TestOfflineStimLER(unittest.TestCase):

    def _base_cfg(self, num_samples: int):
        return SimpleNamespace(
            code="surface",
            datapipe="memory",
            distance=3,
            n_rounds=3,
            data=SimpleNamespace(
                error_mode="circuit_level_surface_custom",
                code_rotation="XV",
            ),
            test=SimpleNamespace(
                num_samples=num_samples,
                n_rounds=3,
                p_error=0.003,
                meas_basis_test="both",
                noise_model="none",
                latency_num_samples=0,
                th_data=0.0,
                th_syn=0.0,
                sampling_mode="threshold",
                temperature=1.0,
                dataloader={
                    "batch_size": 16,
                    "num_workers": 0,
                    "persistent_workers": False,
                },
            ),
        )

    def _run_file_ler(self, root: Path, *, decode_mode: str, model: nn.Module, cfg):
        old_env = {
            "PREDECODER_STIM_SAMPLES_DIR": os.environ.get("PREDECODER_STIM_SAMPLES_DIR"),
            "PREDECODER_DECODE_MODE": os.environ.get("PREDECODER_DECODE_MODE"),
            "PREDECODER_TORCH_COMPILE": os.environ.get("PREDECODER_TORCH_COMPILE"),
            "PREDECODER_INFERENCE_NUM_WORKERS": os.environ.get("PREDECODER_INFERENCE_NUM_WORKERS"),
        }
        os.environ["PREDECODER_STIM_SAMPLES_DIR"] = str(root)
        os.environ["PREDECODER_DECODE_MODE"] = decode_mode
        os.environ["PREDECODER_TORCH_COMPILE"] = "0"
        os.environ["PREDECODER_INFERENCE_NUM_WORKERS"] = "0"
        try:
            return count_logical_errors_with_errorbar(model, torch.device("cpu"), _DummyDist(), cfg)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_pymatching_only_ler_matches_golden_x_and_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {
                basis: _write_artifact(root, basis=basis, shots=96).ler for basis in ("X", "Z")
            }

            result = self._run_file_ler(
                root,
                decode_mode="pymatching_only",
                model=_UnusedModel(),
                cfg=self._base_cfg(96),
            )

            for basis in ("X", "Z"):
                self.assertIn(basis, result)
                observed = float(result[basis]["logical error ratio (pymatch mean)"])
                self.assertEqual(observed, expected[basis])
                self.assertEqual(
                    float(result[basis]["logical error ratio (mean)"]), expected[basis]
                )
                self.assertGreaterEqual(observed, 0.0)
                self.assertLessEqual(observed, 1.0)

    def test_ising_decoding_pymatching_smoke_matches_baseline_for_zero_correction_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {
                basis: _write_artifact(root, basis=basis, shots=32).ler for basis in ("X", "Z")
            }

            result = self._run_file_ler(
                root,
                decode_mode="ising_decoding_pymatching",
                model=_ZeroCorrectionModel(),
                cfg=self._base_cfg(32),
            )

            for basis in ("X", "Z"):
                baseline = float(result[basis]["logical error ratio (pymatch mean)"])
                after = float(result[basis]["logical error ratio (mean)"])
                self.assertEqual(baseline, expected[basis])
                self.assertEqual(after, expected[basis])
                self.assertGreaterEqual(after, 0.0)
                self.assertLessEqual(after, 1.0)

    def test_pymatching_only_single_basis_x_branch(self):
        """Cover the ``meas_basis_test='X'`` branch of
        ``count_logical_errors_with_errorbar`` — distinct from the ``both``
        branch we exercise above, and previously untested for file mode."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = _write_artifact(root, basis="X", shots=64)

            cfg = self._base_cfg(64)
            cfg.test.meas_basis_test = "X"

            result = self._run_file_ler(
                root,
                decode_mode="pymatching_only",
                model=_UnusedModel(),
                cfg=cfg,
            )

            self.assertIn("X", result)
            self.assertNotIn("Z", result)
            observed = float(result["X"]["logical error ratio (pymatch mean)"])
            self.assertEqual(observed, artifact.ler)


if __name__ == "__main__":
    unittest.main()
