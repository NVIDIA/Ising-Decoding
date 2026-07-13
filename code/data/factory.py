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
Factory module for creating datapipes.

Provides DatapipeFactory for instantiating data generators/datapipes from config.
"""
import os

import torch


class DatapipeFactory:
    """
    Factory for creating datapipes.
    
    Training: Uses Torch on-the-fly generation in training.train
    Inference: Uses Stim-based datapipes
    """

    @staticmethod
    def create_datapipe(cfg):
        """Create datapipe for training - always returns None for on-the-fly mode."""
        if cfg.code == "surface":
            return DatapipeFactory._create_surface_datapipe(cfg)
        elif cfg.code == "color":
            return DatapipeFactory._create_color_datapipe(cfg)
        else:
            raise ValueError("Invalid datapipe code")

    @staticmethod
    def create_datapipe_inference(cfg):
        """Create datapipe for inference using Stim."""
        if cfg.code == "surface":
            return DatapipeFactory._create_surface_datapipe_inference(cfg)
        elif cfg.code == "color":
            return DatapipeFactory._create_color_datapipe_inference(cfg)
        else:
            raise ValueError("Invalid datapipe code")

    @staticmethod
    def _create_surface_datapipe(cfg):
        """
        Datapipe for training - on-the-fly generation only.
        
        Returns (None, None) to signal on-the-fly mode - train.py will create
        the generators directly.
        """
        if cfg.datapipe == "memory":
            # On-the-fly data generation
            # No datasets needed - will create generators directly in train.py
            return None, None
        else:
            raise ValueError(f"Datapipe not implemented: {cfg.datapipe}")

    @staticmethod
    def _create_color_datapipe(cfg):
        """Color training data is generated directly by training.train."""
        if cfg.datapipe == "memory":
            return None, None
        else:
            raise ValueError(f"Datapipe not implemented: {cfg.datapipe}")

    @staticmethod
    def _create_surface_datapipe_inference(cfg):
        """
        Datapipe for inference using Stim.
        """
        if cfg.datapipe == "memory":
            from data.datapipe_stim import (
                QCDataPipePreDecoder_Memory_from_stim_file,
                QCDataPipePreDecoder_Memory_inference,
            )
            from qec.noise_model import resolve_test_noise_model

            error_mode_value = getattr(cfg.data, 'error_mode', 'circuit_level_surface_custom')
            code_rotation = getattr(cfg.data, 'code_rotation', 'XV')
            noise_model_obj, test_nm_mode = resolve_test_noise_model(cfg)

            # Only print from rank 0 in distributed settings
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except:
                rank = 0

            if rank == 0:
                print(
                    f"Creating Stim inference datapipe: d={cfg.distance}, n_rounds={cfg.n_rounds}, "
                    f"num_samples={cfg.test.num_samples}, error_mode={error_mode_value}, "
                    f"test.noise_model={test_nm_mode}, p_error={cfg.test.p_error}, "
                    f"measure_basis={cfg.test.meas_basis_test}, code_rotation={code_rotation}"
                )
                if noise_model_obj is not None:
                    print(f"[Inference] Using explicit noise_model (25p): {noise_model_obj!r}")

            stim_samples_dir = os.environ.get("PREDECODER_STIM_SAMPLES_DIR", "").strip()
            if not stim_samples_dir:
                stim_samples_dir = str(getattr(cfg.test, "stim_samples_dir", "") or "").strip()
            measure_basis = cfg.test.meas_basis_test
            if stim_samples_dir and str(measure_basis).upper() in ("BOTH", "MIXED"):
                # The file pipe holds one basis per instance; callers that need
                # both bases re-instantiate per basis (the LER loop does this).
                # Pick X for shape probing here; choice does not affect tensor
                # shapes because they only depend on (distance, n_rounds).
                measure_basis = "X"

            dataset_kwargs: dict = {}
            if stim_samples_dir:
                dataset_cls = QCDataPipePreDecoder_Memory_from_stim_file
                dataset_kwargs["stim_samples_dir"] = stim_samples_dir
                # PREDECODER_STIM_STRICT_NOISE=0 downgrades the noise-fingerprint
                # mismatch from an error to a UserWarning. Default is strict.
                strict_env = os.environ.get("PREDECODER_STIM_STRICT_NOISE", "").strip().lower()
                if strict_env in ("0", "false", "no", "off"):
                    dataset_kwargs["strict_noise"] = False
            else:
                dataset_cls = QCDataPipePreDecoder_Memory_inference

            test_dataset = dataset_cls(
                distance=cfg.distance,
                n_rounds=cfg.n_rounds,
                num_samples=cfg.test.num_samples,
                error_mode=error_mode_value,
                p_error=cfg.test.p_error,
                measure_basis=measure_basis,
                code_rotation=code_rotation,
                noise_model=noise_model_obj,
                **dataset_kwargs,
            )
            return test_dataset
        else:
            raise ValueError(f"Datapipe not implemented: {cfg.datapipe}")

    @staticmethod
    def _create_color_datapipe_inference(cfg):
        """Datapipe for color-code inference/testing using Stim + Chromobius-compatible circuits."""
        if cfg.datapipe == "memory":
            from data.datapipe_stim_color import QCDataPipePreDecoder_ColorCode_inference
            from qec.noise_model import (
                normalize_noise_instruction_semantics,
                normalize_noise_model_family,
                resolve_test_noise_model,
            )

            test_cfg = getattr(cfg, "test", None)
            family = normalize_noise_model_family(
                getattr(test_cfg, "noise_model_family", None),
                fallback_noise_mode=getattr(test_cfg, "noise_mode", None),
            )
            semantics = normalize_noise_instruction_semantics(
                getattr(test_cfg, "noise_instruction_semantics", None)
            )
            gidney_style_noise = bool(getattr(test_cfg, "gidney_style_noise", False))
            if semantics == "reference":
                noise_model_obj = None
                test_nm_mode = "reference"
            else:
                noise_model_obj, test_nm_mode = resolve_test_noise_model(cfg)

            schedule = getattr(cfg.data, "schedule", "nearest-neighbor")
            error_mode_value = getattr(cfg.data, 'error_mode', 'circuit_level_color_code')

            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except:
                rank = 0

            if rank == 0:
                print(
                    f"Creating color Stim inference datapipe: d={cfg.distance}, "
                    f"n_rounds={cfg.n_rounds}, num_samples={cfg.test.num_samples}, "
                    f"error_mode={error_mode_value}, test.noise_model={test_nm_mode}, "
                    f"p_error={cfg.test.p_error}, measure_basis={cfg.test.meas_basis_test}, "
                    f"schedule={schedule}, noise_family={family}, semantics={semantics}"
                )
                if noise_model_obj is not None:
                    print(
                        f"[Color Inference] Using explicit noise_model (25p): "
                        f"{noise_model_obj!r}"
                    )

            return QCDataPipePreDecoder_ColorCode_inference(
                distance=cfg.distance,
                n_rounds=cfg.n_rounds,
                num_samples=cfg.test.num_samples,
                error_mode=error_mode_value,
                p_error=cfg.test.p_error,
                measure_basis=cfg.test.meas_basis_test,
                noise_model=noise_model_obj,
                gidney_style_noise=gidney_style_noise,
                noise_model_family=family,
                noise_instruction_semantics=semantics,
                schedule=schedule,
            )
        else:
            raise ValueError(f"Datapipe not implemented: {cfg.datapipe}")


# Utility function for debugging
def inspect_sample(sample, label: str):
    """
    Inspect a sample from the datapipe for debugging.
    
    Args:
        sample: dict from the datapipe with keys 'x_syn_diff', 'z_syn_diff', 'trainX'
        label: "X" or "Z" (what we expect for this sample)
    """
    x_syn_diff = sample["x_syn_diff"]  # (Sx, T) int32
    z_syn_diff = sample["z_syn_diff"]  # (Sz, T) int32
    trainX = sample["trainX"]  # (4, T, D, D) float32

    assert trainX.ndim == 4 and trainX.shape[0] == 4, f"Unexpected trainX shape: {trainX.shape}"
    assert trainX.dtype == torch.float32, f"Unexpected dtype for trainX: {trainX.dtype}"
    C, T, D, _ = trainX.shape

    # Channels: [0]=x_type, [1]=z_type, [2]=x_present, [3]=z_present
    x_type = trainX[0]
    z_type = trainX[1]
    x_pres = trainX[2]
    z_pres = trainX[3]

    # Basis masks for sanity
    mask_is_X = (z_pres[-1] == 0).all().item()
    mask_is_Z = (x_pres[-1] == 0).all().item()

    print(f"\n=== Sample ({label}) ===")
    print(f"trainX: shape={tuple(trainX.shape)}, dtype={trainX.dtype}")
    print(f"x_syn_diff (Sx,T) shape={tuple(x_syn_diff.shape)}, dtype={x_syn_diff.dtype}")
    print(f"z_syn_diff (Sz,T) shape={tuple(z_syn_diff.shape)}, dtype={z_syn_diff.dtype}")

    # Sanity checks against expected basis
    if label == "X":
        assert mask_is_X and not mask_is_Z, "Expected X-basis sample: z_present last round should be zeros."
    else:
        assert mask_is_Z and not mask_is_X, "Expected Z-basis sample: x_present last round should be zeros."

    # Basic shape expectations
    Sx, T_x = x_syn_diff.shape
    Sz, T_z = z_syn_diff.shape
    assert T_x == T_z == T, "Time dimension mismatch between syn diffs and trainX."
    expected_half = (D * D - 1) // 2
    assert Sx == expected_half and Sz == expected_half, \
        f"Unexpected stabilizer count: Sx={Sx}, Sz={Sz}, expected={expected_half}"

    print("OK ✓")
