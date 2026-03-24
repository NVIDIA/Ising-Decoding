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

import torch

_STIM_INFERENCE_DATAPIPE_PRINTED = False


class DatapipeFactory:
    """
    Factory for creating datapipes.
    
    Training: Uses on-the-fly generation (train.py creates generators directly)
    Inference: Uses Stim-based QCDataPipePreDecoder_Memory_inference
    """

    @staticmethod
    def create_datapipe(cfg):
        """Create datapipe for training - returns None to signal generator mode."""
        if cfg.code == "surface":
            return DatapipeFactory._create_surface_datapipe(cfg)
        else:
            raise ValueError("Invalid datapipe code")

    @staticmethod
    def create_datapipe_inference(cfg):
        """Create datapipe for inference using Stim."""
        if cfg.code == "surface":
            return DatapipeFactory._create_surface_datapipe_inference(cfg)
        else:
            raise ValueError("Invalid datapipe code")

    @staticmethod
    def _create_surface_datapipe(cfg):
        """
        Datapipe for training - on-the-fly generation only.
        
        Returns (None, None) to signal generator mode - train.py will create
        generators directly.
        """
        if cfg.datapipe == "memory":
            # No datasets needed - will create generators directly in train.py
            return None, None
        else:
            raise ValueError(f"Datapipe not implemented: {cfg.datapipe}")

    @staticmethod
    def _create_surface_datapipe_inference(cfg):
        """
        Datapipe for inference using Stim.
        """
        if cfg.datapipe == "memory":
            from data.datapipe_stim import QCDataPipePreDecoder_Memory_inference
            from qec.noise_model import NoiseModel

            error_mode_value = getattr(cfg.data, 'error_mode', 'circuit_level_surface_custom')
            code_rotation = getattr(cfg.data, 'code_rotation', 'XV')
            # Test-time noise model selection:
            # - cfg.test.noise_model='train': use cfg.data.noise_model (if present)
            # - cfg.test.noise_model='none': ignore cfg.data.noise_model, use cfg.test.p_error (legacy single-p)
            # Takes priority over cfg.test.p_error.
            test_nm_mode = getattr(getattr(cfg, "test", None), "noise_model", None)
            if test_nm_mode is None:
                # Backwards-compat default: use training noise model if specified, else none.
                test_nm_mode = "train"
            test_nm_mode = str(test_nm_mode).lower()

            noise_model_obj = None
            if test_nm_mode == "train":
                noise_model_cfg = getattr(cfg.data, "noise_model", None)
                if noise_model_cfg is not None:
                    from omegaconf import OmegaConf
                    nm_dict = OmegaConf.to_container(noise_model_cfg, resolve=True) if hasattr(
                        noise_model_cfg, "items"
                    ) else noise_model_cfg
                    if nm_dict is not None:
                        noise_model_obj = NoiseModel.from_config_dict(dict(nm_dict))
            elif test_nm_mode == "none":
                noise_model_obj = None
            else:
                raise ValueError(
                    f"Invalid cfg.test.noise_model={test_nm_mode!r} (expected 'train' or 'none')"
                )

            # Fail fast: if the user provided an explicit 25p noise model and asked to use it,
            # do not silently fall back to legacy p_error-based generation.
            if test_nm_mode == "train" and getattr(
                cfg.data, "noise_model", None
            ) is not None and noise_model_obj is None:
                raise ValueError(
                    "cfg.test.noise_model='train' but failed to construct NoiseModel from cfg.data.noise_model. "
                    "Refusing to fall back to legacy cfg.test.p_error."
                )

            test_dataset = QCDataPipePreDecoder_Memory_inference(
                distance=cfg.distance,
                n_rounds=cfg.n_rounds,
                num_samples=cfg.test.num_samples,
                error_mode=error_mode_value,
                p_error=cfg.test.p_error,
                measure_basis=cfg.test.meas_basis_test,
                code_rotation=code_rotation,
                noise_model=noise_model_obj,
            )
            return test_dataset
        else:
            raise ValueError(f"Datapipe not implemented: {cfg.datapipe}")
