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
Config-driven NoiseModel loader.

The legacy frame precompute CLI has been replaced by the Torch
augmented-DEM pipeline in ``qec.precompute_dem``. The remaining helpers below
load a 25p NoiseModel from a YAML / JSON / OmegaConf config; they are reused by
``qec.precompute_dem.main`` (via a deferred import) and by tests.
"""

import json
from pathlib import Path

from qec.noise_model import NoiseModel


def _load_config_mapping(path: str) -> dict:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        with path_obj.open("r", encoding="utf-8") as f:
            return json.load(f)
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path_obj)
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML noise model configs require omegaconf or PyYAML to be installed"
            ) from exc
        with path_obj.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _load_noise_model(path: str) -> NoiseModel:
    cfg = _load_config_mapping(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Noise model config must be a mapping, got {type(cfg).__name__}")
    if isinstance(cfg.get("data"), dict) and isinstance(cfg["data"].get("noise_model"), dict):
        noise_model_cfg = cfg["data"]["noise_model"]
    elif isinstance(cfg.get("noise_model"), dict):
        noise_model_cfg = cfg["noise_model"]
    else:
        noise_model_cfg = cfg
    return NoiseModel.from_config_dict(noise_model_cfg)
