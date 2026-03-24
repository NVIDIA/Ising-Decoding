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
SafeTensors save/load utilities for fp16/fp32 pre-decoder models.

No quantization or ModelOpt dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from model.registry import get_model_spec
from model.factory import ModelFactory
from workflows.config_validator import apply_public_defaults_and_model
from omegaconf import OmegaConf


def _build_minimal_cfg(model_id: int):
    """Build a minimal inference config for model_id without a full Hydra setup."""
    spec = get_model_spec(model_id)
    cfg = OmegaConf.create(
        {
            "model_id": model_id,
            "distance": spec.receptive_field,
            "n_rounds": spec.receptive_field,
            "data": {
                "code_rotation": "XV"
            },
        }
    )
    return apply_public_defaults_and_model(cfg, spec)


def save_safetensors(
    model: torch.nn.Module,
    path: str,
    model_id: int,
    dtype: str = "fp32",
    extra_metadata: Optional[dict] = None,
) -> None:
    """
    Save a pre-decoder model to a SafeTensors file.

    Args:
        model: The model to save (should already be on cpu or target device).
        path: Output file path (e.g. "model_fp32.safetensors").
        model_id: Public model ID (1..5).
        dtype: "fp32" or "fp16".
        extra_metadata: Optional dict of additional string metadata to embed.
    """
    if dtype not in ("fp32", "fp16"):
        raise ValueError(f"dtype must be 'fp32' or 'fp16', got: {dtype!r}")

    spec = get_model_spec(model_id)

    metadata = {
        "model_id": str(model_id),
        "quant_format": dtype,
        "model_version": spec.model_version,
        "receptive_field": str(spec.receptive_field),
        "num_filters": json.dumps(list(spec.num_filters)),
        "kernel_size": json.dumps(list(spec.kernel_size)),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            metadata[str(k)] = str(v)

    save_file(model.state_dict(), path, metadata=metadata)


def load_safetensors(
    safetensors_path: str,
    model_id: Optional[int] = None,
    device: str = "cuda",
):
    """
    Load a pre-decoder model from a SafeTensors file.

    Args:
        safetensors_path: Path to the .safetensors file.
        model_id: If provided, overrides the model_id stored in metadata.
        device: Target device string (e.g. "cuda", "cpu", "cuda:0").

    Returns:
        (model, metadata) where model is the loaded nn.Module and metadata is a dict.
    """
    # Read metadata without loading tensors
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata())

    # Resolve model_id
    if model_id is None:
        if "model_id" not in metadata:
            raise ValueError(
                f"SafeTensors file has no 'model_id' in metadata and model_id was not provided: "
                f"{safetensors_path}"
            )
        model_id = int(metadata["model_id"])
    else:
        model_id = int(model_id)

    cfg = _build_minimal_cfg(model_id)
    model = ModelFactory.create_model(cfg).to(device)

    if metadata.get("quant_format") == "fp16":
        model = model.half()

    state_dict = load_file(safetensors_path, device=device)
    model.load_state_dict(state_dict, strict=True)

    return model, metadata
