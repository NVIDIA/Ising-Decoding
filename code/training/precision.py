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
"""Precision helpers for training and validation."""

from contextlib import nullcontext
from typing import Optional

import torch


def validate_precision_flags(enable_fp16: bool, enable_bf16: bool) -> None:
    """Ensure precision modes are mutually exclusive."""
    if enable_fp16 and enable_bf16:
        raise ValueError("enable_fp16 and enable_bf16 are mutually exclusive")


def get_amp_dtype(enable_fp16: bool, enable_bf16: bool) -> Optional[torch.dtype]:
    """Return the autocast dtype for mixed-precision training."""
    validate_precision_flags(enable_fp16, enable_bf16)
    if enable_fp16:
        return torch.float16
    if enable_bf16:
        return torch.bfloat16
    return None


def _device_type(device) -> str:
    if hasattr(device, "type"):
        return device.type
    return str(device).split(":", 1)[0]


def autocast_for_precision(device, enable_fp16: bool, enable_bf16: bool):
    """Create an autocast context without changing parameter or optimizer dtype."""
    amp_dtype = get_amp_dtype(enable_fp16, enable_bf16)
    if amp_dtype is None:
        return nullcontext()

    device_type = _device_type(device)
    if device_type == "cpu" and amp_dtype is torch.float16:
        # CPU fp16 autocast is not the training target and is poorly supported.
        return nullcontext()

    return torch.amp.autocast(device_type=device_type, dtype=amp_dtype)


def targets_for_bce(targets: torch.Tensor) -> torch.Tensor:
    """BCE targets should stay in fp32 even when the forward pass is autocast."""
    return targets.float()


def should_use_grad_scaler(enable_fp16: bool, device) -> bool:
    """GradScaler is required for CUDA fp16 AMP and unnecessary for bf16/fp32."""
    return bool(enable_fp16 and _device_type(device) == "cuda" and torch.cuda.is_available())


def should_use_channels_last_3d(enable_fp16: bool, enable_bf16: bool, device) -> bool:
    """Decide whether to run the 5D Conv3D stack in channels-last (NDHWC) layout.

    On CUDA, half-precision (fp16/bf16) ``Conv3d`` in the default contiguous
    NCDHW layout dispatches to a dramatically slower kernel (measured ~30ms per
    conv on A100 / cuDNN 9.13 vs ~0.05ms for fp32). Converting the model and its
    inputs to ``torch.channels_last_3d`` restores Tensor-Core-friendly kernels,
    making fp16/bf16 forward comparable to (or faster than) fp32. This is a no-op
    benefit for fp32, so we only enable it when autocast is active.
    """
    return bool(
        (enable_fp16 or enable_bf16) and _device_type(device) == "cuda" and
        torch.cuda.is_available()
    )


def module_to_channels_last_3d(module, enabled: bool):
    """Convert a module's 5D parameters/buffers to channels_last_3d when enabled."""
    if enabled:
        return module.to(memory_format=torch.channels_last_3d)
    return module


def input_to_channels_last_3d(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Convert a 5D input batch to channels_last_3d when enabled (no-op otherwise)."""
    if enabled and tensor.dim() == 5:
        return tensor.to(memory_format=torch.channels_last_3d)
    return tensor


def model_is_channels_last_3d(model) -> bool:
    """True if the model's 5D conv weights are stored in channels_last_3d layout.

    Works for plain modules and ``torch.compile``-wrapped (OptimizedModule)
    models, since both expose ``parameters()``.
    """
    try:
        params = model.parameters()
    except AttributeError:
        return False
    for p in params:
        if p.dim() == 5:
            return bool(p.is_contiguous(memory_format=torch.channels_last_3d))
    return False


def match_input_to_model_memory_format(tensor: torch.Tensor, model) -> torch.Tensor:
    """Lay out a 5D eval input to match a channels_last_3d model.

    Inference/eval paths build inputs in the default contiguous (NCDHW) layout.
    If the model runs in channels_last_3d, a contiguous half-precision input
    forces the slow Conv3D kernel; converting the input keeps eval on the fast
    Tensor-Core path and consistent with training. No-op for contiguous models.

    Skipped during ONNX export: the legacy exporter cannot lower
    ``contiguous(memory_format=channels_last_3d)``, and ONNX has no
    memory-format concept — layout only affects kernel dispatch, not values.
    """
    if getattr(torch.onnx, "is_in_onnx_export", lambda: False)():
        return tensor
    if tensor.dim() == 5 and model_is_channels_last_3d(model):
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor
