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
Convert a .pt checkpoint to a .safetensors file (fp16 or fp32).

Post-processing after training: pre-trained models are in .pt format; use this script
when you need to provide them in SafeTensors format (e.g. for inference or downstream tools).

Usage:
    PYTHONPATH=code python code/export/checkpoint_to_safetensors.py \\
        --checkpoint models/PreDecoderModelMemory_r9_v1.0.77.pt \\
        --model-id 1 [--fp16]

Then run inference with:
    PREDECODER_SAFETENSORS_CHECKPOINT=models/PreDecoderModelMemory_r9_v1.0.77_fp16.safetensors \\
    WORKFLOW=inference DISTANCE=9 N_ROUNDS=9 EXPERIMENT_NAME=predecoder_model_1 \\
    bash code/scripts/local_run.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from export.safetensors_utils import _build_minimal_cfg, save_safetensors
from model.factory import ModelFactory


def _load_checkpoint_state_dict(checkpoint_path: str, device: str) -> dict:
    """
    Load a state dict from a .pt checkpoint, handling multiple saved formats:
    - bare state dict (keys are layer names)
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    Also strips the DDP "module." prefix if present.
    """
    raw = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )  # legacy .pt may contain non-tensor objects

    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            # Assume it is a bare state dict
            state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: expected a dict, got {type(raw).__name__}")

    # Strip DDP "module." prefix if present
    fixed = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        fixed[new_key] = v
    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Convert a .pt pre-decoder checkpoint to SafeTensors format."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the input .pt checkpoint file.",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        required=True,
        help="Public model ID (1..5).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Save model weights in float16 (default: float32).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .safetensors file path. "
            "If not provided, auto-generates '{stem}_fp16.safetensors' or '{stem}_fp32.safetensors' "
            "next to the input checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the checkpoint on (default: cpu).",
    )
    args = parser.parse_args()

    dtype = "fp16" if args.fp16 else "fp32"
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_{dtype}.safetensors"

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = _load_checkpoint_state_dict(str(checkpoint_path), args.device)

    print(f"Building model architecture for model_id={args.model_id} ...")
    cfg = _build_minimal_cfg(args.model_id)
    model = ModelFactory.create_model(cfg).to(args.device)

    model.load_state_dict(state_dict, strict=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if dtype == "fp16":
        model = model.half()
        print("  Converted to float16")

    print(f"Saving to: {output_path}")
    save_safetensors(model, str(output_path), model_id=args.model_id, dtype=dtype)

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"Done. File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
