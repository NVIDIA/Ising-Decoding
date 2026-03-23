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

import hydra, sys, torch, os, json, numpy as np
from omegaconf import DictConfig, OmegaConf
from training.train import main as train_main
from model.factory import ModelFactory
from data.factory import DatapipeFactory
from hydra.utils import to_absolute_path
from workflows.config_validator import (
    apply_public_defaults_and_model,
    validate_public_config,
)

from training.distributed import DistributedManager

from torch.utils.data import DataLoader


def _ensure_inference_io_channels(cfg):
    # 1) Ensure out_channels matches the model’s heads (4: z_data, x_data, syn_x, syn_z)
    if not getattr(cfg.model, "out_channels", None) or cfg.model.out_channels == 0:
        cfg.model.out_channels = 4

    # 2) Infer input_channels from a single inference sample if not set
    if not getattr(cfg.model, "input_channels", None) or cfg.model.input_channels == 0:
        ds = DatapipeFactory.create_datapipe_inference(cfg)
        tmp = DataLoader(ds, batch_size=1)
        sample = next(iter(tmp))
        cfg.model.input_channels = int(sample["trainX"].shape[1])

    # 3) Keep num_filters consistent with out_channels
    if hasattr(cfg.model, "num_filters"):
        filters = list(cfg.model.num_filters)
        if filters and filters[-1] != cfg.model.out_channels:
            print(
                f"[run] Adjusting model.num_filters[-1] {filters[-1]} -> {cfg.model.out_channels}"
            )
            filters[-1] = cfg.model.out_channels
            cfg.model.num_filters = filters


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # Early-access public release: validate public surface, then merge in hidden defaults.
    # NOTE: Validation is done BEFORE merging defaults so we can fail fast on injected fields.
    model_spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, model_spec)

    torch.backends.cuda.matmul.allow_tf32 = cfg.enable_matmul_tf32
    torch.backends.cudnn.allow_tf32 = cfg.enable_cudnn_tf32

    if cfg.code == "surface" or cfg.code == "surface_partition":
        run_surface(cfg)


def run_surface(cfg: DictConfig):
    if cfg.workflow.task == "train":
        train_main(cfg)
    elif cfg.workflow.task == "threshold":
        raise ValueError(
            "workflow.task='threshold' has been renamed to workflow.task='inference'. "
            "Please update your config/env var to WORKFLOW=inference."
        )
    elif cfg.workflow.task == "inference":
        from evaluation.inference import run_inference
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        run_inference(model, dist.device, dist, cfg)
    elif cfg.workflow.task == "data":
        DistributedManager.initialize()
        dist = DistributedManager()
        train_loader, _ = DatapipeFactory.create_dataloader(cfg, dist.world_size, dist.rank)
        for j, dl in enumerate(train_loader):
            print(f"Batch {j}: syndrome_shape: {dl['syndrome'].shape}")
    elif cfg.workflow.task == "decoder_ablation":
        from evaluation.failure_analysis import decoder_ablation_study
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        decoder_ablation_study(model, dist.device, dist, cfg)
    elif cfg.workflow.task in ("sampling", "visualize"):
        raise ValueError(
            f"workflow.task={cfg.workflow.task!r} is not supported in the early-access public release. "
            "Supported workflows: train, inference, decoder_ablation."
        )


def find_best_model(path, *, rank: int = 0):
    if rank == 0:
        print(f"Searching for best model in: {path}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    max_value = -1  # Start with -1 to include epoch 0
    best_file = None
    model_files = []

    for filename in os.listdir(path):
        if not filename.startswith("PreDecoderModelMemory_"):
            continue
        try:
            value = float(filename.split(".")[2])  # Gets epoch number
            model_files.append((filename, value))
            if value > max_value:
                max_value = value
                best_file = filename
        except (IndexError, ValueError) as e:
            print(f"Warning: could not parse epoch from filename {filename}: {e}")
            continue

    if rank == 0:
        print(f"Found {len(model_files)} model files:")
        for filename, epoch in sorted(model_files, key=lambda x: x[1]):
            marker = "*" if filename == best_file else " "
            print(f"  [{marker}] {filename} (epoch {epoch})")

    if best_file is None:
        raise FileNotFoundError(f"No valid PreDecoderModelMemory files found in {path}")

    best_model_path = os.path.join(path, best_file)
    if rank == 0:
        print(f"Selected best model: {best_file} (epoch {max_value})")

    return best_model_path


def _resolve_dir(path: str) -> str:
    """Return an absolute version of path, resolving relative paths from the repo root."""
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, path)


def _load_state_dict_from_pt(model_path: str, device) -> dict:
    """Load a state dict from a .pt checkpoint, handling multiple saved formats.

    Supports:
    - bare state dict (keys are layer names)
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    Also strips the DDP "module." prefix if present.
    """
    raw = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: expected a dict, got {type(raw).__name__}")
    return {
        (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }


def _load_model(cfg, dist):
    if dist.rank == 0:
        print(f"Loading model for task: {cfg.workflow.task}")

    _ensure_inference_io_channels(cfg)

    # SafeTensors path: load fp16/fp32 model from SafeTensors file
    safetensors_path = os.environ.get("PREDECODER_SAFETENSORS_CHECKPOINT", "").strip()
    if safetensors_path:
        from export.safetensors_utils import load_safetensors
        if dist.rank == 0:
            print(f"Loading model from SafeTensors: {safetensors_path}")

        # Auto-detect model_id from SafeTensors metadata (don't override with config)
        model, metadata = load_safetensors(
            safetensors_path,
            model_id=None,
            device=str(dist.device),
        )
        if dist.rank == 0:
            loaded_model_id = metadata.get("model_id", "unknown")
            dtype = metadata.get("quant_format", "fp32")
            receptive_field = metadata.get("receptive_field", "unknown")
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  model_id: {loaded_model_id} (from SafeTensors metadata)")
            print(f"  receptive_field: {receptive_field}")
            print(f"  dtype: {dtype}")
            print(f"  parameters: {param_count:,}")

            # Warn if config model_id doesn't match file metadata
            config_model_id = getattr(cfg, "model_id", None)
            if config_model_id is not None and str(config_model_id) != str(loaded_model_id):
                print(
                    f"  Warning: config model_id={config_model_id} differs from "
                    f"file model_id={loaded_model_id}; using {loaded_model_id}"
                )

        if metadata.get("quant_format") == "fp16":
            cfg.enable_fp16 = True
        return model

    model = ModelFactory.create_model(cfg).to(dist.device)

    if cfg.enable_fp16:
        model = model.half()
        if dist.rank == 0:
            print("Model converted to float16 for fp16 inference")

    # Determine model directory
    # Priority: 1) model_checkpoint_dir (for inference configs)
    #           2) cfg.output/models (for training configs)
    model_checkpoint_dir = getattr(cfg, 'model_checkpoint_dir', None)
    use_checkpoint = getattr(cfg.test, 'use_model_checkpoint', -1)

    if use_checkpoint == -1:
        model_dir = _resolve_dir(
            os.path.join(model_checkpoint_dir, "best_model")
            if model_checkpoint_dir else f"{cfg.output}/models/best_model"
        )
        if dist.rank == 0:
            print(f"Loading best model from: {model_dir}")

        # Fallback: older runs may not have a best_model/ folder
        if not os.path.isdir(model_dir):
            fallback_dir = _resolve_dir(
                model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
            )
            if dist.rank == 0:
                print(f"best_model/ not found; falling back to: {fallback_dir}")
            model_dir = fallback_dir

        model_path = find_best_model(model_dir, rank=dist.rank)
    else:
        checkpoint_dir = _resolve_dir(
            model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
        )
        if dist.rank == 0:
            print(f"Loading checkpoint {use_checkpoint} from: {checkpoint_dir}")

        # Prefer any PreDecoderModelMemory_* file ending with .0.{use_checkpoint}.pt
        target_suffix = f".0.{use_checkpoint}.pt"
        checkpoint_filename = None
        try:
            for f in os.listdir(checkpoint_dir):
                if f.startswith("PreDecoderModelMemory_") and f.endswith(target_suffix):
                    checkpoint_filename = f
                    break
        except OSError:
            pass
        if checkpoint_filename is None:
            checkpoint_filename = f"PreDecoderModelMemory_v1.0.{use_checkpoint}.pt"
        model_path = os.path.join(checkpoint_dir, checkpoint_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if dist.rank == 0:
        print(f"Loading model parameters from: {model_path}")

    state_dict = _load_state_dict_from_pt(model_path, dist.device)
    model.load_state_dict(state_dict)

    if dist.rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model loaded ({param_count:,} parameters)")

    return model


if __name__ == "__main__":
    run()
