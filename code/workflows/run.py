# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
    elif cfg.workflow.task in ("sampling", "visualize"):
        raise ValueError(
            f"workflow.task={cfg.workflow.task!r} is not supported in the early-access public release. "
            "Supported workflows: train, inference."
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
            print(f"⚠️  Warning: Could not parse epoch from filename {filename}: {e}")
            continue

    if rank == 0:
        print(f"📊 Found {len(model_files)} model files:")
        for filename, epoch in sorted(model_files, key=lambda x: x[1]):
            marker = "👑" if filename == best_file else "  "
            print(f"  {marker} {filename} (epoch {epoch})")

    if best_file is None:
        raise FileNotFoundError(f"❌ No valid PreDecoderModelMemory files found in {path}")

    best_model_path = path + "/" + best_file
    if rank == 0:
        print(f"✅ Selected best model: {best_file} (epoch {max_value})")
        print(f"📁 Full path: {best_model_path}")

    return best_model_path


def _load_model(cfg, dist):
    if dist.rank == 0:
        print(f"🚀 Loading model for task: {cfg.workflow.task}")

    _ensure_inference_io_channels(cfg)
    model = ModelFactory.create_model(cfg).to(dist.device)

    if dist.rank == 0:
        print(f"Model architecture created and moved to device: {dist.device}")

    # Convert model to fp16 if enabled (consistent with training)
    if cfg.enable_fp16:
        model = model.half()
        if dist.rank == 0:
            print(f"Model converted to float16 for fp16 inference")

    model = torch.compile(model, disable=True)

    if dist.rank == 0:
        print(f"Model compilation disabled (for compatibility)")

    # Determine model directory
    # Priority: 1) model_checkpoint_dir (for inference configs)
    #           2) cfg.output/models (for training configs)
    model_checkpoint_dir = getattr(cfg, 'model_checkpoint_dir', None)

    # Determine which model to load based on use_model_checkpoint
    use_checkpoint = getattr(cfg.test, 'use_model_checkpoint', -1)

    if use_checkpoint == -1:
        # Load best model from best_model folder
        if model_checkpoint_dir:
            model_dir = os.path.join(model_checkpoint_dir, "best_model")
        else:
            model_dir = f"{cfg.output}/models/best_model"

        if dist.rank == 0:
            print(f"📂 Loading best model (use_model_checkpoint=-1)")

        # If model_dir is relative, make it absolute
        if not os.path.isabs(model_dir):
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            model_dir = os.path.join(project_root, model_dir)

        if dist.rank == 0:
            print(f"🔍 Resolved model directory: {model_dir}")

        # Fallback: older runs may not create a best_model/ folder; fall back to cfg.output/models.
        if not os.path.isdir(model_dir):
            fallback_dir = model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
            if not os.path.isabs(fallback_dir):
                current_file = os.path.abspath(__file__)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                fallback_dir = os.path.join(project_root, fallback_dir)
            if dist.rank == 0:
                print(f"⚠️  best_model folder not found; falling back to: {fallback_dir}")
            model_dir = fallback_dir

        model_path = find_best_model(model_dir, rank=dist.rank)
    else:
        # Load specific checkpoint from models folder
        if model_checkpoint_dir:
            checkpoint_dir = model_checkpoint_dir
        else:
            checkpoint_dir = f"{cfg.output}/models"

        if dist.rank == 0:
            print(f"📂 Loading checkpoint {use_checkpoint} (use_model_checkpoint={use_checkpoint})")

        # If checkpoint_dir is relative, make it absolute
        if not os.path.isabs(checkpoint_dir):
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            checkpoint_dir = os.path.join(project_root, checkpoint_dir)

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

        if dist.rank == 0:
            print(f"🔍 Resolved checkpoint path: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Checkpoint not found: {model_path}")

    if dist.rank == 0:
        print(f"📥 Loading model parameters from: {model_path}")

    model_params = torch.load(model_path, map_location=dist.device)
    model.load_state_dict(model_params)

    if dist.rank == 0:
        print(f"✅ Model loaded successfully!")
        # Show model size info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {param_count:,}")

    return model


if __name__ == "__main__":
    run()
