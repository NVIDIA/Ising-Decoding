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

import os
import re
import glob
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

# import modulus
# from modulus.distributed import DistributedManager
# from modulus.utils.capture import _StaticCapture
# from modulus.launch.logging import PythonLogger
from training.distributed import DistributedManager
from training.capture import _StaticCapture
from training.logging import PythonLogger

# Type aliases for clarity
optimizer = Optimizer
scheduler = LambdaLR
scaler = GradScaler

# Logger instance
checkpoint_logging = PythonLogger("checkpoint")


def _is_external_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "save") and hasattr(model, "load") and hasattr(model, "meta")


def create_directory(filepath):
    """Function to create directories"""
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def should_stop_due_to_time(cfg, epoch_times, current_epoch, rank=0):
    """
    Check if we should stop training due to time constraints.
    
    Args:
        cfg: Configuration object with job timing info
        epoch_times: List of epoch durations in seconds
        current_epoch: Current epoch number
        rank: Process rank (only rank 0 will print messages)
        
    Returns:
        bool: True if we should stop, False otherwise
    """
    if not hasattr(cfg, 'time_based_early_stopping') or not cfg.time_based_early_stopping.enabled:
        return False

    if not hasattr(cfg, 'job_start_timestamp') or not hasattr(cfg, 'job_time_limit_seconds'):
        return False

    if not cfg.job_time_limit_seconds:
        return False

    # Calculate elapsed time since job started
    current_time = time.time()
    elapsed_time = current_time - cfg.job_start_timestamp
    remaining_time = cfg.job_time_limit_seconds - elapsed_time

    # Estimate next epoch time based on recent epochs
    if len(epoch_times) == 0:
        return False  # No data yet

    # Use average of recent epochs (up to last 3) for prediction
    recent_epochs = epoch_times[-3:] if len(epoch_times) >= 3 else epoch_times
    estimated_next_epoch_time = sum(recent_epochs) / len(recent_epochs)

    # Add safety margin
    safety_margin_seconds = cfg.time_based_early_stopping.safety_margin_minutes * 60
    required_time = estimated_next_epoch_time + safety_margin_seconds

    if remaining_time < required_time:
        if rank == 0:
            print(f"\n⏰ TIME-BASED EARLY STOPPING TRIGGERED:")
            print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"   Remaining time: {remaining_time/60:.1f} minutes")
            print(f"   Estimated next epoch: {estimated_next_epoch_time/60:.1f} minutes")
            print(f"   Safety margin: {safety_margin_seconds/60:.1f} minutes")
            print(f"   Required time: {required_time/60:.1f} minutes")
            print(f"   Stopping before epoch {current_epoch + 1} to avoid timeout\n")
        return True

    return False


def compare_receptive_field_with_window_data(cfg):
    """Check if model receptive field is compatible with data window size."""
    R = 1
    for k in cfg.model.kernel_size:
        R += k
    R -= len(cfg.model.kernel_size)

    window_size = min(cfg.distance, cfg.n_rounds)

    if R > window_size:
        print("#" * 50)
        print(f"WARNING: Receptive field {R} is larger than window size {window_size}")
        print("#" * 50)


"""
def dict_to_device(state_dict, device):
    # Function to load dictionary to device
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.to(device)
    return new_state_dict
"""


def dict_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _get_checkpoint_filename(
    path: str,
    base_name: str = "checkpoint",
    index: Union[int, None] = None,
    saving: bool = False,
    model_type: str = "mdlus",
) -> str:
    """Gets the file name /path of checkpoint

    This function has three different ways of providing a checkout filename:
    - If supplied an index this will return the checkpoint name using that index.
    - If index is None and saving is false, this will get the checkpoint with the
    largest index (latest save).
    - If index is None and saving is true, it will return the next valid index file name
    which is calculated by indexing the largest checkpoint index found by one.

    Parameters
    ----------
    path : str
        Path to checkpoints
    base_name: str, optional
        Base file name, by default checkpoint
    index : Union[int, None], optional
        Checkpoint index, by default None
    saving : bool, optional
        Get filename for saving a new checkpoint, by default False
    model_type : str
        Model type, by default "mdlus" for Modulus models and "pt" for PyTorch models


    Returns
    -------
    str
        Checkpoint file name
    """
    # Get model parallel rank so all processes in the first model parallel group
    # can save their checkpoint. In the case without model parallelism,
    # model_parallel_rank should be the same as the process rank itself and
    # only rank 0 saves
    if not DistributedManager.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            checkpoint_logging.warning(
                "`DistributedManager` not initialized; initializing now for multi-GPU checkpointing."
            )
            DistributedManager.initialize()
        else:
            checkpoint_logging.info(
                "`DistributedManager` not initialized; proceeding with single-process checkpointing."
            )
    manager = DistributedManager()
    model_parallel_rank = (
        manager.group_rank("model_parallel") if "model_parallel" in manager.group_names else 0
    )

    # Input file name
    checkpoint_filename = str(Path(path).resolve() / f"{base_name}.{model_parallel_rank}")

    # File extension for Modulus models or PyTorch models
    file_extension = ".mdlus" if model_type == "mdlus" else ".pt"

    # If epoch is provided load that file
    if index is not None:
        checkpoint_filename = checkpoint_filename + f".{index}"
        checkpoint_filename += file_extension
    # Otherwise try loading the latest epoch or rolling checkpoint
    else:
        file_names = [
            Path(fname).name
            for fname in glob.glob(checkpoint_filename + "*" + file_extension, recursive=False)
        ]

        if len(file_names) > 0:
            # If checkpoint from a null index save exists load that
            # This is the most likely line to error since it will fail with
            # invalid checkpoint names
            file_idx = [
                int(re.sub(
                    f"^{base_name}.{model_parallel_rank}.|" + file_extension,
                    "",
                    fname,
                )) for fname in file_names
            ]
            file_idx.sort()
            # If we are saving index by 1 to get the next free file name
            if saving:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]+1}"
            else:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]}"
            checkpoint_filename += file_extension
        else:
            checkpoint_filename += ".0" + file_extension

    return checkpoint_filename


def _unique_model_names(models: List[torch.nn.Module],) -> Dict[str, torch.nn.Module]:
    """Util to clean model names and index if repeat names, will also strip DDP wrappers
    if they exist.

    Parameters
    ----------
    model :  List[torch.nn.Module]
        List of models to generate names for

    Returns
    -------
    Dict[str, torch.nn.Module]
        Dictionary of model names and respective modules
    """
    # Loop through provided models and set up base names
    model_dict = {}
    for model0 in models:
        if hasattr(model0, "module"):
            # Strip out DDP layer
            model0 = model0.module
        # Base name of model is meta.name unless pytorch model
        base_name = model0.__class__.__name__
        # if isinstance(model0, modulus.models.Module):
        if _is_external_model(model0) and hasattr(model0.meta, "name"):
            base_name = model0.meta.name
        # If we have multiple models of the same name, introduce another index
        if base_name in model_dict:
            model_dict[base_name].append(model0)
        else:
            model_dict[base_name] = [model0]

    # Set up unique model names if needed
    output_dict = {}
    for key, model in model_dict.items():
        if len(model) > 1:
            for i, model0 in enumerate(model):
                output_dict[key + str(i)] = model0
        else:
            output_dict[key] = model[0]

    return output_dict


def save_checkpoint(
    path: str,
    models: Union[torch.nn.Module, List[torch.nn.Module], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
    global_step: Optional[int] = None,
) -> None:
    """Training checkpoint saving utility

    This will save a training checkpoint in the provided path following the file naming
    convention "checkpoint.{model parallel id}.{epoch/index}.mdlus". The load checkpoint
    method in Modulus core can then be used to read this file.

    Parameters
    ----------
    path : str
        Path to save the training checkpoint
    models : Union[torch.nn.Module, List[torch.nn.Module], None], optional
        A single or list of PyTorch models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler. Will attempt to save on in static capture if none provided, by
        default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none this will save the checkpoint in the next
        valid index, by default None
    metadata : Optional[Dict[str, Any]], optional
        Additional metadata to save, by default None
    """
    # Create checkpoint directory if it does not exist
    if not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Output directory {path} does not exist, will "
            "attempt to create"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

    # == Saving model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)
        for name, model in models.items():
            # Get model type
            # model_type = "mdlus" if isinstance(model, modulus.models.Module) else "pt"
            model_type = "mdlus" if _is_external_model(model) else "pt"

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, saving=True, model_type=model_type
            )

            # Save state dictionary
            # if isinstance(model, modulus.models.Module):
            if _is_external_model(model):
                model.save(file_name)
            else:
                torch.save(model.state_dict(), file_name)
            checkpoint_logging.success(f"Saved model state dictionary: {file_name}")

    # == Saving training checkpoint ==
    checkpoint_dict = {}
    # Optimizer state dict
    if optimizer:
        checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()

    # Scheduler state dict
    if scheduler:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    # Scheduler state dict
    if scaler:
        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
    # Static capture is being used, save its grad scaler
    if _StaticCapture._amp_scalers:
        checkpoint_dict["static_capture_state_dict"] = _StaticCapture.state_dict()

    # Output file name
    output_filename = _get_checkpoint_filename(path, index=epoch, saving=True, model_type="pt")
    if epoch is not None:
        checkpoint_dict["epoch"] = epoch
    if metadata:
        checkpoint_dict["metadata"] = metadata
    if global_step is not None:
        checkpoint_dict["global_step"] = global_step

    # Save checkpoint to memory
    if bool(checkpoint_dict):
        torch.save(
            checkpoint_dict,
            output_filename,
        )
        checkpoint_logging.success(f"Saved training checkpoint: {output_filename}")


def load_checkpoint(
    path: str,
    models: Union[torch.nn.Module, List[torch.nn.Module], None] = None,
    optimizer: Union[Optimizer, None] = None,
    scheduler: Union[LambdaLR, None] = None,
    scaler: Union[GradScaler, None] = None,
    epoch: Union[int, None] = None,
    metadata_dict: Optional[Dict[str, Any]] = {},
    device: Union[str, torch.device] = "cpu",
    steps_per_epoch_estimate: Optional[int] = None,
    rank: int = 0
) -> Tuple[int, int]:
    """Checkpoint loading utility

    Returns:
    -------
    Tuple[int, int]
        Loaded epoch and global_step
    """
    if not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0, 0

    # If there is nothing to load (fresh run), don't emit scary warnings/errors.
    # Note: `_get_checkpoint_filename(..., index=None, saving=False)` returns a default `.0` path
    # even when the directory is empty, so we explicitly glob for any real checkpoint files.
    ckpt_dir = Path(path)
    has_any_training_ckpt = any(ckpt_dir.glob("checkpoint.*.pt"))

    # === Load model checkpoint(s) ===
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)

        expected_model_files: List[Path] = []
        for name, model in models.items():
            model_type = "mdlus" if _is_external_model(model) else "pt"
            expected_model_files.append(
                Path(_get_checkpoint_filename(path, name, index=epoch, model_type=model_type))
            )
        has_any_model_file = any(p.exists() for p in expected_model_files)
        if not has_any_training_ckpt and not has_any_model_file:
            checkpoint_logging.info(
                f"No checkpoints found in {path}; starting fresh (skipping load)."
            )
            return 0, 0

        for name, model in models.items():
            # model_type = "mdlus" if isinstance(model, modulus.models.Module) else "pt"
            model_type = "mdlus" if _is_external_model(model) else "pt"
            file_name = _get_checkpoint_filename(path, name, index=epoch, model_type=model_type)

            if not Path(file_name).exists():
                checkpoint_logging.warning(f"Missing model file {file_name}, skipping load")
                continue

            # if isinstance(model, modulus.models.Module):
            if _is_external_model(model):
                model.load(file_name)
            else:
                model.load_state_dict(torch.load(file_name, map_location=device, weights_only=True))

            if rank == 0:
                checkpoint_logging.success(f"Loaded model state dict {file_name} to {device}")

    # === Load training state ===
    checkpoint_filename = _get_checkpoint_filename(path, index=epoch, model_type="pt")
    if not Path(checkpoint_filename).is_file():
        # If there were checkpoint files at all, warn; otherwise keep it quiet (fresh run).
        if has_any_training_ckpt:
            checkpoint_logging.warning("Could not find valid checkpoint file, skipping load")
        return 0, 0

    checkpoint_dict = torch.load(checkpoint_filename, map_location=device, weights_only=False)
    if rank == 0:
        checkpoint_logging.success(f"Loaded checkpoint file {checkpoint_filename} to {device}")

    if optimizer and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        if rank == 0:
            checkpoint_logging.success("Loaded optimizer state")

    if scheduler and "scheduler_state_dict" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
        if rank == 0:
            checkpoint_logging.success("Loaded scheduler state")

    if scaler and "scaler_state_dict" in checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
        if rank == 0:
            checkpoint_logging.success("Loaded grad scaler state")

    if "static_capture_state_dict" in checkpoint_dict:
        _StaticCapture.load_state_dict(checkpoint_dict["static_capture_state_dict"])
        checkpoint_logging.success("Loaded static capture state")

    # Load epoch and global step
    epoch = checkpoint_dict.get("epoch", 0)
    global_step = checkpoint_dict.get("global_step")
    if global_step is None:
        if steps_per_epoch_estimate is not None:
            global_step = epoch * steps_per_epoch_estimate
        else:
            global_step = epoch * 1000  # Generic fallback

    # Load metadata
    metadata = checkpoint_dict.get("metadata", {})
    for key, value in metadata.items():
        metadata_dict[key] = value

    return epoch, global_step
