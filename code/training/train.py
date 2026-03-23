# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Training module for quantum error correction pre-decoder.

This module provides training functionality with on-the-fly data generation.
All file-based dataset and epoch-config paths have been removed.

V3 Training Configuration Reference
====================================

Noise model scaling:
  Surface-code training noise upscaling via get_training_upscaled_noise_model.
  cfg.data.skip_noise_upscaling  (bool, default: false)
  PREDECODER_SKIP_NOISE_UPSCALING  (0/1, default: 0)
      Skips upscaling entirely; training uses exact user-specified parameters.
  Evaluation/inference always uses the user-specified noise model as-is.

Performance:
  cfg.torch_compile          (bool, default: true)  -- torch.compile on model
  PREDECODER_TORCH_COMPILE   (0/1/auto)             -- env override
  PREDECODER_TORCH_COMPILE_MODE                     -- compile mode override
  Faster EMA via torch._foreach_mul_/_foreach_add_ (auto-fallback on old PyTorch).
  gc.collect()/empty_cache() removed (100-400ms stall per epoch).

SDR threshold gate (early stopping requires minimum SDR):
  cfg.syndrome_density_threshold  (float, default: 1.5x or 33.3%)
  cfg.sdr_as_percent              (bool, default: false)
      false = threshold is a factor (e.g. 1.5 means 1.5x reduction)
      true  = threshold is a percentage (e.g. 33.3 means 33.3% reduction)
  When SDR is not computed, the gate is bypassed (original behavior).

HE acceleration (forwarded to QCDataGeneratorTorch):
  cfg.data.use_compile, cfg.data.compile_chunk_size, cfg.data.compute_dtype,
  cfg.data.use_weight2, cfg.data.max_passes_w2, cfg.data.use_coset_search,
  cfg.data.coset_max_generators, cfg.data.use_dense_overlap

Logging:
  Compact epoch summary: loss | LER | SDR | wall time | throughput
  TensorBoard scalars: Speed/throughput_samp_per_s, Speed/ms_per_batch,
  Speed/epoch_wall_s
"""
import time
import sys
import os
import re
import gc
import math
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
try:
    import torchinfo
except Exception:  # pragma: no cover - optional dependency for summaries
    torchinfo = None
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional for training logs
    SummaryWriter = None

from training.distributed import DistributedManager

from training.utils import (
    load_checkpoint, save_checkpoint, create_directory, should_stop_due_to_time,
    compare_receptive_field_with_window_data
)
from model.factory import ModelFactory

# Import optimizers
from training.optimizers import Lion, DebugLion, get_lr_scheduler

# Import evaluation metrics for validation
from evaluation.metrics import (
    configure_metrics,
    compute_syndrome_density,
    compute_validation_ler,
    HAS_LER_MODULE,
)

# Load mapping functions for the data
from qec.surface_code.data_mapping import (
    compute_stabX_to_data_index_map, compute_stabZ_to_data_index_map,
    normalized_weight_mapping_Xstab_memory, normalized_weight_mapping_Zstab_memory,
    reshape_Xstabilizers_to_grid_vectorized, reshape_Zstabilizers_to_grid_vectorized
)


def _missing_dem_artifacts(frames_dir, distance, n_rounds, bases):
    if frames_dir is None:
        return False
    d = Path(frames_dir)
    if not d.exists():
        return True
    for basis in bases:
        prefix = f"surface_d{int(distance)}_r{int(n_rounds)}_{str(basis).upper()}_frame_predecoder"
        hx_path = d / f"{prefix}.X.npz"
        hz_path = d / f"{prefix}.Z.npz"
        p_path = d / f"{prefix}.p.npz"
        if not (hx_path.exists() and hz_path.exists() and p_path.exists()):
            return True
    return False


def resolve_precomputed_frames_dir(precomputed_frames_dir, distance, n_rounds, meas_basis, rank):
    bases_needed = ["X", "Z"
                   ] if str(meas_basis).lower() in ("both", "mixed") else [str(meas_basis).upper()]
    if _missing_dem_artifacts(precomputed_frames_dir, distance, n_rounds, bases_needed):
        if int(rank) == 0:
            print(
                "[Train] Precomputed DEM artifacts not found. Falling back to in-memory DEM generation. "
                "To precompute, run: python code/data/precompute_frames.py "
                f"--distance {int(distance)} --n_rounds {int(n_rounds)} --basis X --basis Z"
            )
        return None
    return precomputed_frames_dir


def get_current_per_device_batch_size(epoch, cfg):
    """
    Get current per-device batch size based on epoch-based schedule.
    
    Args:
        epoch: Current epoch number (0-indexed)
        cfg: Config with batch_schedule settings
        
    Semantics:
        - start_epoch: After this epoch completes, start ramping (epoch 0 = first epoch)
        - end_epoch: After this epoch completes, reach final batch size
        - Schedule is linear with values rounded to nearest multiple of 8
    """
    if not cfg.batch_schedule.enabled:
        return cfg.batch_schedule.initial

    start_epoch = cfg.batch_schedule.start_epoch
    end_epoch = cfg.batch_schedule.end_epoch
    initial = cfg.batch_schedule.initial
    final = cfg.batch_schedule.final

    # Before start_epoch completes, use initial batch size
    if epoch <= start_epoch:
        return initial

    # After end_epoch completes, use final batch size
    if epoch > end_epoch:
        return final

    # Linear interpolation between start_epoch and end_epoch
    # epoch is current epoch (0-indexed), we ramp during epochs (start_epoch+1) to end_epoch
    progress = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    raw_batch_size = initial + (final - initial) * progress

    # Round to nearest multiple of 8 for nice GPU utilization
    batch_size = int(round(raw_batch_size / 8) * 8)

    # Clamp to valid range
    return max(min(batch_size, final), initial)


def get_accumulate_steps(epoch, cfg):
    """
    Get gradient accumulation steps based on current epoch.
    
    When batch scheduling is enabled, accumulation increases proportionally
    to keep effective batch size = per_device_batch_size * accumulate_steps.
    """
    # If user asked for no accumulation, bail early
    if cfg.train.accumulate_steps == 1:
        return 1

    # If batch scheduling is off, use the static accumulate_steps
    if not cfg.batch_schedule.enabled:
        return cfg.train.accumulate_steps

    # Otherwise compute dynamic accumulation from the schedule
    current_per_device_batch_size = get_current_per_device_batch_size(epoch, cfg)
    per_device_batch_size = cfg.batch_schedule.initial
    unbounded_accumulate = current_per_device_batch_size // per_device_batch_size
    return min(max(unbounded_accumulate, 1), cfg.train.accumulate_steps)


def get_curriculum_batch_sizes(cfg, epoch, num_pairs):
    """
    Get per-(d, n_rounds) batch sizes for curriculum learning.
    
    Returns a list of batch sizes, one per pair, based on linear interpolation
    between initial_batch and final_batch over the scheduled epoch range.
    """
    curriculum = getattr(cfg, 'curriculum_schedule', None)

    if curriculum is None or not getattr(curriculum, 'enabled', False):
        return None

    pairs_config = getattr(curriculum, 'pairs', None)
    if pairs_config is None or len(pairs_config) != num_pairs:
        print(
            f"[Curriculum] Warning: pairs config length ({len(pairs_config) if pairs_config else 0}) "
            f"!= num_pairs ({num_pairs}). Falling back to global batch_schedule."
        )
        return None

    end_epoch = getattr(curriculum, 'end_epoch', 10)
    progress = min(max(epoch / max(1, end_epoch), 0.0), 1.0)

    batch_sizes = []
    for pair_cfg in pairs_config:
        initial = pair_cfg.get('initial_batch', 64)
        final = pair_cfg.get('final_batch', 64)
        batch_size = int(initial + (final - initial) * progress)
        batch_sizes.append(max(batch_size, 1))

    return batch_sizes


def calculate_curriculum_steps_per_epoch(batch_sizes, num_samples, world_size):
    """
    Calculate steps per epoch when using curriculum learning with variable batch sizes.
    """
    num_pairs = len(batch_sizes)
    steps_per_cycle = num_pairs * 2  # Each pair has X and Z basis
    samples_per_cycle = sum(bs * 2 * world_size for bs in batch_sizes)
    num_cycles = math.ceil(num_samples / samples_per_cycle)
    total_steps = num_cycles * steps_per_cycle

    return total_steps, samples_per_cycle, num_cycles


def validation_step(
    generator, model, num_samples, batch_size, device, enable_fp16, enable_bf16=False, rank=0
):
    """Validation using on-the-fly data generation."""
    loss_fn = torch.nn.BCEWithLogitsLoss()
    running_vloss = 0.0
    data_gen_s = 0.0
    model_s = 0.0

    if isinstance(batch_size, (list, tuple)):
        num_batches, samples_per_cycle, num_cycles = calculate_curriculum_steps_per_epoch(
            batch_size, num_samples, world_size=1
        )
        curriculum_mode = True
    else:
        num_batches = (num_samples + batch_size - 1) // batch_size
        curriculum_mode = False

    val_start_time = time.time()
    if rank == 0:
        if curriculum_mode:
            print(
                f"[Validation] Starting validation with generator, {num_batches} batches (curriculum mode)..."
            )
        else:
            print(f"[Validation] Starting validation with generator, {num_batches} batches...")
        # Explicitly state whether the validation generator is using a noise model.
        # Even with a noise_model, the simulator may still have a fixed scalar p placeholder
        # (used for buffer sizing), so we detect noise_model by the presence of the object itself.
        use_nm = False
        try:
            sim_x = getattr(generator, "sim_X", None)
            sim_z = getattr(generator, "sim_Z", None)
            sim = getattr(generator, "sim", None)
            sims = [s for s in (sim_x, sim_z, sim) if s is not None]
            use_nm = bool(getattr(generator, "noise_model", None)
                         ) or any(getattr(s, "noise_model", None) is not None for s in sims)
            if use_nm:
                nm = None
                for s in sims:
                    nm = getattr(s, "noise_model", None)
                    if nm is not None:
                        break
                if nm is None:
                    nm = getattr(generator, "noise_model", None)
                if nm is not None:
                    print(f"[Validation] Using explicit noise_model (25p): {nm!r}")
        except Exception as e:
            print(f"[Validation] (noise_model log skipped due to error: {e})")

        # Only print legacy scalar-p settings when no explicit noise_model is active.
        if not use_nm:
            if hasattr(generator, 'sim_X') and hasattr(generator, 'sim_Z'):
                # Check if it's a Torch generator (no fixed_p attribute)
                if hasattr(generator.sim_X, 'fixed_p'):
                    print(f"[Validation] Generator p settings - X: ", end='')
                    if generator.sim_X.fixed_p:
                        print(f"p={generator.sim_X.p_min:.6f} (fixed)")
                    else:
                        print(f"p∈[{generator.sim_X.p_min:.6f}, {generator.sim_X.p_max:.6f}]")
                    print(f"[Validation] Generator p settings - Z: ", end='')
                    if generator.sim_Z.fixed_p:
                        print(f"p={generator.sim_Z.p_min:.6f} (fixed)")
                    else:
                        print(f"p∈[{generator.sim_Z.p_min:.6f}, {generator.sim_Z.p_max:.6f}]")
                else:
                    print(f"[Validation] Using Torch generator (p from precomputed DEM)")
            elif hasattr(generator, 'sim'):
                if hasattr(generator.sim, 'fixed_p'):
                    print(f"[Validation] Generator p settings: ", end='')
                    if generator.sim.fixed_p:
                        print(f"p={generator.sim.p_min:.6f} (fixed)")
                    else:
                        print(f"p∈[{generator.sim.p_min:.6f}, {generator.sim.p_max:.6f}]")
                else:
                    print(f"[Validation] Using Torch generator (p from precomputed DEM)")

    with torch.no_grad():
        for step in range(num_batches):
            val_step = 10000 + step
            t_gen_start = time.perf_counter()
            trainX, trainY = generator.generate_batch(step=val_step, batch_size=batch_size)
            data_gen_s += time.perf_counter() - t_gen_start

            if step < 8 and rank == 0 and hasattr(generator, 'get_current_pair'):
                d, r = generator.get_current_pair(val_step)
                basis = 'X' if (val_step % 2 == 0) else 'Z'
                print(
                    f"[Val Batch {step}] Using (d={d}, r={r}, basis={basis}) | trainX shape: {trainX.shape}"
                )

            if enable_fp16:
                trainX = trainX.half()
                trainY = trainY.half()
            elif enable_bf16:
                trainX = trainX.to(torch.bfloat16)
                trainY = trainY.to(torch.bfloat16)
            trainX = trainX.to(device, non_blocking=True)
            trainY = trainY.to(device, non_blocking=True)

            if enable_fp16:
                autocast_dtype = torch.float16
            elif enable_bf16:
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float32

            t_model_start = time.perf_counter()
            with autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(trainX)
                loss = loss_fn(outputs, trainY)
            model_s += time.perf_counter() - t_model_start

            running_vloss += loss.item()

    avg_vloss = running_vloss / num_batches
    val_time = time.time() - val_start_time

    if rank == 0:
        time_per_batch = val_time / num_batches
        print(
            f"[Validation] Completed {num_batches} batches in {val_time:.1f}s "
            f"({time_per_batch*1000:.1f}ms/batch), avg_vloss={avg_vloss:.5f}"
        )
        print(f"[Validation Timing] data_gen={data_gen_s:.2f}s, model={model_s:.2f}s")

    return avg_vloss, {
        "data_gen_s": data_gen_s,
        "model_s": model_s,
        "num_batches": num_batches,
        "total_s": val_time
    }


def train_epoch(
    generator,
    steps_per_epoch,
    batch_size,
    cumulative_steps_before_epoch,
    epoch_number,
    model,
    optimizer,
    scaler,
    scheduler,
    tb_writer,
    device,
    enable_fp16,
    enable_bf16=False,
    rank=0,
    use_ema=False,
    ema_model=None,
    ema_decay=0.0,
    global_step=0,
    accumulate_steps=1,
):
    """Training epoch using on-the-fly data generation."""
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    gradient_clipping_counter = 0

    epoch_start_time = time.time()
    last_log_time = epoch_start_time
    data_gen_s = 0.0
    model_s = 0.0

    if rank == 0:
        print(f"train_epoch: Starting {steps_per_epoch} batches...")

    running_loss = 0.0
    last_loss = 0.0
    epoch_total_loss = 0.0
    accumulated_samples = 0

    for step in range(steps_per_epoch):
        if rank == 0 and (step < 2 or step % 200 == 0):
            current_time = time.time()
            elapsed = current_time - epoch_start_time
            if step > 0:
                time_per_batch = (current_time - last_log_time) / min(step, 200)
                remaining = time_per_batch * (steps_per_epoch - step)
                display_loss = epoch_total_loss / step if step > 0 else 0.0
                warmup_note = " (warmup; ETA overestimated)" if step == 1 else ""
                print(
                    f"[Epoch {epoch_number}] Batch {step}/{steps_per_epoch} | "
                    f"Loss: {display_loss:.5f} | "
                    f"Elapsed: {elapsed:.1f}s | Per-batch: {time_per_batch*1000:.1f}ms | "
                    f"ETA: {remaining:.1f}s{warmup_note}"
                )
            else:
                print(
                    f"[Epoch {epoch_number}] Batch {step}/{steps_per_epoch} | Elapsed: {elapsed:.1f}s"
                )
            if step % 200 == 0 and step > 0:
                last_log_time = current_time

        global_step_for_gen = cumulative_steps_before_epoch + step
        t_gen_start = time.perf_counter()
        trainX, trainY = generator.generate_batch(step=global_step_for_gen, batch_size=batch_size)
        data_gen_s += time.perf_counter() - t_gen_start

        if step < 8 and rank == 0 and hasattr(generator, 'get_current_pair'):
            d, r = generator.get_current_pair(global_step_for_gen)
            basis = 'X' if (global_step_for_gen % 2 == 0) else 'Z'
            print(
                f"[Batch {step}] Using (d={d}, r={r}, basis={basis}) | trainX shape: {trainX.shape}"
            )

        trainX = trainX.to(device, non_blocking=True)
        trainY = trainY.to(device, non_blocking=True)

        if (enable_fp16 or enable_bf16) and step < 2:
            if rank == 0:
                dtype_name = "float16" if enable_fp16 else "bfloat16"
                print(
                    f"[Precision Check] trainX dtype: {trainX.dtype}, trainY dtype: {trainY.dtype}"
                )

        if enable_fp16:
            autocast_dtype = torch.float16
        elif enable_bf16:
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float32

        current_batch_size = trainX.shape[0]

        t_model_start = time.perf_counter()
        with autocast(device_type='cuda', dtype=autocast_dtype):
            outputs = model(trainX)
            loss = loss_fn(outputs, trainY)

        scaler.scale(loss).backward()
        accumulated_samples += current_batch_size

        if (step + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(accumulated_samples)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if grad_norm > 1.0:
                gradient_clipping_counter += 1

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if use_ema and ema_model is not None:
                with torch.no_grad():
                    try:
                        ema_params = [
                            p.data for p in ema_model.parameters() if p.dtype.is_floating_point
                        ]
                        model_params = [
                            p.data.to(ep.dtype)
                            for p, ep in zip(model.parameters(), ema_model.parameters())
                            if ep.dtype.is_floating_point
                        ]
                        torch._foreach_mul_(ema_params, ema_decay)
                        torch._foreach_add_(ema_params, model_params, alpha=1.0 - ema_decay)
                    except AttributeError:
                        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                            if ema_param.dtype.is_floating_point:
                                ema_param.data.mul_(ema_decay).add_(
                                    param.data.to(ema_param.dtype), alpha=1.0 - ema_decay
                                )

            global_step += 1
            accumulated_samples = 0
        model_s += time.perf_counter() - t_model_start

        num_elements = outputs.numel()
        batch_loss_mean = loss.item() / num_elements
        running_loss += batch_loss_mean
        epoch_total_loss += batch_loss_mean

        if (step + 1) % accumulate_steps == 0:
            last_loss = running_loss / accumulate_steps
            if rank == 0:
                if tb_writer is not None:
                    tb_writer.add_scalar('Loss/train_step', last_loss, global_step)
                    tb_writer.add_scalar(
                        'LearningRate/train',
                        scheduler.get_last_lr()[0], global_step
                    )
            running_loss = 0.0

    avg_loss = epoch_total_loss / steps_per_epoch

    total_time = time.time() - epoch_start_time
    if rank == 0:
        time_per_batch = total_time / steps_per_epoch
        print(
            f"train_epoch: Completed {steps_per_epoch} batches in {total_time:.1f}s "
            f"({time_per_batch*1000:.1f}ms/batch), avg_loss={avg_loss:.5f}"
        )
        print(f"[Train Timing] data_gen={data_gen_s:.2f}s, model={model_s:.2f}s")

    return avg_loss, global_step, {
        "data_gen_s": data_gen_s,
        "model_s": model_s,
        "steps": steps_per_epoch,
        "total_s": total_time
    }


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using on-the-fly data generation."""
    OmegaConf.set_struct(cfg, False)

    # Suppress torch.compile verbose output
    import logging
    os.environ.setdefault('TORCH_LOGS', '-all')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
    logging.getLogger('torch._inductor.select_algorithm').setLevel(logging.ERROR)
    logging.getLogger('torch._inductor').setLevel(logging.ERROR)
    logging.getLogger('torch._dynamo').setLevel(logging.ERROR)

    # Set TF32 flags
    torch.backends.cuda.matmul.allow_tf32 = cfg.enable_matmul_tf32
    torch.backends.cudnn.allow_tf32 = cfg.enable_cudnn_tf32

    # Set global precision
    if cfg.enable_fp16:
        torch.set_default_dtype(torch.float16)
    elif getattr(cfg, 'enable_bf16', False):
        torch.set_default_dtype(torch.bfloat16)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.enable_cudnn_benchmark)
        torch.backends.cudnn.deterministic = False

    epochs_since_best = 0

    # Initialize distributed manager
    custom_timeout = int(os.environ.get('CUSTOM_DIST_TIMEOUT', 600))

    try:
        import torch.distributed as torch_dist
        original_init_process_group = torch_dist.init_process_group

        def init_process_group_with_timeout(*args, **kwargs):
            if 'timeout' not in kwargs:
                from datetime import timedelta
                kwargs['timeout'] = timedelta(seconds=custom_timeout)
            return original_init_process_group(*args, **kwargs)

        torch_dist.init_process_group = init_process_group_with_timeout
        DistributedManager.initialize()
        dist = DistributedManager()
        torch_dist.init_process_group = original_init_process_group

    except Exception as e:
        print(f"⚠️  Could not apply custom timeout: {e}")
        DistributedManager.initialize()
        dist = DistributedManager()

    # Training is not supported without a GPU; fail fast instead of silently using CPU.
    if dist.rank == 0 and not torch.cuda.is_available():
        print("ERROR: Training requires a GPU. CUDA is not available.")
        print("  - Check: python3 -c \"import torch; print(torch.cuda.is_available())\"")
        print(
            "  - Ensure PyTorch is installed with CUDA (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu121)"
        )
        print(
            "  - Free GPU: run code/scripts/free_gpu.sh to list or kill other processes using the GPU."
        )
        raise RuntimeError("Training requires a GPU; CUDA is not available.")

    # Job timing broadcast
    job_start_timestamp = None
    job_start_datetime = None
    job_time_limit_seconds = None

    if dist.rank == 0:
        job_start_timestamp = os.getenv('JOB_START_TIMESTAMP')
        job_start_datetime = os.getenv('JOB_START_DATETIME')
        job_time_limit = os.getenv('JOB_TIME_LIMIT')

        if not job_start_timestamp:
            try:
                with open('job_start_timestamp.txt', 'r') as f:
                    job_start_timestamp = f.read().strip()
                with open('job_start_datetime.txt', 'r') as f:
                    job_start_datetime = f.read().strip()
                with open('job_time_limit.txt', 'r') as f:
                    job_time_limit = f.read().strip()
            except FileNotFoundError:
                pass

        if job_start_timestamp:
            job_start_timestamp = float(job_start_timestamp)
            if job_time_limit:
                try:
                    time_parts = job_time_limit.split(':')
                    if len(time_parts) == 3:
                        hours, minutes, seconds = map(int, time_parts)
                        job_time_limit_seconds = hours * 3600 + minutes * 60 + seconds
                except ValueError:
                    pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        timing_data = torch.zeros(2, dtype=torch.float64, device=dist.device)
        if dist.rank == 0 and job_start_timestamp is not None:
            timing_data[0] = job_start_timestamp
            timing_data[1] = job_time_limit_seconds if job_time_limit_seconds is not None else 0.0
        torch.distributed.broadcast(timing_data, src=0)
        if dist.rank != 0:
            job_start_timestamp = float(timing_data[0].item()
                                       ) if timing_data[0].item() != 0.0 else None
            job_time_limit_seconds = int(timing_data[1].item()
                                        ) if timing_data[1].item() != 0.0 else None

    if job_start_timestamp is not None:
        cfg.job_start_timestamp = job_start_timestamp
        cfg.job_start_datetime = job_start_datetime
        cfg.job_time_limit_seconds = job_time_limit_seconds

    # === Public-release training defaults (runtime) ===
    # Auto-scale shots / epoch based on detected world size to keep epochs reasonably fast
    # across different user machines, while preserving production-like behavior.
    #
    # Production reference: train.num_samples is interpreted as "shots / epoch at world_size=8".
    # This allows temporarily reducing train.num_samples in config_validator.py for faster iteration.
    try:
        ws = int(getattr(dist, "world_size", 1) or 1)
    except Exception:
        ws = 1
    ws_eff = max(1, min(ws, 8))
    base_samples_8gpu = int(getattr(getattr(cfg, "train", None), "num_samples", 2**26))
    scaled_samples = int(base_samples_8gpu * (ws_eff / 8.0))
    # Make sure it's positive and at least one effective batch worth of samples.
    scaled_samples = max(int(scaled_samples), 1)
    cfg.train.num_samples = scaled_samples
    # Epoch count defaults to production value; allow explicit overrides for quick runs.
    if getattr(cfg.train, "epochs", None) is None:
        cfg.train.epochs = 100

    # Optional timing-mode overrides (env-based) for short measurement runs.
    if os.environ.get("PREDECODER_TIMING_RUN", "0") == "1":
        train_samples_env = os.environ.get("PREDECODER_TRAIN_SAMPLES")
        val_samples_env = os.environ.get("PREDECODER_VAL_SAMPLES")
        test_samples_env = os.environ.get("PREDECODER_TEST_SAMPLES")
        epochs_env = os.environ.get("PREDECODER_TRAIN_EPOCHS")
        try:
            if train_samples_env:
                cfg.train.num_samples = int(train_samples_env)
        except Exception:
            pass
        try:
            if val_samples_env:
                cfg.val.num_samples = int(val_samples_env)
        except Exception:
            pass
        try:
            if test_samples_env:
                cfg.test.num_samples = int(test_samples_env)
        except Exception:
            pass
        try:
            if epochs_env:
                cfg.train.epochs = int(epochs_env)
        except Exception:
            pass
        milestones_env = os.environ.get("PREDECODER_LR_MILESTONES")
        try:
            if milestones_env:
                cfg.lr_scheduler.milestones = [float(x) for x in milestones_env.split(",")]
        except Exception:
            pass

    if dist.rank == 0:
        print(f"Effective workflow.task: {cfg.workflow.task}")
        print(f"Using LR scheduler type: {cfg.lr_scheduler.type}")
        print(
            f"[Train] Auto-scaled train.num_samples={int(cfg.train.num_samples):,} "
            f"(base={base_samples_8gpu:,} @ world_size=8; detected world_size={ws}, using {ws_eff})"
        )
        print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    print(f"Rank {dist.rank} running on {dist.device}")
    if torch.cuda.is_available() and dist.rank == 0:
        mem_alloc = torch.cuda.memory_allocated(dist.device) / 2**20
        mem_reserved = torch.cuda.memory_reserved(dist.device) / 2**20
        print(
            f"[GPU] {dist.device} in use (allocated: {mem_alloc:.1f} MiB, reserved: {mem_reserved:.1f} MiB)"
        )

    # Configure QEC metrics (LER, syndrome density)
    configure_metrics(rank=dist.rank)

    # === Torch Generator Setup ===
    if dist.rank == 0:
        print("=" * 80)
        print("🚀 Using Torch Generator for on-the-fly data generation")
        print("=" * 80)

    from data.generator_torch import QCDataGeneratorTorch

    # Generate random base seed on rank 0, broadcast to all
    import random
    if dist.rank == 0:
        base_seed = random.randint(0, 2**31 - 1)
    else:
        base_seed = 0

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seed_tensor = torch.tensor([base_seed], dtype=torch.int64, device=dist.device)
        torch.distributed.broadcast(seed_tensor, src=0)
        base_seed = int(seed_tensor.item())

    if dist.rank == 0:
        print(f"🎲 Random base seed for this session: {base_seed}")

    # Get p settings
    p_error_value = getattr(cfg.data, 'p_error', None)
    p_min_value = getattr(cfg.data, 'p_min', 0.001)
    p_max_value = getattr(cfg.data, 'p_max', 0.008)

    # Optional explicit circuit-level noise model (overrides p_error/p_min/p_max when provided)
    noise_model_cfg = getattr(cfg.data, "noise_model", None)
    noise_model_user_obj = None
    noise_model_train_obj = None
    if noise_model_cfg is not None:
        from qec.noise_model import NoiseModel
        nm_dict = OmegaConf.to_container(noise_model_cfg, resolve=True
                                        ) if hasattr(noise_model_cfg, "items") else noise_model_cfg
        # Allow configs to specify `noise_model: null`
        if nm_dict is not None:
            noise_model_user_obj = NoiseModel.from_config_dict(dict(nm_dict))

            # Compute grouped noise totals for logging and scaling decisions.
            p_prep = float(noise_model_user_obj.p_prep_X + noise_model_user_obj.p_prep_Z)
            p_meas = float(noise_model_user_obj.p_meas_X + noise_model_user_obj.p_meas_Z)
            p_idle_cnot = float(noise_model_user_obj.get_total_idle_cnot_probability())
            p_idle_spam = float(noise_model_user_obj.get_total_idle_spam_probability())
            p_cnot = float(noise_model_user_obj.get_total_cnot_probability())
            max_group = max(p_prep, p_meas, p_idle_cnot, p_idle_spam, p_cnot)
            if max_group <= 0.0:
                raise ValueError(
                    "Invalid noise_model: all grouped totals are <= 0 "
                    f"(prep={p_prep}, meas={p_meas}, idle_cnot={p_idle_cnot}, idle_spam={p_idle_spam}, cnot={p_cnot})."
                )

            # Surface-code training upscaling: bring max(P's) to target for training
            # data only. Evaluation uses the user-specified noise model as-is.
            from qec.noise_model import (
                get_training_upscaled_noise_model,
                SURFACE_CODE_TRAINING_UPSCALE_TARGET,
                SURFACE_CODE_THRESHOLD_APPROX,
            )
            code_type = getattr(cfg.data, "code_type", "surface_code")
            skip_upscale = bool(getattr(cfg.data, "skip_noise_upscaling", False))
            if os.environ.get("PREDECODER_SKIP_NOISE_UPSCALING", "0") == "1":
                skip_upscale = True
            noise_model_train_obj, upscale_info = get_training_upscaled_noise_model(
                noise_model_user_obj,
                code_type=code_type,
                skip_upscale=skip_upscale,
            )

            # Force fixed-p mode with a conservative scalar placeholder when using noise_model.
            # IMPORTANT: during training we apply drift (±2%) around the *training* noise model reference, so we
            # size buffers using 1.25× the maximum reference probability to avoid overflow.
            p_error_value = float(1.25 * noise_model_train_obj.get_max_probability())
            p_min_value = p_error_value
            p_max_value = p_error_value
            if dist.rank == 0:
                # Always print the grouped totals + decision to make verification easy from logs.
                print(
                    "[Train] noise_model grouped totals: "
                    f"prep={p_prep:.6g}, meas={p_meas:.6g}, "
                    f"idle_cnot={p_idle_cnot:.6g}, idle_spam={p_idle_spam:.6g}, cnot={p_cnot:.6g}; "
                    f"max_group={max_group:.6g}"
                )
                print(f"[Train] {upscale_info['message']}")
                if upscale_info.get("skipped_by_user"):
                    print(
                        "[Train] noise_model upscaling SKIPPED (skip_noise_upscaling=true or "
                        "PREDECODER_SKIP_NOISE_UPSCALING=1). Training uses exact user-specified parameters."
                    )
                if upscale_info.get("applied_upscale"):
                    print(
                        f"[Train] noise_model upscaling (surface code): scale_factor={upscale_info['scale_factor']:.6g}, "
                        f"target={SURFACE_CODE_TRAINING_UPSCALE_TARGET:.1e}. "
                        "Evaluation uses user-specified noise model as-is."
                    )
                    print(
                        f"[Train] noise_model (training, upscaled) summary: {noise_model_train_obj!r}"
                    )
                if upscale_info.get("downscale_skipped"):
                    print(
                        "\n" + "!" * 80 + "\n" +
                        "[Train] WARNING: Noise model DOWNSCALE was NOT applied (max_group > target). "
                        "Parameters are unchanged. If you intended a lower noise regime, check your noise model "
                        "parameter values (e.g. a typo may have set a single p too high).\n" +
                        "!" * 80
                    )
                if upscale_info.get("above_target_warning"):
                    print(
                        "\n" + "#" * 80 + "\n" +
                        "[Train] WARNING: Your noise model max_group ({:.6g}) is ABOVE the surface-code training "
                        "target ({:.1e}). Surface code threshold is approximately {:.1e}. "
                        "You may be above threshold or have introduced a noise model that is not the one you intended "
                        "(e.g. a typo in one of the 25 p's). Please verify your noise_model configuration.\n"
                        .format(
                            max_group, SURFACE_CODE_TRAINING_UPSCALE_TARGET,
                            SURFACE_CODE_THRESHOLD_APPROX
                        ) + "#" * 80
                    )
                print(
                    f"[Train] Using explicit noise_model from config (25p). Overriding p_error/p_min/p_max -> {p_error_value:.6g}"
                )
                print(f"[Train] noise_model (user) summary: {noise_model_user_obj!r}")
                if upscale_info.get("applied_upscale"):
                    print(
                        f"[Train] noise_model (training, upscaled) summary: {noise_model_train_obj!r}"
                    )
                print(
                    "[Train] noise_model idle semantics: "
                    "bulk/CNOT-layer idles use p_idle_cnot_*, "
                    "data-idle during ancilla prep/reset uses p_idle_spam_*, "
                    "data-idle during ancilla measurement is ignored."
                )
                print(
                    "[Train] noise_model totals: "
                    f"prep_total={p_prep:.6g}, meas_total={p_meas:.6g}, "
                    f"idle_cnot_total={noise_model_user_obj.get_total_idle_cnot_probability():.6g}, "
                    f"idle_spam_total={noise_model_user_obj.get_total_idle_spam_probability():.6g}, "
                    f"cnot_total={noise_model_user_obj.get_total_cnot_probability():.6g}"
                )
        elif dist.rank == 0:
            print("[Train] noise_model: null (using legacy single-p / p-range sampling)")
    elif dist.rank == 0:
        print("[Train] noise_model: (missing in config) (using legacy single-p / p-range sampling)")

    # Check for multi-patch mode
    use_multiple_patches = getattr(cfg.data, 'use_multiple_patches', False)
    multi_d = getattr(cfg, 'multiple_distances', None)
    multi_r = getattr(cfg, 'multiple_rounds', None)

    def is_list_like(obj):
        return obj is not None and hasattr(obj, '__len__') and hasattr(obj, '__getitem__')

    use_multi_pairs = (
        use_multiple_patches and is_list_like(multi_d) and is_list_like(multi_r) and
        len(multi_d) == len(multi_r) and len(multi_d) > 0
    )

    # Get HE settings
    timelike_he = getattr(cfg.data, 'timelike_he', False)
    num_he_cycles = getattr(cfg.data, 'num_he_cycles', 1)
    max_passes = getattr(cfg.data, 'max_passes_w1', 32)
    decompose_y = getattr(cfg.data, 'decompose_y', True)
    precomputed_frames_dir = getattr(cfg.data, 'precomputed_frames_dir', None)
    code_rotation = getattr(cfg.data, 'code_rotation', 'XV')
    precomputed_frames_dir = resolve_precomputed_frames_dir(
        precomputed_frames_dir, cfg.distance, cfg.n_rounds, cfg.meas_basis, dist.rank
    )

    _compute_dtype_raw = getattr(cfg.data, 'compute_dtype', None)
    _compute_dtype = None
    if _compute_dtype_raw is not None:
        _dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        _compute_dtype = _dtype_map.get(str(_compute_dtype_raw), None)

    _he_accel_kwargs = dict(
        use_compile=bool(getattr(cfg.data, 'use_compile', False)),
        compile_chunk_size=int(getattr(cfg.data, 'compile_chunk_size', 2)),
        compute_dtype=_compute_dtype,
        use_weight2=bool(getattr(cfg.data, 'use_weight2', False)),
        max_passes_w2=int(getattr(cfg.data, 'max_passes_w2', 4)),
        use_coset_search=bool(getattr(cfg.data, 'use_coset_search', False)),
        coset_max_generators=int(getattr(cfg.data, 'coset_max_generators', 20)),
        use_dense_overlap=bool(getattr(cfg.data, 'use_dense_overlap', False)),
    )

    if use_multi_pairs:
        # Torch-only: no multi-pair support yet; fall back to single pair
        if dist.rank == 0:
            print(
                "[Train] Note: multi_pairs not yet supported in Torch generator; using single pair mode"
            )
        train_generator = None
        val_generator = None
    else:
        train_generator = QCDataGeneratorTorch(
            distance=cfg.distance,
            n_rounds=cfg.n_rounds,
            p_error=p_error_value,
            p_min=p_min_value,
            p_max=p_max_value,
            measure_basis=cfg.meas_basis,
            rank=dist.local_rank,
            global_rank=dist.rank,
            mode='train',
            verbose=(dist.rank == 0),
            base_seed=base_seed,
            timelike_he=timelike_he,
            num_he_cycles=num_he_cycles,
            max_passes_w1=max_passes,
            decompose_y=False,
            precomputed_frames_dir=precomputed_frames_dir,
            code_rotation=code_rotation,
            **_he_accel_kwargs,
        )
        val_generator = QCDataGeneratorTorch(
            distance=cfg.distance,
            n_rounds=cfg.n_rounds,
            p_error=p_error_value,
            p_min=p_min_value,
            p_max=p_max_value,
            measure_basis=cfg.meas_basis,
            rank=dist.local_rank,
            global_rank=dist.rank,
            mode='test',
            verbose=False,
            base_seed=base_seed + 100_000_000,
            timelike_he=timelike_he,
            num_he_cycles=num_he_cycles,
            max_passes_w1=max_passes,
            decompose_y=False,
            precomputed_frames_dir=precomputed_frames_dir,
            code_rotation=code_rotation,
            **_he_accel_kwargs,
        )

    # Create test generator
    test_distance_override = getattr(cfg.test, 'distance', None)
    test_rounds_override = getattr(cfg.test, 'n_rounds', None)
    test_timelike_he = getattr(cfg.test, 'timelike_he', timelike_he)
    test_num_he_cycles = getattr(cfg.test, 'num_he_cycles', num_he_cycles)
    test_max_passes = getattr(cfg.test, 'max_passes_w1', max_passes)

    if test_distance_override is not None and test_rounds_override is not None:
        test_d, test_r = int(test_distance_override), int(test_rounds_override)
    elif use_multi_pairs:
        largest_idx = max(range(len(multi_d)), key=lambda i: int(multi_d[i]))
        test_d, test_r = int(multi_d[largest_idx]), int(multi_r[largest_idx])
    else:
        test_d, test_r = cfg.distance, cfg.n_rounds

    # Determine if we can reuse val_generator for testing
    test_matches_training = False
    if not use_multi_pairs and test_d == cfg.distance and test_r == cfg.n_rounds:
        test_matches_training = True
    elif use_multi_pairs:
        for d, r in zip(multi_d, multi_r):
            if int(d) == test_d and int(r) == test_r:
                test_matches_training = True
                break

    # Test generator (Stim-based evaluation path)
    test_generator = None

    if dist.rank == 0:
        print("✅ Torch generator initialized successfully (Stim simulation + PyMatching decoding)")

        def _print_gen(name, g):
            if g is None:
                print(f"[Train] {name}: <none>")
                return
            # Basic shape config
            d = getattr(g, "distance", None)
            r = getattr(g, "n_rounds", None)
            mode = getattr(g, "mode", None)
            # Noise model usage
            nm = getattr(g, "noise_model", None)
            has_nm = nm is not None
            drift_en = bool(getattr(g, "_noise_model_drift_enabled", False))
            drift_frac = getattr(g, "_noise_model_drift_frac", None)
            drift_s = f", drift=±{float(drift_frac):.3f}" if drift_en and drift_frac is not None else ""
            print(
                f"[Train] {name}: d={d}, n_rounds={r}, mode={mode}, noise_model={'yes' if has_nm else 'no'}{drift_s}"
            )

        _print_gen("train_generator", train_generator)
        _print_gen("val_generator", val_generator)
        _print_gen(f"test_generator (target d={test_d}, r={test_r})", test_generator)
        print("=" * 80)

    # Generate sample batch for shape info
    sample_trainX, sample_trainY = train_generator.generate_batch(step=0, batch_size=1)
    cfg.model.input_channels = sample_trainX.shape[1]
    cfg.model.out_channels = sample_trainY.shape[1]

    # Create model
    base_model = ModelFactory.create_model(cfg).to(dist.device)

    if cfg.enable_fp16:
        base_model = base_model.half()
    elif getattr(cfg, 'enable_bf16', False):
        base_model = base_model.to(torch.bfloat16)

    ema_model = deepcopy(base_model).to(dist.device).eval()

    # Load checkpoint before compiling
    init_epoch_temp = 0
    global_step_temp = 0
    if cfg.load_checkpoint:
        early_stoping_path_temp = os.path.join(cfg.output, "early_stopping.json")
        if os.path.exists(to_absolute_path(early_stoping_path_temp)):
            if dist.rank == 0:
                print(f"Early stopping file found. Finish training.")
            return
        init_epoch_temp, global_step_temp = load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=base_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=dist.device,
            steps_per_epoch_estimate=None,
            rank=dist.rank
        )

    # Optional torch.compile (with graceful fallback on failure)
    env_compile = os.environ.get("PREDECODER_TORCH_COMPILE")
    env_compile_mode = os.environ.get("PREDECODER_TORCH_COMPILE_MODE")
    compile_mode = str(env_compile_mode or getattr(cfg, "torch_compile_mode", "max-autotune"))
    should_compile = bool(getattr(cfg, "torch_compile", True))
    if env_compile is not None:
        env_val = str(env_compile).strip().lower()
        if env_val == "auto":
            should_compile = True
        elif env_val in ("0", "false", "no", "off"):
            should_compile = False
        elif env_val in ("1", "true", "yes", "on"):
            should_compile = True
    if should_compile:
        if dist.rank == 0:
            print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        try:
            base_model = torch.compile(base_model, mode=compile_mode)
        except Exception as exc:
            if dist.rank == 0:
                print(f"[Train] torch.compile failed ({exc}); continuing without compile.")

    # Wrap for DDP
    model = base_model
    if dist.world_size > 1:
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    ema_decay = cfg.ema.decay if cfg.ema.use_ema else 0.0

    # Print model summary
    if dist.rank == 0:
        summary_input = sample_trainX.to(dist.device)
        if cfg.enable_fp16:
            summary_input = summary_input.half()
        elif getattr(cfg, 'enable_bf16', False):
            summary_input = summary_input.to(torch.bfloat16)

        if torchinfo is None:
            if dist.rank == 0:
                print("Model summary skipped (torchinfo not installed).")
        else:
            summary = torchinfo.summary(
                ema_model if cfg.ema.use_ema else base_model,
                input_data=summary_input,
                verbose=0,
                depth=2,
            )
            print(f"Model summary:\n{summary}\n")

    # Create optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(0.9, cfg.optimizer.beta2),
            eps=1e-5
        )
    elif cfg.optimizer_type == "Lion":
        optimizer = DebugLion(
            trainable_params,
            lr=cfg.optimizer.lr,
            betas=(0.9, cfg.optimizer.beta2),
            weight_decay=cfg.optimizer.weight_decay,
            log_nan=True
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimizer_type}")

    # Calculate total steps for scheduler
    effective_num_samples = cfg.train.num_samples
    if dist.rank == 0:
        print(f"Calculating total_steps with {effective_num_samples:,} samples per epoch")

    total_steps = 0
    for epoch in range(cfg.train.epochs):
        per_device_bs = get_current_per_device_batch_size(epoch, cfg)
        acc = cfg.train.accumulate_steps
        batches = effective_num_samples // (per_device_bs * dist.world_size)
        steps = max(1, math.ceil(batches / acc))
        total_steps += steps

    # Quick-validation guard: keep short runs from tripping scheduler constraints.
    # Full training runs should keep the default warmup.
    if cfg.lr_scheduler.warmup_steps >= total_steps:
        if dist.rank == 0:
            print(
                "[Train] Warning: warmup_steps "
                f"({cfg.lr_scheduler.warmup_steps}) >= total_steps ({total_steps}); "
                "reducing warmup_steps to keep the schedule valid for short runs."
            )
        cfg.lr_scheduler.warmup_steps = max(0, total_steps - 1)
    assert cfg.lr_scheduler.warmup_steps < total_steps, \
        f"Warm-up steps ({cfg.lr_scheduler.warmup_steps}) must be less than total training steps ({total_steps})"

    scheduler = get_lr_scheduler(cfg, optimizer, total_steps)

    if dist.rank == 0:
        print(f"Learning Rate Scheduler: {cfg.lr_scheduler.type}, Total steps: {total_steps:,}")

    # Initialize scaler
    use_grad_scaler = cfg.enable_fp16 and torch.cuda.is_available() and torch.get_default_dtype(
    ) != torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    # TensorBoard writer
    writer = None
    if dist.rank == 0:
        if SummaryWriter is None:
            print("TensorBoard logging disabled (tensorboard not installed).")
        else:
            writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    # Setup paths
    model_save_path = os.path.join(cfg.output, "models")
    best_model_path = os.path.join(model_save_path, "best_model")
    early_stoping_path = os.path.join(cfg.output, "early_stopping.json")
    if dist.rank == 0:
        create_directory(model_save_path)
        create_directory(best_model_path)

    if dist.world_size > 1:
        torch.distributed.barrier()

    # Calculate steps per epoch
    per_device_bs0 = get_current_per_device_batch_size(0, cfg)
    acc0 = cfg.train.accumulate_steps
    batches_per_epoch = effective_num_samples // (per_device_bs0 * dist.world_size)
    steps_per_epoch_estimate = math.ceil(batches_per_epoch / acc0)

    # Load optimizer/scheduler/scaler state
    if cfg.load_checkpoint:
        _, _ = load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=None,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dist.device,
            steps_per_epoch_estimate=steps_per_epoch_estimate,
            rank=dist.rank
        )
        init_epoch = init_epoch_temp
        global_step = global_step_temp
    else:
        init_epoch = 0
        global_step = 0

    # Load best_vloss (but reset if metric type changed from loss to LER or vice versa)
    best_vloss = 1_000_000.0
    use_ler_for_early_stopping = getattr(cfg, 'validation_ler', False)
    try:
        checkpoint_files = [
            f for f in os.listdir(best_model_path) if f.endswith('.pt') and 'checkpoint' in f
        ]
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(best_model_path, f))
            )
            checkpoint_dict = torch.load(
                os.path.join(best_model_path, latest_checkpoint),
                map_location="cpu",
                weights_only=False
            )
            if 'metadata' in checkpoint_dict and 'best_vloss' in checkpoint_dict['metadata']:
                saved_using_ler = checkpoint_dict['metadata'].get('using_ler', False)
                # Only restore best_vloss if the metric type matches (both LER or both loss)
                if saved_using_ler == use_ler_for_early_stopping:
                    best_vloss = checkpoint_dict['metadata']['best_vloss']
                    if 'epochs_since_best' in checkpoint_dict['metadata']:
                        epochs_since_best = checkpoint_dict['metadata']['epochs_since_best']
                    if dist.rank == 0:
                        metric_name = "LER" if use_ler_for_early_stopping else "validation loss"
                        print(f"[Checkpoint] Restored best {metric_name}: {best_vloss:.6f}")
                else:
                    if dist.rank == 0:
                        old_metric = "LER" if saved_using_ler else "validation loss"
                        new_metric = "LER" if use_ler_for_early_stopping else "validation loss"
                        print(
                            f"[Checkpoint] Metric type changed ({old_metric} → {new_metric}), resetting best metric"
                        )
    except Exception:
        pass

    epoch_times = []
    cumulative_steps = 0

    # === TRAINING LOOP ===
    for epoch in range(init_epoch, cfg.train.epochs):
        epoch_start_time = time.time()
        epoch_number = epoch

        if should_stop_due_to_time(cfg, epoch_times, epoch, dist.rank):
            break

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dist.device)

        if dist.rank == 0:
            print(f"Device {dist.device}, epoch {epoch_number}:")

        # Get batch sizes
        num_pairs = len(multi_d) if use_multi_pairs else 1
        curriculum_batch_sizes = get_curriculum_batch_sizes(
            cfg, epoch, num_pairs
        ) if use_multi_pairs else None

        if curriculum_batch_sizes is not None:
            steps_per_epoch, samples_per_cycle, num_cycles = calculate_curriculum_steps_per_epoch(
                curriculum_batch_sizes, effective_num_samples, dist.world_size
            )
            accumulate_steps = cfg.train.accumulate_steps
            per_device_batch_size = curriculum_batch_sizes
        else:
            per_device_batch_size = get_current_per_device_batch_size(epoch, cfg)
            accumulate_steps = cfg.train.accumulate_steps
            effective_batch_size = per_device_batch_size * accumulate_steps * dist.world_size
            steps_per_epoch = effective_num_samples // effective_batch_size

            if dist.rank == 0:
                print(
                    f"[Epoch {epoch_number}] Effective batch size = per_device × accumulate_steps × world_size"
                )
                print(
                    f"[Epoch {epoch_number}] Effective batch size: {effective_batch_size} "
                    f"({per_device_batch_size} × {accumulate_steps} × {dist.world_size})"
                )
                # Log batch size to TensorBoard
                if writer is not None:
                    writer.add_scalar("BatchSize", effective_batch_size, epoch_number)

        epoch_wall_start = time.perf_counter()
        train_total_s = 0.0
        val_total_s = 0.0
        loss_sync_s = 0.0
        gc_s = 0.0
        sdr_s = 0.0
        ler_s = 0.0
        log_s = 0.0
        ckpt_best_s = 0.0
        ckpt_periodic_s = 0.0
        barrier_s = 0.0

        model.train(True)

        if epoch == init_epoch:
            cumulative_steps = 0

        t_train_start = time.perf_counter()
        avg_loss, global_step, train_timing = train_epoch(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            cumulative_steps_before_epoch=cumulative_steps,
            batch_size=per_device_batch_size,
            epoch_number=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            tb_writer=writer,
            device=dist.device,
            enable_fp16=cfg.enable_fp16,
            enable_bf16=getattr(cfg, 'enable_bf16', False),
            rank=dist.rank,
            use_ema=cfg.ema.use_ema,
            ema_model=ema_model,
            ema_decay=ema_decay,
            global_step=global_step,
            accumulate_steps=accumulate_steps,
        )
        train_total_s = float(train_timing.get("total_s", 0.0)
                             ) if train_timing else (time.perf_counter() - t_train_start)

        cumulative_steps += steps_per_epoch

        model.eval()
        model_to_eval = ema_model if cfg.ema.use_ema else model
        model_for_ckpt = model_to_eval.module if isinstance(model_to_eval, DDP) else model_to_eval

        val_samples_per_gpu = cfg.val.num_samples // dist.world_size

        t_val_start = time.perf_counter()
        avg_vloss, val_timing = validation_step(
            generator=val_generator,
            model=model_to_eval,
            num_samples=val_samples_per_gpu,
            batch_size=per_device_batch_size,
            device=dist.device,
            enable_fp16=cfg.enable_fp16,
            enable_bf16=getattr(cfg, 'enable_bf16', False),
            rank=dist.rank
        )
        val_total_s = float(val_timing.get("total_s", 0.0)
                           ) if val_timing else (time.perf_counter() - t_val_start)

        # Synchronize losses
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t_sync_start = time.perf_counter()
            t = torch.tensor([avg_loss], device=dist.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            avg_loss = float(t.item())

            v = torch.tensor([avg_vloss], device=dist.device)
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.AVG)
            avg_vloss = float(v.item())
            loss_sync_s = time.perf_counter() - t_sync_start

        # Removed: gc.collect() + empty_cache() cause 100-400ms stall per epoch
        # and empty_cache() causes memory fragmentation on next allocation.
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        gc_s = 0.0

        # Compute LER (+ PyMatching speedup) and SDR (syndrome density reduction) if enabled
        use_ler_for_early_stopping = getattr(cfg, 'validation_ler', False)
        disable_sdr = os.environ.get("PREDECODER_DISABLE_SDR", "0") == "1"
        ler_final_only = os.environ.get("PREDECODER_LER_FINAL_ONLY", "0") == "1"
        run_ler_this_epoch = use_ler_for_early_stopping and (
            (not ler_final_only) or (epoch_number == (cfg.train.epochs - 1))
        )
        validation_ler = None
        ler_reduction_factor = None
        pymatching_speedup_avg = None
        syndrome_density_reduction = None

        if run_ler_this_epoch:
            try:
                orig_cfg_distance, orig_cfg_n_rounds = cfg.distance, cfg.n_rounds
                cfg.distance, cfg.n_rounds = test_d, test_r

                # Syndrome density reduction (SDR): computed for TensorBoard visibility.
                if disable_sdr:
                    syndrome_density_reduction = None
                    sdr_s = 0.0
                    if dist.rank == 0:
                        print("[Syndrome Density] Skipped (PREDECODER_DISABLE_SDR=1)")
                else:
                    t_sdr_start = time.perf_counter()
                    sdr_as_percent = bool(getattr(cfg, 'sdr_as_percent', False))
                    syndrome_density_reduction = compute_syndrome_density(
                        model=model_to_eval,
                        device=dist.device,
                        dist=dist,
                        cfg=cfg,
                        generator=val_generator,
                        rank=dist.rank,
                        sdr_as_percent=sdr_as_percent,
                    )
                    sdr_s = time.perf_counter() - t_sdr_start
                # If multi-pair dict, reduce to a single scalar for logging (average over pairs).
                if isinstance(syndrome_density_reduction,
                              dict) and len(syndrome_density_reduction) > 0:
                    syndrome_density_reduction = sum(syndrome_density_reduction.values()
                                                    ) / len(syndrome_density_reduction)

                # Average SDR across ranks for clean TensorBoard curves.
                if syndrome_density_reduction is not None and torch.distributed.is_available(
                ) and torch.distributed.is_initialized():
                    sd_tensor = torch.tensor(
                        [float(syndrome_density_reduction)], device=dist.device
                    )
                    torch.distributed.all_reduce(sd_tensor, op=torch.distributed.ReduceOp.AVG)
                    syndrome_density_reduction = float(sd_tensor.item())

                t_ler_start = time.perf_counter()
                ler_result = compute_validation_ler(
                    model=model_to_eval,
                    device=dist.device,
                    dist=dist,
                    cfg=cfg,
                    generator=None,
                    rank=dist.rank,
                )
                ler_s = time.perf_counter() - t_ler_start

                if isinstance(ler_result, tuple):
                    # (validation_ler, ler_reduction_factor, pymatching_speedup_avg)
                    if len(ler_result) >= 1:
                        validation_ler = ler_result[0]
                    if len(ler_result) >= 2:
                        ler_reduction_factor = ler_result[1]
                    if len(ler_result) >= 3:
                        pymatching_speedup_avg = ler_result[2]
                elif isinstance(ler_result, dict):
                    ler_values = [
                        v[0]
                        for v in ler_result.values()
                        if isinstance(v, tuple) and len(v) >= 1 and v[0] is not None
                    ]
                    validation_ler = sum(ler_values) / len(ler_values) if ler_values else None
                    speedups = [
                        v[2]
                        for v in ler_result.values()
                        if isinstance(v, tuple) and len(v) >= 3 and v[2] is not None
                    ]
                    pymatching_speedup_avg = sum(speedups) / len(speedups) if speedups else None
                else:
                    validation_ler = ler_result

                if validation_ler is not None and torch.distributed.is_available(
                ) and torch.distributed.is_initialized():
                    ler_tensor = torch.tensor([validation_ler], device=dist.device)
                    torch.distributed.all_reduce(ler_tensor, op=torch.distributed.ReduceOp.AVG)
                    validation_ler = float(ler_tensor.item())

            finally:
                cfg.distance, cfg.n_rounds = orig_cfg_distance, orig_cfg_n_rounds
        elif use_ler_for_early_stopping and dist.rank == 0:
            print("[LER Validation] Skipped (PREDECODER_LER_FINAL_ONLY=1)")

        # Log metrics
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_log_start = time.perf_counter()
        if dist.rank == 0:
            # --- Compute speed/throughput metrics ---
            _epoch_wall = time.perf_counter() - epoch_wall_start
            _samples_this_epoch = steps_per_epoch * per_device_batch_size * dist.world_size
            _throughput = _samples_this_epoch / _epoch_wall if _epoch_wall > 0 else 0.0
            _train_steps = train_timing.get(
                "steps", steps_per_epoch
            ) if train_timing else steps_per_epoch
            _ms_per_batch = (train_total_s / _train_steps * 1000) if _train_steps > 0 else 0.0
            _data_pct = (
                train_timing.get("data_gen_s", 0) / train_total_s * 100
            ) if train_total_s > 0 and train_timing else 0.0

            # --- Main summary line ---
            parts = [f"[{timestamp}] Epoch {epoch_number}"]
            parts.append(f"loss={avg_loss:.5f}/{avg_vloss:.5f}")
            if use_ler_for_early_stopping and validation_ler is not None:
                parts.append(f"LER={validation_ler:.6f}")
            if syndrome_density_reduction is not None:
                _sdr_unit = "%" if bool(getattr(cfg, 'sdr_as_percent', False)) else "x"
                parts.append(f"SDR={syndrome_density_reduction:.2f}{_sdr_unit}")
            parts.append(
                f"{_epoch_wall:.0f}s ({_throughput:.0f} samp/s, {_ms_per_batch:.1f}ms/batch)"
            )
            print(" | ".join(parts))

            # --- Speed breakdown line ---
            _speed_parts = [f"[Speed] train={train_total_s:.1f}s val={val_total_s:.1f}s"]
            if sdr_s > 0:
                _speed_parts.append(f"sdr={sdr_s:.1f}s")
            if ler_s > 0:
                _speed_parts.append(f"ler={ler_s:.1f}s")
            if _data_pct > 1.0:
                _speed_parts.append(f"data_gen={_data_pct:.0f}% of train")
            print(" ".join(_speed_parts))
            dem_timing = None
            try:
                from evaluation import logical_error_rate as ler_stim
                dem_timing = getattr(ler_stim, "LAST_DEM_TIMING", None)
            except Exception:
                dem_timing = None
            if train_timing or val_timing or dem_timing:
                train_data_s = float(train_timing.get("data_gen_s", 0.0)) if train_timing else 0.0
                train_model_s = float(train_timing.get("model_s", 0.0)) if train_timing else 0.0
                val_data_s = float(val_timing.get("data_gen_s", 0.0)) if val_timing else 0.0
                val_model_s = float(val_timing.get("model_s", 0.0)) if val_timing else 0.0
                dem_build_s = float(dem_timing.get("dem_build_s", 0.0)
                                   ) if isinstance(dem_timing, dict) else 0.0
                dem_decode_s = float(dem_timing.get("dem_decode_s", 0.0)
                                    ) if isinstance(dem_timing, dict) else 0.0
                total_bucket_s = train_data_s + train_model_s + val_data_s + val_model_s + dem_build_s + dem_decode_s
                if total_bucket_s > 0:
                    pct = lambda s: 100.0 * s / total_bucket_s
                else:
                    pct = lambda s: 0.0
                print(
                    "[Epoch Timing] train_data_gen={:.2f}s train_model={:.2f}s "
                    "val_data_gen={:.2f}s val_model={:.2f}s dem_build={:.2f}s dem_decode={:.2f}s".
                    format(
                        train_data_s, train_model_s, val_data_s, val_model_s, dem_build_s,
                        dem_decode_s
                    )
                )
                print(
                    "[Epoch Timing %] train_data_gen={:.1f}% train_model={:.1f}% "
                    "val_data_gen={:.1f}% val_model={:.1f}% dem_build={:.2f}% dem_decode={:.2f}%".
                    format(
                        pct(train_data_s), pct(train_model_s), pct(val_data_s), pct(val_model_s),
                        pct(dem_build_s), pct(dem_decode_s)
                    )
                )
                if ler_s > 0:
                    ler_other_s = max(ler_s - (dem_build_s + dem_decode_s), 0.0)
                    print(
                        "[Epoch LER Timing] total={:.2f}s dem_build={:.2f}s dem_decode={:.2f}s other={:.2f}s"
                        .format(ler_s, dem_build_s, dem_decode_s, ler_other_s)
                    )

            # Log Loss to TensorBoard
            if writer is not None:
                writer.add_scalars(
                    "Loss", {
                        "Training": avg_loss,
                        "Validation": avg_vloss
                    }, epoch_number
                )

                # Log LER to TensorBoard (important evaluation metric)
                if validation_ler is not None:
                    writer.add_scalar("Metrics/LER", validation_ler, epoch_number)
                    if ler_reduction_factor is not None:
                        writer.add_scalar(
                            "Metrics/LER_Reduction_Factor", ler_reduction_factor, epoch_number
                        )

                # Log PyMatching decode speedup (baseline / after pre-decoder), avg across X/Z
                if pymatching_speedup_avg is not None:
                    writer.add_scalar(
                        "Metrics/PyMatching_Speedup", float(pymatching_speedup_avg), epoch_number
                    )

                # Log syndrome density reduction (SDR) as an auxiliary metric (requested for visibility).
                if syndrome_density_reduction is not None:
                    writer.add_scalar(
                        "Metrics/SDR", float(syndrome_density_reduction), epoch_number
                    )

                # Log speed metrics for tracking training efficiency over epochs.
                writer.add_scalar("Speed/throughput_samp_per_s", _throughput, epoch_number)
                writer.add_scalar("Speed/ms_per_batch", _ms_per_batch, epoch_number)
                writer.add_scalar("Speed/epoch_wall_s", _epoch_wall, epoch_number)

                writer.flush()
        log_s = time.perf_counter() - t_log_start

        if dist.world_size > 1:
            t_barrier_start = time.perf_counter()
            torch.distributed.barrier()
            barrier_s = time.perf_counter() - t_barrier_start

        # Early stopping logic
        if use_ler_for_early_stopping and validation_ler is not None:
            current_metric = validation_ler
        else:
            current_metric = avg_vloss

        # SDR threshold gate: only count an epoch as an improvement if SDR qualifies.
        # If SDR was not computed (None), fall back to always qualifying.
        # Default thresholds: 1.5x in factor mode, 33.3% in percent mode (equivalent).
        _sdr_pct = bool(getattr(cfg, 'sdr_as_percent', False))
        _default_threshold = 33.3 if _sdr_pct else 1.5
        syndrome_density_threshold = cfg.get('syndrome_density_threshold', _default_threshold)
        syndrome_qualifies = (
            syndrome_density_reduction is not None and
            syndrome_density_reduction >= syndrome_density_threshold
        )
        sdr_not_computed = syndrome_density_reduction is None

        if current_metric < best_vloss and (syndrome_qualifies or sdr_not_computed):
            best_vloss = current_metric
            epochs_since_best = 0

            if dist.rank == 0:
                t_ckpt_best = time.perf_counter()
                save_checkpoint(
                    path=to_absolute_path(best_model_path),
                    models=model_for_ckpt,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch_number + 1,
                    metadata={
                        "best_vloss": best_vloss,
                        "epochs_since_best": epochs_since_best,
                        "using_ler": use_ler_for_early_stopping,
                    },
                    global_step=global_step,
                )
                ckpt_best_s = time.perf_counter() - t_ckpt_best
        else:
            if not (
                syndrome_qualifies or sdr_not_computed
            ) and current_metric < best_vloss and dist.rank == 0:
                _es_unit = "%" if _sdr_pct else "x"
                print(
                    f"[Early Stopping] Metric improved ({current_metric:.6f} < {best_vloss:.6f}) "
                    f"but SDR {syndrome_density_reduction:.2f}{_es_unit} < threshold {syndrome_density_threshold}{_es_unit}; "
                    f"not counting as improvement."
                )
            epochs_since_best += 1

            if cfg.early_stopping.enabled and epochs_since_best >= cfg.early_stopping.patience:
                print(
                    f"Early stopping triggered after {cfg.early_stopping.patience} epochs without improvement."
                )
                with open(to_absolute_path(early_stoping_path), "w") as f:
                    f.write(
                        f"Early stopping at epoch {epoch_number} with best metric {best_vloss:.6f}\n"
                    )
                break

        # Log early stopping metrics to TensorBoard
        if dist.rank == 0 and writer is not None:
            writer.add_scalar("EarlyStopping/epochs_since_best", epochs_since_best, epoch_number)
            writer.add_scalar("EarlyStopping/best_metric", best_vloss, epoch_number)

        # Track epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        if dist.rank == 0:
            _avg_epoch_s = sum(epoch_times) / len(epoch_times) if epoch_times else epoch_duration
            _eta_epochs = cfg.early_stopping.patience - epochs_since_best if cfg.early_stopping.enabled else 0
            _eta_str = f" ETA-stop={_eta_epochs * _avg_epoch_s / 60:.0f}m" if _eta_epochs > 0 else ""
            print(
                f"[{timestamp}] Best={best_vloss:.6f} wait={epochs_since_best}/{cfg.early_stopping.patience}{_eta_str}"
            )

        # Removed: gc.collect() + empty_cache() cause 100-400ms stall per epoch
        # and empty_cache() causes memory fragmentation on next allocation.
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # Save periodic checkpoint
        if dist.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0:
            t_ckpt_periodic = time.perf_counter()
            save_checkpoint(
                path=to_absolute_path(model_save_path),
                models=model_for_ckpt,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_number + 1,
                metadata={"epochs_completed": epoch + 1},
                global_step=global_step,
            )
            ckpt_periodic_s = time.perf_counter() - t_ckpt_periodic

        if dist.rank == 0:
            epoch_wall_s = time.perf_counter() - epoch_wall_start
            measured_sum = (
                train_total_s + val_total_s + loss_sync_s + gc_s + sdr_s + ler_s + ckpt_best_s +
                ckpt_periodic_s + barrier_s + log_s
            )
            overhead_s = max(epoch_wall_s - measured_sum, 0.0)
            if epoch_wall_s > 0:
                pct_wall = lambda s: 100.0 * s / epoch_wall_s
            else:
                pct_wall = lambda s: 0.0
            print(
                "[Epoch Wall] total={:.2f}s train={:.2f}s val={:.2f}s sync={:.2f}s "
                "gc={:.2f}s sdr={:.2f}s ler={:.2f}s ckpt_best={:.2f}s ckpt_periodic={:.2f}s "
                "barrier={:.2f}s log={:.2f}s overhead={:.2f}s".format(
                    epoch_wall_s, train_total_s, val_total_s, loss_sync_s, gc_s, sdr_s, ler_s,
                    ckpt_best_s, ckpt_periodic_s, barrier_s, log_s, overhead_s
                )
            )
            print(
                "[Epoch Wall %] train={:.1f}% val={:.1f}% sync={:.1f}% gc={:.1f}% sdr={:.1f}% ler={:.1f}% "
                "ckpt_best={:.1f}% ckpt_periodic={:.1f}% barrier={:.1f}% log={:.1f}% overhead={:.1f}%"
                .format(
                    pct_wall(train_total_s), pct_wall(val_total_s), pct_wall(loss_sync_s),
                    pct_wall(gc_s), pct_wall(sdr_s), pct_wall(ler_s), pct_wall(ckpt_best_s),
                    pct_wall(ckpt_periodic_s), pct_wall(barrier_s), pct_wall(log_s),
                    pct_wall(overhead_s)
                )
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        r = os.environ.get("RANK", "?")
        lr = os.environ.get("LOCAL_RANK", "?")
        print(f"\n[!!] RANK={r} LOCAL_RANK={lr} crashed:\n{traceback.format_exc()}\n", flush=True)
        raise
