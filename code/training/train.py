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
Training module for the quantum error correction pre-decoder.

This module provides training functionality with on-the-fly data generation.
All file-based dataset and epoch-config paths have been removed.
"""
import time
import sys
import os
import re
import gc
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
try:
    import torchinfo
except ImportError:
    torchinfo = None
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from torch.cuda.amp import GradScaler
from training.precision import (
    autocast_for_precision,
    should_use_grad_scaler,
    should_use_channels_last_3d,
    module_to_channels_last_3d,
    input_to_channels_last_3d,
    targets_for_bce,
)
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
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


def _sync_cuda_if_needed(device=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def _accumulate_numeric_timing(accumulator, sample):
    if sample is None:
        return
    for key, value in sample.items():
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            accumulator[key] = accumulator.get(key, 0.0) + float(value)


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
    bases_needed = (
        ["X", "Z"] if str(meas_basis).lower() in ("both", "mixed") else [str(meas_basis).upper()]
    )
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
    generator,
    model,
    num_samples,
    batch_size,
    device,
    enable_fp16,
    enable_bf16=False,
    rank=0,
    use_channels_last_3d=False,
):
    """Validation using the configured on-the-fly data generator."""
    loss_fn = torch.nn.BCEWithLogitsLoss()
    running_vloss = 0.0

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
                f"[Validation] Starting validation with data generator, {num_batches} batches (curriculum mode)..."
            )
        else:
            print(f"[Validation] Starting validation with data generator, {num_batches} batches...")
        # Explicitly state whether the validation generator is using a noise model.
        try:
            sim_x = getattr(generator, "sim_X", None)
            sim_z = getattr(generator, "sim_Z", None)
            sim = getattr(generator, "sim", None)
            sims = [s for s in (sim_x, sim_z, sim) if s is not None]
            use_nm = any(getattr(s, "use_noise_model", False) for s in sims)
            if use_nm:
                nm = getattr(sims[0], "noise_model", None)
                if nm is not None:
                    print(f"[Validation] Using explicit noise_model (25p): {nm!r}")
                    print(
                        "[Validation] noise_model idle semantics: "
                        "bulk/CNOT-layer idles use p_idle_cnot_*, "
                        "data-idle during ancilla prep/reset uses p_idle_spam_*, "
                        "data-idle during ancilla measurement is ignored."
                    )
        except Exception as e:
            print(f"[Validation] (noise_model log skipped due to error: {e})")

        # Suppress legacy scalar-p prints when using an explicit noise model.
        # The Torch surface generator's MemoryCircuitTorch doesn't carry the
        # legacy fixed_p / p_min / p_max attributes, so guard each access.
        def _print_p_settings(label, s):
            if not all(hasattr(s, a) for a in ("fixed_p", "p_min", "p_max")):
                return
            print(f"[Validation] Generator p settings{label}: ", end='')
            if s.fixed_p:
                print(f"p={s.p_min:.6f} (fixed)")
            else:
                print(f"p∈[{s.p_min:.6f}, {s.p_max:.6f}]")

        if not use_nm:
            if hasattr(generator, 'sim_X') and hasattr(generator, 'sim_Z') \
                    and generator.sim_X is not None and generator.sim_Z is not None:
                _print_p_settings(" - X", generator.sim_X)
                _print_p_settings(" - Z", generator.sim_Z)
            elif hasattr(generator, 'sim') and generator.sim is not None:
                _print_p_settings("", generator.sim)

    with torch.no_grad():
        for step in range(num_batches):
            val_step = 10000 + step
            trainX, trainY = generator.generate_batch(step=val_step, batch_size=batch_size)

            if step < 8 and rank == 0 and hasattr(generator, 'get_current_pair'):
                d, r = generator.get_current_pair(val_step)
                basis = 'X' if (val_step % 2 == 0) else 'Z'
                print(
                    f"[Val Batch {step}] Using (d={d}, r={r}, basis={basis}) | trainX shape: {trainX.shape}"
                )

            # Keep BCE targets in fp32; never .half() the model or labels.
            trainY = targets_for_bce(trainY)
            # Leave trainX fp32 (autocast casts it); only fix its memory layout
            # so half-precision Conv3D hits the fast Tensor-Core kernel.
            trainX = input_to_channels_last_3d(trainX, use_channels_last_3d)

            with autocast_for_precision(device, enable_fp16, enable_bf16):
                outputs = model(trainX)
                loss = loss_fn(outputs, trainY)

            running_vloss += loss.item()

    avg_vloss = running_vloss / num_batches
    val_time = time.time() - val_start_time

    if rank == 0:
        time_per_batch = val_time / num_batches
        print(
            f"[Validation] Completed {num_batches} batches in {val_time:.1f}s "
            f"({time_per_batch*1000:.1f}ms/batch), avg_vloss={avg_vloss:.5f}"
        )

    return avg_vloss


def _generator_uses_compiled_generation(generator) -> bool:
    """True if the generator runs torch.compile'd HE during batch generation.

    Compiled HE artifacts must not be compiled on one thread and re-entered from
    another (Dynamo raises "FX to symbolically trace a dynamo-optimized
    function"), so when this is True the outer batch prefetch -- which runs
    ``generate_batch`` in a worker thread -- must be disabled and generation kept
    on the main thread.
    """
    for attr in ("sim", "sim_X", "sim_Z"):
        sim = getattr(generator, attr, None)
        if sim is not None and bool(getattr(sim, "use_compile", False)):
            return True
    return False


def _should_use_outer_batch_prefetch(generator) -> bool:
    """Outer ThreadPoolExecutor batch prefetch is safe only with eager generation."""
    return not _generator_uses_compiled_generation(generator)


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
    use_channels_last_3d=False,
    rank=0,
    use_ema=False,
    ema_model=None,
    ema_decay=0.0,
    global_step=0,
    accumulate_steps=1,
    profile_enabled=False,
    profile_log_every=50,
    profile_warmup_steps=2,
    profile_generator_subphases=False,
):
    """Training epoch using the configured on-the-fly data generator."""
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    gradient_clipping_counter = 0

    epoch_start_time = time.time()
    last_log_time = epoch_start_time

    if rank == 0:
        print(f"train_epoch_generator: Starting {steps_per_epoch} batches...")

    running_loss = 0.0
    last_loss = 0.0
    epoch_total_loss = 0.0
    accumulated_samples = 0
    profile_sums = {}
    profile_count = 0

    # Pre-generate first batch before loop
    global_step_for_gen = cumulative_steps_before_epoch
    next_data = generator.generate_batch(
        step=global_step_for_gen,
        batch_size=batch_size,
        return_timing=profile_enabled,
        profile_generator_subphases=profile_generator_subphases,
    )

    # Compiled HE generation must stay on the main thread: it is compiled there
    # (warmup + batch 0) and re-entering it from a prefetch worker triggers a
    # Dynamo FX re-trace error. Only overlap generation via the prefetch worker
    # when generation is eager.
    use_outer_prefetch = _should_use_outer_batch_prefetch(generator)
    if rank == 0 and not use_outer_prefetch:
        print(
            "[Train] Compiled data generation detected; disabling outer batch "
            "prefetch (generation runs on the main thread)."
        )

    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        for step in range(steps_per_epoch):
            if rank == 0 and (step < 2 or step % 200 == 0):
                current_time = time.time()
                elapsed = current_time - epoch_start_time
                if step > 0:
                    time_per_batch = (current_time - last_log_time) / min(step, 200)
                    remaining = time_per_batch * (steps_per_epoch - step)
                    display_loss = epoch_total_loss / step if step > 0 else 0.0
                    print(
                        f"[Epoch {epoch_number}] Batch {step}/{steps_per_epoch} | "
                        f"Loss: {display_loss:.5f} | "
                        f"Elapsed: {elapsed:.1f}s | Per-batch: {time_per_batch*1000:.1f}ms | "
                        f"ETA: {remaining:.1f}s"
                    )
                else:
                    print(
                        f"[Epoch {epoch_number}] Batch {step}/{steps_per_epoch} | Elapsed: {elapsed:.1f}s"
                    )
                if step % 200 == 0 and step > 0:
                    last_log_time = current_time

            global_step_for_gen = cumulative_steps_before_epoch + step
            if profile_enabled:
                trainX, trainY, batch_timing = next_data
            else:
                trainX, trainY = next_data
                batch_timing = None

            # Submit next batch generation in background (overlaps with training below)
            prefetch_submit_s = 0.0
            if step + 1 < steps_per_epoch and use_outer_prefetch:
                next_gen_step = cumulative_steps_before_epoch + step + 1
                submit_t0 = time.perf_counter()
                future = prefetch_pool.submit(
                    generator.generate_batch,
                    step=next_gen_step,
                    batch_size=batch_size,
                    return_timing=profile_enabled,
                    profile_generator_subphases=profile_generator_subphases,
                )
                prefetch_submit_s = time.perf_counter() - submit_t0

            if step < 8 and rank == 0 and hasattr(generator, 'get_current_pair'):
                d, r = generator.get_current_pair(global_step_for_gen)
                basis = 'X' if (global_step_for_gen % 2 == 0) else 'Z'
                print(
                    f"[Batch {step}] Using (d={d}, r={r}, basis={basis}) | trainX shape: {trainX.shape}"
                )

            # Keep BCE targets in fp32; never .half() the model or labels.
            trainY = targets_for_bce(trainY)
            # Leave trainX fp32 (autocast casts it); only fix its memory layout
            # so half-precision Conv3D hits the fast Tensor-Core kernel.
            trainX = input_to_channels_last_3d(trainX, use_channels_last_3d)
            optimizer_step_skipped = False

            current_batch_size = trainX.shape[0]
            model_fwd_bwd_s = 0.0
            optimizer_step_s = 0.0

            if profile_enabled:
                _sync_cuda_if_needed(device)
                model_t0 = time.perf_counter()

            with autocast_for_precision(device, enable_fp16, enable_bf16):
                outputs = model(trainX)
                loss = loss_fn(outputs, trainY)

            scaler.scale(loss).backward()
            if profile_enabled:
                _sync_cuda_if_needed(device)
                model_fwd_bwd_s = time.perf_counter() - model_t0
            accumulated_samples += current_batch_size

            if (step + 1) % accumulate_steps == 0:
                if profile_enabled:
                    _sync_cuda_if_needed(device)
                    opt_t0 = time.perf_counter()
                scaler.unscale_(optimizer)

                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(accumulated_samples)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if grad_norm > 1.0:
                    gradient_clipping_counter += 1

                scale_before_step = scaler.get_scale() if scaler.is_enabled() else None
                scaler.step(optimizer)
                scaler.update()
                # On a skipped step (fp16 grad overflow) the scaler lowers the
                # scale and does NOT apply the optimizer update; don't advance
                # scheduler/EMA/global_step in that case.
                optimizer_step_skipped = (
                    scaler.is_enabled() and scaler.get_scale() < scale_before_step
                )
                optimizer.zero_grad()

                if not optimizer_step_skipped:
                    scheduler.step()

                if not optimizer_step_skipped and use_ema and ema_model is not None:
                    with torch.no_grad():
                        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                            if ema_param.dtype.is_floating_point:
                                ema_param.data.mul_(ema_decay).add_(
                                    param.data.to(ema_param.dtype), alpha=1.0 - ema_decay
                                )
                        # Copy buffers (BatchNorm running_mean/running_var) from the
                        # training model to the EMA model.  Without this, the EMA model
                        # (which is always in eval mode) keeps its initial BN stats and
                        # produces garbage during validation/inference.
                        for ema_buf, model_buf in zip(ema_model.buffers(), model.buffers()):
                            ema_buf.data.copy_(model_buf.data)

                if not optimizer_step_skipped:
                    global_step += 1
                accumulated_samples = 0
                if profile_enabled:
                    _sync_cuda_if_needed(device)
                    optimizer_step_s = time.perf_counter() - opt_t0

            num_elements = outputs.numel()
            batch_loss_mean = loss.item() / num_elements
            running_loss += batch_loss_mean
            epoch_total_loss += batch_loss_mean

            if (step + 1) % accumulate_steps == 0:
                last_loss = running_loss / accumulate_steps
                if rank == 0:
                    tb_writer.add_scalar('Loss/train_step', last_loss, global_step)
                    tb_writer.add_scalar(
                        'LearningRate/train',
                        scheduler.get_last_lr()[0], global_step
                    )
                running_loss = 0.0

            # Get the next batch: from the prefetch worker (eager generation) or
            # synchronously on the main thread (compiled generation).
            future_wait_s = 0.0
            if step + 1 < steps_per_epoch:
                wait_t0 = time.perf_counter()
                if use_outer_prefetch:
                    next_data = future.result()
                else:
                    next_data = generator.generate_batch(
                        step=cumulative_steps_before_epoch + step + 1,
                        batch_size=batch_size,
                        return_timing=profile_enabled,
                        profile_generator_subphases=profile_generator_subphases,
                    )
                future_wait_s = time.perf_counter() - wait_t0

            if profile_enabled and step >= profile_warmup_steps:
                step_profile = {
                    "prefetch_submit_s": prefetch_submit_s,
                    "future_wait_s": future_wait_s,
                    "model_fwd_bwd_s": model_fwd_bwd_s,
                    "optimizer_step_s": optimizer_step_s,
                }
                if batch_timing is not None:
                    step_profile.update(batch_timing)
                _accumulate_numeric_timing(profile_sums, step_profile)
                profile_count += 1

                if (
                    rank == 0 and profile_log_every > 0 and
                    profile_count % int(profile_log_every) == 0
                ):
                    avg = {k: v / profile_count for k, v in profile_sums.items()}
                    print(
                        f"[Epoch {epoch_number}] Timing avg over {profile_count} batches | "
                        f"gen={avg.get('generator_total_s', 0.0) * 1000:.1f}ms | "
                        f"wait={avg.get('future_wait_s', 0.0) * 1000:.1f}ms | "
                        f"model={avg.get('model_fwd_bwd_s', 0.0) * 1000:.1f}ms | "
                        f"opt={avg.get('optimizer_step_s', 0.0) * 1000:.1f}ms"
                    )
                    if "raw_sample_s" in avg:
                        print(
                            f"  raw: sample={avg.get('raw_sample_s', 0.0) * 1000:.1f}ms | "
                            f"agg={avg.get('raw_aggregate_s', 0.0) * 1000:.1f}ms | "
                            f"he={avg.get('he_total_s', 0.0) * 1000:.1f}ms | "
                            f"format={avg.get('format_s', 0.0) * 1000:.1f}ms"
                        )

    avg_loss = epoch_total_loss / steps_per_epoch

    if rank == 0:
        total_time = time.time() - epoch_start_time
        time_per_batch = total_time / steps_per_epoch
        print(
            f"train_epoch_generator: Completed {steps_per_epoch} batches in {total_time:.1f}s "
            f"({time_per_batch*1000:.1f}ms/batch), avg_loss={avg_loss:.5f}"
        )
        if profile_enabled and profile_count > 0:
            avg = {k: v / profile_count for k, v in profile_sums.items()}
            print(
                f"train_epoch_generator timing summary ({profile_count} profiled batches, "
                f"after warmup={profile_warmup_steps})"
            )
            print(
                f"  generator={avg.get('generator_total_s', 0.0) * 1000:.1f}ms | "
                f"wait={avg.get('future_wait_s', 0.0) * 1000:.1f}ms | "
                f"model={avg.get('model_fwd_bwd_s', 0.0) * 1000:.1f}ms | "
                f"opt={avg.get('optimizer_step_s', 0.0) * 1000:.1f}ms | "
                f"submit={avg.get('prefetch_submit_s', 0.0) * 1000:.3f}ms"
            )
            if "raw_sample_s" in avg:
                print(
                    f"  raw breakdown: carry={avg.get('raw_propagate_carry_s', 0.0) * 1000:.1f}ms | "
                    f"sample={avg.get('raw_sample_s', 0.0) * 1000:.1f}ms | "
                    f"y={avg.get('raw_decompose_y_s', 0.0) * 1000:.1f}ms | "
                    f"agg={avg.get('raw_aggregate_s', 0.0) * 1000:.1f}ms | "
                    f"ff/meas={avg.get('raw_measure_ff_s', 0.0) * 1000:.1f}ms | "
                    f"s1s2={avg.get('raw_s1s2_propagate_s', 0.0) * 1000:.1f}ms"
                )
                print(
                    f"  post-raw: he={avg.get('he_total_s', 0.0) * 1000:.1f}ms | "
                    f"dlpack={avg.get('dlpack_s', 0.0) * 1000:.1f}ms | "
                    f"format={avg.get('format_s', 0.0) * 1000:.1f}ms"
                )
            tb_writer.add_scalar(
                'Timing/generator_total_ms',
                avg.get('generator_total_s', 0.0) * 1000, epoch_number
            )
            tb_writer.add_scalar(
                'Timing/future_wait_ms',
                avg.get('future_wait_s', 0.0) * 1000, epoch_number
            )
            tb_writer.add_scalar(
                'Timing/model_fwd_bwd_ms',
                avg.get('model_fwd_bwd_s', 0.0) * 1000, epoch_number
            )
            tb_writer.add_scalar(
                'Timing/optimizer_step_ms',
                avg.get('optimizer_step_s', 0.0) * 1000, epoch_number
            )

    return avg_loss, global_step


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using on-the-fly data generation."""
    OmegaConf.set_struct(cfg, False)

    # Env-based overrides for samples and epochs. These apply unconditionally so
    # CI smoke jobs and quick local runs can shrink the workload without needing
    # to edit the YAML. Mirrors public/main's train.py behaviour.
    _train_samples_env = os.environ.get("PREDECODER_TRAIN_SAMPLES")
    _val_samples_env = os.environ.get("PREDECODER_VAL_SAMPLES")
    _test_samples_env = os.environ.get("PREDECODER_TEST_SAMPLES")
    _epochs_env = os.environ.get("PREDECODER_TRAIN_EPOCHS")
    try:
        if _train_samples_env:
            cfg.train.num_samples = int(_train_samples_env)
    except Exception:
        pass
    try:
        if _val_samples_env:
            cfg.val.num_samples = int(_val_samples_env)
    except Exception:
        pass
    try:
        if _test_samples_env:
            cfg.test.num_samples = int(_test_samples_env)
    except Exception:
        pass
    try:
        if _epochs_env:
            cfg.train.epochs = int(_epochs_env)
    except Exception:
        pass

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

    if dist.rank == 0:
        print(f"Effective workflow.task: {cfg.workflow.task}")
        print(f"Using LR scheduler type: {cfg.lr_scheduler.type}")
        print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    print(f"Rank {dist.rank} running on {dist.device}")

    # Configure QEC metrics (LER, syndrome density) based on code family.
    #
    # Color code uses Chromobius-based LER via logical_error_rate_color.py
    # Surface code uses Stim-based LER via logical_error_rate.py
    code_name = str(getattr(cfg, "code", "surface")).lower()
    configure_metrics(rank=dist.rank, code=code_name)

    # === Data Generator Setup ===
    if dist.rank == 0:
        print("=" * 80)
        print("🚀 Setting up on-the-fly data generation")
        print("=" * 80)

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
    noise_model_obj = None
    if noise_model_cfg is not None:
        from qec.noise_model import (
            NoiseModel,
            get_grouped_totals,
            get_training_upscaled_noise_model,
        )
        nm_dict = OmegaConf.to_container(noise_model_cfg, resolve=True
                                        ) if hasattr(noise_model_cfg, "items") else noise_model_cfg
        # Allow configs to specify `noise_model: null`
        if nm_dict is not None:
            user_noise_model_obj = NoiseModel.from_config_dict(dict(nm_dict))
            skip_upscale = bool(getattr(cfg.data, "skip_noise_upscaling", False)) or str(
                os.environ.get("PREDECODER_SKIP_NOISE_UPSCALING", "")
            ).strip().lower() in ("1", "true", "yes", "on")
            noise_model_obj, upscale_info = get_training_upscaled_noise_model(
                user_noise_model_obj,
                code_type=code_name,
                skip_upscale=skip_upscale,
            )
            # Force fixed-p mode with a conservative scalar placeholder when using noise_model.
            # The actual sampling probabilities come from `noise_model_obj`.
            # IMPORTANT: during training we may apply drift (±25%) around the reference noise model,
            # so buffer sizing uses 1.25× the maximum active reference probability.
            p_error_value = float(1.25 * noise_model_obj.get_max_probability())
            p_min_value = p_error_value
            p_max_value = p_error_value
            if dist.rank == 0:
                print(
                    "[Train] Using explicit noise_model from config (25p). "
                    f"p_error/p_min/p_max sizing placeholder -> {p_error_value:.6g}"
                )
                print(f"[Train] configured noise_model summary: {user_noise_model_obj!r}")
                print(f"[Train] training noise_model summary: {noise_model_obj!r}")
                print(f"[Train] training noise upscaling: {upscale_info['message']}")
                totals = get_grouped_totals(noise_model_obj)
                print(
                    "[Train] training noise grouped totals: "
                    f"max_group={totals['max_group']:.6g}, "
                    f"prep_X={totals['p_prep_X']:.6g}, prep_Z={totals['p_prep_Z']:.6g}, "
                    f"meas_X={totals['p_meas_X']:.6g}, meas_Z={totals['p_meas_Z']:.6g}, "
                    f"idle_cnot={totals['p_idle_cnot']:.6g}, "
                    f"idle_spam_effective={totals['p_idle_spam_effective']:.6g}, "
                    f"cnot={totals['p_cnot']:.6g}"
                )
                print(
                    "[Train] noise_model idle semantics: "
                    "bulk/CNOT-layer idles use p_idle_cnot_*, "
                    "data-idle during ancilla prep/reset uses p_idle_spam_*, "
                    "data-idle during ancilla measurement is ignored."
                )
                print(
                    "[Train] noise_model totals: "
                    f"idle_cnot_total={noise_model_obj.get_total_idle_cnot_probability():.6g}, "
                    f"idle_spam_total={noise_model_obj.get_total_idle_spam_probability():.6g}, "
                    f"cnot_total={noise_model_obj.get_total_cnot_probability():.6g}"
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
    use_weight2_timelike = getattr(cfg.data, 'use_weight2_timelike', False)
    max_passes_w1 = getattr(cfg.data, 'max_passes_w1', 32)
    max_passes_w2 = getattr(cfg.data, 'max_passes_w2', 32)
    decompose_y = getattr(cfg.data, 'decompose_y', True)
    # Color-code specific: superdense schedule toggle influences Y-decomposition ruleset.
    # Default to surface-code rules unless we are explicitly training a (superdense) color-code circuit.
    code_name = str(getattr(cfg, "code", "surface")).lower()
    superdense = bool(getattr(cfg.data, "superdense", True))
    y_decomposition_ruleset = "superdense_color_code" if (
        code_name.startswith("color") and superdense
    ) else "surface_code"
    precomputed_frames_dir = getattr(cfg.data, 'precomputed_frames_dir', None)
    code_rotation = getattr(cfg.data, 'code_rotation', 'XV')

    # Code-family toggle: surface vs color code.
    gen_code = str(getattr(cfg, "code", "surface")).lower()
    schedule = getattr(cfg.data, "schedule", "nearest-neighbor")
    enable_z_feedforward = bool(getattr(cfg.data, "enable_z_feedforward", True))
    # Default on: the sampled-only FF override is a strict trainY-label
    # improvement on the color-code path (validated exhaustively against
    # fake r1->r2 diffs at d=5/7/9) and a no-op on the surface-code path
    # (the surface-code memory circuit does not accept this flag, and
    # the generator only forwards it on the color-code branch).
    apply_data_x_override = bool(getattr(cfg.data, "apply_data_x_override", True))

    # Color-code spacelike HE knobs for the Torch+cuStabilizer generator.
    color_apply_spacelike_he = bool(getattr(cfg.data, "apply_spacelike_he", True))
    color_he_max_iterations = int(getattr(cfg.data, "he_max_iterations", 16))
    color_use_coset_search = bool(getattr(cfg.data, "use_coset_search", False))

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
        use_parallel_spacelike=bool(getattr(cfg.data, 'use_parallel_spacelike', False)),
    )

    if use_multi_pairs:
        from data.generator_torch_multi import MultiQCDataGeneratorTorch
        _multi_common = dict(
            distances=[int(d) for d in multi_d],
            rounds=[int(r) for r in multi_r],
            code=gen_code,
            p_error=p_error_value,
            p_min=p_min_value,
            p_max=p_max_value,
            measure_basis=cfg.meas_basis,
            rank=dist.local_rank,
            global_rank=dist.rank,
            timelike_he=timelike_he,
            num_he_cycles=num_he_cycles,
            use_weight2_timelike=use_weight2_timelike,
            max_passes_w1=max_passes_w1,
            max_passes_w2=max_passes_w2,
            decompose_y=False,
            precomputed_frames_dir=precomputed_frames_dir,
            code_rotation=code_rotation,
            noise_model=noise_model_obj,
            schedule=schedule,
            enable_z_feedforward=enable_z_feedforward,
            apply_data_x_override=apply_data_x_override,
            apply_spacelike_he=color_apply_spacelike_he,
            he_max_iterations=color_he_max_iterations,
            use_coset_search=color_use_coset_search,
            use_compile=bool(getattr(cfg.data, "use_compile", False)),
            compile_chunk_size=int(getattr(cfg.data, "compile_chunk_size", 2)),
            compute_dtype=_compute_dtype,
            use_dense_overlap=bool(getattr(cfg.data, "use_dense_overlap", False)),
            use_parallel_spacelike=bool(getattr(cfg.data, "use_parallel_spacelike", False)),
        )
        train_generator = MultiQCDataGeneratorTorch(
            mode='train',
            verbose=(dist.rank == 0),
            base_seed=base_seed,
            **_multi_common,
        )
        val_generator = MultiQCDataGeneratorTorch(
            mode='test',
            verbose=False,
            base_seed=base_seed + 100_000_000,
            **_multi_common,
        )
    else:
        if code_name.startswith("color"):
            if precomputed_frames_dir is None:
                raise ValueError(
                    "code=color requires data.precomputed_frames_dir to point at a color "
                    "augmented DEM bundle (see qec.precompute_dem --code color)."
                )
            from data.generator_torch_color import ColorQCDataGeneratorTorch
            train_generator = ColorQCDataGeneratorTorch(
                distance=cfg.distance,
                n_rounds=cfg.n_rounds,
                schedule=schedule,
                measure_basis=cfg.meas_basis,
                precomputed_frames_dir=precomputed_frames_dir,
                enable_z_feedforward=enable_z_feedforward,
                apply_data_x_override=apply_data_x_override,
                apply_spacelike_he=color_apply_spacelike_he,
                he_max_iterations=color_he_max_iterations,
                use_coset_search=color_use_coset_search,
                rank=dist.local_rank,
                global_rank=dist.rank,
                base_seed=base_seed,
                verbose=(dist.rank == 0),
                noise_model=noise_model_obj,
                p_error=p_error_value,
                p_min=p_min_value,
                p_max=p_max_value,
            )
            val_generator = ColorQCDataGeneratorTorch(
                distance=cfg.distance,
                n_rounds=cfg.n_rounds,
                schedule=schedule,
                measure_basis=cfg.meas_basis,
                precomputed_frames_dir=precomputed_frames_dir,
                enable_z_feedforward=enable_z_feedforward,
                apply_data_x_override=apply_data_x_override,
                apply_spacelike_he=color_apply_spacelike_he,
                he_max_iterations=color_he_max_iterations,
                use_coset_search=color_use_coset_search,
                rank=dist.local_rank,
                global_rank=dist.rank,
                base_seed=base_seed + 100_000_000,
                verbose=False,
                noise_model=noise_model_obj,
                p_error=p_error_value,
                p_min=p_min_value,
                p_max=p_max_value,
            )
        else:
            # Surface code: Torch-only path (matches public/main).
            from data.generator_torch import QCDataGeneratorTorch
            _gen_torch_common = dict(
                distance=cfg.distance,
                n_rounds=cfg.n_rounds,
                p_error=p_error_value,
                p_min=p_min_value,
                p_max=p_max_value,
                measure_basis=cfg.meas_basis,
                rank=dist.local_rank,
                global_rank=dist.rank,
                timelike_he=timelike_he,
                num_he_cycles=num_he_cycles,
                use_weight2=bool(getattr(cfg.data, "use_weight2", use_weight2_timelike)),
                max_passes_w1=max_passes_w1,
                max_passes_w2=max_passes_w2,
                decompose_y=False,
                precomputed_frames_dir=precomputed_frames_dir,
                code_rotation=code_rotation,
                noise_model=noise_model_obj,
                use_compile=bool(getattr(cfg.data, "use_compile", False)),
                compile_chunk_size=int(getattr(cfg.data, "compile_chunk_size", 2)),
                compute_dtype=_compute_dtype,
                use_coset_search=bool(getattr(cfg.data, "use_coset_search", False)),
                coset_max_generators=int(getattr(cfg.data, "coset_max_generators", 20)),
                use_dense_overlap=bool(getattr(cfg.data, "use_dense_overlap", False)),
                use_parallel_spacelike=bool(getattr(cfg.data, "use_parallel_spacelike", False)),
            )
            train_generator = QCDataGeneratorTorch(
                mode="train",
                verbose=(dist.rank == 0),
                base_seed=base_seed,
                **_gen_torch_common,
            )
            val_generator = QCDataGeneratorTorch(
                mode="test",
                verbose=False,
                base_seed=base_seed,
                seed_offset=100_000_000,
                **_gen_torch_common,
            )

    # Create test generator
    test_distance_override = getattr(cfg.test, 'distance', None)
    test_rounds_override = getattr(cfg.test, 'n_rounds', None)
    test_timelike_he = getattr(cfg.test, 'timelike_he', timelike_he)
    test_num_he_cycles = getattr(cfg.test, 'num_he_cycles', num_he_cycles)
    test_use_weight2_timelike = getattr(cfg.test, 'use_weight2_timelike', use_weight2_timelike)
    test_max_passes_w1 = getattr(cfg.test, 'max_passes_w1', max_passes_w1)
    test_max_passes_w2 = getattr(cfg.test, 'max_passes_w2', max_passes_w2)

    if test_distance_override is not None and test_rounds_override is not None:
        test_d, test_r = int(test_distance_override), int(test_rounds_override)
    elif use_multi_pairs:
        largest_idx = max(range(len(multi_d)), key=lambda i: int(multi_d[i]))
        test_d, test_r = int(multi_d[largest_idx]), int(multi_r[largest_idx])
    else:
        test_d, test_r = cfg.distance, cfg.n_rounds

    def _build_test_generator():
        if code_name.startswith("color"):
            from data.generator_torch_color import ColorQCDataGeneratorTorch
            return ColorQCDataGeneratorTorch(
                distance=test_d,
                n_rounds=test_r,
                schedule=schedule,
                measure_basis=cfg.meas_basis,
                precomputed_frames_dir=precomputed_frames_dir,
                enable_z_feedforward=enable_z_feedforward,
                apply_data_x_override=apply_data_x_override,
                apply_spacelike_he=color_apply_spacelike_he,
                he_max_iterations=color_he_max_iterations,
                use_coset_search=color_use_coset_search,
                rank=dist.local_rank,
                global_rank=dist.rank,
                base_seed=base_seed + 200_000_000,
                verbose=False,
                noise_model=noise_model_obj,
                p_error=p_error_value,
                p_min=p_min_value,
                p_max=p_max_value,
            )
        from data.generator_torch import QCDataGeneratorTorch
        return QCDataGeneratorTorch(
            distance=test_d,
            n_rounds=test_r,
            p_error=p_error_value,
            p_min=p_min_value,
            p_max=p_max_value,
            measure_basis=cfg.meas_basis,
            rank=dist.local_rank,
            global_rank=dist.rank,
            mode="test",
            verbose=(dist.rank == 0),
            timelike_he=test_timelike_he,
            num_he_cycles=test_num_he_cycles,
            use_weight2=bool(getattr(cfg.data, "use_weight2", test_use_weight2_timelike)),
            max_passes_w1=test_max_passes_w1,
            max_passes_w2=test_max_passes_w2,
            decompose_y=False,
            precomputed_frames_dir=precomputed_frames_dir,
            code_rotation=code_rotation,
            noise_model=noise_model_obj,
            base_seed=base_seed,
            seed_offset=200_000_000,
            use_compile=bool(getattr(cfg.data, "use_compile", False)),
            compile_chunk_size=int(getattr(cfg.data, "compile_chunk_size", 2)),
            compute_dtype=_compute_dtype,
        )

    # The Torch validation/LER path always passes generator=None to compute_*
    # helpers, so an explicit test_generator is no longer needed. Keep the
    # `_build_test_generator` helper in scope for any future Torch-side reuse.
    _ = _build_test_generator  # keep reference; harmless no-op
    test_generator = None

    if dist.rank == 0:
        print("✅ Data generator initialized successfully")
        print("=" * 80)

    # Generate sample batch for shape info
    sample_trainX, sample_trainY = train_generator.generate_batch(step=0, batch_size=1)
    cfg.model.input_channels = sample_trainX.shape[1]
    cfg.model.out_channels = sample_trainY.shape[1]

    profile_enabled = bool(OmegaConf.select(cfg, "profiling.enabled", default=False))
    profile_log_every = int(OmegaConf.select(cfg, "profiling.log_every", default=50) or 50)
    profile_warmup_steps = int(OmegaConf.select(cfg, "profiling.warmup_steps", default=2) or 2)
    profile_generator_subphases = bool(
        OmegaConf.select(cfg, "profiling.generator_subphases", default=False)
    )

    # Create model
    base_model = ModelFactory.create_model(cfg).to(dist.device)

    # Mixed precision is handled by autocast in the train/val steps; keep
    # parameters, BatchNorm state, and optimizer state in fp32. Only switch the
    # 5D Conv3D stack to channels_last_3d (NDHWC) under AMP for fast kernels.
    use_channels_last_3d = (
        should_use_channels_last_3d(
            cfg.enable_fp16, getattr(cfg, 'enable_bf16', False), dist.device
        ) and bool(
            getattr(cfg, 'amp', {}).get('channels_last_3d', True) if hasattr(cfg, 'amp') else True
        )
    )
    if use_channels_last_3d:
        base_model = module_to_channels_last_3d(base_model, True)

    # Load checkpoint before creating EMA model, so EMA starts from checkpoint weights
    # (and correct BatchNorm running stats) rather than random initialization.
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

    # Create EMA model AFTER checkpoint load so it inherits trained weights AND
    # correct BatchNorm running statistics from the checkpoint.
    ema_model = deepcopy(base_model).to(dist.device).eval()
    if use_channels_last_3d:
        ema_model = module_to_channels_last_3d(ema_model, True)

    # Optional torch.compile
    if getattr(cfg, "torch_compile", False):
        compile_mode = getattr(cfg, "torch_compile_mode", "max-autotune")
        if dist.rank == 0:
            print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        base_model = torch.compile(base_model, mode=compile_mode)

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

    # Print model summary (skipped if torchinfo isn't installed)
    if dist.rank == 0 and torchinfo is not None:
        summary_input = sample_trainX.to(dist.device)
        # Leave summary_input fp32; autocast handles precision. Match layout only.
        summary_input = input_to_channels_last_3d(summary_input, use_channels_last_3d)

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
    # Full training runs keep the default warmup.
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

    # Initialize scaler. fp16 needs loss scaling; bf16/fp32 do not. master
    # weights stay fp32, which is what makes scaling meaningful.
    use_grad_scaler = should_use_grad_scaler(cfg.enable_fp16, dist.device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    # TensorBoard writer (skipped if tensorboard isn't installed)
    writer = None
    if dist.rank == 0 and SummaryWriter is not None:
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
                # With PREDECODER_LER_FINAL_ONLY=1 the per-epoch metric is validation
                # loss even when LER validation is enabled, so expect a loss-based best.
                ler_final_only = os.environ.get("PREDECODER_LER_FINAL_ONLY", "0") == "1"
                expect_ler_metric = use_ler_for_early_stopping and not ler_final_only
                # Only restore best_vloss if the metric type matches (both LER or both loss)
                if saved_using_ler == expect_ler_metric:
                    best_vloss = checkpoint_dict['metadata']['best_vloss']
                    if 'epochs_since_best' in checkpoint_dict['metadata']:
                        epochs_since_best = checkpoint_dict['metadata']['epochs_since_best']
                    if dist.rank == 0:
                        metric_name = "LER" if expect_ler_metric else "validation loss"
                        print(f"[Checkpoint] Restored best {metric_name}: {best_vloss:.6f}")
                else:
                    if dist.rank == 0:
                        old_metric = "LER" if saved_using_ler else "validation loss"
                        new_metric = "LER" if expect_ler_metric else "validation loss"
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
                writer.add_scalar("BatchSize", effective_batch_size, epoch_number)

        model.train(True)

        if epoch == init_epoch:
            cumulative_steps = 0

        avg_loss, global_step = train_epoch(
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
            use_channels_last_3d=use_channels_last_3d,
            rank=dist.rank,
            use_ema=cfg.ema.use_ema,
            ema_model=ema_model,
            ema_decay=ema_decay,
            global_step=global_step,
            accumulate_steps=accumulate_steps,
            profile_enabled=profile_enabled,
            profile_log_every=profile_log_every,
            profile_warmup_steps=profile_warmup_steps,
            profile_generator_subphases=profile_generator_subphases,
        )

        cumulative_steps += steps_per_epoch

        model.eval()
        model_to_eval = ema_model if cfg.ema.use_ema else model
        model_for_ckpt = model_to_eval.module if isinstance(model_to_eval, DDP) else model_to_eval

        val_samples_per_gpu = cfg.val.num_samples // dist.world_size

        avg_vloss = validation_step(
            generator=val_generator,
            model=model_to_eval,
            num_samples=val_samples_per_gpu,
            batch_size=per_device_batch_size,
            device=dist.device,
            enable_fp16=cfg.enable_fp16,
            enable_bf16=getattr(cfg, 'enable_bf16', False),
            use_channels_last_3d=use_channels_last_3d,
            rank=dist.rank
        )

        # Synchronize losses
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([avg_loss], device=dist.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            avg_loss = float(t.item())

            v = torch.tensor([avg_vloss], device=dist.device)
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.AVG)
            avg_vloss = float(v.item())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute LER and syndrome density if enabled
        use_ler_for_early_stopping = getattr(cfg, 'validation_ler', False)
        # CI smoke-run env knobs (smoke_run.sh sets both to 1):
        # PREDECODER_DISABLE_SDR     — skip syndrome density reduction
        # PREDECODER_LER_FINAL_ONLY  — only compute LER on the final epoch
        disable_sdr = os.environ.get("PREDECODER_DISABLE_SDR", "0") == "1"
        ler_final_only = os.environ.get("PREDECODER_LER_FINAL_ONLY", "0") == "1"
        run_ler_this_epoch = use_ler_for_early_stopping and (
            (not ler_final_only) or (epoch_number == (cfg.train.epochs - 1))
        )
        validation_ler = None
        ler_reduction_factor = None
        syndrome_density_reduction = None
        syndrome_density_threshold = 1.5

        if run_ler_this_epoch:
            try:
                orig_cfg_distance, orig_cfg_n_rounds = cfg.distance, cfg.n_rounds
                cfg.distance, cfg.n_rounds = test_d, test_r

                if disable_sdr:
                    if dist.rank == 0:
                        print("[Syndrome Density] Skipped (PREDECODER_DISABLE_SDR=1)")
                    syndrome_density_reduction = None
                else:
                    syndrome_density_reduction = compute_syndrome_density(
                        model=model_to_eval,
                        device=dist.device,
                        dist=dist,
                        cfg=cfg,
                        generator=None,
                        rank=dist.rank,
                    )

                    if isinstance(syndrome_density_reduction, dict):
                        syndrome_density_reduction = sum(syndrome_density_reduction.values()
                                                        ) / len(syndrome_density_reduction)

                if syndrome_density_reduction is not None and torch.distributed.is_available(
                ) and torch.distributed.is_initialized():
                    sd_tensor = torch.tensor([syndrome_density_reduction], device=dist.device)
                    torch.distributed.all_reduce(sd_tensor, op=torch.distributed.ReduceOp.AVG)
                    syndrome_density_reduction = float(sd_tensor.item())

                ler_result = compute_validation_ler(
                    model=model_to_eval,
                    device=dist.device,
                    dist=dist,
                    cfg=cfg,
                    generator=None,
                    rank=dist.rank,
                )

                if isinstance(ler_result, tuple):
                    # public's compute_validation_ler returns
                    # (validation_ler, ler_reduction_factor, pymatching_speedup_avg)
                    if len(ler_result) >= 1:
                        validation_ler = ler_result[0]
                    if len(ler_result) >= 2:
                        ler_reduction_factor = ler_result[1]
                elif isinstance(ler_result, dict):
                    ler_values = [
                        v[0]
                        for v in ler_result.values()
                        if isinstance(v, tuple) and len(v) >= 1 and v[0] is not None
                    ]
                    validation_ler = sum(ler_values) / len(ler_values) if ler_values else None
                else:
                    validation_ler = ler_result

                if validation_ler is not None and torch.distributed.is_available(
                ) and torch.distributed.is_initialized():
                    ler_tensor = torch.tensor([validation_ler], device=dist.device)
                    torch.distributed.all_reduce(ler_tensor, op=torch.distributed.ReduceOp.AVG)
                    validation_ler = float(ler_tensor.item())

            finally:
                cfg.distance, cfg.n_rounds = orig_cfg_distance, orig_cfg_n_rounds

        # Log metrics
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        is_color = code_name == "color"
        ler_label = "LER/round" if is_color else "LER"
        if dist.rank == 0:
            if use_ler_for_early_stopping and validation_ler is not None:
                print(
                    f"[{timestamp}] LOSS train {avg_loss:.5f} valid {avg_vloss:.5f} {ler_label} {validation_ler:.6f}"
                )
            else:
                print(f"[{timestamp}] LOSS train {avg_loss:.5f} valid {avg_vloss:.5f}")

            # Log Loss to TensorBoard
            writer.add_scalars(
                "Loss", {
                    "Training": avg_loss,
                    "Validation": avg_vloss
                }, epoch_number
            )

            # Log LER to TensorBoard (important evaluation metric)
            # Color code reports LER per round; surface code reports total LER
            if validation_ler is not None:
                writer.add_scalar(f"Metrics/{ler_label}", validation_ler, epoch_number)
                if ler_reduction_factor is not None:
                    writer.add_scalar(
                        "Metrics/LER_Reduction_Factor", ler_reduction_factor, epoch_number
                    )

            # Log Syndrome Density Reduction to TensorBoard (important evaluation metric)
            if syndrome_density_reduction is not None:
                writer.add_scalar("Metrics/SDR", syndrome_density_reduction, epoch_number)

            writer.flush()

        if dist.world_size > 1:
            torch.distributed.barrier()

        # Early stopping logic
        if use_ler_for_early_stopping and validation_ler is not None:
            current_metric = validation_ler
            syndrome_qualifies = syndrome_density_reduction is not None and syndrome_density_reduction >= syndrome_density_threshold
        else:
            current_metric = avg_vloss
            syndrome_qualifies = True
        # SDR may be disabled in CI smoke runs (PREDECODER_DISABLE_SDR=1) or
        # before the first SDR validation — treat "no SDR computed yet" as
        # qualifying so we still snapshot the best-loss checkpoint.
        sdr_not_computed = syndrome_density_reduction is None

        if current_metric < best_vloss and (syndrome_qualifies or sdr_not_computed):
            best_vloss = current_metric
            epochs_since_best = 0

            if dist.rank == 0:
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
                        # Record the metric that actually produced best_vloss: when LER
                        # extraction fails, current_metric falls back to validation loss,
                        # and resume relies on this flag to tell the two scales apart.
                        "using_ler": use_ler_for_early_stopping and validation_ler is not None,
                    },
                    global_step=global_step,
                )
        elif current_metric >= best_vloss and syndrome_qualifies:
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
        if dist.rank == 0:
            writer.add_scalar("EarlyStopping/epochs_since_best", epochs_since_best, epoch_number)
            writer.add_scalar("EarlyStopping/best_metric", best_vloss, epoch_number)
            if syndrome_density_reduction is not None:
                writer.add_scalar(
                    "EarlyStopping/syndrome_qualifies", float(syndrome_qualifies), epoch_number
                )

        # Track epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        if dist.rank == 0:
            print(
                f"[{timestamp}] Best metric {best_vloss:.6f}, Epochs since best: {epochs_since_best}, Epoch time {epoch_duration/60:.1f}m"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save periodic checkpoint
        if dist.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0:
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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        r = os.environ.get("RANK", "?")
        lr = os.environ.get("LOCAL_RANK", "?")
        print(f"\n[!!] RANK={r} LOCAL_RANK={lr} crashed:\n{traceback.format_exc()}\n", flush=True)
        raise
