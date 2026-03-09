#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Generic SLURM submission script for pre-decoder training.
# Adapt the #SBATCH directives below to your cluster, then submit with:
#
#   export SHARED_OUTPUT_DIR=$HOME/predecoder_outputs
#   sbatch code/scripts/sbatch_train.sh
#
# ──────────────────────────────────────────────────────────────
# Environment variables (all optional, sensible defaults):
#
#   SHARED_OUTPUT_DIR   Persistent directory for checkpoints, logs, models.
#                       (default: $HOME/predecoder_outputs)
#   EXPERIMENT_NAME     Sub-directory under outputs/ for this run.
#                       (default: qec-decoder-depolarizing-r9-fp8)
#   CONFIG_NAME         Hydra config name without .yaml extension.
#                       (default: config_qec_decoder_r9_fp8)
#   GPUS                Number of GPUs to use (must match --gres). (default: 1)
#   FRESH_START         Set to 1 to ignore existing checkpoints. (default: 0)
#   PREDECODER_TRAIN_EPOCHS   Override epoch count.
#   PREDECODER_TRAIN_SAMPLES  Override samples per epoch (bypasses auto-scaling).
#   PREDECODER_LR_MILESTONES  Comma-separated LR milestone fractions.
#   DOCKER_IMAGE        Pre-built Docker image name. (default: predecoder-train)
#   DOCKER_BASE_IMAGE   Fallback CUDA base image for install-from-scratch.
#                       (default: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04)
#   TORCH_CUDA          PyTorch CUDA wheel tag, e.g. cu121 or cu124.
#   INSTALL_DIR         Where to install Python/venv on bare-metal nodes.
#                       (default: $HOME/predecoder_env)
#   PREDECODER_DISABLE_SDR  Set to 1 to skip SDR (useful for cluster runs).
#   PREDECODER_TORCH_COMPILE  Set to 0 to disable torch.compile.
#
# ──────────────────────────────────────────────────────────────
# Example submissions for common scenarios:
#
#   # R=9, 1 GPU (Model 1)
#   sbatch code/scripts/sbatch_train.sh
#
#   # R=13, 1 GPU (Model 4)
#   EXPERIMENT_NAME=qec-decoder-depolarizing-r13-fp8 \
#   CONFIG_NAME=config_qec_decoder_r13_fp8 \
#     sbatch code/scripts/sbatch_train.sh
#
#   # R=13, 4 GPUs — override partition and resources on the command line:
#   EXPERIMENT_NAME=qec-decoder-depolarizing-r13-fp8 \
#   CONFIG_NAME=config_qec_decoder_r13_fp8 \
#   GPUS=4 FRESH_START=1 \
#     sbatch --partition=<4gpu-partition> --nodes=1 --gres=gpu:4 \
#            --cpus-per-task=80 --mem=240G \
#            code/scripts/sbatch_train.sh
#
#   # Resume a 1-GPU checkpoint on 4 GPUs (fixed LR schedule):
#   EXPERIMENT_NAME=qec-decoder-depolarizing-r13-fp8 \
#   CONFIG_NAME=config_qec_decoder_r13_fp8 \
#   GPUS=4 \
#   PREDECODER_TRAIN_SAMPLES=8388608 \
#   PREDECODER_LR_MILESTONES="1.0,2.0,4.0" \
#     sbatch --partition=<4gpu-partition> --nodes=1 --gres=gpu:4 \
#            --cpus-per-task=80 --mem=240G \
#            code/scripts/sbatch_train.sh
# ──────────────────────────────────────────────────────────────

# === SLURM directives — EDIT THESE for your cluster ===
#SBATCH --job-name=predecoder-train
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=predecoder_train_%j.out
#SBATCH --error=predecoder_train_%j.err

set -euo pipefail

log() { echo "[$(date -Iseconds)] $*"; }
export PREDECODER_VERBOSE=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR:-$HOME/predecoder_outputs}"
GPUS="${GPUS:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qec-decoder-depolarizing-r9-fp8}"
CONFIG_NAME="${CONFIG_NAME:-config_qec_decoder_r9_fp8}"
DOCKER_IMAGE="${DOCKER_IMAGE:-predecoder-train}"
DOCKER_BASE_IMAGE="${DOCKER_BASE_IMAGE:-nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04}"

export EXPERIMENT_NAME CONFIG_NAME

mkdir -p "$SHARED_OUTPUT_DIR"
cd "$REPO_ROOT"

# Prevent two jobs writing to the same experiment dir concurrently
if [ -n "${SLURM_JOB_ID:-}" ]; then
  LOCK_FILE="$SHARED_OUTPUT_DIR/.lock_${EXPERIMENT_NAME}"
  if [ -f "$LOCK_FILE" ]; then
    OTHER_JOB=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -n "$OTHER_JOB" ] && [ "$OTHER_JOB" != "$SLURM_JOB_ID" ]; then
      if command -v squeue >/dev/null 2>&1 && squeue -j "$OTHER_JOB" -h 2>/dev/null | grep -q .; then
        log "ERROR: Another job ($OTHER_JOB) is already running for $EXPERIMENT_NAME. Exiting."
        exit 1
      fi
    fi
  fi
  echo "$SLURM_JOB_ID" > "$LOCK_FILE"
fi

log "========== Pre-decoder training (${GPUS} GPU(s)) =========="
log "Node: $(hostname)"
log "REPO_ROOT=$REPO_ROOT"
log "SHARED_OUTPUT_DIR=$SHARED_OUTPUT_DIR"
log "EXPERIMENT_NAME=$EXPERIMENT_NAME"
log "CONFIG_NAME=$CONFIG_NAME"
log "GPUS=$GPUS"
log "FRESH_START=${FRESH_START:-0}"
nvidia-smi -L 2>/dev/null || log "(nvidia-smi not available on submit node)"
log "========== Checking for Docker =========="

HOST_UID=$(id -u)
HOST_GID=$(id -g)

# Cluster-run defaults: disable SDR (expensive) and torch.compile (can crash in some envs).
# Users can override either by setting the variable before calling sbatch.
PREDECODER_DISABLE_SDR="${PREDECODER_DISABLE_SDR:-1}"
PREDECODER_TORCH_COMPILE="${PREDECODER_TORCH_COMPILE:-0}"

COMMON_ENV=(
  -e SHARED_OUTPUT_DIR=/data
  -e PYTHONUNBUFFERED=1
  -e PREDECODER_VERBOSE=1
  -e "PREDECODER_TORCH_COMPILE=${PREDECODER_TORCH_COMPILE}"
  -e "PREDECODER_DISABLE_SDR=${PREDECODER_DISABLE_SDR}"
  -e "GPUS=${GPUS}"
  -e "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  -e "CONFIG_NAME=${CONFIG_NAME}"
  -e "FRESH_START=${FRESH_START:-0}"
)
# Forward optional overrides
[ -n "${PREDECODER_TRAIN_EPOCHS:-}" ]   && COMMON_ENV+=(-e "PREDECODER_TRAIN_EPOCHS=${PREDECODER_TRAIN_EPOCHS}")
[ -n "${PREDECODER_TRAIN_SAMPLES:-}" ]  && COMMON_ENV+=(-e "PREDECODER_TRAIN_SAMPLES=${PREDECODER_TRAIN_SAMPLES}")
[ -n "${PREDECODER_LR_MILESTONES:-}" ]  && COMMON_ENV+=(-e "PREDECODER_LR_MILESTONES=${PREDECODER_LR_MILESTONES}")
[ -n "${PREDECODER_TIMING_RUN:-}" ]     && COMMON_ENV+=(-e "PREDECODER_TIMING_RUN=${PREDECODER_TIMING_RUN}")
[ -n "${TORCH_CUDA:-}" ]               && COMMON_ENV+=(-e "TORCH_CUDA=${TORCH_CUDA}")

if command -v docker >/dev/null 2>&1 && docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  log "Using pre-built image $DOCKER_IMAGE as user ${HOST_UID}:${HOST_GID}."
  docker run --rm --gpus all \
    --user "${HOST_UID}:${HOST_GID}" \
    -v "${REPO_ROOT}:/app:ro" \
    -v "${SHARED_OUTPUT_DIR}:/data" \
    "${COMMON_ENV[@]}" \
    "$DOCKER_IMAGE" 2>&1
elif command -v docker >/dev/null 2>&1 && docker run --rm --gpus all "$DOCKER_BASE_IMAGE" nvidia-smi -L >/dev/null 2>&1; then
  log "Setting sticky bit on $SHARED_OUTPUT_DIR for NFS/Docker UID compatibility."
  chmod 1777 "$SHARED_OUTPUT_DIR" 2>/dev/null || true
  log "Using Docker base image $DOCKER_BASE_IMAGE (install + train)."
  docker run --rm --gpus all \
    -v "${REPO_ROOT}:/app:ro" \
    -v "${SHARED_OUTPUT_DIR}:/data" \
    -e INSTALL_DIR=/opt/predecoder_env \
    "${COMMON_ENV[@]}" \
    "$DOCKER_BASE_IMAGE" \
    bash -c 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv python3.11-dev curl git coreutils build-essential cmake >/dev/null 2>&1; exec stdbuf -oL -eL bash /app/code/scripts/cluster_container_install_and_train.sh 2>&1'
else
  log "No Docker. Running install + train on node."
  export SHARED_OUTPUT_DIR GPUS EXPERIMENT_NAME CONFIG_NAME
  export FRESH_START="${FRESH_START:-0}"
  export PREDECODER_DISABLE_SDR PREDECODER_TORCH_COMPILE
  INSTALL_DIR="${INSTALL_DIR:-$HOME/predecoder_env}"
  bash code/scripts/cluster_install_deps.sh
  export PREDECODER_PYTHON="${INSTALL_DIR}/venv/bin/python"
  [ -f "${INSTALL_DIR}/miniconda3/envs/predecoder/bin/python" ] && export PREDECODER_PYTHON="${INSTALL_DIR}/miniconda3/envs/predecoder/bin/python"
  bash code/scripts/cluster_train.sh
fi

log "Done. Checkpoints in $SHARED_OUTPUT_DIR/outputs/$EXPERIMENT_NAME/"
