#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run 1-epoch d9 training with live logs and GPU. Logs go to terminal and to a file.
# Usage: ./code/scripts/run_1epoch_with_logs.sh [log_file]
# Example: ./code/scripts/run_1epoch_with_logs.sh
#          ./code/scripts/run_1epoch_with_logs.sh logs/run_$(date +%Y%m%d_%H%M%S).log
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"

LOG_FILE="${1:-}"
FRAMES_DIR="${FRAMES_DIR:-$REPO_ROOT/frames_data}"
OUT_DIR="$REPO_ROOT/outputs/one_epoch_$(date +%Y%m%d_%H%M%S)"
export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Force 2M train samples so Hydra override is respected (train.py applies when PREDECODER_TIMING_RUN=1)
export PREDECODER_TIMING_RUN=1
export PREDECODER_TRAIN_SAMPLES=2048000
export PREDECODER_VAL_SAMPLES=65536

if [ ! -d "$FRAMES_DIR" ]; then
  echo "ERROR: frames_data not found at $FRAMES_DIR"
  echo "Set FRAMES_DIR or create $FRAMES_DIR with precomputed d9 DEM artifacts."
  exit 1
fi

echo "=============================================="
echo "1-epoch d9 training (streaming logs)"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  FRAMES_DIR=$FRAMES_DIR"
echo "  OUTPUT=$OUT_DIR"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  LOG_FILE=${LOG_FILE:-<stdout only>}"
echo "=============================================="

CMD=(
  python3 -u code/workflows/run.py
  --config-name=config_pre_decoder_memory_surface_model_1_d9
  workflow.task=train
  exp_tag=one_epoch
  train.epochs=1
  train.num_samples=2048000
  val.num_samples=65536
  "data.precomputed_frames_dir=$FRAMES_DIR"
  "output=$OUT_DIR"
  "resume_dir=$OUT_DIR/models"
  load_checkpoint=False
)

if [ -n "$LOG_FILE" ]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  echo "Logging to $LOG_FILE"
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  "${CMD[@]}"
fi
