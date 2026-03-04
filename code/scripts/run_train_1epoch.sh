#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Run training for 1 full epoch (8M samples). Verification / long-run test on GPU.
# Same as bisect/public one-epoch run: 8M train, 64k val, 1 epoch.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"

# Optional: free GPU from other processes (if free_gpu.sh exists)
[ -x "code/scripts/free_gpu.sh" ] && code/scripts/free_gpu.sh --kill 2>/dev/null || true

export PREDECODER_TIMING_RUN=1
export PREDECODER_TRAIN_SAMPLES=8388608
export PREDECODER_VAL_SAMPLES=65536
export PREDECODER_TRAIN_EPOCHS=1
export CODE_ROOT="$REPO_ROOT/code"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu_he_1epoch}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$REPO_ROOT/outputs}"
export LOG_BASE_DIR="${LOG_BASE_DIR:-$REPO_ROOT/logs}"

echo "Training 1 epoch: $PREDECODER_TRAIN_SAMPLES samples (experiment: $EXPERIMENT_NAME)"
exec python3 -u code/workflows/run.py --config-name config_public \
  workflow.task=train \
  +exp_tag="$EXPERIMENT_NAME" \
  ++load_checkpoint=False \
  "$@"
