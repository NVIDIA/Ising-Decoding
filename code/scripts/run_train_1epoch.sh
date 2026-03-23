#!/usr/bin/env bash
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
