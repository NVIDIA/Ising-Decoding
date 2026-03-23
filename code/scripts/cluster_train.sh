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

# Generic cluster training entry point.
# Configures output dirs, exports env, and delegates to local_run.sh.
# Requires SHARED_OUTPUT_DIR (e.g. /data in container, or ~/predecoder_outputs on a node).
#
# Key env vars (all have defaults):
#   EXPERIMENT_NAME   - subdirectory under outputs/ (default: qec-decoder-depolarizing-r9-fp8)
#   CONFIG_NAME       - Hydra config in conf/ without .yaml (default: config_qec_decoder_r9_fp8)
#   WORKFLOW          - train | inference (default: train)
#   FRESH_START       - 1 to skip checkpoint resume (default: 0)
#   PREDECODER_PYTHON - explicit path to python binary (auto-detected if unset)
#   PREDECODER_TRAIN_EPOCHS - override epoch count
#   PREDECODER_TRAIN_SAMPLES - override samples per epoch
#   PREDECODER_LR_MILESTONES - comma-separated LR milestone fractions

set -euo pipefail
say() { echo "[$(date -Iseconds)] $*" >&2; }
log() { echo "[$(date -Iseconds)] $*"; }

say "cluster_train.sh: START"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
say "cluster_train.sh: REPO_ROOT=$REPO_ROOT"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qec-decoder-depolarizing-r9-fp8}"
CONFIG_NAME="${CONFIG_NAME:-config_qec_decoder_r9_fp8}"
WORKFLOW="${WORKFLOW:-train}"
EPOCHS="${PREDECODER_TRAIN_EPOCHS:-100}"

SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR:-}"
SHARED_LOG_DIR="${SHARED_LOG_DIR:-}"
if [ -z "$SHARED_OUTPUT_DIR" ]; then
  say "ERROR: SHARED_OUTPUT_DIR is not set."
  exit 1
fi
say "cluster_train.sh: SHARED_OUTPUT_DIR=$SHARED_OUTPUT_DIR"

BASE_OUTPUT_DIR="${SHARED_OUTPUT_DIR}/outputs"
say "cluster_train.sh: mkdir $BASE_OUTPUT_DIR ..."
mkdir -p "$BASE_OUTPUT_DIR"
say "cluster_train.sh: mkdir done"

if [ -n "$SHARED_LOG_DIR" ]; then
  LOG_BASE_DIR="$SHARED_LOG_DIR"
else
  LOG_BASE_DIR="${SHARED_OUTPUT_DIR}/logs"
fi
mkdir -p "$LOG_BASE_DIR"

mkdir -p "${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/models"
echo "Wrote marker: $(date -Iseconds)" > "${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}/.train_started"
log "Created ${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME} and .train_started marker"

export PREDECODER_BASE_OUTPUT_DIR="$BASE_OUTPUT_DIR"
export PREDECODER_LOG_BASE_DIR="$LOG_BASE_DIR"
export PREDECODER_TRAIN_EPOCHS="$EPOCHS"
export EXPERIMENT_NAME
export CONFIG_NAME
export WORKFLOW
[ -n "${PREDECODER_PYTHON:-}" ] && export PREDECODER_PYTHON
export CODE_ROOT="${REPO_ROOT}/code"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

FRESH_START="${FRESH_START:-0}"
export FRESH_START

log "========== Cluster train: $EXPERIMENT_NAME =========="
log "Config: $CONFIG_NAME"
log "Epochs: $EPOCHS"
log "Outputs: $BASE_OUTPUT_DIR"
log "Resume: $([ "$FRESH_START" = "1" ] && echo "no" || echo "yes")"
log "========== Calling local_run.sh =========="

cd "$REPO_ROOT"
bash code/scripts/local_run.sh
log "Done."
