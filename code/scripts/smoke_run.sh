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

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-short}"
CONFIG_NAME="${CONFIG_NAME:-config_public}"

# Short training overrides (timing mode in train.py)
export PREDECODER_TIMING_RUN=1
export PREDECODER_TRAIN_SAMPLES="${PREDECODER_TRAIN_SAMPLES:-4096}"
export PREDECODER_VAL_SAMPLES="${PREDECODER_VAL_SAMPLES:-512}"
export PREDECODER_TEST_SAMPLES="${PREDECODER_TEST_SAMPLES:-512}"
export PREDECODER_TRAIN_EPOCHS="${PREDECODER_TRAIN_EPOCHS:-1}"
export PREDECODER_DISABLE_SDR="${PREDECODER_DISABLE_SDR:-1}"
export PREDECODER_LER_FINAL_ONLY="${PREDECODER_LER_FINAL_ONLY:-1}"

# Short inference overrides (handled inside evaluation/inference.py)
export PREDECODER_INFERENCE_NUM_SAMPLES="${PREDECODER_INFERENCE_NUM_SAMPLES:-32}"
export PREDECODER_INFERENCE_LATENCY_SAMPLES="${PREDECODER_INFERENCE_LATENCY_SAMPLES:-0}"
export PREDECODER_INFERENCE_MEAS_BASIS="${PREDECODER_INFERENCE_MEAS_BASIS:-both}"
export PREDECODER_INFERENCE_NUM_WORKERS="${PREDECODER_INFERENCE_NUM_WORKERS:-0}"

TRAIN_EXTRA_PARAMS="${TRAIN_EXTRA_PARAMS:-}"
INFER_EXTRA_PARAMS="${INFER_EXTRA_PARAMS:-}"

echo "=== Short training ==="
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CONFIG_NAME="${CONFIG_NAME}" \
WORKFLOW=train \
GPUS=1 \
EXTRA_PARAMS="${TRAIN_EXTRA_PARAMS}" \
bash "${REPO_ROOT}/code/scripts/local_run.sh"

echo "=== Short inference ==="
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CONFIG_NAME="${CONFIG_NAME}" \
WORKFLOW=inference \
GPUS=1 \
EXTRA_PARAMS="${INFER_EXTRA_PARAMS}" \
bash "${REPO_ROOT}/code/scripts/local_run.sh"
