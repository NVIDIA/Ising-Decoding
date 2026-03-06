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
