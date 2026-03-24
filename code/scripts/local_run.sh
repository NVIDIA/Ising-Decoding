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

# Minimal local runner.
#
# Examples:
#   bash code/scripts/local_run.sh
#   WORKFLOW=inference bash code/scripts/local_run.sh
#   GPUS=4 bash code/scripts/local_run.sh
#   CUDA_VISIBLE_DEVICES=1 bash code/scripts/local_run.sh        # use only GPU 1
#
# Notes:
# - Public config is `conf/config_public.yaml`. Users should edit only that file.
# - Training knobs are auto-managed in code (epochs, shots/epoch, batch schedule, etc.).
# - SafeTensors (optional): after training, convert the best .pt checkpoint with
#     code/export/checkpoint_to_safetensors.py (see README), then pass the result as:
#     PREDECODER_SAFETENSORS_CHECKPOINT=<path>.safetensors WORKFLOW=inference bash code/scripts/local_run.sh

EXPERIMENT_NAME="${EXPERIMENT_NAME:-test1}"
CONFIG_NAME="${CONFIG_NAME:-config_public}"   # conf/<name>.yaml (no extension)
WORKFLOW="${WORKFLOW:-train}"                 # train | inference
WORKFLOW="$(echo "${WORKFLOW}" | tr '[:upper:]' '[:lower:]')"
GPUS="${GPUS:-}"                              # if empty, auto-detect
FRESH_START="${FRESH_START:-0}"               # 1 => don't load checkpoint
EXTRA_PARAMS="${EXTRA_PARAMS:-}"              # advanced hydra overrides (discouraged)
TORCH_COMPILE="${TORCH_COMPILE:-}"            # 0/1 to disable/enable torch.compile
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-}"  # optional: default | reduce-overhead | max-autotune

DISTANCE="${DISTANCE:-}"
N_ROUNDS="${N_ROUNDS:-}"
if [ $# -eq 1 ]; then DISTANCE="$1"; fi
if [ $# -eq 2 ]; then DISTANCE="$1"; N_ROUNDS="$2"; fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# local_run.sh lives at: <repo_root>/code/scripts/local_run.sh
# so repo_root is two levels up from SCRIPT_DIR.
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${REPO_ROOT}/code}"

# Default output locations live inside the repo (avoid surprises from generic env vars).
# Some environments set BASE_OUTPUT_DIR/LOG_BASE_DIR globally; ignore those by default to
# prevent creating confusing extra folders like /root/outputs or /root/logs.
if [ -n "${BASE_OUTPUT_DIR:-}" ] || [ -n "${LOG_BASE_DIR:-}" ]; then
  echo "[local_run.sh] Note: ignoring BASE_OUTPUT_DIR/LOG_BASE_DIR from the environment."
  echo "[local_run.sh] To override paths, use PREDECODER_BASE_OUTPUT_DIR / PREDECODER_LOG_BASE_DIR."
fi
BASE_OUTPUT_DIR="${PREDECODER_BASE_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
LOG_BASE_DIR="${PREDECODER_LOG_BASE_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_BASE_DIR}"

if [ "${FRESH_START}" -eq 1 ]; then
  RESUME_FLAG="++load_checkpoint=False"
else
  RESUME_FLAG="++load_checkpoint=True"
fi

# GPU-only runs: require a visible GPU and nvidia-smi.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[local_run.sh] Error: GPU-only mode requires nvidia-smi on PATH." >&2
  echo "[local_run.sh] Hint: run on a GPU host or pass CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

# Respect CUDA_VISIBLE_DEVICES if set; otherwise auto-detect via nvidia-smi.
if [ -z "${GPUS}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPUS="$(python3 - <<'PY'
import os
v=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
print(len([x for x in v.split(',') if x.strip()]) or 1)
PY
)"
  else
    GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  fi
fi

if [ "${GPUS}" -le 0 ]; then
  echo "[local_run.sh] Error: no GPUs detected. GPU-only mode requires CUDA." >&2
  exit 1
fi

if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT="$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)"
  export MASTER_PORT
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# Add nanoseconds to avoid collisions when launching multiple runs within the same second.
TIMESTAMP_NS="$(date +%Y%m%d_%H%M%S_%N)"
RUN_ID="${EXPERIMENT_NAME}_${TIMESTAMP}"
LOG_DIR="${LOG_BASE_DIR}/${RUN_ID}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${OUTPUT_DIR}/models"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

# Force Hydra run dir to writable OUTPUT_DIR (avoids read-only repo/outputs in containers)
OVERRIDES="hydra.run.dir=${OUTPUT_DIR}"
if [ -n "${DISTANCE}" ]; then OVERRIDES+=" distance=${DISTANCE}"; fi
if [ -n "${N_ROUNDS}" ]; then OVERRIDES+=" n_rounds=${N_ROUNDS}"; fi
if [ -n "${EXTRA_PARAMS}" ]; then OVERRIDES+=" ${EXTRA_PARAMS}"; fi

CONFIG_SNAPSHOT_DIR="${OUTPUT_DIR}/config"
mkdir -p "${CONFIG_SNAPSHOT_DIR}"
CONFIG_PATH="${REPO_ROOT}/conf/${CONFIG_NAME}.yaml"
if [ -f "${CONFIG_PATH}" ]; then
  # Never overwrite existing snapshots: keep full history.
  base_yaml="${CONFIG_SNAPSHOT_DIR}/${CONFIG_NAME}_${TIMESTAMP_NS}.yaml"
  dest_yaml="${base_yaml}"
  i=0
  while [ -e "${dest_yaml}" ]; do
    i=$((i+1))
    dest_yaml="${base_yaml%.yaml}_${i}.yaml"
  done
  cp "${CONFIG_PATH}" "${dest_yaml}"
  # Also save the exact CLI overrides used for this run (useful when configs change over time).
  base_ovr="${CONFIG_SNAPSHOT_DIR}/${CONFIG_NAME}_${TIMESTAMP_NS}.overrides.txt"
  dest_ovr="${base_ovr}"
  j=0
  while [ -e "${dest_ovr}" ]; do
    j=$((j+1))
    dest_ovr="${base_ovr%.txt}_${j}.txt"
  done
  {
    echo "workflow.task=${WORKFLOW}"
    echo "exp_tag=${EXPERIMENT_NAME}"
    echo "${RESUME_FLAG}"
    echo "${OVERRIDES:-}"
  } > "${dest_ovr}"
else
  echo "[local_run.sh] Warning: could not find config file to snapshot: ${CONFIG_PATH}"
fi

echo "=========================================="
echo "Local run"
echo "=========================================="
echo "workflow.task: ${WORKFLOW}"
echo "config: ${CONFIG_NAME}"
echo "GPUS: ${GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
echo "output: ${OUTPUT_DIR}"
echo "logs: ${LOG_DIR}"
echo "overrides: ${OVERRIDES:-<none>}"
echo "=========================================="

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS="$(nproc)"
export JOB_START_TIMESTAMP="$(date +%s)"
export JOB_START_DATETIME="$(date)"
if [ -n "${TORCH_COMPILE}" ]; then
  export PREDECODER_TORCH_COMPILE="${TORCH_COMPILE}"
fi
if [ -n "${TORCH_COMPILE_MODE}" ]; then
  export PREDECODER_TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE}"
fi

# Prefer PREDECODER_PYTHON (cluster/container venv) when set
PYTHON_BIN="${PYTHON_BIN:-${PREDECODER_PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[local_run.sh] Error: no python interpreter found on PATH." >&2
    exit 1
  fi
fi

# Ensure CUDA is usable before launching the workflow.
if ! "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"[local_run.sh] Error: PyTorch is required for GPU-only runs ({exc}).", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("[local_run.sh] Error: torch.cuda.is_available() is false. GPU-only mode requires CUDA.", file=sys.stderr)
    sys.exit(1)
PY
then
  exit 1
fi

# Run from repo root so config defaults like `output: outputs/${exp_tag}` land in <repo_root>/outputs.
cd "${REPO_ROOT}"

LOG_FILE="${LOG_DIR}/${WORKFLOW}.log"

if [ "${GPUS}" -gt 1 ]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_port="${MASTER_PORT}" \
    code/workflows/run.py \
    --config-name="${CONFIG_NAME}" \
    workflow.task="${WORKFLOW}" \
    +exp_tag="${EXPERIMENT_NAME}" \
    ${RESUME_FLAG} \
    ${OVERRIDES} \
    2>&1 | tee -a "${LOG_FILE}"
else
  "${PYTHON_BIN}" -u code/workflows/run.py \
    --config-name="${CONFIG_NAME}" \
    workflow.task="${WORKFLOW}" \
    +exp_tag="${EXPERIMENT_NAME}" \
    ${RESUME_FLAG} \
    ${OVERRIDES} \
    2>&1 | tee -a "${LOG_FILE}"
fi

cp -f "${LOG_FILE}" "${OUTPUT_DIR}/run.log"
echo "Done. Log: ${LOG_FILE}"
