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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-offline_stim_run}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${REPO_ROOT}/models/Ising-Decoder-SurfaceCode-1-Fast.pt}"
PYTHON_BIN="${PREDECODER_PYTHON:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${REPO_ROOT}/.venv_gpu/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv_gpu/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

SAMPLES_DIR="${REPO_ROOT}/outputs/${EXPERIMENT_NAME}/stim_samples"
LOG_PATH="${REPO_ROOT}/outputs/${EXPERIMENT_NAME}/run.log"

echo "=========================================="
echo "Offline Stim smoke test"
echo "=========================================="
echo "experiment: ${EXPERIMENT_NAME}"
echo "python: ${PYTHON_BIN}"
echo "samples: ${SAMPLES_DIR}"
echo "model: ${MODEL_CHECKPOINT}"
echo "=========================================="

cd "${REPO_ROOT}"

PREDECODER_PYTHON="${PYTHON_BIN}" \
WORKFLOW=generate_stim_data \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
bash code/scripts/local_run.sh

PREDECODER_PYTHON="${PYTHON_BIN}" \
PREDECODER_STIM_SAMPLES_DIR="${SAMPLES_DIR}" \
PREDECODER_DECODE_MODE=pymatching_only \
PREDECODER_EMIT_INFERENCE_SUMMARY=1 \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
GPUS=1 \
WORKFLOW=inference \
bash code/scripts/local_run.sh

if [ -f "${MODEL_CHECKPOINT}" ]; then
  PREDECODER_PYTHON="${PYTHON_BIN}" \
  PREDECODER_STIM_SAMPLES_DIR="${SAMPLES_DIR}" \
  PREDECODER_DECODE_MODE=ising_decoding_pymatching \
  PREDECODER_EMIT_INFERENCE_SUMMARY=1 \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  GPUS=1 \
  WORKFLOW=inference \
  EXTRA_PARAMS="++model_checkpoint_file=${MODEL_CHECKPOINT}${MODEL_ID:+ ++model_id=${MODEL_ID}}" \
  bash code/scripts/local_run.sh
else
  echo "[offline_smoketest.sh] Model not found; skipped ising_decoding_pymatching:"
  echo "  ${MODEL_CHECKPOINT}"
fi

if [ -f "${LOG_PATH}" ]; then
  "${PYTHON_BIN}" - "${LOG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

# Inference prints a single-line JSON marker:
#     [Inference Summary] {"marker": "inference_summary", ...}
# Parse the LAST such marker (so we pick up the most recent run when the log
# file accumulates multiple inference passes, e.g. pymatching_only followed by
# ising_decoding_pymatching).
text = Path(sys.argv[1]).read_text(encoding="utf-8")
marker_prefix = "[Inference Summary] "
records = []
for line in text.splitlines():
    idx = line.find(marker_prefix)
    if idx < 0:
        continue
    payload = line[idx + len(marker_prefix):].strip()
    if not payload:
        continue
    try:
        record = json.loads(payload)
    except json.JSONDecodeError:
        continue
    if record.get("marker") == "inference_summary":
        records.append(record)

if not records:
    raise SystemExit(f"No [Inference Summary] JSON marker found in {sys.argv[1]}")

summary = records[-1]
ler = summary.get("ler", {})
speedup = summary.get("pymatching_speedup_avg_xz", float("nan"))

# Full per-basis latency/LER/speedup table is already printed by
# code/evaluation/inference.py; just emit one headline line here.
print(
    f"\n[offline_smoketest.sh] Avg LER {ler.get('avg_no_predecoder')} "
    f"(no pre-decoder) -> {ler.get('avg_after_predecoder')} (after); "
    f"PyMatching speedup {speedup}"
)
PY
fi
