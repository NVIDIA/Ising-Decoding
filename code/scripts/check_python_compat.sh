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

PYTHON_BIN="${PYTHON_BIN:-${1:-python3}}"
MODE="${MODE:-inference}"  # inference | train
SKIP_TESTS="${SKIP_TESTS:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
TORCH_CUDA="${TORCH_CUDA:-}"  # e.g., cu118, cu121, cu128
TORCH_WHL_INDEX="${TORCH_WHL_INDEX:-}"  # override full index URL
if [[ -n "${TORCH_CUDA}" ]]; then
  CUDA_MAJOR_VERSION=${TORCH_CUDA:2:2}  # e.g., cu121 -> 12
else
  CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION:-12}
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${REQUIRE_GPU}" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[SKIP] GPU required but nvidia-smi not available."
    exit 0
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[SKIP] GPU required but no GPU detected."
    exit 0
  fi
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

REQ_FILE="${REQ_FILE:-}"
EXTRA_PKGS="${EXTRA_PKGS:-}"
if [[ -z "${REQ_FILE}" ]]; then
  if [[ "${MODE}" == "train" ]]; then
    REQ_FILE="${ROOT_DIR}/code/requirements_public_train-cu${CUDA_MAJOR_VERSION}.txt"
  else
    REQ_FILE="${ROOT_DIR}/code/requirements_public_inference.txt"
  fi
fi

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_${MODE}_${PY_VER}}"

echo "=== Python compatibility check ==="
echo "Python: ${PYTHON_BIN} (${PY_VER})"
echo "Mode: ${MODE}"
echo "Requirements: ${REQ_FILE}"
echo "Venv: ${VENV_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_WHL_INDEX}" || -n "${TORCH_CUDA}" ]]; then
  if [[ -n "${TORCH_WHL_INDEX}" ]]; then
    TORCH_INDEX_URL="${TORCH_WHL_INDEX}"
  else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"
  fi
  export PIP_INDEX_URL="${TORCH_INDEX_URL}"
  if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL} https://pypi.org/simple"
  else
    export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
  fi
  echo "PyTorch index: ${PIP_INDEX_URL}"
fi

pip install -r "${REQ_FILE}"

# nvidia-modelopt[onnx] caps its Python requirement at <3.13, but works on 3.13
# in practice.  Re-install with --ignore-requires-python so train CI on 3.13+
# uses modelopt (INT8+FP8) rather than the onnxruntime-only fallback.
if [[ "${MODE}" == "train" ]]; then
  if python -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
    echo "Python 3.13+: installing nvidia-modelopt[onnx] --ignore-requires-python"
    pip install "nvidia-modelopt[onnx]" --ignore-requires-python
  fi
fi

if [[ -n "${EXTRA_PKGS}" ]]; then
  pip install ${EXTRA_PKGS}
fi

python - <<'PY'
import importlib
mods = [
    "hydra",
    "omegaconf",
    "numpy",
    "torch",
    "stim",
    "pymatching",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append((m, str(exc)))
if missing:
    raise SystemExit(f"Missing imports: {missing}")
print("Core imports OK")
PY

if [[ "${SKIP_TESTS}" != "1" ]]; then
  cd "${ROOT_DIR}"
  # Discover and run all test_*.py files; new test files are picked up automatically
  PYTHONPATH="${ROOT_DIR}/code" python -m unittest discover -s code/tests -p "test_*.py"
fi

echo "=== Done ==="
