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

# Install Python and training deps on a node (or inside container). Run from repo root.
# Usage: INSTALL_DIR=/opt/predecoder_env bash code/scripts/cluster_install_deps.sh

set -euo pipefail
INSTALL_DIR="${INSTALL_DIR:-$HOME/predecoder_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================="
echo "Pre-decoder cluster dependency install"
echo "=========================================="
echo "INSTALL_DIR=$INSTALL_DIR REPO_ROOT=$REPO_ROOT"

use_existing_python() {
  local py="$1"
  command -v "$py" >/dev/null 2>&1 || return 1
  local ver; ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || return 1
  local major="${ver%%.*}" minor="${ver#*.}"; minor="${minor%%.*}"
  [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && echo "$py" && return 0
  return 1
}

PYTHON_BIN=""
for c in python3.11 python3.12 python3; do
  if py=$(use_existing_python "$c"); then PYTHON_BIN="$py"; echo "Using existing: $PYTHON_BIN"; break; fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "No Python 3.11+ found. Installing Miniconda..."
  mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR"
  MINICONDA_DIR="${INSTALL_DIR}/miniconda3"
  if [ ! -d "$MINICONDA_DIR" ]; then
    ARCH=$(uname -m)
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    [ -n "$(command -v wget)" ] && wget -q "$MINICONDA_URL" -O miniconda.sh || curl -sL -o miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p "$MINICONDA_DIR" && rm -f miniconda.sh
  fi
  # shellcheck disable=SC1090
  set +u && source "${MINICONDA_DIR}/bin/activate" && set -u
  conda create -n predecoder "python=${PYTHON_VERSION}" -y 2>/dev/null || conda create -n predecoder "python=${PYTHON_VERSION}" -y
  conda activate predecoder
  PYTHON_BIN="${MINICONDA_DIR}/envs/predecoder/bin/python"
else
  mkdir -p "$INSTALL_DIR"
  VENV_DIR="${INSTALL_DIR}/venv"
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    PYTHON_BIN="${VENV_DIR}/bin/python"
  elif [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
  fi
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel -q
cd "$REPO_ROOT"

# Use PyTorch CUDA index so torch is CUDA-built (on aarch64, PyPI serves CPU-only).
# TORCH_CUDA e.g. cu121 or cu124; default cu121 to match nvidia/cuda:12.1 base image.
PYTORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "Installing requirements (torch from CUDA index: ${TORCH_CUDA})..."
"$PYTHON_BIN" -m pip install -r code/requirements_public_train.txt \
  --index-url "${PYTORCH_INDEX}" --extra-index-url https://pypi.org/simple

"$PYTHON_BIN" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('PyTorch CUDA:', torch.version.cuda)
print('Install OK')
"
echo "=========================================="
echo "Install complete. Use: export PREDECODER_PYTHON=$PYTHON_BIN"
echo "=========================================="
