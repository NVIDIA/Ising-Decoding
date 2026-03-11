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

# Run inside container: install deps (if needed) then train.
# Mount repo at /app (read-only), persistent output at /data.
#
# With the pre-built image (docker build -t predecoder-train .):
#   docker run --rm --gpus all -v $(pwd):/app:ro -v $OUTPUT:/data \
#     -e SHARED_OUTPUT_DIR=/data predecoder-train
#
# With a bare CUDA base image (deps installed at runtime):
#   docker run --rm --gpus all -v $(pwd):/app:ro -v $OUTPUT:/data \
#     -e SHARED_OUTPUT_DIR=/data -e INSTALL_DIR=/opt/predecoder_env \
#     nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
#     bash /app/code/scripts/cluster_container_install_and_train.sh
#
# Env: SHARED_OUTPUT_DIR=/data, INSTALL_DIR=/opt/predecoder_env
#      SKIP_INSTALL=1 to force-skip install, PREDECODER_PYTHON to set python path

set -euo pipefail
log() { echo "[$(date -Iseconds)] $*"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-/opt/predecoder_env}"
SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR:-/data}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

log "========== Container: install + train =========="
log "REPO_ROOT=$REPO_ROOT INSTALL_DIR=$INSTALL_DIR SHARED_OUTPUT_DIR=$SHARED_OUTPUT_DIR"

# Auto-detect: skip install when PREDECODER_PYTHON is already set and working
# (i.e. deps are baked into the image via Dockerfile).
if [ "$SKIP_INSTALL" != "1" ] && [ -n "${PREDECODER_PYTHON:-}" ]; then
  if "$PREDECODER_PYTHON" -c "import torch" 2>/dev/null; then
    log "Step 1: Deps already installed (PREDECODER_PYTHON=$PREDECODER_PYTHON). Skipping install."
    SKIP_INSTALL=1
  fi
fi

if [ "$SKIP_INSTALL" != "1" ]; then
  log "Step 1: Installing dependencies..."
  INSTALL_DIR="$INSTALL_DIR" SHARED_OUTPUT_DIR="$SHARED_OUTPUT_DIR" bash "$REPO_ROOT/code/scripts/cluster_install_deps.sh"
  log "Step 1: Done."
else
  log "Step 1: Skipping install."
fi

log "Step 2: Running training..."
export SHARED_OUTPUT_DIR INSTALL_DIR PYTHONUNBUFFERED=1
if [ -z "${PREDECODER_PYTHON:-}" ]; then
  [ -x "${INSTALL_DIR}/venv/bin/python" ] && export PREDECODER_PYTHON="${INSTALL_DIR}/venv/bin/python"
  [ -x "${INSTALL_DIR}/miniconda3/envs/predecoder/bin/python" ] && export PREDECODER_PYTHON="${INSTALL_DIR}/miniconda3/envs/predecoder/bin/python"
fi
log "PREDECODER_PYTHON=${PREDECODER_PYTHON:-<unset>}"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL bash "$REPO_ROOT/code/scripts/cluster_train.sh"
else
  bash "$REPO_ROOT/code/scripts/cluster_train.sh"
fi
log "Done."
