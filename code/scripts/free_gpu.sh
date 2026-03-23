#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# List or kill processes using the GPU so training can utilize it.
# Usage:
#   ./code/scripts/free_gpu.sh           # list PIDs and memory per process
#   ./code/scripts/free_gpu.sh --kill    # kill all processes using the GPU (except this script)
#   ./code/scripts/free_gpu.sh --kill 0  # kill only processes on GPU 0
set -euo pipefail

KILL_MODE=false
GPU_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill)
      KILL_MODE=true
      shift
      if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
        GPU_ID="$1"
        shift
      fi
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--kill [gpu_id]]" >&2
      exit 1
      ;;
  esac
done

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found. Install NVIDIA drivers." >&2
  exit 1
fi

# Get PIDs using the GPU: nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
# We need GPU index when filtering by GPU. Use nvidia-smi -L to get index; --query-compute-apps does not give index directly.
# So we list all compute PIDs, then optionally filter by GPU index using memory query per GPU.
if [[ -n "$GPU_ID" ]]; then
  # Get PIDs on this GPU only (by querying processes on GPU $GPU_ID)
  PIDS=$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
else
  PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
fi

MY_PID=$$
PIDS=$(echo "$PIDS" | tr -d ' ' | grep -v "^$" | grep -v "^${MY_PID}$" || true)

if [[ -z "$PIDS" ]]; then
  echo "No other processes are using the GPU."
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null || true
  exit 0
fi

echo "Processes using the GPU:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true

if [[ "$KILL_MODE" == true ]]; then
  for pid in $PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Killing PID $pid ..."
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "Done. Wait a few seconds then run nvidia-smi to confirm GPU is free."
else
  echo ""
  echo "To free the GPU, run: $0 --kill"
  if [[ -n "$GPU_ID" ]]; then
    echo "To kill only processes on GPU $GPU_ID: $0 --kill $GPU_ID"
  fi
fi
