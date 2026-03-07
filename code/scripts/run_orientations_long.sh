#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Long-running runs for all four surface code orientations (O1..O4).
# Not intended for pre-commit CI: use in scheduled pipelines or on dedicated servers.
# See code/tests/README_ORIENTATIONS_LONG_RUNNING.md for details.
#
# Usage:
#   ORIENTATIONS_LONG_TASK=train    bash code/scripts/run_orientations_long.sh
#   ORIENTATIONS_LONG_TASK=inference bash code/scripts/run_orientations_long.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

TASK="${ORIENTATIONS_LONG_TASK:-train}"
TASK="$(echo "${TASK}" | tr '[:upper:]' '[:lower:]')"
if [ "${TASK}" != "train" ] && [ "${TASK}" != "inference" ]; then
  echo "Usage: ORIENTATIONS_LONG_TASK=train|inference $0" >&2
  exit 1
fi

ORIENTATIONS=(O1 O2 O3 O4)
WORKFLOW="${TASK}"
CONFIG_NAME="${CONFIG_NAME:-config_public}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-orientations}"

echo "=========================================="
echo "Long-running orientation runs: ${TASK}"
echo "Orientations: ${ORIENTATIONS[*]}"
echo "Config: ${CONFIG_NAME}"
echo "=========================================="

for orient in "${ORIENTATIONS[@]}"; do
  echo ""
  echo "=== Running ${TASK} for orientation ${orient} ==="
  if [ "${orient}" = "O1" ]; then
    ORIENT_EXTRA="${EXTRA_PARAMS:-}"
  else
    ORIENT_EXTRA="data.code_rotation=${orient} ${EXTRA_PARAMS:-}"
  fi
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_${orient}" \
  CONFIG_NAME="${CONFIG_NAME}" \
  WORKFLOW="${WORKFLOW}" \
  EXTRA_PARAMS="${ORIENT_EXTRA}" \
  bash "${REPO_ROOT}/code/scripts/local_run.sh" || exit 1
done

echo ""
echo "=========================================="
echo "All orientation runs completed."
echo "=========================================="
