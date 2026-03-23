#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run tests by tier: short (pre-merge), mid (~5-10 min), long (30 min+).
# See code/tests/README_TEST_TIERS.md for purposes and when to run each tier.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TIER="${1:-short}"
TIER="$(echo "${TIER}" | tr '[:upper:]' '[:lower:]')"

export PYTHONPATH="${REPO_ROOT}/code:${PYTHONPATH:-}"

case "${TIER}" in
  short)
    echo "=========================================="
    echo "Tier: SHORT (pre-merge required)"
    echo "=========================================="
    python3 -m unittest discover -s "${REPO_ROOT}/code/tests" -p "test_*.py"
    ;;
  mid)
    echo "=========================================="
    echo "Tier: MID (~5-10 min, pre-merge GPU)"
    echo "=========================================="
    if [ -n "${RUN_TIER_MID_CMD:-}" ]; then
      eval "${RUN_TIER_MID_CMD}"
    else
      export PREDECODER_TRAIN_SAMPLES="${PREDECODER_TRAIN_SAMPLES:-32768}"
      export PREDECODER_VAL_SAMPLES="${PREDECODER_VAL_SAMPLES:-4096}"
      export PREDECODER_TEST_SAMPLES="${PREDECODER_TEST_SAMPLES:-4096}"
      export PREDECODER_TRAIN_EPOCHS="${PREDECODER_TRAIN_EPOCHS:-2}"
      export EXPERIMENT_NAME="${EXPERIMENT_NAME:-tier_mid}"
      bash "${REPO_ROOT}/code/scripts/smoke_run.sh"
    fi
    echo "------------------------------------------"
    echo "Mid-tier HE compile tests"
    echo "------------------------------------------"
    python3 -m unittest discover -s "${REPO_ROOT}/code/tests/mid" -p "test_*.py" -v
    ;;
  long)
    echo "=========================================="
    echo "Tier: LONG (30 min+, scheduled / on-demand)"
    echo "=========================================="
    TASK="${ORIENTATIONS_LONG_TASK:-train}"
    ORIENTATIONS_LONG_TASK="${TASK}" bash "${REPO_ROOT}/code/scripts/run_orientations_long.sh"
    ;;
  *)
    echo "Usage: $0 short|mid|long" >&2
    echo "  short  - Unit/integration tests (pre-merge, required)" >&2
    echo "  mid    - Extended train+inference with LER check (~5-10 min, GPU)" >&2
    echo "  long   - Full orientation matrix (30 min+, on-demand)" >&2
    exit 1
    ;;
esac

echo "Tier '${TIER}' completed."
