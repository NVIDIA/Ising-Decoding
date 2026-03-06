#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Parse training log for "[LER Validation] Logical error rate: X.XXXXX" and exit 1 if above threshold.
# Used by CI to turn "run and hope it completes" into a real correctness check.
#
# Usage:
#   python check_ler_from_log.py <log_file_or_-_for_stdin> [--max-ler 0.1]

import argparse
import re
import sys
from pathlib import Path

# Same pattern as run_one_epoch_ci.py
LER_PATTERN = re.compile(r"\[LER Validation\]\s+Logical error rate:\s+([\d.]+)")


def main():
    parser = argparse.ArgumentParser(
        description="Check that the last LER in a training log is at or below a threshold."
    )
    parser.add_argument(
        "log",
        type=str,
        help="Path to log file, or '-' for stdin",
    )
    parser.add_argument(
        "--max-ler",
        type=float,
        default=0.1,
        help="Maximum allowed LER (default: 0.1)",
    )
    args = parser.parse_args()

    if args.log == "-":
        content = sys.stdin.read()
    else:
        path = Path(args.log)
        if not path.exists():
            print(f"[check_ler_from_log] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        content = path.read_text()

    matches = LER_PATTERN.findall(content)
    if not matches:
        print(
            "[check_ler_from_log] No '[LER Validation] Logical error rate: X.XXXXX' line found.",
            file=sys.stderr
        )
        sys.exit(1)

    ler = float(matches[-1])
    print(f"[check_ler_from_log] Last LER: {ler:.6f} (max allowed: {args.max_ler})")

    if ler > args.max_ler:
        print(f"[check_ler_from_log] FAIL: LER {ler} > {args.max_ler}", file=sys.stderr)
        sys.exit(1)

    print("[check_ler_from_log] PASS.")
    sys.exit(0)


if __name__ == "__main__":
    main()
