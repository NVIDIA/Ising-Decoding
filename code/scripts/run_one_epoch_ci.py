#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Long-running test: run 1 epoch training + validation (5-10 min), assert LER <= threshold.
# Skipped (exit 0) if no GPU. Intended for a separate CI job with GPU and sufficient timeout.
"""Run one-epoch training and pass if validation LER is at or below threshold."""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
LER_THRESHOLD = 0.1
# QUICK=10: ~2M samples, ~1000 steps, ~5-10 min
QUICK = "10"


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[run_one_epoch_ci] No GPU available; skipping long-running test (exit 0).")
            return 0
    except Exception as e:
        print(f"[run_one_epoch_ci] Could not check GPU: {e}; skipping (exit 0).")
        return 0

    frames_dir = REPO_ROOT / "frames_data"
    if not frames_dir.is_dir():
        print(f"[run_one_epoch_ci] frames_data not found at {frames_dir}; skipping (exit 0).")
        return 0

    out_dir = REPO_ROOT / "outputs" / "ci_one_epoch"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(CODE_DIR),
        "EXPERIMENT_NAME": "ci_one_epoch",
        "QUICK": QUICK,
        "FRAMES_DIR": str(frames_dir),
    }

    cmd = [
        sys.executable,
        "-u",
        str(CODE_DIR / "workflows" / "run.py"),
        "--config-name=config_pre_decoder_memory_surface_model_1_d9",
        "workflow.task=train",
        f"exp_tag=ci_one_epoch",
        "train.epochs=1",
        "train.num_samples=2048000",
        "val.num_samples=65536",
        f"data.precomputed_frames_dir={frames_dir}",
        f"output={out_dir}",
        f"resume_dir={out_dir}/models",
        "load_checkpoint=False",
    ]

    # Stream logs to terminal when PREDECODER_STREAM_LOGS=1 (e.g. interactive runs)
    stream_logs = os.environ.get("PREDECODER_STREAM_LOGS", "0") == "1"
    print(f"[run_one_epoch_ci] Running 1 epoch (QUICK=10, ~5-10 min)...")
    if stream_logs:
        print(
            "[run_one_epoch_ci] Logs streaming (PREDECODER_STREAM_LOGS=1). LER check after completion."
        )
        lines = []
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout:
                print(line, end="")
                lines.append(line)
            proc.wait()
        except Exception as e:
            proc.kill()
            proc.wait()
            raise
        combined = "".join(lines)
        proc = type("Proc", (), {"returncode": proc.returncode})()
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = stdout + "\n" + stderr

    # Parse last "[LER Validation] Logical error rate: X.XXXXX"
    match = re.search(r"\[LER Validation\]\s+Logical error rate:\s+([\d.]+)", combined)
    if not match:
        print("[run_one_epoch_ci] Could not find LER in output.")
        print(combined[-8000:])  # last 8k chars
        return 1

    ler = float(match.group(1))
    print(f"[run_one_epoch_ci] Validation LER: {ler:.5f} (threshold: {LER_THRESHOLD})")

    if proc.returncode != 0:
        print(f"[run_one_epoch_ci] Training exited with {proc.returncode}.")
        print(combined[-8000:])
        return 1

    if ler > LER_THRESHOLD:
        print(f"[run_one_epoch_ci] FAIL: LER {ler} > {LER_THRESHOLD}.")
        return 1

    print("[run_one_epoch_ci] PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
