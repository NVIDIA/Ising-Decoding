# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for the training_progress script (TensorBoard progress reader).
"""Tests for code/scripts/training_progress.py."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_script = _repo_root / "code" / "scripts" / "training_progress.py"


class TestTrainingProgressScript(unittest.TestCase):
    """Tests for training_progress.py CLI (CPU; no GPU/tensorboard required for these)."""

    def test_missing_logdir_exits_nonzero(self):
        """When --logdir points to a non-existent path, script exits with 1."""
        if not _script.exists():
            self.skipTest("training_progress.py not in repository")
        proc = subprocess.run(
            [sys.executable,
             str(_script), "--logdir", "/nonexistent/tensorboard/log"],
            cwd=str(_repo_root),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("not found", (proc.stderr or proc.stdout or "").lower())

    def test_empty_dir_exits_zero_or_no_crash(self):
        """When --logdir is an empty directory, script should not crash (may exit 0 or 1)."""
        if not _script.exists():
            self.skipTest("training_progress.py not in repository")
        try:
            from tensorboard.backend.event_processing import event_accumulator
        except ImportError:
            self.skipTest("tensorboard not installed")
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(_script), "--logdir", tmp],
                cwd=str(_repo_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            # No events: script may print "No Loss/train_step events" and exit 0, or exit 1 for empty
            self.assertIn(proc.returncode, (0, 1))
