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

import os
import subprocess
import sys

import pytest


def _repo_root() -> str:
    # This file lives at code/tests/..., so repo root is 2 levels up.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.integration
def test_color_code_run_py_train_smoke(tmp_path):
    """
    Integration smoke test: run the Hydra entrypoint (run.py) end-to-end for a tiny
    color-code training config and ensure it exits successfully.

    This is meant to validate *plumbing* across:
      run.py -> training/train.py -> configured data generator -> model factory -> forward/loss/step.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch not importable: {e}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; training entrypoint uses CUDA autocast.")

    repo_root = _repo_root()
    out_dir = tmp_path / "predecoder_out"
    code_dir = os.path.join(repo_root, "code")

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    # Keep runs deterministic-ish and avoid grabbing all GPUs in CI.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Reduce log spam in test output.
    env.setdefault("TORCH_LOGS", "-all")
    # The repo's importable modules live under `code/` (not an installed package).
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        "code/workflows/run.py",
        "--config-name",
        "config_color_smoke",
        f"output={out_dir}",
    ]

    res = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )

    # Show the tail of logs to help debug failures.
    tail = res.stdout[-8000:]
    assert res.returncode == 0, tail

    # Basic sanity: the training code prints a config summary including `code: color`.
    assert "code: color" in res.stdout, tail
    assert "Setting up on-the-fly data generation" in res.stdout, tail
