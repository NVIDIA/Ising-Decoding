# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch decisions of ``code/scripts/local_run.sh`` under a Slurm-like environment.

The script is run with a stub ``python`` (records its argv, exits 0) and a stub
``nvidia-smi`` on PATH, so the test needs neither a GPU nor torch. Checks:

  * ``generate_stim_data`` is a serial export: no ``torch.distributed.run``
    launch even when ``SLURM_NNODES`` says the allocation has several nodes;
  * a single-node multi-GPU launch always passes ``--node_rank=0``, also when
    ``SLURM_NODEID`` is non-zero on the node that runs it;
  * a genuine multi-node launch still forwards the Slurm node rank.
"""
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUN = REPO_ROOT / "code" / "scripts" / "local_run.sh"


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TestLocalRunLaunch(unittest.TestCase):

    def _run(self, env_overrides: dict) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            argv_log = tmp / "argv.log"
            _write_exe(
                bin_dir / "python",
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$@\" >> '{argv_log}'\n"
                "cat >/dev/null\n"  # swallow heredoc scripts
                "exit 0\n"
            )
            _write_exe(bin_dir / "nvidia-smi", "#!/usr/bin/env bash\necho GPU0\nexit 0\n")
            env = {
                k: v for k, v in os.environ.items() if
                not k.startswith(("SLURM_", "NNODES", "NODE_RANK", "MASTER_", "GPUS", "WORKFLOW"))
            }
            env.update(
                {
                    "PATH": f"{bin_dir}:{env.get('PATH', '')}",
                    "PYTHON_BIN": str(bin_dir / "python"),
                    "PREDECODER_BASE_OUTPUT_DIR": str(tmp / "out"),
                    "PREDECODER_LOG_BASE_DIR": str(tmp / "logs"),
                    "CONFIG_NAME": "config_public",
                    "EXPERIMENT_NAME": "launch_test",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29999",
                }
            )
            env.update({k: str(v) for k, v in env_overrides.items()})
            proc = subprocess.run(
                ["bash", str(LOCAL_RUN)],
                env=env,
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            return argv_log.read_text() if argv_log.exists() else ""

    def test_generate_stim_data_never_uses_the_distributed_launcher(self):
        argv = self._run({
            "WORKFLOW": "generate_stim_data",
            "SLURM_NNODES": 4,
            "SLURM_NODEID": 2,
        })
        self.assertIn("code/workflows/run.py", argv)
        self.assertNotIn("torch.distributed.run", argv)
        self.assertNotIn("--node_rank", argv)

    def test_single_node_multi_gpu_in_a_slurm_allocation_is_node_rank_zero(self):
        argv = self._run({
            "WORKFLOW": "inference",
            "GPUS": 2,
            "NNODES": 1,
            "SLURM_NODEID": 3,
        })
        self.assertIn("torch.distributed.run", argv)
        self.assertIn("--nnodes=1", argv)
        self.assertIn("--node_rank=0", argv)
        self.assertNotIn("--node_rank=3", argv)

    def test_multi_node_launch_forwards_the_slurm_node_rank(self):
        argv = self._run({
            "WORKFLOW": "train",
            "GPUS": 4,
            "SLURM_NNODES": 2,
            "SLURM_NODEID": 1,
        })
        self.assertIn("--nnodes=2", argv)
        self.assertIn("--node_rank=1", argv)


if __name__ == "__main__":
    unittest.main()
