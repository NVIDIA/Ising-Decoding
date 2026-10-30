#!/usr/bin/env python3
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
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model
from evaluation.logical_error_rate import count_logical_errors_with_errorbar
from training.distributed import DistributedManager
from model.factory import ModelFactory
from export.safetensors_utils import _build_minimal_cfg

# Tolerance for "LER after <= baseline + tolerance".
# At d=13 with 262k shots and LER ~2e-4, the standard error per basis is
# ~sqrt(p/N) ≈ 2.8e-5; comparing two independent estimates (before/after)
# gives combined SE ~4e-5, so 2e-4 is a ~5-sigma guard against flakes.
LER_IMPROVEMENT_TOLERANCE = 2e-4

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# Each model is accepted as a .pt checkpoint or as the fp16 SafeTensors file
# distributed via Hugging Face. The published file name carries the release
# version, so it is matched by pattern and the newest version wins.
MODEL_R9 = {
    "filename": "Ising-Decoder-SurfaceCode-1-Fast.pt",
    "safetensors_glob": "ising_decoder_surface_code_1_fast_r9_*_fp16.safetensors",
    "hf_repo": "nvidia/Ising-Decoder-SurfaceCode-1-Fast",
    "model_id": 1,
}
MODEL_R13 = {
    "filename": "Ising-Decoder-SurfaceCode-1-Accurate.pt",
    "safetensors_glob": "ising_decoder_surface_code_1_accurate_r13_*_fp16.safetensors",
    "hf_repo": "nvidia/Ising-Decoder-SurfaceCode-1-Accurate",
    "model_id": 4,
}

REQUIRED_MODELS = [MODEL_R9, MODEL_R13]


def _version_key(path: Path):
    """Sort key for published file names: the version after "_v" (v1.0.9 < v1.0.77)."""
    m = re.search(r"_v(\d+(?:\.\d+)*)_", path.name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def _resolve_model_file(model_info: dict, models_dir: Path = MODELS_DIR):
    """Return the .pt checkpoint if present, else the newest published SafeTensors file, else None."""
    pt_path = models_dir / model_info["filename"]
    if pt_path.exists():
        return pt_path
    published = sorted(models_dir.glob(model_info["safetensors_glob"]), key=_version_key)
    return published[-1] if published else None


def _missing_model_msg(model_info: dict, models_dir: Path = MODELS_DIR) -> str:
    return (
        f"Model file missing: expected {models_dir / model_info['safetensors_glob']} "
        f"(download it from https://huggingface.co/{model_info['hf_repo']}) or "
        f"{models_dir / model_info['filename']}."
    )


def _load_public_model(merged, model_file: Path, dist):
    """Load model_file the way the public inference workflow does.

    SafeTensors files carry their model_id and precision in their metadata, so
    they go through PREDECODER_SAFETENSORS_CHECKPOINT; .pt checkpoints go
    through model_checkpoint_file.
    """
    use_safetensors = model_file.suffix == ".safetensors"
    if not use_safetensors:
        merged.model_checkpoint_file = str(model_file)
    old = os.environ.get("PREDECODER_SAFETENSORS_CHECKPOINT")
    if use_safetensors:
        os.environ["PREDECODER_SAFETENSORS_CHECKPOINT"] = str(model_file)
    else:
        os.environ.pop("PREDECODER_SAFETENSORS_CHECKPOINT", None)
    try:
        return _load_model(merged, dist)
    finally:
        if old is None:
            os.environ.pop("PREDECODER_SAFETENSORS_CHECKPOINT", None)
        else:
            os.environ["PREDECODER_SAFETENSORS_CHECKPOINT"] = old


def _run_inference_rtest(distance: int, n_rounds: int, model_info: dict):
    """Load public config and a specific model, run inference; returns result dict."""
    cfg_path = REPO_ROOT / "conf" / "config_public.yaml"
    cfg = OmegaConf.load(str(cfg_path))

    cfg.model_id = model_info["model_id"]
    cfg.distance = distance
    cfg.n_rounds = n_rounds
    cfg.workflow.task = "inference"

    model_spec = validate_public_config(cfg)
    merged = apply_public_defaults_and_model(cfg, model_spec)

    model_file = _resolve_model_file(model_info)
    if model_file is None:
        raise FileNotFoundError(_missing_model_msg(model_info))
    model_file = model_file.resolve()

    merged.test.latency_num_samples = 0
    merged.test.verbose_inference = False
    if "dataloader" in merged.test:
        merged.test.dataloader.num_workers = 0

    env_overrides = {
        "PREDECODER_INFERENCE_NUM_SAMPLES": None,
        "PREDECODER_INFERENCE_LATENCY_SAMPLES": None,
        "PREDECODER_INFERENCE_MEAS_BASIS": None,
        "PREDECODER_INFERENCE_NUM_WORKERS": "0",
    }
    old_env = {k: os.environ.get(k) for k in env_overrides}
    for key, value in env_overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_public_model(merged, model_file, dist)
        return count_logical_errors_with_errorbar(model, dist.device, dist, merged)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_ler_improvement_vs_baseline(self, result, tolerance: float = LER_IMPROVEMENT_TOLERANCE):
    """Assert LER after pre-decoder <= baseline + tolerance for both X and Z."""
    for basis in ("X", "Z"):
        ler_after = float(result[basis]["logical error ratio (mean)"])
        ler_baseline = float(result[basis]["logical error ratio (pymatch mean)"])
        self.assertLessEqual(
            ler_after,
            ler_baseline + tolerance,
            msg=
            f"{basis}: LER after ({ler_after:.6f}) > baseline ({ler_baseline:.6f}) + {tolerance}",
        )


class TestPublicInferenceModels(unittest.TestCase):
    """Tests for pre-trained model files (r9 and r13)."""

    def test_required_model_files_present(self):
        """Fail if any required model file is missing. Must not skip."""
        for model_info in REQUIRED_MODELS:
            self.assertIsNotNone(
                _resolve_model_file(model_info),
                msg=_missing_model_msg(model_info),
            )

    # -- R=9 model tests --

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d9_r9_ler_quality(self):
        """d=9, n_rounds=9, r9 model: LER avg in range, per-basis, improvement."""
        result = _run_inference_rtest(9, 9, MODEL_R9)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        LER_AVG_D9_MAX = 5e-3
        LER_BASIS_D9_MAX = 1e-2
        self.assertGreaterEqual(ler_avg, 0.0, msg="LER avg negative")
        self.assertLessEqual(ler_avg, LER_AVG_D9_MAX, msg=f"LER avg > {LER_AVG_D9_MAX}")
        self.assertGreaterEqual(ler_x, 0.0)
        self.assertLessEqual(ler_x, LER_BASIS_D9_MAX, msg="LER X too high")
        self.assertGreaterEqual(ler_z, 0.0)
        self.assertLessEqual(ler_z, LER_BASIS_D9_MAX, msg="LER Z too high")

        _assert_ler_improvement_vs_baseline(self, result, tolerance=5e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d13_r9_ler_quality(self):
        """d=13, n_rounds=13, r9 model: evaluate at distance larger than R."""
        result = _run_inference_rtest(13, 13, MODEL_R9)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        self.assertAlmostEqual(ler_avg, 2e-4, delta=1.5e-4)
        self.assertAlmostEqual(ler_x, 2e-4, delta=2e-4, msg="LER X out of range")
        self.assertAlmostEqual(ler_z, 2e-4, delta=2e-4, msg="LER Z out of range")

        _assert_ler_improvement_vs_baseline(self, result)

    # -- R=13 model tests --

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference rtest.")
    def test_inference_d13_r13_ler_quality(self):
        """d=13, n_rounds=13, r13 model: LER avg, per-basis, improvement over baseline."""
        result = _run_inference_rtest(13, 13, MODEL_R13)

        ler_x = float(result["X"]["logical error ratio (mean)"])
        ler_z = float(result["Z"]["logical error ratio (mean)"])
        ler_avg = 0.5 * (ler_x + ler_z)

        self.assertAlmostEqual(ler_avg, 2e-4, delta=1.5e-4)
        self.assertAlmostEqual(ler_x, 2e-4, delta=2e-4, msg="LER X out of range")
        self.assertAlmostEqual(ler_z, 2e-4, delta=2e-4, msg="LER Z out of range")

        _assert_ler_improvement_vs_baseline(self, result)


def _export_safetensors(directory: Path, name: str, seed: int, model_id: int = 1) -> dict:
    """Export a random model_id checkpoint with code/export/checkpoint_to_safetensors.py
    under a published-style name; return the fp16 weights it should load back as."""
    torch.manual_seed(seed)
    state_dict = ModelFactory.create_model(_build_minimal_cfg(model_id)).state_dict()
    pt_path = directory / f"seed{seed}.pt"
    torch.save(state_dict, pt_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "code" / "export" / "checkpoint_to_safetensors.py"),
            "--checkpoint",
            str(pt_path),
            "--model-id",
            str(model_id),
            "--fp16",
            "--output",
            str(directory / name),
        ],
        cwd=str(REPO_ROOT),
        env={
            **os.environ, "PYTHONPATH": str(REPO_ROOT / "code")
        },
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"checkpoint_to_safetensors.py failed with exit code {proc.returncode}:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    pt_path.unlink()
    return {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}


class TestPublishedSafeTensorsLoading(unittest.TestCase):
    """A published-style SafeTensors file is found and loaded through _load_model."""

    def test_newest_published_file_loads_through_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            # Two releases with different weights; v1.0.77 is newer than v1.0.9.
            _export_safetensors(
                models_dir, "ising_decoder_surface_code_1_fast_r9_v1.0.9_fp16.safetensors", seed=1
            )
            expected = _export_safetensors(
                models_dir, "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors", seed=2
            )

            model_file = _resolve_model_file(MODEL_R9, models_dir)
            self.assertEqual(
                model_file.name, "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"
            )

            cfg = OmegaConf.load(str(REPO_ROOT / "conf" / "config_public.yaml"))
            cfg.model_id = MODEL_R9["model_id"]
            cfg.workflow.task = "inference"
            merged = apply_public_defaults_and_model(cfg, validate_public_config(cfg))
            merged.enable_fp16 = False

            DistributedManager.initialize()
            model = _load_public_model(merged, model_file, DistributedManager())

            self.assertTrue(
                merged.enable_fp16, "fp16 metadata in the file should enable fp16 inference"
            )
            loaded = model.state_dict()
            self.assertEqual(set(loaded), set(expected))
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(loaded[name].cpu(), tensor), msg=name)


if __name__ == "__main__":
    unittest.main()
