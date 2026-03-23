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
"""
Automated LER collection using pre-generated models over multiple p values
and noise model modes (single-p vs explicit 25p).

Env overrides (optional):
  PREDECODER_TEST_MODEL_FILE=/path/to/PreDecoderModelMemory_v1.0.94.pt
  PREDECODER_TEST_MODEL_DIR=/path/to/models
  PREDECODER_TEST_ALL_MODELS=1
  PREDECODER_TEST_MODEL_GLOB=PreDecoderModelMemory_v1.0.*.pt
  PREDECODER_TEST_MODEL_ID=1
  PREDECODER_TEST_DISTANCE=9
  PREDECODER_TEST_N_ROUNDS=9
  PREDECODER_TEST_NUM_SAMPLES=2048
  PREDECODER_TEST_P_VALUES=0.002,0.003,0.004,0.005,0.006,0.007,0.008
  PREDECODER_TEST_P_REF=0.003
  PREDECODER_TEST_NOISE_SEED=1234
  PREDECODER_TEST_NOISE_JITTER=0.2
  PREDECODER_TEST_RANDOM_VARIANTS=2
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from omegaconf import OmegaConf

# Ensure repo's code/ is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.logical_error_rate import count_logical_errors_with_errorbar
from training.distributed import DistributedManager
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model


def _find_model_file() -> Path | None:
    """Find the pre-generated model file in common locations or env overrides."""
    repo_root = Path(__file__).resolve().parents[2]
    env_file = os.environ.get("PREDECODER_TEST_MODEL_FILE")
    env_dir = os.environ.get("PREDECODER_TEST_MODEL_DIR")
    candidates = []
    if env_file:
        candidates.append(Path(env_file))
    if env_dir:
        candidates.append(Path(env_dir) / "PreDecoderModelMemory_v1.0.94.pt")
    candidates.extend(
        [
            repo_root / "models" / "PreDecoderModelMemory_v1.0.94.pt",
            repo_root / "outputs" / "predecoder_model_1" / "models" /
            "PreDecoderModelMemory_v1.0.94.pt",
        ]
    )
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, str(int(default))).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _get_env_csv_floats(name: str, default: Iterable[float]) -> list[float]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_checkpoint_from_name(path: Path) -> int | None:
    parts = path.stem.split(".")
    if not parts:
        return None
    try:
        return int(parts[-1])
    except Exception:
        return None


def _collect_model_paths(repo_root: Path) -> list[tuple[Path, int]]:
    if _get_env_bool("PREDECODER_TEST_ALL_MODELS", False):
        model_dir = Path(os.environ.get("PREDECODER_TEST_MODEL_DIR", repo_root / "models"))
        pattern = os.environ.get("PREDECODER_TEST_MODEL_GLOB", "PreDecoderModelMemory_v1.0.*.pt")
        paths = sorted(model_dir.glob(pattern))
        out = []
        for path in paths:
            ckpt = _parse_checkpoint_from_name(path)
            if ckpt is not None:
                out.append((path, ckpt))
        return out
    model_file = _find_model_file()
    if model_file is None:
        return []
    ckpt = _parse_checkpoint_from_name(model_file)
    if ckpt is None:
        return []
    return [(model_file, ckpt)]


def _clone_cfg(cfg):
    """Deep copy an OmegaConf config with resolved values."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _scale_noise_dict(noise_dict: Dict[str, float], scale: float) -> Dict[str, float]:
    """Scale a 25p noise model dict by a factor, clamping to [0, 1]."""
    out = {}
    for k in sorted(noise_dict.keys()):
        val = float(noise_dict[k]) * float(scale)
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        out[k] = val
    return out


def _randomize_noise_dict(
    noise_dict: Dict[str, float],
    scale: float,
    rng: random.Random,
    jitter: float,
) -> Dict[str, float]:
    """Scale + jitter a 25p noise dict deterministically (seeded RNG), clamped to [0, 1]."""
    out = {}
    for k in sorted(noise_dict.keys()):
        base = float(noise_dict[k]) * float(scale)
        factor = rng.uniform(1.0 - jitter, 1.0 + jitter)
        val = base * factor
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        out[k] = val
    return out


class TestLERPretrainedModels(unittest.TestCase):
    """Collect LER data across p values and noise model modes using pre-generated models."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for LER collection tests.")
    def test_ler_collection_pretrained_models(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        model_entries = _collect_model_paths(repo_root)
        if not model_entries:
            raise unittest.SkipTest(
                "Missing pre-generated model. Set PREDECODER_TEST_MODEL_FILE or "
                "PREDECODER_TEST_MODEL_DIR, or PREDECODER_TEST_ALL_MODELS=1 to run LER collection tests."
            )

        cfg_path = repo_root / "conf" / "config_public.yaml"
        cfg = OmegaConf.load(str(cfg_path))
        cfg.model_id = _get_env_int("PREDECODER_TEST_MODEL_ID", 1)
        cfg.distance = _get_env_int("PREDECODER_TEST_DISTANCE", 9)
        cfg.n_rounds = _get_env_int("PREDECODER_TEST_N_ROUNDS", cfg.distance)
        cfg.workflow.task = "inference"

        model_spec = validate_public_config(cfg)
        merged = apply_public_defaults_and_model(cfg, model_spec)

        merged.test.latency_num_samples = 0
        merged.test.verbose_inference = False
        # Keep this reasonably fast; adjust if you want tighter bounds.
        merged.test.num_samples = _get_env_int("PREDECODER_TEST_NUM_SAMPLES", 2048)
        if "dataloader" in merged.test:
            merged.test.dataloader.num_workers = 0
            merged.test.dataloader.persistent_workers = False
            if hasattr(merged.test.dataloader, "prefetch_factor"):
                merged.test.dataloader.prefetch_factor = None

        # Keep dataloader stable in containers and avoid env overrides.
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

        # Standard p sweep (from config_validator threshold defaults).
        p_values = _get_env_csv_floats(
            "PREDECODER_TEST_P_VALUES",
            [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
        )
        # Noise model variants (keep small but tweakable):
        # - single_p: legacy p_error
        # - explicit_25p: scale config 25p noise dict by p/p_ref and apply small random jitter
        base_25p = OmegaConf.to_container(merged.data.noise_model, resolve=True)
        p_ref = _get_env_float("PREDECODER_TEST_P_REF", 0.003)
        rng = random.Random(_get_env_int("PREDECODER_TEST_NOISE_SEED", 1234))
        jitter = _get_env_float("PREDECODER_TEST_NOISE_JITTER", 0.2)  # +/- 20% per-parameter
        random_variants = _get_env_int("PREDECODER_TEST_RANDOM_VARIANTS", 2)

        cases: Iterable[Tuple[str, float, str | None, Dict[str, float] | None]] = []
        for p_value in p_values:
            cases.append(("single_p", p_value, None, None))
            scale = float(p_value) / float(p_ref)
            for i in range(random_variants):
                noise = _randomize_noise_dict(base_25p, scale, rng, jitter)
                cases.append(("explicit_25p_random", p_value, f"rand{i + 1}", noise))

        try:
            DistributedManager.initialize()
            dist = DistributedManager()

            for model_path, checkpoint in model_entries:
                merged.model_checkpoint_dir = str(model_path.parent)
                merged.test.use_model_checkpoint = int(checkpoint)
                model = _load_model(merged, dist)

                results = []
                for mode, p_value, tag, noise_dict in cases:
                    case_cfg = _clone_cfg(merged)
                    if mode == "single_p":
                        case_cfg.test.noise_model = "none"
                        case_cfg.test.p_error = float(p_value)
                        case_cfg.data.noise_model = None
                    else:
                        case_cfg.test.noise_model = "train"
                        # Apply scaled + jittered 25p noise dict.
                        case_cfg.data.noise_model = noise_dict
                        case_cfg.test.p_error = float(p_value)

                    result = count_logical_errors_with_errorbar(model, dist.device, dist, case_cfg)
                    results.append((mode, p_value, tag, result))

                    # Basic sanity checks on collected metrics.
                    for basis in ("X", "Z"):
                        self.assertIn(basis, result)
                        ler = float(result[basis]["logical error ratio (mean)"])
                        baseline = float(result[basis]["logical error ratio (pymatch mean)"])
                        err = float(result[basis]["logical error ratio (standard error)"])
                        base_err = float(
                            result[basis]["logical error ratio (pymatch standard error)"]
                        )
                        self.assertGreaterEqual(ler, 0.0)
                        self.assertLessEqual(ler, 1.0)
                        self.assertGreaterEqual(baseline, 0.0)
                        self.assertLessEqual(baseline, 1.0)
                        # Predecoder should not be worse than baseline within tolerance.
                        # Use a base tolerance plus a few standard errors for stability.
                        tol = 5e-4 + 3.0 * max(err, base_err)
                        self.assertLessEqual(
                            ler,
                            baseline + tol,
                            msg=
                            f"{basis}: LER after ({ler:.6f}) > baseline ({baseline:.6f}) + {tol:.6f}",
                        )

                if dist.rank == 0:
                    print(f"[LER] model={model_path.name} (checkpoint={checkpoint})")
                    for mode, p_value, tag, result in results:
                        x = result["X"]
                        z = result["Z"]
                        ler_x = float(x["logical error ratio (mean)"])
                        ler_z = float(z["logical error ratio (mean)"])
                        base_x = float(x["logical error ratio (pymatch mean)"])
                        base_z = float(z["logical error ratio (pymatch mean)"])
                        ler_avg = 0.5 * (ler_x + ler_z)
                        tag_note = "" if tag is None else f" {tag}"
                        print(
                            f"[LER] mode={mode} p={p_value:.4f}{tag_note} | "
                            f"X: {ler_x:.6f} (base {base_x:.6f}) "
                            f"Z: {ler_z:.6f} (base {base_z:.6f}) "
                            f"avg: {ler_avg:.6f}"
                        )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
