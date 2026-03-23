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
"""One-off script to measure LER at d=9, n_rounds=9 for TestPublicInferenceModelV1 baseline.

Who: Maintainers or developers working on the public inference model or d=9 test bounds.
When: Run when you want to tighten test_inference_d9_noise25p_ler_quality bounds (e.g. after
  changing the model, noise, or config), or to recalibrate after flaky CI. Not run by CI.
Use the printed "Suggested test bounds" to update LER_AVG_D9_MAX / LER_BASIS_D9_MAX in
  test_inference_public_model.py if desired.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from workflows.config_validator import apply_public_defaults_and_model, validate_public_config
from workflows.run import _load_model
from evaluation.logical_error_rate import count_logical_errors_with_errorbar
from training.distributed import DistributedManager


def main():
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "conf" / "config_public.yaml"
    cfg = OmegaConf.load(str(cfg_path))

    cfg.model_id = 1
    cfg.distance = 9
    cfg.n_rounds = 9
    cfg.workflow.task = "inference"

    model_spec = validate_public_config(cfg)
    merged = apply_public_defaults_and_model(cfg, model_spec)

    model_file = repo_root / "models" / "PreDecoderModelMemory_v1.0.94.pt"
    if not model_file.exists():
        print(f"Missing model: {model_file}")
        return 1

    merged.model_checkpoint_dir = str(model_file.parent)
    merged.test.use_model_checkpoint = 94
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
        model = _load_model(merged, dist)
        result = count_logical_errors_with_errorbar(model, dist.device, dist, merged)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    x = result["X"]
    z = result["Z"]
    ler_x = float(x["logical error ratio (mean)"])
    ler_z = float(z["logical error ratio (mean)"])
    base_x = float(x["logical error ratio (pymatch mean)"])
    base_z = float(z["logical error ratio (pymatch mean)"])
    ler_avg = 0.5 * (ler_x + ler_z)

    print("d=9, n_rounds=9 (25p noise, model v1.0.94):")
    print(f"  LER X (after):  {ler_x:.6f}  baseline: {base_x:.6f}")
    print(f"  LER Z (after):  {ler_z:.6f}  baseline: {base_z:.6f}")
    print(f"  LER avg:        {ler_avg:.6f}")
    print("\n# Suggested test bounds (mean ± delta): use ler_avg with delta e.g. 1e-3 or 5e-4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
