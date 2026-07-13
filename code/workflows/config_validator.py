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
Public config normalization / validation for the public release.

Responsibilities:
- Fail-fast if the user tries to set hidden/experimental fields (via Hydra CLI `+foo=...`)
- Merge in hidden defaults (sourced from model_1_d9 config) so training runs with a minimal public config
- Apply the selected public model architecture (model_id -> model.*)
- Clamp distance/n_rounds to the model receptive field:
    D = min(distance, R)
    N_R = min(n_rounds, R)
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, Iterable, Tuple

from omegaconf import DictConfig, OmegaConf

from model.registry import PublicModelSpec, get_model_spec

_PUBLIC_ROTATION_TO_INTERNAL = {
    # Public user-facing aliases
    "O1": "XV",
    "O2": "XH",
    "O3": "ZV",
    "O4": "ZH",
}
_INTERNAL_ROTATION_TO_PUBLIC = {v: k for k, v in _PUBLIC_ROTATION_TO_INTERNAL.items()}

_PUBLIC_MODEL_ID_TO_LR = {
    1: 3e-4,
    2: 2e-4,
    3: 1e-4,
    4: 2e-4,
    5: 1e-4,
}
_PUBLIC_COLOR_LR = 1e-5


def _default_public_noise_model() -> Dict[str, float]:
    return {"p": 0.003}


def _default_precomputed_frames_dir() -> str:
    """
    Default location for precomputed frames shipped with (or generated inside) this repo.

    We compute this path relative to the codebase so it is stable regardless of the user's
    current working directory.
    """
    # .../<repo>/code/workflows/config_validator.py -> repo root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / "frames_data").resolve())


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    if val in ("0", "false", "no", "off", ""):
        return False
    return True


def _normalize_code(value: Any) -> str:
    """Normalize the public code-family selector."""
    if value is None:
        return "surface"
    code = str(value).strip().lower()
    if code in ("surface", "surface_code"):
        return "surface"
    if code in ("color", "color_code"):
        return "color"
    raise ValueError("Invalid code={!r}. Use 'surface' or 'color'.".format(value))


def _normalize_code_rotation(value: Any) -> str:
    """
    Normalize code rotation values.

    Public config accepts O1..O4 for user convenience. Internally we keep using:
    XV, XH, ZV, ZH (as expected by SurfaceCode / MemoryCircuit).
    """
    if value is None:
        return value
    s = str(value).strip().upper()
    if s in _PUBLIC_ROTATION_TO_INTERNAL:
        return _PUBLIC_ROTATION_TO_INTERNAL[s]
    if s in _INTERNAL_ROTATION_TO_PUBLIC:
        return s
    raise ValueError(
        f"Invalid data.code_rotation={value!r}. "
        f"Use one of {sorted(_PUBLIC_ROTATION_TO_INTERNAL.keys())} (public) "
        f"or {sorted(_INTERNAL_ROTATION_TO_PUBLIC.keys())} (internal)."
    )


def _base_hidden_defaults_dict() -> Dict[str, Any]:
    """
    Baseline config used as the source-of-truth for hidden defaults.

    IMPORTANT: We intentionally embed these defaults directly in code so the public
    release does not ship internal/legacy config files. These values were copied
    from the historical `config_pre_decoder_memory_surface_model_1_d9.yaml`.
    """
    base_output_dir = os.environ.get("PREDECODER_BASE_OUTPUT_DIR", "outputs")
    output_root = f"{base_output_dir}/${{exp_tag}}"
    return {
        "exp_tag": "pre-decoder",
        "output": output_root,
        "hydra": {
            "run": {
                "dir": "${output}"
            },
            "output_subdir": "hydra"
        },
        "resume_dir": f"{output_root}/models",
        "enable_fp16": False,
        "enable_bf16": False,
        "enable_matmul_tf32": True,
        "enable_cudnn_tf32": True,
        "enable_cudnn_benchmark": True,
        "torch_compile": _get_env_bool("PREDECODER_TORCH_COMPILE", True),
        "torch_compile_mode": os.environ.get("PREDECODER_TORCH_COMPILE_MODE", "default"),
        "load_checkpoint": False,
        "code": "surface",
        "distance": 9,
        "n_rounds": 9,
        "multiple_distances": [13, 13],
        "multiple_rounds": [13, 13],
        "use_multiple_patches": False,
        "meas_basis": "both",
        "workflow": {
            "task": "train"
        },
        "data":
            {
                "timelike_he": True,
                "num_he_cycles": 1,
                "use_weight2_timelike": False,
                "use_parallel_spacelike": False,
                "max_passes_w1": 8,
                "max_passes_w2": 4,
                "decompose_y": True,
                "p_error": None,
                "p_min": 0.001,
                "p_max": 0.006,
                "error_mode": "circuit_level_surface_custom",
                # Public config overrides this; keep the historical default for completeness.
                "precomputed_frames_dir": _default_precomputed_frames_dir(),
                "enable_correlated_pymatching": False,
                "code_rotation": "XV",
                "noise_model": None,
                "skip_noise_upscaling": False,
            },
        "model":
            {
                "version": "predecoder_memory_v1",
                "dropout_p": 0.05,
                "activation": "gelu",
                "num_filters": [128, 128, 128, 4],
                "kernel_size": [3, 3, 3, 3],
                "input_channels": 4,
                "out_channels": 4,
            },
        "datapipe": "memory",
        "data_method": "train",
        "train":
            {
                # Production baseline: 2^26 shots / epoch when training with 8 GPUs.
                # The training script will auto-scale this based on detected world size / GPU count.
                "num_samples": 67108864,
                "accumulate_steps": 2,
                "checkpoint_interval": 1,
                "save_every_datasets": 5,
                "epochs": 100,
            },
        # NOTE: temporarily reduced for faster iteration during refactor/testing.
        "val": {
            "num_samples": 65536,
            "threshold": 0.5,
            "trials": 1
        },
        "optimizer_type": "Lion",
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-7,
            "beta2": 0.95
        },
        "lr_scheduler":
            {
                "type": "warmup_then_decay",
                "warmup_steps": 100,
                "milestones": [0.25, 0.5, 1.0],
                "gamma": 0.7,
                "min_lr": 1e-6,
            },
        "batch_schedule":
            {
                "enabled": True,
                "initial": 256,
                "final": 1024,
                "start_epoch": 1,
                "end_epoch": 3,
            },
        "validation_ler": True,
        "early_stopping": {
            "enabled": True,
            "patience": 100
        },
        "time_based_early_stopping": {
            "enabled": False,
            "safety_margin_minutes": 5
        },
        "ema": {
            "use_ema": True,
            "decay": 0.0001
        },
        "test":
            {
                "num_samples": 262144,
                "trials": 1,
                "distance": 9,
                "n_rounds": 9,
                "noise_model": "train",
                "p_error": 0.006,
                "dataloader":
                    {
                        "batch_size": 2048,
                        "num_workers": 4,
                        "persistent_workers": True,
                        "prefetch_factor": 2,
                    },
                "sampler": {
                    "shuffle": False,
                    "drop_last": False
                },
                "syn_red": "full",
                "th_data": 0.0,
                "th_syn": 0.0,
                "sampling_mode": "threshold",
                "temperature": 0.0,
                "temperature_data": None,
                "temperature_syn": None,
                "per_round": False,
                "meas_basis_test": "both",
                "use_model_checkpoint": -1,
            },
        "threshold":
            {
                "p_values": [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
                "distances": [5, 7, 9, 11, 13],
                "n_rounds": None,
            },
    }


def _apply_code_specific_defaults(
    merged: DictConfig, code: str, model_spec: PublicModelSpec
) -> None:
    """Patch hidden defaults that differ by code family."""
    merged.code = code

    if "data" not in merged:
        merged.data = {}
    if "test" not in merged:
        merged.test = {}
    if "model" not in merged:
        merged.model = {}

    if merged.data.noise_model is None:
        merged.data.noise_model = _default_public_noise_model()

    if code == "surface":
        merged.data.timelike_he = True
        merged.data.decompose_y = True
        merged.data.error_mode = "circuit_level_surface_custom"
        merged.test.p_error = 0.006
        return

    # Color-code public defaults. These mirror the Torch/cuStabilizer color
    # path validated during PR68 follow-up smoke runs.
    merged.data.superdense = True
    merged.data.schedule = "nearest-neighbor"
    merged.data.enable_z_feedforward = True
    merged.data.apply_data_x_override = True
    merged.data.apply_spacelike_he = True
    merged.data.he_max_iterations = 16
    merged.data.use_coset_search = False
    merged.data.timelike_he = False
    merged.data.use_weight2_timelike = False
    merged.data.use_weight3_timelike = False
    merged.data.decompose_y = False
    merged.data.error_mode = "circuit_level_color_code"
    merged.data.p_error = None
    merged.data.p_min = 0.003
    merged.data.p_max = 0.003

    # The Conv3D head may be widened beyond the four public output channels;
    # PreDecoderModelMemory_v1 slices back to out_channels in forward().
    filters = list(model_spec.num_filters)
    if filters:
        filters[-1] = max(int(filters[-1]), 16)
        merged.model.num_filters = filters
    merged.model.dropout_p = 0.01

    merged.test.p_error = 0.003
    merged.test.noise_model_family = "legacy"
    merged.test.noise_instruction_semantics = "current"
    merged.test.noise_mode = "legacy"
    merged.test.gidney_style_noise = False
    if "dataloader" not in merged.test:
        merged.test.dataloader = {}
    merged.test.dataloader.num_workers = 0
    merged.test.dataloader.persistent_workers = False
    merged.test.dataloader.prefetch_factor = None


def _select(cfg: DictConfig, key: str) -> Tuple[bool, Any]:
    """
    Return (exists, value) for a dot-path in cfg.
    Note: OmegaConf.select returns None both for missing keys and explicit nulls,
    so we treat a key as existing iff it is present in the underlying container.
    """
    # OmegaConf doesn't provide a direct 'has_key' for dotted paths; implement via container walk.
    cur: Any = cfg
    parts = key.split(".")
    for p in parts:
        if not isinstance(cur, DictConfig) or p not in cur:
            return False, None
        cur = cur[p]
    return True, cur


def _assert_not_present(cfg: DictConfig, keys: Iterable[str], *, context: str) -> None:
    for k in keys:
        exists, _ = _select(cfg, k)
        if exists:
            raise ValueError(
                f"Config field '{k}' is not supported in the public release ({context}). "
                f"Remove it from the config/CLI overrides."
            )


def validate_public_config(cfg: DictConfig) -> PublicModelSpec:
    """
    Validate the user-facing config BEFORE we merge in hidden defaults.

    Returns:
        PublicModelSpec for cfg.model_id (validated).
    """
    # model_id must exist in public config
    if "model_id" not in cfg:
        raise ValueError("Missing required field: 'model_id' (choose 1..5 or 'B').")

    code = _normalize_code(getattr(cfg, "code", "surface"))
    model_spec = get_model_spec(cfg.model_id)
    if code == "color" and int(model_spec.receptive_field) > 13:
        raise ValueError(
            "code='color' currently supports public model_ids with receptive field <= 13 "
            "(choose model_id 1, 2, 4, 5, or B)."
        )
    # The cascade/bottleneck model "B" is only released for the color code.
    if model_spec.model_version == "predecoder_memory_cascade" and code != "color":
        raise ValueError(
            "model_id='B' (cascade/bottleneck) is only supported with code='color' "
            "in this release."
        )

    # Public config requires distance/n_rounds (evaluation targets)
    if "distance" not in cfg or "n_rounds" not in cfg:
        raise ValueError("Missing required fields: 'distance' and 'n_rounds'.")
    try:
        d = int(cfg.distance)
        r = int(cfg.n_rounds)
    except Exception as e:
        raise ValueError(
            f"Invalid distance/n_rounds: distance={cfg.distance!r}, n_rounds={cfg.n_rounds!r}"
        ) from e
    if d <= 0 or r <= 0:
        raise ValueError(
            f"Invalid distance/n_rounds: distance={d}, n_rounds={r} (must be positive integers)"
        )

    if "train" in cfg:
        raise ValueError("Config field 'train' is not supported in the public release.")
    if "val" in cfg:
        raise ValueError("Config field 'val' is not supported in the public release.")
    if "test" in cfg:
        raise ValueError("Config field 'test' is not supported in the public release.")

    # Fail-fast on known hidden fields if the user tries to inject them.
    _assert_not_present(
        cfg,
        keys=(
            # output paths are managed by the runner scripts; not user-configurable in public release
            "output",
            "resume_dir",
            # precision / tf32 knobs (always fp32 + tf32 enabled)
            "enable_fp16",
            "enable_bf16",
            "enable_matmul_tf32",
            "enable_cudnn_tf32",
            # always both bases
            "meas_basis",
            # multi-patch curriculum mode (hidden)
            "use_multiple_patches",
            "multiple_distances",
            "multiple_rounds",
            # optimizer knobs (only optimizer.lr exposed)
            "optimizer",
            "optimizer_type",
            "lr_scheduler",
            "batch_schedule",
            # obsolete/confusing
            "train.save_every_datasets",
            # validation hidden knobs
            "val.threshold",
            "val.trials",
            # early stopping extras hidden
            "time_based_early_stopping",
            "ema",
        ),
        context="hidden field override",
    )

    # Restrict cfg.data to a small public surface (others can be too experimental).
    if "data" in cfg and isinstance(cfg.data, DictConfig):
        # NOTE: precomputed frames path is intentionally hidden from the public config.
        # We default it internally to <repo>/frames_data (see _default_precomputed_frames_dir).
        if "precomputed_frames_dir" in cfg.data:
            raise ValueError(
                "Config field 'data.precomputed_frames_dir' is not supported in the public release. "
                "Remove it from the config/CLI overrides."
            )
        # `use_compile` and `use_parallel_spacelike` are HE-acceleration flags
        # surfaced in the public release. Both default False; users opt in via
        # `conf/config_public.yaml` or CLI override. See README.md, section
        # "HE acceleration (advanced): parallel spacelike" for the contract.
        allowed_data_keys = {
            "code_rotation",
            "noise_model",
            "skip_noise_upscaling",
            "use_compile",
            "use_parallel_spacelike",
        }
        for k in cfg.data.keys():
            if k not in allowed_data_keys:
                raise ValueError(
                    f"Config field 'data.{k}' is not supported in the public release. "
                    f"Allowed data fields are: {sorted(allowed_data_keys)}"
                )
        # These two flags are part of the public config surface, so keep their
        # accepted type stricter than hidden/internal HE knobs that are merged
        # from trusted defaults. OmegaConf accepts strings like "True"/"yes",
        # which would otherwise flow into downstream `bool(...)` casts and
        # become truthy regardless of the user's intent.
        for bool_key in ("skip_noise_upscaling", "use_compile", "use_parallel_spacelike"):
            if bool_key in cfg.data and not isinstance(cfg.data[bool_key], bool):
                raise ValueError(
                    f"Config field 'data.{bool_key}' must be a boolean "
                    f"(got {type(cfg.data[bool_key]).__name__}: {cfg.data[bool_key]!r})."
                )
        # Validate rotation value (accept O1..O4; also allow internal XV/XH/ZV/ZH for compatibility).
        if "code_rotation" in cfg.data:
            _normalize_code_rotation(cfg.data.code_rotation)

    # Restrict optimizer sub-keys: only lr is public.
    if "optimizer" in cfg and isinstance(cfg.optimizer, DictConfig):
        for k in cfg.optimizer.keys():
            if k != "lr":
                raise ValueError(
                    f"Config field 'optimizer.{k}' is not supported in the public release. "
                    f"Only 'optimizer.lr' is user-configurable."
                )

    return model_spec


def clamp_to_receptive_field(cfg: DictConfig, R: int) -> None:
    """In-place clamp of cfg.distance and cfg.n_rounds to receptive field R."""
    if not isinstance(R, int) or R <= 0:
        raise ValueError(f"Invalid receptive field R={R!r}")
    if "distance" not in cfg or "n_rounds" not in cfg:
        raise ValueError("Both 'distance' and 'n_rounds' must be present in config.")
    cfg.distance = int(min(int(cfg.distance), R))
    cfg.n_rounds = int(min(int(cfg.n_rounds), R))


def apply_public_defaults_and_model(cfg: DictConfig, model_spec: PublicModelSpec) -> DictConfig:
    """
    Merge hidden defaults and apply public model settings.

    Returns a new DictConfig (does not mutate input).
    """
    base_cfg = OmegaConf.create(_base_hidden_defaults_dict())

    # Merge: base provides full training-ready config; public cfg overrides user-visible fields.
    merged = OmegaConf.merge(base_cfg, cfg)
    OmegaConf.set_struct(merged, False)
    code = _normalize_code(getattr(merged, "code", "surface"))

    # In the public release:
    # - cfg.distance / cfg.n_rounds are the *evaluation targets* the user cares about
    # - training always uses distance=n_rounds=R (the model receptive field)
    requested_distance = int(merged.distance)
    requested_n_rounds = int(merged.n_rounds)

    # Enforce public invariants (hidden from user)
    merged.enable_fp16 = False
    merged.enable_bf16 = False
    merged.enable_matmul_tf32 = True
    merged.enable_cudnn_tf32 = True

    merged.meas_basis = "both"

    # Disable multi-patch mode explicitly
    if "data" not in merged:
        merged.data = {}
    merged.data.use_multiple_patches = False
    merged.multiple_distances = None
    merged.multiple_rounds = None

    # Always use repo-relative frames_data by default (hidden from public config).
    merged.data.precomputed_frames_dir = _default_precomputed_frames_dir()

    # Apply model architecture from registry
    if "model" not in merged:
        merged.model = {}
    if model_spec.model_overrides:
        # Non-convolutional model (e.g. cascade "B"): the registry carries the
        # full model.* block. Write it verbatim and drop conv-stack-only fields
        # inherited from the hidden defaults so they can't confuse the builder.
        merged.model.version = model_spec.model_version
        for key, value in model_spec.model_overrides.items():
            merged.model[key] = value
        if "num_filters" in merged.model:
            merged.model.pop("num_filters")
    else:
        merged.model.version = model_spec.model_version
        merged.model.num_filters = list(model_spec.num_filters)
        merged.model.kernel_size = list(model_spec.kernel_size)

    _apply_code_specific_defaults(merged, code, model_spec)

    # Public release: hard-code optimizer.lr based on code family/model choice.
    # (User is not allowed to override optimizer settings.)
    if "optimizer" not in merged:
        merged.optimizer = {}
    if code == "color":
        lr = _PUBLIC_COLOR_LR
    else:
        lr = _PUBLIC_MODEL_ID_TO_LR.get(int(model_spec.model_id))
        if lr is None:
            raise ValueError(f"No public LR mapping for model_id={model_spec.model_id!r}")
    merged.optimizer.lr = float(lr)

    # Public release: production-like batch schedule defaults.
    # Target behavior: per-GPU batch size is 512 in the first epoch, 2048 thereafter.
    # Model 3 is heavier; use a smaller schedule there.
    if "batch_schedule" not in merged:
        merged.batch_schedule = {}
    merged.batch_schedule.enabled = True
    # Heavier models use a smaller batch schedule: model 3 (k=5 stack) and the
    # cascade/bottleneck model "B" (embed_dim=512).
    is_cascade = model_spec.model_version == "predecoder_memory_cascade"
    if str(model_spec.model_id) == "3" or is_cascade:
        merged.batch_schedule.initial = 256
        merged.batch_schedule.final = 1024
    else:
        merged.batch_schedule.initial = 512
        merged.batch_schedule.final = 2048
    # "First epoch only" initial, then final for all later epochs.
    merged.batch_schedule.start_epoch = 0
    merged.batch_schedule.end_epoch = 0

    # Public release: training epochs default to production values,
    # but honor explicit user overrides for quick validation runs.
    if "train" not in merged:
        merged.train = {}
    if not ("train" in cfg and isinstance(cfg.train, DictConfig) and "epochs" in cfg.train):
        merged.train.epochs = 100

    # Public release: validation sample count defaults to production values,
    # but honor explicit user overrides for quick validation runs.
    if "val" not in merged:
        merged.val = {}
    # NOTE: temporarily reduced for faster iteration during refactor/testing.
    if not ("val" in cfg and isinstance(cfg.val, DictConfig) and "num_samples" in cfg.val):
        merged.val.num_samples = 65536

    # Train vs inference window semantics (public release):
    # - Top-level cfg.distance / cfg.n_rounds are the user-specified *evaluation* targets.
    # - Training always runs on the model receptive field R (distance=n_rounds=R).
    task = str(getattr(getattr(merged, "workflow", None), "task", "train")).strip().lower()
    R = int(model_spec.receptive_field)
    if R <= 0:
        raise ValueError(f"Invalid receptive field R={R!r}")
    if task == "train":
        merged.distance = R
        merged.n_rounds = R
    else:
        merged.distance = int(requested_distance)
        merged.n_rounds = int(requested_n_rounds)

    # Public code_rotation aliases: normalize O1..O4 -> internal XV/XH/ZV/ZH.
    if "data" in merged and "code_rotation" in merged.data:
        merged.data.code_rotation = _normalize_code_rotation(merged.data.code_rotation)

    # Test/evaluation config is hidden and always uses the user-requested window.
    if "test" not in merged:
        merged.test = {}
    if not ("test" in cfg and isinstance(cfg.test, DictConfig) and "num_samples" in cfg.test):
        merged.test.num_samples = 262144
    merged.test.distance = int(requested_distance)
    merged.test.n_rounds = int(requested_n_rounds)
    merged.test.noise_model = "train"
    return merged
