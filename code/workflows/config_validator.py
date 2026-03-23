# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
Public config normalization / validation for the early-access public release.

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
                "use_weight2": False,
                "max_passes_w1": 8,
                "max_passes_w2": 4,
                "use_compile": True,
                "compile_chunk_size": 2,
                "compute_dtype": None,
                "use_coset_search": False,
                "coset_max_generators": 20,
                "use_dense_overlap": False,
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
                "num_samples": int(os.environ.get("PREDECODER_NUM_SAMPLES", 67108864)),
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
        raise ValueError("Missing required field: 'model_id' (choose 1..5).")

    model_spec = get_model_spec(cfg.model_id)

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
        allowed_data_keys = {"code_rotation", "noise_model"}
        for k in cfg.data.keys():
            if k not in allowed_data_keys:
                raise ValueError(
                    f"Config field 'data.{k}' is not supported in the public release. "
                    f"Allowed data fields are: {sorted(allowed_data_keys)}"
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
    merged.model.version = model_spec.model_version
    merged.model.num_filters = list(model_spec.num_filters)
    merged.model.kernel_size = list(model_spec.kernel_size)

    # Public release: hard-code optimizer.lr based on model choice.
    # (User is not allowed to override optimizer settings.)
    if "optimizer" not in merged:
        merged.optimizer = {}
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
    if int(model_spec.model_id) == 3:
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
