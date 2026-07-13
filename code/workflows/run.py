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

import hydra, sys, torch, os, json, numpy as np
from omegaconf import DictConfig, OmegaConf
from training.train import main as train_main
from model.factory import ModelFactory
from data.factory import DatapipeFactory
from hydra.utils import to_absolute_path
from workflows.config_validator import (
    apply_public_defaults_and_model,
    validate_public_config,
)

from training.distributed import DistributedManager

from torch.utils.data import DataLoader


def _ensure_inference_io_channels(cfg):
    # 1) Ensure out_channels matches the model’s heads (4: z_data, x_data, syn_x, syn_z)
    if not getattr(cfg.model, "out_channels", None) or cfg.model.out_channels == 0:
        cfg.model.out_channels = 4

    # 2) Infer input_channels from a single inference sample if not set
    if not getattr(cfg.model, "input_channels", None) or cfg.model.input_channels == 0:
        ds = DatapipeFactory.create_datapipe_inference(cfg)
        tmp = DataLoader(ds, batch_size=1)
        sample = next(iter(tmp))
        cfg.model.input_channels = int(sample["trainX"].shape[1])

    # 3) Keep num_filters consistent with out_channels
    if hasattr(cfg.model, "num_filters"):
        filters = list(cfg.model.num_filters)
        if filters and filters[-1] != cfg.model.out_channels:
            print(
                f"[run] Adjusting model.num_filters[-1] {filters[-1]} -> {cfg.model.out_channels}"
            )
            filters[-1] = cfg.model.out_channels
            cfg.model.num_filters = filters


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray, dict)):
        return list(value)
    return [value]


def _basis_list(value):
    bases = [str(item).upper() for item in _as_list(value)]
    if any(basis in ("BOTH", "MIXED") for basis in bases):
        return ["X", "Z"]
    for basis in bases:
        if basis not in ("X", "Z"):
            raise ValueError(f"Unsupported color threshold basis {basis!r}")
    return bases


def _threshold_value(cfg, name, fallback=None):
    threshold_cfg = getattr(cfg, "threshold", None)
    if threshold_cfg is not None and hasattr(threshold_cfg, name):
        value = getattr(threshold_cfg, name)
        if value is not None:
            return value
    return fallback


def _resolve_color_threshold_settings(cfg):
    test_cfg = getattr(cfg, "test", cfg)
    distances = [
        int(item) for item in _as_list(
            _threshold_value(
                cfg, "distances", [getattr(test_cfg, "distance", getattr(cfg, "distance"))]
            )
        )
    ]
    p_values = [
        float(item) for item in _as_list(
            _threshold_value(
                cfg, "p_values", getattr(test_cfg, "p_values", [getattr(test_cfg, "p_error")])
            )
        )
    ]

    threshold_cfg = getattr(cfg, "threshold", None)
    n_rounds_cfg = getattr(threshold_cfg, "n_rounds", None) if threshold_cfg is not None else None
    if n_rounds_cfg is None:
        n_rounds = {int(d): int(d) for d in distances}
    else:
        n_rounds_values = [int(item) for item in _as_list(n_rounds_cfg)]
        if len(n_rounds_values) == 1:
            n_rounds = {int(d): int(n_rounds_values[0]) for d in distances}
        elif len(n_rounds_values) == len(distances):
            n_rounds = {int(d): int(r) for d, r in zip(distances, n_rounds_values)}
        else:
            raise ValueError(
                "threshold.n_rounds must be null, scalar, or match threshold.distances length"
            )

    num_samples = int(_threshold_value(cfg, "num_samples", getattr(test_cfg, "num_samples", 0)))
    if num_samples <= 0:
        raise ValueError("Color threshold requires threshold.num_samples or test.num_samples > 0")

    bases = _basis_list(_threshold_value(cfg, "basis", getattr(test_cfg, "meas_basis_test", "X")))
    return {
        "distances": distances,
        "p_values": p_values,
        "n_rounds": n_rounds,
        "num_samples": num_samples,
        "bases": bases,
    }


def _record_resolved_model_path(cfg, model_path):
    """Record the resolved checkpoint path on cfg so color aggregations can find it."""
    try:
        cfg.resolved_model_checkpoint_path = model_path
    except Exception:
        OmegaConf.update(
            cfg, "resolved_model_checkpoint_path", model_path, merge=False, force_add=True
        )


def _apply_public_inference_env_overrides(cfg) -> None:
    try:
        test_cfg = getattr(cfg, "test", None)
        if test_cfg is None:
            return
        env_samples = os.environ.get("PREDECODER_INFERENCE_NUM_SAMPLES")
        if env_samples:
            test_cfg.num_samples = int(env_samples)
        env_latency = os.environ.get("PREDECODER_INFERENCE_LATENCY_SAMPLES")
        if env_latency:
            test_cfg.latency_num_samples = int(env_latency)
        env_basis = os.environ.get("PREDECODER_INFERENCE_MEAS_BASIS")
        if env_basis:
            test_cfg.meas_basis_test = str(env_basis)
    except Exception:
        pass


def _is_standalone_color_config(cfg: DictConfig, code_name: str) -> bool:
    """Whether a color config should bypass the public-config validator.

    The public config (``conf/config_public.yaml``) is intentionally narrow: it
    has no ``test``/``train``/``val`` sections — the validator builds those from
    defaults — so it always flows through ``validate_public_config`` for both
    surface and color.

    Color can also be driven by the standalone color configs
    (``conf/config_color_*.yaml``, ``conf/config_inference_color_model_5.yaml``)
    that spell out the full config schema themselves (explicit
    ``test``/``train``/``val`` sections, threshold sweeps). Those bypass the
    validator — it would reject those sections — and go straight to
    ``run_color``.
    """
    if not code_name.startswith("color"):
        return False
    return any(section in cfg for section in ("test", "train", "val"))


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    code_name = str(getattr(cfg, "code", "surface")).lower()

    # The narrow public config (conf/config_public.yaml) is validated and then
    # has defaults merged in. This covers BOTH surface and color: the validator
    # builds a fully-populated, code-specific config (see
    # apply_public_defaults_and_model). The standalone color configs spell out
    # the full schema themselves and bypass the validator (see below).
    if not _is_standalone_color_config(cfg, code_name):
        # Validate BEFORE merging defaults so we fail fast on unsupported fields.
        model_spec = validate_public_config(cfg)
        cfg = apply_public_defaults_and_model(cfg, model_spec)

    torch.backends.cuda.matmul.allow_tf32 = cfg.enable_matmul_tf32
    torch.backends.cudnn.allow_tf32 = cfg.enable_cudnn_tf32

    if code_name in ("surface", "surface_partition"):
        run_surface(cfg)
    elif code_name.startswith("color"):
        run_color(cfg)
    else:
        raise ValueError(
            f"Invalid cfg.code={cfg.code!r} (expected 'surface'/'surface_partition' or 'color*')."
        )


def run_surface(cfg: DictConfig):
    if cfg.workflow.task == "train":
        train_main(cfg)
    elif cfg.workflow.task == "threshold":
        raise ValueError(
            "workflow.task='threshold' has been renamed to workflow.task='inference'. "
            "Please update your config/env var to WORKFLOW=inference."
        )
    elif cfg.workflow.task == "inference":
        from evaluation.inference import run_inference
        DistributedManager.initialize()
        dist = DistributedManager()
        decode_mode = os.environ.get("PREDECODER_DECODE_MODE", "").strip().lower()
        if decode_mode == "pymatching_only":
            model = torch.nn.Identity()
        else:
            model = _load_model(cfg, dist)
        run_inference(model, dist.device, dist, cfg)
    elif cfg.workflow.task == "generate_stim_data":
        from export.generate_test_data import generate_test_data
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        basis_cfg = str(getattr(cfg.test, "meas_basis_test", "both")).upper()
        bases = ["X", "Z"] if basis_cfg in ("BOTH", "MIXED") else [basis_cfg]
        if any(b not in ("X", "Z") for b in bases):
            raise ValueError(f"Invalid test.meas_basis_test={basis_cfg!r}; expected X, Z, or both.")

        num_samples = int(getattr(cfg.test, "num_samples", 1000))
        output_dir = os.path.join(HydraConfig.get().runtime.output_dir, "stim_samples")
        noise_model_cfg = getattr(cfg.data, "noise_model", None)
        noise_model_params = None
        if noise_model_cfg is not None:
            noise_model_params = OmegaConf.to_container(noise_model_cfg, resolve=True)

        # The generate_stim_data workflow ONLY writes Stim sample artifacts
        # (samples_{basis}.dets + metadata_{basis}.json). The CUDA-Q .bin
        # artifacts are produced by a separate workflow (see generate_test_data
        # CLI with --stim-artifacts/--no-cudaq-artifacts) to keep the offline
        # decoding output dir narrowly scoped.
        write_cudaq_artifacts = False

        print(
            "[generate_stim_data] Writing Stim detector samples "
            f"to {output_dir} for basis={bases}, shots={num_samples}"
        )
        for basis in bases:
            generate_test_data(
                distance=int(cfg.distance),
                n_rounds=int(cfg.n_rounds),
                basis=basis,
                p_error=float(getattr(cfg.test, "p_error", 0.003)),
                code_rotation=str(getattr(cfg.data, "code_rotation", "XV")),
                noise_model_params=noise_model_params,
                num_samples=num_samples,
                output_dir=output_dir,
                write_stim_artifacts=True,
                write_cudaq_artifacts=write_cudaq_artifacts,
            )
    elif cfg.workflow.task == "data":
        DistributedManager.initialize()
        dist = DistributedManager()
        train_loader, _ = DatapipeFactory.create_dataloader(cfg, dist.world_size, dist.rank)
        for j, dl in enumerate(train_loader):
            print(f"Batch {j}: syndrome_shape: {dl['syndrome'].shape}")
    elif cfg.workflow.task == "decoder_ablation":
        from evaluation.failure_analysis import decoder_ablation_study
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        decoder_ablation_study(model, dist.device, dist, cfg)
    elif cfg.workflow.task in ("sampling", "visualize"):
        raise ValueError(
            f"workflow.task={cfg.workflow.task!r} is not supported. "
            "Supported workflows: train, inference, decoder_ablation."
        )


def run_color(cfg: DictConfig):
    """
    Color-code workflow runner.

    Supports:
    - train: Training with Chromobius-based LER validation
    - inference: Run inference and compute LER with a trained model
    - threshold: Threshold sweep across multiple p values
    - sdr: Syndrome-density-reduction sweep
    - chromobius_timing: Single-shot chromobius timing sweep

    Driven either by the public config (conf/config_public.yaml with
    `code: color`) or by the standalone color configs (conf/config_color_*.yaml,
    conf/config_inference_color_model_5.yaml), which spell out the full config
    schema (test/train/val sections, threshold sweeps) and bypass the
    public-config validator.
    """
    task = str(getattr(cfg.workflow, "task", "train")).lower()

    if task == "train":
        train_main(cfg)
    elif task == "inference":
        from evaluation.logical_error_rate_color import count_logical_errors_color

        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        _apply_public_inference_env_overrides(cfg)

        if dist.rank == 0:
            print("[Color Code Inference] Running LER computation...")
            print(
                f"[Color Code Inference] Distance: {cfg.test.distance}, Rounds: {cfg.test.n_rounds}"
            )
            print(
                f"[Color Code Inference] p_error: {cfg.test.p_error}, Basis: {cfg.test.meas_basis_test}"
            )

        result = count_logical_errors_color(
            model,
            dist.device,
            dist,
            cfg,
            include_diagnostics=True,
            log_summary=True,
        )

        if dist.rank == 0:
            print("\n" + "=" * 60)
            print("Color Code Inference Results")
            print("=" * 60)
            for basis, data in result.items():
                print(f"\n{basis}-basis:")
                for key, value in data.items():
                    print(f"  {key}: {value}")

            result_path = os.path.join(cfg.output, "inference_results.json")
            os.makedirs(cfg.output, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {result_path}")

    elif task == "threshold":
        from evaluation.color_threshold_results import append_threshold_results
        from evaluation.logical_error_rate_color import count_logical_errors_color

        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        model_checkpoint_path = getattr(cfg, "resolved_model_checkpoint_path", None)
        if model_checkpoint_path is None:
            raise RuntimeError(
                "Color threshold aggregation requires a resolved model checkpoint path"
            )

        settings = _resolve_color_threshold_settings(cfg)

        if dist.rank == 0:
            print("[Color Code Threshold] Running LER count sweep")
            print(f"[Color Code Threshold] distances: {settings['distances']}")
            print(f"[Color Code Threshold] p_values:  {settings['p_values']}")
            print(f"[Color Code Threshold] bases:     {settings['bases']}")
            print(f"[Color Code Threshold] num_samples={settings['num_samples']}")

        rows = []
        original = {
            "distance": getattr(cfg, "distance", None),
            "n_rounds": getattr(cfg, "n_rounds", None),
            "test.distance": getattr(cfg.test, "distance", None),
            "test.n_rounds": getattr(cfg.test, "n_rounds", None),
            "test.p_error": getattr(cfg.test, "p_error", None),
            "test.meas_basis_test": getattr(cfg.test, "meas_basis_test", None),
            "test.num_samples": getattr(cfg.test, "num_samples", None),
        }
        try:
            for d in settings["distances"]:
                n_rounds = settings["n_rounds"][int(d)]
                cfg.distance = int(d)
                cfg.n_rounds = int(n_rounds)
                cfg.test.distance = int(d)
                cfg.test.n_rounds = int(n_rounds)
                cfg.test.num_samples = int(settings["num_samples"])

                for p in settings["p_values"]:
                    cfg.test.p_error = float(p)
                    for basis in settings["bases"]:
                        cfg.test.meas_basis_test = basis
                        if dist.rank == 0:
                            print(
                                f"[Color Code Threshold] Evaluating d={d}, R={n_rounds}, p={p:g}, basis={basis}"
                            )

                        result = count_logical_errors_color(
                            model,
                            dist.device,
                            dist,
                            cfg,
                            include_diagnostics=False,
                            log_summary=False,
                        )
                        data = result[basis]
                        row = {
                            "distance": int(d),
                            "n_rounds": int(n_rounds),
                            "p": float(p),
                            "basis": basis,
                            "logical_errors": int(data["logical_errors"]),
                            "num_shots": int(data["num_shots"]),
                            "chromobius_errors": int(data["chromobius_errors"]),
                        }
                        rows.append(row)
                        if dist.rank == 0:
                            print(
                                "[Color Code Threshold] "
                                f"d={d}, R={n_rounds}, p={p:g}, basis={basis}: "
                                f"PD+Chromobius {row['logical_errors']}/{row['num_shots']} logical errors, "
                                f"Chromobius {row['chromobius_errors']}/{row['num_shots']} logical errors"
                            )
        finally:
            cfg.distance = original["distance"]
            cfg.n_rounds = original["n_rounds"]
            cfg.test.distance = original["test.distance"]
            cfg.test.n_rounds = original["test.n_rounds"]
            cfg.test.p_error = original["test.p_error"]
            cfg.test.meas_basis_test = original["test.meas_basis_test"]
            cfg.test.num_samples = original["test.num_samples"]

        if dist.rank == 0:
            result_path, _ = append_threshold_results(cfg, rows, model_checkpoint_path)
            print(f"[Color Code Threshold] Aggregated results saved to: {result_path}")

    elif task == "sdr":
        from evaluation.color_sdr_results import append_sdr_results
        from evaluation.logical_error_rate_color import compute_syndrome_density_reduction_color

        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        model_checkpoint_path = getattr(cfg, "resolved_model_checkpoint_path", None)
        if model_checkpoint_path is None:
            raise RuntimeError("Color SDR aggregation requires a resolved model checkpoint path")

        settings = _resolve_color_threshold_settings(cfg)

        if dist.rank == 0:
            print("[Color Code SDR] Running syndrome-density reduction sweep")
            print(f"[Color Code SDR] distances: {settings['distances']}")
            print(f"[Color Code SDR] p_values:  {settings['p_values']}")
            print(f"[Color Code SDR] bases:     {settings['bases']}")
            print(f"[Color Code SDR] num_samples={settings['num_samples']}")

        rows = []
        original = {
            "distance": getattr(cfg, "distance", None),
            "n_rounds": getattr(cfg, "n_rounds", None),
            "test.distance": getattr(cfg.test, "distance", None),
            "test.n_rounds": getattr(cfg.test, "n_rounds", None),
            "test.p_error": getattr(cfg.test, "p_error", None),
            "test.meas_basis_test": getattr(cfg.test, "meas_basis_test", None),
            "test.num_samples": getattr(cfg.test, "num_samples", None),
        }
        try:
            for d in settings["distances"]:
                n_rounds = settings["n_rounds"][int(d)]
                cfg.distance = int(d)
                cfg.n_rounds = int(n_rounds)
                cfg.test.distance = int(d)
                cfg.test.n_rounds = int(n_rounds)
                cfg.test.num_samples = int(settings["num_samples"])

                for p in settings["p_values"]:
                    cfg.test.p_error = float(p)
                    for basis in settings["bases"]:
                        cfg.test.meas_basis_test = basis
                        if dist.rank == 0:
                            print(
                                f"[Color Code SDR] Evaluating d={d}, R={n_rounds}, p={p:g}, basis={basis}"
                            )

                        result = compute_syndrome_density_reduction_color(
                            model, dist.device, dist, cfg
                        )
                        row = {
                            "distance": int(d),
                            "n_rounds": int(n_rounds),
                            "p": float(p),
                            "basis": basis,
                            "input_syndrome_ones": int(result["input_syndrome_ones"]),
                            "residual_syndrome_ones": int(result["residual_syndrome_ones"]),
                            "syndrome_elements": int(result["syndrome_elements"]),
                        }
                        rows.append(row)
                        if dist.rank == 0:
                            print(
                                "[Color Code SDR] "
                                f"d={d}, R={n_rounds}, p={p:g}, basis={basis}: "
                                f"input={result['input_syndrome_density']:.6g}, "
                                f"residual={result['residual_syndrome_density']:.6g}, "
                                f"SDR={result['reduction_factor']:.6g}"
                            )
        finally:
            cfg.distance = original["distance"]
            cfg.n_rounds = original["n_rounds"]
            cfg.test.distance = original["test.distance"]
            cfg.test.n_rounds = original["test.n_rounds"]
            cfg.test.p_error = original["test.p_error"]
            cfg.test.meas_basis_test = original["test.meas_basis_test"]
            cfg.test.num_samples = original["test.num_samples"]

        if dist.rank == 0:
            result_path, _ = append_sdr_results(cfg, rows, model_checkpoint_path)
            print(f"[Color Code SDR] Aggregated results saved to: {result_path}")

    elif task == "chromobius_timing":
        from evaluation.color_chromobius_timing_results import append_chromobius_timing_results
        from evaluation.logical_error_rate_color import compute_chromobius_single_shot_timing_color

        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        model_checkpoint_path = getattr(cfg, "resolved_model_checkpoint_path", None)
        if model_checkpoint_path is None:
            raise RuntimeError(
                "Color Chromobius timing aggregation requires a resolved model checkpoint path"
            )

        settings = _resolve_color_threshold_settings(cfg)

        if dist.rank == 0:
            print("[Color Code Chromobius Timing] Running single-shot timing sweep")
            print(f"[Color Code Chromobius Timing] distances: {settings['distances']}")
            print(f"[Color Code Chromobius Timing] p_values:  {settings['p_values']}")
            print(f"[Color Code Chromobius Timing] bases:     {settings['bases']}")
            print(f"[Color Code Chromobius Timing] num_samples={settings['num_samples']}")

        rows = []
        original = {
            "distance": getattr(cfg, "distance", None),
            "n_rounds": getattr(cfg, "n_rounds", None),
            "test.distance": getattr(cfg.test, "distance", None),
            "test.n_rounds": getattr(cfg.test, "n_rounds", None),
            "test.p_error": getattr(cfg.test, "p_error", None),
            "test.meas_basis_test": getattr(cfg.test, "meas_basis_test", None),
            "test.num_samples": getattr(cfg.test, "num_samples", None),
        }
        try:
            for d in settings["distances"]:
                n_rounds = settings["n_rounds"][int(d)]
                cfg.distance = int(d)
                cfg.n_rounds = int(n_rounds)
                cfg.test.distance = int(d)
                cfg.test.n_rounds = int(n_rounds)
                cfg.test.num_samples = int(settings["num_samples"])

                for p in settings["p_values"]:
                    cfg.test.p_error = float(p)
                    for basis in settings["bases"]:
                        cfg.test.meas_basis_test = basis
                        if dist.rank == 0:
                            print(
                                "[Color Code Chromobius Timing] "
                                f"Evaluating d={d}, R={n_rounds}, p={p:g}, basis={basis}"
                            )

                        timing = compute_chromobius_single_shot_timing_color(
                            model, dist.device, dist, cfg
                        )

                        row = {
                            "distance": int(d),
                            "n_rounds": int(n_rounds),
                            "p": float(p),
                            "basis": basis,
                            "original_syndromes": timing["original_syndromes"],
                            "residual_syndromes": timing["residual_syndromes"],
                        }
                        rows.append(row)
                        if dist.rank == 0:
                            original_stats = timing["original_syndromes"]
                            residual_stats = timing["residual_syndromes"]
                            print(
                                "[Color Code Chromobius Timing] "
                                f"d={d}, R={n_rounds}, p={p:g}, basis={basis}: "
                                f"original avg={original_stats['avg_us_per_round']:.6g} us/round "
                                f"({original_stats['shots']} shots), "
                                f"residual avg={residual_stats['avg_us_per_round']:.6g} us/round "
                                f"({residual_stats['shots']} shots)"
                            )
        finally:
            cfg.distance = original["distance"]
            cfg.n_rounds = original["n_rounds"]
            cfg.test.distance = original["test.distance"]
            cfg.test.n_rounds = original["test.n_rounds"]
            cfg.test.p_error = original["test.p_error"]
            cfg.test.meas_basis_test = original["test.meas_basis_test"]
            cfg.test.num_samples = original["test.num_samples"]

        if dist.rank == 0:
            result_path, _ = append_chromobius_timing_results(cfg, rows, model_checkpoint_path)
            print(f"[Color Code Chromobius Timing] Aggregated results saved to: {result_path}")

    else:
        raise NotImplementedError(
            f"Color-code workflow.task={cfg.workflow.task!r} is not supported. "
            "Supported tasks: 'train', 'inference', 'threshold', 'sdr', 'chromobius_timing'."
        )


def find_best_model(path, *, rank: int = 0):
    if rank == 0:
        print(f"Searching for best model in: {path}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    max_value = -1  # Start with -1 to include epoch 0
    best_file = None
    model_files = []
    # Named .pt files without epoch numbers (e.g. Ising-Decoder-SurfaceCode-1-Fast.pt)
    named_pt_files = []

    for filename in os.listdir(path):
        if not filename.endswith(".pt"):
            continue
        if filename.startswith("PreDecoderModelMemory_"):
            try:
                value = float(filename.split(".")[2])  # Gets epoch number
                model_files.append((filename, value))
                if value > max_value:
                    max_value = value
                    best_file = filename
            except (IndexError, ValueError) as e:
                print(f"Warning: could not parse epoch from filename {filename}: {e}")
        else:
            named_pt_files.append(filename)

    # Fall back to named .pt files when no epoch-numbered checkpoints are present
    if best_file is None and named_pt_files:
        named_pt_files.sort()
        best_file = named_pt_files[-1]
        model_files = [(f, None) for f in named_pt_files]

    if rank == 0:
        print(f"Found {len(model_files)} model file(s):")
        for filename, epoch in sorted(model_files, key=lambda x: (x[1] is None, x[1] or 0)):
            marker = "*" if filename == best_file else " "
            epoch_str = str(epoch) if epoch is not None else "n/a"
            print(f"  [{marker}] {filename} (epoch {epoch_str})")

    if best_file is None:
        raise FileNotFoundError(
            f"No valid model checkpoint files found in {path}\n"
            f"Expected .pt files (e.g. Ising-Decoder-SurfaceCode-1-Fast.pt or "
            f"PreDecoderModelMemory_*.pt).\n"
            f"Hint: download the pretrained weights and place them in this directory, "
            f"or set model_checkpoint_file in your config to an explicit path."
        )

    best_model_path = os.path.join(path, best_file)
    if rank == 0:
        epoch_str = str(max_value) if max_value >= 0 else "n/a"
        print(f"Selected best model: {best_file} (epoch {epoch_str})")

    return best_model_path


def _resolve_dir(path: str) -> str:
    """Return an absolute version of path, resolving relative paths from the repo root."""
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, path)


def _load_state_dict_from_pt(model_path: str, device) -> dict:
    """Load a state dict from a .pt checkpoint, handling multiple saved formats.

    Supports:
    - bare state dict (keys are layer names)
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    Also strips the DDP "module." prefix if present.
    """
    raw = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: expected a dict, got {type(raw).__name__}")
    return {
        (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }


def _load_model(cfg, dist):
    if dist.rank == 0:
        print(f"Loading model for task: {cfg.workflow.task}")

    _ensure_inference_io_channels(cfg)

    # SafeTensors path: load fp16/fp32 model from SafeTensors file
    safetensors_path = os.environ.get("PREDECODER_SAFETENSORS_CHECKPOINT", "").strip()
    if safetensors_path:
        from export.safetensors_utils import load_safetensors
        if dist.rank == 0:
            print(f"Loading model from SafeTensors: {safetensors_path}")

        # Auto-detect model_id from SafeTensors metadata (don't override with config)
        model, metadata = load_safetensors(
            safetensors_path,
            model_id=None,
            device=str(dist.device),
        )
        if dist.rank == 0:
            loaded_model_id = metadata.get("model_id", "unknown")
            dtype = metadata.get("quant_format", "fp32")
            receptive_field = metadata.get("receptive_field", "unknown")
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  model_id: {loaded_model_id} (from SafeTensors metadata)")
            print(f"  receptive_field: {receptive_field}")
            print(f"  dtype: {dtype}")
            print(f"  parameters: {param_count:,}")

            # Warn if config model_id doesn't match file metadata
            config_model_id = getattr(cfg, "model_id", None)
            if config_model_id is not None and str(config_model_id) != str(loaded_model_id):
                print(
                    f"  Warning: config model_id={config_model_id} differs from "
                    f"file model_id={loaded_model_id}; using {loaded_model_id}"
                )

        if metadata.get("quant_format") == "fp16":
            cfg.enable_fp16 = True
        _record_resolved_model_path(cfg, safetensors_path)
        return model

    # Direct file path override (for named pretrained models without epoch numbers)
    model_checkpoint_file = getattr(cfg, 'model_checkpoint_file', None)
    if model_checkpoint_file:
        model_checkpoint_file = _resolve_dir(str(model_checkpoint_file))
        if not os.path.exists(model_checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint_file}")
        if dist.rank == 0:
            print(f"Loading model from: {model_checkpoint_file}")
        model = ModelFactory.create_model(cfg).to(dist.device)
        if cfg.enable_fp16:
            model = model.half()
        state_dict = _load_state_dict_from_pt(model_checkpoint_file, dist.device)
        model.load_state_dict(state_dict)
        _record_resolved_model_path(cfg, model_checkpoint_file)
        if dist.rank == 0:
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model loaded ({param_count:,} parameters)")
        return model

    model = ModelFactory.create_model(cfg).to(dist.device)

    if cfg.enable_fp16:
        model = model.half()
        if dist.rank == 0:
            print("Model converted to float16 for fp16 inference")

    # Determine model directory
    # Priority: 1) model_checkpoint_dir (for inference configs)
    #           2) cfg.output/models (for training configs)
    model_checkpoint_dir = getattr(cfg, 'model_checkpoint_dir', None)
    use_checkpoint = getattr(cfg.test, 'use_model_checkpoint', -1)

    if use_checkpoint == -1:
        model_dir = _resolve_dir(
            os.path.join(model_checkpoint_dir, "best_model")
            if model_checkpoint_dir else f"{cfg.output}/models/best_model"
        )
        if dist.rank == 0:
            print(f"Loading best model from: {model_dir}")

        # Fallback: older runs may not have a best_model/ folder
        if not os.path.isdir(model_dir):
            fallback_dir = _resolve_dir(
                model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
            )
            if dist.rank == 0:
                print(f"best_model/ not found; falling back to: {fallback_dir}")
            model_dir = fallback_dir

        model_path = find_best_model(model_dir, rank=dist.rank)
    else:
        checkpoint_dir = _resolve_dir(
            model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
        )
        if dist.rank == 0:
            print(f"Loading checkpoint {use_checkpoint} from: {checkpoint_dir}")

        # Prefer any PreDecoderModelMemory_* file ending with .0.{use_checkpoint}.pt
        target_suffix = f".0.{use_checkpoint}.pt"
        checkpoint_filename = None
        try:
            for f in os.listdir(checkpoint_dir):
                if f.startswith("PreDecoderModelMemory_") and f.endswith(target_suffix):
                    checkpoint_filename = f
                    break
        except OSError:
            pass
        if checkpoint_filename is None:
            checkpoint_filename = f"PreDecoderModelMemory_v1.0.{use_checkpoint}.pt"
        model_path = os.path.join(checkpoint_dir, checkpoint_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if dist.rank == 0:
        print(f"Loading model parameters from: {model_path}")

    state_dict = _load_state_dict_from_pt(model_path, dist.device)
    model.load_state_dict(state_dict)
    _record_resolved_model_path(cfg, model_path)

    if dist.rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model loaded ({param_count:,} parameters)")

    return model


if __name__ == "__main__":
    run()
