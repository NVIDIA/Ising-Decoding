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
"""Export color-code model-only ONNX files for TensorRT timing.

This helper measures only the neural predecoder model from an already-built
``trainX`` tensor input to model logits. It is useful for isolating model
latency from color-code detector preprocessing, postprocessing, residual
packing, and Chromobius/global decoding.

Use ``export_detector_input_trtexec.py`` when the timing target should include
the tensorized detector-input preprocessing and residual detector assembly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _install_optional_import_stubs() -> None:
    """Keep this standalone in lean benchmark/export environments."""
    try:
        import physicsnemo  # noqa: F401
    except ImportError:
        physicsnemo = types.ModuleType("physicsnemo")
        distributed = types.ModuleType("physicsnemo.distributed")

        class DistributedManager:
            pass

        distributed.DistributedManager = DistributedManager
        sys.modules.setdefault("physicsnemo", physicsnemo)
        sys.modules.setdefault("physicsnemo.distributed", distributed)


def _repo_code_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_cfg(args: argparse.Namespace, n_rows: int, n_cols: int) -> SimpleNamespace:
    return SimpleNamespace(
        code="color",
        distance=int(args.distance),
        n_rounds=int(args.rounds),
        enable_fp16=False,
        model=SimpleNamespace(
            version="predecoder_memory_v1",
            dropout_p=float(args.dropout),
            activation=str(args.activation),
            num_filters=[int(v) for v in args.filters.split(",")],
            kernel_size=[int(v) for v in args.kernel_sizes.split(",")],
            input_channels=4,
            out_channels=4,
        ),
        benchmark=SimpleNamespace(
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            batch_size=int(args.batch_size),
        ),
    )


def _load_checkpoint_if_requested(model: torch.nn.Module, checkpoint: str | None) -> str:
    if not checkpoint:
        return "random_initialization"

    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    clean_state = {}
    for key, value in payload.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        clean_state[clean_key] = value
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not match model architecture: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return str(checkpoint_path)


def _export_onnx(
    model: torch.nn.Module,
    example: torch.Tensor,
    output_path: Path,
    *,
    opset: int,
    dynamic_batch: bool,
) -> None:
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "trainX": {
                0: "batch"
            },
            "logits": {
                0: "batch"
            },
        }

    torch.onnx.export(
        model,
        example,
        output_path,
        opset_version=int(opset),
        input_names=["trainX"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )


def _quantize_fp8(
    fp32_path: Path,
    output_path: Path,
    example_shape: tuple[int, ...],
    calibration_samples: int,
) -> None:
    import modelopt.onnx.quantization as mq

    calibration = np.random.default_rng(1234).integers(
        0,
        2,
        size=(int(calibration_samples),) + tuple(example_shape[1:]),
        dtype=np.int32,
    ).astype(np.float32)
    mq.quantize(
        onnx_path=str(fp32_path),
        quantize_mode="fp8",
        calibration_data={"trainX": calibration},
        output_path=str(output_path),
        op_types_to_quantize=["Conv"],
        high_precision_dtype="fp16",
    )


def _trtexec_command(
    onnx_path: Path,
    output_dir: Path,
    precision: str,
    *,
    docker_image: str | None,
    warmup_ms: int,
    duration_s: int,
    iterations: int,
    avg_runs: int,
) -> list[str]:
    basename = onnx_path.name
    stem = onnx_path.stem
    if precision == "fp8-quantized" and stem.endswith(".fp8"):
        stem = stem[:-len(".fp8")]
    timing_name = f"{stem}.{precision}.times.json"
    engine_name = f"{stem}.{precision}.engine"
    common = [
        "trtexec",
        f"--onnx=/workspace/{basename}",
        "--useCudaGraph",
        "--noDataTransfers",
        "--useSpinWait",
        f"--warmUp={int(warmup_ms)}",
        f"--duration={int(duration_s)}",
        f"--iterations={int(iterations)}",
        f"--avgRuns={int(avg_runs)}",
        f"--exportTimes=/workspace/{timing_name}",
        f"--saveEngine=/workspace/{engine_name}",
    ]
    if precision == "fp16":
        common.insert(2, "--fp16")
    elif precision == "fp8-direct":
        common.insert(2, "--fp8")
    elif precision == "fp8-quantized":
        common.insert(2, "--stronglyTyped")
    elif precision != "fp32":
        raise ValueError(f"Unknown precision: {precision}")

    if docker_image:
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{output_dir.resolve()}:/workspace",
            docker_image,
            *common,
        ]
    return common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=13)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/color-paper-trtexec"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--activation", default="gelu", choices=["gelu", "relu", "leakyrelu"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--filters", default="256,256,256,256,256,4")
    parser.add_argument("--kernel-sizes", default="3,3,3,3,3,3")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--quantize-fp8", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--docker-image", default="nvcr.io/nvidia/tensorrt:25.03-py3")
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument(
        "--duration-s",
        type=int,
        default=0,
        help="trtexec measurement duration. 0 keeps the run iteration-count driven.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--avg-runs", type=int, default=100)
    args = parser.parse_args()

    _install_optional_import_stubs()
    sys.path.insert(0, str(_repo_code_dir()))

    from model.factory import ModelFactory
    from qec.color_code.color_code import ColorCode

    code = ColorCode(args.distance)
    n_rows = int(code.n_rows)
    n_cols = int(code.n_cols)
    cfg = _make_cfg(args, n_rows, n_cols)

    model = ModelFactory.create_model(cfg)
    checkpoint_used = _load_checkpoint_if_requested(model, args.checkpoint)
    model.eval()

    example_shape = (int(args.batch_size), 4, int(args.rounds), n_rows, n_cols)
    example = torch.randint(0, 2, example_shape, dtype=torch.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"color-model5-model-only-d{args.distance}-t{args.rounds}"
            f"-b{args.batch_size}")
    fp32_path = args.output_dir / f"{stem}.onnx"
    _export_onnx(
        model,
        example,
        fp32_path,
        opset=int(args.opset),
        dynamic_batch=bool(args.dynamic_batch),
    )

    quantized_path = None
    quantization_status = "not_requested"
    if args.quantize_fp8:
        quantized_path = args.output_dir / f"{stem}.fp8.onnx"
        try:
            _quantize_fp8(
                fp32_path,
                quantized_path,
                example_shape,
                int(args.calibration_samples),
            )
            quantization_status = "ok"
        except Exception as exc:
            quantized_path = None
            quantization_status = f"failed: {type(exc).__name__}: {exc}"

    trtexec = {
        "fp32":
            _trtexec_command(
                fp32_path,
                args.output_dir,
                "fp32",
                docker_image=args.docker_image,
                warmup_ms=args.warmup_ms,
                duration_s=args.duration_s,
                iterations=args.iterations,
                avg_runs=args.avg_runs,
            ),
        "fp16":
            _trtexec_command(
                fp32_path,
                args.output_dir,
                "fp16",
                docker_image=args.docker_image,
                warmup_ms=args.warmup_ms,
                duration_s=args.duration_s,
                iterations=args.iterations,
                avg_runs=args.avg_runs,
            ),
        "fp8_direct_unquantized":
            _trtexec_command(
                fp32_path,
                args.output_dir,
                "fp8-direct",
                docker_image=args.docker_image,
                warmup_ms=args.warmup_ms,
                duration_s=args.duration_s,
                iterations=args.iterations,
                avg_runs=args.avg_runs,
            ),
    }
    if quantized_path is not None:
        trtexec["fp8_quantized"] = _trtexec_command(
            quantized_path,
            args.output_dir,
            "fp8-quantized",
            docker_image=args.docker_image,
            warmup_ms=args.warmup_ms,
            duration_s=args.duration_s,
            iterations=args.iterations,
            avg_runs=args.avg_runs,
        )

    manifest = {
        "experiment": "paper_style_color_model_only_trtexec",
        "distance": int(args.distance),
        "rounds": int(args.rounds),
        "batch_size": int(args.batch_size),
        "input_shape": list(example_shape),
        "output_shape": list(model(example).shape),
        "color_grid":
            {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "spacetime_cells": int(args.rounds) * n_rows * n_cols,
            },
        "model":
            {
                "version": cfg.model.version,
                "filters": cfg.model.num_filters,
                "kernel_sizes": cfg.model.kernel_size,
                "activation": cfg.model.activation,
                "dropout": cfg.model.dropout_p,
                "checkpoint": checkpoint_used,
            },
        "onnx":
            {
                "fp32_path": str(fp32_path),
                "fp8_quantized_path": str(quantized_path) if quantized_path is not None else None,
                "opset": int(args.opset),
                "dynamic_batch": bool(args.dynamic_batch),
                "quantization_status": quantization_status,
            },
        "paper_style_trtexec_flags":
            [
                "--useCudaGraph",
                "--noDataTransfers",
                "--useSpinWait",
                f"--warmUp={int(args.warmup_ms)}",
                f"--duration={int(args.duration_s)}",
                f"--iterations={int(args.iterations)}",
                f"--avgRuns={int(args.avg_runs)}",
            ],
        "trtexec_commands": trtexec,
    }
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
