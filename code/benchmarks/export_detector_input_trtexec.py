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
"""Export detector-input predecoder ONNX models for TensorRT timing.

The ONNX input is a flattened detector vector named ``dets``. This lets us time
the tensorized predecoder interface that starts from decoder-style detector
bits, rather than from already-built model tensors.

Export modes:
- ``preprocess``: detector vector -> model input tensor ``trainX``.
- ``logits``: detector vector -> ``trainX`` -> Conv3D model logits.
- ``residual``: detector vector -> logical-frame bit plus residual detector
  vector, using the same color-code preprocessing transform as the production
  datapipe.

Chromobius/global decoding stays outside this TensorRT comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _install_optional_import_stubs() -> None:
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


def _make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    code_name = str(args.code).lower()
    if args.filters is not None:
        filters = [int(v) for v in str(args.filters).split(",")]
    elif code_name == "color":
        filters = [256, 256, 256, 256, 256, 4]
    else:
        filters = [128, 128, 128, 4]
    if args.kernel_sizes is not None:
        kernel_sizes = [int(v) for v in str(args.kernel_sizes).split(",")]
    else:
        kernel_sizes = [3] * len(filters)

    return SimpleNamespace(
        code=code_name,
        distance=int(args.distance),
        n_rounds=int(args.rounds),
        enable_fp16=False,
        model=SimpleNamespace(
            version="predecoder_memory_v1",
            dropout_p=float(args.dropout),
            activation=str(args.activation),
            num_filters=filters,
            kernel_size=kernel_sizes,
            input_channels=4,
            out_channels=4,
        ),
        test=SimpleNamespace(
            th_data=0.0,
            th_syn=0.0,
            sampling_mode="threshold",
            temperature=1.0,
            temperature_data=1.0,
            temperature_syn=1.0,
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


class DetectorInputModel(torch.nn.Module):

    def __init__(self, input_transform: torch.nn.Module, model: torch.nn.Module):
        super().__init__()
        self.input_transform = input_transform
        self.model = model

    def build_train_x(
        self,
        dets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_transform.build_train_x(dets)

    def forward(self, dets: torch.Tensor) -> torch.Tensor:
        train_x, _, _, _ = self.build_train_x(dets)
        return self.model(train_x.contiguous())


class DetectorInputPreprocess(torch.nn.Module):

    def __init__(self, detector_module: DetectorInputModel):
        super().__init__()
        self.detector_module = detector_module

    def forward(self, dets: torch.Tensor) -> torch.Tensor:
        train_x, _, _, _ = self.detector_module.build_train_x(dets)
        return train_x.contiguous()


class DetectorInputColorEval(torch.nn.Module):

    def __init__(
        self,
        detector_module: DetectorInputModel,
        eval_module: torch.nn.Module,
    ):
        super().__init__()
        self.detector_module = detector_module
        self.eval_module = eval_module

    def forward(self, dets: torch.Tensor) -> torch.Tensor:
        train_x, x_syn, z_syn, boundary = self.detector_module.build_train_x(dets)
        return self.eval_module(
            train_x.contiguous(),
            x_syn.to(dtype=torch.int32),
            z_syn.to(dtype=torch.int32),
            boundary.to(dtype=torch.int32),
        )


def _build_detector_module(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[DetectorInputModel, dict]:
    code_name = str(args.code).lower()
    basis = str(args.basis).upper()
    distance = int(args.distance)
    rounds = int(args.rounds)

    if code_name == "surface":
        from qec.surface_code.detector_input import SurfaceDetectorInputTransform

        input_transform = SurfaceDetectorInputTransform(
            distance=distance,
            basis=basis,
            rounds=rounds,
            rotation=str(args.rotation),
            preprocess_strategy=str(args.preprocess_strategy),
        )
        height = int(input_transform.height)
        width = int(input_transform.width)
        num_stabs = int(input_transform.num_stabs)
        num_data = int(input_transform.num_data)
    elif code_name == "color":
        from qec.color_code.detector_input import ColorDetectorInputTransform

        input_transform = ColorDetectorInputTransform(
            distance=distance,
            rounds=rounds,
            basis=basis,
            preprocess_strategy=str(args.preprocess_strategy),
        )
        height = int(input_transform.height)
        width = int(input_transform.width)
        num_stabs = int(input_transform.num_stabs)
        num_data = int(input_transform.num_data)
    else:
        raise ValueError(f"Unsupported code: {args.code!r}")

    module = DetectorInputModel(input_transform, model)
    output_shape = [int(args.batch_size), 4, rounds, height, width]
    metadata = {
        "height": height,
        "width": width,
        "num_stabs": num_stabs,
        "num_data": num_data,
        "detector_width": int(input_transform.detector_width),
        "input_shape": [int(args.batch_size),
                        int(input_transform.detector_width)],
        "output_shape": output_shape,
    }
    return module, metadata


def _build_color_residual_module(
    detector_module: DetectorInputModel,
    model: torch.nn.Module,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    metadata: dict,
) -> DetectorInputColorEval:
    from evaluation.logical_error_rate_color import (
        PreDecoderColorEvalModule,
        _build_color_code_parity_maps,
    )

    maps = _build_color_code_parity_maps(int(args.distance))
    obs_support = torch.zeros(int(maps["num_data"]), dtype=torch.float32)
    obs_support[::2] = 1.0
    eval_module = PreDecoderColorEvalModule(
        model,
        cfg,
        maps,
        basis=str(args.basis),
        obs_support=obs_support,
        num_boundary_dets=int(metadata["num_stabs"]),
        enable_delta_s2_correction=False,
        enable_z_ff=True,
    )
    return DetectorInputColorEval(detector_module, eval_module)


def _export_onnx(
    module: torch.nn.Module,
    example: torch.Tensor,
    output_path: Path,
    *,
    opset: int,
    output_name: str,
) -> None:
    torch.onnx.export(
        module,
        example,
        output_path,
        opset_version=int(opset),
        input_names=["dets"],
        output_names=[output_name],
        dynamic_axes={
            "dets": {
                0: "batch"
            },
            output_name: {
                0: "batch"
            },
        },
        do_constant_folding=True,
        dynamo=False,
    )


def _validate_onnx_export(
    module: torch.nn.Module,
    example: torch.Tensor,
    onnx_path: Path,
    output_name: str,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "--validate-onnx requires onnxruntime in the current Python environment"
        ) from exc

    with torch.no_grad():
        expected = module(example).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run([output_name], {"dets": example.detach().cpu().numpy()})[0]
    max_abs_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    is_close = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-4))
    result = {
        "status": "ok" if is_close else "failed",
        "max_abs_diff": max_abs_diff,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
    }
    if not is_close:
        raise RuntimeError(f"ONNX validation failed: {result}")
    return result


def _quantize_fp8(
    fp32_path: Path,
    output_path: Path,
    detector_width: int,
    calibration_samples: int,
) -> None:
    import modelopt.onnx.quantization as mq

    calibration = np.random.default_rng(1234).integers(
        0,
        2,
        size=(int(calibration_samples), int(detector_width)),
        dtype=np.int32,
    ).astype(np.float32)
    mq.quantize(
        onnx_path=str(fp32_path),
        quantize_mode="fp8",
        calibration_data={"dets": calibration},
        output_path=str(output_path),
        op_types_to_quantize=["Conv"],
        high_precision_dtype="fp16",
    )


def _trtexec_commands(
    onnx_path: Path,
    quantized_path: Path | None,
    output_dir: Path,
    detector_width: int,
    docker_image: str | None,
) -> dict:
    common = [
        "trtexec",
        f"--shapes=dets:1x{int(detector_width)}",
        "--duration=1",
        "--iterations=100",
        "--warmUp=5",
        "--builderOptimizationLevel=5",
        "--useCudaGraph=true",
        "--noDataTransfers",
    ]

    def wrap(parts: list[str]) -> list[str]:
        if not docker_image:
            return parts
        return [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{output_dir.resolve()}:/workspace",
            docker_image,
            *parts,
        ]

    commands = {
        "fp16_unquantized":
            wrap([
                "trtexec",
                f"--onnx=/workspace/{onnx_path.name}",
                "--fp16",
                *common[1:],
            ]),
        "fp8_direct_unquantized":
            wrap(
                [
                    "trtexec",
                    f"--onnx=/workspace/{onnx_path.name}",
                    "--fp16",
                    "--fp8",
                    *common[1:],
                ]
            ),
    }
    if quantized_path is not None:
        commands["fp8_quantized_exact_style"] = wrap(
            [
                "trtexec",
                f"--onnx=/workspace/{quantized_path.name}",
                "--fp16",
                "--fp8",
                *common[1:],
            ]
        )
        commands["fp8_quantized_strongly_typed"] = wrap(
            [
                "trtexec",
                f"--onnx=/workspace/{quantized_path.name}",
                "--stronglyTyped",
                *common[1:],
            ]
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", choices=["surface", "color"], required=True)
    parser.add_argument("--distance", type=int, default=13)
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--basis", choices=["X", "Z"], default="Z")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/detector-input-trtexec"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rotation", default="XV")
    parser.add_argument("--activation", default="gelu", choices=["gelu", "relu", "leakyrelu"])
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--filters", default=None)
    parser.add_argument("--kernel-sizes", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--output-mode",
        choices=["preprocess", "logits", "residual"],
        default="logits",
    )
    parser.add_argument(
        "--preprocess-strategy",
        choices=["dense_matmul", "gather"],
        default="dense_matmul",
        help="Implementation used to place syndrome detectors onto the model grid.",
    )
    parser.add_argument("--quantize-fp8", action="store_true")
    parser.add_argument(
        "--validate-onnx",
        action="store_true",
        help="Run the exported FP32 ONNX with onnxruntime and compare it to PyTorch.",
    )
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--docker-image", default="nvcr.io/nvidia/tensorrt:25.03-py3")
    args = parser.parse_args()

    _install_optional_import_stubs()
    sys.path.insert(0, str(_repo_code_dir()))

    from model.factory import ModelFactory

    cfg = _make_cfg(args)
    model = ModelFactory.create_model(cfg)
    checkpoint_used = _load_checkpoint_if_requested(model, args.checkpoint)
    model.eval()

    module, metadata = _build_detector_module(model, args)
    output_name = "logits"
    if args.output_mode == "preprocess":
        module = DetectorInputPreprocess(module)
        output_name = "trainX"
    elif args.output_mode == "residual":
        if str(args.code).lower() != "color":
            raise ValueError("--output-mode residual is currently implemented for --code color")
        module = _build_color_residual_module(module, model, cfg, args, metadata)
        output_name = "L_and_residual_dets"
        metadata["output_shape"] = [
            int(args.batch_size),
            1 + int(metadata["detector_width"]),
        ]
    module.eval()

    detector_width = int(metadata["detector_width"])
    example = torch.randint(
        0,
        2,
        (int(args.batch_size), detector_width),
        dtype=torch.float32,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem_kind = "model" if args.output_mode == "logits" else str(args.output_mode)
    stem = (
        f"{args.code}-detector-input-{stem_kind}-{args.preprocess_strategy}-d{args.distance}"
        f"-t{args.rounds}-{args.basis.lower()}"
    )
    fp32_path = args.output_dir / f"{stem}.onnx"
    _export_onnx(module, example, fp32_path, opset=int(args.opset), output_name=output_name)
    onnx_validation = {"status": "not_requested"}
    if args.validate_onnx:
        onnx_validation = _validate_onnx_export(module, example, fp32_path, output_name)

    quantized_path = None
    quantization_status = "not_requested"
    if args.quantize_fp8 and args.output_mode != "preprocess":
        quantized_path = args.output_dir / f"{stem}.fp8.onnx"
        try:
            _quantize_fp8(
                fp32_path,
                quantized_path,
                detector_width,
                int(args.calibration_samples),
            )
            quantization_status = "ok"
        except Exception as exc:
            quantized_path = None
            quantization_status = f"failed: {type(exc).__name__}: {exc}"
    elif args.quantize_fp8:
        quantization_status = "skipped: preprocess mode has no Conv nodes to quantize"

    manifest = {
        "experiment":
            "detector_input_trtexec",
        "code":
            str(args.code),
        "distance":
            int(args.distance),
        "rounds":
            int(args.rounds),
        "basis":
            str(args.basis),
        "output_mode":
            str(args.output_mode),
        "output_name":
            output_name,
        "preprocess_strategy":
            str(args.preprocess_strategy),
        "global_decoder":
            "excluded",
        "checkpoint":
            checkpoint_used,
        "model":
            {
                "version": cfg.model.version,
                "filters": cfg.model.num_filters,
                "kernel_sizes": cfg.model.kernel_size,
                "activation": cfg.model.activation,
                "dropout": cfg.model.dropout_p,
            },
        "included_stages":
            (
                [
                    "detector_vector_to_grid_trainX",
                ] if args.output_mode == "preprocess" else [
                    "detector_vector_to_grid_trainX",
                    "cnn_forward",
                ] if args.output_mode == "logits" else [
                    "detector_vector_to_grid_trainX",
                    "cnn_forward",
                    "threshold_sampling",
                    "parity_reconstruction",
                    "logical_frame",
                    "residual_detector_assembly",
                    "boundary_detector_append",
                ]
            ),
        "metadata":
            metadata,
        "onnx":
            {
                "fp32_path": str(fp32_path),
                "fp8_quantized_path": str(quantized_path) if quantized_path is not None else None,
                "opset": int(args.opset),
                "validation": onnx_validation,
                "quantization_status": quantization_status,
            },
        "reference_style_flags":
            [
                f"--shapes=dets:1x{detector_width}",
                "--duration=1",
                "--iterations=100",
                "--warmUp=5",
                "--builderOptimizationLevel=5",
                "--useCudaGraph=true",
                "--fp16",
                "--fp8",
                "--noDataTransfers",
            ],
        "trtexec_commands":
            _trtexec_commands(
                fp32_path,
                quantized_path,
                args.output_dir,
                detector_width,
                args.docker_image,
            ),
    }
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
