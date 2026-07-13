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
"""Sweep color-code detector-input TensorRT latency across model architectures."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_CANDIDATES = {
    "model5": [256, 256, 256, 256, 256, 4],
    "model1": [128, 128, 128, 4],
    "thin96": [96, 96, 96, 4],
    "thin64": [64, 64, 64, 4],
}


def _parse_int_list(value: str) -> list[int]:
    return [int(part) for part in str(value).split(",") if part]


def _parse_candidates(values: list[str]) -> dict[str, list[int]]:
    candidates: dict[str, list[int]] = {}
    for value in values:
        if ":" in str(value):
            name, filters = str(value).split(":", 1)
            candidates[name] = _parse_int_list(filters)
            continue
        for item in str(value).split(","):
            if not item:
                continue
            if item not in DEFAULT_CANDIDATES:
                known = ", ".join(sorted(DEFAULT_CANDIDATES))
                raise ValueError(
                    f"Unknown candidate {item!r}. Use one of {known}, or name:f1,f2,..."
                )
            candidates[item] = DEFAULT_CANDIDATES[item]
    return candidates


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def _timing_summary(times_path: Path) -> dict[str, float | int | str]:
    data = _read_json(times_path)
    values = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for key in ("computeMs", "gpuComputeMs", "latencyMs"):
            if key in entry:
                values.append(float(entry[key]))
                break
    if not values:
        raise RuntimeError(f"No timing values found in {times_path}")
    values.sort()
    count = len(values)

    def percentile(q: float) -> float:
        index = round((q / 100.0) * (count - 1))
        return values[min(count - 1, max(0, int(index)))]

    return {
        "times_path": str(times_path),
        "count": count,
        "mean_ms": sum(values) / count,
        "median_ms": percentile(50),
        "p90_ms": percentile(90),
        "p99_ms": percentile(99),
    }


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _export_command(
    args: argparse.Namespace,
    *,
    filters: list[int],
    rounds: int,
    output_dir: Path,
) -> list[str]:
    exporter = Path(__file__).with_name("export_detector_input_trtexec.py")
    return [
        sys.executable,
        str(exporter),
        "--code",
        "color",
        "--distance",
        str(args.distance),
        "--rounds",
        str(rounds),
        "--basis",
        str(args.basis),
        "--output-mode",
        "residual",
        "--preprocess-strategy",
        "gather",
        "--filters",
        ",".join(str(v) for v in filters),
        "--quantize-fp8",
        "--calibration-samples",
        str(args.calibration_samples),
        "--output-dir",
        str(output_dir),
        "--docker-image",
        str(args.docker_image),
    ]


def _trtexec_command(
    args: argparse.Namespace,
    *,
    onnx_path: Path,
    output_dir: Path,
    detector_width: int,
    times_path: Path,
) -> list[str]:
    trtexec = [
        "trtexec",
        f"--onnx=/workspace/{onnx_path.name}",
        "--stronglyTyped",
        f"--shapes=dets:1x{detector_width}",
        "--duration=1",
        "--iterations=100",
        "--warmUp=5",
        "--builderOptimizationLevel=5",
        "--useCudaGraph=true",
        "--noDataTransfers",
        f"--exportTimes=/workspace/{times_path.name}",
    ]
    if not args.docker_image:
        return [
            part.replace(f"/workspace/{onnx_path.name}",
                         str(onnx_path)).replace(f"/workspace/{times_path.name}", str(times_path))
            for part in trtexec
        ]
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{output_dir.resolve()}:/workspace",
        str(args.docker_image),
        *trtexec,
    ]


def _write_rows(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "candidate",
        "filters",
        "distance",
        "rounds",
        "basis",
        "detector_width",
        "median_ms",
        "mean_ms",
        "p90_ms",
        "p99_ms",
        "times_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=13)
    parser.add_argument("--rounds", default="13,52,104")
    parser.add_argument("--basis", choices=["X", "Z"], default="Z")
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Candidate name or name:f1,f2,...; repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/detector-input-trtexec/color-architecture-sweep"),
    )
    parser.add_argument("--docker-image", default="nvcr.io/nvidia/tensorrt:25.03-py3")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--run-trtexec", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = _parse_candidates(args.candidate or ["model5,model1"])
    rounds_values = _parse_int_list(args.rounds)
    rows: list[dict[str, object]] = []
    stem_basis = str(args.basis).lower()

    for candidate, filters in candidates.items():
        for rounds in rounds_values:
            case_dir = args.output_dir / f"{candidate}-d{args.distance}-t{rounds}-{stem_basis}"
            stem = f"color-detector-input-residual-gather-d{args.distance}-t{rounds}-{stem_basis}"
            manifest_path = case_dir / f"{stem}.manifest.json"
            fp8_path = case_dir / f"{stem}.fp8.onnx"
            times_path = case_dir / f"{stem}.fp8-strongly-typed.times.json"

            if not args.skip_export:
                _run(
                    _export_command(args, filters=filters, rounds=rounds, output_dir=case_dir),
                    dry_run=bool(args.dry_run),
                )

            if args.run_trtexec:
                manifest = _read_json(manifest_path)
                metadata = manifest["metadata"]
                _run(
                    _trtexec_command(
                        args,
                        onnx_path=fp8_path,
                        output_dir=case_dir,
                        detector_width=int(metadata["detector_width"]),
                        times_path=times_path,
                    ),
                    dry_run=bool(args.dry_run),
                )

            if times_path.exists():
                manifest = _read_json(manifest_path)
                row = {
                    "candidate": candidate,
                    "filters": ",".join(str(v) for v in filters),
                    "distance": int(args.distance),
                    "rounds": int(rounds),
                    "basis": str(args.basis),
                    "detector_width": int(manifest["metadata"]["detector_width"]),
                    **_timing_summary(times_path),
                }
                rows.append(row)

    _write_rows(args.output_dir, rows)
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
