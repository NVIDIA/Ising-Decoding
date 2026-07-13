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
"""Locked JSON aggregation for color-code syndrome-density reduction sweeps."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def normalize_sdr_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "distance": _as_int(row["distance"]),
        "n_rounds": _as_int(row["n_rounds"]),
        "p": _as_float(row["p"]),
        "basis": str(row["basis"]).upper(),
        "input_syndrome_ones": _as_int(row["input_syndrome_ones"]),
        "residual_syndrome_ones": _as_int(row["residual_syndrome_ones"]),
        "syndrome_elements": _as_int(row["syndrome_elements"]),
    }


def resolve_rounds_mode(rows: list[dict[str, Any]]) -> str:
    normalized = [normalize_sdr_row(row) for row in rows]
    if not normalized:
        raise ValueError("Cannot resolve SDR rounds mode without rows")

    if all(row["n_rounds"] == row["distance"] for row in normalized):
        return "n_rounds_eq_d"
    if all(row["n_rounds"] == 4 * row["distance"] for row in normalized):
        return "n_rounds_eq_4d"

    combos = sorted({(row["distance"], row["n_rounds"]) for row in normalized})
    raise ValueError(
        "Color SDR aggregation supports only n_rounds=d or n_rounds=4*d; "
        f"got {combos}"
    )


def sdr_results_path(cfg, model_checkpoint_path: str, rows: list[dict[str, Any]]) -> str:
    rounds_mode = resolve_rounds_mode(rows)
    checkpoint_path = os.path.abspath(str(model_checkpoint_path))
    checkpoint_dir = os.path.dirname(checkpoint_path)
    use_checkpoint = int(getattr(getattr(cfg, "test", cfg), "use_model_checkpoint", -1))

    if use_checkpoint == -1:
        filename = f"sdr_results_{rounds_mode}.json"
    else:
        filename = f"{Path(checkpoint_path).stem}_sdr_results_{rounds_mode}.json"
    return os.path.join(checkpoint_dir, filename)


def _empty_payload(cfg, model_checkpoint_path: str, rounds_mode: str) -> dict[str, Any]:
    now = _now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "model":
            {
                "checkpoint_path":
                    os.path.abspath(str(model_checkpoint_path)),
                "model_version":
                    str(getattr(getattr(cfg, "model", object()), "version", "unknown")),
            },
        "rounds_mode": rounds_mode,
        "points": {},
    }


def _validate_payload(
    payload: dict[str, Any], cfg, model_checkpoint_path: str, rounds_mode: str
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported SDR results schema version: {payload.get('schema_version')}")
    if payload.get("rounds_mode") != rounds_mode:
        raise ValueError(
            f"SDR results rounds mode mismatch: existing={payload.get('rounds_mode')}, "
            f"new={rounds_mode}"
        )

    expected_model = os.path.abspath(str(model_checkpoint_path))
    existing_model = payload.get("model", {}).get("checkpoint_path")
    if existing_model != expected_model:
        raise ValueError(
            "SDR results model mismatch: "
            f"existing={existing_model!r}, new={expected_model!r}"
        )


def _sdr_payload(input_ones: int, residual_ones: int, elements: int) -> dict[str, Any]:
    input_density = float(input_ones) / float(elements) if elements > 0 else None
    residual_density = float(residual_ones) / float(elements) if elements > 0 else None
    if residual_density is None:
        reduction = None
    elif residual_density > 0:
        reduction = float(input_density) / residual_density
    else:
        reduction = float("inf")

    return {
        "input_syndrome_ones": int(input_ones),
        "residual_syndrome_ones": int(residual_ones),
        "syndrome_elements": int(elements),
        "input_syndrome_density": input_density,
        "residual_syndrome_density": residual_density,
        "reduction_factor": reduction,
    }


def _merge_row(payload: dict[str, Any], row: dict[str, Any]) -> None:
    d_key = str(row["distance"])
    p_key = str(float(row["p"]))
    basis = str(row["basis"]).upper()
    points = payload.setdefault("points", {})
    basis_points = points.setdefault(d_key, {}).setdefault(p_key, {})
    existing = basis_points.get(basis)

    if existing is None:
        input_ones = row["input_syndrome_ones"]
        residual_ones = row["residual_syndrome_ones"]
        elements = row["syndrome_elements"]
        contributions = 1
    else:
        if int(existing["n_rounds"]) != int(row["n_rounds"]):
            raise ValueError(
                f"Refusing to merge d={d_key}, p={p_key}, basis={basis}: "
                f"existing n_rounds={existing['n_rounds']} but new n_rounds={row['n_rounds']}"
            )
        input_ones = int(existing["input_syndrome_ones"]) + row["input_syndrome_ones"]
        residual_ones = int(existing["residual_syndrome_ones"]) + row["residual_syndrome_ones"]
        elements = int(existing["syndrome_elements"]) + row["syndrome_elements"]
        contributions = int(existing.get("contributions", 1)) + 1

    basis_points[basis] = {
        "distance": int(row["distance"]),
        "p": float(row["p"]),
        "basis": basis,
        "n_rounds": int(row["n_rounds"]),
        "contributions": contributions,
        "updated_at": _now_iso(),
        **_sdr_payload(input_ones, residual_ones, elements),
    }


def append_sdr_results(
    cfg,
    rows: list[dict[str, Any]],
    model_checkpoint_path: str,
) -> tuple[str, dict[str, Any]]:
    """Atomically add SDR rows into the model-local aggregate JSON."""
    normalized = [normalize_sdr_row(row) for row in rows]
    if not normalized:
        raise ValueError("Cannot append empty SDR results")

    json_path = sdr_results_path(cfg, model_checkpoint_path, normalized)
    rounds_mode = resolve_rounds_mode(normalized)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    lock_path = f"{json_path}.lock"
    tmp_path = f"{json_path}.tmp.{os.getpid()}"

    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(json_path):
                with open(json_path) as f:
                    payload = json.load(f)
                _validate_payload(payload, cfg, model_checkpoint_path, rounds_mode)
            else:
                payload = _empty_payload(cfg, model_checkpoint_path, rounds_mode)

            for row in normalized:
                _merge_row(payload, row)

            payload["updated_at"] = _now_iso()
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, json_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return json_path, payload
