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
"""Locked JSON aggregation for color-code Chromobius single-shot timing."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STREAMS = ("original_syndromes", "residual_syndromes")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def normalize_timing_counter(counter: dict[str, Any]) -> dict[str, Any]:
    shots = _as_int(counter["shots"])
    min_value = counter.get("min_us_per_round")
    max_value = counter.get("max_us_per_round")
    return {
        "shots": shots,
        "sum_us_per_round": _as_float(counter["sum_us_per_round"]),
        "sum_sq_us_per_round": _as_float(counter["sum_sq_us_per_round"]),
        "min_us_per_round": None if min_value is None else _as_float(min_value),
        "max_us_per_round": None if max_value is None else _as_float(max_value),
    }


def normalize_timing_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "distance": _as_int(row["distance"]),
        "n_rounds": _as_int(row["n_rounds"]),
        "p": _as_float(row["p"]),
        "basis": str(row["basis"]).upper(),
        "original_syndromes": normalize_timing_counter(row["original_syndromes"]),
        "residual_syndromes": normalize_timing_counter(row["residual_syndromes"]),
    }


def resolve_rounds_mode(rows: list[dict[str, Any]]) -> str:
    normalized = [normalize_timing_row(row) for row in rows]
    if not normalized:
        raise ValueError("Cannot resolve Chromobius timing rounds mode without rows")

    if all(row["n_rounds"] == row["distance"] for row in normalized):
        return "n_rounds_eq_d"
    if all(row["n_rounds"] == 4 * row["distance"] for row in normalized):
        return "n_rounds_eq_4d"

    combos = sorted({(row["distance"], row["n_rounds"]) for row in normalized})
    raise ValueError(
        "Color Chromobius timing aggregation supports only n_rounds=d or n_rounds=4*d; "
        f"got {combos}"
    )


def chromobius_timing_results_path(
    cfg,
    model_checkpoint_path: str,
    rows: list[dict[str, Any]],
) -> str:
    rounds_mode = resolve_rounds_mode(rows)
    checkpoint_path = os.path.abspath(str(model_checkpoint_path))
    checkpoint_dir = os.path.dirname(checkpoint_path)
    use_checkpoint = int(getattr(getattr(cfg, "test", cfg), "use_model_checkpoint", -1))

    if use_checkpoint == -1:
        filename = f"chromobius_timing_results_{rounds_mode}.json"
    else:
        filename = f"{Path(checkpoint_path).stem}_chromobius_timing_results_{rounds_mode}.json"
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
        "stat_units": "us_per_round",
        "points": {},
    }


def _validate_payload(
    payload: dict[str, Any], cfg, model_checkpoint_path: str, rounds_mode: str
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Chromobius timing results schema version: {payload.get('schema_version')}"
        )
    if payload.get("rounds_mode") != rounds_mode:
        raise ValueError(
            f"Chromobius timing rounds mode mismatch: existing={payload.get('rounds_mode')}, "
            f"new={rounds_mode}"
        )

    expected_model = os.path.abspath(str(model_checkpoint_path))
    existing_model = payload.get("model", {}).get("checkpoint_path")
    if existing_model != expected_model:
        raise ValueError(
            "Chromobius timing results model mismatch: "
            f"existing={existing_model!r}, new={expected_model!r}"
        )


def _derived_counter(counter: dict[str, Any]) -> dict[str, Any]:
    shots = int(counter["shots"])
    sum_value = float(counter["sum_us_per_round"])
    sum_sq = float(counter["sum_sq_us_per_round"])
    avg = sum_value / shots if shots > 0 else None
    if shots > 1:
        variance = max((sum_sq - (sum_value * sum_value) / shots) / (shots - 1), 0.0)
    else:
        variance = None
    return {
        "shots": shots,
        "sum_us_per_round": sum_value,
        "sum_sq_us_per_round": sum_sq,
        "avg_us_per_round": avg,
        "variance_us_per_round_sq": variance,
        "min_us_per_round": counter["min_us_per_round"],
        "max_us_per_round": counter["max_us_per_round"],
    }


def _merge_counter(existing: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return _derived_counter(new)

    shots = int(existing["shots"]) + int(new["shots"])
    sum_value = float(existing["sum_us_per_round"]) + float(new["sum_us_per_round"])
    sum_sq = float(existing["sum_sq_us_per_round"]) + float(new["sum_sq_us_per_round"])
    existing_min = existing.get("min_us_per_round")
    existing_max = existing.get("max_us_per_round")
    new_min = new.get("min_us_per_round")
    new_max = new.get("max_us_per_round")

    mins = [value for value in (existing_min, new_min) if value is not None]
    maxes = [value for value in (existing_max, new_max) if value is not None]
    return _derived_counter(
        {
            "shots": shots,
            "sum_us_per_round": sum_value,
            "sum_sq_us_per_round": sum_sq,
            "min_us_per_round": min(mins) if mins else None,
            "max_us_per_round": max(maxes) if maxes else None,
        }
    )


def _merge_row(payload: dict[str, Any], row: dict[str, Any]) -> None:
    d_key = str(row["distance"])
    p_key = str(float(row["p"]))
    basis = str(row["basis"]).upper()
    points = payload.setdefault("points", {})
    basis_points = points.setdefault(d_key, {}).setdefault(p_key, {})
    existing = basis_points.get(basis)

    if existing is not None and int(existing["n_rounds"]) != int(row["n_rounds"]):
        raise ValueError(
            f"Refusing to merge d={d_key}, p={p_key}, basis={basis}: "
            f"existing n_rounds={existing['n_rounds']} but new n_rounds={row['n_rounds']}"
        )

    basis_points[basis] = {
        "distance":
            int(row["distance"]),
        "p":
            float(row["p"]),
        "basis":
            basis,
        "n_rounds":
            int(row["n_rounds"]),
        "contributions":
            int(existing.get("contributions", 0)) + 1 if existing else 1,
        "original_syndromes":
            _merge_counter(
                existing.get("original_syndromes") if existing else None,
                row["original_syndromes"],
            ),
        "residual_syndromes":
            _merge_counter(
                existing.get("residual_syndromes") if existing else None,
                row["residual_syndromes"],
            ),
        "updated_at":
            _now_iso(),
    }


def append_chromobius_timing_results(
    cfg,
    rows: list[dict[str, Any]],
    model_checkpoint_path: str,
) -> tuple[str, dict[str, Any]]:
    """Atomically add timing rows into the model-local aggregate JSON."""
    normalized = [normalize_timing_row(row) for row in rows]
    if not normalized:
        raise ValueError("Cannot append empty Chromobius timing results")

    json_path = chromobius_timing_results_path(cfg, model_checkpoint_path, normalized)
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
