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
"""Locked JSON aggregation for color-code threshold logical-error counts."""

from __future__ import annotations

import fcntl
import json
import math
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


def _safe_rate(errors: int, shots: int) -> float | None:
    if shots <= 0:
        return None
    return float(errors) / float(shots)


def _logical_rate_per_round(total_ler: float | None, n_rounds: int) -> float | None:
    if total_ler is None or n_rounds <= 0:
        return None
    total_ler = min(max(float(total_ler), 0.0), 1.0)
    return 1.0 - (1.0 - total_ler)**(1.0 / float(n_rounds))


def _stderr(errors: int, shots: int) -> float | None:
    if shots <= 0:
        return None
    p = float(errors) / float(shots)
    return math.sqrt(max(p * (1.0 - p), 0.0) / float(shots))


def normalize_threshold_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize color threshold rows to one count schema."""
    distance = _as_int(row["distance"])
    n_rounds = _as_int(row["n_rounds"])
    p_value = _as_float(row["p"])
    basis = str(row["basis"]).upper()

    if "pd_chromobius_errors" in row:
        pd_errors = _as_int(row["pd_chromobius_errors"])
        pd_shots = _as_int(row["pd_chromobius_shots"])
    else:
        pd_errors = _as_int(row["logical_errors"])
        pd_shots = _as_int(row["num_shots"])

    chromobius_shots = _as_int(row.get("chromobius_shots", row.get("num_shots", pd_shots)))
    return {
        "distance": distance,
        "n_rounds": n_rounds,
        "p": p_value,
        "basis": basis,
        "pd_chromobius_errors": pd_errors,
        "pd_chromobius_shots": pd_shots,
        "chromobius_errors": _as_int(row["chromobius_errors"]),
        "chromobius_shots": chromobius_shots,
    }


def resolve_rounds_mode(rows: list[dict[str, Any]]) -> str:
    normalized = [normalize_threshold_row(row) for row in rows]
    if not normalized:
        raise ValueError("Cannot resolve threshold rounds mode without rows")

    if all(row["n_rounds"] == row["distance"] for row in normalized):
        return "n_rounds_eq_d"
    if all(row["n_rounds"] == 4 * row["distance"] for row in normalized):
        return "n_rounds_eq_4d"

    combos = sorted({(row["distance"], row["n_rounds"]) for row in normalized})
    raise ValueError(
        "Color threshold aggregation supports only n_rounds=d or n_rounds=4*d; "
        f"got {combos}"
    )


def threshold_results_path(
    cfg,
    model_checkpoint_path: str,
    rows: list[dict[str, Any]],
) -> str:
    rounds_mode = resolve_rounds_mode(rows)
    checkpoint_path = os.path.abspath(str(model_checkpoint_path))
    checkpoint_dir = os.path.dirname(checkpoint_path)
    use_checkpoint = int(getattr(getattr(cfg, "test", cfg), "use_model_checkpoint", -1))

    if use_checkpoint == -1:
        filename = f"threshold_results_{rounds_mode}.json"
    else:
        filename = f"{Path(checkpoint_path).stem}_threshold_results_{rounds_mode}.json"
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
        raise ValueError(
            f"Unsupported threshold results schema version: {payload.get('schema_version')}"
        )
    if payload.get("rounds_mode") != rounds_mode:
        raise ValueError(
            f"Threshold results rounds mode mismatch: existing={payload.get('rounds_mode')}, "
            f"new={rounds_mode}"
        )

    expected_model = os.path.abspath(str(model_checkpoint_path))
    existing_model = payload.get("model", {}).get("checkpoint_path")
    if existing_model != expected_model:
        raise ValueError(
            "Threshold results model mismatch: "
            f"existing={existing_model!r}, new={expected_model!r}"
        )


def _counter_payload(errors: int, shots: int, n_rounds: int) -> dict[str, Any]:
    ler_total = _safe_rate(errors, shots)
    return {
        "logical_errors": int(errors),
        "shots": int(shots),
        "ler_total": ler_total,
        "ler_per_round": _logical_rate_per_round(ler_total, n_rounds),
        "ler_stderr": _stderr(errors, shots),
    }


def _merge_row(payload: dict[str, Any], row: dict[str, Any]) -> None:
    d_key = str(row["distance"])
    p_key = str(float(row["p"]))
    basis = str(row["basis"]).upper()
    points = payload.setdefault("points", {})
    basis_points = points.setdefault(d_key, {}).setdefault(p_key, {})
    existing = basis_points.get(basis)

    if existing is None:
        pd_errors = row["pd_chromobius_errors"]
        pd_shots = row["pd_chromobius_shots"]
        chromobius_errors = row["chromobius_errors"]
        chromobius_shots = row["chromobius_shots"]
        contributions = 1
    else:
        if int(existing["n_rounds"]) != int(row["n_rounds"]):
            raise ValueError(
                f"Refusing to merge d={d_key}, p={p_key}, basis={basis}: "
                f"existing n_rounds={existing['n_rounds']} but new n_rounds={row['n_rounds']}"
            )
        pd_errors = int(existing["pd_chromobius"]["logical_errors"]) + row["pd_chromobius_errors"]
        pd_shots = int(existing["pd_chromobius"]["shots"]) + row["pd_chromobius_shots"]
        chromobius_errors = int(existing["chromobius"]["logical_errors"]) + row["chromobius_errors"]
        chromobius_shots = int(existing["chromobius"]["shots"]) + row["chromobius_shots"]
        contributions = int(existing.get("contributions", 1)) + 1

    basis_points[basis] = {
        "distance": int(row["distance"]),
        "p": float(row["p"]),
        "basis": basis,
        "n_rounds": int(row["n_rounds"]),
        "contributions": contributions,
        "pd_chromobius": _counter_payload(pd_errors, pd_shots, int(row["n_rounds"])),
        "chromobius": _counter_payload(chromobius_errors, chromobius_shots, int(row["n_rounds"])),
        "updated_at": _now_iso(),
    }


def append_threshold_results(
    cfg,
    rows: list[dict[str, Any]],
    model_checkpoint_path: str,
) -> tuple[str, dict[str, Any]]:
    """Atomically add threshold rows into the model-local aggregate JSON."""
    normalized = [normalize_threshold_row(row) for row in rows]
    if not normalized:
        raise ValueError("Cannot append empty threshold results")

    json_path = threshold_results_path(cfg, model_checkpoint_path, normalized)
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
