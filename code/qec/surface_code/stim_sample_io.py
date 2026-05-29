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
"""Helpers for Stim detector-sample files used by offline decoding.

The pair of files written by :func:`write_stim_detector_samples` /
:func:`write_metadata_json` and consumed by :func:`read_stim_detector_samples`
forms the on-disk *Stim sample contract* between data generators (the local
simulator, a QPU wrapper, or a third-party producer) and the offline decoder.

The contract has two layers:

1. **Structural fields** — distance, rounds, basis, orientation, detector and
   observable counts, and the on-disk format. These are always validated and
   any mismatch is a hard error.
2. **Noise-model fingerprint** — ``p_error`` and the 25-parameter ``NoiseModel``
   parameters (via :func:`qec.noise_model.NoiseModel.sha256`). When the decoder
   passes in an active noise model and ``strict=True``, mismatches are a hard
   error; when ``strict=False`` they emit a warning. Older files that predate
   this field bypass the check, so legacy artifacts keep loading.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import stim

_ROTATION_ALIASES = {"O1": "XV", "O2": "XH", "O3": "ZV", "O4": "ZH"}
_SUPPORTED_FORMATS = {"dets"}

#: Current Stim sample metadata schema version.
SCHEMA_VERSION = 2


def normalize_code_rotation(value: Any) -> str:
    rotation = str(value).strip().upper()
    return _ROTATION_ALIASES.get(rotation, rotation)


def build_stim_sample_metadata(
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    num_detectors: int,
    num_observables: int,
    num_shots: int,
    sample_format: str = "dets",
    append_observables: bool = True,
    p_error: Optional[float] = None,
    noise_model_label: Optional[str] = None,
    noise_model_params: Optional[Mapping[str, float]] = None,
    noise_model_sha256: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a metadata dict matching the current Stim sample schema.

    Args:
        distance, n_rounds, basis, code_rotation, num_detectors, num_observables,
            num_shots: Structural fields the offline decoder must agree on.
        sample_format: Only ``"dets"`` is supported today; the argument exists
            so that future formats can be added without an API break.
        append_observables: ``True`` if logical observables are appended to each
            shot. The offline decoder requires this.
        p_error: Scalar physical error rate used when generating samples (or
            ``None`` if the generator uses an explicit ``NoiseModel``).
        noise_model_label: Human-readable label, e.g. ``"25-param"`` or
            ``"simple"``. Used for warnings and never for strict checks.
        noise_model_params: The full 25-parameter dict (sorted) used to build
            the ``NoiseModel`` instance, or ``None`` for the simple-noise case.
        noise_model_sha256: Deterministic fingerprint of ``noise_model_params``
            (typically ``NoiseModel.sha256()``). Used for strict comparison.
        extra: Optional additional fields to record alongside the contract.

    Returns:
        A JSON-serializable dict describing one ``samples_{basis}.dets`` file.
    """
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "stim_detector_samples",
        "format": sample_format,
        "append_observables": bool(append_observables),
        "distance": int(distance),
        "n_rounds": int(n_rounds),
        "basis": str(basis).strip().upper(),
        "code_rotation": str(code_rotation).strip().upper(),
        "num_detectors": int(num_detectors),
        "num_observables": int(num_observables),
        "num_shots": int(num_shots),
    }
    if p_error is not None:
        meta["p_error"] = float(p_error)
    if noise_model_label is not None:
        meta["noise_model"] = str(noise_model_label)
    if noise_model_params is not None:
        meta["noise_model_params"] = {
            str(k): float(v) for k, v in sorted(dict(noise_model_params).items())
        }
    if noise_model_sha256 is not None:
        meta["noise_model_sha256"] = str(noise_model_sha256)
    if extra:
        for k, v in dict(extra).items():
            meta.setdefault(str(k), v)
    return meta


def write_metadata_json(path: str | Path, metadata: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, sort_keys=True)
        f.write("\n")


def read_metadata_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Stim metadata must be a JSON object: {path}")
    return metadata


def validate_stim_sample_metadata(
    metadata: Mapping[str, Any],
    *,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    num_detectors: int,
    num_observables: int,
    p_error: Optional[float] = None,
    noise_model_sha256: Optional[str] = None,
    noise_model_label: Optional[str] = None,
    p_error_atol: float = 0.0,
    strict_noise: bool = True,
) -> None:
    """Validate metadata against the circuit/config used for decoding.

    Structural checks (distance, rounds, basis, orientation, detector and
    observable counts, format, observable appending) are always strict. The
    noise-model checks are opt-in: pass ``p_error`` and/or
    ``noise_model_sha256`` to compare against the recorded values. When the
    decoder does not provide a noise model (``noise_model_sha256=None``), older
    files that never recorded one are accepted as-is.

    Args:
        metadata: Loaded metadata dict.
        distance, n_rounds, basis, code_rotation, num_detectors, num_observables:
            Structural fields the decoder expects.
        p_error: Decoder's active scalar error rate, or ``None`` to skip.
        noise_model_sha256: Decoder's active noise-model fingerprint, or
            ``None`` to skip.
        noise_model_label: Decoder's active noise-model label
            (e.g. ``"25-param"``), used only for clarifying messages.
        p_error_atol: Absolute tolerance when comparing scalar ``p_error``.
        strict_noise: If ``True``, mismatches in ``p_error`` or
            ``noise_model_sha256`` raise. If ``False``, they emit a
            :class:`UserWarning` and the call continues.

    Raises:
        ValueError: With one line per structural mismatch (and per noise
            mismatch when ``strict_noise=True``). The messages are intentionally
            explicit because these files are a cross-team contract.
    """
    errors: list[str] = []
    # Legacy files (no schema_version key) predate the noise fingerprint; treat them as v1.
    sv = metadata.get("schema_version", 1)
    if not isinstance(sv, int) or isinstance(sv, bool) or sv > SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {sv!r} (max {SCHEMA_VERSION})")
    sample_format = str(metadata.get("format", "")).strip().lower()
    if sample_format not in _SUPPORTED_FORMATS:
        errors.append(
            f"metadata format mismatch: file has {metadata.get('format')!r}, "
            f"supported formats are {sorted(_SUPPORTED_FORMATS)}"
        )

    if metadata.get("append_observables") is not True:
        errors.append(
            "metadata append_observables mismatch: expected true because LER requires logical labels"
        )

    checks = (
        ("distance", int(distance), lambda v: int(v)),
        ("n_rounds", int(n_rounds), lambda v: int(v)),
        ("basis", str(basis).strip().upper(), lambda v: str(v).strip().upper()),
        ("num_detectors", int(num_detectors), lambda v: int(v)),
        ("num_observables", int(num_observables), lambda v: int(v)),
    )
    for key, expected, cast in checks:
        if key not in metadata:
            errors.append(f"metadata missing required field: {key}")
            continue
        try:
            actual = cast(metadata[key])
        except Exception:
            errors.append(
                f"metadata {key} mismatch: file has {metadata[key]!r}, expected {expected!r}"
            )
            continue
        if actual != expected:
            errors.append(f"metadata {key} mismatch: file has {actual!r}, expected {expected!r}")

    if "code_rotation" not in metadata:
        errors.append("metadata missing required field: code_rotation")
    else:
        actual_rotation = normalize_code_rotation(metadata["code_rotation"])
        expected_rotation = normalize_code_rotation(code_rotation)
        if actual_rotation != expected_rotation:
            errors.append(
                "metadata code_rotation mismatch: "
                f"file has {metadata['code_rotation']!r}/{actual_rotation}, "
                f"decode requested {code_rotation!r}/{expected_rotation}"
            )

    if int(num_observables) <= 0:
        errors.append("missing observables: rebuilt circuit has num_observables=0")
    else:
        try:
            file_num_obs = int(metadata.get("num_observables", 0))
        except Exception:
            file_num_obs = 0
        if file_num_obs <= 0:
            errors.append("missing observables: metadata num_observables must be positive")

    noise_messages: list[str] = []
    if p_error is not None and "p_error" in metadata:
        try:
            file_p_error = float(metadata["p_error"])
        except Exception:
            file_p_error = None
        if file_p_error is None:
            noise_messages.append(
                f"metadata p_error mismatch: file has {metadata['p_error']!r}, "
                f"decoder uses {float(p_error)!r}"
            )
        elif abs(file_p_error - float(p_error)) > float(p_error_atol):
            noise_messages.append(
                f"metadata p_error mismatch: file has {file_p_error!r}, "
                f"decoder uses {float(p_error)!r} (atol={p_error_atol})"
            )

    if noise_model_sha256 is not None and "noise_model_sha256" in metadata:
        file_sha = str(metadata.get("noise_model_sha256", "")).strip()
        if file_sha != str(noise_model_sha256).strip():
            file_label = metadata.get("noise_model", "?")
            local_label = noise_model_label or "?"
            noise_messages.append(
                "metadata noise_model_sha256 mismatch: "
                f"file has {file_sha!r} (label={file_label!r}), "
                f"decoder uses {noise_model_sha256!r} (label={local_label!r})"
            )

    if noise_messages and strict_noise:
        errors.extend(noise_messages)
    elif noise_messages:
        warnings.warn(
            "Stim sample noise-model drift (continuing because strict_noise=False):\n- " +
            "\n- ".join(noise_messages),
            UserWarning,
            stacklevel=2,
        )

    if errors:
        raise ValueError("Invalid Stim sample metadata:\n- " + "\n- ".join(errors))


def write_stim_detector_samples(
    *,
    path: str | Path,
    dets_and_obs: np.ndarray,
    num_detectors: int,
    num_observables: int,
    sample_format: str = "dets",
) -> None:
    """Write a ``samples_*.dets`` file using Stim's sparse format.

    Args:
        path: Destination path (parent directories are created).
        dets_and_obs: ``(num_shots, num_detectors + num_observables)`` array of
            detector bits with logical observables appended. The width must
            match exactly; mismatches raise rather than producing a malformed
            file that would fail validation on read.
        num_detectors, num_observables: Counts used to direct Stim's writer.
        sample_format: Currently only ``"dets"`` is supported.
    """
    sample_format = str(sample_format).lower()
    if sample_format not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported Stim sample format {sample_format!r}")
    data = np.asarray(dets_and_obs, dtype=np.bool_)
    if data.ndim != 2:
        raise ValueError(
            f"dets_and_obs must be 2-D (num_shots, num_detectors + num_observables); "
            f"got shape {data.shape}"
        )
    expected_width = int(num_detectors) + int(num_observables)
    if int(data.shape[1]) != expected_width:
        raise ValueError(
            f"dets_and_obs width mismatch: array has {int(data.shape[1])} columns, "
            f"expected num_detectors + num_observables = {expected_width} "
            f"(num_detectors={int(num_detectors)}, num_observables={int(num_observables)})."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stim.write_shot_data_file(
        data=data,
        path=str(path),
        format=sample_format,
        num_detectors=int(num_detectors),
        num_observables=int(num_observables),
    )


def read_stim_detector_samples(
    *,
    samples_path: str | Path,
    metadata_path: str | Path,
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    num_detectors: int,
    num_observables: int,
    p_error: Optional[float] = None,
    noise_model_sha256: Optional[str] = None,
    noise_model_label: Optional[str] = None,
    strict_noise: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a ``samples_*.dets`` file after validating its metadata.

    Args:
        samples_path, metadata_path: File paths for the sample data and JSON
            metadata, typically produced by :func:`resolve_stim_sample_paths`.
        distance, n_rounds, basis, code_rotation, num_detectors,
            num_observables: Structural parameters the decoder expects.
        p_error, noise_model_sha256, noise_model_label: Optional noise-model
            fingerprint the decoder is using. See
            :func:`validate_stim_sample_metadata` for semantics.
        strict_noise: When ``True`` (the default), noise-fingerprint
            mismatches raise. When ``False``, they emit a warning.

    Returns:
        ``(dets_and_obs, metadata)`` where ``dets_and_obs`` has shape
        ``(num_shots, num_detectors + num_observables)`` and dtype ``uint8``.

    Raises:
        ValueError: If the metadata is inconsistent with the structural
            parameters (always strict) or, when ``strict_noise=True``, with
            the noise fingerprint.
    """
    metadata = read_metadata_json(metadata_path)
    validate_stim_sample_metadata(
        metadata,
        distance=distance,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=code_rotation,
        num_detectors=num_detectors,
        num_observables=num_observables,
        p_error=p_error,
        noise_model_sha256=noise_model_sha256,
        noise_model_label=noise_model_label,
        strict_noise=strict_noise,
    )
    data = stim.read_shot_data_file(
        path=str(samples_path),
        format=str(metadata["format"]).lower(),
        num_detectors=int(num_detectors),
        num_observables=int(num_observables),
    )
    arr = np.asarray(data, dtype=np.uint8)
    expected_width = int(num_detectors) + int(num_observables)
    if arr.ndim != 2 or arr.shape[1] != expected_width:
        raise ValueError(
            f"Stim sample shape mismatch: file produced shape {arr.shape}, "
            f"expected (*, {expected_width})"
        )
    expected_shots = int(metadata["num_shots"])
    if arr.shape[0] != expected_shots:
        raise ValueError(
            f"metadata num_shots mismatch: file has {arr.shape[0]} shots, "
            f"metadata has {expected_shots}"
        )
    return arr, dict(metadata)


def resolve_stim_sample_paths(root: str | Path, basis: str) -> tuple[Path, Path]:
    """Resolve either flat or per-basis Stim artifact layouts."""
    root = Path(root)
    basis = str(basis).strip().upper()
    candidates = (
        (root / f"samples_{basis}.dets", root / f"metadata_{basis}.json"),
        (root / basis / "samples.dets", root / basis / "metadata.json"),
        (root / "samples.dets", root / "metadata.json"),
    )
    for samples_path, metadata_path in candidates:
        if samples_path.exists() and metadata_path.exists():
            return samples_path, metadata_path
    expected = ", ".join(f"{s} + {m}" for s, m in candidates)
    raise FileNotFoundError(
        f"No Stim sample artifact found for basis {basis}. Expected one of: {expected}"
    )
