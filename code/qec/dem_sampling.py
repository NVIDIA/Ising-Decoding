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
DEM sampling utilities for training data generation.

Sampling runs on the GPU via cuQuantum's cuStabilizer BitMatrixSampler with
optional CuPy zero-copy DLPack transfers.  cuquantum>=26.3.0 is required.

This module provides the sampling functions needed by MemoryCircuitTorch
to generate training batches from precomputed DEM matrices (H, p, A).
"""

from __future__ import annotations

import threading
import time
import weakref
from collections import deque

import torch
import numpy as np

try:
    from cuquantum.stabilizer.dem_sampling import BitMatrixSampler
    from cuquantum.stabilizer.simulator import Options
    _CUSTAB_AVAILABLE = True
except ImportError:
    # This should only happen if cuquantum is not installed. That is expected
    # for some test environments that don't need DEM sampling, so handle that
    # gracefully here.
    BitMatrixSampler = None  # type: ignore[misc, assignment]
    Options = None  # type: ignore[misc, assignment]
    _CUSTAB_AVAILABLE = False

try:
    import cupy as _cp  # noqa: F401
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def _custab_available() -> bool:
    """True if custabilizer (cuquantum.stabilizer) is present. For use by tests/skip logic."""
    return _CUSTAB_AVAILABLE


class _SamplerEntry:
    """Cached sampler state for one DEM matrix ``H`` (and the ``p`` it was built with)."""

    __slots__ = ("_H_ref", "csr", "p", "sampler", "max_shots", "device_id", "seed")

    def __init__(self, H: torch.Tensor) -> None:
        # Weak: the cache must not keep a caller's matrix alive (a view such as
        # ``H.T`` would, through its ``_base``). The sampler owns its own copy
        # of the matrix (CSR on CUDA, numpy on CPU); once the caller drops H the
        # entry is stale and evicted on the next call.
        self._H_ref = weakref.ref(H)
        self.csr = None  # BitMatrixCSR of H^T, built once per H on the CUDA path
        self.p = None
        self.sampler = None
        self.max_shots = 0
        self.device_id = None
        self.seed = None

    @property
    def H(self) -> "torch.Tensor | None":
        return self._H_ref()


# One entry per DEM matrix, keyed by id(H). Mixed-basis training alternates two
# simulators (X and Z, distinct H) every step; a single-slot cache rebuilt the
# sampler on every basis switch. Insertion order doubles as LRU order: a hit
# re-inserts its entry at the end, and the cache is bounded so matrices from
# earlier runs in the same process (tests, sweeps) do not pile up with their
# GPU samplers. Entries whose H was garbage-collected are dropped on sight.
_SAMPLER_CACHE_MAX = 8
_sampler_cache: dict[int, _SamplerEntry] = {}
# Sampler destruction can release the GIL while cache entries are removed.
# Reentrant because touching an entry also evicts stale entries.
_sampler_cache_lock = threading.RLock()


def _evict_stale_sampler_entries() -> None:
    """Drop entries whose matrix has been garbage-collected."""
    with _sampler_cache_lock:
        for stale_key in [k for k, e in _sampler_cache.items() if e.H is None]:
            del _sampler_cache[stale_key]


def _touch_sampler_entry(H: torch.Tensor) -> _SamplerEntry:
    """Return the live cache entry for ``H``, creating/evicting as needed."""
    with _sampler_cache_lock:
        key = id(H)
        entry = _sampler_cache.pop(key, None)
        if entry is None or entry.H is not H:
            entry = _SamplerEntry(H)
        _evict_stale_sampler_entries()
        _sampler_cache[key] = entry  # most recently used goes last
        while len(_sampler_cache) > _SAMPLER_CACHE_MAX:
            del _sampler_cache[next(iter(_sampler_cache))]
        return entry


# Most recently used entry, mirrored into module globals for introspection/tests.
_cached_sampler = None
_cached_H: "torch.Tensor | None" = None
_cached_HT: "torch.Tensor | None" = None
_cached_max_shots: int = 0
_cached_device_id: int | None = None
_cached_seed: "int | None" = None

# Row-chunk size (in matrix elements) for the CSR build. Bounds the temporary
# memory of torch.nonzero on one chunk; the dense H^T itself is never
# materialised contiguously. 2**28 uint8 elements = 256 MiB per chunk.
_CSR_BUILD_CHUNK_ELEMENTS = 2**28

_DEM_TIMINGS_S: deque[float] = deque(maxlen=200)
_custab_path_logged: bool = False

_MIN_MAX_SHOTS = 1024


def get_dem_sampling_avg_ms() -> float:
    """Average duration of recent dem_sampling calls in milliseconds (for training log)."""
    if not _DEM_TIMINGS_S:
        return 0.0
    return (sum(_DEM_TIMINGS_S) / len(_DEM_TIMINGS_S)) * 1000.0


def _reset_sampler_cache() -> None:
    """Reset the module-level sampler cache."""
    global _cached_sampler, _cached_H, _cached_HT, _cached_max_shots
    global _cached_device_id, _cached_seed
    with _sampler_cache_lock:
        _sampler_cache.clear()
    _cached_sampler = None
    _cached_H = None
    _cached_HT = None
    _cached_max_shots = 0
    _cached_device_id = None
    _cached_seed = None


def _bit_matrix_csr_from_dense(HT: torch.Tensor):
    """Build a cuStabilizer ``BitMatrixCSR`` (rows = errors) from dense ``H^T`` on CUDA.

    Works in row chunks so that neither a contiguous copy of the dense matrix
    nor a full-size index tensor is ever allocated: the reference d17 color
    model has H^T of 437,479 x 14,722 = 6.4e9 elements, which also overflows
    the int32 element count in CuPy's ``packbits`` that the dense sampler input
    goes through. The CSR carries only the non-zeros.
    """
    from cuquantum.stabilizer.dem_sampling import BitMatrixCSR
    import cupy as cp

    n_errors, n_outcomes = int(HT.shape[0]), int(HT.shape[1])
    rows_per_chunk = max(1, _CSR_BUILD_CHUNK_ELEMENTS // max(1, n_outcomes))
    counts = torch.zeros(n_errors, dtype=torch.int64, device=HT.device)
    col_chunks = []
    for start in range(0, n_errors, rows_per_chunk):
        block = HT[start:start + rows_per_chunk]
        nz = torch.nonzero(block, as_tuple=False)  # row-major order -> CSR order
        if nz.numel():
            counts[start:start +
                   block.shape[0]] += torch.bincount(nz[:, 0], minlength=block.shape[0])
            col_chunks.append(nz[:, 1].to(torch.int64))
    col_indices = (
        torch.cat(col_chunks)
        if col_chunks else torch.empty(0, dtype=torch.int64, device=HT.device)
    )
    row_offsets = torch.zeros(n_errors + 1, dtype=torch.int64, device=HT.device)
    torch.cumsum(counts, dim=0, out=row_offsets[1:])
    nnz = int(col_indices.shape[0])
    with cp.cuda.Device(HT.device.index if HT.device.index is not None else 0):
        return BitMatrixCSR(
            row_offsets=cp.from_dlpack(row_offsets.contiguous()),
            col_indices=cp.from_dlpack(col_indices.contiguous()),
            nnz=nnz,
            shape=(n_errors, n_outcomes),
        )


def dem_sampling(
    H: torch.Tensor,
    p: torch.Tensor,
    batch_size: int,
    device_id: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample errors from a detector error model (DEM) via cuST BitMatrixSampler.

    Args:
        H: (2*num_detectors, num_errors) uint8 - Detector-error incidence matrix
        p: (num_errors,) float32 - Per-error probabilities
        batch_size: int - Number of samples to generate
        device_id: Optional int - Device ID for cuST. If omitted, infer from
            H.device when H is on CUDA.
        seed: Optional int - RNG seed for this call. On the CUDA path the
            cached sampler is reused and the seed is passed to ``sample``, so
            repeated calls with the same seed produce identical outputs without
            rebuilding the sampler. On the CPU path a fresh sampler is created.
            When None (default), the sampler's own RNG stream continues.

    Returns:
        frames_xz: (batch_size, 2*num_detectors) uint8 - Detector outcomes
    """
    from cuquantum.stabilizer.dem_sampling import BitMatrixSampler
    from cuquantum.stabilizer.simulator import Options

    global _cached_sampler, _cached_H, _cached_HT, _cached_max_shots
    global _cached_device_id, _cached_seed, _custab_path_logged

    if H.ndim != 2:
        raise ValueError(f"H must be 2-D, got ndim={H.ndim}")
    if p.ndim != 1:
        raise ValueError(f"p must be 1-D, got ndim={p.ndim}")
    if H.shape[1] != p.shape[0]:
        raise ValueError(f"H has {H.shape[1]} columns but p has {p.shape[0]} entries")

    if device_id is None:
        if H.is_cuda:
            device_index = H.device.index
            device_id = int(torch.cuda.current_device() if device_index is None else device_index)
        else:
            device_id = 0

    gpu_native = _CUPY_AVAILABLE and H.is_cuda

    entry = _touch_sampler_entry(H)

    # CUDA: one sparse sampler per (H, p, device); a per-call seed goes to
    # ``sample`` and resets its RNG, so repeated calls with the same seed
    # reproduce exactly without rebuilding anything. CPU (numpy) samplers take
    # the seed at construction, so that path still rebuilds when a seed is set.
    need_new = (
        entry.sampler is None or entry.p is not p or batch_size > entry.max_shots or
        entry.device_id != device_id or (seed is not None and not gpu_native)
    )

    if need_new:
        max_shots = max(batch_size, _MIN_MAX_SHOTS)
        if gpu_native:
            import cupy as cp
            with cp.cuda.Device(device_id):
                if entry.csr is None:
                    entry.csr = _bit_matrix_csr_from_dense(H.T.detach())
                p_in = cp.from_dlpack(p.detach().to(torch.float64).contiguous())
                entry.sampler = BitMatrixSampler(
                    entry.csr,
                    p_in,
                    max_shots,
                    package="cupy",
                    options=Options(device_id=device_id)
                )
        else:
            H_in = H.T.detach().cpu().numpy().astype(np.uint8)
            p_in = p.detach().cpu().numpy().astype(np.float64)
            bms_kwargs: dict = {"package": "numpy", "options": Options(device_id=device_id)}
            if seed is not None:
                bms_kwargs["seed"] = seed
            entry.sampler = BitMatrixSampler(H_in, p_in, max_shots, **bms_kwargs)
        entry.p = p
        entry.max_shots = max_shots
        entry.device_id = device_id
    entry.seed = seed

    _cached_sampler = entry.sampler
    _cached_H = entry.H
    _cached_HT = H.T
    _cached_max_shots = entry.max_shots
    _cached_device_id = entry.device_id
    _cached_seed = entry.seed
    # The mirrors above released their reference to the previous matrix; if
    # that was its last one, its entry is stale now.
    _evict_stale_sampler_entries()

    t0 = time.perf_counter()
    if gpu_native:
        import cupy as cp
        with cp.cuda.Device(device_id):
            entry.sampler.sample(batch_size, seed=seed)
            out = entry.sampler.get_outcomes(bit_packed=False)
    else:
        entry.sampler.sample(batch_size)
        out = entry.sampler.get_outcomes(bit_packed=False)
    if isinstance(out, np.ndarray):
        out = torch.as_tensor(out, device=H.device).to(dtype=torch.uint8)
    else:
        # ``out`` is CuPy-owned memory (the sampler's outcome buffer). Hand the
        # caller a torch-owned copy and wait for that copy before the CuPy
        # array is released: the sampler's next ``sample`` may reuse the same
        # pool block on its own stream while kernels queued on the caller's
        # stream (e.g. the superbatch producer) are still reading it. One
        # stream sync per sampling call; the sample itself is far larger.
        out_cp = out
        out = torch.from_dlpack(out_cp).to(dtype=torch.uint8, copy=True)
        torch.cuda.current_stream(out.device).synchronize()
        del out_cp
    _DEM_TIMINGS_S.append(time.perf_counter() - t0)

    if not _custab_path_logged:
        print(
            f"---- [dem_sampling] Using cuST BitMatrixSampler path "
            f"(max_shots={_cached_max_shots}, gpu_native={gpu_native}, "
            f"matrix={'sparse CSR' if gpu_native else 'dense'}, device_id={device_id})"
        )
        _custab_path_logged = True

    return out


def measure_from_stacked_frames(
    frames_xz: torch.Tensor,
    meas_qubits: torch.Tensor,
    meas_bases: torch.Tensor,
    nq: int,
) -> torch.Tensor:
    """
    Extract measurement outcomes from stacked frame data.

    Convention: Z-basis measurement reads the X-component of the Pauli frame
    (anti-commutation), and X-basis measurement reads the Z-component.

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8 - Stacked [X|Z] detector frames
        meas_qubits: (num_meas,) long - Qubit indices for measurements
        meas_bases: (num_meas,) long - Basis for each measurement (0=X, 1=Z)
        nq: int - Total number of qubits

    Returns:
        meas_old: (batch_size, n_rounds, num_meas) uint8 - Measurement outcomes
    """
    meas_qubits = torch.as_tensor(meas_qubits, device=frames_xz.device,
                                  dtype=torch.long).reshape(-1)
    meas_bases = torch.as_tensor(meas_bases, device=frames_xz.device, dtype=torch.long).reshape(-1)
    D = frames_xz.shape[1] // 2
    R = D // int(nq)
    assert D == R * int(nq), f"Detector count {D} must be divisible by nq={nq}"

    idx = (torch.arange(R, device=frames_xz.device)[:, None] * int(nq) +
           meas_qubits[None, :]).reshape(-1)
    x = frames_xz[:, :D].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    z = frames_xz[:, D:].index_select(1, idx).reshape(frames_xz.shape[0], R, -1)
    return torch.where(meas_bases[None, None, :] == 1, x, z).to(torch.uint8)


def timelike_syndromes(
    frames_xz: torch.Tensor,
    A: torch.Tensor,
    meas_old: torch.Tensor,
) -> torch.Tensor:
    """
    Apply timelike corrections to measurements using the A matrix.

    A is a linear map over GF(2) producing s2 from frames_xz;
    meas_new = s2 ^ meas_old.

    Args:
        frames_xz: (batch_size, 2*num_detectors) uint8
        A: (n_rounds*num_meas, 2*num_detectors) uint8
        meas_old: (batch_size, n_rounds, num_meas) uint8

    Returns:
        meas_new: (batch_size, n_rounds, num_meas) uint8
    """
    s2 = torch.remainder(frames_xz.float() @ A.t().float(), 2).to(torch.uint8)
    return (s2 ^ meas_old.reshape(meas_old.shape[0], -1)).reshape_as(meas_old)
