# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA path of dem_sampling: cached sparse CSR sampler with per-call seeds.

The dense sampler input packed H^T through CuPy's ``packbits``, whose element
count is int32: the d17 color model's 6.4e9-element matrix overflowed it, and
every seeded call rebuilt the sampler (repacking the whole matrix). The CUDA
path now builds one ``BitMatrixCSR`` per H and passes the seed to ``sample``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from qec import dem_sampling as dem  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and dem._custab_available() and dem._CUPY_AVAILABLE),
    reason="CUDA + cuStabilizer + CuPy required",
)


def _random_h(n_outcomes=37, n_errors=530, density=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    H = (torch.rand((n_outcomes, n_errors), generator=g) < density).to(torch.uint8)
    p = torch.full((n_errors,), 0.02, dtype=torch.float32)
    return H.cuda(), p.cuda()


def test_csr_matches_the_dense_transpose():
    H, _ = _random_h()
    csr = dem._bit_matrix_csr_from_dense(H.T)
    dense_back = csr.to_scipy_sparse().toarray().astype(np.uint8)
    np.testing.assert_array_equal(dense_back, H.T.cpu().numpy())
    assert csr.nnz == int(H.sum().item())
    assert csr.shape == (H.shape[1], H.shape[0])


def test_csr_build_is_chunk_invariant(monkeypatch):
    H, _ = _random_h(n_outcomes=41, n_errors=1000)
    full = dem._bit_matrix_csr_from_dense(H.T).to_scipy_sparse().toarray()
    monkeypatch.setattr(dem, "_CSR_BUILD_CHUNK_ELEMENTS", 41 * 7)  # 7 rows per chunk
    chunked = dem._bit_matrix_csr_from_dense(H.T).to_scipy_sparse().toarray()
    np.testing.assert_array_equal(full, chunked)


def test_seeded_calls_reuse_one_sampler_and_reproduce():
    dem._reset_sampler_cache()
    H, p = _random_h()
    a = dem.dem_sampling(H, p, 256, seed=11)
    sampler = dem._cached_sampler
    b = dem.dem_sampling(H, p, 256, seed=11)
    c = dem.dem_sampling(H, p, 256, seed=12)
    assert dem._cached_sampler is sampler, "seeded CUDA calls must not rebuild the sampler"
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (256, H.shape[0]) and a.dtype == torch.uint8
    assert set(torch.unique(a).tolist()) <= {0, 1}


def test_outcomes_are_consistent_with_the_matrix():
    """Deterministic probabilities pin the sampler's arithmetic on the sparse matrix."""
    dem._reset_sampler_cache()
    H, p = _random_h(n_outcomes=16, n_errors=40, density=0.2)
    # Zero-probability errors never fire -> all-zero outcomes.
    zeros = dem.dem_sampling(H, torch.zeros_like(p), 64, seed=1)
    assert int(zeros.sum().item()) == 0
    # A new p for the same H must rebuild the sampler (not reuse the p=0 one):
    # probability-one errors all fire -> outcome = parity of each row of H.
    ones = dem.dem_sampling(H, torch.ones_like(p), 8, seed=1)
    expected = (H.sum(dim=1) % 2).to(torch.uint8)
    assert torch.equal(ones[0], expected)


def test_alternating_matrices_keep_their_own_samplers():
    """X and Z simulators alternate every step; neither may evict the other."""
    dem._reset_sampler_cache()
    Hx, px = _random_h(seed=1)
    Hz, pz = _random_h(seed=2)
    dem.dem_sampling(Hx, px, 32, seed=0)
    sx = dem._cached_sampler
    dem.dem_sampling(Hz, pz, 32, seed=1)
    sz = dem._cached_sampler
    assert sx is not sz
    for step in range(2, 8):
        H, p = (Hx, px) if step % 2 == 0 else (Hz, pz)
        dem.dem_sampling(H, p, 32, seed=step)
        assert dem._cached_sampler is (sx if step % 2 == 0 else sz)
    assert len(dem._sampler_cache) == 2


def test_cache_is_bounded_and_drops_collected_matrices():
    """Distinct H tensors from repeated runs must not accumulate GPU samplers."""
    import gc
    dem._reset_sampler_cache()
    kept = []
    for i in range(dem._SAMPLER_CACHE_MAX + 3):
        H, p = _random_h(seed=100 + i)
        dem.dem_sampling(H, p, 32, seed=i)
        kept.append((H, p))
    assert len(dem._sampler_cache) == dem._SAMPLER_CACHE_MAX
    # LRU: the oldest three were evicted, the newest are present.
    live = {id(h) for h, _ in kept[3:]}
    assert set(dem._sampler_cache) == live
    # A hit moves an entry to the most-recent end, so it survives the next insert.
    H_old, p_old = kept[3]
    dem.dem_sampling(H_old, p_old, 32, seed=1)
    H_new, p_new = _random_h(seed=999)
    dem.dem_sampling(H_new, p_new, 32, seed=2)
    assert id(H_old) in dem._sampler_cache and id(kept[4][0]) not in dem._sampler_cache
    # Dropping every reference to a matrix frees its entry on the next call.
    gone = id(H_new)
    del H_new, p_new
    gc.collect()
    dem.dem_sampling(H_old, p_old, 32, seed=3)
    assert gone not in dem._sampler_cache


def test_returned_outcomes_do_not_alias_the_sampler_buffer():
    """The caller's tensor must survive the sampler's next call.

    The sampler hands out CuPy-owned memory; if dem_sampling returned a view of
    it, the next sample() (same seed or not) could overwrite or reuse that
    block while the caller -- possibly a kernel queued on another stream -- is
    still reading it.
    """
    dem._reset_sampler_cache()
    H, p = _random_h(seed=7)
    first = dem.dem_sampling(H, p, 64, seed=0)
    snapshot = first.clone()
    torch.cuda.synchronize()
    for seed in (1, 2, 0, 3):
        dem.dem_sampling(H, p, 64, seed=seed)
    torch.cuda.synchronize()
    assert torch.equal(first, snapshot)
