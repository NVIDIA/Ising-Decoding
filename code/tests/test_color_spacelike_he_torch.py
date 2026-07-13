# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity tests for the Torch color-code spacelike HE.

These checks assert two invariants that hold for the default (heuristic)
spacelike pipeline: the map is a projector, and it preserves the syndrome.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

torch = pytest.importorskip("torch")

from qec.color_code import ColorCode  # noqa: E402
from qec.color_code.homological_equivalence_torch import (  # noqa: E402
    apply_homological_equivalence_color_torch,
    build_color_spacelike_he_cache,
    simplify_color_batched_torch,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rand_binary(rng: np.random.Generator, shape, p: float) -> np.ndarray:
    return (rng.random(shape) < p).astype(np.uint8)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_simplify_is_idempotent(distance: int) -> None:
    """A canonicalization map must be a projector: HE(HE(x)) == HE(x)."""
    device = _device()
    cc = ColorCode(distance)
    cache = build_color_spacelike_he_cache(cc, device=device)

    rng = np.random.default_rng(13 * distance + 1)
    errors_np = _rand_binary(rng, (32, cc.num_data), 0.3)
    errors = torch.as_tensor(errors_np, dtype=torch.uint8, device=device)

    once = simplify_color_batched_torch(errors, cache)
    twice = simplify_color_batched_torch(once, cache)
    torch.testing.assert_close(once, twice)


def test_apply_he_entrypoint_idempotent() -> None:
    """The (B, T, D) entrypoint must also be idempotent under repeated application."""
    device = _device()
    cc = ColorCode(5)
    cache = build_color_spacelike_he_cache(cc, device=device)

    rng = np.random.default_rng(2718)
    z_np = _rand_binary(rng, (4, 3, cc.num_data), 0.22)
    x_np = _rand_binary(rng, (4, 3, cc.num_data), 0.31)
    z = torch.as_tensor(z_np, dtype=torch.uint8, device=device)
    x = torch.as_tensor(x_np, dtype=torch.uint8, device=device)

    z1, x1 = apply_homological_equivalence_color_torch(z, x, cache)
    z2, x2 = apply_homological_equivalence_color_torch(z1, x1, cache)
    torch.testing.assert_close(z1, z2)
    torch.testing.assert_close(x1, x2)


def test_apply_he_entrypoint_preserves_syndrome() -> None:
    """HE must not change which stabilizers fire (syndrome ≡ parity_matrix @ error mod 2)."""
    device = _device()
    cc = ColorCode(5)
    cache = build_color_spacelike_he_cache(cc, device=device)

    rng = np.random.default_rng(31415)
    z_np = _rand_binary(rng, (5, 4, cc.num_data), 0.18)
    x_np = _rand_binary(rng, (5, 4, cc.num_data), 0.27)
    z = torch.as_tensor(z_np, dtype=torch.uint8, device=device)
    x = torch.as_tensor(x_np, dtype=torch.uint8, device=device)

    z_can, x_can = apply_homological_equivalence_color_torch(z, x, cache)

    parity = cache.parity_matrix.to(torch.float32)  # (P, D)

    def syndrome(err: torch.Tensor) -> torch.Tensor:
        B, T, D = err.shape
        flat = err.reshape(B * T, D).to(torch.float32)
        return ((flat @ parity.T).to(torch.int64)) % 2

    torch.testing.assert_close(syndrome(z), syndrome(z_can))
    torch.testing.assert_close(syndrome(x), syndrome(x_can))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
