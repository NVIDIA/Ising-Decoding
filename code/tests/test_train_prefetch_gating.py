# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the outer-batch-prefetch gating in ``training.train``.

Compiled HE generation must run on the main thread, so the outer
``ThreadPoolExecutor`` prefetch is disabled when the generator's sims use
``torch.compile``. These are CPU-only and use lightweight fakes.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

try:
    from training.train import (
        _generator_uses_compiled_generation,
        _should_use_outer_batch_prefetch,
    )
except Exception:  # heavy transitive imports may be unavailable on some runners
    pytest.skip("training.train not importable in this environment", allow_module_level=True)


def _gen(**sims):
    return types.SimpleNamespace(
        **{
            k: types.SimpleNamespace(use_compile=v) for k, v in sims.items()
        }
    )


def test_eager_single_sim_uses_prefetch():
    g = _gen(sim=False)
    assert _generator_uses_compiled_generation(g) is False
    assert _should_use_outer_batch_prefetch(g) is True


def test_compiled_single_sim_disables_prefetch():
    g = _gen(sim=True)
    assert _generator_uses_compiled_generation(g) is True
    assert _should_use_outer_batch_prefetch(g) is False


def test_compiled_in_either_basis_sim_disables_prefetch():
    # Mixed-basis generator: X compiled, Z not -> still compiled overall.
    g = _gen(sim_X=True, sim_Z=False)
    assert _generator_uses_compiled_generation(g) is True
    assert _should_use_outer_batch_prefetch(g) is False


def test_eager_both_bases_uses_prefetch():
    g = _gen(sim_X=False, sim_Z=False)
    assert _generator_uses_compiled_generation(g) is False
    assert _should_use_outer_batch_prefetch(g) is True


def test_generator_without_sims_defaults_to_prefetch():
    g = types.SimpleNamespace()
    assert _generator_uses_compiled_generation(g) is False
    assert _should_use_outer_batch_prefetch(g) is True
