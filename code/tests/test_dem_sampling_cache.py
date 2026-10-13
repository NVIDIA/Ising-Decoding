# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for concurrent sampler-cache cleanup."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

import pytest
import torch

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from qec import dem_sampling as dem  # noqa: E402


@pytest.mark.parametrize("concurrent_operation", ["evict", "touch", "reset"])
def test_cleanup_survives_concurrent_cache_operations(concurrent_operation):
    """Native sampler destruction releases the GIL during stale-entry deletion."""
    dem._reset_sampler_cache()
    destroying = threading.Event()
    release_destructor = threading.Event()
    second_started = threading.Event()
    second_finished = threading.Event()

    class YieldingSampler:

        def __del__(self):
            destroying.set()
            release_destructor.wait(timeout=5)

    old = [torch.zeros((2, 2), dtype=torch.uint8) for _ in range(2)]
    dem._touch_sampler_entry(old[0]).sampler = YieldingSampler()
    dem._touch_sampler_entry(old[1])
    hx = torch.zeros((2, 2), dtype=torch.uint8)
    hz = torch.ones((2, 2), dtype=torch.uint8)
    del old

    def competing_operation():
        second_started.set()
        try:
            if concurrent_operation == "evict":
                dem._evict_stale_sampler_entries()
            elif concurrent_operation == "touch":
                dem._touch_sampler_entry(hz)
            else:
                dem._reset_sampler_cache()
        finally:
            second_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(dem._touch_sampler_entry, hx)
            try:
                assert destroying.wait(timeout=5)
                second = executor.submit(competing_operation)
                assert second_started.wait(timeout=5)
                # Without serialization the second operation removes the next
                # stale key. With serialization it waits for the destructor.
                second_finished.wait(timeout=0.1)
            finally:
                release_destructor.set()
            assert first.result(timeout=5).H is hx
            second.result(timeout=5)

        expected = set() if concurrent_operation == "reset" else {id(hx)}
        if concurrent_operation == "touch":
            expected.add(id(hz))
        assert set(dem._sampler_cache) == expected
    finally:
        release_destructor.set()
        dem._reset_sampler_cache()
