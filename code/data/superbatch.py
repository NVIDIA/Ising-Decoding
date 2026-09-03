# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic resident superbatch buffering for Torch data generators."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class _BuiltSuperbatch:
    train_x: torch.Tensor
    train_y: torch.Tensor
    start_step: int
    batch_size: int
    num_slices: int
    timing: dict
    ready_event: Optional[torch.cuda.Event]
    offset: int = 0


class SuperbatchBuffer:
    """Build large GPU batches and serve deterministic minibatch views."""

    def __init__(
        self,
        *,
        superbatch_size: int,
        prefetch: bool,
        device: torch.device,
        build_fn: Callable[[int, int, bool], tuple],
    ) -> None:
        self.superbatch_size = int(superbatch_size or 0)
        if self.superbatch_size < 0:
            raise ValueError("superbatch_size must be non-negative")
        self.prefetch = bool(prefetch) and self.superbatch_size > 0
        self.device = torch.device(device)
        self._build_fn = build_fn
        self._lock = threading.RLock()
        self._resident: Optional[_BuiltSuperbatch] = None
        self._future: Optional[Future] = None
        self._future_key: Optional[tuple[int, int]] = None
        self._closed = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._producer_stream: Optional[torch.cuda.Stream] = None

        if self.active:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="superbatch")
            if self.device.type == "cuda":
                self._producer_stream = torch.cuda.Stream(device=self.device)

    @property
    def active(self) -> bool:
        return self.superbatch_size > 0

    def _validate_batch_size(self, batch_size: int) -> int:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.superbatch_size < batch_size:
            raise ValueError(
                f"superbatch_size={self.superbatch_size} is smaller than "
                f"batch_size={batch_size}"
            )
        if self.superbatch_size % batch_size:
            raise ValueError(
                f"superbatch_size={self.superbatch_size} must be divisible by "
                f"batch_size={batch_size}"
            )
        return batch_size

    def _build(self, start_step: int, batch_size: int, collect_timing: bool) -> _BuiltSuperbatch:
        build_t0 = time.perf_counter()
        ready_event = None
        if self.device.type == "cuda":
            assert self._producer_stream is not None
            with torch.cuda.device(self.device), torch.cuda.stream(self._producer_stream):
                train_x, train_y, timing = self._build_fn(
                    int(start_step), self.superbatch_size, collect_timing
                )
                ready_event = torch.cuda.Event()
                ready_event.record(self._producer_stream)
        else:
            train_x, train_y, timing = self._build_fn(
                int(start_step), self.superbatch_size, collect_timing
            )

        expected = self.superbatch_size
        if int(train_x.shape[0]) != expected or int(train_y.shape[0]) != expected:
            raise ValueError(
                "superbatch builder returned the wrong leading dimension: "
                f"train_x={tuple(train_x.shape)}, train_y={tuple(train_y.shape)}, "
                f"expected B={expected}"
            )
        timing = dict(timing or {})
        timing["superbatch_generate_s"] = time.perf_counter() - build_t0
        return _BuiltSuperbatch(
            train_x=train_x,
            train_y=train_y,
            start_step=int(start_step),
            batch_size=int(batch_size),
            num_slices=self.superbatch_size // int(batch_size),
            timing=timing,
            ready_event=ready_event,
        )

    def _submit_locked(self, start_step: int, batch_size: int, collect_timing: bool) -> None:
        if self._future is not None:
            return
        if self._executor is None:
            raise RuntimeError("superbatch executor is not available")
        self._future_key = (int(start_step), int(batch_size))
        self._future = self._executor.submit(
            self._build,
            int(start_step),
            int(batch_size),
            bool(collect_timing),
        )

    def _clear_locked(self) -> Optional[Future]:
        future = self._future
        self._resident = None
        self._future = None
        self._future_key = None
        return future

    def _reset_for_request(self) -> None:
        with self._lock:
            future = self._clear_locked()
        if future is not None:
            future.cancel()
            try:
                future.result()
            except Exception:
                pass

    def _activate(self, start_step: int, batch_size: int,
                  collect_timing: bool) -> tuple[_BuiltSuperbatch, float]:
        key = (int(start_step), int(batch_size))
        with self._lock:
            if self._future_key != key:
                stale = self._clear_locked()
            else:
                stale = None
            if stale is None and self._future is None:
                self._submit_locked(start_step, batch_size, collect_timing)
            future = self._future

        if stale is not None:
            stale.cancel()
            try:
                stale.result()
            except Exception:
                pass
            with self._lock:
                self._submit_locked(start_step, batch_size, collect_timing)
                future = self._future

        assert future is not None
        wait_t0 = time.perf_counter()
        built = future.result()
        wait_s = time.perf_counter() - wait_t0

        with self._lock:
            if self._future is future:
                self._future = None
                self._future_key = None
            self._resident = built
            if self.prefetch:
                next_step = built.start_step + built.num_slices
                self._submit_locked(next_step, built.batch_size, collect_timing)
        return built, wait_s

    def _wait_for_consumer(self, built: _BuiltSuperbatch) -> None:
        if built.ready_event is None:
            return
        with torch.cuda.device(self.device):
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(built.ready_event)
            built.train_x.record_stream(consumer)
            built.train_y.record_stream(consumer)

    def generate(
        self,
        *,
        step: int,
        batch_size: int,
        return_timing: bool,
        direct_fn: Callable[[int, int, bool], tuple],
    ):
        """Return one minibatch, building/refilling a resident batch as needed."""
        if not self.active:
            train_x, train_y, timing = direct_fn(int(step), int(batch_size), bool(return_timing))
            if return_timing:
                return train_x, train_y, timing
            return train_x, train_y
        if self._closed:
            raise RuntimeError("superbatch buffer is closed")

        batch_size = self._validate_batch_size(batch_size)
        step = int(step)
        call_t0 = time.perf_counter()
        wait_s = 0.0

        with self._lock:
            resident = self._resident
            expected_step = (
                resident.start_step + resident.offset if resident is not None else None
            )
            valid = (
                resident is not None and resident.batch_size == batch_size and expected_step == step
            )
        if not valid:
            if resident is not None:
                self._reset_for_request()
            resident, wait_s = self._activate(step, batch_size, bool(return_timing))

        self._wait_for_consumer(resident)
        slice_t0 = time.perf_counter()
        with self._lock:
            start = resident.offset * batch_size
            end = start + batch_size
            train_x = resident.train_x[start:end]
            train_y = resident.train_y[start:end]
            first_slice = resident.offset == 0
            resident.offset += 1
            if resident.offset >= resident.num_slices:
                self._resident = None
        slice_s = time.perf_counter() - slice_t0

        if return_timing:
            timing = dict(resident.timing) if first_slice else {}
            timing["generator_total_s"] = time.perf_counter() - call_t0
            timing["superbatch_wait_s"] = wait_s
            timing["superbatch_slice_s"] = slice_s
            timing["superbatch_size"] = self.superbatch_size
            return train_x, train_y, timing
        return train_x, train_y

    def release(self, *, wait: bool = True, keep_pending: bool = False) -> None:
        """Drop the resident superbatch.

        ``keep_pending=True`` leaves an in-flight prefetch alone: at an epoch
        boundary the prefetched build is exactly the next epoch's first
        superbatch whenever the step sequence continues and the batch size is
        unchanged, so it is served from the buffer instead of being awaited and
        thrown away. If the next request does not match it, ``generate`` cancels
        and rebuilds as usual.
        """
        with self._lock:
            if keep_pending:
                self._resident = None
                return
            future = self._clear_locked()
        if future is not None:
            future.cancel()
            if wait:
                try:
                    future.result()
                except Exception:
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self.release(wait=True)
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._closed = True


__all__ = ["SuperbatchBuffer"]
