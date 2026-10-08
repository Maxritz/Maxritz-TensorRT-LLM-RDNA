# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small bounded request scheduler used by the portable async facade.

This is intentionally a serialized lifecycle scheduler, not a paged-KV or
continuous-batching scheduler. It establishes truthful queueing and cancellation
semantics before adding a backend-specific batching executor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar


Result = TypeVar("Result")


@dataclass
class _Job(Generic[Result]):
    run: Callable[[], Result]
    future: asyncio.Future[Result]


class SerializedRequestScheduler:
    """Bound pending requests and execute blocking model calls one at a time."""

    def __init__(self, max_queue_size: int = 32) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(max_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self.submitted = 0
        self.completed = 0
        self.cancelled = 0

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="trtllm-rocm-request-scheduler")

    async def submit(self, operation: Callable[[], Result]) -> Result:
        if self._closed:
            raise RuntimeError("Request scheduler has been shut down")
        self._ensure_worker()
        future: asyncio.Future[Result] = asyncio.get_running_loop().create_future()
        job = _Job(operation, future)
        self.submitted += 1
        await self._queue.put(job)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Queued jobs are skipped. Active blocking generation cannot safely
            # be preempted by Python, but its result is discarded.
            future.cancel()
            self.cancelled += 1
            raise

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                if job.future.cancelled():
                    continue
                try:
                    result = await asyncio.to_thread(job.run)
                except BaseException as error:
                    if not job.future.cancelled():
                        job.future.set_exception(error)
                else:
                    if not job.future.cancelled():
                        job.future.set_result(result)
                        self.completed += 1
            finally:
                self._queue.task_done()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker


__all__ = ["SerializedRequestScheduler"]
