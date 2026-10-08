# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async facade for the portable, serialized ROCm executor."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import AsyncIterator
from typing import Any

from .llm import LLM
from .sampling import RequestOutput, SamplingParams, StreamOutput
from .scheduler import SerializedRequestScheduler


class AsyncLLM(LLM):
    """`LLM` with awaitable execution and a bounded async streaming bridge.

    Generation remains serialized by ``LLM``'s model lock; this prevents blocking
    an ASGI event loop without falsely promising continuous batching. Ending an
    async consumer stops forwarding chunks, but cannot interrupt a currently
    executing Transformers ``generate`` call.
    """

    def __init__(self, *args: Any, max_queue_size: int = 32, **kwargs: Any) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        super().__init__(*args, **kwargs)
        self._max_queue_size = max_queue_size
        self._scheduler: SerializedRequestScheduler | None = None
        self._scheduler_loop: asyncio.AbstractEventLoop | None = None

    def _request_scheduler(self) -> SerializedRequestScheduler:
        loop = asyncio.get_running_loop()
        if self._scheduler is None:
            self._scheduler = SerializedRequestScheduler(self._max_queue_size)
            self._scheduler_loop = loop
        elif self._scheduler_loop is not loop:
            raise RuntimeError("An AsyncLLM instance must be used from one asyncio event loop")
        return self._scheduler

    async def generate_async(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        **kwargs: Any,
    ) -> list[RequestOutput]:
        scheduler = self._request_scheduler()
        return await scheduler.submit(lambda: self.generate(prompts, sampling_params, **kwargs))

    async def generate_stream_async(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        *,
        max_queue_size: int = 8,
    ) -> AsyncIterator[StreamOutput]:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamOutput | BaseException | None] = asyncio.Queue(max_queue_size)
        stopped = threading.Event()

        def put(item: StreamOutput | BaseException | None) -> bool:
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while True:
                try:
                    future.result(timeout=0.1)
                    return True
                except concurrent.futures.TimeoutError:
                    if stopped.is_set():
                        future.cancel()
                        return False

        def worker() -> None:
            try:
                for output in self.generate_stream(prompt, sampling_params):
                    if stopped.is_set() or not put(output):
                        return
            except BaseException as error:  # raised inside the awaiting task
                put(error)
            finally:
                put(None)

        thread = threading.Thread(target=worker, daemon=True, name="trtllm-rocm-async-stream")
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stopped.set()

    async def shutdown_async(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.aclose()
        await asyncio.to_thread(self.shutdown)


__all__ = ["AsyncLLM"]
