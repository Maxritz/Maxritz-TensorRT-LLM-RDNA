# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backend-neutral token-step continuous batching lifecycle.

The executor supplies prefill/decode callbacks. This scheduler owns admission,
cancellation and fair one-token decode rounds, so a paged attention backend can
batch live sequences without changing HTTP request semantics.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .paged_kv import PagedSequence


Result = TypeVar("Result")


@dataclass
class DecodeRequest(Generic[Result]):
    prompt_token_ids: list[int]
    max_tokens: int
    sequence: PagedSequence = field(default_factory=PagedSequence)
    generated_token_ids: list[int] = field(default_factory=list)
    cancelled: bool = False
    future: asyncio.Future[Result] | None = None


class ContinuousBatchScheduler(Generic[Result]):
    """Admit requests while decoding prior ones in token-sized rounds.

    ``prefill`` and ``decode`` receive the live batch and must return completed
    results keyed by request identity. A decode callback normally appends exactly
    one token to every unfinished request and uses each request's `sequence` page
    table for paged attention.
    """

    def __init__(
        self,
        prefill: Callable[[list[DecodeRequest[Result]]], None],
        decode: Callable[[list[DecodeRequest[Result]]], dict[int, Result]],
        *,
        max_waiting: int = 128,
        max_active: int = 32,
    ) -> None:
        if min(max_waiting, max_active) < 1:
            raise ValueError("Continuous batch capacities must be positive")
        self.prefill = prefill
        self.decode = decode
        self.max_active = max_active
        self._waiting: asyncio.Queue[DecodeRequest[Result]] = asyncio.Queue(max_waiting)
        self._active: list[DecodeRequest[Result]] = []
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def submit(self, request: DecodeRequest[Result]) -> Result:
        if self._closed:
            raise RuntimeError("Continuous batch scheduler has been shut down")
        loop = asyncio.get_running_loop()
        request.future = loop.create_future()
        await self._waiting.put(request)
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="trtllm-rocm-continuous-batch")
        try:
            return await asyncio.shield(request.future)
        except asyncio.CancelledError:
            request.cancelled = True
            raise

    def _admit(self) -> None:
        new = []
        while len(self._active) + len(new) < self.max_active and not self._waiting.empty():
            request = self._waiting.get_nowait()
            self._waiting.task_done()
            if not request.cancelled:
                new.append(request)
        if new:
            self.prefill(new)
            self._active.extend(new)

    async def _run(self) -> None:
        while not self._closed:
            self._admit()
            self._active = [request for request in self._active if not request.cancelled]
            if not self._active:
                await asyncio.sleep(0)
                if self._waiting.empty():
                    self._worker = None
                    return
                continue
            completed = self.decode(self._active)
            survivors = []
            for request in self._active:
                result = completed.get(id(request))
                if (
                    result is not None
                    and request.future is not None
                    and not request.future.cancelled()
                ):
                    request.future.set_result(result)
                elif not request.cancelled:
                    survivors.append(request)
            self._active = survivors
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        self._closed = True
        for request in [*self._active]:
            request.cancelled = True
            if request.future is not None and not request.future.done():
                request.future.cancel()


__all__ = ["ContinuousBatchScheduler", "DecodeRequest"]
