# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

import pytest

from tensorrt_llm.rocm.scheduler import SerializedRequestScheduler

pytestmark = pytest.mark.cpu_only


def test_serialized_scheduler_orders_operations_and_skips_cancelled_work():
    async def exercise():
        scheduler = SerializedRequestScheduler(max_queue_size=2)
        started = threading.Event()
        release = threading.Event()
        calls = []

        def first():
            calls.append("first")
            started.set()
            assert release.wait(timeout=2)
            return "one"

        def second():
            calls.append("second")
            return "two"

        first_task = asyncio.create_task(scheduler.submit(first))
        await asyncio.to_thread(started.wait, 2)
        second_task = asyncio.create_task(scheduler.submit(second))
        await asyncio.sleep(0)
        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task
        release.set()
        assert await first_task == "one"
        await scheduler.aclose()
        return calls, scheduler

    calls, scheduler = asyncio.run(exercise())
    assert calls == ["first"]
    assert scheduler.cancelled == 1
