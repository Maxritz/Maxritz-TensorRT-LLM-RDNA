# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

from tensorrt_llm.rocm.continuous import ContinuousBatchScheduler, DecodeRequest


def test_continuous_scheduler_interleaves_token_rounds():
    async def exercise():
        rounds = []

        def prefill(requests):
            rounds.append(("prefill", [request.prompt_token_ids[0] for request in requests]))

        def decode(requests):
            rounds.append(("decode", [request.prompt_token_ids[0] for request in requests]))
            completed = {}
            for request in requests:
                request.generated_token_ids.append(1)
                if len(request.generated_token_ids) == request.max_tokens:
                    completed[id(request)] = list(request.generated_token_ids)
            return completed

        scheduler = ContinuousBatchScheduler(prefill, decode, max_active=2)
        first = asyncio.create_task(scheduler.submit(DecodeRequest([10], 3)))
        await asyncio.sleep(0)
        second = asyncio.create_task(scheduler.submit(DecodeRequest([20], 2)))
        result = await asyncio.gather(first, second)
        await scheduler.aclose()
        return rounds, result

    rounds, results = asyncio.run(exercise())
    assert results == [[1, 1, 1], [1, 1]]
    assert ("decode", [10, 20]) in rounds
