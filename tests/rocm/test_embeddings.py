# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from tensorrt_llm.rocm.embeddings import EmbeddingLLM
from tensorrt_llm.rocm.validation import tiny_model_and_tokenizer


def test_embedding_llm_pools_and_normalizes_tiny_cpu_model():
    model, tokenizer = tiny_model_and_tokenizer()
    with EmbeddingLLM(model, tokenizer, device="cpu", dtype="float32", max_batch_size=2) as engine:
        results = engine.encode(["hello", "world"])

    assert [result.index for result in results] == [0, 1]
    assert all(len(result.token_ids) > 0 for result in results)
    assert all(len(result.embedding) == model.config.n_embd for result in results)
    assert all(
        math.isclose(sum(value * value for value in result.embedding), 1.0, rel_tol=1e-5)
        for result in results
    )


def test_embedding_llm_accepts_token_ids():
    model, tokenizer = tiny_model_and_tokenizer()
    with EmbeddingLLM(model, tokenizer, device="cpu", dtype="float32") as engine:
        result = engine.encode([[2, 3, 4]])[0]

    assert result.token_ids == [2, 3, 4]
