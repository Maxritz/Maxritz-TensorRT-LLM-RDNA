# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi.testclient import TestClient

from tensorrt_llm.rocm.embedding_server import create_embedding_app
from tensorrt_llm.rocm.embeddings import EmbeddingLLM
from tensorrt_llm.rocm.validation import tiny_model_and_tokenizer


def test_openai_embedding_endpoint_tiny_cpu(monkeypatch):
    monkeypatch.delenv("TRTLLM_API_KEY", raising=False)
    model, tokenizer = tiny_model_and_tokenizer()
    engine = EmbeddingLLM(model, tokenizer, device="cpu", dtype="float32")
    with TestClient(create_embedding_app(engine, "tiny-embed")) as client:
        response = client.post(
            "/v1/embeddings", json={"model": "tiny-embed", "input": ["hello", "world"]}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert [entry["index"] for entry in body["data"]] == [0, 1]
        assert body["usage"]["total_tokens"] > 0
        wrong_model = client.post(
            "/v1/embeddings", json={"model": "not-here", "input": "hello"}
        )
        assert wrong_model.status_code == 404
