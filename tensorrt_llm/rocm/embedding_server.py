# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible embedding service for the portable ROCm embedder."""

from __future__ import annotations

import hmac
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from tensorrt_llm._config import StrictBaseModel

from .embeddings import EmbeddingLLM


class EmbeddingRequest(StrictBaseModel):
    model: str = Field(description="The served model name.")
    input: str | list[str] | list[int] | list[list[int]] = Field(
        description="Text or nonempty token-ID inputs."
    )
    encoding_format: Literal["float"] = "float"
    dimensions: None = Field(
        default=None,
        description="Projection to a requested dimensionality is not enabled on this backend.",
    )
    user: str | None = None


def create_embedding_app(engine: EmbeddingLLM, served_model_name: str | None = None) -> FastAPI:
    """Create a single-model `/v1/embeddings` server with optional bearer auth."""
    model_name = served_model_name or engine.model_id
    api_key = os.environ.get("TRTLLM_API_KEY")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            engine.shutdown()

    app = FastAPI(title="RDNA4 ROCm embeddings", lifespan=lifespan)

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        supplied = (
            authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        )
        if not hmac.compare_digest(supplied, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/embeddings", dependencies=[Depends(authorize)])
    async def embeddings(request: EmbeddingRequest) -> dict:
        if request.model != model_name:
            raise HTTPException(status_code=404, detail=f"Model {request.model!r} is not served")
        try:
            results = await run_in_threadpool(engine.encode, request.input)
        except (TypeError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        input_tokens = sum(len(result.token_ids) for result in results)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": result.embedding, "index": result.index}
                for result in results
            ],
            "model": model_name,
            "usage": {"prompt_tokens": input_tokens, "total_tokens": input_tokens},
            "created": int(time.time()),
            "id": f"embd-{uuid.uuid4().hex}",
        }

    return app


__all__ = ["EmbeddingRequest", "create_embedding_app"]
