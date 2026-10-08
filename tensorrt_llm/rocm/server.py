# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI-compatible text/chat serving for the ROCm backend."""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from tensorrt_llm._config import StrictBaseModel

from .llm import LLM
from .sampling import RequestOutput, SamplingParams


class CompletionRequest(SamplingParams):
    model: str = Field(description="The served model name.")
    prompt: str | list[str] | list[int] | list[list[int]] = Field(
        description="Text or token-ID prompt(s)."
    )
    stream: bool = Field(
        default=False,
        description="Return OpenAI-compatible server-sent events for one completion.",
    )
    adapter: str | None = Field(default=None, description="Name of a loaded PEFT adapter.")


class ChatMessage(StrictBaseModel):
    role: Literal["system", "user", "assistant"] = Field(description="Conversation role.")
    content: str = Field(
        description="Text message content; multimodal and tool messages are not enabled."
    )


class ChatRequest(SamplingParams):
    model: str = Field(description="The served model name.")
    messages: list[ChatMessage] = Field(min_length=1, description="Conversation messages.")
    stream: bool = Field(
        default=False,
        description="Return OpenAI-compatible server-sent events for one completion.",
    )
    adapter: str | None = Field(default=None, description="Name of a loaded PEFT adapter.")


def _usage(results: list[RequestOutput]) -> dict[str, int]:
    prompt = sum(len(result.prompt_token_ids) for result in results)
    completion = sum(len(output.token_ids) for result in results for output in result.outputs)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _sse(payload: dict | str) -> str:
    """Encode one OpenAI server-sent event without trusting model text as JSON."""
    content = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {content}\n\n"


def create_app(engine: LLM, served_model_name: str | None = None) -> FastAPI:
    """Serve one engine; model execution is serialized by its lock and runs off-loop.

    Set TRTLLM_API_KEY to require a bearer token on /v1 routes. No key is embedded
    in profiling command metadata. The caller must provision TLS/access controls
    before exposing the default 0.0.0.0 listener to an untrusted network.
    """
    model_name = served_model_name or engine.model_id
    api_key = os.environ.get("TRTLLM_API_KEY")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            engine.shutdown()

    app = FastAPI(title="RDNA4 ROCm LLM", lifespan=lifespan)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if api_key is not None:
            candidate = (
                authorization.removeprefix("Bearer ")
                if authorization and authorization.startswith("Bearer ")
                else ""
            )
            if not hmac.compare_digest(candidate, api_key):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid API key",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    def check_request(request: CompletionRequest | ChatRequest) -> None:
        if request.model != model_name:
            raise HTTPException(status_code=404, detail=f"Model {request.model!r} is not served")

    def sampling(request: CompletionRequest | ChatRequest) -> SamplingParams:
        return SamplingParams(
            **request.model_dump(exclude={"model", "prompt", "messages", "stream", "adapter"})
        )

    async def generate(
        prompt, params: SamplingParams, adapter_name: str | None
    ) -> list[RequestOutput]:
        try:
            result = await run_in_threadpool(
                engine.generate, prompt, params, adapter_name=adapter_name
            )
            # ``streaming=False`` above guarantees the ordinary list surface.
            return result  # type: ignore[return-value]
        except (ValueError, TypeError, NotImplementedError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    def stream_events(
        prompt: str | list[int], params: SamplingParams, *, chat: bool, adapter_name: str | None
    ) -> Iterator[str]:
        """Bridge the synchronous portable streamer to OpenAI SSE framing."""
        stream_id = ("chatcmpl" if chat else "cmpl") + f"-{uuid.uuid4().hex}"
        created = int(time.time())
        if chat:
            yield _sse(
                {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
            )
        cancelled = threading.Event()
        try:
            for output in engine.generate_stream(
                prompt,
                params,
                cancellation_event=cancelled,
                adapter_name=adapter_name,
            ):
                if chat:
                    choice = {
                        "index": 0,
                        "delta": ({"content": output.text} if output.text else {}),
                        "finish_reason": output.finish_reason,
                    }
                    object_name = "chat.completion.chunk"
                else:
                    choice = {
                        "index": 0,
                        "text": output.text,
                        "finish_reason": output.finish_reason,
                        "logprobs": None,
                    }
                    object_name = "text_completion"
                yield _sse(
                    {
                        "id": stream_id,
                        "object": object_name,
                        "created": created,
                        "model": model_name,
                        "choices": [choice],
                    }
                )
        except (ValueError, TypeError, NotImplementedError) as error:
            yield _sse({"error": {"message": str(error), "type": type(error).__name__}})
        except RuntimeError as error:
            yield _sse({"error": {"message": str(error), "type": "service_unavailable"}})
        finally:
            cancelled.set()
        yield _sse("[DONE]")

    def streaming_response(
        prompt: str | list[int], params: SamplingParams, *, chat: bool, adapter_name: str | None
    ) -> StreamingResponse:
        if params.n != 1 or params.beam_width != 1:
            raise HTTPException(
                status_code=400,
                detail="Streaming supports n=1 and beam_width=1; use non-streaming for other modes",
            )
        return StreamingResponse(
            stream_events(prompt, params, chat=chat, adapter_name=adapter_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": "rocm",
            "model": model_name,
            "device": str(engine.device),
        }

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/completions", dependencies=[Depends(authenticate)])
    async def completions(request: CompletionRequest) -> object:
        check_request(request)
        params = sampling(request)
        if request.stream:
            if not isinstance(request.prompt, str) and not (
                isinstance(request.prompt, list)
                and request.prompt
                and all(
                    isinstance(token, int) and not isinstance(token, bool)
                    for token in request.prompt
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Streaming accepts one text prompt or one nonempty list of token IDs",
                )
            return streaming_response(
                request.prompt, params, chat=False, adapter_name=request.adapter
            )
        results = await generate(request.prompt, params, request.adapter)
        choices = [
            {
                "index": index,
                "text": output.text,
                "finish_reason": output.finish_reason,
                "logprobs": None,
            }
            for index, output in enumerate(
                output for result in results for output in result.outputs
            )
        ]
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": choices,
            "usage": _usage(results),
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    async def chat(request: ChatRequest) -> object:
        check_request(request)
        params = sampling(request)
        try:
            prompt = engine.tokenizer.apply_chat_template(
                [message.model_dump() for message in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"Tokenizer chat template required: {error}"
            ) from error
        if request.stream:
            return streaming_response(prompt, params, chat=True, adapter_name=request.adapter)
        results = await generate(prompt, params, request.adapter)
        choices = [
            {
                "index": output.index,
                "message": {"role": "assistant", "content": output.text},
                "finish_reason": output.finish_reason,
            }
            for output in results[0].outputs
        ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": choices,
            "usage": _usage(results),
        }

    return app


__all__ = ["create_app"]
