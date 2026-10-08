# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable encoder/embedding execution for the ROCm backend."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trtllm_profile import component, trace_active

from .runtime import resolve_device, resolve_dtype


@dataclass
class EmbeddingOutput:
    index: int
    embedding: list[float]
    token_ids: list[int]


class EmbeddingLLM:
    """Run an HF encoder (or decoder hidden-state) model as a local embedder.

    This is a correctness-first, single-device path. It deliberately uses normal
    PyTorch operators so it runs in explicit CPU reference mode and on HIP. It
    is not the CUDA-graph/dynamic-batching encoder executor from upstream.
    """

    def __init__(
        self,
        model: str | Path | PreTrainedModel,
        tokenizer: str | Path | PreTrainedTokenizerBase | None = None,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "auto",
        max_batch_size: int = 1,
        pooling: Literal["mean", "cls", "last_token"] = "mean",
        normalize: bool = True,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self._max_batch_size = max_batch_size
        self.pooling = pooling
        self.normalize = normalize
        self._lock = threading.RLock()
        self._closed = False
        self.last_stats: dict[str, float | int] = {}
        self.model_id = (
            str(model)
            if isinstance(model, (str, Path))
            else model.config.name_or_path or type(model).__name__
        )
        loading = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        with trace_active(), torch.inference_mode(False):
            self.model = (
                model
                if isinstance(model, PreTrainedModel)
                else AutoModel.from_pretrained(str(model), torch_dtype=self.dtype, **loading)
            )
            self.model = self.model.eval().to(device=self.device, dtype=self.dtype)
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = tokenizer
            else:
                source = str(tokenizer) if tokenizer is not None else self.model_id
                self.tokenizer = AutoTokenizer.from_pretrained(source, **loading)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer must define a pad token or an EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare(
        self, inputs: str | list[str] | list[int] | list[list[int]]
    ) -> tuple[list[str], list[list[int]], dict[str, torch.Tensor]]:
        if isinstance(inputs, str):
            prompts: list[str] | list[list[int]] = [inputs]
        elif isinstance(inputs, list) and inputs and isinstance(inputs[0], int):
            prompts = [inputs]
        elif isinstance(inputs, list) and inputs:
            prompts = inputs
        else:
            raise ValueError("Embedding input must contain text or token IDs")
        if isinstance(prompts[0], str):
            if not all(isinstance(item, str) for item in prompts):
                raise TypeError("All embedding inputs must have the same type")
            encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
            token_ids = [
                ids[mask.bool()].tolist()
                for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
            ]
            texts = list(prompts)
        else:
            if not all(
                isinstance(item, list)
                and item
                and all(isinstance(token, int) and not isinstance(token, bool) for token in item)
                for item in prompts
            ):
                raise ValueError("Token embedding inputs must be nonempty integer ID lists")
            token_ids = [list(item) for item in prompts]
            width = max(map(len, token_ids))
            encoded = {
                "input_ids": torch.tensor(
                    [ids + [self.tokenizer.pad_token_id] * (width - len(ids)) for ids in token_ids]
                ),
                "attention_mask": torch.tensor(
                    [[1] * len(ids) + [0] * (width - len(ids)) for ids in token_ids]
                ),
            }
            texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        with component("transfer"):
            tensors = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        return texts, token_ids, tensors

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        if self.pooling == "mean":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        if self.pooling == "cls":
            first = torch.where(mask.bool(), positions, mask.shape[1]).min(dim=1).values
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), first]
        last = torch.where(mask.bool(), positions, -1).max(dim=1).values
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]

    def encode(
        self, inputs: str | list[str] | list[int] | list[list[int]]
    ) -> list[EmbeddingOutput]:
        with self._lock, trace_active(), torch.inference_mode():
            if self._closed:
                raise RuntimeError("EmbeddingLLM has been shut down")
            started = time.perf_counter()
            outputs: list[EmbeddingOutput] = []
            input_count = 1 if isinstance(inputs, str) else len(inputs)
            for offset in range(0, input_count, self._max_batch_size):
                batch = (
                    inputs
                    if isinstance(inputs, str)
                    else inputs[offset : offset + self._max_batch_size]
                )
                _, token_ids, tensors = self._prepare(batch)
                with component("embed"):
                    result = self.model(**tensors, output_hidden_states=True, return_dict=True)
                    hidden = getattr(result, "last_hidden_state", None)
                    if hidden is None:
                        states = getattr(result, "hidden_states", None)
                        if not states:
                            raise RuntimeError(
                                "Model does not expose hidden states required for embeddings"
                            )
                        hidden = states[-1]
                    vectors = self._pool(hidden, tensors["attention_mask"])
                    if self.normalize:
                        vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)
                for ids, vector in zip(token_ids, vectors.cpu()):
                    outputs.append(EmbeddingOutput(len(outputs), vector.tolist(), ids))
            elapsed = time.perf_counter() - started
            input_tokens = sum(len(output.token_ids) for output in outputs)
            self.last_stats = {
                "wall_s": elapsed,
                "input_tokens": input_tokens,
                "embeddings": len(outputs),
                "embeddings_per_s": len(outputs) / elapsed if elapsed else 0.0,
            }
            return outputs

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self.model = None

    def __enter__(self) -> "EmbeddingLLM":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()


__all__ = ["EmbeddingLLM", "EmbeddingOutput"]
