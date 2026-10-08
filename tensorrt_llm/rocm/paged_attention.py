# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged-KV attention execution with a portable PyTorch/HIP fallback.

This path deliberately materializes only the logical pages required by each
sequence and then dispatches the tested ROCm attention primitive. It is a
correctness implementation for page-table semantics. A Triton/HIP decode kernel
can replace the per-sequence materialization behind the same function after
RDNA4 qualification.
"""

from __future__ import annotations

import torch

from .ops import attention
from .paged_kv import PagedKVCache, PagedSequence
from .runtime import KernelBackend


def paged_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    sequences: list[PagedSequence],
    *,
    layer: int,
    causal: bool = True,
    query_start: int | list[int] | None = None,
    scale: float | None = None,
    backend: KernelBackend = "torch",
    implementation: str = "portable",
) -> torch.Tensor:
    """Attend `[batch, query_heads, query_tokens, dim]` over paged K/V storage.

    ``query_start`` is the logical KV position of each query row. Leaving it as
    ``None`` right-aligns decode queries to the materialized KV length. Each
    sequence may have a distinct number of pages, which is why this reference
    implementation dispatches one ragged row at a time.
    """
    if query.ndim != 4 or query.shape[0] != len(sequences):
        raise ValueError("query batch dimension must equal the number of paged sequences")
    if implementation not in ("portable", "triton"):
        raise ValueError("implementation must be portable or triton")
    if not 0 <= layer < cache.num_layers:
        raise ValueError("layer is outside this paged KV cache")
    if query.dtype != cache.dtype or query.device != cache.keys.device:
        raise ValueError("query must match the cache logical dtype and device")
    starts: list[int | None]
    if query_start is None or isinstance(query_start, int):
        starts = [query_start] * len(sequences)
    else:
        if len(query_start) != len(sequences):
            raise ValueError("query_start must have one value per sequence")
        starts = list(query_start)
    if implementation == "triton":
        if query.shape[2] != 1 or not causal or any(start is not None for start in starts):
            raise NotImplementedError("Triton paged decode supports one right-aligned causal query")
        if cache.quantization != "none":
            raise NotImplementedError("Triton paged decode does not yet support INT8 KV pages")
        from .triton_paged_decode import triton_paged_decode

        table, lengths = cache.page_table(sequences)
        return triton_paged_decode(
            query[:, :, 0], cache.keys[layer], cache.values[layer], table, lengths, scale=scale
        ).unsqueeze(2)
    outputs = []
    for row, (sequence, start) in enumerate(zip(sequences, starts)):
        if sequence.length < 1:
            raise ValueError("Paged attention requires at least one KV token per sequence")
        keys, values = cache.materialize(sequence)
        key = keys[layer].transpose(0, 1).unsqueeze(0)
        value = values[layer].transpose(0, 1).unsqueeze(0)
        outputs.append(
            attention(
                query[row : row + 1],
                key,
                value,
                causal=causal,
                query_start=start,
                scale=scale,
                backend=backend,
            )
        )
    return torch.cat(outputs, dim=0)


__all__ = ["paged_attention"]
