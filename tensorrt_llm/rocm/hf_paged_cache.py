# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformers Cache adapter backed by portable RDNA4 paged KV pages."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers.cache_utils import Cache

from .paged_kv import PagedKVCache, PagedSequence


class PagedDynamicCache(Cache):
    """A Transformers-compatible cache using one refcounted page pool per layer.

    Hugging Face attention modules still receive the standard contiguous key/value
    tensors returned by :meth:`update`. The cache lifecycle, prefill reuse and
    decode storage are paged already; replacing that materialization with
    ``triton_paged_decode`` is an attention-backend optimization, not a change to
    the model's cache contract.
    """

    def __init__(
        self,
        *,
        num_hidden_layers: int,
        num_pages: int,
        page_size: int,
        num_key_value_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        quantization: str = "none",
    ) -> None:
        super().__init__()
        self.num_layers = num_hidden_layers
        self.dtype = dtype
        self.device = torch.device(device)
        self._pools = [
            PagedKVCache(
                num_layers=1,
                num_pages=num_pages,
                page_size=page_size,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                quantization=quantization,
            )
            for _ in range(num_hidden_layers)
        ]
        self._sequences: list[list[PagedSequence]] = [[] for _ in range(num_hidden_layers)]
        self._seen_tokens = 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def _ensure_batch(self, batch_size: int) -> None:
        for layer, pool in enumerate(self._pools):
            sequences = self._sequences[layer]
            while len(sequences) < batch_size:
                sequences.append(pool.new_sequence())
            if len(sequences) != batch_size:
                raise ValueError("PagedDynamicCache batch size cannot shrink during a generation")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append `[batch, kv_heads, tokens, dim]` tensors and return HF layout."""
        del cache_kwargs
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is outside this cache")
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("Cache updates require matching [batch, heads, tokens, dim] K/V")
        if key_states.dtype != self.dtype or key_states.device != self.device:
            raise ValueError("Cache updates must match configured dtype and device")
        batch_size, _, tokens, _ = key_states.shape
        if tokens < 1:
            raise ValueError("Cache update requires at least one token")
        self._ensure_batch(batch_size)
        pool = self._pools[layer_idx]
        for batch in range(batch_size):
            pool.append(
                self._sequences[layer_idx][batch],
                key_states[batch].transpose(0, 1).unsqueeze(0),
                value_states[batch].transpose(0, 1).unsqueeze(0),
            )
        if layer_idx == 0:
            self._seen_tokens += tokens
        keys, values = zip(
            *(pool.materialize(sequence) for sequence in self._sequences[layer_idx])
        )
        lengths = {key.shape[1] for key in keys}
        if len(lengths) != 1:
            raise ValueError("Ragged cache rows require a paged attention backend")
        return (
            torch.cat([key[0].transpose(0, 1).unsqueeze(0) for key in keys], dim=0),
            torch.cat([value[0].transpose(0, 1).unsqueeze(0) for value in values], dim=0),
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._sequences[layer_idx]:
            return 0
        return self._sequences[layer_idx][0].length

    def get_max_cache_shape(self) -> int | None:
        return None

    def get_max_length(self) -> int | None:
        return self.get_max_cache_shape()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        indices = beam_idx.tolist()
        for layer, pool in enumerate(self._pools):
            previous = self._sequences[layer]
            if any(index < 0 or index >= len(previous) for index in indices):
                raise ValueError("beam index is outside the paged cache batch")
            reordered = [pool.clone(previous[index]) for index in indices]
            for sequence in previous:
                pool.release(sequence)
            self._sequences[layer] = reordered

    def release(self) -> None:
        for pool, sequences in zip(self._pools, self._sequences):
            for sequence in sequences:
                pool.release(sequence)
            sequences.clear()

    def cache_prefix(self, token_ids: Iterable[list[int]]) -> None:
        for layer, pool in enumerate(self._pools):
            for tokens, sequence in zip(token_ids, self._sequences[layer]):
                pool.cache_prefix(tokens, sequence)


__all__ = ["PagedDynamicCache"]
