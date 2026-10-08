# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable paged KV storage and page-reference prefix sharing.

The class is a backend-neutral cache allocator. It is intentionally separate from
Transformers' opaque ``past_key_values`` format; an attention backend must opt in
before this storage can be used for decode. This keeps page ownership/copy-on-write
correct on CPU and HIP instead of pretending a CUDA paged-attention plugin works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PagedSequence:
    """Logical sequence state backed by physical page IDs."""

    page_ids: list[int] = field(default_factory=list)
    length: int = 0


class PagedKVCache:
    """Fixed-size K/V page pool with refcounted prefix sharing and copy-on-write."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        quantization: str = "none",
    ) -> None:
        if min(num_layers, num_pages, page_size, num_kv_heads, head_dim) < 1:
            raise ValueError("Paged KV dimensions must be positive")
        if quantization not in ("none", "int8"):
            raise ValueError("Paged KV quantization must be none or int8")
        self.page_size = page_size
        self.num_pages = num_pages
        self.num_layers = num_layers
        self.dtype = dtype
        self.quantization = quantization
        shape = (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        storage_dtype = torch.int8 if quantization == "int8" else dtype
        self.keys = torch.empty(shape, dtype=storage_dtype, device=device)
        self.values = torch.empty_like(self.keys)
        scale_shape = (*shape[:-1], 1)
        self.key_scales = (
            torch.empty(scale_shape, dtype=torch.float32, device=device)
            if quantization == "int8"
            else None
        )
        self.value_scales = (
            torch.empty_like(self.key_scales) if self.key_scales is not None else None
        )
        self._free = list(range(num_pages - 1, -1, -1))
        self._refs = [0] * num_pages
        self._prefixes: dict[tuple[int, ...], PagedSequence] = {}

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def _allocate(self) -> int:
        if not self._free:
            raise RuntimeError("Paged KV cache is full")
        page = self._free.pop()
        self._refs[page] = 1
        return page

    def _retain(self, page: int) -> None:
        self._refs[page] += 1

    def _release(self, page: int) -> None:
        if self._refs[page] < 1:
            raise RuntimeError("Paged KV page refcount underflow")
        self._refs[page] -= 1
        if self._refs[page] == 0:
            self._free.append(page)

    def new_sequence(self, prefix_tokens: list[int] | None = None) -> PagedSequence:
        """Create a sequence, sharing a cached full-page prefix when available."""
        if prefix_tokens is None:
            return PagedSequence()
        cached = self._prefixes.get(tuple(prefix_tokens))
        if cached is None:
            return PagedSequence()
        for page in cached.page_ids:
            self._retain(page)
        return PagedSequence(list(cached.page_ids), cached.length)

    def clone(self, sequence: PagedSequence) -> PagedSequence:
        """Retain page references for a beam/cache branch."""
        for page in sequence.page_ids:
            self._retain(page)
        return PagedSequence(list(sequence.page_ids), sequence.length)

    def release(self, sequence: PagedSequence) -> None:
        for page in sequence.page_ids:
            self._release(page)
        sequence.page_ids.clear()
        sequence.length = 0

    def cache_prefix(self, token_ids: list[int], sequence: PagedSequence) -> None:
        """Retain only page-aligned prefixes; partial pages need copy-on-write."""
        aligned = sequence.length - (sequence.length % self.page_size)
        if aligned == 0 or aligned > len(token_ids):
            return
        pages = sequence.page_ids[: aligned // self.page_size]
        key = tuple(token_ids[:aligned])
        old = self._prefixes.pop(key, None)
        if old is not None:
            self.release(old)
        for page in pages:
            self._retain(page)
        self._prefixes[key] = PagedSequence(list(pages), aligned)

    def append(self, sequence: PagedSequence, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append `[layers, tokens, kv_heads, head_dim]` K/V tensors."""
        expected = self.keys.shape[0], self.keys.shape[3], self.keys.shape[4]
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError(
                "K/V append tensors must have matching [layers, tokens, heads, dim] shape"
            )
        if (keys.shape[0], keys.shape[2], keys.shape[3]) != expected:
            raise ValueError("K/V append shape does not match this page pool")
        if keys.device != self.keys.device or keys.dtype != self.dtype:
            raise ValueError("K/V append tensors must match pool device and logical dtype")
        for offset in range(keys.shape[1]):
            page_offset = sequence.length % self.page_size
            if page_offset == 0:
                sequence.page_ids.append(self._allocate())
            elif self._refs[sequence.page_ids[-1]] > 1:
                old_page = sequence.page_ids[-1]
                new_page = self._allocate()
                self.keys[:, new_page, :page_offset].copy_(self.keys[:, old_page, :page_offset])
                self.values[:, new_page, :page_offset].copy_(self.values[:, old_page, :page_offset])
                if self.key_scales is not None:
                    self.key_scales[:, new_page, :page_offset].copy_(
                        self.key_scales[:, old_page, :page_offset]
                    )
                    self.value_scales[:, new_page, :page_offset].copy_(
                        self.value_scales[:, old_page, :page_offset]
                    )
                sequence.page_ids[-1] = new_page
                self._release(old_page)
            page = sequence.page_ids[-1]
            if self.key_scales is None:
                self.keys[:, page, page_offset].copy_(keys[:, offset])
                self.values[:, page, page_offset].copy_(values[:, offset])
            else:
                self._quantize(
                    keys[:, offset],
                    self.keys[:, page, page_offset],
                    self.key_scales[:, page, page_offset],
                )
                self._quantize(
                    values[:, offset],
                    self.values[:, page, page_offset],
                    self.value_scales[:, page, page_offset],
                )
            sequence.length += 1

    @staticmethod
    def _quantize(source: torch.Tensor, destination: torch.Tensor, scale: torch.Tensor) -> None:
        value = source.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
        destination.copy_(torch.round(source.float() / value).clamp_(-127, 127).to(torch.int8))
        scale.copy_(value)

    def page_table(self, sequences: list[PagedSequence]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device page IDs and logical lengths for paged decode kernels."""
        if not sequences:
            raise ValueError("page_table requires at least one sequence")
        width = max(len(sequence.page_ids) for sequence in sequences)
        table = torch.full(
            (len(sequences), width), -1, dtype=torch.int32, device=self.keys.device
        )
        for row, sequence in enumerate(sequences):
            if sequence.page_ids:
                table[row, : len(sequence.page_ids)] = torch.tensor(
                    sequence.page_ids, dtype=torch.int32, device=self.keys.device
                )
        lengths = torch.tensor(
            [sequence.length for sequence in sequences], dtype=torch.int32, device=self.keys.device
        )
        return table, lengths

    def materialize(self, sequence: PagedSequence) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous `[layers, tokens, heads, dim]` tensors for a backend adapter."""
        key_parts, value_parts = [], []
        remaining = sequence.length
        for page in sequence.page_ids:
            width = min(remaining, self.page_size)
            keys = self.keys[:, page, :width]
            values = self.values[:, page, :width]
            if self.key_scales is not None:
                keys = (keys.float() * self.key_scales[:, page, :width]).to(self.dtype)
                values = (values.float() * self.value_scales[:, page, :width]).to(self.dtype)
            key_parts.append(keys)
            value_parts.append(values)
            remaining -= width
        if not key_parts:
            empty = torch.empty(
                (self.num_layers, 0, self.keys.shape[3], self.keys.shape[4]),
                dtype=self.dtype,
                device=self.keys.device,
            )
            return empty, empty.clone()
        return torch.cat(key_parts, dim=1), torch.cat(value_parts, dim=1)


__all__ = ["PagedKVCache", "PagedSequence"]
