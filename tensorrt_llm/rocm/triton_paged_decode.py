# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional Triton paged-KV decode kernel for RDNA4 qualification.

This is intentionally opt-in: it requires an AMD Triton installation and a real
RDNA4 numerical/performance qualification. The portable paged_attention fallback
remains the safe default.
"""

from __future__ import annotations

import torch


def triton_paged_decode(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one query token per sequence from `[pages, page, kv_heads, dim]` K/V.

    Supports FP16/BF16 input, GQA and page-table indirection. The head dimension
    is compiled as a Triton constexpr and must be <=256; the sequence bound is
    the padded page-table width times page size.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise RuntimeError("triton_paged_decode requires an AMD Triton installation") from error
    if query.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("Expected Q [batch, heads, dim] and K/V [pages, page, kv_heads, dim]")
    batch, query_heads, head_dim = query.shape
    if (
        keys.shape[-1] != head_dim
        or query_heads % keys.shape[2]
        or page_table.shape[0] != batch
        or lengths.shape != (batch,)
    ):
        raise ValueError("Invalid GQA/page-table dimensions")
    if head_dim > 256 or head_dim & (head_dim - 1):
        raise ValueError("Triton paged decode requires a power-of-two head dimension <=256")
    if query.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("triton_paged_decode requires a HIP device")
    if any(t.device != query.device for t in (keys, values, page_table, lengths)):
        raise ValueError("Paged decode tensors must share one HIP device")
    if page_table.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("page_table and lengths must be integer tensors")
    page_size, kv_heads = keys.shape[1], keys.shape[2]
    max_tokens = page_table.shape[1] * page_size
    if max_tokens > 8192:
        raise ValueError("Triton paged decode qualification currently caps max tokens at 8192")

    @triton.jit
    def kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        table_ptr,
        lengths_ptr,
        out_ptr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kp: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_tb: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        SCALE: tl.constexpr,
        PAGE: tl.constexpr,
        GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_TOKENS: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + batch_index * stride_qb + head * stride_qh + offsets).to(tl.float32)
        length = tl.load(lengths_ptr + batch_index)
        kv_head = head // GROUP
        maximum = -float("inf")
        denominator = 0.0
        accumulator = tl.zeros((HEAD_DIM,), tl.float32)
        for token in range(0, MAX_TOKENS):
            page_slot = token // PAGE
            page = tl.load(table_ptr + batch_index * stride_tb + page_slot)
            valid = (token < length) & (page >= 0)
            offset = page * stride_kp + (token % PAGE) * stride_kt + kv_head * stride_kh + offsets
            key = tl.load(k_ptr + offset, mask=valid, other=0.0).to(tl.float32)
            value = tl.load(v_ptr + offset, mask=valid, other=0.0).to(tl.float32)
            score = tl.sum(q * key, axis=0) * SCALE
            score = tl.where(valid, score, -float("inf"))
            next_maximum = tl.maximum(maximum, score)
            alpha = tl.exp(maximum - next_maximum)
            beta = tl.exp(score - next_maximum)
            accumulator = accumulator * alpha + value * beta
            denominator = denominator * alpha + beta
            maximum = next_maximum
        result = accumulator / denominator
        tl.store(out_ptr + batch_index * stride_ob + head * stride_oh + offsets, result)

    output = torch.empty_like(query)
    kernel[(batch, query_heads)](
        query,
        keys,
        values,
        page_table,
        lengths,
        output,
        query.stride(0),
        query.stride(1),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        page_table.stride(0),
        output.stride(0),
        output.stride(1),
        scale if scale is not None else head_dim**-0.5,
        PAGE=page_size,
        GROUP=query_heads // kv_heads,
        HEAD_DIM=head_dim,
        MAX_TOKENS=max_tokens,
        num_warps=4,
    )
    return output


__all__ = ["triton_paged_decode"]
