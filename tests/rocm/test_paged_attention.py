# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from tensorrt_llm.rocm.ops import attention
from tensorrt_llm.rocm.paged_attention import paged_attention
from tensorrt_llm.rocm.paged_kv import PagedKVCache


def test_paged_attention_matches_contiguous_reference_cpu():
    cache = PagedKVCache(
        num_layers=1,
        num_pages=8,
        page_size=2,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    sequences = [cache.new_sequence(), cache.new_sequence()]
    kv = []
    for sequence, tokens in zip(sequences, (3, 5)):
        keys = torch.randn(1, tokens, 2, 4)
        values = torch.randn_like(keys)
        cache.append(sequence, keys, values)
        kv.append((keys, values))
    query = torch.randn(2, 4, 1, 4)

    actual = paged_attention(query, cache, sequences, layer=0, backend="torch")
    expected = torch.cat(
        [
            attention(
                query[index : index + 1],
                key[0].transpose(0, 1).unsqueeze(0),
                value[0].transpose(0, 1).unsqueeze(0),
                causal=True,
                backend="torch",
            )
            for index, (key, value) in enumerate(kv)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_int8_paged_kv_materializes_close_to_source_cpu():
    cache = PagedKVCache(
        num_layers=1,
        num_pages=4,
        page_size=2,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
        quantization="int8",
    )
    sequence = cache.new_sequence()
    source = torch.randn(1, 3, 1, 8)
    cache.append(sequence, source, source)
    keys, values = cache.materialize(sequence)
    torch.testing.assert_close(keys, source, rtol=0.03, atol=0.03)
    torch.testing.assert_close(values, source, rtol=0.03, atol=0.03)
