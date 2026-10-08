# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from tensorrt_llm.rocm.paged_kv import PagedKVCache


def test_paged_kv_shares_prefix_and_copy_on_writes_cpu():
    cache = PagedKVCache(
        num_layers=1,
        num_pages=4,
        page_size=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    original = cache.new_sequence()
    values = torch.arange(8, dtype=torch.float32).reshape(1, 4, 1, 2)
    cache.append(original, values, values + 10)
    cache.cache_prefix([1, 2, 3, 4], original)
    shared = cache.new_sequence([1, 2, 3, 4])
    extension = torch.full((1, 1, 1, 2), 99.0)
    cache.append(shared, extension, extension)

    original_key, _ = cache.materialize(original)
    shared_key, _ = cache.materialize(shared)
    assert original.length == 4 and shared.length == 5
    assert torch.equal(original_key, values)
    assert torch.equal(shared_key[:, :4], values)
    assert torch.equal(shared_key[:, 4:], extension)
