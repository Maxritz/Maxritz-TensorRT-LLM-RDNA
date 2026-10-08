# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from tensorrt_llm.rocm.hf_paged_cache import PagedDynamicCache


def test_hf_paged_cache_updates_and_reorders_cpu():
    cache = PagedDynamicCache(
        num_hidden_layers=2,
        num_pages=8,
        page_size=2,
        num_key_value_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    key = torch.randn(2, 1, 2, 4)
    value = torch.randn_like(key)
    returned_key, returned_value = cache.update(key, value, 0)
    torch.testing.assert_close(returned_key, key)
    torch.testing.assert_close(returned_value, value)
    next_key, next_value = torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4)
    returned_key, returned_value = cache.update(next_key, next_value, 0)
    torch.testing.assert_close(returned_key, torch.cat((key, next_key), dim=2))
    torch.testing.assert_close(returned_value, torch.cat((value, next_value), dim=2))
    cache.reorder_cache(torch.tensor([1, 0]))
    returned_key, _ = cache.update(torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4), 0)
    assert returned_key.shape == (2, 1, 4, 4)
    cache.release()
