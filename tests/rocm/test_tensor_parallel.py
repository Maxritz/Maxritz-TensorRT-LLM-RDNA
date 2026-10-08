# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from tensorrt_llm.rocm.distributed import RCCLContext
from tensorrt_llm.rocm.tensor_parallel import ColumnParallelLinear, RowParallelLinear


def test_tensor_parallel_linear_world_one_matches_dense_cpu(monkeypatch):
    context = RCCLContext.__new__(RCCLContext)
    context.device = torch.device("cpu")
    monkeypatch.setattr(RCCLContext, "world_size", property(lambda _self: 1))
    monkeypatch.setattr(RCCLContext, "rank", property(lambda _self: 0))
    linear = torch.nn.Linear(4, 6)
    inputs = torch.randn(3, 4)
    torch.testing.assert_close(ColumnParallelLinear(linear, context)(inputs), linear(inputs))
    torch.testing.assert_close(RowParallelLinear(linear, context)(inputs), linear(inputs))
