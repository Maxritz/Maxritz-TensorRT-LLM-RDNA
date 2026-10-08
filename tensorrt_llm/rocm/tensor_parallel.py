# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""RCCL tensor-parallel linear building blocks for ROCm model integration."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .distributed import RCCLContext


class ColumnParallelLinear(torch.nn.Module):
    """Shard output features; all-gather returns the original dense output layout."""

    def __init__(self, linear: torch.nn.Linear, context: RCCLContext) -> None:
        super().__init__()
        if linear.out_features % context.world_size:
            raise ValueError("Output features must divide evenly across tensor-parallel ranks")
        self.context = context
        width = linear.out_features // context.world_size
        start = context.rank * width
        self.weight = torch.nn.Parameter(linear.weight.detach()[start : start + width].clone())
        self.bias = (
            torch.nn.Parameter(linear.bias.detach()[start : start + width].clone())
            if linear.bias is not None
            else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        local = functional.linear(input, self.weight, self.bias)
        if self.context.world_size == 1:
            return local
        return torch.cat(self.context.all_gather(local), dim=-1)


class RowParallelLinear(torch.nn.Module):
    """Shard input features; RCCL all-reduce produces the dense output layout."""

    def __init__(self, linear: torch.nn.Linear, context: RCCLContext) -> None:
        super().__init__()
        if linear.in_features % context.world_size:
            raise ValueError("Input features must divide evenly across tensor-parallel ranks")
        self.context = context
        width = linear.in_features // context.world_size
        start = context.rank * width
        self.weight = torch.nn.Parameter(linear.weight.detach()[:, start : start + width].clone())
        self.bias = (
            torch.nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] % self.context.world_size:
            raise ValueError(
                "Input hidden dimension must divide evenly across tensor-parallel ranks"
            )
        width = input.shape[-1] // self.context.world_size
        shard = input[..., self.context.rank * width : (self.context.rank + 1) * width]
        local = functional.linear(shard, self.weight)
        if self.context.world_size == 1:
            return local + self.bias if self.bias is not None else local
        result = self.context.all_reduce(local)
        return result + self.bias if self.bias is not None else result


__all__ = ["ColumnParallelLinear", "RowParallelLinear"]
