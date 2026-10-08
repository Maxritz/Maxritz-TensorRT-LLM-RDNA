# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adapter-grouped LoRA GEMM reference path for ROCm execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class LoRAWeights:
    """One inference LoRA update: `scale * (x @ A.T) @ B.T`."""

    a: torch.Tensor
    b: torch.Tensor
    scale: float = 1.0

    def validate(self, input_features: int, output_features: int, reference: torch.Tensor) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2 or self.a.shape[1] != input_features:
            raise ValueError("LoRA A must have shape [rank, input_features]")
        if self.b.shape != (output_features, self.a.shape[0]):
            raise ValueError("LoRA B must have shape [output_features, rank]")
        if any(weight.device != reference.device for weight in (self.a, self.b)):
            raise ValueError("LoRA weights must be on the input device")


def grouped_lora_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    adapter_ids: list[str | None],
    adapters: dict[str, LoRAWeights],
) -> torch.Tensor:
    """Apply base GEMM once and batched LoRA updates grouped by adapter identity.

    Rows selecting the same adapter are processed together, avoiding serialized
    `set_adapter()` model mutation. This portable reference uses ROCm PyTorch
    GEMMs; it is the correctness contract for a future fused grouped GEMM kernel.
    """
    if input.ndim != 2 or weight.ndim != 2 or input.shape[1] != weight.shape[1]:
        raise ValueError(
            "Expected input [rows, input_features] and weight [output_features, input_features]"
        )
    if len(adapter_ids) != input.shape[0]:
        raise ValueError("adapter_ids must have one entry per input row")
    output = functional.linear(input, weight, bias)
    groups: dict[str, list[int]] = {}
    for index, adapter in enumerate(adapter_ids):
        if adapter is not None:
            groups.setdefault(adapter, []).append(index)
    for name, rows in groups.items():
        if name not in adapters:
            raise ValueError(f"No weights registered for adapter {name!r}")
        lora = adapters[name]
        lora.validate(input.shape[1], weight.shape[0], input)
        indices = torch.tensor(rows, dtype=torch.long, device=input.device)
        selected = input.index_select(0, indices)
        update = functional.linear(functional.linear(selected, lora.a), lora.b) * lora.scale
        output.index_add_(0, indices, update.to(output.dtype))
    return output


__all__ = ["LoRAWeights", "grouped_lora_linear"]
