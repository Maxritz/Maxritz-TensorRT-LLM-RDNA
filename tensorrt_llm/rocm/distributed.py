# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit RCCL/PyTorch collective primitives for future RDNA4 parallel executors."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .runtime import resolve_device


class RCCLContext:
    """Own one initialized ROCm process group; never silently fall back to Gloo."""

    def __init__(self, device: str | torch.device = "cuda:0") -> None:
        self.device = resolve_device(device)
        self._owns_group = False

    def initialize(self, *, init_method: str = "env://") -> None:
        if dist.is_initialized():
            if dist.get_backend() != "nccl":
                raise RuntimeError("ROCm distributed execution requires an RCCL/NCCL process group")
            return
        dist.init_process_group(backend="nccl", init_method=init_method)
        self._owns_group = True

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            raise RuntimeError("Call RCCLContext.initialize before using collectives")
        if tensor.device != self.device:
            raise ValueError("Collective tensor must reside on this context's HIP device")
        dist.all_reduce(tensor)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if not dist.is_initialized():
            raise RuntimeError("Call RCCLContext.initialize before using collectives")
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output, tensor)
        return output

    def close(self) -> None:
        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_group = False


__all__ = ["RCCLContext"]
