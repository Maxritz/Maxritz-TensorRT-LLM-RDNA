# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-effect-free probes for optional ROCm feature layers.

A package being importable is not proof that its kernels support RDNA4. Callers
must additionally apply architecture, version and shape qualification gates.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class OptionalCapability:
    name: str
    module: str
    purpose: str
    installed: bool
    version: str | None
    qualification: str


def _probe(
    name: str,
    module: str,
    purpose: str,
    qualification: str,
    distribution: str | None = None,
) -> OptionalCapability:
    try:
        installed = importlib.util.find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        installed = False
    try:
        version = importlib.metadata.version(distribution or name) if installed else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    return OptionalCapability(name, module, purpose, installed, version, qualification)


def optional_capabilities() -> dict[str, dict[str, str | bool | None]]:
    """Report optional layers without importing their native extensions."""
    capabilities = (
        _probe(
            "aiter", "aiter", "experimental RDNA4 attention/operator candidate",
            "gfx1200/gfx1201 and exact dtype/shape tests required; "
            "Navi paged attention is disabled",
        ),
        _probe(
            "triton", "triton", "portable custom-kernel language",
            "requires an AMD Triton compile+numerical smoke test on the target GPU",
        ),
        _probe(
            "peft", "peft", "static Hugging Face LoRA adapter loading",
            "load/merge correctness test required for the installed Transformers version",
        ),
        _probe(
            "bitsandbytes", "bitsandbytes", "low-bit HF loading candidate",
            "requires a real gfx120x kernel and accuracy/VRAM qualification",
        ),
        _probe(
            "outlines", "outlines", "host-side constrained decoding candidate",
            "requires tokenizer-specific allowed-token correctness tests",
        ),
        _probe(
            "lm-eval", "lm_eval", "external OpenAI-compatible evaluation client",
            "run through the local API; not imported into the serving process",
        ),
        _probe(
            "rccl", "torch.distributed", "ROCm collective transport through PyTorch",
            "requires multi-GPU topology and collective correctness/performance tests",
            distribution="torch",
        ),
    )
    return {capability.name: asdict(capability) for capability in capabilities}


__all__ = ["OptionalCapability", "optional_capabilities"]
