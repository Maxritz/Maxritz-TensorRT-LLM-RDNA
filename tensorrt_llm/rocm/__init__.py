# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROCm/RDNA4 inference API; heavyweight model imports remain lazy."""

import importlib

from .sampling import CompletionOutput, RequestOutput, SamplingParams, StreamOutput

__all__ = [
    "LLM",
    "AsyncLLM",
    "EmbeddingLLM",
    "RCCLContext",
    "PagedKVCache",
    "PagedDynamicCache",
    "paged_attention",
    "ContinuousBatchScheduler",
    "DecodeRequest",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "LoRAWeights",
    "grouped_lora_linear",
    "SamplingParams",
    "CompletionOutput",
    "RequestOutput",
    "StreamOutput",
]


def __getattr__(name: str):
    if name == "LLM":
        value = importlib.import_module(".llm", __name__).LLM
    elif name == "AsyncLLM":
        value = importlib.import_module(".async_llm", __name__).AsyncLLM
    elif name == "EmbeddingLLM":
        value = importlib.import_module(".embeddings", __name__).EmbeddingLLM
    elif name == "RCCLContext":
        value = importlib.import_module(".distributed", __name__).RCCLContext
    elif name == "PagedKVCache":
        value = importlib.import_module(".paged_kv", __name__).PagedKVCache
    elif name == "PagedDynamicCache":
        value = importlib.import_module(".hf_paged_cache", __name__).PagedDynamicCache
    elif name == "paged_attention":
        value = importlib.import_module(".paged_attention", __name__).paged_attention
    elif name in ("ContinuousBatchScheduler", "DecodeRequest"):
        value = getattr(importlib.import_module(".continuous", __name__), name)
    elif name in ("ColumnParallelLinear", "RowParallelLinear"):
        value = getattr(importlib.import_module(".tensor_parallel", __name__), name)
    elif name in ("LoRAWeights", "grouped_lora_linear"):
        value = getattr(importlib.import_module(".adapter_gemm", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
