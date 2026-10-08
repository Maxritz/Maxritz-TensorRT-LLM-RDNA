# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn.functional as functional

from tensorrt_llm.rocm.adapter_gemm import LoRAWeights, grouped_lora_linear


def test_grouped_lora_linear_matches_per_row_reference_cpu():
    inputs = torch.randn(4, 3)
    weight, bias = torch.randn(2, 3), torch.randn(2)
    adapters = {
        "a": LoRAWeights(torch.randn(2, 3), torch.randn(2, 2), 0.5),
        "b": LoRAWeights(torch.randn(1, 3), torch.randn(2, 1), 1.25),
    }
    identities = ["a", None, "b", "a"]
    actual = grouped_lora_linear(inputs, weight, bias, identities, adapters)
    expected = functional.linear(inputs, weight, bias)
    for row, identity in enumerate(identities):
        if identity is not None:
            lora = adapters[identity]
            expected[row] += functional.linear(
                functional.linear(inputs[row : row + 1], lora.a), lora.b
            )[0] * lora.scale
    torch.testing.assert_close(actual, expected)
