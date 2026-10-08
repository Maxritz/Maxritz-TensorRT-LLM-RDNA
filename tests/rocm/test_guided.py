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
"""Finite guided-constraint coverage: JSON-Schema expansion, regex, mutual exclusion."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from tensorrt_llm.rocm.guided import compile_partial_regex, json_schema_alternatives
from tensorrt_llm.rocm.llm import LLM, _ChoiceProcessor
from tensorrt_llm.rocm.sampling import SamplingParams

pytestmark = pytest.mark.cpu_only


def test_const_enum_boolean_null_expand_to_canonical_alternatives() -> None:
    assert json_schema_alternatives({"const": "tok4"}) == ['"tok4"']
    assert json_schema_alternatives({"enum": ["tok5", "tok4", "tok4"]}) == ['"tok4"', '"tok5"']
    assert json_schema_alternatives({"type": "boolean"}) == ["false", "true"]
    assert json_schema_alternatives({"type": "null"}) == ["null"]
    assert json_schema_alternatives({"type": ["boolean", "null"]}) == ["false", "null", "true"]


def test_const_values_canonicalize_with_sorted_keys_and_no_whitespace() -> None:
    value = [1, {"z": True, "a": None}]
    assert json_schema_alternatives({"const": value}) == ['[1,{"a":null,"z":true}]']


def test_fixed_length_arrays_expand_cartesian_products() -> None:
    schema = {"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 2}
    assert json_schema_alternatives(schema) == [
        "[false,false]",
        "[false,true]",
        "[true,false]",
        "[true,true]",
    ]
    empty = {"type": "array", "items": {"type": "boolean"}, "minItems": 0, "maxItems": 0}
    assert json_schema_alternatives(empty) == ["[]"]


def test_closed_objects_compose_required_properties_in_canonical_order() -> None:
    schema = {
        "type": "object",
        "properties": {"b": {"const": "tok4"}, "a": {"enum": ["tok5", "tok4"]}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    assert json_schema_alternatives(schema) == [
        '{"a":"tok4","b":"tok4"}',
        '{"a":"tok5","b":"tok4"}',
    ]
    constant = {
        "type": "object",
        "properties": {"v": {"const": "tok4"}},
        "required": ["v"],
    }
    assert json_schema_alternatives(constant) == ['{"v":"tok4"}']


def test_nested_arrays_and_objects_stay_finite() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"enum": ["tok4", "tok5"]},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    assert json_schema_alternatives(schema) == [
        '{"items":["tok4","tok4"]}',
        '{"items":["tok4","tok5"]}',
        '{"items":["tok5","tok4"]}',
        '{"items":["tok5","tok5"]}',
    ]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": "integer"},
        {"type": "number"},
        {"enum": []},
        {"enum": "tok4"},
        {},
        {"pattern": "tok.*"},
        {"type": "boolean", "pattern": "tok"},
        {"type": "array", "minItems": 2, "maxItems": 2},
        {"type": "array", "items": {"type": "boolean"}},
        {"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 2},
        {"type": "array", "items": {"type": "boolean"}, "minItems": -1, "maxItems": -1},
        {"type": "object", "properties": {}, "required": ["missing"]},
        {"type": "object", "properties": {}, "additionalProperties": True},
        {"type": "object", "properties": {"a": {"const": 1}}},
        {"type": "object", "properties": {"a": {"const": 1}}, "required": "a"},
        {"const": float("inf")},
        {"const": {1: "tok4"}},
    ],
)
def test_unbounded_or_unsupported_schema_constructs_are_rejected(schema) -> None:
    with pytest.raises(ValueError):
        json_schema_alternatives(schema)


def test_schema_nesting_beyond_the_finite_depth_limit_is_rejected() -> None:
    schema: dict = {"type": "boolean"}
    for _ in range(40):
        schema = {
            "type": "array",
            "items": schema,
            "minItems": 1,
            "maxItems": 1,
        }
    with pytest.raises(ValueError, match="depth"):
        json_schema_alternatives(schema)


def test_compile_partial_regex_rejects_invalid_patterns() -> None:
    compiled = compile_partial_regex("tok4( tok5)?")
    assert compiled.fullmatch("tok4")
    assert compiled.fullmatch("tok4 tok5")
    assert compiled.fullmatch("tok4 to", partial=True)
    assert not compiled.fullmatch("tok5", partial=True)
    with pytest.raises(ValueError, match="nonempty"):
        compile_partial_regex("")
    with pytest.raises(ValueError):
        compile_partial_regex("(unclosed")


def test_guided_constraints_are_mutually_exclusive() -> None:
    schema = {"enum": ["tok4", "tok5"]}
    with pytest.raises(ValueError, match="mutually exclusive"):
        SamplingParams(guided_choice=["tok4"], guided_regex="tok5")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SamplingParams(guided_choice=["tok4"], guided_json_schema=schema)
    with pytest.raises(ValueError, match="mutually exclusive"):
        SamplingParams(guided_json_schema=schema, guided_regex="tok4")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SamplingParams(guided_choice=["tok4"], guided_json_schema=schema, guided_regex="tok4")


def test_guided_parameters_compile_eagerly() -> None:
    with pytest.raises(ValueError, match="regular expression"):
        SamplingParams(guided_regex="(unclosed")
    with pytest.raises(ValueError):
        SamplingParams(guided_json_schema={"type": "string"})
    SamplingParams(guided_regex="tok4|tok5")
    SamplingParams(guided_json_schema={"enum": ["tok4", "tok5"]})


def test_guided_json_schema_reuses_the_choice_token_prefix_mask(engine) -> None:
    params = SamplingParams(
        temperature=0, max_tokens=4, guided_json_schema={"enum": ["tok4", "tok5"]}
    )
    processors = engine._guided_processors(params, prompt_width=1, eos_ids=[2])
    assert isinstance(processors[0], _ChoiceProcessor)
    assert processors[0].choices == [
        engine.tokenizer.encode('"tok4"', add_special_tokens=False),
        engine.tokenizer.encode('"tok5"', add_special_tokens=False),
    ]
    scores = torch.randn(1, engine.model.get_input_embeddings().num_embeddings)
    masked = processors[0](torch.tensor([[4]]), scores)
    assert masked[0].isfinite().sum().item() == 1


def test_engine_rejects_conflicting_guides(engine) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        engine.generate(
            "tok4",
            SamplingParams(temperature=0, guided_choice=["tok4"], guided_regex="tok5"),
        )


def test_guided_regex_generation_stays_within_the_pattern(engine) -> None:
    params = SamplingParams(temperature=0, max_tokens=4, guided_regex="tok4|tok5")
    result = engine.generate("tok6", params)[0].outputs[0]
    assert result.text in ("tok4", "tok5")
    assert result.finish_reason == "stop"
    assert len(result.token_ids) == 1


def test_guided_regex_partial_match_allows_only_completable_tokens(engine) -> None:
    params = SamplingParams(temperature=0, max_tokens=4, guided_regex="tok4( tok5)?")
    result = engine.generate("tok6", params)[0].outputs[0]
    assert result.text == "tok4"
    assert result.finish_reason == "stop"


def test_guided_regex_streams_one_exact_match(engine) -> None:
    params = SamplingParams(temperature=0, max_tokens=4, guided_regex="tok4|tok5")
    chunks = list(engine.generate_stream("tok6", params))
    assert chunks[-1].finish_reason == "stop"
    assert len(chunks[-1].token_ids) == 1
    assert chunks[-1].token_ids[0] in (4, 5)
    assert "".join(chunk.text for chunk in chunks[:-1]) == engine.tokenizer.decode(
        chunks[-1].token_ids, skip_special_tokens=True
    )


@pytest.fixture
def json_engine():
    """Tiny Llama whose vocabulary holds whole canonical JSON fragments."""
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        '"tok4"': 4,
        '"tok5"': 5,
        '{"v":"tok4"}': 6,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=128,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    config = LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model = LlamaForCausalLM(config).eval()
    with LLM(model, fast, device="cpu", dtype="float32") as instance:
        yield instance


def test_guided_json_schema_generates_canonical_json(json_engine) -> None:
    params = SamplingParams(
        temperature=0, max_tokens=4, guided_json_schema={"enum": ["tok4", "tok5"]}
    )
    result = json_engine.generate('"tok4"', params)[0].outputs[0]
    assert result.text in ('"tok4"', '"tok5"')
    assert result.finish_reason == "stop"


def test_guided_json_schema_closed_object_end_to_end(json_engine) -> None:
    schema = {
        "type": "object",
        "properties": {"v": {"enum": ["tok4"]}},
        "required": ["v"],
        "additionalProperties": False,
    }
    params = SamplingParams(temperature=0, max_tokens=4, guided_json_schema=schema)
    result = json_engine.generate('"tok4"', params)[0].outputs[0]
    assert result.text == '{"v":"tok4"}'
    assert result.finish_reason == "stop"
