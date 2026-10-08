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
"""Finite guided-decoding constraint compilation for the portable backend.

JSON-Schema documents from the deliberately finite subset compile to canonical
JSON alternative strings, and regular expressions compile to partial-match
patterns. Both feed the exact token-prefix masking path in
:mod:`tensorrt_llm.rocm.llm` without a GPU grammar runtime. This module is
intentionally dependency-free so expansion is testable on any host.
"""

from __future__ import annotations

import json
import math
from itertools import product
from typing import Any

# Schema keywords that never change the finite alternative set.
_ANNOTATION_KEYWORDS = frozenset(
    {"$schema", "$id", "title", "description", "default", "examples", "deprecated"}
)
_STRUCTURAL_KEYWORDS = frozenset(
    {
        "const",
        "enum",
        "type",
        "items",
        "minItems",
        "maxItems",
        "properties",
        "required",
        "additionalProperties",
    }
)
# Types with finite alternative sets; string/number/integer are unbounded.
_SUPPORTED_TYPES = ("boolean", "null", "array", "object")
# Bound recursive expansion so hostile schemas fail loudly instead of hanging.
_MAX_DEPTH = 32

__all__ = ["json_schema_alternatives", "compile_partial_regex"]


def json_schema_alternatives(schema: dict[str, Any]) -> list[str]:
    """Compile a finite JSON-Schema subset to canonical JSON alternative texts.

    Supported constructs: ``const``, ``enum``, the ``boolean`` and ``null``
    types, fixed-length arrays (equal ``minItems``/``maxItems``), and closed
    objects whose required properties are drawn from ``properties``. Anything
    that could need infinite or ambiguous expansion raises ``ValueError``.

    Returns the sorted, deduplicated set of canonical encodings: compact
    separators, sorted object keys, no insignificant whitespace. Generation
    then constrains itself to these exact strings through the same
    token-prefix mask used for ``guided_choice``.

    Canonical encodings are composed bottom-up from child encodings instead of
    re-encoding every full alternative, keeping compilation linear in the
    schema size plus the alternative count rather than paying one full
    ``json.dumps`` traversal per alternative.
    """
    texts = _schema_texts(schema, depth=0)
    return sorted(set(texts))


def compile_partial_regex(pattern: str):
    """Compile ``pattern`` for host-side partial matching.

    Uses the ``regex`` package (a Transformers dependency) so decoding can ask
    whether a prefix can still grow into a full match. Invalid or empty
    patterns raise ``ValueError``; a missing dependency raises ``RuntimeError``.
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("guided_regex must be a nonempty pattern string")
    try:
        import regex
    except ImportError as error:  # pragma: no cover - transformers always ships it
        raise RuntimeError(
            "guided_regex requires the optional 'regex' package for partial matching"
        ) from error
    try:
        return regex.compile(pattern)
    except regex.error as error:
        raise ValueError(f"guided_regex is not a valid regular expression: {error}") from error


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _schema_texts(schema: Any, depth: int) -> list[str]:
    if depth > _MAX_DEPTH:
        raise ValueError("guided_json_schema nesting exceeds the finite expansion depth limit")
    if not isinstance(schema, dict):
        raise ValueError("A guided JSON schema must be a JSON object")
    unsupported = set(schema) - _ANNOTATION_KEYWORDS - _STRUCTURAL_KEYWORDS
    if unsupported:
        raise ValueError(
            "Unsupported JSON-Schema keywords for guided_json_schema: "
            + ", ".join(sorted(unsupported))
        )
    if "const" in schema:
        _check_serializable(schema["const"], "const")
        return [_canonical(schema["const"])]
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError("guided_json_schema enum must be a nonempty list")
        for value in values:
            _check_serializable(value, "enum")
        return [_canonical(value) for value in values]
    if "type" not in schema:
        raise ValueError("guided_json_schema must declare const, enum, or type")
    declared = schema["type"]
    kinds = [declared] if isinstance(declared, str) else declared
    if (
        not isinstance(kinds, list)
        or not kinds
        or not all(kind in _SUPPORTED_TYPES for kind in kinds)
    ):
        raise ValueError(
            "guided_json_schema type must be one of "
            f"{', '.join(_SUPPORTED_TYPES)}; string, number and integer are unbounded"
        )
    texts: list[str] = []
    for kind in kinds:
        if kind == "boolean":
            texts.extend(("false", "true"))
        elif kind == "null":
            texts.append("null")
        elif kind == "array":
            texts.extend(_array_texts(schema, depth))
        else:
            texts.extend(_object_texts(schema, depth))
    return texts


def _array_texts(schema: dict[str, Any], depth: int) -> list[str]:
    items = schema.get("items")
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if not isinstance(items, dict):
        raise ValueError("A guided JSON array must declare an object items schema")
    lengths_are_fixed = (
        isinstance(minimum, int)
        and not isinstance(minimum, bool)
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and minimum == maximum
        and minimum >= 0
    )
    if not lengths_are_fixed:
        raise ValueError(
            "A guided JSON array must declare equal nonnegative integer "
            "minItems and maxItems; variable length is not finite"
        )
    item_texts = _schema_texts(items, depth + 1)
    return [
        "[" + ",".join(combination) + "]" for combination in product(item_texts, repeat=minimum)
    ]


def _object_texts(schema: dict[str, Any], depth: int) -> list[str]:
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not all(isinstance(key, str) for key in properties):
        raise ValueError("A guided JSON object must declare a properties object")
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise ValueError("A guided JSON object must declare a required list of property names")
    if "additionalProperties" in schema and schema["additionalProperties"] is not False:
        raise ValueError("A guided JSON object must be closed: additionalProperties must be false")
    missing = [key for key in required if key not in properties]
    if missing:
        raise ValueError(
            "A guided JSON object requires undeclared properties: " + ", ".join(sorted(missing))
        )
    keys = sorted(set(required))
    per_key = [_schema_texts(properties[key], depth + 1) for key in keys]
    return [
        "{" + ",".join(f"{_canonical(key)}:{text}" for key, text in zip(keys, combination)) + "}"
        for combination in product(*per_key)
    ]


def _check_serializable(value: Any, context: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"guided_json_schema {context} numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _check_serializable(item, context)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"guided_json_schema {context} objects must have string keys")
            _check_serializable(item, context)
        return
    raise ValueError(
        f"guided_json_schema {context} values must be JSON-serializable; got {type(value).__name__}"
    )
