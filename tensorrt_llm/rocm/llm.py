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
"""Single-device ROCm LLM execution, independent of TensorRT and native CUDA bindings."""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Literal

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from trtllm_profile import active_session, component, trace_active

from .runtime import KernelBackend, resolve_device, resolve_dtype
from .sampling import CompletionOutput, RequestOutput, SamplingParams, StreamOutput


class _StopStrings(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        strings: list[str],
        prompt_width: int,
        minimum: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.strings = strings
        self.prompt_width = prompt_width
        self.minimum = minimum

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor | None, **kwargs
    ) -> torch.Tensor:
        if input_ids.shape[1] - self.prompt_width < self.minimum:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        texts = self.tokenizer.batch_decode(
            input_ids[:, self.prompt_width :], skip_special_tokens=True
        )
        return torch.tensor(
            [any(stop in text for stop in self.strings) for text in texts],
            dtype=torch.bool,
            device=input_ids.device,
        )


class _CancellationCriteria(StoppingCriteria):
    """Stop generation at the next decoding step when a request is cancelled."""

    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor | None, **kwargs
    ) -> torch.Tensor:
        return torch.full(
            (input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device
        )


class _ChoiceProcessor(LogitsProcessor):
    """Portable exact-choice constrained decoding without a CUDA grammar runtime."""

    def __init__(self, choices: list[list[int]], prompt_width: int, eos_ids: list[int]) -> None:
        self.choices = choices
        self.prompt_width = prompt_width
        self.eos_ids = eos_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        constrained = torch.full_like(scores, -float("inf"))
        for row, sequence in enumerate(input_ids.tolist()):
            generated = sequence[self.prompt_width :]
            allowed = set()
            for choice in self.choices:
                if choice[: len(generated)] != generated:
                    continue
                if len(generated) == len(choice):
                    allowed.update(self.eos_ids)
                else:
                    allowed.add(choice[len(generated)])
            if not allowed:
                raise ValueError("guided_choice has no valid token continuation")
            constrained[row, list(allowed)] = scores[row, list(allowed)]
        return constrained


class LLM:
    """Generate text with ROCm PyTorch/Hugging Face on one genuine RDNA4 GPU.

    Unquantized causal language models supported by the installed Transformers
    version can use SDPA or eager attention. ``device='cpu'`` is explicitly for
    reference testing, never an automatic fallback. TensorRT plans, NVIDIA
    quantization plugins, distributed scheduling, and CUDA-specific options are
    rejected instead of being ignored or silently emulated.
    """

    def __init__(
        self,
        model: str | Path | PreTrainedModel,
        tokenizer: str | Path | PreTrainedTokenizerBase | None = None,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "auto",
        quantization: Literal["none", "bitsandbytes-4bit", "bitsandbytes-8bit"] = "none",
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        attn_backend: Literal["sdpa", "eager", "hip"] = "sdpa",
        kernels: KernelBackend = "torch",
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        draft_model: str | Path | PreTrainedModel | None = None,
        draft_tokenizer: str | Path | PreTrainedTokenizerBase | None = None,
        num_draft_tokens: int = 4,
        paged_kv_cache: bool = False,
        kv_cache_pages: int = 1024,
        kv_cache_page_size: int = 16,
        kv_cache_quantization: Literal["none", "int8"] = "none",
        lora_adapter: str | Path | None = None,
        merge_lora: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        **unsupported,
    ) -> None:
        if unsupported:
            raise NotImplementedError(f"Unsupported ROCm options: {', '.join(sorted(unsupported))}")
        if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
            raise NotImplementedError(
                "The ROCm backend currently supports single-device execution only"
            )
        if max_batch_size < 1 or (max_seq_len is not None and max_seq_len < 1):
            raise ValueError("max_batch_size and max_seq_len must be positive")
        if num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be positive")
        if kv_cache_pages < 1 or kv_cache_page_size < 1:
            raise ValueError("kv_cache_pages and kv_cache_page_size must be positive")
        if kv_cache_quantization not in ("none", "int8"):
            raise ValueError("kv_cache_quantization must be none or int8")
        if quantization not in ("none", "bitsandbytes-4bit", "bitsandbytes-8bit"):
            raise ValueError("Unsupported quantization mode")
        if attn_backend not in ("sdpa", "eager", "hip") or kernels not in ("torch", "hip"):
            raise ValueError("Use attn_backend='sdpa'/'eager'/'hip' and kernels='torch'/'hip'")
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        if quantization != "none" and self.device.type != "cuda":
            raise ValueError("bitsandbytes quantization requires a real ROCm device")
        self.quantization = quantization
        if kernels == "hip" and self.device.type != "cuda":
            raise ValueError(
                "kernels='hip' requires a real RDNA4 GPU; use kernels='torch' for CPU validation"
            )
        if attn_backend == "hip" and kernels != "hip":
            raise ValueError("attn_backend='hip' requires kernels='hip'")
        self._max_batch_size = max_batch_size
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._closed = False
        self.last_stats: dict[str, float | int] = {}
        self.native_norm_count = 0
        self._lora_adapters: set[str] = set()
        self._active_lora_adapter: str | None = None
        self.model_id = (
            str(model)
            if isinstance(model, (str, Path))
            else model.config.name_or_path or type(model).__name__
        )
        loading = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        quantization_options: dict = {}
        if quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as error:
                raise RuntimeError(
                    "bitsandbytes quantization requires Transformers BitsAndBytesConfig "
                    "and bitsandbytes"
                ) from error
            quantization_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "bitsandbytes-4bit",
                load_in_8bit=quantization == "bitsandbytes-8bit",
            )
            quantization_options["device_map"] = {"": str(self.device)}
        # Loading must create ordinary versioned parameters, even when called
        # from an outer inference-mode scope (important for dispatch profiling).
        with trace_active(), torch.inference_mode(False):
            if isinstance(model, PreTrainedModel):
                loaded = model
                config = model.config
                if attn_backend == "hip":
                    from .attention import configure_native_attention

                    config._attn_implementation = configure_native_attention(config)
            else:
                path = Path(model)
                if path.suffix in (".engine", ".plan") or (
                    path.is_dir() and not (path / "config.json").is_file()
                ):
                    raise ValueError(
                        "Load a Hugging Face checkpoint, not a TensorRT engine directory/plan"
                    )
                config = AutoConfig.from_pretrained(str(model), **loading)
                self._check_quantization(config)
                implementation = attn_backend
                if attn_backend == "hip":
                    from .attention import configure_native_attention

                    implementation = configure_native_attention(config)
                loaded = AutoModelForCausalLM.from_pretrained(
                    str(model),
                    config=config,
                    torch_dtype=self.dtype,
                    attn_implementation=implementation,
                    **quantization_options,
                    **loading,
                )
            self._check_quantization(config)
            if attn_backend == "hip" and not getattr(loaded, "_supports_attention_backend", False):
                raise NotImplementedError(
                    "This Transformers model does not support the native attention interface"
                )
            self.model = loaded.eval()
            if quantization == "none":
                self.model = self.model.to(device=self.device, dtype=self.dtype)
            if lora_adapter is not None and quantization != "none":
                raise NotImplementedError(
                    "PEFT adapters with bitsandbytes weights are not qualified"
                )
            if lora_adapter is not None:
                try:
                    from peft import PeftModel
                except ImportError as error:
                    raise RuntimeError(
                        "LoRA adapters require the optional 'peft' package; install PEFT "
                        "before passing lora_adapter"
                    ) from error
                adapter = str(lora_adapter)
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter,
                    adapter_name="default",
                    is_trainable=False,
                    local_files_only=local_files_only,
                ).eval()
                self._lora_adapters.add("default")
                self._active_lora_adapter = "default"
                if merge_lora:
                    self.model = self.model.merge_and_unload().eval()
                    self._lora_adapters.clear()
                    self._active_lora_adapter = None
                self.model = self.model.to(device=self.device, dtype=self.dtype)
            elif merge_lora:
                raise ValueError("merge_lora requires lora_adapter")
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = tokenizer
            else:
                source = str(tokenizer) if tokenizer is not None else self.model_id
                self.tokenizer = AutoTokenizer.from_pretrained(source, **loading)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer must define a pad token or an EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.draft_model: PreTrainedModel | None = None
            self._num_draft_tokens = num_draft_tokens
            if draft_model is not None:
                if isinstance(draft_model, PreTrainedModel):
                    candidate = draft_model
                else:
                    draft_config = AutoConfig.from_pretrained(str(draft_model), **loading)
                    self._check_quantization(draft_config)
                    candidate = AutoModelForCausalLM.from_pretrained(
                        str(draft_model), config=draft_config, torch_dtype=self.dtype, **loading
                    )
                self.draft_model = candidate.eval().to(device=self.device, dtype=self.dtype)
                if isinstance(draft_tokenizer, PreTrainedTokenizerBase):
                    candidate_tokenizer = draft_tokenizer
                elif isinstance(draft_model, PreTrainedModel) and draft_tokenizer is None:
                    candidate_tokenizer = self.tokenizer
                else:
                    source = (
                        str(draft_tokenizer) if draft_tokenizer is not None else str(draft_model)
                    )
                    candidate_tokenizer = AutoTokenizer.from_pretrained(source, **loading)
                if candidate_tokenizer.get_vocab() != self.tokenizer.get_vocab():
                    raise ValueError("draft_model and target model must use identical tokenizers")
            capacities = [
                value
                for value in (
                    getattr(config, "max_position_embeddings", None),
                    getattr(config, "n_positions", None),
                    self.tokenizer.model_max_length,
                )
                if isinstance(value, int) and 0 < value < 100000000
            ]
            capacity = min(capacities) if capacities else None
            if max_seq_len is not None and capacity is not None and max_seq_len > capacity:
                raise ValueError(
                    f"max_seq_len={max_seq_len} exceeds the model/tokenizer capacity {capacity}"
                )
            self._max_seq_len = max_seq_len or capacity
            self._paged_kv_config: dict | None = None
            if paged_kv_cache:
                kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
                head_dim = getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
                self._paged_kv_config = {
                    "num_hidden_layers": config.num_hidden_layers,
                    "num_pages": kv_cache_pages,
                    "page_size": kv_cache_page_size,
                    "num_key_value_heads": kv_heads,
                    "head_dim": head_dim,
                    "dtype": self.dtype,
                    "device": self.device,
                    "quantization": kv_cache_quantization,
                }
            if kernels == "hip":
                from .kernels import load_kernels
                from .ops import apply_native_norms

                load_kernels(self.device)
                self.native_norm_count = apply_native_norms(self.model)

    @staticmethod
    def _check_quantization(config) -> None:
        if getattr(config, "quantization_config", None):
            raise NotImplementedError(
                "Pre-quantized checkpoints are not enabled by the portable backend. "
                "Use FP32/FP16/BF16 weights or explicit bitsandbytes loading."
            )

    def load_lora_adapter(
        self, name: str, adapter: str | Path, *, local_files_only: bool = False
    ) -> None:
        """Load a PEFT adapter for serialized per-request selection."""
        if not name or name in self._lora_adapters:
            raise ValueError("Adapter name must be new and nonempty")
        with self._lock:
            if not self._lora_adapters or not hasattr(self.model, "load_adapter"):
                raise NotImplementedError("Load an initial lora_adapter before adding adapters")
            self.model.load_adapter(
                str(adapter),
                adapter_name=name,
                is_trainable=False,
                local_files_only=local_files_only,
            )
            self._lora_adapters.add(name)

    def set_lora_adapter(self, name: str | None) -> None:
        """Select a loaded adapter; callers must not switch it during an active request."""
        with self._lock:
            if name is None:
                if self._lora_adapters:
                    self.model.disable_adapter_layers()
                self._active_lora_adapter = None
                return
            if name not in self._lora_adapters:
                raise ValueError(f"Unknown LoRA adapter {name!r}")
            self.model.set_adapter(name)
            self._active_lora_adapter = name

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def _encode(
        self, prompts: list[str] | list[list[int]]
    ) -> tuple[dict[str, torch.Tensor], list[list[int]], list[str]]:
        if isinstance(prompts[0], str):
            if not all(isinstance(prompt, str) for prompt in prompts):
                raise ValueError("All prompts must have the same type")
            encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
            tokens = [
                ids[mask.bool()].tolist()
                for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
            ]
            texts = list(prompts)
            tensors = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        else:
            if not all(
                isinstance(prompt, list)
                and prompt
                and all(isinstance(token, int) and not isinstance(token, bool) for token in prompt)
                for prompt in prompts
            ):
                raise ValueError("Token prompts must be nonempty lists of integer token IDs")
            tokens = [list(prompt) for prompt in prompts]
            width = max(map(len, tokens))
            tensors = {
                "input_ids": torch.tensor(
                    [[self.tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in tokens]
                ),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(ids)) + [1] * len(ids) for ids in tokens]
                ),
            }
            texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=False)
        if any(not ids for ids in tokens):
            raise ValueError("Prompts must contain at least one token")
        vocabulary_size = self.model.get_input_embeddings().num_embeddings
        if any(token < 0 or token >= vocabulary_size for ids in tokens for token in ids):
            raise ValueError("Prompt token ID is outside the model vocabulary")
        with component("transfer"):
            tensors = {name: tensor.to(self.device) for name, tensor in tensors.items()}
        return tensors, tokens, texts

    def _new_paged_cache(self):
        if self._paged_kv_config is None:
            return None
        from .hf_paged_cache import PagedDynamicCache

        return PagedDynamicCache(**self._paged_kv_config)

    def _speculative_options(self, params: SamplingParams) -> dict:
        if self.draft_model is None:
            return {}
        if params.n != 1 or params.beam_width != 1 or params.guided_choice is not None:
            raise NotImplementedError(
                "draft-model decoding requires n=1, beam_width=1, and no guided_choice"
            )
        return {
            "assistant_model": self.draft_model,
            "num_assistant_tokens": self._num_draft_tokens,
        }

    def _stopping_criteria(
        self,
        strings: list[str],
        prompt_width: int,
        minimum: int,
        cancellation_event: threading.Event | None,
    ) -> StoppingCriteriaList:
        criteria: list[StoppingCriteria] = []
        if strings:
            criteria.append(_StopStrings(self.tokenizer, strings, prompt_width, minimum))
        if cancellation_event is not None:
            criteria.append(_CancellationCriteria(cancellation_event))
        return StoppingCriteriaList(criteria)

    def _guided_processors(
        self, params: SamplingParams, prompt_width: int, eos_ids: list[int]
    ) -> list[LogitsProcessor] | None:
        if params.guided_choice is None:
            return None
        choices = [
            self.tokenizer.encode(choice, add_special_tokens=False)
            for choice in params.guided_choice
        ]
        if any(not choice for choice in choices):
            raise ValueError("A guided_choice cannot encode to an empty token sequence")
        if not eos_ids:
            raise ValueError("guided_choice requires an EOS token")
        if any(len(choice) > params.max_tokens for choice in choices):
            raise ValueError("guided_choice length exceeds max_tokens")
        return [_ChoiceProcessor(choices, prompt_width, eos_ids)]

    def generate(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        *,
        streaming: bool = False,
        cancellation_event: threading.Event | None = None,
        adapter_name: str | None = None,
    ) -> list[RequestOutput] | Iterator[StreamOutput]:
        """Generate a batch, preserving prompt order and excluding EOS from token IDs.

        Stop strings are excluded from text, but their complete boundary tokens
        remain in token_ids (a stop string may end inside a subword token). Use
        :meth:`generate_stream` when one completion must be delivered incrementally.
        """
        if streaming:
            return self.generate_stream(  # type: ignore[arg-type]
                prompts,
                sampling_params,
                cancellation_event=cancellation_event,
                adapter_name=adapter_name,
            )
        params = sampling_params or SamplingParams()
        if not isinstance(params, SamplingParams):
            raise TypeError("Use tensorrt_llm.rocm.SamplingParams with the ROCm backend")
        if isinstance(prompts, str):
            batches = [prompts]
        elif isinstance(prompts, list) and prompts and isinstance(prompts[0], int):
            batches = [prompts]
        elif isinstance(prompts, list):
            batches = prompts
        else:
            raise TypeError("prompts must be text or a list of text/token prompts")
        with self._lock, trace_active(), torch.inference_mode():
            if self._closed:
                raise RuntimeError("LLM has been shut down")
            if adapter_name is not None:
                self.set_lora_adapter(adapter_name)
            results: list[RequestOutput] = []
            started = time.perf_counter()
            model_hooks = nullcontext()
            if active_session() is not None:
                from trtllm_profile.torch_trace import instrument_model

                model_hooks = instrument_model(self.model)
            with model_hooks:
                for offset in range(0, len(batches), self._max_batch_size):
                    batch = batches[offset : offset + self._max_batch_size]
                    encoded, prompt_ids, texts = self._encode(batch)
                    width = encoded["input_ids"].shape[1]
                    if (
                        self._max_seq_len is not None
                        and width + params.max_tokens > self._max_seq_len
                    ):
                        raise ValueError(
                            f"Prompt plus max_tokens exceeds max_seq_len={self._max_seq_len}"
                        )
                    strings = [params.stop] if isinstance(params.stop, str) else (params.stop or [])
                    stopping = self._stopping_criteria(
                        strings, width, params.min_tokens, cancellation_event
                    )
                    do_sample = params.temperature > 0
                    end_id = (
                        params.end_id
                        if params.end_id is not None
                        else self.model.generation_config.eos_token_id
                    )
                    if end_id is None:
                        end_id = self.tokenizer.eos_token_id
                    pad_id = (
                        params.pad_id if params.pad_id is not None else self.tokenizer.pad_token_id
                    )
                    vocabulary_size = self.model.get_input_embeddings().num_embeddings
                    eos_ids = (
                        end_id
                        if isinstance(end_id, list)
                        else ([end_id] if end_id is not None else [])
                    )
                    if any(token >= vocabulary_size for token in [pad_id, *eos_ids]):
                        raise ValueError("EOS/pad token ID is outside the model vocabulary")
                    processors = self._guided_processors(params, width, eos_ids)
                    options = {
                        "max_new_tokens": params.max_tokens,
                        "min_new_tokens": params.min_tokens,
                        "do_sample": do_sample,
                        "num_beams": params.beam_width,
                        "num_return_sequences": params.n,
                        "repetition_penalty": params.repetition_penalty,
                        "pad_token_id": pad_id,
                        "eos_token_id": None if params.ignore_eos else end_id,
                        "stopping_criteria": stopping,
                        "use_cache": True,
                        "return_dict_in_generate": False,
                    }
                    options.update(self._speculative_options(params))
                    paged_cache = self._new_paged_cache()
                    if paged_cache is not None:
                        options["past_key_values"] = paged_cache
                    if processors:
                        options["logits_processor"] = processors
                    if do_sample:
                        options.update(
                            temperature=params.temperature, top_p=params.top_p, top_k=params.top_k
                        )
                    rng = (
                        torch.random.fork_rng(
                            devices=[self.device.index] if self.device.type == "cuda" else []
                        )
                        if params.seed is not None
                        else nullcontext()
                    )
                    with rng:
                        if params.seed is not None:
                            torch.random.default_generator.manual_seed(params.seed)
                            if self.device.type == "cuda":
                                with torch.cuda.device(self.device):
                                    torch.cuda.manual_seed(params.seed)
                        generated = self.model.generate(**encoded, **options)
                    with component("transfer"):
                        generated_ids = generated[:, width:].cpu().tolist()
                    for index, (text, ids) in enumerate(zip(texts, prompt_ids)):
                        outputs = []
                        for completion in range(params.n):
                            tokens = generated_ids[index * params.n + completion]
                            reason = "length"
                            if not params.ignore_eos:
                                endings = [
                                    position
                                    for position, token in enumerate(tokens)
                                    if token in eos_ids
                                ]
                                if endings:
                                    tokens = tokens[: endings[0]]
                                    reason = "stop"
                            if cancellation_event is not None and cancellation_event.is_set():
                                reason = "cancelled"
                            output_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                            stop_positions = [
                                output_text.find(stop) for stop in strings if stop in output_text
                            ]
                            if stop_positions:
                                output_text = output_text[: min(stop_positions)]
                                reason = "stop"
                                while tokens and tokens[-1] == pad_id:
                                    tokens = tokens[:-1]
                            outputs.append(
                                CompletionOutput(completion, output_text, tokens, reason)
                            )
                        results.append(RequestOutput(next(self._ids), text, ids, outputs))
            elapsed = time.perf_counter() - started
            input_tokens = sum(len(result.prompt_token_ids) for result in results)
            output_tokens = sum(
                len(output.token_ids) for result in results for output in result.outputs
            )
            self.last_stats = {
                "wall_s": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_s": output_tokens / elapsed if elapsed else 0.0,
            }
            return results

    def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        *,
        cancellation_event: threading.Event | None = None,
        adapter_name: str | None = None,
    ) -> Iterator[StreamOutput]:
        """Generate one completion as lossless text deltas.

        This is the portable streaming counterpart to :meth:`generate`. It uses
        Transformers' streamer rather than a CUDA-only executor, so it works on
        CPU reference mode and HIP. Streaming one sequence is intentional:
        Transformers' text streamer does not preserve per-request ordering for
        a batch, beam search, or multiple returned sequences. Those modes remain
        available through the non-streaming ``generate`` API.

        Stop strings are withheld until they are known not to be the start of a
        configured stop sequence. The terminal item has an empty ``text`` field,
        a finish reason, and exact generated token IDs.
        """
        params = sampling_params or SamplingParams()
        if not isinstance(params, SamplingParams):
            raise TypeError("Use tensorrt_llm.rocm.SamplingParams with the ROCm backend")
        if params.n != 1 or params.beam_width != 1:
            raise NotImplementedError(
                "Streaming supports one completion with beam_width=1; use non-streaming generate"
            )
        if self.draft_model is not None:
            raise NotImplementedError("Streaming with a draft model is not qualified yet")
        if not isinstance(prompt, str) and not (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(token, int) and not isinstance(token, bool) for token in prompt)
        ):
            raise TypeError("Streaming prompt must be text or one nonempty list of token IDs")

        # Tokenizer decoding is stateful at word boundaries; TextIteratorStreamer
        # preserves that state and supplies deltas suitable for SSE output.
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        request_id = next(self._ids)
        state: dict[str, object] = {}

        def _run() -> None:
            try:
                with self._lock, trace_active(), torch.inference_mode():
                    if self._closed:
                        raise RuntimeError("LLM has been shut down")
                    if adapter_name is not None:
                        self.set_lora_adapter(adapter_name)
                    started = time.perf_counter()
                    encoded, prompt_ids, _ = self._encode([prompt])
                    width = encoded["input_ids"].shape[1]
                    if (
                        self._max_seq_len is not None
                        and width + params.max_tokens > self._max_seq_len
                    ):
                        raise ValueError(
                            f"Prompt plus max_tokens exceeds max_seq_len={self._max_seq_len}"
                        )
                    strings = (
                        [params.stop] if isinstance(params.stop, str) else (params.stop or [])
                    )
                    stopping = self._stopping_criteria(
                        strings, width, params.min_tokens, cancellation_event
                    )
                    end_id = (
                        params.end_id
                        if params.end_id is not None
                        else self.model.generation_config.eos_token_id
                    )
                    if end_id is None:
                        end_id = self.tokenizer.eos_token_id
                    pad_id = (
                        params.pad_id
                        if params.pad_id is not None
                        else self.tokenizer.pad_token_id
                    )
                    vocabulary_size = self.model.get_input_embeddings().num_embeddings
                    eos_ids = (
                        end_id
                        if isinstance(end_id, list)
                        else ([end_id] if end_id is not None else [])
                    )
                    if any(token >= vocabulary_size for token in [pad_id, *eos_ids]):
                        raise ValueError("EOS/pad token ID is outside the model vocabulary")
                    processors = self._guided_processors(params, width, eos_ids)
                    options = {
                        "max_new_tokens": params.max_tokens,
                        "min_new_tokens": params.min_tokens,
                        "do_sample": params.temperature > 0,
                        "repetition_penalty": params.repetition_penalty,
                        "pad_token_id": pad_id,
                        "eos_token_id": None if params.ignore_eos else end_id,
                        "stopping_criteria": stopping,
                        "streamer": streamer,
                        "use_cache": True,
                        "return_dict_in_generate": False,
                    }
                    paged_cache = self._new_paged_cache()
                    if paged_cache is not None:
                        options["past_key_values"] = paged_cache
                    if processors:
                        options["logits_processor"] = processors
                    if params.temperature > 0:
                        options.update(
                            temperature=params.temperature,
                            top_p=params.top_p,
                            top_k=params.top_k,
                        )
                    rng = (
                        torch.random.fork_rng(
                            devices=[self.device.index] if self.device.type == "cuda" else []
                        )
                        if params.seed is not None
                        else nullcontext()
                    )
                    with rng:
                        if params.seed is not None:
                            torch.random.default_generator.manual_seed(params.seed)
                            if self.device.type == "cuda":
                                with torch.cuda.device(self.device):
                                    torch.cuda.manual_seed(params.seed)
                        generated = self.model.generate(**encoded, **options)
                    tokens = generated[0, width:].cpu().tolist()
                    finish_reason = "length"
                    if not params.ignore_eos:
                        endings = [
                            position for position, token in enumerate(tokens) if token in eos_ids
                        ]
                        if endings:
                            tokens = tokens[: endings[0]]
                            finish_reason = "stop"
                    if cancellation_event is not None and cancellation_event.is_set():
                        finish_reason = "cancelled"
                    elapsed = time.perf_counter() - started
                    self.last_stats = {
                        "wall_s": elapsed,
                        "input_tokens": len(prompt_ids[0]),
                        "output_tokens": len(tokens),
                        "tokens_per_s": len(tokens) / elapsed if elapsed else 0.0,
                    }
                    state.update(tokens=tokens, finish_reason=finish_reason)
            except BaseException as error:
                # A generator exception would otherwise leave the consumer
                # blocked forever waiting for TextIteratorStreamer's sentinel.
                state["error"] = error
                streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(target=_run, name="trtllm-rocm-stream", daemon=True)
        thread.start()
        pending = ""
        stopped_by_string = False
        strings = [params.stop] if isinstance(params.stop, str) else (params.stop or [])
        holdback = max((len(string) - 1 for string in strings), default=0)
        try:
            for delta in streamer:
                candidate = pending + delta
                stops = [candidate.find(string) for string in strings if string in candidate]
                if stops:
                    text = candidate[: min(stops)]
                    pending = ""
                    stopped_by_string = True
                elif holdback:
                    split = max(0, len(candidate) - holdback)
                    text, pending = candidate[:split], candidate[split:]
                else:
                    text, pending = candidate, ""
                if text:
                    yield StreamOutput(request_id=request_id, text=text)
        finally:
            thread.join()
        if "error" in state:
            raise state["error"]  # type: ignore[misc]
        if pending and not stopped_by_string:
            yield StreamOutput(request_id=request_id, text=pending)
        finish_reason = "stop" if stopped_by_string else state["finish_reason"]
        yield StreamOutput(
            request_id=request_id,
            text="",
            finish_reason=finish_reason,  # type: ignore[arg-type]
            token_ids=state["tokens"],  # type: ignore[arg-type]
        )

    def shutdown(self) -> None:
        """Release model references; do not clear another application's GPU allocator."""
        with self._lock:
            self._closed = True
            self.model = None

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()


__all__ = ["LLM"]
