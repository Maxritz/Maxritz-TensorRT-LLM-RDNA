# RDNA4 porting roadmap and extension ledger

This document is the implementation backlog for the ROCm/RDNA4 backend. It is
intentionally more precise than a generic list of CUDA dependencies: a feature
is either already usable, can be implemented with portable PyTorch/Hugging Face
code, needs a separately qualified ROCm extension, or has no meaningful
"port" because it is a TensorRT/NVIDIA artifact.

**Snapshot date:** 2026-10-08

**Scope:** Linux + genuine `gfx1200`/`gfx1201`; no architecture spoofing.

**Evidence:** the current backend, the RDNA4 ISA text files in this checkout,
and the linked upstream projects at the revisions inspected on this date.

> [!IMPORTANT]
> “Feature parity” must not mean accepting a CUDA option and silently doing
> something else. Every optional extension needs a capability probe, explicit
> selection, a CPU/reference correctness test, an RDNA4 GPU test, and a
> performance result before it becomes the default.

## Current foundation

| Area | Current state | Boundary |
| --- | --- | --- |
| Model execution | Hugging Face causal-LM generation, eager/SDPA attention, one device | Unquantized FP32/FP16/BF16 checkpoints only |
| Native HIP primitives | RMSNorm, add+RMSNorm, 1D LayerNorm, gated SiLU/GELU, partial RoPE, GQA attention | Correctness-first; not a flash/paged implementation |
| Model integration | Llama, Mistral, Qwen2/Qwen3 native attention; Phi3/Llama/Mistral/Qwen native norms | Only when `kernels="hip"` is selected |
| Sampling | Greedy, temperature, top-p/k, repetition penalty, seed, EOS/stop strings, HF beam search | No logits/logprob materialization or advanced penalties |
| HTTP API | OpenAI-style completions/chat, health, model list; single-sequence SSE streaming | Generation is serialized; no in-flight batching |
| Qualification | CPU reference suite and target-machine HIP validator | No RDNA4 GPU result has been checked in this sandbox |
| Observability | Opt-in component spans, host profile, device event spans, CPU/RAM/GPU/VRAM telemetry | Not a ROCm kernel-instruction profiler |

The new portable streaming implementation is deliberately narrow: it uses the
Transformers streamer and therefore supports one prompt, `n=1`, and
`beam_width=1`. A batch/beam stream must wait for an actual request scheduler
rather than risk interleaved or incorrectly attributed deltas.

## Complete upstream feature ledger

The table includes every feature area represented by `docs/source/features/`
and the relevant public API/serve surfaces. “Portable first” means it can be
implemented without an RDNA4-specific third-party kernel. “Extension” means
there is a credible candidate, **not** that it is already an approved runtime
dependency.

| Upstream capability | ROCm state today | Replacement route | Priority / prerequisite |
| --- | --- | --- | --- |
| TensorRT engine build, deserialize, plugins and plans | **Not portable** | Define ROCm as an HF/safetensors (and later GGUF) execution format; do not attempt to load `.plan`/`.engine` | Architecture boundary; no compatibility shim |
| Checkpoint loading / ModelExpress | HF local checkpoint only | Add explicit safetensors/sharded-index loader first; evaluate ModelExpress only if it has a non-CUDA contract | P2 |
| Dense MHA/MQA/GQA attention | SDPA/eager plus small native HIP kernel | Default PyTorch SDPA/AOTriton; optional RDNA AITER FlashAttention | P1 kernel path |
| Flash attention / FMHA / FlashInfer | **Not ported** | AITER gfx1201 FlyDSL flash-attention kernel; AOTriton/PyTorch SDPA fallback | P1, target GPU qualification |
| MLA, sparse/skip-softmax attention | **Not ported** | AITER/vLLM ROCm implementations are reference material; port only algorithms with a native RDNA4 path | P4; model-specific |
| Paged attention and paged KV manager | **Partial** | `PagedKVCache` has page allocation/refcount/COW prefix sharing and `paged_attention` provides a portable HIP/PyTorch execution fallback; a fused RDNA decode kernel and HF cache adapter remain | P1 scheduler + P2 kernel |
| Prefix cache / prefix tokenization cache | **Not ported** | Token-ID prefix trie first, then page-level KV sharing after paged cache exists | P2; depends on paged KV |
| Chunked prefill, IFB, request scheduling, overlap scheduler | **Partial** | `ContinuousBatchScheduler` now defines admission and token-step decode rounds over live page-table requests; an HF cache adapter must supply its prefill/decode callbacks before serving integration | P1; must precede throughput claims |
| CUDA graphs / piecewise graphs | **Not ported** | Qualify PyTorch `torch.cuda.CUDAGraph` under HIP and call the result a HIP graph | P3; fixed-shape scheduler required |
| Additional outputs / prompt and completion logprobs | **Not ported** | Portable logits processors and selected-token `log_softmax`; do not materialize all vocabulary logits by default | P1 |
| Advanced sampling (bad/stop token IDs, penalties, min-p, logits processors) | Partial | Extend portable sampler before replacing it with a fused GPU sampler | P1 |
| Guided decoding (JSON, regex, grammar, structural tags) | **Partial** | Portable exact `guided_choice` token-prefix mask is implemented; JSON/regex/grammar require an Outlines constraint adapter and tokenizer qualification | P2; portable sampler required |
| Streaming, OpenAI SSE | **Implemented, constrained** | Transformers `TextIteratorStreamer` and SSE adapter | P0 complete; GPU qualification still required |
| AsyncLLM, cancellation, backpressure | **Partial** | `AsyncLLM` has a bounded serialized queue and async stream bridge; queued cancellation works, while active model cancellation still needs scheduler/request lifecycle support in the generation loop | P1 |
| LoRA / multi-LoRA and adapter cache | **Partial** | Optional PEFT single-adapter load/optional merge is implemented; add an adapter registry and batched fused LoRA only after the scheduler/GEMM path | P2 then P4 |
| Quantized weights, quantized KV, ModelOpt formats | **Partial** | Explicit bitsandbytes 4/8-bit HF loading is available behind a real ROCm-device dependency; quantized KV, ModelOpt formats, accuracy and performance gates remain | P2, each format needs accuracy/perf gates |
| KV cache compression/offload | **Partial** | `PagedKVCache(quantization="int8")` stores per-token/head scaled INT8 K/V and dequantizes at portable attention dispatch; host offload, NVFP4 and fused quantized decode remain | P4 |
| Speculative decoding (draft/target, n-gram, EAGLE, MTP, PARD, DFlash) | **Partial** | HF assisted draft/target generation is available for non-streaming `n=1`/beam-1 requests; scheduler-aware acceptance kernels and advanced variants remain | P3, requires paged KV + logprobs |
| Encoder-only embeddings / reranking | **Partial** | `EmbeddingLLM`, AutoModel hidden-state mean/CLS/last-token pooling and a standalone `/v1/embeddings` server are implemented; reranker, dynamic batching and dimension projection remain | P1 |
| Multimodal LLM | **Not ported** | Transformers processor/model path, then batched vision encoder scheduling | P3, per-model qualification |
| Visual generation, quantized/sparse VisualGen, VisualGen graph | **Not ported** | Separate Diffusers/ROCm pipeline; NVIDIA CUTEDSL/FlashInfer code cannot be transplanted | P5 |
| MoE routing / fused MoE | **Not ported** | Dense/eager correctness baseline, then AITER RDNA-compatible GEMM/router kernels | P4; target-specific |
| Tensor/pipeline/data/context/expert/Helix parallelism | **Not ported** | PyTorch distributed backed by RCCL; introduce TP before PP/EP/CP | P4; multi-GPU RDNA4 lab needed |
| Disaggregated serving / KV connectors / NIXL/Mooncake | **Not ported** | Begin with a host-copy reference transport; evaluate RCCL point-to-point and Iris only after ownership/lifetime tests | P5 |
| Ray orchestration, sub-agent routing, post-processor hooks | **Not ported** | Keep these backend-neutral at the HTTP/request layer | P3/P5 depending on scheduler |
| Evaluation CLI | **Not ported** | Adapter to `lm-evaluation-harness`; verify offline/local model behavior | P2 |
| Native attention extras: attention maps, softcap, dropout | **Not ported** | Keep native kernel inference-only; SDPA/eager is the correct fallback for these requirements | P3 only if needed |

## Extension candidates and decisions

### 1. AITER — primary kernel integration candidate

- **Project:** [ROCm/aiter](https://github.com/ROCm/aiter) (MIT).
- **Why:** AMD describes it as its high-performance ROCm operator library. It
  includes attention, RoPE/KV-cache, RMSNorm, GEMM, fused MoE, quantization and
  communication code; it is used by ROCm paths in vLLM and SGLang.
- **RDNA4 evidence:** its README lists `gfx1201` as *experimental*, explicitly
  stating that Triton, most FlyDSL, and most HIP kernels can run, while many CK
  and assembly kernels remain CDNA-only. It contains a dedicated
  [`flash_attn_func_gfx1201.py`](https://github.com/ROCm/aiter/blob/main/aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py)
  that uses wave32 `16x16x16` WMMA and online softmax.
- **Adoption:** use a lazy optional import with a version/architecture/shape
  gate; never make it part of `requirements-rocm.txt` until a released wheel or
  source pin has passed target tests. Start with flash-attention forward only.
- **Important blocker:** AITER's current `paged_attn.py` explicitly reports
  that its ROCm custom paged-attention path is unsupported on Navi (`gfx1*`).
  Therefore it is **not** a drop-in RDNA4 paged-KV solution today.

### 2. PyTorch SDPA + AOTriton — safe attention baseline

- **Projects:** [PyTorch](https://github.com/pytorch/pytorch) and
  [ROCm/aotriton](https://github.com/ROCm/aotriton) (MIT).
- **Why:** the present backend already uses SDPA. AOTriton is the ROCm
  FlashAttention library used by PyTorch's HIP SDPA builds.
- **Decision:** retain SDPA as the correctness and broad-model fallback. Do not
  vendor an AOTriton API directly: it is a fast-moving internal library and its
  README warns that the FlashAttention API changes. Qualify the installed HIP
  PyTorch wheel instead.

### 3. Triton AMD — portable custom-kernel language

- **Project:** [Triton](https://github.com/triton-lang/triton) (MIT).
- **Why:** its upstream README advertises AMD GPU support on ROCm 6.2+ and can
  dump generated AMDGCN for inspection. It is the preferred language for a
  portable page table, cache write, sampling, and small fused kernels.
- **Decision:** add as an optional development dependency only. Build a
  capability probe (`torch.version.hip`, real `gfx1200/gfx1201`, compile smoke
  test) before exposing any Triton backend.

### 4. hipBLASLt / ROCm libraries — projections, not an engine replacement

- **Project:** [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)
  (the standalone hipBLASLt repository is retired).
- **Why:** GEMMs should go through PyTorch/hipBLASLt before maintaining a custom
  WMMA projection kernel.
- **Decision:** use PyTorch's selected ROCm BLAS path first. Only add a custom
  GEMM provider when shape traces show it wins and it can dispatch safely on
  RDNA4.

### 5. Quantization: bitsandbytes plus AITER, with separate formats

- **Projects:** [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
  (MIT), AITER quantization ops, and [llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT) for a separately served GGUF route.
- **Evidence:** bitsandbytes' support table lists ROCm RDNA `gfx120X`; AITER
  has HIP/Triton quantization helpers including integer, FP8 and block-scaled
  paths; llama.cpp supports HIP and GGUF quantization.
- **Decision:** do **not** label these interchangeable. First implement a
  `bitsandbytes`-backed HF loading option with an explicit support probe. Treat
  GGUF as a different model/runtime adapter, not a TensorRT-LLM checkpoint.
  FP8/FP4/MX formats each need their own numerical and hardware-performance
  qualification.

### 6. LoRA: PEFT for correctness, AITER GEMM later for throughput

- **Project:** [Hugging Face PEFT](https://github.com/huggingface/peft)
  (Apache-2.0).
- **Decision:** add a lazy, optional PEFT loader for a single adapter and
  explicit merge/unmerge semantics. Multi-LoRA request routing and cache
  residency belong after the scheduler/page manager. Do not couple initial LoRA
  support to a fused CUDA-style GEMM.

### 7. Guided decoding: Outlines first

- **Project:** [Outlines](https://github.com/dottxt-ai/outlines)
  (Apache-2.0).
- **Decision:** adapt its grammar/schema result to a CPU-built allowed-token
  mask consumed by the portable sampler. This makes correctness testable on CPU
  and avoids claiming an xgrammar CUDA integration works on HIP. Cache compiled
  constraints by tokenizer/model/schema hash.

### 8. Distributed execution: RCCL and Iris, only with real topology tests

- **Projects:** [RCCL](https://github.com/ROCm/rccl) and
  [Iris](https://github.com/ROCm/iris).
- **Why:** RCCL supplies standard all-reduce/all-gather/reduce-scatter/all-to-all
  and point-to-point primitives; AITER can use Iris for GPU-initiated Triton
  collectives.
- **Decision:** build a simple PyTorch-distributed tensor-parallel prototype on
  RCCL first. Do not assert a NIXL/Mooncake replacement or multi-node support
  until request ownership, cache-layout conversion, failure recovery and target
  interconnect tests are present.

### 9. vLLM and SGLang — reference implementations, not vendored dependencies

- **Projects:** [vLLM](https://github.com/vllm-project/vllm) and
  [SGLang](https://github.com/sgl-project/sglang), both Apache-2.0.
- **Why:** they show production scheduler, paged cache, streaming, speculative
  and ROCm dispatch designs. vLLM's current ROCm platform recognizes `gfx1201`
  devices and has explicit RDNA AITER backend selection.
- **Decision:** use them to validate API contracts and architecture; do not
  copy their scheduler into this repository piecemeal. The portable backend
  needs its own small, tested request/page abstraction.

## RDNA4 ISA implications

The local `rdna4-instruction-set-architecture.txt` resolves a key question:
RDNA4 has wave32 WMMA and sparse WMMA instructions, including `16x16x16`
FP16/BF16-to-FP32 and FP8/BF8-to-FP32 operations. It also has
`GLOBAL_LOAD_TR_B128` and `GLOBAL_LOAD_TR_B64` load-transpose operations for
WMMA fragments.

That makes an RDNA4-native dense FlashAttention/GEMM implementation plausible,
but it does **not** make CDNA MFMA code or NVIDIA CUTLASS fragments valid. The
fragment layout, execution width, `EXEC` requirements and hazards are different.
The safest order is:

1. profile SDPA/hipBLASLt first;
2. qualify AITER's existing gfx1201 wave32 implementation;
3. use `ROCm/amd_matrix_instruction_calculator` to generate the exact register
   maps for an uncovered shape;
4. add one isolated HIP/FlyDSL/Triton operator with reference tests;
5. inspect emitted AMDGCN and measure it on actual hardware.

Do not hand-emit code objects or transplant an MFMA/CUTLASS tile mapping. In
particular, VOPD is wave32-only, and WMMA load-transpose instructions require a
full active execution mask; neither condition is satisfied by simply changing a
CUDA warp constant.

## Parallel workstreams and merge order

The workstreams can proceed independently, but the merge order is deliberate.
Each workstream owns its own tests and must leave unsupported inputs explicit.

| Stream | Deliverables | Depends on | Definition of done |
| --- | --- | --- | --- |
| A — portable API | Streaming, logprobs, sampler features, async request lifecycle | Current HF backend | CPU API tests + OpenAI wire tests; no silent option loss |
| B — scheduler/KV | Request states, cancellation, page allocator, prefix cache, chunked prefill | A | CPU scheduler invariants, GPU page-table tests, load test |
| C — kernels | AITER capability adapter, SDPA selection, flash attention, cache writes | B for paged decode | Per-op reference parity on gfx1200 and gfx1201; benchmark decision recorded |
| D — adapters/models | PEFT, embeddings, guided decoding, spec baseline, multimodal | A; B for batching | Offline fixture tests and model-family qualification matrix |
| E — quant/MoE | bnb loading, AITER quant/GEMM, dense MoE baseline | C | Accuracy delta, memory, latency and fallback tests by format |
| F — distributed/serve | RCCL TP, cache transport, PP/EP, Ray/disaggregation | B/C/E | 2+ GPU correctness, fault/restart and topology-specific benchmarks |
| G — VisualGen | ROCm-native visual-generation runtime | C/D | Separate model suites; no CUDA/CUTEDSL code path exposed as RDNA4 |

### Recommended implementation sequence

1. **P0: preserve correctness and finish portable HTTP/API gaps.** Streaming is
   now in this category. Add logprobs, deterministic tests and cancellation
   next.
2. **P1: unlock real serving throughput.** Design the request/page interfaces,
   add the scheduler, embeddings and a first target-qualified flash-attention
   adapter. Do not advertise IFB before this step.
3. **P2: improve usability and fit.** PEFT, guided decoding, evaluation,
   quantized loader, prefix reuse and safe host cache offload.
4. **P3/P4: model and multi-GPU depth.** Speculative decoding, multimodal,
   graphs, MoE and RCCL tensor parallelism.
5. **P5: advanced deployment.** Disaggregation, cache connectors, Ray and
   visual generation.

## Required qualification record for every extension

Every candidate must add a row to the support matrix containing:

- exact project version/commit, license and installation source;
- ROCm version, HIP PyTorch version, `gfx1200` or `gfx1201`, and host compiler;
- enabled shapes/dtypes/model families and explicit fallback/rejection rules;
- CPU reference test and RDNA4 numerical tolerances;
- stream/nondefault-stream/memory-lifetime coverage where applicable;
- cold-start/JIT behavior, latency, throughput and VRAM delta against SDPA/eager;
- failure-mode test: missing package, incompatible architecture, unsupported
  layout, compilation failure and multi-request cancellation;
- a removal/rollback path that leaves the pure PyTorch backend usable.

A candidate that works only by setting `HSA_OVERRIDE_GFX_VERSION` or that has
only CDNA test coverage is not an RDNA4 port.
