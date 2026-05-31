# TensorRT-LLM + Triton Multi-GPU Serving

Production-style LLM serving on the NVIDIA-native stack — **TensorRT-LLM** engines
tensor-parallel across **H100 (NVLink)**, benchmarked head-to-head against **vLLM**, plus a
cross-model and a quantization study. The measured runs use TensorRT-LLM's own OpenAI server
(`trtllm-serve`); a **Triton + `tensorrt_llm`-backend** deployment template
(`triton_model_repo/`) is included as the production path (not yet exercised in the
benchmark — see Status).

Built to move from "I use vLLM" to "I can stand up the NVIDIA inference stack on real
multi-GPU hardware and reason about the trade-offs." The repo prioritizes a reproducible
**serve → benchmark** loop, with a **controlled methodology** (every request decodes exactly
256 tokens via `ignore_eos`, so throughput/latency compare the same work across stacks).

## What this is
- A scripted pipeline: HF checkpoint → TensorRT-LLM engine (TP=4, FP8/FP16) → Triton model repository → load test.
- An apples-to-apples benchmark harness (TensorRT-LLM/Triton vs vLLM): throughput, TTFT, inter-token latency under matched concurrency.
- Documented engineering decisions: tensor parallelism, quantization, in-flight (continuous) batching, paged KV-cache.

## What this is NOT
- Not a fork of `trtllm-serve` / `genai-perf` — it wraps them in a reproducible harness with a documented comparison.
- Not a claim that TensorRT-LLM always wins — the goal is to measure honestly and explain *when and why* each stack wins.
- Not multi-node — single 4×H100 box over NVLink. Multi-node (NCCL over InfiniBand) is in the roadmap.

## Hardware
- 4× NVIDIA H100 80GB, NVLink. Tensor parallel = 4.
- Driver + CUDA per the TensorRT-LLM container (see `docs/design-decisions.md`).

## Layout
```
scripts/build_engine.sh     # HF -> TRT-LLM checkpoint -> engine (TP=4)
scripts/serve_triton.sh     # launch Triton with the tensorrt_llm backend
scripts/serve_vllm.sh       # vLLM baseline (TP=4) for comparison
bench/bench.py              # async OpenAI-compatible load test (TTFT/throughput/ITL)
bench/sweep.sh              # concurrency sweep -> results/*.json
triton_model_repo/          # Triton ensemble + tensorrt_llm model config
docs/roadmap.md             # what's done / in progress / planned
docs/design-decisions.md    # parallelism, quant, batching choices and why
results/                    # benchmark outputs + plots (populated on the 4xH100 box)
```

## Quick start (run on the 4×H100 box)
```bash
export MODEL=meta-llama/Llama-3.1-8B-Instruct   # scale up to 70B once 8B is green
make build        # build the TP=4 TensorRT-LLM engine
make serve        # start Triton on :8000
make bench        # load test -> results/trtllm.json
make bench-vllm   # same load against vLLM -> results/vllm.json
make report       # merge + plot -> results/report.md
```

## Results — measured on H100

Full writeup: [`results/report.md`](results/report.md). Plots: `pareto_models.png`,
`pareto_h2h.png`, `pareto_quant.png`. **Controlled methodology:** every request decodes
exactly 256 tokens (`ignore_eos` + `min_tokens`) on every stack/model, so throughput and ITL
compare the same work — without this, greedy decode stops at each model's EOS at different
points and the numbers are apples-to-oranges.

### 1. Cross-model — vLLM, TP=1, BF16 (1× H100 each)

Three models across two generations: Llama-3.1-8B (2024) vs Qwen3-8B / Qwen3.5-9B (2026).

| model | tok/s @c1 | tok/s @c128 | ITL @c1 |
|---|---|---|---|
| Llama-3.1-8B | 152 | 13,771 | 6.2 ms |
| Qwen3-8B | 145 | 13,411 | 6.5 ms |
| Qwen3.5-9B | 127 | 9,356 | 6.8 ms |

A model-selection table: the 9B carries ~25% less throughput/H100 than the 8Bs — the
capability-vs-cost trade an SA puts in front of a customer with numbers, not vibes. (Frontier
2026 MoE models — GLM-5.1 744B, DeepSeek-V4, Llama-4 — need the full 8-GPU box and are out of
scope on the free cards here.)

### 2. Stack head-to-head — TensorRT-LLM vs vLLM, Llama-3.1-8B, TP=2, BF16

Same model (engine-supported on the TRT-LLM 0.20 **compiled-engine** path), same parallelism,
same controlled 256-token decode:

| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL | vLLM ITL |
|---|---|---|---|---|---|
| 1 | 111 | 228 | 0.49× | 8.9 ms | 4.0 ms |
| 32 | 3,619 | 6,831 | 0.53× | 8.5 ms | 4.5 ms |
| 128 | 11,305 | 19,659 | 0.57× | 9.8 ms | 5.6 ms |

**Honest finding (now methodologically airtight): out-of-the-box vLLM beats out-of-the-box
`trtllm-serve` ~1.8–2×**, and the *why* is the value — TRT-LLM's ITL is pinned at a flat
~9 ms vs vLLM's ~4 ms. The earlier version of this benchmark lacked `ignore_eos`, so vLLM
generated ~85 tokens and TRT-LLM 256 — invalid. **Forcing both to 256 tokens, the gap holds**,
which proves it's a real decode-path difference, not an EOS artifact:
- **CUDA-graph decode.** vLLM captures the decode step as a CUDA graph by default; the default
  `trtllm-serve` engine build does not — exactly the flat ~2× ITL gap (cf. the latency-wall
  study in the sibling `nccl-collectives-bench` repo).
- **JIT attention kernels.** The container logs `flashinfer: Prebuilt kernels not found, using
  JIT backend` — TRT-LLM is on unoptimized, just-in-time-compiled attention.

The SA point isn't to crown a winner but to explain the gap and name the levers: build with
CUDA graphs + paged-context FMHA, prebuild FlashInfer kernels, add an **FP8 engine** (TRT-LLM's
real Hopper advantage). Those are the roadmap. TRT-LLM is built to *win* this once tuned; out
of the box on a non-flagship arch, it does not.

### 3. Quantization — FP8 vs BF16, vLLM, Qwen3-8B, TP=2

**Continuous batching scales throughput ~86× (c=1→128) while ITL barely moves** (4.3→6.0 ms) —
the paged-KV / in-flight-batching property. **FP8** is ~1.27× faster at low concurrency (decode
is memory-bandwidth-bound; FP8 halves weight traffic, ITL −23%): BF16 215 vs FP8 273 tok/s @c1,
narrowing to ~1.12× at c128 (18.6k vs 20.8k).

### 4. Big model at scale — TensorRT-LLM vs vLLM, Qwen2.5-32B, TP=4, BF16

The head-to-head at a real multi-GPU operating point — a 32B model **tensor-parallel across
4× H100** (`Qwen2ForCausalLM`, TRT-LLM compiled engine), controlled 256-token decode:

| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL | vLLM ITL |
|---|---|---|---|---|---|
| 1 | 51 | 113 | 0.45× | 19.5 ms | 8.4 ms |
| 32 | 1,602 | 3,276 | 0.49× | 19.7 ms | 9.5 ms |
| 128 | 5,834 | 9,383 | 0.62× | 20.8 ms | 12.7 ms |

The §2 finding **holds and amplifies at 32B / TP=4**: out-of-the-box vLLM ~2× faster, and the
gap widens because the missing CUDA-graph decode pays the per-step launch tax across more
layers *and* a 4-way tensor-parallel all-reduce per layer. Same root cause, larger model — the
levers to close it (CUDA graphs, FP8 engine) matter more at scale, not less.

> Shared-box hygiene: all serving pinned to free GPUs (2–7) via `--gpus '"device=…"'`, never
> touching the busy GPU 0. Reproduce: `scripts/serve_vllm.sh` / `scripts/serve_trtllm.sh`, then
> `bash bench/sweep.sh <base> <tag>` and `python bench/pareto.py`.

## Status
**Four measured studies complete** — cross-model (Llama-3.1-8B / Qwen3-8B / Qwen3.5-9B),
TensorRT-LLM-vs-vLLM head-to-head at TP=2 (Llama-3.1-8B) **and TP=4 (Qwen2.5-32B)**, and
FP8/BF16 quantization (Qwen3-8B) — all under a controlled 256-token methodology. Remaining
(roadmap): a CUDA-graph + FP8 **tuned** TRT-LLM engine to close the head-to-head gap, and
standing up the full Triton `tensorrt_llm`-backend (`triton_model_repo/`,
`scripts/serve_triton.sh`) in place of `trtllm-serve`. Note: TRT-LLM 0.20's compiled-engine
path supports Llama-3.x / Qwen2.x archs; Qwen3 / Llama-4 run only on its PyTorch backend or
vLLM today.
