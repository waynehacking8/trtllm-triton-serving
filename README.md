# TensorRT-LLM + Triton Multi-GPU Serving

Production-style LLM serving on the NVIDIA-native stack — **TensorRT-LLM** engines
served through **Triton Inference Server**, tensor-parallel across **4× H100 (NVLink)**,
benchmarked head-to-head against **vLLM**.

Built to move from "I use vLLM" to "I can stand up the NVIDIA inference stack on real
multi-GPU hardware and reason about the trade-offs." The repo prioritizes a reproducible
**build → serve → benchmark** loop over breadth.

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

## Results — measured on 4× H100

Full writeup: [`results/report.md`](results/report.md). Plots: `results/pareto_h2h.png`,
`results/pareto_quant.png`. Open models used so runs reproduce without gated weights.

### 1. TensorRT-LLM vs vLLM — Qwen2.5-7B-Instruct, TP=2, BF16 (same model, same hardware)

TensorRT-LLM 0.20 served via `trtllm-serve` (compiled engine, `Qwen2ForCausalLM`) vs vLLM,
both out-of-the-box defaults:

| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL | vLLM ITL |
|---|---|---|---|---|---|
| 1 | 119 | 197 | 0.60× | 8.3 ms | 3.9 ms |
| 32 | 3,750 | 6,331 | 0.59× | 8.2 ms | 4.3 ms |
| 128 | 11,917 | 14,680 | 0.81× | 8.9 ms | 5.6 ms |

**Honest finding: out-of-the-box vLLM beats out-of-the-box `trtllm-serve` on this 7B model**
(1.2–1.8× throughput), and the *why* is the interesting part — TRT-LLM's inter-token latency
is pinned at a flat ~8.3 ms vs vLLM's ~4 ms. Two concrete causes, not hand-waving:
- **CUDA-graph decode.** vLLM captures the decode step as a CUDA graph by default; the
  default `trtllm-serve` engine build does not, so every decode step pays kernel-launch
  dispatch — which is exactly the flat ~2× ITL gap (see the latency-wall study in the
  sibling `nccl-collectives-bench` repo).
- **JIT attention kernels.** The container logs `flashinfer: Prebuilt kernels not found,
  using JIT backend` — TRT-LLM is running unoptimized, just-in-time-compiled attention.

The point of an SA benchmark is not to crown a winner but to explain the gap and name the
levers to close it: build the engine with CUDA graphs + paged-context FMHA, prebuild
FlashInfer kernels, and add an **FP8 engine** (TRT-LLM's real advantage on Hopper). Those
are the documented next steps in [`docs/roadmap.md`](docs/roadmap.md). TRT-LLM is built to
*win* this comparison once tuned; out of the box on a non-flagship arch, it does not.

### 2. FP8 vs BF16 — vLLM, Qwen3-8B, TP=2

Same model, same GPUs. **Continuous batching scales throughput ~88× (c=1→128) while ITL
barely moves** (4.3→5.8 ms) — the paged-KV / in-flight-batching property. **FP8** is
1.25–1.30× faster at low concurrency (decode is memory-bandwidth-bound; FP8 halves weight
traffic, ITL −22%), narrowing to 1.07× under heavy batching. At a TTFT-p99 ≤ 200 ms SLA:
BF16 11.5k vs FP8 13.5k tok/s.

> Shared-box hygiene: TRT-LLM on GPUs 2,3 and vLLM on GPUs 4,5 via `--gpus '"device=…"'`,
> never touching the busy GPU 0. Reproduce: serve each stack, `bash bench/sweep.sh <base>
> <tag>`, then `python bench/pareto.py`.

## Status
**Measured head-to-head complete** — TensorRT-LLM (`trtllm-serve`, compiled engine) vs vLLM
on Qwen2.5-7B, plus an FP8/BF16 study on Qwen3-8B. Remaining (roadmap): a CUDA-graph + FP8
**tuned** TRT-LLM engine to close the gap, and the full Triton `tensorrt_llm`-backend
deployment (`scripts/serve_triton.sh`) in place of `trtllm-serve`.
