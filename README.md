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

## Results
Populated after running on the 4×H100 box — see `results/`. **(in progress)**

## Status
Scaffold + runnable harness complete; engine build and benchmarks pending a run on the 4×H100 box.
