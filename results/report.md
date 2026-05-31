# Serving benchmark — TensorRT-LLM vs vLLM, and FP8 vs BF16 (4×H100)

vLLM and TensorRT-LLM OpenAI servers, TP=2, same GPUs class, fixed prompt, max_tokens=256, temp=0. Load from `bench/bench.py` (async, streaming, per-request TTFT + inter-token latency).

> **Note:** these numbers predate the `ignore_eos` fix in `bench.py`, so the two stacks decoded different output lengths (vLLM stopped at EOS, TRT-LLM ran to 256). The **per-token ITL** comparison is length-independent and stands; the **aggregate throughput ratio** will tighten on a re-run with fixed-length decode. The README Results section is the up-to-date narrative.

## 1. TensorRT-LLM vs vLLM — Qwen2.5-7B-Instruct, TP=2, BF16

Same model, same parallelism — TensorRT-LLM uses the compiled engine path (`Qwen2ForCausalLM`, supported in TRT-LLM 0.20); vLLM is the baseline.

**TensorRT-LLM** (`trtllm_qwen25`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 119.0 | 0.0173 | 0.0464 | 8.29 | 8.34 |
| 4 | 511.8 | 0.0235 | 0.0307 | 7.73 | 7.8 |
| 16 | 1883.1 | 0.0539 | 0.0855 | 8.3 | 8.37 |
| 32 | 3749.5 | 0.0793 | 0.1222 | 8.21 | 8.41 |
| 64 | 6891.1 | 0.1461 | 0.1882 | 8.71 | 8.9 |
| 128 | 11917.1 | 0.3082 | 1.6864 | 8.94 | 9.53 |

**vLLM** (`vllm_qwen25`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 197.4 | 0.0162 | 0.0166 | 3.89 | 3.89 |
| 4 | 903.5 | 0.0252 | 0.0301 | 3.95 | 4.96 |
| 16 | 3345.5 | 0.0443 | 0.0908 | 4.11 | 4.18 |
| 32 | 6330.6 | 0.0533 | 0.0959 | 4.28 | 4.69 |
| 64 | 10555.7 | 0.0984 | 0.1741 | 4.6 | 4.81 |
| 128 | 14679.5 | 0.2267 | 0.3483 | 5.6 | 6.67 |

Throughput ratio (TensorRT-LLM ÷ vLLM) by concurrency:

| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL ms | vLLM ITL ms |
|---|---|---|---|---|---|
| 1 | 119 | 197 | 0.60× | 8.29 | 3.89 |
| 4 | 512 | 904 | 0.57× | 7.73 | 3.95 |
| 16 | 1883 | 3346 | 0.56× | 8.30 | 4.11 |
| 32 | 3750 | 6331 | 0.59× | 8.21 | 4.28 |
| 64 | 6891 | 10556 | 0.65× | 8.71 | 4.60 |
| 128 | 11917 | 14680 | 0.81× | 8.94 | 5.60 |

### Throughput at a TTFT-p99 SLA (tok/s)

| stack | p99≤100ms | p99≤200ms | p99≤500ms |
|---|---|---|---|
| TensorRT-LLM | 1883 | 6891 | 6891 |
| vLLM | 6331 | 10556 | 14680 |

## 2. FP8 vs BF16 — vLLM, Qwen3-8B, TP=2

**vLLM BF16** (`vllm_bf16`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 215.3 | 0.0171 | 0.0173 | 4.29 | 4.29 |
| 4 | 897.5 | 0.0248 | 0.0294 | 4.36 | 4.4 |
| 16 | 3291.2 | 0.048 | 0.0945 | 4.6 | 4.97 |
| 32 | 6299.3 | 0.0566 | 0.0798 | 4.84 | 4.89 |
| 64 | 11474.0 | 0.1109 | 0.1752 | 5.07 | 5.4 |
| 128 | 18915.0 | 0.2096 | 0.3292 | 5.82 | 6.18 |

**vLLM FP8** (`vllm_fp8`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 272.3 | 0.0153 | 0.0159 | 3.33 | 3.34 |
| 4 | 1136.8 | 0.0235 | 0.0295 | 3.38 | 3.75 |
| 16 | 4263.0 | 0.0447 | 0.0825 | 3.54 | 3.57 |
| 32 | 7895.3 | 0.0683 | 0.087 | 3.76 | 3.87 |
| 64 | 13469.3 | 0.1396 | 0.179 | 4.17 | 4.42 |
| 128 | 20232.4 | 0.2262 | 0.3155 | 5.3 | 5.76 |

| concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup | BF16 ITL ms | FP8 ITL ms |
|---|---|---|---|---|---|
| 1 | 215 | 272 | 1.26× | 4.29 | 3.33 |
| 4 | 898 | 1137 | 1.27× | 4.36 | 3.38 |
| 16 | 3291 | 4263 | 1.30× | 4.60 | 3.54 |
| 32 | 6299 | 7895 | 1.25× | 4.84 | 3.76 |
| 64 | 11474 | 13469 | 1.17× | 5.07 | 4.17 |
| 128 | 18915 | 20232 | 1.07× | 5.82 | 5.30 |

FP8 (Hopper FP8 tensor cores) wins most at low concurrency where decode is memory-bandwidth-bound; the edge narrows under heavy batching.

### Throughput at a TTFT-p99 SLA (tok/s)

| stack | p99≤100ms | p99≤200ms | p99≤500ms |
|---|---|---|---|
| vLLM BF16 | 6299 | 11474 | 18915 |
| vLLM FP8 | 7895 | 13469 | 20232 |

