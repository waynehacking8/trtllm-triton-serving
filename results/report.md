# Serving benchmark — cross-model, stack head-to-head, and quantization (4×H100)

vLLM / TensorRT-LLM OpenAI servers, fixed prompt, max_tokens=256, temp=0, **ignore_eos + min_tokens** so every request decodes exactly 256 tokens on every stack/model (a fair, controlled comparison). Load: `bench/bench.py` (async, streaming, TTFT + inter-token latency). ITL is per-streamed-chunk (≈ per token).

## 1. Cross-model — vLLM, TP=1, BF16

Same server, same hardware (1× H100 each), three models across two generations/families: Llama-3.1-8B (2024) vs Qwen3-8B and Qwen3.5-9B (2026).

**Qwen3-8B** (`xm_qwen3_8b`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 145.4 | 0.0221 | 0.0231 | 6.51 | 6.51 |
| 16 | 2262.2 | 0.0566 | 0.1682 | 6.79 | 6.81 |
| 64 | 7862.5 | 0.1271 | 0.163 | 7.59 | 7.79 |
| 128 | 13410.7 | 0.2033 | 0.3166 | 8.59 | 8.73 |

**Qwen3.5-9B** (`xm_qwen35_9b`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 127.1 | 0.0284 | 0.0338 | 6.81 | 6.81 |
| 16 | 1978.4 | 0.0694 | 0.4282 | 7.63 | 8.26 |
| 64 | 6233.8 | 0.1404 | 0.2206 | 9.63 | 9.75 |
| 128 | 9355.9 | 0.2675 | 0.5493 | 12.41 | 12.82 |

**Llama-3.1-8B** (`xm_llama31_8b`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 152.4 | 0.0224 | 0.0239 | 6.19 | 6.2 |
| 16 | 2378.1 | 0.0559 | 0.1591 | 6.44 | 6.83 |
| 64 | 8315.1 | 0.1045 | 0.1617 | 7.24 | 7.35 |
| 128 | 13770.7 | 0.2096 | 0.2627 | 8.39 | 8.51 |

Throughput (tok/s) by concurrency:

| model | c1 | c16 | c64 | c128 |
|---|---|---|---|---|
| Qwen3-8B | 145 | 2262 | 7862 | 13411 |
| Qwen3.5-9B | 127 | 1978 | 6234 | 9356 |
| Llama-3.1-8B | 152 | 2378 | 8315 | 13771 |

Reads as a model-selection table: at a target concurrency, which model gives the most tok/s per H100. Larger/newer models trade throughput for capability — the SA job is to put that trade in front of the customer with numbers.

### Throughput at a TTFT-p99 SLA (tok/s)

| stack/model | p99≤100ms | p99≤200ms | p99≤500ms |
|---|---|---|---|
| Qwen3-8B | 145 | 7862 | 13411 |
| Qwen3.5-9B | 127 | 127 | 6234 |
| Llama-3.1-8B | 152 | 8315 | 13771 |

## 2. Stack head-to-head — TensorRT-LLM vs vLLM, Llama-3.1-8B, TP=2, BF16

Same model (Llama-3.1-8B, supported on the TRT-LLM 0.20 **compiled-engine** path), same parallelism, same controlled 256-token decode.

**TensorRT-LLM (Llama-3.1-8B)** (`trtllm_llama31`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 111.1 | 0.0186 | 0.0501 | 8.88 | 8.9 |
| 4 | 507.2 | 0.0243 | 0.0318 | 7.81 | 7.84 |
| 16 | 1937.8 | 0.046 | 0.0623 | 8.07 | 8.16 |
| 32 | 3619.3 | 0.0815 | 0.1292 | 8.53 | 8.63 |
| 64 | 6513.8 | 0.134 | 0.205 | 9.28 | 9.38 |
| 128 | 11305.3 | 0.2879 | 0.7875 | 9.75 | 10.18 |

**vLLM (Llama-3.1-8B)** (`vllm_llama31`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 227.7 | 0.0159 | 0.0176 | 4.04 | 4.04 |
| 4 | 937.0 | 0.023 | 0.1103 | 4.14 | 4.5 |
| 16 | 3529.9 | 0.046 | 0.0791 | 4.34 | 4.37 |
| 32 | 6831.4 | 0.0526 | 0.084 | 4.45 | 4.48 |
| 64 | 12031.2 | 0.1075 | 0.1567 | 4.85 | 4.93 |
| 128 | 19658.7 | 0.2104 | 0.3598 | 5.55 | 5.92 |

| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL ms | vLLM ITL ms |
|---|---|---|---|---|---|
| 1 | 111 | 228 | 0.49× | 8.88 | 4.04 |
| 4 | 507 | 937 | 0.54× | 7.81 | 4.14 |
| 16 | 1938 | 3530 | 0.55× | 8.07 | 4.34 |
| 32 | 3619 | 6831 | 0.53× | 8.53 | 4.45 |
| 64 | 6514 | 12031 | 0.54× | 9.28 | 4.85 |
| 128 | 11305 | 19659 | 0.58× | 9.75 | 5.55 |

### Throughput at a TTFT-p99 SLA (tok/s)

| stack/model | p99≤100ms | p99≤200ms | p99≤500ms |
|---|---|---|---|
| TensorRT-LLM (Llama-3.1-8B) | 1938 | 3619 | 6514 |
| vLLM (Llama-3.1-8B) | 6831 | 12031 | 19659 |

## 3. Quantization — vLLM, Qwen3-8B, TP=2 (FP8 vs BF16)

**vLLM BF16 (Qwen3-8B)** (`vllm_bf16`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 215.3 | 0.0165 | 0.0175 | 4.29 | 4.3 |
| 4 | 884.7 | 0.0266 | 0.0322 | 4.38 | 4.74 |
| 16 | 3353.4 | 0.0412 | 0.0886 | 4.59 | 4.63 |
| 32 | 6189.0 | 0.0738 | 0.096 | 4.88 | 4.97 |
| 64 | 11525.9 | 0.1029 | 0.1581 | 5.08 | 5.15 |
| 128 | 18584.7 | 0.2072 | 0.3159 | 5.97 | 6.17 |

**vLLM FP8 (Qwen3-8B)** (`vllm_fp8`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 273.0 | 0.0159 | 0.0172 | 3.31 | 3.32 |
| 4 | 1135.0 | 0.0242 | 0.0303 | 3.38 | 3.76 |
| 16 | 4321.2 | 0.0389 | 0.0765 | 3.53 | 3.56 |
| 32 | 7840.9 | 0.0664 | 0.1298 | 3.76 | 3.89 |
| 64 | 13792.4 | 0.1067 | 0.1762 | 4.13 | 4.25 |
| 128 | 20793.7 | 0.2027 | 0.2815 | 5.26 | 5.51 |

| concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup | BF16 ITL ms | FP8 ITL ms |
|---|---|---|---|---|---|
| 1 | 215 | 273 | 1.27× | 4.29 | 3.31 |
| 4 | 885 | 1135 | 1.28× | 4.38 | 3.38 |
| 16 | 3353 | 4321 | 1.29× | 4.59 | 3.53 |
| 32 | 6189 | 7841 | 1.27× | 4.88 | 3.76 |
| 64 | 11526 | 13792 | 1.20× | 5.08 | 4.13 |
| 128 | 18585 | 20794 | 1.12× | 5.97 | 5.26 |

FP8 (Hopper FP8 tensor cores) wins most at low concurrency where decode is memory-bandwidth-bound; the edge narrows under heavy batching.

