# Serving benchmark — TensorRT-LLM vs vLLM, cross-model & quantization (H100)

OpenAI-compatible servers, fixed prompt, **every request decodes exactly 256 tokens** (`ignore_eos`+`min_tokens`) so throughput/latency compare the same work. **All TRT-LLM runs use CUDA graphs**, correctly applied via `extra_llm_api_options` (`pytorch_backend_config.use_cuda_graph: true`) — see the verification note in the README for how a silent mis-config (CUDA graphs *off*) was caught with a memory-bandwidth roofline check and fixed (it had made TRT-LLM look ~2× slower). ITL is per-streamed-chunk (≈ per token).

## 1. Cross-model — vLLM, TP=1, BF16 (1× H100 each)

Llama-3.1-8B (2024) vs Qwen3-8B / Qwen3.5-9B (2026):

| model | c1 tok/s | c16 tok/s | c64 tok/s | c128 tok/s |
|---|---|---|---|---|
| Qwen3-8B | 145 | 2262 | 7862 | 13411 |
| Qwen3.5-9B | 127 | 1978 | 6234 | 9356 |
| Llama-3.1-8B | 152 | 2378 | 8315 | 13771 |

The 9B carries ~25% less throughput/H100 than the 8Bs — the capability-vs-cost trade, with numbers.

## 2. Head-to-head FP8 — Llama-3.1-8B, TP=2 (headline)

Same model & precision (FP8, `nvidia/Llama-3.1-8B-Instruct-FP8`), TRT-LLM's PyTorch backend + CUDA graphs (`--backend pytorch`, *not* a pre-compiled TRT engine — a compiled-engine comparison is future work) vs vLLM. **TRT-LLM wins the low/mid-concurrency (latency) regime; vLLM wins high concurrency (throughput).** This is the textbook split and only appears once CUDA graphs are correctly on.

**TensorRT-LLM+CG** (`trtllm_llama31_fp8`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 374.3 | 0.0241 | 0.0295 | 2.56 | 2.56 |
| 4 | 1361.5 | 0.0425 | 0.0484 | 2.77 | 2.8 |
| 16 | 4855.9 | 0.0573 | 0.1018 | 3.04 | 3.05 |
| 32 | 9256.2 | 0.0854 | 0.1175 | 3.09 | 3.23 |
| 64 | 13919.3 | 0.1527 | 0.9153 | 3.59 | 4.88 |
| 128 | 13802.5 | 0.2848 | 2.1801 | 7.82 | 9.25 |

**vLLM** (`vllm_llama31_fp8`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 299.6 | 0.0151 | 0.0165 | 2.98 | 2.99 |
| 4 | 1290.9 | 0.0215 | 0.0258 | 3.01 | 3.03 |
| 16 | 4707.5 | 0.041 | 0.0859 | 3.16 | 3.53 |
| 32 | 8808.9 | 0.0718 | 0.1037 | 3.33 | 3.45 |
| 64 | 15447.1 | 0.099 | 0.161 | 3.69 | 3.96 |
| 128 | 22782.6 | 0.213 | 0.3429 | 4.67 | 4.95 |

| concurrency | TRT-LLM+CG tok/s | vLLM tok/s | ratio | winner |
|---|---|---|---|---|
| 1 | 374 | 300 | 1.25× | TRT-LLM |
| 4 | 1362 | 1291 | 1.05× | TRT-LLM |
| 16 | 4856 | 4708 | 1.03× | TRT-LLM |
| 32 | 9256 | 8809 | 1.05× | TRT-LLM |
| 64 | 13919 | 15447 | 0.90× | vLLM |
| 128 | 13802 | 22783 | 0.61× | vLLM |

## 3. Head-to-head BF16 — Llama-3.1-8B, TP=2

Same as above, BF16. The two are ~tied at concurrency 1; vLLM's scheduler pulls ahead as the batch grows.

**TensorRT-LLM+CG** (`trtllm_llama31`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 230.2 | 0.0243 | 0.0254 | 4.26 | 4.27 |
| 4 | 892.8 | 0.0399 | 0.0487 | 4.33 | 4.36 |
| 16 | 3068.9 | 0.0683 | 0.1052 | 4.94 | 4.97 |
| 32 | 5748.5 | 0.0852 | 0.3679 | 5.08 | 5.2 |
| 64 | 10613.5 | 0.1361 | 0.193 | 5.45 | 5.54 |
| 128 | 14194.1 | 0.2831 | 1.3698 | 7.88 | 8.41 |

**vLLM** (`vllm_llama31`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 227.7 | 0.0159 | 0.0176 | 4.04 | 4.04 |
| 4 | 937.0 | 0.023 | 0.1103 | 4.14 | 4.5 |
| 16 | 3529.9 | 0.046 | 0.0791 | 4.34 | 4.37 |
| 32 | 6831.4 | 0.0526 | 0.084 | 4.45 | 4.48 |
| 64 | 12031.2 | 0.1075 | 0.1567 | 4.85 | 4.93 |
| 128 | 19658.7 | 0.2104 | 0.3598 | 5.55 | 5.92 |

| concurrency | TRT-LLM+CG tok/s | vLLM tok/s | ratio | winner |
|---|---|---|---|---|
| 1 | 230 | 228 | 1.01× | tie |
| 4 | 893 | 937 | 0.95× | vLLM |
| 16 | 3069 | 3530 | 0.87× | vLLM |
| 32 | 5748 | 6831 | 0.84× | vLLM |
| 64 | 10614 | 12031 | 0.88× | vLLM |
| 128 | 14194 | 19659 | 0.72× | vLLM |

## 4. Head-to-head BF16 — Qwen2.5-32B, TP=4 (big model, 4 cards)

32B tensor-parallel across 4× H100. Same crossover shape — competitive at low concurrency, vLLM ahead at saturation.

**TensorRT-LLM+CG** (`trtllm_qwen25_32b`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 114.1 | 0.0341 | 0.0355 | 8.66 | 8.66 |
| 4 | 449.0 | 0.0501 | 0.0679 | 8.73 | 8.75 |
| 16 | 1610.2 | 0.083 | 0.0979 | 9.61 | 9.72 |
| 32 | 2924.7 | 0.1188 | 0.5506 | 10.48 | 10.62 |
| 64 | 5326.1 | 0.1634 | 0.2192 | 11.33 | 11.64 |
| 128 | 5686.3 | 0.2798 | 0.6616 | 21.24 | 22.63 |

**vLLM** (`vllm_qwen25_32b`)

| concurrency | throughput_tok_s | ttft_p50_s | ttft_p99_s | itl_p50_ms | itl_p99_ms |
|---|---|---|---|---|---|
| 1 | 113.4 | 0.0216 | 0.0218 | 8.43 | 8.44 |
| 4 | 463.4 | 0.0329 | 0.0449 | 8.48 | 8.82 |
| 16 | 1738.4 | 0.0608 | 0.1205 | 8.95 | 8.98 |
| 32 | 3275.6 | 0.0809 | 0.1093 | 9.45 | 9.51 |
| 64 | 5826.0 | 0.1166 | 0.1747 | 10.48 | 10.55 |
| 128 | 9382.9 | 0.2128 | 0.2762 | 12.72 | 12.87 |

| concurrency | TRT-LLM+CG tok/s | vLLM tok/s | ratio | winner |
|---|---|---|---|---|
| 1 | 114 | 113 | 1.01× | tie |
| 4 | 449 | 463 | 0.97× | vLLM |
| 16 | 1610 | 1738 | 0.93× | vLLM |
| 32 | 2925 | 3276 | 0.89× | vLLM |
| 64 | 5326 | 5826 | 0.91× | vLLM |
| 128 | 5686 | 9383 | 0.61× | vLLM |

## 5. Quantization — vLLM, Qwen3-8B, TP=2 (FP8 vs BF16)

| concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup |
|---|---|---|---|
| 1 | 215 | 273 | 1.27× |
| 4 | 885 | 1135 | 1.28× |
| 16 | 3353 | 4321 | 1.29× |
| 32 | 6189 | 7841 | 1.27× |
| 64 | 11526 | 13792 | 1.20× |
| 128 | 18585 | 20794 | 1.12× |

FP8 wins most at low concurrency (memory-bandwidth-bound decode).

