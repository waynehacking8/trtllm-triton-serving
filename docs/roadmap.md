# Roadmap

The phases below double as the list of interview talking points that have empirical
evidence in this repo. Unchecked = not yet run/verified.

## Phase 1 — Single-model TP=4 serving
- [ ] Build Llama-3.1-8B TRT-LLM engine, TP=4, FP16; load in Triton; smoke test.
- [ ] Async benchmark client: TTFT, throughput (tok/s), inter-token latency (ITL).
- [ ] vLLM baseline (TP=4) under identical prompts/concurrency.

## Phase 2 — Quantization + batching
- [ ] FP8 engine; measure quality (a small eval set) vs FP16 and the throughput delta.
- [ ] In-flight batching on; sweep max_batch_size / max_num_tokens.
- [ ] Paged KV-cache free-fraction sweep; document the latency/throughput Pareto front.

## Phase 3 — Scale up
- [ ] 70B-class model, TP=4 (and TP=2/PP=2 comparison).
- [ ] genai-perf cross-check of the custom harness numbers.

## Phase 4 — Reference architecture
- [ ] One-command deploy (compose); a short "how an SA would hand this to a partner" guide.

## Out of scope (for now)
- Multi-node (NCCL over InfiniBand) — see the `nccl-collectives-bench` repo first.
