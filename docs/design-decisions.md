# Design decisions

**D1 — Tensor parallel = 4, not pipeline parallel.** On a single NVLink box, TP keeps
all GPUs busy per token and avoids pipeline bubbles. PP is only worth it across slow
links (multi-node). We measure TP=4 vs TP=2/PP=2 in Phase 3 to make this concrete.

**D2 — Triton tensorrt_llm backend, not bare trtllm-serve.** Triton gives the
in-flight batcher, metrics, and an ensemble (pre/post) that mirrors how partners deploy.
`trtllm-serve` is simpler but hides the pieces an SA needs to explain.

**D3 — FP8 before INT4.** On Hopper, FP8 (E4M3) hits Tensor Cores natively with small
quality loss; INT4 (AWQ/GPTQ) trades more quality for memory and is a Phase-2+ option.

**D4 — Honest baseline.** vLLM is run with matched TP, dtype, and max context so the
comparison isn't rigged. Where vLLM wins (often ergonomics / TTFT at low concurrency),
the report says so.

**D5 — Reproducibility.** Fixed container tag, fixed seeds for the eval set, and the
exact build/serve commands are scripted — no hand-run steps that don't show up in git.
