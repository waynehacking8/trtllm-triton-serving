# Roadmap

The phases below double as the list of key technical claims that have empirical
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

## Phase 5 — Follow-ups to the measured studies (specified)

These are the "Remaining (roadmap)" items referenced in README Status, each with the question it
answers, the exact method, and how to read the result.

- [x] **Tuned-vs-tuned re-run** (TRT-LLM tuned vs vLLM defaults). **DONE — README study 7 /
  report study 6.** Result: c128 throughput unchanged (13,803 → 13,828; 0.2%); TTFT p99
  improves 25% (2.18s → 1.64s). The deficit is NOT the defaults. Note the roadmap's proposed
  YAML nesting was wrong — `enable_chunked_prefill` and `scheduler_config` are top-level
  LlmArgs keys (verified against the installed wheel), not `pytorch_backend_config` children;
  the correct config is `configs/trtllm_pytorch_tuned.yaml`.
  - **Question:** at c128, default-config TRT-LLM trails vLLM by 65% in throughput with
    TTFT p99 = 2.18 s — is that the out-of-box defaults (`GUARANTEED_NO_EVICT` scheduler +
    chunked prefill off) or an inherent engine limit? SqueezeBits' controlled study found tuned
    TRT-LLM *wins* at large batch — this run decides the final form of this repo's headline.
  - **Method:** new config `configs/trtllm_pytorch_tuned.yaml` = the existing CUDA-graph config
    plus two switches:
    ```yaml
    pytorch_backend_config:
      use_cuda_graph: true
      cuda_graph_padding_enabled: true
      cuda_graph_max_batch_size: 256
      enable_chunked_prefill: true        # default off (GitHub issue #4947)
      scheduler_policy: MAX_UTILIZATION   # default GUARANTEED_NO_EVICT
    ```
    Serve with `CFG=/models/trtllm_pytorch_tuned.yaml bash scripts/serve_trtllm.sh
    /models/Llama-3.1-8B-Instruct-FP8 2 8012`, **verify the settings actually took**
    (`docker logs trtllm_serve | grep -iE 'chunked|scheduler|use_cuda_graph'` — this repo's
    hard-won lesson: 0.20 silently ignores mis-nested keys), then
    `bash bench/sweep.sh http://localhost:8012 trtllm_llama31_fp8_tuned` and
    `python bench/pareto.py && python bench/roofline_check.py`.
  - **Read-out:** tuned TRT-LLM c128 ≥ vLLM (22,783 tok/s) → the deficit was defaults; the
    headline becomes "tuned engines are comparable; the difference is robustness of defaults".
    Still clearly behind → defaults are not the main cause; profile the scheduler. Either way,
    whether TTFT p99 @c128 drops from 2.18 s directly tests the scheduler/chunked-prefill
    hypothesis.

- [x] **Compiled engine vs PyTorch backend** head-to-head. **DONE — README study 8 / report
  study 7.** Result: compiled engine ≈ PyTorch backend (±5% at every concurrency); both trail
  vLLM ~25% at c128 → further evidence the gap is engine-runtime, not kernels. CUDA graphs add
  only ~6% to the compiled engine (vs 2.3× for the PyTorch backend). Triton `tensorrt_llm`
  backend also deployed (`scripts/setup_triton_repo.sh`, ensemble + BLS, TP=2 via MPI) and
  measured: ~187 tok/s c1 through the ensemble (~15% frontend overhead vs trtllm-serve).
  - **Question:** all measured TRT-LLM numbers use the PyTorch backend. How much does the
    pre-compiled engine add — enough to change the c128 conclusion?
  - **Method:** `bash scripts/build_engine.sh` → `bash scripts/serve_triton.sh` (this also
    completes the Triton `tensorrt_llm`-backend deployment item) →
    `bash bench/sweep.sh http://localhost:8000 trtllm_compiled`. Constraint: TRT-LLM 0.20's
    compiled path supports Llama-3.x / Qwen2.x only (not Qwen3).
  - **Read-out:** the compiled-vs-PyTorch delta at c1 and c128. If compiled still loses to vLLM
    at c128 → further evidence the gap is scheduler/defaults, not kernel efficiency.

- [x] **Speculative decoding under concurrency.** **DONE — README study 9 / report study 8.**
  Result: speedup decays 3.5× (c1) → 1.18× (c128) while draft acceptance stays flat ~97% —
  the decay is the memory-bound→compute-bound transition, not draft quality. No <1.0×
  crossover up to c128; extrapolated break-even ≈ c256. Guidance: enable below ~c32.
  - **Question:** the 2.82× extractive speedup is batch=1. Where does the benefit reach zero as
    concurrency rises, and where does it become a net loss?
  - **Method:** serve vLLM with the n-gram speculative config; run the existing sweep against the
    extractive task set (`bash bench/sweep.sh <base> spec_extractive`, c1→c128).
  - **Read-out:** plot speedup vs concurrency; find the ≤1.0× crossover → complete usage
    guidance: "enable spec decode for RAG-style workloads below concurrency X, disable above".

- [x] **Triton ensemble path: full concurrency sweep** (currently c1 smoke only, ~187 tok/s).
  **DONE — README study 12 / report section 10 / `bench/bench_triton.py` /
  `results/trtllm_triton_bf16-c*.json` / `results/pareto_triton.png`.** Result: the roadmap's
  either/or question gets a both-with-a-boundary answer. **c1–c32: the Python hop fully
  amortizes (0±1% vs the same executor through trtllm-serve)** — and this revises study 8's
  smoke estimate: the "~15% c1 overhead" was trtllm-serve's CUDA-graph advantage, not ensemble
  cost. **c64+: the hop serializes catastrophically** — at c128 the ensemble regresses in
  absolute terms (8,710 → 5,769 tok/s) while the engine underneath scales to 14.7k; TTFT p99
  6.0 s with ITL still 8.9 ms = queueing at single-instance Python preprocessing
  (`preprocessing_instance_count: 1`), not in the engine. Guidance: ensemble OK ≤c32;
  throughput serving needs more preprocessing instances, BLS, or trtllm-serve.
  - **Question:** the ~15% ensemble overhead vs `trtllm-serve` was measured at c1 only. Under
    concurrency, does the Python pre/post-processing hop amortize (fixed cost spread over more
    requests) or become the bottleneck (GIL / single-instance preprocessing serializes)?
  - **Method:** serve via `scripts/serve_triton.sh`; run the existing sweep against the Triton
    OpenAI-compatible frontend (or genai-perf against the ensemble endpoint) for c1→c128, tag
    `trtllm_triton_bf16`; commit the JSONs (closes the smoke-only caveat noted in README
    study 8).
  - **Read-out:** overhead vs `trtllm-serve` at each concurrency. Amortizes to <5% by c32 →
    the ensemble path is fine for throughput serving, only latency-sensitive paths need BLS /
    the OpenAI frontend. Grows with concurrency → quantifies why production deployments bypass
    the Python ensemble.

- [x] **bench.py: write the model field into output JSON.** **DONE.** bench.py now writes
  `model` into every result JSON; sweep.sh passes the served model name through. All new-run
  JSONs (tuned, compiled, spec-concurrency) are self-describing; pre-existing JSONs keep their
  filename-based attribution (re-running ~3 GPU-hours of sweeps to add a metadata field that
  the README mapping already documents was judged not worth the burn — flagged here honestly).
  - **Question:** (data traceability, not hypothesis testing) 60 of 61 raw JSONs lack a model
    field; attribution currently relies on filenames.
  - **Method:** one-line change in `bench/bench.py` to write `--model` into the output JSON;
    re-run `bash bench/sweep.sh` for all tags.
  - **Read-out:** every JSON becomes self-describing.

## Phase 6 — Literature-ceiling reproductions (specified)

Goal: turn published serving-performance claims into measured, attributed results from this
repo's own harness and hardware.

- [x] **NVIDIA perf-overview 26,401 tok/s: per-knob waterfall attribution.**
  Published target: Llama-3.1-8B FP8, 1×H100 SXM, ISL/OSL 128/128, TP=1, kv_cache_free_gpu_mem_fraction=0.95.
  **DONE — README study 11 / `scripts/waterfall.sh` / `results/waterfall/` / `results/waterfall.png`.**
  Result: **published number reproduced at 100.3%** (27,785 vs 27,688, the release/0.20 docs
  figure matching the installed version). Attribution: workload-shape knobs are *favorable*
  (+66% offline); the entire serving deficit is **concurrency cap (−50%)** + **serving stack
  (−48%)**, partially recovered by TP2 (+16%) → the committed 13,828. The "53% of published"
  framing was apples-to-oranges by construction: the published figure is offline
  unbounded-batching, not serving. Two 0.20 tool findings documented with kept logs: NVIDIA's
  exact YAML OOMs on 80GB (needs max_batch_size 2048), and trtllm-bench --tp 2 crashes at init
  (TP measured on the serving side instead).
  - **Question:** this repo's best TRT-LLM number (13.8k tok/s, c128, 256-token decode, TP=2,
    kv 0.8) is 53% of NVIDIA's published 26,401. How much of that gap is methodology
    (ISL/OSL 128/128 vs our decode-256, TP=1 vs TP=2, kv 0.95 vs 0.8, prefill-heavy vs
    decode-heavy) and how much is real? Community report (TRT-LLM issue #6294) only reached
    7,099 on PCIe H100 — so the published number is not trivially reproducible.
  - **Method:** on 1 idle H100: first replicate NVIDIA's exact config (trtllm-bench or
    genai-perf, 128/128 synthetic, TP=1, kv 0.95, large CUDA-graph batch list) and try to hit
    ~26k. Then change ONE knob at a time toward this repo's methodology: OSL 128→256 →
    kv 0.95→0.8 → TP 1→2 → real-prompt workload. Record tok/s after each step.
  - **Read-out:** a waterfall chart decomposing 26,401 → (our methodology's number). Each bar
    is an attributed loss. This converts "we only reach 53% of the published ceiling" into a
    reproducibility analysis of what the published ceiling actually measures.

- [x] **NVFP4 W4A4 serving on RTX PRO 6000 (sm_120). DONE — README study 10 / report study 9.**
  Published target: ~1.77–2.1× over BF16 at high concurrency, accuracy parity on GPQA/ARC
  (NVIDIA NVFP4 / Jarvis Labs).
  - **Question:** all serving numbers in this repo are Hopper. Does NVFP4 quantized serving
    deliver the published speedup on a Blackwell workstation card — the first sm_120 data
    point in this repo?
  - **Method (as run):** Llama-3.1-8B-Instruct → NVFP4 W4A4 via TensorRT-Model-Optimizer PTQ
    (`scripts/quantize_nvfp4.py`, 512-sample cnn_dailymail calibration); served with vLLM 0.18 +
    flashinfer 0.6.6 on the RTX PRO 6000 (`scripts/serve_vllm_sm120.sh`, TP=1; no-compile +
    decode-CUDA-graph workaround documented in `results/sm120_environment.txt`); existing sweep
    (c1→c128, 256-token decode) against BF16 and FP8 baselines on the same card; ARC-Challenge
    300-question spot-check (`bench/accuracy_mc.py`; GPQA is HF-gated, roadmap allowed either).
  - **Result: the published speedup reproduces — and is exceeded at high concurrency.**
    NVFP4/BF16 = 1.61× (c1) → 1.74× (c16) → 1.93× (c32) → **2.52× (c64) → 2.13× (c128)** vs the
    published ~1.77–2.1×. Crucially, vLLM selected the **native FP4 path**
    (`NvFp4LinearBackend.FLASHINFER_CUTLASS`, logged in `results/sm120_environment.txt`) — not
    the Marlin dequant fallback — so this is real 5th-gen Tensor Core FP4 math. Two findings on
    top of the headline: (a) **below c16, FP8 beats NVFP4** (0.92–0.94×; activation-quant
    overhead not yet amortized) — precision choice is concurrency-dependent; (b) accuracy
    spot-check: BF16 0.830 / FP8 0.847 / **NVFP4 0.800** (−3.0 pp, ~1.4σ at n=300) — consistent
    with published near-parity but a real downward trend, flagged for a full-eval follow-up.
    Raw data: `results/vllm_sm120_{bf16,fp8,nvfp4}-c*.json`, `results/acc_arc_sm120_*.json`,
    chart `results/pareto_sm120.png`.

- [ ] **Candidates (spec on demand):** EAGLE-3 speculative decoding (published +38% at batch=64,
  arXiv:2503.01840 — directly extends study 9's n-gram curve, which decays to 1.18× there);
  FP8 KV-cache (published ~1.45× decode-heavy); SGLang as a third engine (published ~29% over
  vLLM on shared-prefix workloads).

## Out of scope (for now)
- Multi-node (NCCL over InfiniBand) — see the `nccl-collectives-bench` repo first.
