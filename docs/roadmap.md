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

## Out of scope (for now)
- Multi-node (NCCL over InfiniBand) — see the `nccl-collectives-bench` repo first.
