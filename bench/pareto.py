#!/usr/bin/env python3
"""Build serving reports from concurrency sweeps. Studies (all TRT-LLM runs use the
correctly-applied CUDA-graph config — see README "verification" note):

  1. Cross-model (vLLM TP=1)            — Llama-3.1-8B / Qwen3-8B / Qwen3.5-9B
  2. Head-to-head FP8  (Llama-3.1-8B TP=2)   — TensorRT-LLM+CUDA-graph vs vLLM  [headline]
  3. Head-to-head BF16 (Llama-3.1-8B TP=2)   — TensorRT-LLM+CUDA-graph vs vLLM
  4. Head-to-head BF16 (Qwen2.5-32B TP=4)    — TensorRT-LLM+CUDA-graph vs vLLM
  5. Quantization (Qwen3-8B vLLM TP=2)       — FP8 vs BF16
  6. Tuned-vs-tuned (Llama-3.1-8B FP8 TP=2)  — TRT-LLM defaults vs +chunked-prefill+MAX_UTILIZATION
  7. Compiled engine vs PyTorch backend (Llama-3.1-8B BF16 TP=2)
  9. NVFP4 W4A4 vs BF16/FP8 (Llama-3.1-8B, vLLM TP=1, RTX PRO 6000 sm_120)  [first non-Hopper data]

All requests decode exactly 256 tokens (ignore_eos). Consumes results/<tag>-c<N>.json.
"""
import glob, json, os, pathlib, re
from collections import defaultdict

_REPO = pathlib.Path(__file__).resolve().parent.parent

LABEL = {
    "xm_qwen3_8b": "Qwen3-8B", "xm_qwen35_9b": "Qwen3.5-9B", "xm_llama31_8b": "Llama-3.1-8B",
    "trtllm_llama31": "TensorRT-LLM+CG", "vllm_llama31": "vLLM",
    "trtllm_llama31_fp8": "TensorRT-LLM+CG", "vllm_llama31_fp8": "vLLM",
    "trtllm_qwen25_32b": "TensorRT-LLM+CG", "vllm_qwen25_32b": "vLLM",
    "vllm_bf16": "vLLM BF16", "vllm_fp8": "vLLM FP8",
    "trtllm_llama31_fp8_tuned": "TRT-LLM tuned (chunked prefill + MAX_UTIL)",
    "trtllm_compiled_bf16": "TRT-LLM compiled engine",
    "trtllm_compiled_bf16_cg": "TRT-LLM compiled engine + CUDA graphs",
    "vllm_sm120_bf16": "vLLM BF16 (sm_120)", "vllm_sm120_fp8": "vLLM FP8 (sm_120)",
    "vllm_sm120_nvfp4": "vLLM NVFP4 W4A4 (sm_120)",
    "trtllm_triton_bf16": "Triton ensemble (tensorrt_llm backend)",
}
GROUP_A = ["xm_qwen3_8b", "xm_qwen35_9b", "xm_llama31_8b"]
GROUP_B = ["trtllm_llama31", "vllm_llama31"]
GROUP_FP8 = ["trtllm_llama31_fp8", "vllm_llama31_fp8"]
GROUP_D = ["trtllm_qwen25_32b", "vllm_qwen25_32b"]
GROUP_C = ["vllm_bf16", "vllm_fp8"]
GROUP_TUNED = ["trtllm_llama31_fp8", "trtllm_llama31_fp8_tuned", "vllm_llama31_fp8"]
GROUP_ENGINE = ["trtllm_llama31", "trtllm_compiled_bf16", "trtllm_compiled_bf16_cg", "vllm_llama31"]
GROUP_SM120 = ["vllm_sm120_bf16", "vllm_sm120_fp8", "vllm_sm120_nvfp4"]
# Triton ensemble vs the same compiled engine through trtllm-serve (study 12)
GROUP_TRITON = ["trtllm_triton_bf16", "trtllm_compiled_bf16", "trtllm_compiled_bf16_cg"]
# accuracy spot-check JSONs (bench/accuracy_mc.py) paired with the sm_120 sweep tags
ACC_SM120 = {"vllm_sm120_bf16": "results/acc_arc_sm120_bf16.json",
             "vllm_sm120_fp8": "results/acc_arc_sm120_fp8.json",
             "vllm_sm120_nvfp4": "results/acc_arc_sm120_nvfp4.json"}


def load_sweeps():
    runs = defaultdict(list)
    for f in glob.glob(str(_REPO / "results" / "*-c*.json")):
        m = re.match(r"(.+)-c(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        tag, c = m.group(1), int(m.group(2))
        with open(f) as fh:
            d = {**json.load(fh), "concurrency": c}
        runs[tag].append(d)
    for t in runs:
        runs[t].sort(key=lambda r: r["concurrency"])
    return runs


def sla_throughput(rows, sla_s):
    ok = [r for r in rows if r["ttft_p99_s"] <= sla_s]
    return max((r["throughput_tok_s"] for r in ok), default=0.0)


def sweep_table(w, runs, tags):
    cols = ["concurrency", "throughput_tok_s", "ttft_p50_s", "ttft_p99_s", "itl_p50_ms", "itl_p99_ms"]
    for tag in tags:
        if tag not in runs:
            continue
        w(f"**{LABEL.get(tag, tag)}** (`{tag}`)\n")
        w("| " + " | ".join(cols) + " |")
        w("|" + "---|" * len(cols))
        for r in runs[tag]:
            w("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        w("")


def h2h(w, runs, trt, vl, title, blurb):
    w(f"## {title}\n"); w(blurb + "\n")
    sweep_table(w, runs, [trt, vl])
    if trt in runs and vl in runs:
        tr = {r["concurrency"]: r for r in runs[trt]}
        vv = {r["concurrency"]: r for r in runs[vl]}
        w("| concurrency | TRT-LLM+CG tok/s | vLLM tok/s | ratio | winner |")
        w("|---|---|---|---|---|")
        for c in sorted(set(tr) & set(vv)):
            a, b = tr[c]["throughput_tok_s"], vv[c]["throughput_tok_s"]
            r = a / b if b else 0
            win = "TRT-LLM" if r > 1.02 else ("vLLM" if r < 0.98 else "tie")
            n_note = " (n=8, interpret with caution)" if c == 1 else ""
            w(f"| {c} | {a:.0f} | {b:.0f} | {r:.2f}×{n_note} | {win} |")
        w("")
        w("*Note: c=1 rows are based on n=8 requests — p99 latency at this sample size has "
          "high variance and should be interpreted with caution.*\n")
        w("*Note: vLLM was run with default serve settings while TRT-LLM had explicit config "
          "tuning (`extra_llm_api_options`); the low-concurrency TRT-LLM advantage could narrow "
          "under a comparably tuned vLLM config.*\n")


def main():
    runs = load_sweeps()
    if not runs:
        print("no sweep results"); return
    L = []; w = L.append
    w("# Serving benchmark — TensorRT-LLM vs vLLM, cross-model & quantization (H100)\n")
    w("OpenAI-compatible servers, fixed prompt, **every request decodes exactly 256 tokens** "
      "(`ignore_eos`+`min_tokens`) so throughput/latency compare the same work. **All TRT-LLM "
      "runs use CUDA graphs**, correctly applied via `extra_llm_api_options` "
      "(`pytorch_backend_config.use_cuda_graph: true`) — see the verification note in the "
      "README for how a silent mis-config (CUDA graphs *off*) was caught with a "
      "memory-bandwidth roofline check and fixed (it had made TRT-LLM look ~2× slower). "
      "ITL is per-streamed-chunk (≈ per token).\n")

    if any(t in runs for t in GROUP_A):
        w("## 1. Cross-model — vLLM, TP=1, BF16 (1× H100 each)\n")
        w("Llama-3.1-8B (2024) vs Qwen3-8B / Qwen3.5-9B (2026):\n")
        present = [t for t in GROUP_A if t in runs]
        cs = sorted({r["concurrency"] for t in present for r in runs[t]})
        w("| model | " + " | ".join(f"c{c} tok/s" for c in cs) + " |")
        w("|---|" + "---|" * len(cs))
        for t in present:
            byc = {r["concurrency"]: r["throughput_tok_s"] for r in runs[t]}
            w(f"| {LABEL[t]} | " + " | ".join(f"{byc.get(c,0):.0f}" for c in cs) + " |")
        w("\nThe 9B carries ~30% less throughput/H100 than the 8Bs (9,356 vs 13,411/13,771 "
          "@c128) — the capability-vs-cost trade, with numbers.\n")

    h2h(w, runs, "trtllm_llama31_fp8", "vllm_llama31_fp8",
        "2. Head-to-head FP8 — Llama-3.1-8B, TP=2 (headline)",
        "Same model & precision (FP8, `nvidia/Llama-3.1-8B-Instruct-FP8`), TRT-LLM's PyTorch "
        "backend + CUDA graphs (`--backend pytorch`) vs vLLM. **TRT-LLM wins the "
        "low/mid-concurrency (latency) regime; vLLM wins high concurrency (throughput).** "
        "This only appears once CUDA graphs are correctly on. The earlier caveat — that the "
        "c128 deficit might just be `trtllm-serve` defaults (`GUARANTEED_NO_EVICT` scheduler, "
        "chunked prefill off, GitHub issue #4947) — has now been **tested and rejected**: "
        "study 6 below re-runs with chunked prefill + `MAX_UTILIZATION` and the c128 "
        "throughput does not move (13.8k both ways), and study 7 shows the compiled engine "
        "lands in the same place. The gap at high concurrency is engine-runtime-level in "
        "TRT-LLM 0.20 for this decode-heavy workload, not a configuration artifact. "
        "(Published comparisons vary — SqueezeBits found tuned TRT-LLM winning at large batch "
        "on older versions/different workloads; BentoML found TTFT collapse at 100 users — "
        "which is exactly why this repo measures rather than quotes.)")

    h2h(w, runs, "trtllm_llama31", "vllm_llama31",
        "3. Head-to-head BF16 — Llama-3.1-8B, TP=2",
        "Same as above, BF16. The two are ~tied at concurrency 1; vLLM's scheduler pulls "
        "ahead as the batch grows.")

    h2h(w, runs, "trtllm_qwen25_32b", "vllm_qwen25_32b",
        "4. Head-to-head BF16 — Qwen2.5-32B, TP=4 (big model, 4 cards)",
        "32B tensor-parallel across 4× H100. Same crossover shape — competitive at low "
        "concurrency, vLLM ahead at saturation.")

    if all(t in runs for t in GROUP_C):
        w("## 5. Quantization — vLLM, Qwen3-8B, TP=2 (FP8 vs BF16)\n")
        bf = {r["concurrency"]: r for r in runs["vllm_bf16"]}
        fp = {r["concurrency"]: r for r in runs["vllm_fp8"]}
        w("| concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup |")
        w("|---|---|---|---|")
        for c in sorted(set(bf) & set(fp)):
            b, f = bf[c]["throughput_tok_s"], fp[c]["throughput_tok_s"]
            w(f"| {c} | {b:.0f} | {f:.0f} | {f/b:.2f}× |")
        w("\nFP8 wins most at low concurrency (memory-bandwidth-bound decode).\n")

    if "trtllm_llama31_fp8_tuned" in runs:
        w("## 6. Tuned-vs-tuned — does TRT-LLM's c128 deficit come from its defaults?\n")
        w("Same FP8 serve command as study 2, plus `enable_chunked_prefill: true` and "
          "`scheduler_config.capacity_scheduler_policy: MAX_UTILIZATION` "
          "(`configs/trtllm_pytorch_tuned.yaml`; key nesting verified against the installed "
          "0.20 wheel — both are TOP-LEVEL LlmArgs keys, *not* `pytorch_backend_config` "
          "children as some docs suggest, which would be silently ignored).\n")
        base = {r["concurrency"]: r for r in runs.get("trtllm_llama31_fp8", [])}
        tuned = {r["concurrency"]: r for r in runs["trtllm_llama31_fp8_tuned"]}
        vl = {r["concurrency"]: r for r in runs.get("vllm_llama31_fp8", [])}
        w("| concurrency | TRT defaults tok/s | TRT tuned tok/s | TRT defaults TTFT p99 | TRT tuned TTFT p99 | vLLM tok/s |")
        w("|---|---|---|---|---|---|")
        for c in sorted(tuned):
            b, t = base.get(c, {}), tuned[c]
            v = vl.get(c, {})
            w(f"| {c} | {b.get('throughput_tok_s', '—')} | {t['throughput_tok_s']} | "
              f"{b.get('ttft_p99_s', '—')}s | {t['ttft_p99_s']}s | {v.get('throughput_tok_s', '—')} |")
        w("")
        w("**Read-out: throughput is unchanged** (c128: 13,803 default vs 13,828 tuned — 0.2%); "
          "the c64 saturation ceiling is identical. **TTFT p99 at c128 improves 25%** "
          "(2.18s → 1.64s) — chunked prefill does what it promises for admission latency — "
          "but the throughput gap to vLLM (22.8k) is *not* a scheduler/defaults artifact. "
          "Combined with study 7 (compiled engine, same ceiling), the deficit is in the "
          "engine runtime itself for this workload on 0.20.\n")

    if any(t in runs for t in ("trtllm_compiled_bf16", "trtllm_compiled_bf16_cg")):
        w("## 7. Compiled TRT engine vs PyTorch backend — BF16 Llama-3.1-8B TP=2\n")
        w("`trtllm-build` engine (bfloat16, TP=2, `--use_paged_context_fmha enable`, "
          "`scripts/build_engine.sh`) served through the same `trtllm-serve` OpenAI frontend "
          "as the PyTorch-backend runs — only the executor differs. The +CG variant adds "
          "`extended_runtime_perf_knob_config.cuda_graph_mode: true` "
          "(`configs/trtllm_engine_cudagraph.yaml`).\n")
        cols = {"trtllm_llama31": "PyTorch backend + CG",
                "trtllm_compiled_bf16": "compiled engine",
                "trtllm_compiled_bf16_cg": "compiled engine + CG",
                "vllm_llama31": "vLLM"}
        present = [t for t in cols if t in runs]
        cs = sorted({r["concurrency"] for t in present for r in runs[t]})
        w("| concurrency | " + " | ".join(cols[t] for t in present) + " |")
        w("|---|" + "---|" * len(present))
        for c in cs:
            vals = []
            for t in present:
                byc = {r["concurrency"]: r["throughput_tok_s"] for r in runs[t]}
                vals.append(f"{byc.get(c, 0):.0f}")
            w(f"| {c} | " + " | ".join(vals) + " |")
        w("")
        w("**Read-out: the compiled engine and the PyTorch backend land within ~5% of each "
          "other at every concurrency** (c1: 220 vs 230; c128: 14.8k vs 14.2k) — and both "
          "still trail vLLM by ~25% at c128. Two further observations: (1) CUDA graphs add "
          "only ~6% to the compiled engine at c1 (TRT already fuses kernels at build time) "
          "versus the 2.3× they added to the PyTorch backend (162→374 FP8) — the lever moves "
          "to wherever launch overhead lives. (2) The same engine served through the Triton "
          "`tensorrt_llm` backend's ensemble path measures ~187 tok/s at c1 (~15% below "
          "trtllm-serve) — the ensemble's Python pre/post-processing hop; see "
          "`scripts/setup_triton_repo.sh`.\n")

    if os.path.exists("results/spec_concurrency.json"):
        with open(str(_REPO / "results" / "spec_concurrency.json")) as fh:
            sc = json.load(fh)
        w("## 8. Speculative decoding under concurrency — where does the benefit end?\n")
        w(f"n-gram (prompt-lookup) speculative decoding, {sc['model']}, extractive/RAG-style "
          "task, non-streaming (see `bench/spec_concurrency.py`). The batch=1 study showed "
          "2.8–3.5×; this study finds where the speedup dies as concurrency rises:\n")
        w("| concurrency | baseline tok/s | ngram tok/s | speedup | draft acceptance |")
        w("|---|---|---|---|---|")
        for r in sc["rows"]:
            w(f"| {r['concurrency']} | {r['baseline_tok_s']} | {r['ngram_tok_s']} | "
              f"**{r['speedup']:.2f}×** | {r['draft_acceptance']:.0%} |")
        cross = sc.get("crossover_concurrency")
        w("")
        w(f"**Read-out: the speedup decays monotonically (3.5× → 1.18×) while draft acceptance "
          "stays ~97% flat** — so the decay is *not* the draft getting worse; it is the "
          "compute-bound transition predicted by the spec-decode literature (Nightjar, "
          "arXiv:2512.22420; vLLM docs): at small batch the GPU is memory-bound and "
          "verification is free, at large batch every verified-then-rejected token competes "
          "with other requests for compute. "
          + (f"Crossover below 1.0× observed at c={cross}." if cross else
             "No <1.0× crossover up to c=128 on this task; extrapolating the decay puts it "
             "near c≈256.")
          + " Deployment guidance: enable n-gram spec decode for RAG-style/extractive "
          "workloads when per-replica concurrency stays below ~32 (≥2× speedup); it is "
          "merely neutral by c≈128.\n")
        w("![spec decode vs concurrency](spec_concurrency.png)\n")

    if any(t in runs for t in GROUP_SM120):
        w("## 9. NVFP4 W4A4 vs BF16/FP8 — Llama-3.1-8B, vLLM TP=1, RTX PRO 6000 Blackwell (sm_120)\n")
        w("The repo's first non-Hopper data point (roadmap Phase 6 literature-ceiling item). "
          "Llama-3.1-8B-Instruct quantized to **NVFP4 (W4A4)** with TensorRT-Model-Optimizer "
          "(`scripts/quantize_nvfp4.py`), served by vLLM on a single RTX PRO 6000 Blackwell "
          "Max-Q, vs the BF16 and FP8 (on-the-fly) baselines on the same card "
          "(`scripts/serve_vllm_sm120.sh` — compilation off / full decode CUDA graphs, the "
          "documented sm_120 workaround, identical for every precision). "
          "**Published target being tested: ~1.77-2.1x over BF16 at high concurrency** "
          "(NVIDIA NVFP4 blog / Jarvis Labs, measured on B200/RTX PRO with native FP4 kernels).\n")
        sweep_table(w, runs, GROUP_SM120)
        bf = {r["concurrency"]: r for r in runs.get("vllm_sm120_bf16", [])}
        fp = {r["concurrency"]: r for r in runs.get("vllm_sm120_fp8", [])}
        nv = {r["concurrency"]: r for r in runs.get("vllm_sm120_nvfp4", [])}
        if bf and nv:
            w("| concurrency | BF16 tok/s | FP8 tok/s | NVFP4 tok/s | NVFP4/BF16 | published NVFP4/BF16 |")
            w("|---|---|---|---|---|---|")
            for c in sorted(set(bf) & set(nv)):
                b, n = bf[c]["throughput_tok_s"], nv[c]["throughput_tok_s"]
                f = fp.get(c, {}).get("throughput_tok_s")
                w(f"| {c} | {b:.0f} | {f:.0f} | {n:.0f} | **{n/b:.2f}x** | ~1.77x |"
                  if f is not None else
                  f"| {c} | {b:.0f} | — | {n:.0f} | **{n/b:.2f}x** | ~1.77x |")
            w("")
        # accuracy spot-check table (bench/accuracy_mc.py results)
        accs = {}
        for t, p in ACC_SM120.items():
            if os.path.exists(p):
                with open(p) as fh:
                    accs[t] = json.load(fh)
        if accs:
            w("**Accuracy spot-check** (ARC-Challenge subset, generation-based MC via the "
              "serving endpoint — only the delta between precisions is meaningful):\n")
            w("| precision | ARC-Challenge accuracy | n | delta vs BF16 |")
            w("|---|---|---|---|")
            base_acc = accs.get("vllm_sm120_bf16", {}).get("accuracy")
            for t in GROUP_SM120:
                if t in accs:
                    a = accs[t]
                    d = (f"{a['accuracy'] - base_acc:+.4f}"
                         if base_acc is not None and t != "vllm_sm120_bf16" else "—")
                    w(f"| {LABEL[t]} | {a['accuracy']:.4f} | {a['n']} | {d} |")
            w("")

    if "trtllm_triton_bf16" in runs:
        w("## 10. Triton ensemble path under concurrency — where the Python hop stops being free\n")
        w("The same compiled BF16 engine as section 7, deployed behind Triton's `tensorrt_llm` "
          "backend (ensemble: preprocessing -> tensorrt_llm -> postprocessing, "
          "`scripts/setup_triton_repo.sh`), swept c1->c128 with `bench/bench_triton.py` "
          "(Triton generate_stream protocol, same forced-256-token methodology as every other "
          "sweep). Baselines: the same engine through `trtllm-serve` (section 7).\n")
        cols = {"trtllm_triton_bf16": "Triton ensemble",
                "trtllm_compiled_bf16": "trtllm-serve (no CG)",
                "trtllm_compiled_bf16_cg": "trtllm-serve + CG"}
        present = [t for t in cols if t in runs]
        cs = sorted({r["concurrency"] for t in present for r in runs[t]})
        w("| concurrency | " + " | ".join(cols[t] for t in present) + " | ensemble vs no-CG serve |")
        w("|---|" + "---|" * (len(present) + 1))
        tri = {r["concurrency"]: r["throughput_tok_s"] for r in runs["trtllm_triton_bf16"]}
        nocg = {r["concurrency"]: r["throughput_tok_s"] for r in runs.get("trtllm_compiled_bf16", [])}
        for c in cs:
            vals = []
            for t in present:
                byc = {r["concurrency"]: r["throughput_tok_s"] for r in runs[t]}
                vals.append(f"{byc[c]:,.0f}" if c in byc else "—")
            delta = (f"{(tri[c] / nocg[c] - 1) * 100:+.1f}%" if c in tri and c in nocg else "—")
            w(f"| {c} | " + " | ".join(vals) + f" | {delta} |")
        w("")
        w("**Read-out: the ensemble's Python pre/post hop is FREE until c32 (0±1% vs the "
          "identical executor through trtllm-serve), then becomes THE bottleneck** — at c64 it "
          "costs ~10%, and at c128 the ensemble *regresses in absolute terms* (its throughput "
          "falls while the engine underneath keeps scaling) with TTFT p99 exploding to ~6 s "
          "while ITL stays <9 ms: requests queue at the single-instance Python preprocessing "
          "stage (`preprocessing_instance_count: 1`), not in the engine. This also revises "
          "study 8's c1 smoke estimate: against the same executor without CUDA graphs, the "
          "c1 ensemble overhead is ~0%, not ~15% — the 15% was mostly trtllm-serve's "
          "CUDA-graph advantage, which the C++ tensorrt_llm backend doesn't have.\n")

    os.makedirs(str(_REPO / "results"), exist_ok=True)
    with open(str(_REPO / "results" / "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote results/report.md")
    _plot(runs, GROUP_FP8, "pareto_fp8.png", "FP8 head-to-head — Llama-3.1-8B TP=2")
    _plot(runs, GROUP_B, "pareto_h2h.png", "BF16 head-to-head — Llama-3.1-8B TP=2")
    _plot(runs, GROUP_D, "pareto_32b.png", "Qwen2.5-32B TP=4 head-to-head")
    _plot(runs, GROUP_A, "pareto_models.png", "Cross-model — vLLM TP=1")
    _plot(runs, GROUP_TUNED, "pareto_tuned.png", "TRT-LLM defaults vs tuned vs vLLM — FP8 TP=2")
    _plot(runs, GROUP_ENGINE, "pareto_engine.png", "Compiled engine vs PyTorch backend — BF16 TP=2")
    _plot(runs, GROUP_SM120, "pareto_sm120.png",
          "NVFP4 W4A4 vs BF16/FP8 — Llama-3.1-8B, RTX PRO 6000 (sm_120)")
    _plot(runs, GROUP_TRITON, "pareto_triton.png",
          "Triton ensemble vs trtllm-serve — same compiled BF16 engine, TP=2")


def _plot(runs, tags, fname, title):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    if not any(t in runs for t in tags):
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    present = [t for t in tags if t in runs]
    # Per-series annotation offsets (in points) so concurrency labels never sit on the
    # markers, and labels from near-coincident series do not collide. The scheme depends on
    # how many series share the axes:
    #   2 series (head-to-heads): the two curves cross, so push series 0 up-right and
    #     series 1 down-right — labels diverge away from each crossover, never stacking.
    #   3 series (cross-model): two curves (Qwen3-8B / Llama-3.1-8B) are near-coincident the
    #     whole sweep, so fan the three labels to three fixed vertical lanes (top/mid/bottom),
    #     all offset right, so coincident markers still get vertically-separated labels.
    offsets = ([(9, 8), (9, -15)] if len(present) <= 2
               else [(11, 13), (11, -7), (11, -22)])
    for i, tag in enumerate(present):
        rs = runs[tag]
        xs = [r["throughput_tok_s"] for r in rs]
        ys = [r["ttft_p99_s"] * 1000 for r in rs]
        ln, = ax.plot(xs, ys, "o-", lw=2, ms=7, label=LABEL.get(tag, tag))
        dx, dy = offsets[i % len(offsets)]
        for r in rs:
            ax.annotate(f"c{r['concurrency']}",
                        (r["throughput_tok_s"], r["ttft_p99_s"] * 1000),
                        textcoords="offset points", xytext=(dx, dy),
                        ha="left" if dx > 0 else "right",
                        fontsize=8.5, color=ln.get_color(), fontweight="bold")
    # log-y: TTFT spans ~20 ms to >2 s across the sweep; linear scale squashes the
    # low-concurrency points into an unreadable band at the bottom.
    ax.set_yscale("log")
    ax.margins(x=0.08, y=0.15)  # headroom so point labels never clip at the frame
    ax.set_xlabel("throughput (tok/s)"); ax.set_ylabel("TTFT p99 (ms, log scale)")
    ax.set_title(title)
    # lower right is always empty on a throughput-vs-TTFT pareto (curves rise to the right)
    ax.legend(framealpha=1.0, loc="lower right")
    ax.grid(True, alpha=0.3); ax.grid(True, alpha=0.12, which="minor")
    out = str(_REPO / "results" / fname)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
