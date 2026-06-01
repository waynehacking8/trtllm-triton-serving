#!/usr/bin/env python3
"""Build serving reports from concurrency sweeps. Studies (all TRT-LLM runs use the
correctly-applied CUDA-graph config — see README "verification" note):

  1. Cross-model (vLLM TP=1)            — Llama-3.1-8B / Qwen3-8B / Qwen3.5-9B
  2. Head-to-head BF16 (Llama-3.1-8B TP=2)   — TensorRT-LLM+CUDA-graph vs vLLM
  3. Head-to-head FP8  (Llama-3.1-8B TP=2)   — TensorRT-LLM+CUDA-graph vs vLLM  [headline]
  4. Head-to-head BF16 (Qwen2.5-32B TP=4)    — TensorRT-LLM+CUDA-graph vs vLLM
  5. Quantization (Qwen3-8B vLLM TP=2)       — FP8 vs BF16

All requests decode exactly 256 tokens (ignore_eos). Consumes results/<tag>-c<N>.json.
"""
import glob, json, os, re
from collections import defaultdict

LABEL = {
    "xm_qwen3_8b": "Qwen3-8B", "xm_qwen35_9b": "Qwen3.5-9B", "xm_llama31_8b": "Llama-3.1-8B",
    "trtllm_llama31": "TensorRT-LLM+CG", "vllm_llama31": "vLLM",
    "trtllm_llama31_fp8": "TensorRT-LLM+CG", "vllm_llama31_fp8": "vLLM",
    "trtllm_qwen25_32b": "TensorRT-LLM+CG", "vllm_qwen25_32b": "vLLM",
    "vllm_bf16": "vLLM BF16", "vllm_fp8": "vLLM FP8",
}
GROUP_A = ["xm_qwen3_8b", "xm_qwen35_9b", "xm_llama31_8b"]
GROUP_B = ["trtllm_llama31", "vllm_llama31"]
GROUP_FP8 = ["trtllm_llama31_fp8", "vllm_llama31_fp8"]
GROUP_D = ["trtllm_qwen25_32b", "vllm_qwen25_32b"]
GROUP_C = ["vllm_bf16", "vllm_fp8"]


def load_sweeps():
    runs = defaultdict(list)
    for f in glob.glob("results/*-c*.json"):
        m = re.match(r"(.+)-c(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        tag, c = m.group(1), int(m.group(2))
        d = json.load(open(f)); d["concurrency"] = c
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
            w(f"| {c} | {a:.0f} | {b:.0f} | {r:.2f}× | {win} |")
        w("")


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
        w("\nThe 9B carries ~25% less throughput/H100 than the 8Bs — the capability-vs-cost "
          "trade, with numbers.\n")

    h2h(w, runs, "trtllm_llama31_fp8", "vllm_llama31_fp8",
        "2. Head-to-head FP8 — Llama-3.1-8B, TP=2 (headline)",
        "Same model & precision (FP8, `nvidia/Llama-3.1-8B-Instruct-FP8`), TRT-LLM's PyTorch "
        "backend + CUDA graphs (`--backend pytorch`, *not* a pre-compiled TRT engine — a "
        "compiled-engine comparison is future work) vs vLLM. **TRT-LLM wins the "
        "low/mid-concurrency (latency) regime; vLLM wins high concurrency (throughput).** "
        "This is the textbook split and only appears once CUDA graphs are correctly on.")

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

    os.makedirs("results", exist_ok=True)
    open("results/report.md", "w").write("\n".join(L) + "\n")
    print("wrote results/report.md")
    _plot(runs, GROUP_FP8, "pareto_fp8.png", "FP8 head-to-head — Llama-3.1-8B TP=2")
    _plot(runs, GROUP_B, "pareto_h2h.png", "BF16 head-to-head — Llama-3.1-8B TP=2")
    _plot(runs, GROUP_D, "pareto_32b.png", "Qwen2.5-32B TP=4 head-to-head")
    _plot(runs, GROUP_A, "pareto_models.png", "Cross-model — vLLM TP=1")


def _plot(runs, tags, fname, title):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    if not any(t in runs for t in tags):
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Per-series annotation offsets (in points) so concurrency labels never sit on the
    # markers, and labels from different series near the same spot do not collide.
    offsets = [(9, 8), (9, -15), (9, -16)]
    present = [t for t in tags if t in runs]
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
    fig.tight_layout(); fig.savefig(f"results/{fname}", dpi=130); plt.close(fig)
    print(f"wrote results/{fname}")


if __name__ == "__main__":
    main()
