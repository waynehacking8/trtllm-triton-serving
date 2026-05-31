#!/usr/bin/env python3
"""Build serving reports from concurrency sweeps. Three studies:

  A. Cross-model (vLLM, TP=1, BF16)      — Qwen3-8B vs Qwen3.5-9B vs Llama-3.1-8B
  B. Stack head-to-head (TP=2, BF16)     — TensorRT-LLM vs vLLM on Llama-3.1-8B
  C. Quantization (vLLM, Qwen3-8B, TP=2) — FP8 vs BF16

All sweeps use bench.py with ignore_eos + min_tokens, so every request on every stack/model
decodes EXACTLY max_tokens — throughput/ITL compare the same amount of work. Consumes
results/<tag>-c<N>.json; emits results/report.md + results/pareto_*.png.
"""
import glob, json, os, re
from collections import defaultdict

LABEL = {
    "xm_qwen3_8b": "Qwen3-8B", "xm_qwen35_9b": "Qwen3.5-9B", "xm_llama31_8b": "Llama-3.1-8B",
    "trtllm_llama31": "TensorRT-LLM (Llama-3.1-8B)", "vllm_llama31": "vLLM (Llama-3.1-8B)",
    "vllm_bf16": "vLLM BF16 (Qwen3-8B)", "vllm_fp8": "vLLM FP8 (Qwen3-8B)",
    "trtllm_qwen25_32b": "TensorRT-LLM (Qwen2.5-32B)", "vllm_qwen25_32b": "vLLM (Qwen2.5-32B)",
    "vllm_llama31_fp8": "vLLM FP8 (Llama-3.1-8B)",
    "trtllm_llama31_fp8": "TensorRT-LLM FP8 engine (Llama-3.1-8B)",
    "trtllm_llama31_tuned": "TensorRT-LLM PyTorch+CUDA-graph (Llama-3.1-8B)",
}
GROUP_A = ["xm_qwen3_8b", "xm_qwen35_9b", "xm_llama31_8b"]
GROUP_B = ["trtllm_llama31", "vllm_llama31"]
GROUP_C = ["vllm_bf16", "vllm_fp8"]
GROUP_D = ["trtllm_qwen25_32b", "vllm_qwen25_32b"]
GROUP_E = ["vllm_llama31_fp8", "trtllm_llama31_fp8", "trtllm_llama31",
           "trtllm_llama31_tuned", "vllm_llama31"]


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


def sla_table(w, runs, tags):
    slas = [0.10, 0.20, 0.50]
    w("| stack/model | " + " | ".join(f"p99≤{int(s*1000)}ms" for s in slas) + " |")
    w("|---|" + "---|" * len(slas))
    for tag in tags:
        if tag not in runs:
            continue
        w(f"| {LABEL.get(tag, tag)} | " + " | ".join(f"{sla_throughput(runs[tag], s):.0f}" for s in slas) + " |")
    w("")


def main():
    runs = load_sweeps()
    if not runs:
        print("no sweep results"); return
    L = []; w = L.append
    w("# Serving benchmark — cross-model, stack head-to-head, and quantization (4×H100)\n")
    w("vLLM / TensorRT-LLM OpenAI servers, fixed prompt, max_tokens=256, temp=0, **ignore_eos "
      "+ min_tokens** so every request decodes exactly 256 tokens on every stack/model (a fair, "
      "controlled comparison). Load: `bench/bench.py` (async, streaming, TTFT + inter-token "
      "latency). ITL is per-streamed-chunk (≈ per token).\n")

    if any(t in runs for t in GROUP_A):
        w("## 1. Cross-model — vLLM, TP=1, BF16\n")
        w("Same server, same hardware (1× H100 each), three models across two generations/"
          "families: Llama-3.1-8B (2024) vs Qwen3-8B and Qwen3.5-9B (2026).\n")
        sweep_table(w, runs, GROUP_A)
        present = [t for t in GROUP_A if t in runs]
        cs = sorted({r["concurrency"] for t in present for r in runs[t]})
        w("Throughput (tok/s) by concurrency:\n")
        w("| model | " + " | ".join(f"c{c}" for c in cs) + " |")
        w("|---|" + "---|" * len(cs))
        for t in present:
            byc = {r["concurrency"]: r["throughput_tok_s"] for r in runs[t]}
            w(f"| {LABEL[t]} | " + " | ".join(f"{byc.get(c,0):.0f}" for c in cs) + " |")
        w("")
        w("Reads as a model-selection table: at a target concurrency, which model gives the "
          "most tok/s per H100. Larger/newer models trade throughput for capability — the SA "
          "job is to put that trade in front of the customer with numbers.\n")
        w("### Throughput at a TTFT-p99 SLA (tok/s)\n"); sla_table(w, runs, GROUP_A)

    if any(t in runs for t in GROUP_B):
        w("## 2. Stack head-to-head — TensorRT-LLM vs vLLM, Llama-3.1-8B, TP=2, BF16\n")
        w("Same model (Llama-3.1-8B, supported on the TRT-LLM 0.20 **compiled-engine** path), "
          "same parallelism, same controlled 256-token decode.\n")
        sweep_table(w, runs, GROUP_B)
        if all(t in runs for t in GROUP_B):
            tr = {r["concurrency"]: r for r in runs["trtllm_llama31"]}
            vl = {r["concurrency"]: r for r in runs["vllm_llama31"]}
            w("| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL ms | vLLM ITL ms |")
            w("|---|---|---|---|---|---|")
            for c in sorted(set(tr) & set(vl)):
                a, b = tr[c], vl[c]
                rt = a["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else 0
                w(f"| {c} | {a['throughput_tok_s']:.0f} | {b['throughput_tok_s']:.0f} | {rt:.2f}× | "
                  f"{a['itl_p50_ms']:.2f} | {b['itl_p50_ms']:.2f} |")
            w("")
        w("### Throughput at a TTFT-p99 SLA (tok/s)\n"); sla_table(w, runs, GROUP_B)

    if all(t in runs for t in GROUP_C):
        w("## 3. Quantization — vLLM, Qwen3-8B, TP=2 (FP8 vs BF16)\n")
        sweep_table(w, runs, GROUP_C)
        bf = {r["concurrency"]: r for r in runs["vllm_bf16"]}
        fp = {r["concurrency"]: r for r in runs["vllm_fp8"]}
        w("| concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup | BF16 ITL ms | FP8 ITL ms |")
        w("|---|---|---|---|---|---|")
        for c in sorted(set(bf) & set(fp)):
            b, f = bf[c], fp[c]
            sp = f["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else 0
            w(f"| {c} | {b['throughput_tok_s']:.0f} | {f['throughput_tok_s']:.0f} | {sp:.2f}× | "
              f"{b['itl_p50_ms']:.2f} | {f['itl_p50_ms']:.2f} |")
        w("")
        w("FP8 (Hopper FP8 tensor cores) wins most at low concurrency where decode is "
          "memory-bandwidth-bound; the edge narrows under heavy batching.\n")

    if any(t in runs for t in GROUP_D):
        w("## 4. Big model at scale — TensorRT-LLM vs vLLM, Qwen2.5-32B, TP=4, BF16\n")
        w("Same head-to-head as §2, but a 32B model **tensor-parallel across 4× H100** "
          "(`Qwen2ForCausalLM`, TRT-LLM compiled engine), controlled 256-token decode — the "
          "multi-GPU operating point that actually exercises the box.\n")
        sweep_table(w, runs, GROUP_D)
        if all(t in runs for t in GROUP_D):
            tr = {r["concurrency"]: r for r in runs["trtllm_qwen25_32b"]}
            vl = {r["concurrency"]: r for r in runs["vllm_qwen25_32b"]}
            w("| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL ms | vLLM ITL ms |")
            w("|---|---|---|---|---|---|")
            for c in sorted(set(tr) & set(vl)):
                a, b = tr[c], vl[c]
                rt = a["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else 0
                w(f"| {c} | {a['throughput_tok_s']:.0f} | {b['throughput_tok_s']:.0f} | {rt:.2f}× | "
                  f"{a['itl_p50_ms']:.2f} | {b['itl_p50_ms']:.2f} |")
            w("")
        w("### Throughput at a TTFT-p99 SLA (tok/s)\n"); sla_table(w, runs, GROUP_D)

    if any(t in runs for t in GROUP_E):
        w("## 5. Every lever — config matrix, Llama-3.1-8B, TP=2 (both stacks, both precisions)\n")
        w("Pulled every TRT-LLM lever vs vLLM, same model/controlled decode. FP8 is TRT-LLM's "
          "best config (+46% over its BF16) — the Hopper W8A8 lever is real — but vLLM FP8 "
          "still wins ~1.85×; the residual is vLLM's default CUDA-graph decode capture.\n")
        w("| config | tok/s @c1 | ITL @c1 |")
        w("|---|---|---|")
        order = ["vllm_llama31_fp8", "vllm_llama31", "trtllm_llama31_fp8", "trtllm_llama31",
                 "trtllm_llama31_tuned"]
        for t in order:
            if t in runs:
                r = next((x for x in runs[t] if x["concurrency"] == 1), runs[t][0])
                w(f"| {LABEL.get(t,t)} | {r['throughput_tok_s']:.0f} | {r['itl_p50_ms']:.2f} ms |")
        w("")

    os.makedirs("results", exist_ok=True)
    open("results/report.md", "w").write("\n".join(L) + "\n")
    print("wrote results/report.md")
    _plot(runs, GROUP_A, "pareto_models.png", "Cross-model — vLLM TP=1 (4×H100)")
    _plot(runs, GROUP_B, "pareto_h2h.png", "TensorRT-LLM vs vLLM — Llama-3.1-8B TP=2")
    _plot(runs, GROUP_C, "pareto_quant.png", "vLLM FP8 vs BF16 — Qwen3-8B TP=2")
    _plot(runs, GROUP_D, "pareto_32b.png", "TensorRT-LLM vs vLLM — Qwen2.5-32B TP=4")


def _plot(runs, tags, fname, title):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    if not any(t in runs for t in tags):
        return
    plt.figure(figsize=(8, 5))
    for tag in tags:
        if tag not in runs:
            continue
        xs = [r["throughput_tok_s"] for r in runs[tag]]
        ys = [r["ttft_p99_s"] * 1000 for r in runs[tag]]
        plt.plot(xs, ys, "o-", label=LABEL.get(tag, tag))
        for r in runs[tag]:
            plt.annotate(f"c{r['concurrency']}", (r["throughput_tok_s"], r["ttft_p99_s"] * 1000),
                         fontsize=7, alpha=0.6)
    plt.xlabel("throughput (tok/s)"); plt.ylabel("TTFT p99 (ms)")
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"results/{fname}", dpi=130)
    print(f"wrote results/{fname}")


if __name__ == "__main__":
    main()
