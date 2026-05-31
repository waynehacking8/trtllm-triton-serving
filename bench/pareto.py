#!/usr/bin/env python3
"""Build serving reports from concurrency sweeps: a TensorRT-LLM vs vLLM head-to-head and
an FP8/BF16 quantization study. Consumes results/<tag>-c<N>.json (from bench.py / sweep.sh).

Tags:
  trtllm_qwen25, vllm_qwen25   — same model (Qwen2.5-7B-Instruct, TP=2, BF16): stack head-to-head
  vllm_bf16, vllm_fp8          — same model (Qwen3-8B, TP=2): quantization study

Emits results/report.md + results/pareto_*.png. The Pareto framing is the serving one:
at a tail-latency SLA, how much throughput can each config sustain?
"""
import glob, json, os, re
from collections import defaultdict

LABEL = {"trtllm_qwen25": "TensorRT-LLM", "vllm_qwen25": "vLLM",
         "vllm_bf16": "vLLM BF16", "vllm_fp8": "vLLM FP8"}


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
    w("| stack | " + " | ".join(f"p99≤{int(s*1000)}ms" for s in slas) + " |")
    w("|---|" + "---|" * len(slas))
    for tag in tags:
        if tag not in runs:
            continue
        cells = [f"{sla_throughput(runs[tag], s):.0f}" for s in slas]
        w(f"| {LABEL.get(tag, tag)} | " + " | ".join(cells) + " |")
    w("")


def main():
    runs = load_sweeps()
    if not runs:
        print("no sweep results"); return
    L = []; w = L.append
    w("# Serving benchmark — TensorRT-LLM vs vLLM, and FP8 vs BF16 (4×H100)\n")
    w("vLLM and TensorRT-LLM OpenAI servers, TP=2, same GPUs class, fixed prompt, "
      "max_tokens=256, temp=0. Load from `bench/bench.py` (async, streaming, per-request "
      "TTFT + inter-token latency).\n")

    h2h = ["trtllm_qwen25", "vllm_qwen25"]
    if any(t in runs for t in h2h):
        w("## 1. TensorRT-LLM vs vLLM — Qwen2.5-7B-Instruct, TP=2, BF16\n")
        w("Same model, same parallelism — TensorRT-LLM uses the compiled engine path "
          "(`Qwen2ForCausalLM`, supported in TRT-LLM 0.20); vLLM is the baseline.\n")
        sweep_table(w, runs, h2h)
        if all(t in runs for t in h2h):
            tr = {r["concurrency"]: r for r in runs["trtllm_qwen25"]}
            vl = {r["concurrency"]: r for r in runs["vllm_qwen25"]}
            w("Throughput ratio (TensorRT-LLM ÷ vLLM) by concurrency:\n")
            w("| concurrency | TRT-LLM tok/s | vLLM tok/s | ratio | TRT-LLM ITL ms | vLLM ITL ms |")
            w("|---|---|---|---|---|---|")
            for c in sorted(set(tr) & set(vl)):
                a, b = tr[c], vl[c]
                rt = a["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else 0
                w(f"| {c} | {a['throughput_tok_s']:.0f} | {b['throughput_tok_s']:.0f} | "
                  f"{rt:.2f}× | {a['itl_p50_ms']:.2f} | {b['itl_p50_ms']:.2f} |")
            w("")
        w("### Throughput at a TTFT-p99 SLA (tok/s)\n")
        sla_table(w, runs, h2h)

    quant = ["vllm_bf16", "vllm_fp8"]
    if all(t in runs for t in quant):
        w("## 2. FP8 vs BF16 — vLLM, Qwen3-8B, TP=2\n")
        sweep_table(w, runs, quant)
        bf = {r["concurrency"]: r for r in runs["vllm_bf16"]}
        fp = {r["concurrency"]: r for r in runs["vllm_fp8"]}
        w("| concurrency | BF16 tok/s | FP8 tok/s | FP8 speedup | BF16 ITL ms | FP8 ITL ms |")
        w("|---|---|---|---|---|---|")
        for c in sorted(set(bf) & set(fp)):
            b, f = bf[c], fp[c]
            sp = f["throughput_tok_s"] / b["throughput_tok_s"] if b["throughput_tok_s"] else 0
            w(f"| {c} | {b['throughput_tok_s']:.0f} | {f['throughput_tok_s']:.0f} | "
              f"{sp:.2f}× | {b['itl_p50_ms']:.2f} | {f['itl_p50_ms']:.2f} |")
        w("")
        w("FP8 (Hopper FP8 tensor cores) wins most at low concurrency where decode is "
          "memory-bandwidth-bound; the edge narrows under heavy batching.\n")
        w("### Throughput at a TTFT-p99 SLA (tok/s)\n")
        sla_table(w, runs, quant)

    os.makedirs("results", exist_ok=True)
    open("results/report.md", "w").write("\n".join(L) + "\n")
    print("wrote results/report.md")
    _plot(runs, h2h, "pareto_h2h.png", "TensorRT-LLM vs vLLM — Qwen2.5-7B TP=2")
    _plot(runs, quant, "pareto_quant.png", "vLLM FP8 vs BF16 — Qwen3-8B TP=2")


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
