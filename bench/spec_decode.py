#!/usr/bin/env python3
"""Frontier study: n-gram (prompt-lookup) speculative decoding — when it pays off.

Speculative decoding proposes several tokens with a cheap "draft" and verifies them in one
target forward pass; accepted tokens are free. The *n-gram* variant needs no draft model — it
drafts by looking up recent n-grams from the prompt/context, so it shines exactly when the
output echoes the input (RAG, summarization, code editing, agentic tool transcripts) and is a
net loss on free-form generation (low acceptance, wasted draft+verify).

Measured here: vLLM, Qwen2.5-7B, TP=1, baseline vs `--speculative-config {method: ngram,
num_speculative_tokens: 5, prompt_lookup_max: 4}`. **Measured NON-STREAMING on purpose** —
spec decode emits tokens in bursts, and per-SSE-chunk streaming serializes them over the
network, hiding the GPU-side speedup (it reads as ~0.85× streamed vs ~2.6× non-streamed; the
same client-side-streaming caveat that the roofline study flagged for the head-to-heads).

Run: python bench/spec_decode.py   (servers on :8090 baseline, :8091 ngram)
"""
import json, os, time

PASSAGE = ("NVIDIA NIM packages optimized inference as microservices with an OpenAI-compatible "
           "API. TensorRT-LLM compiles models into engines with paged KV-cache and FP8 "
           "quantization. NVLS performs in-switch reduction so each GPU sends data once. CUDA "
           "graphs capture the decode step to remove per-kernel launch overhead. ") * 2
TASKS = {
    "extractive (RAG-style, echoes context)": "Repeat the following text exactly, word for word:\n" + PASSAGE,
    "generative (novel text)": "Explain tensor parallelism for LLM inference in detail.",
}
BASE, SPEC = "http://localhost:8090", "http://localhost:8091"


def tps_nonstream(base, model, prompt, maxtok=200):
    import httpx
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/v1/completions",
                   json={"model": model, "prompt": prompt, "max_tokens": maxtok,
                         "temperature": 0, "stream": False}, timeout=120)
    dt = time.perf_counter() - t0
    return r.json()["usage"]["completion_tokens"] / dt


def acceptance(spec_base):
    import httpx
    try:
        m = httpx.get(f"{spec_base}/metrics", timeout=10).text
        d = a = 0
        for line in m.splitlines():
            if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                d = float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                a = float(line.split()[-1])
        return a / d if d else 0.0
    except Exception:
        return 0.0


def plot(src="results/spec_decode.json", fname="results/spec_decode.png"):
    """Render results/spec_decode.json as a grouped bar chart. All numbers come from the
    JSON — nothing is hard-coded. Aesthetic matches bench/pareto.py (matplotlib default
    tab colors, 8×5, dpi 130, grid alpha 0.3)."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    with open(src) as fh:
        d = json.load(fh)
    rows = d["results"]
    # short labels for the x axis (first word before the parenthesis)
    tasks = [r["task"].split(" (")[0] for r in rows]
    baseline = [r["baseline_tok_s"] for r in rows]
    spec = [r["ngram_spec_tok_s"] for r in rows]
    speedup = [r["speedup"] for r in rows]

    x = np.arange(len(rows)); width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, baseline, width, label="baseline", color="tab:gray")
    b2 = ax.bar(x + width / 2, spec, width, label="n-gram spec decode", color="tab:blue")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.0f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)
    # speedup annotation above each spec bar
    for xi, s, sp in zip(x, spec, speedup):
        ax.annotate(f"{sp:.2f}×", (xi + width / 2, s), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color=("tab:green" if sp >= 1.0 else "tab:red"))
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title(f"n-gram speculative decoding — {d['model']}, "
                 f"k={d['num_speculative_tokens']}, accept {d['draft_acceptance_rate']:.0%} "
                 f"(non-streaming)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(spec + baseline) * 1.25)
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def main(base=BASE, spec=SPEC):
    rows = []
    for label, prompt in TASKS.items():
        tps_nonstream(base, "base", prompt); tps_nonstream(spec, "spec", prompt)  # warm
        b = tps_nonstream(base, "base", prompt)
        s = tps_nonstream(spec, "spec", prompt)
        rows.append({"task": label, "baseline_tok_s": round(b, 1),
                     "ngram_spec_tok_s": round(s, 1), "speedup": round(s / b, 2)})
    out = {"model": "Qwen2.5-7B-Instruct", "method": "ngram (prompt-lookup)",
           "num_speculative_tokens": 5, "draft_acceptance_rate": round(acceptance(spec), 2),
           "results": rows}
    os.makedirs("results", exist_ok=True)
    with open("results/spec_decode.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    plot()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        main()
