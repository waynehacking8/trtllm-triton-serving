#!/usr/bin/env python3
"""Speculative decoding under concurrency — where does the benefit cross below 1.0×?

The batch=1 study (bench/spec_decode.py) found n-gram (prompt-lookup) speculative decoding
gives 2.8× on extractive/RAG-style output. But speculation spends extra compute per accepted
token: at small batch the GPU is memory-bound and that compute is free; as concurrency rises
the engine becomes compute-bound and verification overhead turns into a tax (Nightjar,
arXiv:2512.22420; vLLM docs say the same). This study measures the crossover.

Method: the SAME extractive prompt set, non-streaming (spec decode emits token bursts;
client-side SSE streaming hides the GPU-side speedup), against a baseline vLLM server and an
n-gram-spec vLLM server run SEQUENTIALLY on the same GPU. Aggregate tok/s at each concurrency;
speedup = spec/baseline.

Usage:
  python3 bench/spec_concurrency.py bench --base <url> --model <name> --tag <baseline|ngram>
  python3 bench/spec_concurrency.py report   # merge the two tags -> json + png
"""
import argparse
import asyncio
import json
import os
import time

import httpx

PASSAGE = ("NVIDIA NIM packages optimized inference as microservices with an OpenAI-compatible "
           "API. TensorRT-LLM compiles models into engines with paged KV-cache and FP8 "
           "quantization. NVLS performs in-switch reduction so each GPU sends data once. CUDA "
           "graphs capture the decode step to remove per-kernel launch overhead. ") * 2
PROMPT = "Repeat the following text exactly, word for word:\n" + PASSAGE
CONCURRENCIES = [1, 4, 16, 32, 64, 128]
MAX_TOKENS = 200
RESULTS_DIR = "results"


async def one_request(client, base, model):
    r = await client.post(f"{base}/v1/completions",
                          json={"model": model, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
                                "temperature": 0, "stream": False}, timeout=600)
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


async def run_level(base, model, concurrency, total):
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded():
            async with sem:
                return await one_request(client, base, model)

        t0 = time.perf_counter()
        toks = await asyncio.gather(*[guarded() for _ in range(total)])
        wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "requests": total,
            "out_tokens_total": sum(toks), "wall_s": round(wall, 3),
            "throughput_tok_s": round(sum(toks) / wall, 1)}


def spec_counters(base):
    """Cumulative draft/accepted token counters from vLLM /metrics (0,0 if absent)."""
    try:
        m = httpx.get(f"{base}/metrics", timeout=10).text
        d = a = 0.0
        for line in m.splitlines():
            if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                d = float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                a = float(line.split()[-1])
        return d, a
    except Exception:
        return 0.0, 0.0


def bench(base, model, tag):
    rows = []
    for c in CONCURRENCIES:
        # warm-up at this concurrency, then measure
        asyncio.run(run_level(base, model, c, c))
        d0, a0 = spec_counters(base)
        row = asyncio.run(run_level(base, model, c, c * 4))
        d1, a1 = spec_counters(base)
        row["draft_acceptance"] = round((a1 - a0) / (d1 - d0), 3) if d1 > d0 else None
        rows.append(row)
        print(f"  c={c:>3}  {row['throughput_tok_s']:>9} tok/s  "
              f"accept={row['draft_acceptance']}", flush=True)
    out = {"model": model, "tag": tag, "max_tokens": MAX_TOKENS, "task": "extractive",
           "rows": rows}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"spec_concurrency_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


def report():
    base = json.load(open(os.path.join(RESULTS_DIR, "spec_concurrency_baseline.json")))
    spec = json.load(open(os.path.join(RESULTS_DIR, "spec_concurrency_ngram.json")))
    rows = []
    for b, s in zip(base["rows"], spec["rows"]):
        assert b["concurrency"] == s["concurrency"]
        rows.append({"concurrency": b["concurrency"],
                     "baseline_tok_s": b["throughput_tok_s"],
                     "ngram_tok_s": s["throughput_tok_s"],
                     "speedup": round(s["throughput_tok_s"] / b["throughput_tok_s"], 3),
                     "draft_acceptance": s["draft_acceptance"]})
    crossover = next((r["concurrency"] for r in rows if r["speedup"] < 1.0), None)
    out = {"model": spec["model"], "task": "extractive (RAG-style)",
           "crossover_concurrency": crossover, "rows": rows}
    path = os.path.join(RESULTS_DIR, "spec_concurrency.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    plot(out)


def plot(data, fname=os.path.join(RESULTS_DIR, "spec_concurrency.png")):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = data["rows"]
    cs = [r["concurrency"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    accepts = [r["draft_acceptance"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs, speedups, "o-", color="tab:blue", label="ngram spec speedup", zorder=3)
    ax.axhline(1.0, color="tab:red", linestyle="--", alpha=0.7, label="break-even (1.0×)")
    for c, s in zip(cs, speedups):
        ax.annotate(f"{s:.2f}×", (c, s), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color="tab:green" if s >= 1.0 else "tab:red")
    ax2 = ax.twinx()
    ax2.plot(cs, accepts, "s--", color="tab:gray", alpha=0.6, label="draft acceptance")
    ax2.set_ylabel("draft acceptance rate", color="tab:gray")
    ax2.set_ylim(0, 1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("concurrency")
    ax.set_ylabel("speedup vs no-spec baseline")
    cross = data["crossover_concurrency"]
    ax.set_title(f"n-gram speculative decoding vs concurrency — {data['model']}\n"
                 f"extractive task, non-streaming; crossover at c={cross}" if cross else
                 f"n-gram speculative decoding vs concurrency — {data['model']}\n"
                 "extractive task, non-streaming; no crossover up to c=128")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"wrote {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "report"])
    ap.add_argument("--base", default="http://localhost:8090")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--tag", default="baseline", choices=["baseline", "ngram"])
    a = ap.parse_args()
    if a.cmd == "bench":
        bench(a.base, a.model, a.tag)
    else:
        report()


if __name__ == "__main__":
    main()
