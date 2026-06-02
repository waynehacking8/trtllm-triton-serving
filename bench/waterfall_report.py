#!/usr/bin/env python3
"""Waterfall attribution report + chart (roadmap Phase 6).

Parses the trtllm-bench logs produced by scripts/waterfall.sh and decomposes the gap
between NVIDIA's published Llama-3.1-8B-FP8 1xH100 number and this repo's measured
serving throughput, one attributed knob per bar.

Inputs:  results/waterfall/W*.txt        (raw trtllm-bench stdout, committed)
         results/trtllm_llama31_fp8_tuned-c128.json  (the repo's serving measurement)
Outputs: results/waterfall.png + a markdown table printed to stdout (paste into report).

Usage: python bench/waterfall_report.py [results/waterfall]
"""
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WF_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results", "waterfall")

# Published references (docs change between releases - both are recorded):
PUBLISHED_020 = 27688.36   # release/0.20 perf-overview (matches the installed TRT-LLM 0.20.0)
PUBLISHED_CUR = 26401.48   # current (1.x) perf-overview, same row

NVIDIA_GREEN = "#76b900"
RED = "#c0392b"
GREY = "#9aa0a6"

# (file_tag, label, what changed vs previous step)
# Note: TP 1->2 could NOT be measured with trtllm-bench (0.20 crashes at TP2 init - rank 1
# illegal memory access, reproducible across GPU pairs/IPC flags/CG lists; failed logs kept
# as W4_tp2.txt). The TP step therefore happens on the serving side: W5 (serve, TP1) ->
# W6 (serve, TP2 = the repo's committed measurement).
STEPS = [
    ("W0a_kv090", "W0: NVIDIA config\n(0.20-runnable)", "ISL/OSL 128/128, TP1, kv 0.90*, CG list, offline"),
    ("W1_osl256", "W1: OSL 256", "output length 128 -> 256 (repo decodes 256/request)"),
    ("W2_isl12", "W2: ISL 12", "input length 128 -> 12 (repo's real prompt size)"),
    ("W3_kv080", "W3: kv 0.80", "kv_cache_free_gpu_mem_fraction 0.90 -> 0.80 (neutral)"),
    ("W4_conc128", "W4: c128 cap", "offline unbounded -> 128 concurrent requests (still TP1)"),
]
# W5: trtllm-serve TP1 (HTTP streaming serving stack, serve-tuned config: max_batch 256,
# kv 0.85, CG-256, chunked prefill, MAX_UTILIZATION) measured by bench/bench.py at c128.
W5_SERVE_TP1 = "W5_serve_tp1-c128.json"


def parse_bench_txt(path):
    """Total output tok/s from a trtllm-bench log (report JSON preferred when present)."""
    rj = path.replace(".txt", ".report.json")
    if os.path.exists(rj):
        try:
            d = json.load(open(rj))
            # 0.20 report schema: nested under "performance" or top-level keys
            for k in ("total_output_throughput_tok_s", "output_throughput_tok_s"):
                if k in d:
                    return float(d[k])
            perf = d.get("performance", {})
            for k in ("total_output_throughput", "output_throughput"):
                if k in perf:
                    return float(perf[k])
        except (json.JSONDecodeError, ValueError):
            pass
    txt = open(path, errors="replace").read()
    m = re.search(r"Total Output Throughput \(tokens/sec\):\s*([\d.]+)", txt)
    return float(m.group(1)) if m else None


def main():
    rows = []
    for tag, label, knob in STEPS:
        p = os.path.join(WF_DIR, f"{tag}.txt")
        tput = parse_bench_txt(p) if os.path.exists(p) else None
        rows.append({"tag": tag, "label": label, "knob": knob, "tok_s": tput})

    # W5: trtllm-serve TP1 measured by the repo's own client harness (same workload shape)
    w5_path = os.path.join(WF_DIR, W5_SERVE_TP1)
    if os.path.exists(w5_path):
        w5 = json.load(open(w5_path))
        rows.append({"tag": "W5_serve_tp1", "label": "W5: serving stack\n(trtllm-serve, TP1)",
                     "knob": "offline bench harness -> HTTP streaming serving "
                             "(serve config: max_batch 256, kv 0.85, CG-256, chunked prefill)",
                     "tok_s": w5["throughput_tok_s"]})

    # W6: the repo's committed serving measurement (TP2)
    serving = json.load(open(os.path.join(REPO, "results", "trtllm_llama31_fp8_tuned-c128.json")))
    rows.append({"tag": "W6_serve_tp2", "label": "W6: TP 1 -> 2\n(committed measurement)",
                 "knob": "tensor parallelism 1 -> 2 in serving (trtllm-bench TP2 is broken in "
                         "0.20 - see W4_tp2.txt)",
                 "tok_s": serving["throughput_tok_s"]})

    missing = [r["tag"] for r in rows if r["tok_s"] is None]
    if missing:
        print(f"WARNING: missing steps {missing} - chart will skip them")
    done = [r for r in rows if r["tok_s"] is not None]

    # ---- markdown table ----
    print("\n| step | config change | tok/s | delta vs prev | % of W0 |")
    print("|---|---|---|---|---|")
    w0 = done[0]["tok_s"]
    prev = None
    for r in done:
        delta = f"{r['tok_s'] - prev:+,.0f}" if prev is not None else "—"
        print(f"| {r['label'].replace(chr(10), ' ')} | {r['knob']} | {r['tok_s']:,.0f} | "
              f"{delta} | {r['tok_s'] / w0 * 100:.0f}% |")
        prev = r["tok_s"]
    print(f"\npublished references: {PUBLISHED_020:,.2f} (release/0.20 docs) / "
          f"{PUBLISHED_CUR:,.2f} (current docs)")
    print(f"W0 reaches {w0 / PUBLISHED_020 * 100:.0f}% of the 0.20-docs number\n")

    # ---- waterfall chart ----
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [r["label"] for r in done]
    vals = [r["tok_s"] for r in done]
    colors = [NVIDIA_GREEN] + [GREY] * (len(done) - 2) + [RED]
    bars = ax.bar(labels, vals, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 300, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    # delta annotations between bars
    for i in range(1, len(done)):
        d = done[i]["tok_s"] - done[i - 1]["tok_s"]
        x = i - 0.5
        y = max(done[i]["tok_s"], done[i - 1]["tok_s"]) + 1600
        ax.annotate(f"{d:+,.0f}", (x, y), ha="center", fontsize=9,
                    color=RED if d < 0 else "#3a7000", fontweight="bold")
    # published reference lines
    ax.axhline(PUBLISHED_020, color="black", linestyle="--", linewidth=1.2)
    ax.text(len(done) - 0.5, PUBLISHED_020 + 300, f"published (0.20 docs): {PUBLISHED_020:,.0f}",
            ha="right", fontsize=9)
    ax.axhline(PUBLISHED_CUR, color="black", linestyle=":", linewidth=1)
    ax.text(len(done) - 0.5, PUBLISHED_CUR - 1100, f"published (current docs): {PUBLISHED_CUR:,.0f}",
            ha="right", fontsize=9)
    ax.set_ylabel("total output throughput (tok/s)")
    ax.set_title("From NVIDIA's published number to this repo's measurement - one knob at a time\n"
                 "Llama-3.1-8B FP8 on H100 SXM, TRT-LLM 0.20 PyTorch backend (trtllm-bench)",
                 fontsize=12, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    fig.tight_layout()
    out = os.path.join(REPO, "results", "waterfall.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
