#!/usr/bin/env python3
"""Verification: measured single-stream decode vs the memory-bandwidth roofline.

Single-stream (concurrency 1) LLM decode is memory-bandwidth-bound: each token reads the
whole weight set once, so the ceiling is

    tok/s_max = (HBM_bandwidth * n_gpu) / weight_bytes

This is the sanity check that caught a real config bug in this repo: an early TRT-LLM run
measured 162 tok/s for Llama-3.1-8B FP8 (TP=2) = only ~19% of roofline, which is physically
implausible for NVIDIA's own kernels. Root cause: CUDA graphs were silently NOT enabled (wrong
`extra_llm_api_options` schema). After the fix the same config hit 374 tok/s = ~45% of the
TP=2 roofline / ~89% of the 1-GPU roofline — in the expected band. Numbers far below roofline
are a red flag to debug, not to publish.

H100 SXM5 HBM3: ~3.35 TB/s per GPU.
"""
import pathlib
_REPO = pathlib.Path(__file__).resolve().parent.parent

HBM_TBS = 3.35  # TB/s per H100 SXM5

# (label, params_billions, bytes_per_param, n_gpu)
CONFIGS = [
    ("Llama-3.1-8B BF16, TP=2", 8.03, 2, 2),
    ("Llama-3.1-8B FP8,  TP=2", 8.03, 1, 2),
    ("Qwen2.5-32B BF16,  TP=4", 32.8, 2, 4),
]
# measured concurrency-1 throughput (tok/s) — fill from results/*-c1.json
MEASURED = {
    "Llama-3.1-8B BF16, TP=2": {"TRT-LLM+CG": None, "vLLM": None,
                                "TRT compiled engine": None, "TRT compiled engine+CG": None},
    "Llama-3.1-8B FP8,  TP=2": {"TRT-LLM+CG": None, "vLLM": None, "TRT-LLM tuned": None},
    "Qwen2.5-32B BF16,  TP=4": {"TRT-LLM+CG": None, "vLLM": None},
}


def roofline(params_b, bytes_pp, n_gpu):
    weight_bytes = params_b * 1e9 * bytes_pp
    # aggregate HBM across the TP group (weights are sharded, read in parallel)
    return HBM_TBS * 1e12 * n_gpu / weight_bytes


def load_measured():
    import json
    m = {
        "Llama-3.1-8B BF16, TP=2": {"TRT-LLM+CG": "trtllm_llama31", "vLLM": "vllm_llama31",
                                    "TRT compiled engine": "trtllm_compiled_bf16",
                                    "TRT compiled engine+CG": "trtllm_compiled_bf16_cg"},
        "Llama-3.1-8B FP8,  TP=2": {"TRT-LLM+CG": "trtllm_llama31_fp8", "vLLM": "vllm_llama31_fp8",
                                    "TRT-LLM tuned": "trtllm_llama31_fp8_tuned"},
        "Qwen2.5-32B BF16,  TP=4": {"TRT-LLM+CG": "trtllm_qwen25_32b", "vLLM": "vllm_qwen25_32b"},
    }
    for label, engines in m.items():
        for engine, tag in engines.items():
            p = _REPO / "results" / f"{tag}-c1.json"
            if p.exists():
                with open(p) as fh:
                    MEASURED[label][engine] = json.load(fh)["throughput_tok_s"]


def main():
    load_measured()
    print(f"{'config':26s} {'roofline':>10s}  {'engine':12s} {'tok/s':>7s} {'% roof':>7s}")
    print("-" * 70)
    for label, pb, bpp, ng in CONFIGS:
        rl = roofline(pb, bpp, ng)
        for engine, tok in MEASURED[label].items():
            pct = f"{100*tok/rl:.0f}%" if tok else "—"
            toks = f"{tok:.0f}" if tok else "—"
            flag = " <-- implausible, debug" if tok and tok / rl < 0.25 else ""
            print(f"{label:26s} {rl:8.0f}   {engine:12s} {toks:>7s} {pct:>7s}{flag}")
        print()


if __name__ == "__main__":
    main()
