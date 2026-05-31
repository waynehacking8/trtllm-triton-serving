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
import httpx

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
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/v1/completions",
                   json={"model": model, "prompt": prompt, "max_tokens": maxtok,
                         "temperature": 0, "stream": False}, timeout=120)
    dt = time.perf_counter() - t0
    return r.json()["usage"]["completion_tokens"] / dt


def acceptance(spec_base):
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
    json.dump(out, open("results/spec_decode.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
