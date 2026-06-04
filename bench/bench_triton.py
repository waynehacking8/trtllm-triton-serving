#!/usr/bin/env python3
"""Async load test for Triton's tensorrt_llm-backend ENSEMBLE path (Triton protocol).

Mirrors bench/bench.py exactly — same prompt, same forced-256-token methodology, same
metrics (TTFT / ITL / throughput) and the same output JSON schema — but speaks Triton's
/v2/models/<model>/generate_stream SSE protocol instead of the OpenAI API, so the ensemble
(preprocessing -> tensorrt_llm -> postprocessing) is measured WITHOUT any extra frontend hop.

This closes the roadmap Phase 5 item: study 8 measured the ensemble at c1 only (~187 tok/s,
~15% below trtllm-serve); this sweeps c1->c128 to see whether the Python pre/post-processing
hop amortizes under concurrency or becomes the bottleneck.
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

# identical to bench/bench.py
PROMPT = "Explain tensor parallelism for LLM inference in three sentences."


async def one_request(client, base, max_tokens, model):
    t0 = time.perf_counter()
    first = None
    n = 0
    # min_tokens + the engine's exclude_input_in_output force every request to decode exactly
    # max_tokens tokens (TRT-LLM suppresses EOS until min_tokens) — the same controlled
    # methodology as bench.py's ignore_eos+min_tokens, expressed in ensemble parameters.
    payload = {"text_input": PROMPT, "max_tokens": max_tokens, "min_tokens": max_tokens,
               "temperature": 0.0, "stream": True}
    async with client.stream("POST", f"{base}/v2/models/{model}/generate_stream",
                             json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            if first is None:
                first = time.perf_counter()
            n += 1
    t1 = time.perf_counter()
    ttft = (first or t1) - t0
    itl = ((t1 - first) / (n - 1)) if (first and n > 1) else 0.0
    return {"ttft": ttft, "e2e": t1 - t0, "tokens": n, "itl": itl}


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))] if s else 0.0


async def run(base, concurrency, total, max_tokens, model):
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded():
            async with sem:
                return await one_request(client, base, max_tokens, model)

        t0 = time.perf_counter()
        res = await asyncio.gather(*[guarded() for _ in range(total)])
        wall = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in res)
    ttfts = [r["ttft"] for r in res]
    itls = [r["itl"] for r in res if r["itl"] > 0]
    return {
        "model": model,
        "concurrency": concurrency,
        "requests": total,
        "wall_s": round(wall, 3),
        "throughput_tok_s": round(toks / wall, 1),
        "out_tokens_mean": round(toks / len(res), 1),
        "ttft_p50_s": round(pct(ttfts, 0.50), 4),
        "ttft_p99_s": round(pct(ttfts, 0.99), 4),
        "itl_p50_ms": round(pct(itls, 0.50) * 1000, 2),
        "itl_p99_ms": round(pct(itls, 0.99) * 1000, 2),
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 2) if itls else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8020")
    ap.add_argument("--model", default="ensemble", help="Triton model name (ensemble or tensorrt_llm_bls)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--total", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/run.json")
    args = ap.parse_args()

    res = asyncio.run(run(args.base, args.concurrency, args.total, args.max_tokens, args.model))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
