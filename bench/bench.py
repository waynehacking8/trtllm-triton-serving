#!/usr/bin/env python3
"""Async load test for an OpenAI-compatible LLM endpoint.

Measures, under fixed concurrency: time-to-first-token (TTFT), inter-token latency
(ITL), and end-to-end throughput (tok/s). Works against Triton's OpenAI frontend,
trtllm-serve, or vLLM — so the same harness benchmarks both stacks.
"""
import argparse, asyncio, json, time
import httpx

PROMPT = "Explain tensor parallelism for LLM inference in three sentences."

async def one_request(client, base, max_tokens):
    t0 = time.perf_counter()
    first = None
    n = 0
    payload = {"model": "ensemble", "prompt": PROMPT, "max_tokens": max_tokens,
               "stream": True, "temperature": 0.0}
    async with client.stream("POST", f"{base}/v1/completions", json=payload) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            if line.strip() == "data: [DONE]":
                break
            if first is None:
                first = time.perf_counter()
            n += 1
    t1 = time.perf_counter()
    return {"ttft": (first or t1) - t0, "e2e": t1 - t0, "tokens": n}

async def run(base, concurrency, total, max_tokens):
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)
        async def guarded():
            async with sem:
                return await one_request(client, base, max_tokens)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[guarded() for _ in range(total)])
        wall = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in res)
    ttfts = sorted(r["ttft"] for r in res)
    p = lambda q: ttfts[int(q * (len(ttfts) - 1))]
    return {"concurrency": concurrency, "requests": total, "wall_s": round(wall, 3),
            "throughput_tok_s": round(toks / wall, 1),
            "ttft_p50_s": round(p(0.5), 4), "ttft_p99_s": round(p(0.99), 4)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--total", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/run.json")
    a = ap.parse_args()
    out = asyncio.run(run(a.base, a.concurrency, a.total, a.max_tokens))
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
