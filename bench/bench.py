#!/usr/bin/env python3
"""Async load test for an OpenAI-compatible LLM endpoint.

Measures, under fixed concurrency: time-to-first-token (TTFT), inter-token latency
(ITL), and end-to-end throughput (tok/s). Works against Triton's OpenAI frontend,
trtllm-serve, or vLLM — so the same harness benchmarks both stacks.
"""
import argparse, asyncio, json, time, os, statistics
import httpx

PROMPT = "Explain tensor parallelism for LLM inference in three sentences."


async def one_request(client, base, max_tokens, model):
    t0 = time.perf_counter()
    first = None
    n = 0
    payload = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens,
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
    ttft = (first or t1) - t0
    # mean inter-token latency for this request (decode phase only)
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
    return {"concurrency": concurrency, "requests": total, "wall_s": round(wall, 3),
            "throughput_tok_s": round(toks / wall, 1),
            "out_tokens_mean": round(toks / len(res), 1),
            "ttft_p50_s": round(pct(ttfts, 0.5), 4), "ttft_p99_s": round(pct(ttfts, 0.99), 4),
            "itl_p50_ms": round(1000 * pct(itls, 0.5), 2),
            "itl_p99_ms": round(1000 * pct(itls, 0.99), 2),
            "itl_mean_ms": round(1000 * statistics.mean(itls), 2) if itls else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="ensemble", help="served model name (vLLM) or 'ensemble' (Triton)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--total", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/run.json")
    a = ap.parse_args()
    out = asyncio.run(run(a.base, a.concurrency, a.total, a.max_tokens, a.model))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
