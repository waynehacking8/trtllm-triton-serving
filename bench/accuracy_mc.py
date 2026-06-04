#!/usr/bin/env python3
"""Multiple-choice accuracy spot-check against an OpenAI-compatible endpoint.

Used for the Phase 6 NVFP4 study: quantization accuracy parity is checked on an
ARC-Challenge subset (GPQA is HF-gated; the roadmap allows either), generation-based:
the model is prompted with the question + lettered choices and must answer with a letter.

Note on methodology: this is *generation-based* MC eval (what an API endpoint allows),
not loglikelihood-based like lm-eval-harness — absolute scores read lower than published
loglikelihood numbers; only the delta between precisions served identically is meaningful.

    python bench/accuracy_mc.py --base http://localhost:8010 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --n 300 --out results/acc_arc_sm120_bf16.json
"""
import argparse
import asyncio
import json
import os
import re

import httpx
from datasets import load_dataset

PROMPT_TMPL = (
    "The following is a multiple-choice science question. "
    "Answer with the letter of the correct choice only.\n\n"
    "Question: {q}\n{choices}\nAnswer:"
)


def format_choices(labels, texts):
    return "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))


async def ask(client, base, model, prompt, sem):
    async with sem:
        r = await client.post(
            f"{base}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 4, "temperature": 0.0},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]


async def run(base, model, n, concurrency):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = [ds[i] for i in range(min(n, len(ds)))]

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        for it in items:
            prompt = PROMPT_TMPL.format(
                q=it["question"],
                choices=format_choices(it["choices"]["label"], it["choices"]["text"]),
            )
            tasks.append(ask(client, base, model, prompt, sem))
        outs = await asyncio.gather(*tasks)

    correct = 0
    unparsed = 0
    for it, out in zip(items, outs):
        m = re.search(r"\b([A-E1-5])\b", out.strip())
        if not m:
            unparsed += 1
            continue
        if m.group(1) == it["answerKey"]:
            correct += 1
    return {
        "dataset": "ai2_arc/ARC-Challenge[test]",
        "method": "generation-based MC (letter match), temperature 0",
        "n": len(items),
        "correct": correct,
        "unparsed": unparsed,
        "accuracy": round(correct / len(items), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8010")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = asyncio.run(run(a.base, a.model, a.n, a.concurrency))
    res["model"] = a.model
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
