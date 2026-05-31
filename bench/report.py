#!/usr/bin/env python3
"""Merge two benchmark JSONs into a markdown comparison table."""
import json, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default="results/report.md")
a = ap.parse_args()
rows = [json.load(open(f)) for f in a.files]
keys = ["concurrency", "throughput_tok_s", "ttft_p50_s", "ttft_p99_s", "wall_s"]
with open(a.out, "w") as w:
    w.write("# Benchmark report\n\n| stack | " + " | ".join(keys) + " |\n")
    w.write("|---|" + "---|" * len(keys) + "\n")
    for f, r in zip(a.files, rows):
        tag = f.split("/")[-1].replace(".json", "")
        w.write(f"| {tag} | " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")
print(open(a.out).read())
