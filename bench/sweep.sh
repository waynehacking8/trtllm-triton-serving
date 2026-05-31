#!/usr/bin/env bash
# Concurrency sweep against one endpoint -> results/<tag>-c<N>.json
set -euo pipefail
BASE="${1:-http://localhost:8000}"
TAG="${2:-trtllm}"
for C in 1 4 16 32 64 128; do
  python bench/bench.py --base "$BASE" --concurrency "$C" --total $((C*8)) \
    --out "results/${TAG}-c${C}.json"
done
