#!/usr/bin/env bash
# vLLM baseline on the SAME hardware/parallelism for an honest comparison.
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${2:-4}"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" \
  --port 8001 --dtype float16 --disable-log-requests
