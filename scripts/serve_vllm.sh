#!/usr/bin/env bash
# vLLM baseline on the SAME hardware/parallelism for an honest comparison.
#
# NOTE: this is a GENERIC launcher (BF16/float16, TP defaulting to 4), NOT the exact invocation
# used for the committed FP8 head-to-head (study 2). That run served nvidia/Llama-3.1-8B-Instruct-FP8
# at TP=2 on port 8001 with the other knobs left at vLLM defaults — see scripts/serve_vllm_fp8.sh,
# which reconstructs it from the documented run parameters and lists exactly which flags are
# documented vs left at defaults.
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${2:-4}"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --tensor-parallel-size "$TP" \
  --port 8001 --dtype float16 --disable-log-requests
