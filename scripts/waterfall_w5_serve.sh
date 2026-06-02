#!/usr/bin/env bash
# Waterfall step W5: the serving-stack step (roadmap Phase 6).
#
# W4 measured offline trtllm-bench at TP1/c128. W5 measures the SAME model/workload through
# the production serving path this repo's committed numbers use: trtllm-serve (OpenAI HTTP
# streaming) + the serve-tuned config (max_batch 256, kv 0.85, CG-256, chunked prefill,
# MAX_UTILIZATION) — still TP1, still c128, still ~12-token prompt / 256-token decode.
# The W4 -> W5 delta is therefore "what the serving deployment costs vs the offline harness".
# The W5 -> W6 delta (W6 = results/trtllm_llama31_fp8_tuned-c128.json, TP2) isolates TP 1->2.
#
# Run ON the GPU box:  GPUS=2 bash waterfall_w5_serve.sh
# Then from the bench client host:
#   python3 bench/bench.py --base http://<box>:8012 --model <served-name> \
#     --concurrency 128 --total 1024 --max-tokens 256 \
#     --out results/waterfall/W5_serve_tp1-c128.json
set -euo pipefail

MODEL=/models/Llama-3.1-8B-Instruct-FP8
PORT=8012
GPUS="${GPUS:-2}"
IMG=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
CFG=/models/trt_tuned.yaml          # contents = configs/trtllm_pytorch_tuned.yaml

docker rm -f trtllm_serve_w5 2>/dev/null || true
docker run -d --name trtllm_serve_w5 --gpus "\"device=${GPUS}\"" \
  --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/user/models-dl:/models --entrypoint trtllm-serve \
  "$IMG" serve "$MODEL" \
  --backend pytorch --tp_size 1 --host 0.0.0.0 --port "$PORT" \
  --max_batch_size 256 --max_num_tokens 8192 --max_seq_len 8192 \
  --kv_cache_free_gpu_memory_fraction 0.85 --extra_llm_api_options "$CFG"

echo ">> trtllm-serve TP1 starting on :$PORT (GPU $GPUS). Waiting for ready..."
for i in $(seq 1 120); do
  if curl -s -m 2 "http://localhost:$PORT/v1/models" | grep -q '"id"'; then
    echo ">> READY. Served models:"; curl -s "http://localhost:$PORT/v1/models"
    echo; echo ">> verify CUDA graphs: docker logs trtllm_serve_w5 2>&1 | grep -o 'use_cuda_graph=[A-Za-z]*'"
    docker logs trtllm_serve_w5 2>&1 | grep -oE "use_cuda_graph=[A-Za-z]+|enable_chunked_prefill=[A-Za-z]+|MAX_UTILIZATION" | sort -u
    exit 0
  fi
  sleep 5
done
echo "TIMEOUT waiting for serve"; docker logs trtllm_serve_w5 2>&1 | tail -20; exit 1
