#!/usr/bin/env bash
# Serve a model with TensorRT-LLM's OpenAI-compatible server, CUDA graphs CORRECTLY enabled.
# This is the command used for the measured head-to-heads vs vLLM (see README).
#
# Image: nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 (TensorRT-LLM 0.20, public on NGC).
# We use --backend pytorch with cuda graphs (configs/trtllm_pytorch_cudagraph.yaml). CUDA graphs
# are the single biggest lever for single-stream decode latency: 162 -> 374 tok/s on
# Llama-3.1-8B FP8 TP=2. The config MUST be nested under pytorch_backend_config or it is
# silently ignored — verify the log prints use_cuda_graph=True.
set -euo pipefail
MODEL="${1:-/models/Llama-3.1-8B-Instruct-FP8}"
TP="${2:-2}"
PORT="${3:-8012}"
GPUS="${GPUS:-2,3}"            # pin to free GPUs; never the busy GPU 0
IMG=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
CFG="${CFG:-/models/trt_cudagraph.yaml}"   # contents = configs/trtllm_pytorch_cudagraph.yaml

docker run -d --name trtllm_serve --gpus "\"device=${GPUS}\"" \
  --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/user/models-dl:/models --entrypoint trtllm-serve \
  "$IMG" serve "$MODEL" \
  --backend pytorch --tp_size "$TP" --host 0.0.0.0 --port "$PORT" \
  --max_batch_size 256 --max_num_tokens 8192 --max_seq_len 8192 \
  --kv_cache_free_gpu_memory_fraction 0.85 --extra_llm_api_options "$CFG"

echo ">> TensorRT-LLM serving $MODEL on :$PORT (TP=$TP, GPUs $GPUS), CUDA graphs ON"
echo ">> verify: docker logs trtllm_serve | grep -oE 'use_cuda_graph=[A-Za-z]+'  (must be True)"
