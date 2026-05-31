#!/usr/bin/env bash
# Serve a model with TensorRT-LLM's OpenAI-compatible server (compiled-engine path).
# This is the command used for the measured head-to-head vs vLLM (see README).
#
# Image: nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 (TensorRT-LLM 0.20, public on NGC).
# The default (cpp) backend builds an optimized TRT engine for supported archs
# (Qwen2ForCausalLM, Llama, etc.). Qwen3ForCausalLM is not yet supported on the engine path
# in 0.20 — it only runs on `--backend pytorch` (reference flow), which is why the head-to-head
# uses Qwen2.5-7B.
set -euo pipefail
MODEL="${1:-/models/Qwen2.5-7B-Instruct}"
TP="${2:-2}"
PORT="${3:-8012}"
GPUS="${GPUS:-2,3}"            # pin to free GPUs; never the busy GPU 0
IMG=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3

docker run -d --name trtllm_serve --gpus "\"device=${GPUS}\"" \
  --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/user/models-dl:/models --entrypoint trtllm-serve \
  "$IMG" serve "$MODEL" \
  --tp_size "$TP" --host 0.0.0.0 --port "$PORT" \
  --max_seq_len 8192 --kv_cache_free_gpu_memory_fraction 0.85

echo ">> TensorRT-LLM serving $MODEL on :$PORT (TP=$TP, GPUs $GPUS)"
echo ">> tuning levers to beat vLLM (roadmap): CUDA-graph decode, paged-context FMHA,"
echo ">> prebuilt FlashInfer kernels, FP8 engine."
