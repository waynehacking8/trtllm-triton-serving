#!/usr/bin/env bash
# HF checkpoint -> TensorRT-LLM checkpoint -> compiled TP=2 engine.
#
# This is the script that built the measured engine (results/trtllm_compiled_bf16*).
# Run inside the TensorRT-LLM container:
#   docker run --rm --gpus '"device=4,5"' -v /path/to/models:/models --entrypoint bash \
#     nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 /models/_build_engine.sh
#
# Notes from the measured run:
#   * convert_checkpoint.py lives at /app/examples/models/core/llama/ in the 25.06 image
#     (older docs say /app/tensorrt_llm/examples/llama/ — that path does not exist here).
#   * The whole build is fast for an 8B model: ~3 min checkpoint conversion + ~35 s engine build.
#   * dtype bfloat16 matches what the PyTorch backend / vLLM use with dtype=auto for
#     Llama-3.1, so the compiled-vs-PyTorch comparison is apples-to-apples.
set -euo pipefail
MODEL_DIR="${1:-/models/Llama-3.1-8B-Instruct}"   # local HF checkpoint (no download needed)
TP="${2:-2}"
DTYPE="${DTYPE:-bfloat16}"
NAME="$(basename "$MODEL_DIR")"
CKPT="/models/_build/ckpt-${NAME}-tp${TP}-${DTYPE}"
OUT="/models/_build/engine-${NAME}-tp${TP}-${DTYPE}"

echo ">> converting HF -> TRT-LLM checkpoint (TP=$TP, $DTYPE)"
python3 /app/examples/models/core/llama/convert_checkpoint.py \
  --model_dir "$MODEL_DIR" \
  --output_dir "$CKPT" \
  --dtype "$DTYPE" --tp_size "$TP"

echo ">> building engine"
trtllm-build \
  --checkpoint_dir "$CKPT" \
  --output_dir "$OUT" \
  --gemm_plugin auto \
  --max_batch_size 256 \
  --max_num_tokens 8192 \
  --use_paged_context_fmha enable

echo ">> engine at $OUT"
echo ">> serve it (same OpenAI frontend as the PyTorch-backend runs):"
echo "   trtllm-serve serve $OUT --tokenizer $MODEL_DIR --tp_size $TP \\"
echo "     --max_batch_size 256 --max_num_tokens 8192 --kv_cache_free_gpu_memory_fraction 0.85 \\"
echo "     --extra_llm_api_options /models/trt_engine_cg.yaml   # configs/trtllm_engine_cudagraph.yaml"
