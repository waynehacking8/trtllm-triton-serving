#!/usr/bin/env bash
# HF checkpoint -> TensorRT-LLM checkpoint -> TP=4 engine.
# Run inside the TensorRT-LLM container, e.g.:
#   docker run --gpus all -it --rm -v $PWD:/work nvcr.io/nvidia/tensorrt_llm/release:latest
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${2:-4}"
DTYPE="${DTYPE:-float16}"               # set to 'fp8' for the Phase-2 quantized engine
OUT="engines/${MODEL##*/}-tp${TP}-${DTYPE}"

echo ">> downloading $MODEL"
huggingface-cli download "$MODEL" --local-dir "hf_models/${MODEL##*/}"

echo ">> converting HF -> TRT-LLM checkpoint (TP=$TP)"
python /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
  --model_dir "hf_models/${MODEL##*/}" \
  --output_dir "checkpoints/${MODEL##*/}-tp${TP}" \
  --dtype "$DTYPE" --tp_size "$TP"

echo ">> building engine"
trtllm-build \
  --checkpoint_dir "checkpoints/${MODEL##*/}-tp${TP}" \
  --output_dir "$OUT" \
  --gemm_plugin auto \
  --max_batch_size 256 \
  --max_num_tokens 8192 \
  --use_paged_context_fmha enable

echo ">> engine at $OUT"
