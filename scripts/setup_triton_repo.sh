#!/usr/bin/env bash
# Build a Triton model repository for the tensorrt_llm backend from the in-image template,
# then launch tritonserver (TP=2 via MPI).
#
# This is the script behind the deployed-and-smoke-tested Triton stack:
#   ensemble + preprocessing + tensorrt_llm + postprocessing + tensorrt_llm_bls
#   HTTP 8020 / gRPC 8021 / metrics 8022  (offset to avoid colliding with other services)
#
# Run detached:
#   docker run -d --name triton_trtllm --gpus '"device=2,3"' --network host --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 -v /path/to/models:/models \
#     --entrypoint bash nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3 \
#     /models/_setup_triton_repo.sh
#
# Smoke test:
#   curl http://HOST:8020/v2/health/ready
#   curl http://HOST:8020/v2/models/ensemble/generate \
#     -d '{"text_input": "The capital of France is", "max_tokens": 16, "temperature": 0}'
#
# Measured note: single-stream decode through the ensemble (~187 tok/s) is ~15% below
# trtllm-serve on the same engine (~220 tok/s) — the ensemble's Python pre/post-processing
# hop is not free. Use the BLS model or the OpenAI frontend for production latency paths.
set -ex
REPO="${REPO:-/models/_build/triton_repo}"
ENGINE="${ENGINE:-/models/_build/engine-Llama-3.1-8B-Instruct-tp2-bfloat16}"
TOK="${TOK:-/models/Llama-3.1-8B-Instruct}"
WORLD="${WORLD:-2}"
FILL="python3 /app/tools/fill_template.py"

rm -rf "$REPO"
cp -r /app/all_models/inflight_batcher_llm "$REPO"

$FILL -i "$REPO/tensorrt_llm/config.pbtxt" \
"triton_backend:tensorrtllm,triton_max_batch_size:256,decoupled_mode:True,max_beam_width:1,\
engine_dir:${ENGINE},max_tokens_in_paged_kv_cache:,max_attention_window_size:,sink_token_length:,\
kv_cache_free_gpu_mem_fraction:0.85,exclude_input_in_output:True,enable_kv_cache_reuse:False,\
batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,enable_chunked_context:False,\
encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"

$FILL -i "$REPO/preprocessing/config.pbtxt" \
"tokenizer_dir:${TOK},triton_max_batch_size:256,preprocessing_instance_count:1,max_queue_delay_microseconds:0"

$FILL -i "$REPO/postprocessing/config.pbtxt" \
"tokenizer_dir:${TOK},triton_max_batch_size:256,postprocessing_instance_count:1"

$FILL -i "$REPO/ensemble/config.pbtxt" "triton_max_batch_size:256,logits_datatype:TYPE_FP32"

$FILL -i "$REPO/tensorrt_llm_bls/config.pbtxt" \
"triton_max_batch_size:256,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32"

python3 /app/scripts/launch_triton_server.py --world_size "$WORLD" --model_repo "$REPO" \
  --http_port 8020 --grpc_port 8021 --metrics_port 8022 --force
# launch_triton_server.py backgrounds mpirun; keep the container alive
sleep 5
tail -f /dev/null
