#!/usr/bin/env bash
# Launch Triton with the tensorrt_llm backend serving the compiled engine (TP=2).
#
# The model repository setup + tritonserver launch live in scripts/setup_triton_repo.sh
# (run inside the container). This wrapper starts the container detached.
set -euo pipefail
GPUS="${GPUS:-2,3}"            # pin to free GPUs; never the busy GPU 0
MODELS_DIR="${MODELS_DIR:-/home/user/models-dl}"
IMG=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3

# the in-container script needs to be reachable under /models
cp "$(dirname "$0")/setup_triton_repo.sh" "$MODELS_DIR/_setup_triton_repo.sh"

docker run -d --name triton_trtllm --gpus "\"device=${GPUS}\"" \
  --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODELS_DIR":/models --entrypoint bash "$IMG" /models/_setup_triton_repo.sh

echo ">> Triton (tensorrt_llm backend) starting on HTTP :8020 / gRPC :8021 / metrics :8022"
echo ">> ready check: curl http://localhost:8020/v2/health/ready"
