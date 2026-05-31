#!/usr/bin/env bash
# Launch Triton with the tensorrt_llm backend over the model repo.
set -euo pipefail
REPO="${1:-triton_model_repo}"
WORLD="${WORLD:-4}"
mpirun -n "$WORLD" --allow-run-as-root \
  tritonserver --model-repository="$REPO" \
  --grpc-port 8001 --http-port 8000 --metrics-port 8002 \
  --disable-auto-complete-config
