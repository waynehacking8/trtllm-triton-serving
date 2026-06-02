#!/usr/bin/env bash
# vLLM serving on the RTX PRO 6000 Blackwell (sm_120) box — single GPU, TP=1.
# Used for the Phase 6 NVFP4-vs-BF16-vs-FP8 study (the repo's only sm_120 data point).
#
#   bash scripts/serve_vllm_sm120.sh <model-or-checkpoint> [port] [extra vllm args...]
#
# sm_120 box notes (documented measurement conditions, apply to EVERY precision equally):
#   * vLLM 0.18.0 + torch 2.10 on this box needs gpu-memory-utilization <= 0.70
#     (allocator headroom; also an idle 11 GB process is resident on the card).
#   * vLLM 0.18's default torch.compile path crashes on this torch version
#     ("standalone_compile does not have the attribute 'FakeTensorMode'"). The workaround
#     that KEEPS CUDA graphs (unlike --enforce-eager) is compilation mode NONE +
#     cudagraph_mode FULL — vLLM then captures full decode graphs (FULL_DECODE_ONLY with
#     the FlashAttention backend), which is what matters for the 256-token-decode workload.
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${2:-8010}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

# NOTE: request logging is off by default in vLLM 0.18 (the old --disable-log-requests
# flag was removed; its replacement --no-enable-log-requests is already the default).
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.70 \
  --max-model-len 4096 \
  --compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL"}' \
  "$@"
