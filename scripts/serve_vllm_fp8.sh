#!/usr/bin/env bash
# vLLM FP8 serve command for the study-2 head-to-head (the "headline" FP8 comparison vs TRT-LLM).
#
# RECONSTRUCTED from committed run parameters — this exact script was not committed at the time the
# FP8 head-to-head results were produced (commit 962291b); it is rebuilt here from what the repo
# documents so the comparison is reproducible. Sources for each parameter:
#   * model  nvidia/Llama-3.1-8B-Instruct-FP8  — results/report.md:19, README §2 ("the headline"),
#            commit 962291b ("FP8-vs-FP8 head-to-head"). A pre-quantized FP8 checkpoint, so vLLM
#            auto-detects the quantization scheme from the checkpoint config; there is NO
#            on-the-fly --quantization flag for this run.
#   * TP=2   — README Hardware ("TP=2 for the 8B head-to-heads") and §2 heading.
#   * port 8001 — the H100-box vLLM port committed in scripts/serve_vllm.sh.
#
# NOT reconstructable from committed files (left at vLLM defaults for these runs, per the generic
# launcher scripts/serve_vllm.sh — DO NOT treat the values below as measured/verified):
#   * --gpu-memory-utilization : not recorded → vLLM default (0.90 on this vLLM version).
#   * --max-num-seqs           : not recorded → vLLM default.
#   * --dtype                  : not recorded for the FP8 run. The generic serve_vllm.sh hard-codes
#                                float16, but for an FP8 checkpoint vLLM's "auto" is the natural
#                                choice; the committed data does not pin which was used, so this is
#                                left unset (auto). Set explicitly if you need bit-exact repro.
#
# i.e. unlike scripts/serve_trtllm.sh (which carries explicit --max_batch_size / --max_num_tokens /
# --kv_cache_free_gpu_memory_fraction tuning), the vLLM side of the head-to-head ran near-default.
# See the serve-config-asymmetry note in README §2.
set -euo pipefail
MODEL="${1:-nvidia/Llama-3.1-8B-Instruct-FP8}"   # documented: report.md:19 / commit 962291b
TP="${2:-2}"                                      # documented: README Hardware + §2
PORT="${3:-8001}"                                 # documented: scripts/serve_vllm.sh

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --disable-log-requests
# All other knobs (gpu-memory-utilization, max-num-seqs, dtype) intentionally omitted = vLLM
# defaults, matching how these head-to-head numbers were produced. See header for provenance.
