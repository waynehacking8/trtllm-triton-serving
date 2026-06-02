#!/usr/bin/env bash
# NVIDIA perf-overview waterfall attribution (roadmap Phase 6).
#
# Decomposes the gap between NVIDIA's published Llama-3.1-8B-FP8 1xH100 number
# (27,688 tok/s on TRT-LLM 0.20 docs, ISL/OSL 128/128) and this repo's measured
# serving number (13,828 tok/s, c128) by changing ONE knob per step:
#
#   W0  NVIDIA exact: ISL/OSL 128/128, TP1, kv 0.95, full CUDA-graph list, unbounded
#       -> FINDING: this exact config does not run on TRT-LLM 0.20 + 80GB H100. kv 0.95
#          OOMs at CUDA-graph capture; kv 0.90 with the auto-heuristic max_batch_size=4096
#          OOMs mid-benchmark on prefill activations. Reproducing the published number on
#          the published version requires capping max_batch_size at 2048 (decode batch 2048
#          still saturates at ISL/OSL 128/128). Two variants, both with that cap:
#          W0a  kv 0.95 -> 0.90, full CUDA-graph list   (relax the kv knob)
#          W0b  kv 0.95, CUDA-graph list capped at 2048 (relax the graph-pool knob)
#   W1  + OSL 128 -> 256          (this repo decodes 256 tokens per request)
#   W2  + ISL 128 -> 12           (this repo's real prompt is ~12 tokens)
#   W3  + kv -> 0.80              (this repo's serving config)
#   W4  + TP 1 -> 2               (this repo serves TP=2)
#   W5  + concurrency cap 128     (this repo benches at c128, not offline-unbounded)
#   W6  = the committed serving measurement (trtllm-serve + OpenAI streaming client,
#         real prompts): results/trtllm_llama31_fp8_tuned-c128.json -> 13,828 tok/s
#
# Run ON the GPU box (needs docker + the FP8 checkpoint under /home/user/models-dl):
#   bash waterfall.sh [output_dir]
# Produces one raw trtllm-bench log + one report JSON per step in output_dir.
set -euo pipefail

IMG=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
MODEL_DIR=/home/user/models-dl
MODEL=/models/Llama-3.1-8B-Instruct-FP8
HF_NAME=nvidia/Llama-3.1-8B-Instruct-FP8
OUT="${1:-/home/user/sa-portfolio/waterfall/results}"
WF=/home/user/sa-portfolio/waterfall          # prepare_dataset.py lives here
GPU1="${GPU1:-5}"                             # single-GPU steps
GPU2="${GPU2:-5,6}"                           # TP=2 step
mkdir -p "$OUT"

# --- NVIDIA's exact extra_llm_api_options for TRT-LLM 0.20 (perf-overview, release/0.20) ---
cat > "$OUT/llm_options_nvidia.yaml" <<'YAML'
pytorch_backend_config:
  use_cuda_graph: true
  cuda_graph_padding_enabled: true
  cuda_graph_batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
  - 256
  - 384
  - 512
  - 1024
  - 2048
  - 4096
  - 8192
YAML

# --- same, but CUDA-graph list capped at 2048 (W0b: fits beside kv 0.95 on 80 GB) ---
cat > "$OUT/llm_options_cg2048.yaml" <<'YAML'
pytorch_backend_config:
  use_cuda_graph: true
  cuda_graph_padding_enabled: true
  cuda_graph_batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 128
  - 256
  - 384
  - 512
  - 1024
  - 2048
YAML

# --- synthetic datasets (token-norm-dist, stdev 0 = fixed lengths), NVIDIA's generator ---
gen_dataset() { # isl osl num_requests outfile
  local isl=$1 osl=$2 n=$3 f=$4
  [ -s "$OUT/$f" ] && { echo "dataset $f exists, skip"; return; }
  docker run --rm --entrypoint bash -v "$MODEL_DIR:/models" -v "$WF:/wf" -v "$OUT:/out" "$IMG" -c \
    "cd /wf && python3 prepare_dataset.py --tokenizer=$MODEL --stdout token-norm-dist \
     --num-requests=$n --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 \
     > /out/$f" 2>"$OUT/${f%.jsonl}_gen.log"
  echo "generated $f ($(wc -l < "$OUT/$f") requests)"
}

# --- one waterfall step ---
run_step() { # name gpus tp kv dataset yaml extra_args...
  local name=$1 gpus=$2 tp=$3 kv=$4 dataset=$5 yaml=$6; shift 6
  if [ -s "$OUT/$name.txt" ] && grep -q "Total Output Throughput" "$OUT/$name.txt"; then
    echo "== $name already done, skip =="; return
  fi
  echo "== $name (GPUs $gpus, TP$tp, kv $kv, $dataset, $yaml) =="
  # record clock + box state alongside every measurement
  nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,memory.used \
    --format=csv > "$OUT/$name.boxstate.csv"
  # expandable_segments: the OOM error's own recommendation; newer NGC images default to it.
  # --ipc=host + memlock/stack ulimits: required for TP>1 (CUDA IPC between executor ranks --
  # without them rank 1 dies with "illegal memory access" during init).
  docker run --rm --gpus "\"device=$gpus\"" --shm-size=8g \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$MODEL_DIR:/models" -v "$OUT:/out" "$IMG" \
    trtllm-bench --model "$HF_NAME" --model_path "$MODEL" throughput \
      --dataset "/out/$dataset" --backend pytorch \
      --tp "$tp" --kv_cache_free_gpu_mem_fraction "$kv" \
      --extra_llm_api_options "/out/$yaml" \
      --report_json "/out/$name.report.json" \
      "$@" > "$OUT/$name.txt" 2>&1 || { echo "FAILED $name"; tail -5 "$OUT/$name.txt"; return 1; }
  grep -E "Total Output Throughput|Total Token Throughput|Request Throughput" "$OUT/$name.txt" || true
}

# datasets: num_requests scaled down as total tokens/request grows (NVIDIA does the same;
# steady-state throughput is unaffected once runtime >> warmup)
gen_dataset 128 128 30000 ds_128_128.jsonl
gen_dataset 128 256 15000 ds_128_256.jsonl
gen_dataset  12 256 15000 ds_12_256.jsonl

# MBS: required deviation on 0.20/80GB (see header). Constant across every step below, so
# it cancels out of all step-to-step attributions.
MBS=2048

# W0: NVIDIA's exact config OOMs on 0.20 + 80GB (documented above); keep the failed log as
# evidence, then run the two minimal relaxations and use the better one as the reference.
run_step W0_nvidia_exact "$GPU1" 1 0.95 ds_128_128.jsonl llm_options_nvidia.yaml || true
run_step W0a_kv090       "$GPU1" 1 0.90 ds_128_128.jsonl llm_options_nvidia.yaml --max_batch_size $MBS
run_step W0b_cg2048      "$GPU1" 1 0.95 ds_128_128.jsonl llm_options_cg2048.yaml --max_batch_size $MBS

# W1+: carry forward the W0a convention (kv 0.90, full CG list); every step changes ONE knob.
run_step W1_osl256       "$GPU1" 1 0.90 ds_128_256.jsonl llm_options_nvidia.yaml --max_batch_size $MBS
run_step W2_isl12        "$GPU1" 1 0.90 ds_12_256.jsonl  llm_options_nvidia.yaml --max_batch_size $MBS
run_step W3_kv080        "$GPU1" 1 0.80 ds_12_256.jsonl  llm_options_nvidia.yaml --max_batch_size $MBS

# W4: cap concurrency at 128 (this repo benches at c128, not offline-unbounded). Still TP1.
#
# NOTE / documented limitation: the original plan put TP 1->2 here, measured with trtllm-bench.
# That is NOT POSSIBLE on TRT-LLM 0.20: trtllm-bench --backend pytorch --tp 2 crashes
# reproducibly (rank 1 "illegal memory access" during executor init) across GPU pairs
# (2,3 and 5,6), with/without --ipc=host, and with/without large CUDA-graph lists — the failed
# logs are kept as W4_tp2.txt evidence. trtllm-SERVE TP=2 works fine (this repo's committed
# serving numbers). So the TP step moves to the serving side of the chain:
#   W4 (this step)  offline, TP1, c128
#   W5              = the repo's committed trtllm-serve TP1... (measured by bench/bench.py)
#   W6              = the repo's committed trtllm-serve TP2 c128 number (13,828 tok/s)
run_step W4_conc128      "$GPU1" 1 0.80 ds_12_256.jsonl  llm_options_nvidia.yaml --max_batch_size $MBS --concurrency 128

echo "ALL WATERFALL STEPS DONE -> $OUT"
echo "(W5 = trtllm-serve TP1 measurement: scripts/serve_trtllm.sh + bench/bench.py c128 - run separately)"
