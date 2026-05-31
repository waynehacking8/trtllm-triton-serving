MODEL ?= meta-llama/Llama-3.1-8B-Instruct
TP    ?= 4
PORT  ?= 8000

.PHONY: build serve serve-vllm bench bench-vllm report
build:      ; bash scripts/build_engine.sh "$(MODEL)" $(TP)
serve:      ; bash scripts/serve_triton.sh
serve-vllm: ; bash scripts/serve_vllm.sh "$(MODEL)" $(TP)
bench:      ; python bench/bench.py --base http://localhost:$(PORT) --out results/trtllm.json
bench-vllm: ; python bench/bench.py --base http://localhost:8001 --out results/vllm.json
report:     ; python bench/report.py results/trtllm.json results/vllm.json --out results/report.md
