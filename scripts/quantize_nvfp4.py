#!/usr/bin/env python3
"""Quantize an HF causal-LM checkpoint to NVFP4 (W4A4) with TensorRT-Model-Optimizer.

Produces a vLLM-loadable checkpoint (quant_method=modelopt, quant_algo=NVFP4) for the
roadmap Phase 6 "NVFP4 W4A4 serving on RTX PRO 6000 (sm_120)" experiment.

    python scripts/quantize_nvfp4.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --out checkpoints/Llama-3.1-8B-Instruct-NVFP4

PTQ calibration: 512 cnn_dailymail articles x 512 tokens (ModelOpt's documented default
recipe). lm_head stays unquantized (excluded by NVFP4_DEFAULT_CFG).
"""
import argparse
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

CALIB_SAMPLES = 512
CALIB_SEQLEN = 512
CALIB_BATCH = 8


def build_calib_loop(model, tokenizer):
    """Forward-pass calibration loop over cnn_dailymail (ModelOpt's standard PTQ set)."""
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    texts = [ds[i]["article"] for i in range(CALIB_SAMPLES)]

    def calib(_model):
        for i in range(0, CALIB_SAMPLES, CALIB_BATCH):
            batch = tokenizer(
                texts[i : i + CALIB_BATCH],
                return_tensors="pt",
                max_length=CALIB_SEQLEN,
                truncation=True,
                padding="max_length",
            ).to(_model.device)
            with torch.no_grad():
                _model(**batch)
            if (i // CALIB_BATCH) % 8 == 0:
                print(f"  calib {i + CALIB_BATCH}/{CALIB_SAMPLES}", flush=True)

    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", default="checkpoints/Llama-3.1-8B-Instruct-NVFP4")
    args = ap.parse_args()

    t0 = time.time()
    print(f">> loading {args.model} (bf16)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print(">> PTQ calibration + NVFP4 quantization (mtq.NVFP4_DEFAULT_CFG)", flush=True)
    calib_loop = build_calib_loop(model, tokenizer)
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=calib_loop)
    mtq.print_quant_summary(model)

    print(f">> exporting HF checkpoint -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=args.out)
    tokenizer.save_pretrained(args.out)
    print(f">> done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
