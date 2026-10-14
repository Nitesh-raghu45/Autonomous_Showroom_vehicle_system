#!/usr/bin/env bash
# Bare-metal launch of the production model on the L4 (no Docker).  pip install vllm
set -euo pipefail
exec vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 --enable-prefix-caching
# vLLM >= 0.17 also has:  --performance-mode interactivity   (favours latency at small batch sizes; benchmark it)
