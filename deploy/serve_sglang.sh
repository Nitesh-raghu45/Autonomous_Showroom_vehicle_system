#!/usr/bin/env bash
# Alternative to vLLM.  pip install "sglang[all]"
set -euo pipefail
exec python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --host 0.0.0.0 --port 8001 --context-length 8192 --mem-fraction-static 0.85
