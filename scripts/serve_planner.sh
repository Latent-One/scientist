#!/usr/bin/env bash
# Start the planner: Qwen3.6-35B-A3B (BF16) on NGC vLLM 26.09, localhost:8000, model name "planner".
# Memory (measured): 64.69 GiB weights + 13.96 GiB KV cache at --gpu-memory-utilization 0.65,
# leaving room for the science workers. Load takes ~3-4 minutes. Stop: docker rm -f planner
set -euo pipefail
docker rm -f planner >/dev/null 2>&1 || true
docker run -d --name planner --gpus all --ipc=host --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.09-py3 vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name planner \
    --host 127.0.0.1 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.65 --max-num-seqs 4 \
    --enable-prefix-caching --reasoning-parser qwen3 --language-model-only
echo "loading… follow with: docker logs -f planner   (ready when: curl -s localhost:8000/v1/models)"
