#!/usr/bin/env bash
set -euo pipefail

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${INFERENCE_BENCH_BASE_MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${INFERENCE_BENCH_MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-stats
