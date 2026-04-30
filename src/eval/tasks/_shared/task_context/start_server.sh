#!/usr/bin/env bash
set -euo pipefail

# Start an OpenAI-compatible inference server on $HOST:$PORT.
#
# This script is intentionally engine-agnostic: you may use vLLM, SGLang, or
# any other engine, as long as it exposes an OpenAI-compatible API at:
#   - GET  /v1/models
#   - POST /v1/chat/completions  (streaming enabled)
#
# How to use:
#   - Edit this file and write your engine launch command.
#   - Keep the final process in the foreground and prefer ending with `exec ...`.
#   - Use ./test_server.sh during development if you want to launch the server
#     under the same supervised scaffold the benchmark harness uses.
#
# IMPORTANT:
# - Do not daemonize the final server with nohup/setsid/trailing `&`.
# - You MUST use $HOST and $PORT (not hardcoded values).
# - The benchmark will restart this script in a fresh container for final eval.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

echo "=== Inference server ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "HF_HOME=${HF_HOME}"
echo "============================"

echo "ERROR: start_server.sh has no engine configured."
echo "Edit this file and replace this stub with your launch command."
exit 2
