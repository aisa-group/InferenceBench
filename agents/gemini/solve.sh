#!/bin/bash
set -euo pipefail

export GEMINI_SANDBOX="false"
TIMEOUT_HOURS="${NUM_HOURS:-10}"

PROMPT_FILE="${PROMPT_FILE:-/home/agent/prompt.txt}"
PROMPT="${PROMPT:-}"
if [ -f "${PROMPT_FILE}" ]; then
  PROMPT="$(cat "${PROMPT_FILE}")"
fi

set +e
timeout --signal=TERM --kill-after=30s "${TIMEOUT_HOURS}h" \
  gemini --yolo --model "$AGENT_CONFIG" --output-format stream-json -p "$PROMPT"
AGENT_RC=$?
set -e

SERVER_URL=${INFERENCE_BENCH_SERVER_URL:-http://127.0.0.1:8000}

if ! python3 - <<PY
import sys
import requests

try:
    r = requests.get("${SERVER_URL}/v1/models", timeout=2)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
PY
then
  # Kill any lingering inference engine processes before starting the saved server script.
  pkill -9 -f 'vllm' 2>/dev/null || true
  pkill -9 -f 'sglang' 2>/dev/null || true
  sleep 5
  if [[ -f /home/agent/task/run_server.sh ]]; then
    chmod +x /home/agent/task/run_server.sh || true
    nohup /home/agent/task/run_server.sh > /home/agent/task/server.log 2>&1 &
  elif [[ -f /home/agent/task/start_server.sh ]]; then
    chmod +x /home/agent/task/start_server.sh || true
    nohup /home/agent/task/start_server.sh > /home/agent/task/server.log 2>&1 &
  fi
  sleep 2
fi

REQUESTS_FILE=
SCENARIO=${INFERENCE_BENCH_SCENARIO:-}
MODEL_ID=${INFERENCE_BENCH_BASE_MODEL:-}
MODEL_SAFE="$(printf '%s' "${MODEL_ID}" | tr '/:' '_')"
if [[ -n "$SCENARIO" ]]; then
  for candidate in \
    "/home/agent/task/inference/baselines/speed/torch/${SCENARIO}/${MODEL_SAFE}/requests.jsonl" \
    "/home/agent/task/inference/baselines/speed/vllm/${SCENARIO}/${MODEL_SAFE}/requests.jsonl" \
    "/home/agent/task/inference/baselines/speed/torch/${SCENARIO}/requests.jsonl" \
    "/home/agent/task/inference/baselines/speed/vllm/${SCENARIO}/requests.jsonl"
  do
    [[ -f "$candidate" ]] && { REQUESTS_FILE=$candidate; break; }
  done
fi
EVAL_REQUESTS_ARG=()
if [[ -n "$REQUESTS_FILE" ]]; then
  echo "[agent] using requests file for preview eval: ${REQUESTS_FILE}"
  EVAL_REQUESTS_ARG=(--requests-file "$REQUESTS_FILE")
else
  echo "[agent] no requests file found for preview eval; will sample dataset."
fi

PYTHONNOUSERSITE=1 PYTHONPATH="/home/agent${PYTHONPATH:+:$PYTHONPATH}" python /home/agent/task/evaluate.py \
  --server-url "$SERVER_URL" \
  --json-output-file "${INFERENCE_BENCH_METRICS_PATH:-/home/agent/task/metrics.json}" \
  "${EVAL_REQUESTS_ARG[@]}"

exit "${AGENT_RC}"
