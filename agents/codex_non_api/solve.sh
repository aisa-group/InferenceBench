#!/bin/bash
set -euo pipefail

unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Clear API keys so the CLI uses the ChatGPT Pro auth from auth.json
export CODEX_API_KEY=""
export OPENAI_API_KEY=""

# Ensure auth.json is in ~/.codex/ where codex expects it.
# run_task.sh copies it there from agents/<agent>/auth.json before this script runs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/.codex
if [ ! -f ~/.codex/auth.json ]; then
  # Fallback: copy from script directory (when running outside run_task.sh).
  if [ -f "${SCRIPT_DIR}/auth.json" ]; then
    cp "${SCRIPT_DIR}/auth.json" ~/.codex/auth.json
  else
    echo "ERROR: No auth.json found in ~/.codex/ or ${SCRIPT_DIR}/"
    exit 1
  fi
fi

# Parse reasoning effort suffix from AGENT_CONFIG (e.g. gpt-5.4-high -> model=gpt-5.4, effort=high)
CODEX_MODEL="${AGENT_CONFIG:-}"
CODEX_REASONING_EFFORT=""
case "${CODEX_MODEL}" in
  *-high) CODEX_REASONING_EFFORT="high"; CODEX_MODEL="${CODEX_MODEL%-high}" ;;
  *-med)  CODEX_REASONING_EFFORT="medium"; CODEX_MODEL="${CODEX_MODEL%-med}" ;;
  *-low)  CODEX_REASONING_EFFORT="low"; CODEX_MODEL="${CODEX_MODEL%-low}" ;;
esac

# Write config.toml from scratch (avoids duplicate keys on resume runs)
{
  printf 'forced_login_method = "chatgpt"\n'
  if [ -n "${CODEX_REASONING_EFFORT}" ]; then
    printf 'model_reasoning_effort = "%s"\n' "${CODEX_REASONING_EFFORT}"
  fi
} > ~/.codex/config.toml

export BASH_MAX_TIMEOUT_MS=36000000

TIMEOUT_HOURS=${NUM_HOURS:-10}

echo "[agent] $(date --iso-8601=seconds) starting codex (non-api) agent"
echo "[agent] AGENT_CONFIG=${AGENT_CONFIG:-} (model=${CODEX_MODEL}, reasoning_effort=${CODEX_REASONING_EFFORT:-default})"
if command -v codex >/dev/null 2>&1; then
  echo "[agent] codex=$(command -v codex)"
else
  echo "[agent] codex not found on PATH"
fi

PROMPT_FILE=${PROMPT_FILE:-/home/agent/prompt.txt}
PROMPT=${PROMPT:-}
[[ -f "$PROMPT_FILE" ]] && PROMPT="$(cat "$PROMPT_FILE")"
PROMPT_BYTES="$(printf '%s' "$PROMPT" | wc -c | tr -d ' ')"
PROMPT_LINES="$(printf '%s' "$PROMPT" | awk 'END{print NR}')"
echo "[agent] PROMPT_FILE=${PROMPT_FILE} bytes=${PROMPT_BYTES} lines=${PROMPT_LINES}"
echo "[agent] PROMPT_FIRST_LINE=$(printf '%s' "$PROMPT" | head -n 1)"

echo "[agent] RESUME_PROMPT_FIRST_LINE=$(printf '%s' "${RESUME_PROMPT:-}" | head -n 1)"
RUN_INDEX=${INFERENCE_BENCH_RUN_INDEX:-1}
echo "[agent] RUN_INDEX=${RUN_INDEX}"

START_TS=$(date +%s)
if [[ -n "${INFERENCE_BENCH_REMAINING_SECONDS:-}" && "${INFERENCE_BENCH_REMAINING_SECONDS}" -gt 0 ]]; then
    TOTAL_TIMEOUT_SECONDS="${INFERENCE_BENCH_REMAINING_SECONDS}"
else
    TOTAL_TIMEOUT_SECONDS="$(python3 - <<PY
h = float("${TIMEOUT_HOURS}")
print(max(1, int(h * 3600)))
PY
)"
fi
DEADLINE_TS=$((START_TS + TOTAL_TIMEOUT_SECONDS))
echo "[agent] TIMEOUT_HOURS=${TIMEOUT_HOURS} total_timeout_seconds=${TOTAL_TIMEOUT_SECONDS}"

time_left_seconds() {
  local left=$((DEADLINE_TS - $(date +%s)))
  (( left < 0 )) && left=0
  printf '%s' "$left"
}

make_resume_prompt() {
  local remaining
  remaining="$(time_left_seconds)"
  local mins=$(( remaining / 60 ))
  if [ -n "${RESUME_PROMPT:-}" ]; then
    printf '%s' "${RESUME_PROMPT}"
  else
    printf 'Continue where you left off and complete the task. You have approximately %d minutes (%d seconds) of wall-clock time remaining. Make sure you utilize this time fully to achieve the best results you can. Do not ask for user feedback.' \
      "${mins}" "${remaining}"
  fi
}

run_codex_with_remaining_budget() {
  local phase=$1
  shift
  local remaining
  remaining="$(time_left_seconds)"
  if (( remaining <= 0 )); then
    echo "[agent] ${phase}: no budget left, skipping"
    return 124
  fi
  echo "[agent] ${phase}: launching with remaining budget ${remaining}s"
  timeout --signal=TERM --kill-after=30s "${remaining}s" codex "$@"
}

set +e
START_PROMPT=${PROMPT:-$RESUME_PROMPT}
echo "[agent] phase 1/2: starting codex exec run"
if (( RUN_INDEX > 1 )); then
  echo "[agent] rerun detected; attempting to resume previous codex session"
  run_codex_with_remaining_budget exec \
    --search exec resume --last --skip-git-repo-check --yolo --model "$CODEX_MODEL" "$(make_resume_prompt)"
  EXEC_RC=$?
  if (( EXEC_RC != 0 && EXEC_RC != 124 )); then
    echo "[agent] resume-on-rerun failed rc=${EXEC_RC}; falling back to fresh exec prompt"
    run_codex_with_remaining_budget exec \
      --search exec --skip-git-repo-check --yolo --model "$CODEX_MODEL" "$START_PROMPT"
    EXEC_RC=$?
  fi
else
  run_codex_with_remaining_budget exec \
    --search exec --skip-git-repo-check --yolo --model "$CODEX_MODEL" "$START_PROMPT"
  EXEC_RC=$?
fi
AGENT_RC=$EXEC_RC

REMAINING_AFTER_EXEC="$(time_left_seconds)"
echo "[agent] exec finished rc=${EXEC_RC} remaining_seconds=${REMAINING_AFTER_EXEC}"
if (( EXEC_RC != 124 && REMAINING_AFTER_EXEC > 0 )); then
  RESUME_ROUND=1
  while :; do
    CURRENT_REMAINING="$(time_left_seconds)"
    if (( CURRENT_REMAINING <= 0 )); then
      echo "[agent] no time left before resume round ${RESUME_ROUND}; stopping resume loop"
      break
    fi
    echo "[agent] phase 2/2 round ${RESUME_ROUND}: continuing conversation with codex resume"
    run_codex_with_remaining_budget resume \
      --search exec resume --last --skip-git-repo-check --yolo --model "$CODEX_MODEL" "$(make_resume_prompt)"
    AGENT_RC=$?
    CURRENT_REMAINING="$(time_left_seconds)"
    echo "[agent] resume round ${RESUME_ROUND} finished rc=${AGENT_RC} remaining_seconds=${CURRENT_REMAINING}"
    if (( AGENT_RC == 124 )); then
      echo "[agent] resume used remaining budget; proceeding to eval"
      break
    fi
    if (( AGENT_RC != 0 )); then
      echo "[agent] resume failed rc=${AGENT_RC}; proceeding to eval"
      break
    fi
    RESUME_ROUND=$((RESUME_ROUND + 1))
  done
else
  echo "[agent] skipping resume (exec used full budget or no time left)"
fi
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
  # Kill any lingering inference engine processes (including worker subprocesses) before starting fresh.
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
  echo "[agent] waiting for server at ${SERVER_URL}"
  _wait_end=$(( $(date +%s) + 900 ))
  while (( $(date +%s) < _wait_end )); do
    if curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; then
      echo "[agent] server is up"
      break
    fi
    sleep 10
  done
  if ! curl -sf "${SERVER_URL}/v1/models" >/dev/null 2>&1; then
    echo "[agent] WARNING: server did not become ready within 900s"
  fi
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

exit "$AGENT_RC"
