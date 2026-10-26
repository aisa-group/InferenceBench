#!/bin/bash
set -euo pipefail
source /home/agent/agent_common.sh

echo "[agent] $(date --iso-8601=seconds) starting claude agent"
START_CMD=(claude --print --verbose --output-format stream-json --model "${AGENT_CONFIG}" --dangerously-skip-permissions)
CONTINUE_CMD=("${START_CMD[@]}" --continue)
run_agent
