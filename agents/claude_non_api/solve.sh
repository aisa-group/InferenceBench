#!/bin/bash
set -euo pipefail
source /home/agent/agent_common.sh

CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/agent/oauth_token)"
export CLAUDE_CODE_OAUTH_TOKEN
AGENT_LAST_OUTPUT="/tmp/claude_last_output.jsonl"

echo "[agent] $(date --iso-8601=seconds) starting claude (non-api) agent"
START_CMD=(claude --print --verbose --output-format stream-json --model "${AGENT_CONFIG}" --dangerously-skip-permissions)
CONTINUE_CMD=("${START_CMD[@]}" --continue)
rc=0
run_agent || rc=$?

if grep -Eqi 'rate_limit_event|out_of_credits|out of extra usage' "${AGENT_LAST_OUTPUT}" 2>/dev/null; then
  echo "[agent] detected claude usage limit; exiting before eval"
  exit 1
fi
exit "${rc}"
