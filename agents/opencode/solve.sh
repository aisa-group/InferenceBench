#!/bin/bash
set -euo pipefail
source /home/agent/agent_common.sh

OPENCODE_API_KEY="$(cat /home/agent/api_key)"
export OPENCODE_API_KEY

for name in venv .venv env; do
  mkdir -p "/tmp/agent_${name}"
  [ -e "/home/agent/task/${name}" ] || ln -sf "/tmp/agent_${name}" "/home/agent/task/${name}" || true
done

cat > opencode.json <<'JSON'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "opencode": {
      "options": {
        "apiKey": "{env:OPENCODE_API_KEY}"
      }
    }
  }
}
JSON

echo "[agent] $(date --iso-8601=seconds) starting opencode agent"
START_CMD=(opencode run --model "${AGENT_CONFIG}" --format json)
CONTINUE_CMD=("${START_CMD[@]}" --continue)
run_agent
