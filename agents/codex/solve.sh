#!/bin/bash
set -euo pipefail
source /home/agent/agent_common.sh

CODEX=(codex --search)
MODEL="${AGENT_CONFIG}"
case "${MODEL}" in
  *-high) CODEX+=(-c 'model_reasoning_effort="high"'); MODEL="${MODEL%-high}" ;;
  *-med) CODEX+=(-c 'model_reasoning_effort="medium"'); MODEL="${MODEL%-med}" ;;
  *-low) CODEX+=(-c 'model_reasoning_effort="low"'); MODEL="${MODEL%-low}" ;;
esac

echo "[agent] $(date --iso-8601=seconds) starting codex agent"
START_CMD=("${CODEX[@]}" exec --skip-git-repo-check --yolo --model "${MODEL}")
CONTINUE_CMD=("${CODEX[@]}" exec resume --last --skip-git-repo-check --yolo --model "${MODEL}")
run_agent
