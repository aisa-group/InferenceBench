#!/bin/bash
set -euo pipefail

export GEMINI_SANDBOX="false"
timeout --signal=TERM --kill-after=30s "${INFERENCE_BENCH_REMAINING_SECONDS}s" \
  gemini --yolo --model "${AGENT_CONFIG}" --output-format stream-json -p "$(cat "${PROMPT_FILE}")"
