#!/usr/bin/env bash
set -euo pipefail

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

echo "ERROR: start_server.sh has no engine configured."
echo "Edit this file and replace this stub with your launch command. Available: HOST, PORT, INFERENCE_BENCH_BASE_MODEL, INFERENCE_BENCH_MAX_MODEL_LEN, HF_HOME."
exit 2
