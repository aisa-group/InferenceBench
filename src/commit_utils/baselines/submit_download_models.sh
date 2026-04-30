#!/bin/bash
set -euo pipefail
#
# Download models to shared HF cache via HTCondor (no GPU, no container).
# Runs directly on the worker node using the host Python + huggingface_hub.
#
# Usage:
#   bash src/commit_utils/baselines/submit_download_models.sh <model_id> [bid]
#   bash src/commit_utils/baselines/submit_download_models.sh "Qwen/Qwen2.5-14B-Instruct" 50
#   bash src/commit_utils/baselines/submit_download_models.sh "Qwen/Qwen2.5-32B-Instruct" 50
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

MODEL="${1:?Usage: $0 <model_id> [bid]}"
BID="${2:-50}"

echo "Submitting model download job (no container, host Python):"
echo "  model=${MODEL}"
echo "  bid=${BID}"
echo "  HF_HOME=${HF_HOME}"

condor_submit_bid "${BID}" \
  -a "initialdir=${REPO_ROOT}" \
  -a "model=${MODEL}" \
  "${REPO_ROOT}/src/commit_utils/baselines/download_model.sub"
