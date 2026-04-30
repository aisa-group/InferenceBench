#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Source API keys / tokens (HF_TOKEN, etc.)
if [ -f "${REPO_ROOT}/env.sh" ]; then
    # shellcheck source=../../env.sh
    source "${REPO_ROOT}/env.sh"
fi

# Source environment defaults (HF_HOME, INFERENCE_BENCH_CONTAINERS_DIR, etc.)
# shellcheck source=../../src/commit_utils/set_env_vars.sh
source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

CONTAINERS_DIR="${INFERENCE_BENCH_CONTAINERS_DIR:-${REPO_ROOT}/containers}"
CONTAINER_SIF="${CONTAINERS_DIR}/download_hf_cache.sif"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

# Auto-build if missing
if [ ! -f "${CONTAINER_SIF}" ]; then
    echo "[download_hf_cache] building container..."
    bash "${REPO_ROOT}/containers/build_container.sh" download_hf_cache
fi

echo "[download_hf_cache] container=${CONTAINER_SIF}"
echo "[download_hf_cache] hf_cache=${HF_CACHE}"

apptainer run \
    --bind "${HF_CACHE}:${HF_CACHE}" \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env HF_HOME="${HF_CACHE}" \
    --env HF_HUB_CACHE="${HF_CACHE}/hub" \
    --env HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "${CONTAINER_SIF}" \
    "${REPO_ROOT}/containers/download_hf_cache/download_resources.py" "$@"

echo "[download_hf_cache] complete. Cache at: ${HF_CACHE}"
