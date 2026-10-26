#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

CONTAINER_SIF="${INFERENCE_BENCH_CONTAINERS_DIR}/download_hf_cache.sif"

if [ ! -f "${CONTAINER_SIF}" ]; then
    echo "[download_hf_cache] building container"
    bash "${REPO_ROOT}/containers/build_container.sh" download_hf_cache
fi

apptainer run \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HOME}/hub" \
    --env HF_DATASETS_CACHE="${HF_HOME}/datasets" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "${CONTAINER_SIF}" \
    "${REPO_ROOT}/containers/download_hf_cache/download_resources.py" "$@"
