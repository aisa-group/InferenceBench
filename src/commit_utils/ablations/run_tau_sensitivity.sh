#!/bin/bash
# Runs scripts/tau_sensitivity.py on the worker. CPU-only, no container.
# The script only reads metrics.json files from INFERENCE_BENCH_RESULTS_DIR
# and writes its outputs under TAU_OUT_DIR, so no GPU and no apptainer
# setup is needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_ROOT="${INFERENCE_BENCH_RESULTS_DIR:-/fast/${USER}/ptb_results}"
OUT_DIR="${TAU_OUT_DIR:-${REPO_ROOT}/tau_sensitivity_out}"

echo "[tau_sensitivity] repo_root=${REPO_ROOT}"
echo "[tau_sensitivity] results_root=${RESULTS_ROOT}"
echo "[tau_sensitivity] out_dir=${OUT_DIR}"
echo "[tau_sensitivity] python=$(command -v python3)"

mkdir -p "${OUT_DIR}"

python3 -u scripts/tau_sensitivity.py \
    --results "${RESULTS_ROOT}" \
    --baselines "${REPO_ROOT}/src/eval/inference/baselines/speed" \
    --out "${OUT_DIR}"

echo "[tau_sensitivity] done."
echo "[tau_sensitivity] artifacts:"
ls -l "${OUT_DIR}"
