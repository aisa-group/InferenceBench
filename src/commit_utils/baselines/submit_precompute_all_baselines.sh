#!/bin/bash
set -euo pipefail

# ============================================================================
#  Baseline Precompute — Single Job Submission
# ============================================================================
#  Edit these defaults, then run:   bash src/commit_utils/baselines/submit_precompute_all_baselines.sh
#
#  Or override from the command line:
#    bash submit_precompute_all_baselines.sh [BASE_MODEL] [BID] [BACKENDS] \
#         [DATASET_SEED] [QUALITY_SEED] [MMLUPRO_N] [QUALITY_CONCURRENCY]
# ============================================================================

# ──── Edit these ────────────────────────────────────────────────────────────
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"  # Model to benchmark
BACKENDS="torch"                    # Comma-separated: vllm, torch, sglang
BID=100                             # HTCondor bid priority
DATASET_SEED=248                    # Seed for synthetic dataset sampling
QUALITY_SEED=248                    # Seed for MMLU-Pro quality eval
MMLUPRO_N=500                       # Number of MMLU-Pro samples
QUALITY_CONCURRENCY=4               # Concurrent requests for quality eval
SKIP_SPEED=0                        # Set to 1 to skip speed baselines
SKIP_QUALITY=0                      # Set to 1 to skip quality baselines
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

# Command-line args override the defaults above.
BASE_MODEL="${1:-${BASE_MODEL}}"
BID="${2:-${BID}}"
BACKENDS="${3:-${BACKENDS}}"
DATASET_SEED="${4:-${DATASET_SEED}}"
QUALITY_SEED="${5:-${QUALITY_SEED}}"
MMLUPRO_N="${6:-${MMLUPRO_N}}"
QUALITY_CONCURRENCY="${7:-${QUALITY_CONCURRENCY}}"

REQ_EXPR='TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"'

# Export skip flags so HTCondor passes them via $ENV(...) in the .sub file.
export INFERENCE_BENCH_SKIP_SPEED="${SKIP_SPEED}"
export INFERENCE_BENCH_SKIP_QUALITY="${SKIP_QUALITY}"

echo "Submitting baseline precompute jobs:"
echo "  base_model=${BASE_MODEL}"
echo "  backends=${BACKENDS} (one job per backend)"
echo "  bid=${BID}"
echo "  dataset_seed=${DATASET_SEED} quality_seed=${QUALITY_SEED} mmlupro_n=${MMLUPRO_N} quality_concurrency=${QUALITY_CONCURRENCY}"
echo "  skip_speed=${SKIP_SPEED} skip_quality=${SKIP_QUALITY}"
echo "  condor_requirements=${REQ_EXPR}"

IFS=',' read -r -a BACKEND_LIST <<< "${BACKENDS}"
for backend in "${BACKEND_LIST[@]}"; do
  backend="${backend//[[:space:]]/}"
  [ -z "${backend}" ] && continue
  echo "--- submitting backend=${backend}"
  condor_submit_bid "${BID}" \
    -a "initialdir=${REPO_ROOT}" \
    -a "base_model=${BASE_MODEL}" \
    -a "backends=${backend}" \
    -a "dataset_seed=${DATASET_SEED}" \
    -a "quality_seed=${QUALITY_SEED}" \
    -a "mmlupro_n=${MMLUPRO_N}" \
    -a "quality_concurrency=${QUALITY_CONCURRENCY}" \
    -a "requirements=${REQ_EXPR}" \
    "${REPO_ROOT}/src/commit_utils/baselines/precompute_all_baselines.sub"
done
