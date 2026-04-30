#!/bin/bash
set -euo pipefail

# ============================================================================
#  Random-Search Baseline over vLLM Configurations — Condor Submission
# ============================================================================
#  Launches vLLM once per sampled configuration, runs the four speed
#  scenarios against each, and records per-config metrics for Appendix I.6.
#  Config 0 is always the default vLLM launch; configs 1..N are random
#  samples from the parameter space defined in
#  src/eval/inference/random_search_vllm.py (max-num-seqs,
#  max-num-batched-tokens, kv-cache-dtype, gpu-memory-utilization,
#  block-size). A handful of pinned "special" configs at the end cover
#  --enforce-eager, --no-enable-chunked-prefill, and FLASHINFER attention
#  backend, so every one-shot qualitative toggle has at least one data
#  point.
#
#  Uses the same apptainer / venv / HF-cache plumbing as
#  submit_precompute_all_baselines.sh by piggy-backing on
#  run_precompute_all_baselines.sh with INFERENCE_BENCH_INNER_MODE=random_search.
#
#  Usage:
#    bash src/commit_utils/ablations/submit_random_search_vllm.sh [BASE_MODEL] [BID] [NUM_CONFIGS] [SEARCH_SEED] [OUT_ROOT]
#
#  Defaults below match the current main table.
# ============================================================================

BASE_MODEL_DEFAULT="mistralai/Mistral-7B-Instruct-v0.3"
BID_DEFAULT=100
NUM_CONFIGS_DEFAULT=32
SEARCH_SEED_DEFAULT=248
OUT_ROOT_DEFAULT="/fast/${USER}/random_search_vllm"

BASE_MODEL="${1:-${BASE_MODEL_DEFAULT}}"
BID="${2:-${BID_DEFAULT}}"
NUM_CONFIGS="${3:-${NUM_CONFIGS_DEFAULT}}"
SEARCH_SEED="${4:-${SEARCH_SEED_DEFAULT}}"
OUT_ROOT="${5:-${OUT_ROOT_DEFAULT}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

# Piggy-back on the precompute runner but switch its final python call to
# the random-search driver.
export INFERENCE_BENCH_INNER_MODE="random_search"
export RANDOM_SEARCH_N="${NUM_CONFIGS}"
export RANDOM_SEARCH_SEED="${SEARCH_SEED}"
export RANDOM_SEARCH_OUT_ROOT="${OUT_ROOT}"
export RANDOM_SEARCH_START_FROM="${RANDOM_SEARCH_START_FROM:-0}"

# Quality eval is skipped inside the driver (model weights don't change
# across configs), but we still set the env var so the outer precompute
# harness doesn't try to run the quality loop if it falls back.
export INFERENCE_BENCH_SKIP_QUALITY=1

# Baseline precompute uses a 1800s ceiling for vllm first-startup; pin it
# here so every config has the same budget.
export INFERENCE_BENCH_SERVER_START_TIMEOUT_S="${INFERENCE_BENCH_SERVER_START_TIMEOUT_S:-1800}"

mkdir -p "${OUT_ROOT}"

echo "Submitting random-search vLLM job:"
echo "  base_model=${BASE_MODEL}"
echo "  bid=${BID}"
echo "  num_random_configs=${NUM_CONFIGS} (plus 1 default + 3 pinned specials)"
echo "  search_seed=${SEARCH_SEED}"
echo "  out_root=${OUT_ROOT}"
echo "  server_start_timeout_s=${INFERENCE_BENCH_SERVER_START_TIMEOUT_S}"

# Use the same condor requirements filter as submit_precompute_all_baselines.sh.
REQ_EXPR='TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"'

condor_submit_bid "${BID}" \
  -a "initialdir=${REPO_ROOT}" \
  -a "base_model=${BASE_MODEL}" \
  -a "backends=vllm" \
  -a "dataset_seed=248" \
  -a "quality_seed=248" \
  -a "mmlupro_n=200" \
  -a "quality_concurrency=1" \
  -a "requirements=${REQ_EXPR}" \
  "${REPO_ROOT}/src/commit_utils/ablations/random_search_vllm.sub"
