#!/bin/bash
set -euo pipefail

# Submit non-agentic search baselines.
#
# Usage:
#   bash src/commit_utils/baselines/submit_hpo_search_baselines.sh \
#     [BASE_MODEL] [BID] [METHODS] [ENGINES] [SCENARIOS] [BUDGET_S] [OUT_ROOT] [SEED_IDS]
#
# Examples:
#   # Three validation jobs from the handoff:
#   bash src/commit_utils/baselines/submit_hpo_search_baselines.sh \
#     mistralai/Mistral-7B-Instruct-v0.3 100 random vllm a 7200
#   bash src/commit_utils/baselines/submit_hpo_search_baselines.sh \
#     mistralai/Mistral-7B-Instruct-v0.3 100 tpe vllm c 7200
#   bash src/commit_utils/baselines/submit_hpo_search_baselines.sh \
#     mistralai/Mistral-7B-Instruct-v0.3 100 smac vllm b 7200
#
#   # Full 108-run matrix:
#   bash src/commit_utils/baselines/submit_hpo_search_baselines.sh \
#     mistralai/Mistral-7B-Instruct-v0.3 100 random,tpe,smac vllm,sglang,tgi a,b,c,d 7200

BASE_MODEL_DEFAULT="mistralai/Mistral-7B-Instruct-v0.3"
BID_DEFAULT=100
METHODS_DEFAULT="random,tpe,smac"
ENGINES_DEFAULT="vllm,sglang,tgi"
SCENARIOS_DEFAULT="a,b,c,d"
BUDGET_S_DEFAULT=7200
SEED_IDS_DEFAULT="0,1,2"
SERVER_START_TIMEOUT_S_DEFAULT=300
FIRST_SERVER_START_TIMEOUT_S_DEFAULT=360
SERVER_INITIAL_DELAY_S_DEFAULT=60
QUICK_EVAL_TIMEOUT_S_DEFAULT=180
SGLANG_SLOW_QUICK_EVAL_TIMEOUT_S_DEFAULT=600

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

BASE_MODEL="${1:-${BASE_MODEL_DEFAULT}}"
BID="${2:-${BID_DEFAULT}}"
METHODS="${3:-${METHODS_DEFAULT}}"
ENGINES="${4:-${ENGINES_DEFAULT}}"
SCENARIOS="${5:-${SCENARIOS_DEFAULT}}"
BUDGET_S="${6:-${BUDGET_S_DEFAULT}}"
OUT_ROOT="${7:-${INFERENCE_BENCH_RESULTS_DIR}/hpo_search_baselines}"
SEED_IDS="${8:-${SEED_IDS_DEFAULT}}"
SERVER_START_TIMEOUT_S="${HPO_BASELINE_SERVER_START_TIMEOUT_S:-${SERVER_START_TIMEOUT_S_DEFAULT}}"
FIRST_SERVER_START_TIMEOUT_S="${HPO_BASELINE_FIRST_SERVER_START_TIMEOUT_S:-${FIRST_SERVER_START_TIMEOUT_S_DEFAULT}}"
SERVER_INITIAL_DELAY_S="${HPO_BASELINE_SERVER_INITIAL_DELAY_S:-${SERVER_INITIAL_DELAY_S_DEFAULT}}"
QUICK_EVAL_TIMEOUT_S="${HPO_BASELINE_QUICK_EVAL_TIMEOUT_S:-${QUICK_EVAL_TIMEOUT_S_DEFAULT}}"

export HPO_BASELINE_OUT_ROOT="${OUT_ROOT}"
export HPO_BASELINE_AUTO_INSTALL_DEPS="${HPO_BASELINE_AUTO_INSTALL_DEPS:-1}"

REQ_EXPR='TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3"'
QUALITY_SEED=248
MMLUPRO_N=500
QUALITY_CONCURRENCY=4

seed_dev_for_id() {
  case "$1" in
    0) echo 21 ;;
    1) echo 101 ;;
    2) echo 202 ;;
    *) echo "Unsupported seed_id=$1; expected one of 0,1,2" >&2; return 1 ;;
  esac
}

seed_eval_for_id() {
  case "$1" in
    0) echo 1337 ;;
    1) echo 1101 ;;
    2) echo 2202 ;;
    *) echo "Unsupported seed_id=$1; expected one of 0,1,2" >&2; return 1 ;;
  esac
}

mkdir -p "${OUT_ROOT}"

echo "Submitting HPO search baseline jobs:"
echo "  base_model=${BASE_MODEL}"
echo "  methods=${METHODS}"
echo "  engines=${ENGINES}"
echo "  scenarios=${SCENARIOS}"
echo "  seed_ids=${SEED_IDS} quality_seed=${QUALITY_SEED}"
echo "  budget_s=${BUDGET_S}"
echo "  quick_eval_timeout_s=${QUICK_EVAL_TIMEOUT_S} sglang_slow_quick_eval_timeout_s=${HPO_BASELINE_SGLANG_SLOW_QUICK_EVAL_TIMEOUT_S:-${SGLANG_SLOW_QUICK_EVAL_TIMEOUT_S_DEFAULT}}"
echo "  server_start_timeout_s=${SERVER_START_TIMEOUT_S} first_server_start_timeout_s=${FIRST_SERVER_START_TIMEOUT_S} initial_delay_s=${SERVER_INITIAL_DELAY_S}"
echo "  out_root=${OUT_ROOT}"

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
IFS=',' read -r -a ENGINE_LIST <<< "${ENGINES}"
IFS=',' read -r -a SCENARIO_LIST <<< "${SCENARIOS}"
IFS=',' read -r -a SEED_ID_LIST <<< "${SEED_IDS}"

for method in "${METHOD_LIST[@]}"; do
  method="${method//[[:space:]]/}"
  [ -z "${method}" ] && continue
  for engine in "${ENGINE_LIST[@]}"; do
    engine="${engine//[[:space:]]/}"
    [ -z "${engine}" ] && continue
    for scenario in "${SCENARIO_LIST[@]}"; do
      scenario="${scenario//[[:space:]]/}"
      [ -z "${scenario}" ] && continue
      for seed_id in "${SEED_ID_LIST[@]}"; do
        seed_id="${seed_id//[[:space:]]/}"
        [ -z "${seed_id}" ] && continue
        seed_dev="$(seed_dev_for_id "${seed_id}")"
        seed_eval="$(seed_eval_for_id "${seed_id}")"
        job_quick_eval_timeout_s="${QUICK_EVAL_TIMEOUT_S}"
        if [ "${engine}" = "sglang" ] && { [ "${scenario}" = "b" ] || [ "${scenario}" = "c" ]; }; then
          job_quick_eval_timeout_s="${HPO_BASELINE_SGLANG_SLOW_QUICK_EVAL_TIMEOUT_S:-${SGLANG_SLOW_QUICK_EVAL_TIMEOUT_S_DEFAULT}}"
        fi
        echo "--- submitting method=${method} engine=${engine} scenario=${scenario} seed_id=${seed_id} seed_pair=(${seed_dev},${seed_eval})"
        condor_submit_bid "${BID}" \
          -a "initialdir=${REPO_ROOT}" \
          -a "method=${method}" \
          -a "engine=${engine}" \
          -a "scenario=${scenario}" \
          -a "seed_id=${seed_id}" \
          -a "seed_dev=${seed_dev}" \
          -a "seed_eval=${seed_eval}" \
          -a "quality_seed=${QUALITY_SEED}" \
          -a "mmlupro_n=${MMLUPRO_N}" \
          -a "quality_concurrency=${QUALITY_CONCURRENCY}" \
          -a "budget_s=${BUDGET_S}" \
          -a "quick_eval_timeout_s=${job_quick_eval_timeout_s}" \
          -a "server_start_timeout_s=${SERVER_START_TIMEOUT_S}" \
          -a "first_server_start_timeout_s=${FIRST_SERVER_START_TIMEOUT_S}" \
          -a "server_initial_delay_s=${SERVER_INITIAL_DELAY_S}" \
          -a "base_model=${BASE_MODEL}" \
          -a "backends=${engine}" \
          -a "requirements=${REQ_EXPR}" \
          "${REPO_ROOT}/src/commit_utils/baselines/hpo_search_baselines.sub"
      done
    done
  done
done
