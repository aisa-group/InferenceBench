#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
REPO_ROOT="$(pwd)"
source src/commit_utils/set_env_vars.sh

SEED_PAIRS_ARG="${DEFAULT_SEED_PAIRS}"
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --seed-pairs) shift; SEED_PAIRS_ARG="$1" ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done
set -- "${POSITIONAL[@]}"

BASE_MODEL="${1:-${INFERENCE_BENCH_BASE_MODEL}}"
BID="${2:-40}"
METHODS="${3:-random,tpe,smac}"
ENGINES="${4:-vllm,sglang}"
SCENARIOS="${5:-a,b,c,d}"
BUDGET_S="${6:-7200}"
export HPO_BASELINE_OUT_ROOT="${7:-${INFERENCE_BENCH_RESULTS_DIR}/hpo_search_baselines}"

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
IFS=',' read -r -a ENGINE_LIST <<< "${ENGINES}"
IFS=',' read -r -a SCENARIO_LIST <<< "${SCENARIOS}"
IFS=',' read -r -a SEED_PAIRS <<< "${SEED_PAIRS_ARG}"

invalid() {
  echo "$*" >&2
  exit 2
}
[[ "${BASE_MODEL}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?$ ]] || invalid "Invalid Hugging Face model ID: ${BASE_MODEL}"
for method in "${METHOD_LIST[@]}"; do
  [[ "${method}" =~ ^(random|tpe|smac)$ ]] || invalid "Invalid method: ${method}"
done
for engine in "${ENGINE_LIST[@]}"; do
  [ "${engine}" != "tgi" ] || invalid "TGI does not support HPO search because its API cannot ignore EOS."
  [[ "${engine}" =~ ^(vllm|sglang)$ ]] || invalid "Invalid engine: ${engine}"
done
for scenario in "${SCENARIO_LIST[@]}"; do
  [[ "${scenario}" =~ ^[abcd]$ ]] || invalid "Invalid scenario: ${scenario}"
done
[[ "${BUDGET_S}" =~ ^[0-9]+([.][0-9]+)?$ ]] || invalid "Invalid budget: ${BUDGET_S}"
[ "${#SEED_PAIRS[@]}" -gt 0 ] || invalid "At least one seed pair is required."
for pair in "${SEED_PAIRS[@]}"; do
  [[ "${pair}" =~ ^[0-9]+:[0-9]+$ ]] || invalid "Invalid seed pair: ${pair}"
done

echo "Submitting HPO search: base_model=${BASE_MODEL} methods=${METHODS} engines=${ENGINES} scenarios=${SCENARIOS} seed_pairs=${SEED_PAIRS_ARG} budget_s=${BUDGET_S} out_root=${HPO_BASELINE_OUT_ROOT}"
mkdir -p "${HPO_BASELINE_OUT_ROOT}"

DEFAULT_QUICK_EVAL_TIMEOUT_S="${HPO_BASELINE_QUICK_EVAL_TIMEOUT_S:-}"
SLOW_QUICK_EVAL_TIMEOUT_S="${HPO_BASELINE_SGLANG_SLOW_QUICK_EVAL_TIMEOUT_S:-600}"
for method in "${METHOD_LIST[@]}"; do
  for engine in "${ENGINE_LIST[@]}"; do
    for scenario in "${SCENARIO_LIST[@]}"; do
      for seed_id in "${!SEED_PAIRS[@]}"; do
        IFS=':' read -r seed_dev seed_eval <<< "${SEED_PAIRS[seed_id]}"
        export HPO_BASELINE_QUICK_EVAL_TIMEOUT_S="${DEFAULT_QUICK_EVAL_TIMEOUT_S}"
        [[ "${engine}${scenario}" != sglang[bc] ]] || HPO_BASELINE_QUICK_EVAL_TIMEOUT_S="${SLOW_QUICK_EVAL_TIMEOUT_S}"
        echo "Submitting method=${method} engine=${engine} scenario=${scenario} seed_id=${seed_id} seed_pair=${seed_dev}:${seed_eval}"
        condor_submit_bid "${BID}" \
          -a "initialdir=${REPO_ROOT}" \
          -a "method=${method}" \
          -a "engine=${engine}" \
          -a "scenario=${scenario}" \
          -a "seed_id=${seed_id}" \
          -a "seed_dev=${seed_dev}" \
          -a "seed_eval=${seed_eval}" \
          -a "budget_s=${BUDGET_S}" \
          -a "base_model=${BASE_MODEL}" \
          src/commit_utils/baselines/hpo_search_baselines.sub
      done
    done
  done
done
