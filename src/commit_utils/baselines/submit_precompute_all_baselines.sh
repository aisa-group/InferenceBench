#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
REPO_ROOT="$(pwd)"
source src/commit_utils/set_env_vars.sh

REQUIREMENTS=()
POSITIONAL=()
for arg in "$@"; do
  if [ "${arg}" = "--a100" ]; then
    use_a100
  else
    POSITIONAL+=("${arg}")
  fi
done
set -- "${POSITIONAL[@]}"

BASE_MODEL="${1:-${INFERENCE_BENCH_BASE_MODEL}}"
BID="${2:-100}"
BACKENDS="${3:-torch}"
DATASET_SEED="${4:-${INFERENCE_BENCH_DATASET_SEED}}"
QUALITY_SEED="${5:-${INFERENCE_BENCH_QUALITY_SEED}}"
MMLUPRO_N="${6:-${INFERENCE_BENCH_QUALITY_MMLUPRO_N}}"
QUALITY_CONCURRENCY="${7:-${INFERENCE_BENCH_QUALITY_CONCURRENCY}}"

echo "Submitting baseline precompute: base_model=${BASE_MODEL} backends=${BACKENDS} bid=${BID} dataset_seed=${DATASET_SEED} quality_seed=${QUALITY_SEED} mmlupro_n=${MMLUPRO_N} quality_concurrency=${QUALITY_CONCURRENCY} skip_speed=${INFERENCE_BENCH_SKIP_SPEED} skip_quality=${INFERENCE_BENCH_SKIP_QUALITY}"

IFS=',' read -r -a BACKEND_LIST <<< "${BACKENDS}"
[ "${#BACKEND_LIST[@]}" -gt 0 ] || { echo "At least one backend is required." >&2; exit 2; }
for backend in "${BACKEND_LIST[@]}"; do
  case "${backend}" in
    vllm|torch|sglang) ;;
    tgi)
      if [ "${INFERENCE_BENCH_SKIP_SPEED}" != "1" ] && [ "${INFERENCE_BENCH_DOWNLOAD_ONLY}" != "1" ]; then
        echo "TGI cannot run speed baselines because its API cannot ignore EOS; set INFERENCE_BENCH_SKIP_SPEED=1." >&2
        exit 2
      fi
      ;;
    *) echo "Invalid backend: ${backend}" >&2; exit 2 ;;
  esac
done
for backend in "${BACKEND_LIST[@]}"; do
  echo "Submitting backend=${backend}"
  condor_submit_bid "${BID}" \
    -a "initialdir=${REPO_ROOT}" \
    -a "base_model=${BASE_MODEL}" \
    -a "backends=${backend}" \
    -a "dataset_seed=${DATASET_SEED}" \
    -a "quality_seed=${QUALITY_SEED}" \
    -a "mmlupro_n=${MMLUPRO_N}" \
    -a "quality_concurrency=${QUALITY_CONCURRENCY}" \
    "${REQUIREMENTS[@]}" \
    src/commit_utils/baselines/precompute_all_baselines.sub
done
