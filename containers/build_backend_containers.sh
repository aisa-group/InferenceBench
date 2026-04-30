#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

if [[ "${INFERENCE_BENCH_CONTAINERS_DIR}" = /* ]]; then
  CONTAINERS_ROOT="${INFERENCE_BENCH_CONTAINERS_DIR}"
else
  CONTAINERS_ROOT="${REPO_ROOT}/${INFERENCE_BENCH_CONTAINERS_DIR}"
fi
mkdir -p "${CONTAINERS_ROOT}"
echo "[build_backend_containers] containers_root=${CONTAINERS_ROOT}"

build_one() {
  local backend="${1:-}"
  local out_name=""
  local def_file=""
  local out_file=""
  case "${backend}" in
    sglang)
      out_name="${INFERENCE_BENCH_BASELINE_CONTAINER_SGLANG_NAME}"
      def_file="containers/inference_baselines_sglang.def"
      ;;
    vllm)
      out_name="${INFERENCE_BENCH_BASELINE_CONTAINER_VLLM_NAME}"
      def_file="containers/inference_baselines_vllm.def"
      ;;
    torch)
      out_name="${INFERENCE_BENCH_BASELINE_CONTAINER_TORCH_NAME}"
      def_file="containers/inference_baselines_torch.def"
      ;;
    tgi)
      out_name="${INFERENCE_BENCH_BASELINE_CONTAINER_TGI_NAME}"
      def_file="containers/inference_baselines_tgi.def"
      ;;
    *)
      echo "[build_backend_containers] ERROR: unknown backend '${backend}' (expected: sglang|vllm|torch|tgi)" >&2
      return 2
      ;;
  esac
  echo "[build_backend_containers] backend=${backend} -> ${out_name}.sif (${def_file})"
  bash "${REPO_ROOT}/containers/build_container.sh" "${out_name}" "${def_file}"
  out_file="${CONTAINERS_ROOT}/${out_name}.sif"
  if [ -n "${INFERENCE_BENCH_BUILD_LOG_DIR:-}" ]; then
    echo "[build_backend_containers] build log: ${INFERENCE_BENCH_BUILD_LOG_DIR%/}/${out_name}_build.log"
  elif [[ "${CONTAINERS_ROOT}" = /* ]]; then
    echo "[build_backend_containers] build log: ${CONTAINERS_ROOT}/build_logs/${out_name}_build.log"
  else
    echo "[build_backend_containers] build log: ${REPO_ROOT}/${CONTAINERS_ROOT}/build_logs/${out_name}_build.log"
  fi
  if [ ! -f "${out_file}" ]; then
    echo "[build_backend_containers] ERROR: expected output missing: ${out_file}" >&2
    return 3
  fi
  ls -lh "${out_file}"
}

if [ "$#" -eq 0 ]; then
  set -- sglang vllm torch tgi
fi

for backend in "$@"; do
  build_one "${backend}"
done
