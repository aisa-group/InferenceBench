#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source src/commit_utils/set_env_vars.sh

MODE="${INFERENCE_BENCH_INNER_MODE:-precompute}"
case "${MODE}" in
  precompute|hpo_search) ;;
  *) echo "[baseline_precompute] ERROR: INFERENCE_BENCH_INNER_MODE must be precompute or hpo_search" >&2; exit 2 ;;
esac
BASE_MODEL="${1:-${INFERENCE_BENCH_BASE_MODEL}}"
CLUSTER_ID="${2:-local}"
BACKEND="${3:-vllm}"
[ -z "${4:-}" ] || export INFERENCE_BENCH_DATASET_SEED="$4"
[ -z "${5:-}" ] || export INFERENCE_BENCH_QUALITY_SEED="$5"
[ -z "${6:-}" ] || export INFERENCE_BENCH_QUALITY_MMLUPRO_N="$6"
[ -z "${7:-}" ] || export INFERENCE_BENCH_QUALITY_CONCURRENCY="$7"

fail() {
  echo "[baseline_precompute] ERROR: $*" >&2
  exit 2
}

CONTAINER_PATH="/root/.local/bin:/home/agent/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
DISABLE_RUNTIME_SHIMS=1
case "${BACKEND}" in
  vllm) DISABLE_RUNTIME_SHIMS=0 ;;
  torch|sglang) ;;
  tgi)
    CONTAINER_PATH="/usr/src/.venv/bin:${CONTAINER_PATH}"
    if [ "${INFERENCE_BENCH_DOWNLOAD_ONLY}" != "1" ]; then
      [ "${MODE}" != "hpo_search" ] || fail "TGI does not support HPO search because its API cannot ignore EOS."
      [ "${INFERENCE_BENCH_SKIP_SPEED}" = "1" ] || fail "TGI cannot run speed baselines because its API cannot ignore EOS; set INFERENCE_BENCH_SKIP_SPEED=1."
    fi
    ;;
  *) fail "exactly one supported backend is required (vllm, torch, sglang, tgi): ${BACKEND}" ;;
esac
container_var="INFERENCE_BENCH_BASELINE_CONTAINER_${BACKEND^^}_NAME"
venv_var="INFERENCE_BENCH_VENV_${BACKEND^^}_HOST"
CONTAINER_NAME="${!container_var}"
VENV_HOST="${!venv_var}"

RESULT_PREFIX_SAFE="$(echo "${BASE_MODEL}" | tr '/:' '_')"
RESULTS_ROOT="$(realpath -ms "${INFERENCE_BENCH_RESULTS_DIR}")"
CONTAINER_SIF="$(realpath -ms "${INFERENCE_BENCH_CONTAINERS_DIR}")/${CONTAINER_NAME}.sif"
RESULT_DIR="${RESULTS_ROOT}/baseline_precompute/${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"
mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/output.log")
exec 2> >(tee -a "${RESULT_DIR}/error.log" >&2)

echo "[baseline_precompute] base_model=${BASE_MODEL} backend=${BACKEND} dataset_seed=${INFERENCE_BENCH_DATASET_SEED} quality_seed=${INFERENCE_BENCH_QUALITY_SEED} mmlupro_n=${INFERENCE_BENCH_QUALITY_MMLUPRO_N} quality_concurrency=${INFERENCE_BENCH_QUALITY_CONCURRENCY} max_model_len=${INFERENCE_BENCH_MAX_MODEL_LEN}"
echo "[baseline_precompute] results_dir=${RESULT_DIR} container=${CONTAINER_SIF}"

RANDOM_UUID="$(uuidgen)"
TMP_ROOT="${RESULTS_ROOT}/tmp"
mkdir -p "${TMP_ROOT}" 2>/dev/null && [ -w "${TMP_ROOT}" ] || TMP_ROOT="/tmp"
TMP_SUBDIR="${TMP_ROOT}/inferencebench_precompute_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"
JOB_TMP="${TMP_SUBDIR}/tmp"
mkdir -p "${JOB_TMP}" "${VENV_HOST}" "${VENV_HOST}.locks"

if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "[baseline_precompute] building container ${CONTAINER_NAME}"
  bash containers/build_container.sh "${CONTAINER_NAME}"
fi

HF_RESOURCES_FILE="${JOB_TMP}/hf_resources.json"
printf '{"models": ["%s"], "datasets": [{"dataset": "TIGER-Lab/MMLU-Pro", "configs": ["default"], "splits": ["test"]}]}\n' \
  "${BASE_MODEL}" > "${HF_RESOURCES_FILE}"
APPTAINER_BIND="${APPTAINER_BIND:+${APPTAINER_BIND},}${HF_RESOURCES_FILE}:${HF_RESOURCES_FILE}:ro" \
  bash containers/download_hf_cache/download_hf_cache.sh --resources-file "${HF_RESOURCES_FILE}" --workers 4

if [ "${INFERENCE_BENCH_DOWNLOAD_ONLY}" = "1" ]; then
  echo "[baseline_precompute] download complete"
  exit 0
fi

if ! ls /dev/nvidia* >/dev/null 2>&1; then
  echo "[baseline_precompute] ERROR: no GPU is visible (/dev/nvidia* missing); submit with src/commit_utils/baselines/submit_precompute_all_baselines.sh." >&2
  exit 61
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  assigned="${_CONDOR_AssignedGPUs:-}"
  uuid_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
  picked=""
  for tok in ${assigned//[;,]/ }; do
    case "${tok}" in
      CUDA*) tok="${tok#CUDA}" ;;
      GPU-*) tok="$(awk -F, -v want="${tok,,}" 'index(tolower($2), want) == 1 {print $1; exit}' <<< "${uuid_map}")" ;;
    esac
    [[ ! "${tok}" =~ ^[0-9]+$ ]] || picked="${picked:+${picked},}${tok}"
  done
  [ -n "${picked}" ] || picked="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [ -n "${picked}" ]; then
    export CUDA_VISIBLE_DEVICES="${picked}" NVIDIA_VISIBLE_DEVICES="${picked}"
  fi
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${NVIDIA_VISIBLE_DEVICES:-}" = "all" ]; then
  export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi

cat >"${JOB_TMP}/inner_precompute.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
BACKEND="$1"

VENV="/venv"
if [ -x "/usr/src/.venv/bin/python" ] && [ -f "/usr/src/.venv/bin/activate" ]; then
  VENV="/usr/src/.venv"
fi
export HOME="/tmp/home"
export PIP_CACHE_DIR="${VENV}/pip_cache"
export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "${HOME}" "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${FLASHINFER_CACHE_DIR}" "${FLASHINFER_TMPDIR}"

if [ ! -x "${VENV}/bin/python" ] || [ ! -f "${VENV}/bin/activate" ]; then
  lockdir="/venv_locks/bootstrap.lockdir"
  waited=0
  until mkdir "${lockdir}" 2>/dev/null; do
    if ((++waited >= 120)); then
      echo "[baseline_precompute] ERROR: could not acquire lock: ${lockdir}" >&2
      exit 2
    fi
    sleep 1
  done
  trap 'rmdir "${lockdir}" 2>/dev/null || true' EXIT
  if [ ! -x "${VENV}/bin/python" ] || [ ! -f "${VENV}/bin/activate" ]; then
    echo "[baseline_precompute] creating venv at ${VENV}"
    python3 -m venv --clear --system-site-packages "${VENV}"
  fi
  rmdir "${lockdir}" 2>/dev/null || true
  trap - EXIT
fi
source "${VENV}/bin/activate"

python -c "import transformers" || {
  echo "[baseline_precompute] ERROR: the shared venv cannot import transformers." >&2
  exit 4
}

if [ "${INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS}" != "1" ] \
    && ! nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >/dev/null 2>&1; then
  shim_dir="${INFERENCE_BENCH_RUNTIME_CACHE_DIR}/bin"
  mkdir -p "${shim_dir}"
  cat >"${shim_dir}/nvidia-smi" <<'EOSHIM'
#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
import sys
import torch

args = sys.argv[1:]

if not torch.cuda.is_available():
    print("nvidia-smi shim: CUDA is not available", file=sys.stderr)
    raise SystemExit(1)

props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]

query_arg = next((a for a in args if a.startswith("--query-gpu=")), None)
format_arg = next((a for a in args if a.startswith("--format=")), "")
fmt = format_arg.split("=", 1)[1] if format_arg else ""

if query_arg:
    fields = [f.strip() for f in query_arg.split("=", 1)[1].split(",") if f.strip()]
    rows = []
    for p in props:
        values = []
        for f in fields:
            if f == "memory.total":
                values.append(str(int(p.total_memory // (1024 * 1024))))
            elif f == "name":
                values.append(p.name)
            else:
                values.append("unknown")
        rows.append(values)

    if "csv" in fmt:
        if "noheader" not in fmt:
            print(", ".join(fields))
        for vals in rows:
            print(", ".join(vals))
    else:
        for vals in rows:
            print(" ".join(vals))
    raise SystemExit(0)

print("nvidia-smi shim (torch-backed):")
for idx, p in enumerate(props):
    print(f"GPU {idx}: {p.name} {int(p.total_memory // (1024 * 1024))} MiB")
PY
EOSHIM
  chmod +x "${shim_dir}/nvidia-smi"
  export PATH="${shim_dir}:${PATH}"
  echo "[baseline_precompute] installed nvidia-smi shim at ${shim_dir}/nvidia-smi"
fi

python - <<'PY' || { rc=$?; echo "[baseline_precompute] ERROR: CUDA is not usable in the container." >&2; exit "${rc}"; }
import sys
import torch

ok = torch.cuda.is_available()
print(f"[baseline_precompute] torch={torch.__version__} torch_cuda={torch.version.cuda} cuda_available={ok}")
if ok:
    torch.zeros(1, device="cuda")
sys.exit(0 if ok else 1)
PY

if [ "${INFERENCE_BENCH_INNER_MODE}" = "hpo_search" ]; then
  exec python -u -m src.eval.inference.hpo_search_baselines \
    --method "${HPO_BASELINE_METHOD}" \
    --engine "${BACKEND}" \
    --scenario "${HPO_BASELINE_SCENARIO}" \
    --seed-id "${HPO_BASELINE_SEED_ID}"
fi
flags=()
[ -z "${INFERENCE_BENCH_SERVER_START_TIMEOUT_S}" ] || flags+=(--server-start-timeout-s "${INFERENCE_BENCH_SERVER_START_TIMEOUT_S}")
[ "${INFERENCE_BENCH_SKIP_SPEED}" != "1" ] || flags+=(--skip-speed)
[ "${INFERENCE_BENCH_SKIP_QUALITY}" != "1" ] || flags+=(--skip-quality)
exec python -u -m src.eval.inference.precompute_all_baselines \
  --cache-quality-samples \
  --backends "${BACKEND}" \
  --base-model "${INFERENCE_BENCH_BASE_MODEL}" \
  --max-model-len "${INFERENCE_BENCH_MAX_MODEL_LEN}" \
  --dataset-seed "${INFERENCE_BENCH_DATASET_SEED}" \
  --quality-seed "${INFERENCE_BENCH_QUALITY_SEED}" \
  --mmlupro-n "${INFERENCE_BENCH_QUALITY_MMLUPRO_N}" \
  --quality-concurrency "${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
  "${flags[@]}"
SH

EXTRA_ARGS=()
if [ "${MODE}" = "hpo_search" ]; then
  HPO_BASELINE_OUT_ROOT="$(realpath -ms "${HPO_BASELINE_OUT_ROOT:-${RESULTS_ROOT}/hpo_search_baselines}")"
  export HPO_BASELINE_OUT_ROOT
  mkdir -p "${HPO_BASELINE_OUT_ROOT}"
  echo "[baseline_precompute] hpo_baseline_out_root=${HPO_BASELINE_OUT_ROOT}"
  EXTRA_ARGS+=(--bind "${HPO_BASELINE_OUT_ROOT}:${HPO_BASELINE_OUT_ROOT}")
fi
for name in $(compgen -e HPO_BASELINE_ || true); do
  EXTRA_ARGS+=(--env "${name}=${!name}")
done
for name in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES; do
  [ -z "${!name:-}" ] || EXTRA_ARGS+=(--env "${name}=${!name}")
done
[ ! -d /data ] || EXTRA_ARGS+=(--bind /data:/data:ro)
[ ! -d /dev/shm ] || EXTRA_ARGS+=(--bind /dev/shm:/dev/shm)

RUNTIME_CACHE="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}"
echo "[baseline_precompute] starting container"
apptainer exec \
  --nv \
  -c \
  --cleanenv \
  --env USER="${USER:-$(id -un)}" \
  --env PATH="${CONTAINER_PATH}" \
  --env PYTHONNOUSERSITE="1" \
  --env PIP_CONFIG_FILE="/dev/null" \
  --env TMPDIR="/tmp/tmp" \
  --env TMP="/tmp/tmp" \
  --env TEMP="/tmp/tmp" \
  --env XDG_CACHE_HOME="/tmp/xdg_cache" \
  --env SOFT_FILELOCK="1" \
  --env PYTHONPATH="/opt/filelock_workarounds/soft_file_locks" \
  --env INFERENCE_BENCH_SERVER_START_TIMEOUT_S="${INFERENCE_BENCH_SERVER_START_TIMEOUT_S:-}" \
  --env INFERENCE_BENCH_DISABLE_HF_LOCKS="1" \
  --env INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS="${DISABLE_RUNTIME_SHIMS}" \
  --env INFERENCE_BENCH_ALLOW_HF_DOWNLOAD="1" \
  --env HF_HOME="/hf_cache" \
  --env HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  --env HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
  --env HF_HUB_DISABLE_MMAP="1" \
  --env HF_HUB_CACHE="/hf_cache/hub" \
  --env HUGGINGFACE_HUB_CACHE="/hf_cache/hub" \
  --env TRANSFORMERS_CACHE="/hf_cache/hub" \
  --env HF_DATASETS_CACHE="/hf_cache/datasets" \
  --env INFERENCE_BENCH_BASE_MODEL="${BASE_MODEL}" \
  --env INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN}" \
  --env INFERENCE_BENCH_INPUT_TOKEN_MARGIN="${INFERENCE_BENCH_INPUT_TOKEN_MARGIN}" \
  --env INFERENCE_BENCH_RESULTS_DIR="${INFERENCE_BENCH_RESULTS_DIR}" \
  --env INFERENCE_BENCH_DATASET_SEED="${INFERENCE_BENCH_DATASET_SEED}" \
  --env INFERENCE_BENCH_QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED}" \
  --env INFERENCE_BENCH_QUALITY_MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N}" \
  --env INFERENCE_BENCH_QUALITY_CONCURRENCY="${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
  --env INFERENCE_BENCH_QUALITY_TAU="${INFERENCE_BENCH_QUALITY_TAU}" \
  --env INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY="${INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY}" \
  --env INFERENCE_BENCH_QUALITY_BASELINE_BACKEND="${INFERENCE_BENCH_QUALITY_BASELINE_BACKEND}" \
  --env INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE="${INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE}" \
  --env INFERENCE_BENCH_SKIP_SPEED="${INFERENCE_BENCH_SKIP_SPEED}" \
  --env INFERENCE_BENCH_SKIP_QUALITY="${INFERENCE_BENCH_SKIP_QUALITY}" \
  --env INFERENCE_BENCH_INNER_MODE="${MODE}" \
  --env INFERENCE_BENCH_RUNTIME_CACHE_DIR="${RUNTIME_CACHE}" \
  --env VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  --env FLASHINFER_CACHE_DIR="${RUNTIME_CACHE}/flashinfer_cache" \
  --env FLASHINFER_JIT_CACHE_DIR="${RUNTIME_CACHE}/flashinfer_cache" \
  --env FLASHINFER_BUILD_DIR="${RUNTIME_CACHE}/flashinfer_cache" \
  --env FLASHINFER_TMPDIR="${RUNTIME_CACHE}/flashinfer_tmp" \
  --env OUTLINES_CACHE_DIR="${RUNTIME_CACHE}/outlines_cache" \
  --env DISKCACHE_DIRECTORY="${RUNTIME_CACHE}/outlines_cache" \
  --env TVM_FFI_CACHE_DIR="/dev/shm/inferencebench_tvm_ffi_${CLUSTER_ID}_${RANDOM_UUID}" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env HUGGINGFACEHUB_API_TOKEN="${HF_TOKEN:-}" \
  "${EXTRA_ARGS[@]}" \
  --bind "${JOB_TMP}:/tmp" \
  --bind "${VENV_HOST}:/venv" \
  --bind "${VENV_HOST}.locks:/venv_locks" \
  --bind "${HF_HOME}:/hf_cache" \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  --pwd "${REPO_ROOT}" \
  --writable-tmpfs \
  "${CONTAINER_SIF}" \
  bash /tmp/inner_precompute.sh "${BACKEND}"

echo "[baseline_precompute] done"
rm -rf "${TMP_SUBDIR}"
