#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Source API keys / tokens (HF_TOKEN, etc.)
if [ -f "${REPO_ROOT}/env.sh" ]; then
    source "${REPO_ROOT}/env.sh"
fi

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"
source "${REPO_ROOT}/src/baselines/backend_profiles.sh"

BASE_MODEL="${1:-${INFERENCE_BENCH_BASE_MODEL}}"
CLUSTER_ID="${2:-local}"
BACKENDS_RAW="${3:-vllm,torch,sglang,tgi}"
DATASET_SEED_OVERRIDE="${4:-}"
QUALITY_SEED_OVERRIDE="${5:-}"
MMLUPRO_N_OVERRIDE="${6:-}"
QUALITY_CONCURRENCY_OVERRIDE="${7:-}"

# Apply backend-specific profile (per-backend container, venv, shim flags).
# Uses the first backend in the comma-separated list.
_first_backend="$(echo "${BACKENDS_RAW}" | tr ',' '\n' | head -1 | tr -d '[:space:]')"
if [ -n "${_first_backend}" ]; then
  ptb_backend_profile_apply "${_first_backend}" || true
fi
unset _first_backend

if [ -n "${DATASET_SEED_OVERRIDE}" ]; then export INFERENCE_BENCH_DATASET_SEED="${DATASET_SEED_OVERRIDE}"; fi
if [ -n "${QUALITY_SEED_OVERRIDE}" ]; then export INFERENCE_BENCH_QUALITY_SEED="${QUALITY_SEED_OVERRIDE}"; fi
if [ -n "${MMLUPRO_N_OVERRIDE}" ]; then export INFERENCE_BENCH_QUALITY_MMLUPRO_N="${MMLUPRO_N_OVERRIDE}"; fi
if [ -n "${QUALITY_CONCURRENCY_OVERRIDE}" ]; then export INFERENCE_BENCH_QUALITY_CONCURRENCY="${QUALITY_CONCURRENCY_OVERRIDE}"; fi

RESULT_PREFIX_SAFE=$(echo "${BASE_MODEL}" | tr '/:' '_')
if [[ "${INFERENCE_BENCH_RESULTS_DIR}" = /* ]]; then
  RESULTS_ROOT="${INFERENCE_BENCH_RESULTS_DIR}"
else
  RESULTS_ROOT="${REPO_ROOT}/${INFERENCE_BENCH_RESULTS_DIR}"
fi
RESULT_DIR="${RESULTS_ROOT}/baseline_precompute/${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/output.log")
exec 2> >(tee -a "${RESULT_DIR}/error.log" >&2)

echo "[baseline_precompute] repo_root=${REPO_ROOT}"
echo "[baseline_precompute] base_model=${BASE_MODEL}"
echo "[baseline_precompute] backends=${BACKENDS_RAW}"
echo "[baseline_precompute] dataset_seed=${INFERENCE_BENCH_DATASET_SEED}"
echo "[baseline_precompute] quality_seed=${INFERENCE_BENCH_QUALITY_SEED}"
echo "[baseline_precompute] mmlupro_n=${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}"
echo "[baseline_precompute] quality_concurrency=${INFERENCE_BENCH_QUALITY_CONCURRENCY}"
echo "[baseline_precompute] max_model_len=${INFERENCE_BENCH_MAX_MODEL_LEN}"
echo "[baseline_precompute] results_dir=${RESULT_DIR}"
echo "[baseline_precompute] container_name=${INFERENCE_BENCH_BASELINE_CONTAINER_NAME}"
echo "[baseline_precompute] venv_host=${INFERENCE_BENCH_VENV_HOST}"
echo "[baseline_precompute] _CONDOR_AssignedGPUs=${_CONDOR_AssignedGPUs:-}"

restore_core_host_path() {
  local core_dirs=(
    /usr/local/sbin
    /usr/local/bin
    /usr/sbin
    /usr/bin
    /sbin
    /bin
  )
  local d
  for d in "${core_dirs[@]}"; do
    case ":${PATH:-}:" in
      *":${d}:"*) ;;
      *)
        if [ -d "${d}" ]; then
          PATH="${PATH:+${PATH}:}${d}"
        fi
        ;;
    esac
  done
  export PATH
  hash -r || true
}

NEEDS_GPU=0
case ",${BACKENDS_RAW}," in
  *,vllm,*|*,sglang,*|*,torch,*|*,tgi,*) NEEDS_GPU=1 ;;
esac

DOWNLOAD_ONLY="${INFERENCE_BENCH_DOWNLOAD_ONLY:-0}"
if [ "${DOWNLOAD_ONLY}" = "1" ]; then
  NEEDS_GPU=0
fi

# Best-effort host CUDA module bootstrap. Some clusters only expose the full
# CUDA runtime/toolkit after loading the module in non-interactive shells.
if [ "${NEEDS_GPU}" = "1" ] && [ "${INFERENCE_BENCH_LOAD_CUDA_MODULE:-1}" = "1" ]; then
  module_source="${INFERENCE_BENCH_CUDA_MODULE_SOURCE:-/etc/profile.d/modules.sh}"
  if ! command -v module >/dev/null 2>&1; then
    if [ -r "${module_source}" ]; then
      # shellcheck disable=SC1090
      source "${module_source}" || true
    fi
  fi
  if command -v module >/dev/null 2>&1; then
    module_loaded=0
    for cuda_mod in ${INFERENCE_BENCH_CUDA_MODULE_CANDIDATES:-cuda}; do
      if module load "${cuda_mod}" >/dev/null 2>&1; then
        module_loaded=1
        hash -r || true
        echo "[baseline_precompute] host module load success: ${cuda_mod}"
        break
      fi
    done
    if [ "${module_loaded}" != "1" ]; then
      echo "[baseline_precompute] NOTE: could not load any CUDA module candidate: ${INFERENCE_BENCH_CUDA_MODULE_CANDIDATES:-cuda}" >&2
    fi
  else
    echo "[baseline_precompute] NOTE: environment-modules command unavailable; skipping CUDA module load." >&2
  fi
fi
restore_core_host_path

echo "[baseline_precompute] host preflight: checking NVIDIA devices/driver..."
echo "[baseline_precompute] host env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-}"
if ls -l /dev/nvidia* >/dev/null 2>&1; then
  ls -l /dev/nvidia* || true
else
  echo "[baseline_precompute] NOTE: no /dev/nvidia* visible on host"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[baseline_precompute] NOTE: nvidia-smi not found on host PATH"
fi

if [ "${NEEDS_GPU}" = "1" ] && [ "${INFERENCE_BENCH_ALLOW_NO_GPU:-0}" != "1" ]; then
  if ! ls -l /dev/nvidia* >/dev/null 2>&1; then
    if [ -n "${_CONDOR_AssignedGPUs:-}" ] || [ -n "${_CONDOR_SCRATCH_DIR:-}" ]; then
      echo "[baseline_precompute] ERROR: scheduler assigned GPU(s) (${_CONDOR_AssignedGPUs:-unknown}) but /dev/nvidia* is not visible." >&2
      echo "[baseline_precompute] Exiting without retry." >&2
      exit 61
    fi
    echo "[baseline_precompute] ERROR: this run requires a GPU node, but no /dev/nvidia* is visible on this host." >&2
    echo "[baseline_precompute] Run via HTCondor instead:" >&2
    echo "  bash src/commit_utils/baselines/submit_precompute_all_baselines.sh \"${BASE_MODEL}\"" >&2
    exit 61
  fi
fi

# Prefer selecting a concrete device on the host before entering the container.
# This avoids torch CUDA init failures when one physical GPU is unhealthy (e.g. GPU0)
# and CUDA_VISIBLE_DEVICES would otherwise be empty.
if [ "${NEEDS_GPU}" = "1" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  host_picked=""
  if [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
    condor_raw="${_CONDOR_AssignedGPUs}"
    condor_norm="$(echo "${condor_raw}" | tr ';' ',' | tr -d '[:space:]')"
    condor_cuda_visible=""
    host_uuid_map=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      host_uuid_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
    fi
    for tok in $(echo "${condor_norm}" | tr ',' ' '); do
      v=""
      if [[ "${tok}" =~ ^CUDA([0-9]+)$ ]]; then
        v="${BASH_REMATCH[1]}"
      elif [[ "${tok}" =~ ^[0-9]+$ ]]; then
        v="${tok}"
      elif [[ "${tok}" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
        v="$(echo "${host_uuid_map}" | awk -F',' -v want="${tok}" '
          {
            idx=$1; uuid=$2;
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx);
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid);
            low_uuid=tolower(uuid); low_want=tolower(want);
            if (low_uuid == low_want || index(low_uuid, low_want) == 1) { print idx; exit 0; }
          }')"
      fi
      if [ -n "${v}" ]; then
        if [ -n "${condor_cuda_visible}" ]; then
          condor_cuda_visible="${condor_cuda_visible},${v}"
        else
          condor_cuda_visible="${v}"
        fi
      fi
    done
    if [ -n "${condor_cuda_visible}" ]; then
      host_picked="${condor_cuda_visible}"
      export CUDA_VISIBLE_DEVICES="${host_picked}"
      # Keep NVIDIA_VISIBLE_DEVICES aligned so container GPU passthrough can
      # isolate the assigned device instead of exposing all host GPUs.
      export NVIDIA_VISIBLE_DEVICES="${host_picked}"
      echo "[baseline_precompute] NOTE: host selected CUDA_VISIBLE_DEVICES from _CONDOR_AssignedGPUs=${_CONDOR_AssignedGPUs} -> ${CUDA_VISIBLE_DEVICES}"
    fi
  fi

  if [ -z "${host_picked}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    host_table_idx="$(
      (nvidia-smi 2>/dev/null || true) | awk '
        /^\|[[:space:]]*[0-9]+[[:space:]]+NVIDIA/ {
          line=$0
          sub(/^\|[[:space:]]*/, "", line)
          if (match(line, /^[0-9]+/)) {
            print substr(line, RSTART, RLENGTH)
            exit
          }
        }'
    )"
    if [ -n "${host_table_idx}" ]; then
      host_picked="${host_table_idx}"
      export CUDA_VISIBLE_DEVICES="${host_picked}"
      export NVIDIA_VISIBLE_DEVICES="${host_picked}"
      echo "[baseline_precompute] NOTE: host selected CUDA_VISIBLE_DEVICES from nvidia-smi table index=${CUDA_VISIBLE_DEVICES}"
    fi
  fi
fi

if [ "${NEEDS_GPU}" = "1" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${NVIDIA_VISIBLE_DEVICES:-}" = "all" ]; then
  export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  echo "[baseline_precompute] NOTE: overriding NVIDIA_VISIBLE_DEVICES=all -> ${NVIDIA_VISIBLE_DEVICES}"
fi

RANDOM_UUID="$(uuidgen)"
TMP_ROOT="${INFERENCE_BENCH_TMP_ROOT:-}"
if [ -z "${TMP_ROOT}" ] || [ "${TMP_ROOT}" = "UNDEFINED" ]; then
  TMP_ROOT="${RESULTS_ROOT}/tmp"
fi
# Prefer a shared, large tmp area (under results root) when possible.
if [ -n "${TMP_ROOT}" ] && [ "${TMP_ROOT}" != "UNDEFINED" ] && [[ "${TMP_ROOT}" = /* ]]; then
  mkdir -p "${TMP_ROOT}" 2>/dev/null || true
fi
if [ -z "${TMP_ROOT}" ] || [ "${TMP_ROOT}" = "UNDEFINED" ] || [ ! -d "${TMP_ROOT}" ] || [ ! -w "${TMP_ROOT}" ]; then
  TMP_ROOT="/tmp"
fi

TMP_SUBDIR="${TMP_ROOT}/inferencebench_precompute_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"
JOB_TMP="${TMP_SUBDIR}/tmp"

mkdir -p "${JOB_TMP}"

HF_CACHE_HOST="${HF_HOME}"
HF_CACHE_IN_CONTAINER="/hf_cache"
echo "[baseline_precompute] hf_cache_host=${HF_CACHE_HOST} (bound to ${HF_CACHE_IN_CONTAINER} in container)"

VENV_HOST="${INFERENCE_BENCH_VENV_HOST}"
VENV_IN_CONTAINER="/venv"
mkdir -p "${VENV_HOST}/tmp" "${VENV_HOST}/pip_cache" "${VENV_HOST}/xdg_cache" "${VENV_HOST}/runtime_cache"

echo "[baseline_precompute] bootstrapping bound venv (one-time setup)..."
if [ "${INFERENCE_BENCH_SKIP_VENV_BOOTSTRAP:-0}" = "1" ]; then
  echo "[baseline_precompute] NOTE: skipping venv bootstrap (INFERENCE_BENCH_SKIP_VENV_BOOTSTRAP=1)"
else
  echo "[baseline_precompute] NOTE: legacy bootstrap script removed; venv will be created/repaired inside the container if needed."
fi

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${JOB_TMP}/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fast/${USER}/apptainer_cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

if [[ "${INFERENCE_BENCH_CONTAINERS_DIR}" = /* ]]; then
  CONTAINERS_ROOT="${INFERENCE_BENCH_CONTAINERS_DIR}"
else
  CONTAINERS_ROOT="${REPO_ROOT}/${INFERENCE_BENCH_CONTAINERS_DIR}"
fi
# Select the best available container for the requested backends.
# Prefer per-backend containers (e.g. inference_baselines_vllm.sif) since they
# have backend-specific dependencies (flashinfer for vllm, sgl_kernel for sglang).
# Fall back to the generic inference_baselines.sif if no per-backend container exists.
_pick_container_sif() {
  local backends_csv="$1"
  local containers_root="$2"

  # Try per-backend container matching the first backend in the list.
  # These have backend-specific deps (flashinfer for vllm, sgl_kernel for sglang).
  local first_backend
  first_backend="$(echo "${backends_csv}" | tr ',' '\n' | head -1 | tr -d '[:space:]')"
  local upper="${first_backend^^}"
  local env_var="INFERENCE_BENCH_BASELINE_CONTAINER_${upper}_NAME"
  local per_backend_name="${!env_var:-inference_baselines_${first_backend}}"
  local per_backend_sif="${containers_root}/${per_backend_name}.sif"
  if [ -f "${per_backend_sif}" ]; then
    echo "${per_backend_sif}"
    return
  fi

  # Fall back to the generic baseline container.
  local generic="${INFERENCE_BENCH_BASELINE_CONTAINER_NAME:-inference_baselines}"
  echo "${containers_root}/${generic}.sif"
}

CONTAINER_SIF="$(_pick_container_sif "${BACKENDS_RAW}" "${CONTAINERS_ROOT}")"
BASELINE_CONTAINER_NAME="$(basename "${CONTAINER_SIF}" .sif)"
echo "[baseline_precompute] selected container: ${CONTAINER_SIF}"

if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "[baseline_precompute] container image missing: ${CONTAINER_SIF}"
  echo "[baseline_precompute] building baseline container ${BASELINE_CONTAINER_NAME}..."
  BUILD_APPTAINER_TMPDIR="${INFERENCE_BENCH_BUILD_APPTAINER_TMPDIR:-${HOME%/}/apptainer_tmp}"
  mkdir -p "${BUILD_APPTAINER_TMPDIR}"
  APPTAINER_TMPDIR="${BUILD_APPTAINER_TMPDIR}" \
    bash "${REPO_ROOT}/containers/build_container.sh" "${BASELINE_CONTAINER_NAME}"
fi
if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "[baseline_precompute] ERROR: container image still missing after build: ${CONTAINER_SIF}" >&2
  exit 50
fi

# --- Pre-download HF resources on the HOST (before entering the container) ---
python3 - <<'PY'
import json
import os
from pathlib import Path

base = os.environ.get("INFERENCE_BENCH_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
r = {
    "models": [base],
    "datasets": [
        {"dataset": "TIGER-Lab/MMLU-Pro", "configs": ["default"], "splits": ["test"]},
    ],
}
p = Path("/tmp/baseline_precompute_hf_resources.json")
p.write_text(json.dumps(r, indent=2))
print(f"[baseline_precompute] wrote resources file to {p}")
PY

bash "${REPO_ROOT}/containers/download_hf_cache/download_hf_cache.sh" \
  --resources-file /tmp/baseline_precompute_hf_resources.json \
  --workers 4

if [ "${INFERENCE_BENCH_DOWNLOAD_ONLY:-0}" = "1" ]; then
  echo "[baseline_precompute] download-only complete; exiting."
  exit 0
fi

INNER_SH="${JOB_TMP}/inner_precompute.sh"
cat >"${INNER_SH}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

echo "[baseline_precompute] container preflight: python=$(command -v python || true)"

# Fix broken flashinfer stub. ptb_import_shims.py (in the bound venv) creates a
# fake flashinfer module with __spec__=None, which crashes vLLM's model registry.
# Fix by removing the shim file from the venv — it's only needed for SGLang.
echo "[baseline_precompute] checking flashinfer stub..."
SHIM_FILE="/venv/lib/python3.10/site-packages/ptb_import_shims.py"
if [ -f "${SHIM_FILE}" ] && grep -q "ptb_flashinfer_stub" "${SHIM_FILE}" 2>/dev/null; then
  echo "[baseline_precompute] removing ptb_import_shims.py (flashinfer stub breaks vLLM)"
  rm -f "${SHIM_FILE}" 2>/dev/null || true
  rm -f "${SHIM_FILE}c" 2>/dev/null || true  # .pyc
  find /venv -path "*/__pycache__/ptb_import_shims*" -delete 2>/dev/null || true
fi

# Fix huggingface-hub version if too new for transformers (flashinfer pulls >=1.0)
python3 -c "import transformers" 2>/dev/null || {
  echo "[baseline_precompute] fixing huggingface-hub version (downgrading to <1.0)..."
  pip install --quiet "huggingface-hub>=0.34.0,<1.0" 2>/dev/null || true
}

# The bound venv site-packages takes priority over the container's system packages.

VENV="/venv"
MARKER="${VENV}/.ptb_baselines_venv"
HPO_ENGINE="${HPO_BASELINE_ENGINE:-${BACKENDS_RAW:-}}"

# Some backend containers (e.g. the upstream TGI image) ship a ready-made venv at
# /usr/src/.venv with the driver deps already installed (transformers, requests, ...).
# Prefer that for plain baseline precompute, but keep HPO-on-TGI on the bound host
# venv so optimizer deps can be installed there just like vLLM/SGLang.
if [ -f "/usr/src/.venv/bin/activate" ] && [ -x "/usr/src/.venv/bin/python" ] \
  && ! { [ "${INFERENCE_BENCH_INNER_MODE:-precompute}" = "hpo_search" ] && [ "${HPO_ENGINE}" = "tgi" ]; }; then
  VENV="/usr/src/.venv"
  MARKER="${VENV}/.ptb_baselines_venv"
  echo "[baseline_precompute] using container's pre-built venv at ${VENV}"
elif [ "${INFERENCE_BENCH_INNER_MODE:-precompute}" = "hpo_search" ] && [ "${HPO_ENGINE}" = "tgi" ]; then
  echo "[baseline_precompute] using bound host venv for TGI HPO at ${VENV}"
fi

export HOME="/tmp/home"  # always use bound /tmp scratch; host /home/agent isn't writable inside this container and flashinfer's JIT cache defaults to $HOME/.cache
export TMPDIR="${TMPDIR:-/tmp/tmp}"
export TMP="${TMP:-/tmp/tmp}"
export TEMP="${TEMP:-/tmp/tmp}"
export PIP_CACHE_DIR="${VENV}/pip_cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
export INFERENCE_BENCH_RUNTIME_CACHE_DIR="${INFERENCE_BENCH_RUNTIME_CACHE_DIR:-/tmp/runtime_cache}"
# Ensure Apptainer --nv driver libraries win over any CUDA toolkit stubs.
export LD_LIBRARY_PATH="/.singularity.d/libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${HOME}" "${TMPDIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${INFERENCE_BENCH_RUNTIME_CACHE_DIR}"

# FlashInfer JIT can write into ~/.cache by default; keep it on a large bind-mounted path.
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${INFERENCE_BENCH_RUNTIME_CACHE_DIR}/flashinfer_cache}"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_JIT_CACHE_DIR:-${FLASHINFER_CACHE_DIR}}"
export FLASHINFER_BUILD_DIR="${FLASHINFER_BUILD_DIR:-${FLASHINFER_CACHE_DIR}}"
export FLASHINFER_TMPDIR="${FLASHINFER_TMPDIR:-${INFERENCE_BENCH_RUNTIME_CACHE_DIR}/flashinfer_tmp}"
mkdir -p "${FLASHINFER_CACHE_DIR}" "${FLASHINFER_TMPDIR}"

echo "[baseline_precompute] /dev/shm capacity:"
df -h /dev/shm || true

# If the scheduler didn't set CUDA_VISIBLE_DEVICES, only honor HTCondor's explicit
# assignment. Do not probe GPUs here: on partially broken hosts, probing can be flaky
# and forcing index 0 has repeatedly selected a dead device.
if [ "${INFERENCE_BENCH_NEEDS_GPU:-0}" = "1" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  picked=""
  if [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
    # Common HTCondor formats: "1", "1,2", "CUDA1", "CUDA1,CUDA2",
    # or GPU UUID-style tokens such as "GPU-xxxxxxxx-...".
    condor_raw="${_CONDOR_AssignedGPUs}"
    condor_norm="$(echo "${condor_raw}" | tr ';' ',' | tr -d '[:space:]')"
    condor_cuda_visible=""
    uuid_map=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      uuid_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
    fi
    proc_uuid_map="$(
      for info in /proc/driver/nvidia/gpus/*/information; do
        [ -r "${info}" ] || continue
        idx="$(awk -F':' '/Device Minor/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "${info}")"
        uuid="$(awk -F':' '/GPU UUID/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "${info}")"
        if [ -n "${idx}" ] && [ -n "${uuid}" ]; then
          echo "${idx},${uuid}"
        fi
      done
    )"
    for tok in $(echo "${condor_norm}" | tr ',' ' '); do
      v=""
      if [[ "${tok}" =~ ^CUDA([0-9]+)$ ]]; then
        v="${BASH_REMATCH[1]}"
      elif [[ "${tok}" =~ ^[0-9]+$ ]]; then
        v="${tok}"
      elif [[ "${tok}" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
        # Map UUID token/prefix to a numeric index expected by CUDA_VISIBLE_DEVICES.
        # Prefer /proc map (works even when nvidia-smi query is flaky), then fallback.
        v="$(echo "${proc_uuid_map}" | awk -F',' -v want="${tok}" '
          {
            idx=$1; uuid=$2;
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx);
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid);
            low_uuid=tolower(uuid); low_want=tolower(want);
            if (low_uuid == low_want || index(low_uuid, low_want) == 1) { print idx; exit 0; }
          }')"
        if [ -z "${v}" ]; then
          v="$(echo "${uuid_map}" | awk -F',' -v want="${tok}" '
          {
            idx=$1; uuid=$2;
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx);
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid);
            low_uuid=tolower(uuid); low_want=tolower(want);
            if (low_uuid == low_want || index(low_uuid, low_want) == 1) { print idx; exit 0; }
          }')"
        fi
      fi
      if [ -n "${v}" ]; then
        if [ -n "${condor_cuda_visible}" ]; then
          condor_cuda_visible="${condor_cuda_visible},${v}"
        else
          condor_cuda_visible="${v}"
        fi
      fi
    done
    if [ -n "${condor_cuda_visible}" ]; then
      picked="${condor_cuda_visible}"
      export CUDA_VISIBLE_DEVICES="${picked}"
      echo "[baseline_precompute] NOTE: CUDA_VISIBLE_DEVICES was empty; using _CONDOR_AssignedGPUs=${_CONDOR_AssignedGPUs} -> ${CUDA_VISIBLE_DEVICES}" >&2
    else
      echo "[baseline_precompute] NOTE: _CONDOR_AssignedGPUs is set but not mappable (${_CONDOR_AssignedGPUs}); trying nvidia-smi table fallback." >&2
    fi
  fi

  if [ -z "${picked}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    # Parse the first actually listed GPU index from nvidia-smi table output.
    # This avoids direct per-index probing on partially broken hosts.
    table_idx="$(
      (nvidia-smi 2>/dev/null || true) | awk '
        /^\|[[:space:]]*[0-9]+[[:space:]]+NVIDIA/ {
          line=$0
          sub(/^\|[[:space:]]*/, "", line)
          if (match(line, /^[0-9]+/)) {
            print substr(line, RSTART, RLENGTH)
            exit
          }
        }'
    )"
    if [ -n "${table_idx}" ]; then
      picked="${table_idx}"
      export CUDA_VISIBLE_DEVICES="${picked}"
      echo "[baseline_precompute] NOTE: CUDA_VISIBLE_DEVICES was empty; using first listed nvidia-smi table GPU index=${CUDA_VISIBLE_DEVICES}" >&2
    fi
  fi

  if [ -z "${picked}" ]; then
    echo "[baseline_precompute] NOTE: CUDA_VISIBLE_DEVICES was empty and no reliable GPU mapping was found; leaving CUDA_VISIBLE_DEVICES unset." >&2
  fi
fi

venv_ready=0
if [ -x "${VENV}/bin/python" ] && [ -f "${VENV}/bin/activate" ]; then
  venv_ready=1
fi

if [ "${venv_ready}" != "1" ]; then
  lockdir="${VENV}/.bootstrap.lockdir"
  for i in $(seq 1 120); do
    if mkdir "${lockdir}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [ ! -d "${lockdir}" ]; then
    echo "[baseline_precompute] ERROR: could not acquire lock: ${lockdir}" >&2
    exit 2
  fi
  cleanup_lock() { rmdir "${lockdir}" 2>/dev/null || true; }
  trap cleanup_lock EXIT

  if [ ! -f "${VENV}/bin/activate" ]; then
    echo "[baseline_precompute] creating/repairing venv at ${VENV} (host bind)"
    python3 -m venv --clear --system-site-packages "${VENV}"
    touch "${MARKER}" || true
  fi
fi

if [ ! -f "${VENV}/bin/activate" ]; then
  echo "[baseline_precompute] ERROR: venv is missing ${VENV}/bin/activate after bootstrap/repair." >&2
  echo "[baseline_precompute] Delete ${VENV} on host and rerun (or unset INFERENCE_BENCH_SKIP_VENV_BOOTSTRAP)." >&2
  exit 3
fi

source "${VENV}/bin/activate"

if python - <<'PY'
import importlib.metadata as m
import sys

try:
    import transformers  # noqa: F401
except Exception as e:
    msg = f"{type(e).__name__}: {e}"
    print(f"[baseline_precompute] transformers import failed: {msg}", file=sys.stderr)
    if "huggingface_hub" in msg or "huggingface-hub" in msg:
        raise SystemExit(42)
    raise SystemExit(1)

try:
    v = m.version("huggingface-hub")
except Exception:
    v = "unknown"
print(f"[baseline_precompute] huggingface-hub={v}")
PY
then
  :
else
  transformers_check_rc=$?
  if [ "${transformers_check_rc}" -eq 42 ]; then
    echo "[baseline_precompute] repairing huggingface-hub for transformers compatibility (pinning <1.0)"
    python -m pip install --no-cache-dir "huggingface-hub>=0.34.0,<1.0"
    python - <<'PY'
import importlib.metadata as m
import transformers  # noqa: F401
print(f"[baseline_precompute] huggingface-hub(repaired)={m.version('huggingface-hub')}")
PY
  else
    echo "[baseline_precompute] ERROR: transformers import failed for non-huggingface_hub reason." >&2
    exit 4
  fi
fi

# SGLang may hard-require a working `nvidia-smi` binary to read GPU memory.
# Some Apptainer images include a wrapper that exits non-zero when host
# binaries are not bind-mounted. Provide a runtime shim backed by torch CUDA
# queries when `nvidia-smi` is missing or non-functional.
if [ "${INFERENCE_BENCH_NEEDS_GPU:-0}" = "1" ] && [ "${INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS:-0}" != "1" ]; then
  nvidia_smi_ok=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >/dev/null 2>&1; then
      nvidia_smi_ok=1
    fi
  fi
  if [ "${nvidia_smi_ok}" != "1" ]; then
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

gpu_count = torch.cuda.device_count()
props = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]

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
            elif f == "driver_version":
                values.append("unknown")
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
    total_mib = int(p.total_memory // (1024 * 1024))
    print(f"GPU {idx}: {p.name} {total_mib} MiB")
PY
EOSHIM
    chmod +x "${shim_dir}/nvidia-smi"
    export PATH="${shim_dir}:${PATH}"
    echo "[baseline_precompute] installed runtime nvidia-smi shim at ${shim_dir}/nvidia-smi"
  fi
elif [ "${INFERENCE_BENCH_NEEDS_GPU:-0}" = "1" ]; then
  echo "[baseline_precompute] NOTE: runtime shims disabled; not installing nvidia-smi shim."
fi

if [ "${INFERENCE_BENCH_NEEDS_GPU:-0}" = "1" ]; then
  echo "[baseline_precompute] gpu preflight: checking NVIDIA devices/driver..."
  echo "[baseline_precompute] env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-}"
  ls -l /dev/nvidia* 2>/dev/null || echo "[baseline_precompute] NOTE: no /dev/nvidia* visible"
  if [ -r /proc/driver/nvidia/version ]; then
    echo "[baseline_precompute] /proc/driver/nvidia/version:"
    cat /proc/driver/nvidia/version
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "[baseline_precompute] NOTE: nvidia-smi not found in container PATH"
  fi
  _ptb_cuda_preflight_python() {
  python - <<'PY'
import sys
import ctypes
import torch
import os
import re

try:
    ctypes.CDLL("libcuda.so.1")
    print("[baseline_precompute] libcuda.so.1 load=OK")
except OSError as e:
    print("[baseline_precompute] libcuda.so.1 load=FAIL (%s)" % (e,))

try:
    dv = getattr(torch._C, "_cuda_getDriverVersion", lambda: None)()
except Exception as e:
    dv = "error:%s:%s" % (type(e).__name__, e)

ok = bool(torch.cuda.is_available())
torch_cuda = getattr(torch.version, "cuda", None)
print(
    "[baseline_precompute] torch=%s torch_cuda=%s cuda_driver_version=%s cuda_available=%s"
    % (torch.__version__, torch_cuda, dv, ok)
)

if ok:
    try:
        torch.zeros(1, device="cuda")
        print("[baseline_precompute] cuda_smoke=OK")
    except Exception as e:
        print("[baseline_precompute] cuda_smoke=FAIL (%s:%s)" % (type(e).__name__, e))
        sys.exit(1)
else:
    print("[baseline_precompute] HINT: If driver is 545.x, torch cu12.8/cu12.4 builds won't work. Install a cu121 torch into the bound venv.")

sys.exit(0 if ok else 1)
PY
  }

  if ! _ptb_cuda_preflight_python; then
    preflight_rc=$?
    if [ "${preflight_rc}" -eq 64 ]; then
      exit 64
    fi
    echo "[baseline_precompute] ERROR: CUDA runtime not usable in container for this allocation." >&2
    echo "[baseline_precompute] ERROR: Exiting without retry/fallback mask probing." >&2
    exit "${preflight_rc}"
  fi
fi

if [ "${INFERENCE_BENCH_DOWNLOAD_ONLY:-0}" = "1" ]; then
  echo "[baseline_precompute] download-only complete; exiting."
  exit 0
fi

BACKENDS_RAW="${INFERENCE_BENCH_BACKENDS_RAW:-vllm,torch,sglang,tgi}"
BACKENDS_SPACE="${BACKENDS_RAW//,/ }"

_ptb_is_float() {
  local v="${1:-}"
  if [ -z "${v}" ] || [ "${v}" = "UNDEFINED" ]; then
    return 1
  fi
  [[ "${v}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

SERVER_START_TIMEOUT_S="${INFERENCE_BENCH_SERVER_START_TIMEOUT_S:-600}"
if ! _ptb_is_float "${SERVER_START_TIMEOUT_S}"; then
  SERVER_START_TIMEOUT_S="600"
fi
echo "[baseline_precompute] server_start_timeout_s=${SERVER_START_TIMEOUT_S}"

SKIP_FLAGS=""
if [ "${INFERENCE_BENCH_SKIP_SPEED:-0}" = "1" ]; then
  SKIP_FLAGS="${SKIP_FLAGS} --skip-speed"
fi
if [ "${INFERENCE_BENCH_SKIP_QUALITY:-0}" = "1" ]; then
  SKIP_FLAGS="${SKIP_FLAGS} --skip-quality"
fi

INNER_MODE="${INFERENCE_BENCH_INNER_MODE:-precompute}"
echo "[baseline_precompute] inner_mode=${INNER_MODE}"
# Outer shell already rewrote INFERENCE_BENCH_INNER_MODE=random_search_b to
# random_search before apptainer exec (so the /fast bind-mount gets added).
# Inner only needs to pick up the scenario restriction marker it set.
_RSB_FORCE="${INFERENCE_BENCH_SC_RESTRICT:-}"
if [ -n "${_RSB_FORCE}" ]; then
  echo "[baseline_precompute] scenario restriction: ${_RSB_FORCE}"
fi
if [ "${INNER_MODE}" = "random_search" ]; then
  # Random-search ablation over vLLM configurations (Appendix I.6).
  # Reuses all of the container / venv / HF-cache plumbing above, then
  # hands off to src.eval.inference.random_search_vllm which runs its own
  # per-config server-launch loop.
  # HTCondor sets unbound $ENV() vars to the literal string "UNDEFINED"
  # rather than leaving them unset, so ${VAR:-default} doesn't help.
  # Use eval with :- to safely handle truly-unset vars under set -u.
  _rs_clean() { eval "local v=\${$1:-}"; [[ "$v" == "UNDEFINED" || -z "$v" ]] && echo "$2" || echo "$v"; }
  RS_SCENARIO_FLAG=""
  _rs_sc="$(_rs_clean RANDOM_SEARCH_SCENARIOS "${_RSB_FORCE}")"
  if [ -n "${_rs_sc}" ]; then
    # shellcheck disable=SC2086
    RS_SCENARIO_FLAG="--scenarios ${_rs_sc//,/ }"
  fi
  python -u -m src.eval.inference.random_search_vllm \
    --base-model "${INFERENCE_BENCH_BASE_MODEL}" \
    --max-model-len "${INFERENCE_BENCH_MAX_MODEL_LEN}" \
    --server-start-timeout-s "${SERVER_START_TIMEOUT_S}" \
    --dataset-seed "${INFERENCE_BENCH_DATASET_SEED}" \
    --num-configs "$(_rs_clean RANDOM_SEARCH_N 32)" \
    --search-seed "$(_rs_clean RANDOM_SEARCH_SEED 248)" \
    --out-root "$(_rs_clean RANDOM_SEARCH_OUT_ROOT /fast/${USER}/random_search_vllm)" \
    --start-from "$(_rs_clean RANDOM_SEARCH_START_FROM 0)" \
    --eval-timeout "$(_rs_clean RANDOM_SEARCH_EVAL_TIMEOUT 1800)" \
    ${RS_SCENARIO_FLAG}
elif [ "${INNER_MODE}" = "hpo_search" ]; then
  _hpo_clean() { eval "local v=\${$1:-}"; [[ "$v" == "UNDEFINED" || -z "$v" ]] && echo "$2" || echo "$v"; }
  python -u -m src.eval.inference.hpo_search_baselines \
    --method "$(_hpo_clean HPO_BASELINE_METHOD random)" \
    --engine "$(_hpo_clean HPO_BASELINE_ENGINE "${BACKENDS_RAW%%,*}")" \
    --scenario "$(_hpo_clean HPO_BASELINE_SCENARIO a)" \
    --seed-id "$(_hpo_clean HPO_BASELINE_SEED_ID 0)" \
    --seed-dev "$(_hpo_clean HPO_BASELINE_SEED_DEV 21)" \
    --seed-eval "$(_hpo_clean HPO_BASELINE_SEED_EVAL 1337)" \
    --quality-seed "$(_hpo_clean HPO_BASELINE_QUALITY_SEED "${INFERENCE_BENCH_QUALITY_SEED}")" \
    --base-model "${INFERENCE_BENCH_BASE_MODEL}" \
    --max-model-len "${INFERENCE_BENCH_MAX_MODEL_LEN}" \
    --budget-s "$(_hpo_clean HPO_BASELINE_BUDGET_S 7200)" \
    --quick-eval-timeout-s "$(_hpo_clean HPO_BASELINE_QUICK_EVAL_TIMEOUT_S 180)" \
    --server-start-timeout-s "$(_hpo_clean HPO_BASELINE_SERVER_START_TIMEOUT_S 300)" \
    --first-server-start-timeout-s "$(_hpo_clean HPO_BASELINE_FIRST_SERVER_START_TIMEOUT_S 360)" \
    --server-initial-delay-s "$(_hpo_clean HPO_BASELINE_SERVER_INITIAL_DELAY_S 60)" \
    --out-root "$(_hpo_clean HPO_BASELINE_OUT_ROOT "${INFERENCE_BENCH_RESULTS_DIR}/hpo_search_baselines")"
else
  python -u -m src.eval.inference.precompute_all_baselines \
    --cache-quality-samples \
    --backends ${BACKENDS_SPACE} \
    --base-model "${INFERENCE_BENCH_BASE_MODEL}" \
    --max-model-len "${INFERENCE_BENCH_MAX_MODEL_LEN}" \
    --server-start-timeout-s "${SERVER_START_TIMEOUT_S}" \
    --dataset-seed "${INFERENCE_BENCH_DATASET_SEED}" \
    --quality-seed "${INFERENCE_BENCH_QUALITY_SEED}" \
    --mmlupro-n "${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}" \
    --quality-concurrency "${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
    ${SKIP_FLAGS}
fi
SH
chmod +x "${INNER_SH}"

DATA_BIND=""
if [ -d "/data" ]; then
  DATA_BIND="--bind /data:/data:ro"
fi

SHM_BIND=""
SHM_BIND_SOURCE="/dev/shm"
if [ -n "${SHM_BIND_SOURCE}" ] && [ -d "${SHM_BIND_SOURCE}" ]; then
  SHM_BIND="--bind ${SHM_BIND_SOURCE}:/dev/shm"
fi
echo "[baseline_precompute] shm_bind_source=${SHM_BIND_SOURCE}"

# Random-search driver writes per-config artifacts under RANDOM_SEARCH_OUT_ROOT.
# That path is on the host filesystem and is NOT one of the canonical binds
# (repo, venv, HF cache, /tmp), so without an explicit bind-mount the writes
# would fall through into apptainer's writable-tmpfs overlay and exhaust its
# RAM-backed capacity within a couple of configs. Bind-mount it explicitly
# when in random_search mode; the outer submit wrapper creates the dir on the
# host before condor_submit_bid, so it always exists.
# random_search_b is a Sc.B-only variant of random_search. The outer shell has
# to normalize it to "random_search" BEFORE the bind decision below, otherwise
# /fast/${USER}/random_search_vllm is not bound and writes hit the 16 MB apptainer
# overlay → ENOSPC. Also set INFERENCE_BENCH_SC_RESTRICT so the inner heredoc
# can force --scenarios without needing to propagate RANDOM_SEARCH_SCENARIOS
# (which doesn't survive condor→worker env handling reliably).
if [ "${INFERENCE_BENCH_INNER_MODE:-}" = "random_search_b" ]; then
  export INFERENCE_BENCH_INNER_MODE="random_search"
  export INFERENCE_BENCH_SC_RESTRICT="inference_scenario_b_output_heavy"
fi

RANDOM_SEARCH_BIND=""
if [ "${INFERENCE_BENCH_INNER_MODE:-precompute}" = "random_search" ]; then
  RS_OUT="${RANDOM_SEARCH_OUT_ROOT:-/fast/${USER}/random_search_vllm}"
  mkdir -p "${RS_OUT}"
  RANDOM_SEARCH_BIND="--bind ${RS_OUT}:${RS_OUT}"
  echo "[baseline_precompute] random_search_out_root=${RS_OUT}"
fi

HPO_SEARCH_BIND=""
if [ "${INFERENCE_BENCH_INNER_MODE:-precompute}" = "hpo_search" ]; then
  HPO_OUT="${HPO_BASELINE_OUT_ROOT:-${INFERENCE_BENCH_RESULTS_DIR}/hpo_search_baselines}"
  mkdir -p "${HPO_OUT}"
  export HPO_BASELINE_OUT_ROOT="${HPO_OUT}"
  HPO_SEARCH_BIND="--bind ${HPO_OUT}:${HPO_OUT}"
  echo "[baseline_precompute] hpo_baseline_out_root=${HPO_OUT}"
fi

echo "[baseline_precompute] starting apptainer precompute..."
NV_FLAG=""
if [ "${NEEDS_GPU}" = "1" ]; then
  # Default to legacy --nv for robustness on clusters with partially unhealthy GPUs.
  # Allow explicit opt-in to --nvccli via env if desired.
  if [ "${INFERENCE_BENCH_APPTAINER_GPU_MODE:-nv}" = "nvccli" ] && apptainer exec --help 2>/dev/null | grep -q -- '--nvccli'; then
    NV_FLAG="--nvccli"
  else
    NV_FLAG="--nv"
  fi
fi
echo "[baseline_precompute] apptainer_gpu_flag=${NV_FLAG:-none}"

CUDA_ENV_FLAGS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_ENV_FLAGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
if [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
  CUDA_ENV_FLAGS+=(--env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}")
fi

# The TGI upstream image ships its python runtime + text-generation-launcher in a
# uv-managed venv at /usr/src/.venv, which isn't on the default PATH we inject.
CONTAINER_PATH="/root/.local/bin:/home/agent/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
case "${CONTAINER_SIF}" in
  *inference_baselines_tgi*) CONTAINER_PATH="/usr/src/.venv/bin:${CONTAINER_PATH}" ;;
esac

# sgl_kernel's prebuilt sm90 .so links libnuma.so.1, which we don't install in
# the sglang container. Bind the host library into the container to satisfy the loader.
EXTRA_BINDS=()
EXTRA_ENVS=()
case "${CONTAINER_SIF}" in
  *inference_baselines_sglang*)
    if [ -f /lib/x86_64-linux-gnu/libnuma.so.1 ]; then
      EXTRA_BINDS+=(--bind /lib/x86_64-linux-gnu/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libnuma.so.1)
    fi
    ;;
esac

set +e
apptainer exec \
  ${NV_FLAG} \
  -c \
  --cleanenv \
  --env PATH="${CONTAINER_PATH}" \
  --env PYTHONNOUSERSITE="1" \
  --env PIP_CONFIG_FILE="/dev/null" \
  --env TMPDIR="/tmp/tmp" \
  --env TMP="/tmp/tmp" \
  --env TEMP="/tmp/tmp" \
  --env XDG_CACHE_HOME="/tmp/xdg_cache" \
  --env SOFT_FILELOCK="1" \
  --env PYTHONPATH="/opt/filelock_workarounds/soft_file_locks" \
  --env INFERENCE_BENCH_DEBUG_STACK_DUMP="${INFERENCE_BENCH_DEBUG_STACK_DUMP:-0}" \
  --env INFERENCE_BENCH_SERVER_START_TIMEOUT_S="${INFERENCE_BENCH_SERVER_START_TIMEOUT_S:-}" \
  --env INFERENCE_BENCH_STUB_OUTLINES="${INFERENCE_BENCH_STUB_OUTLINES:-1}" \
  --env INFERENCE_BENCH_DISABLE_HF_LOCKS="${INFERENCE_BENCH_DISABLE_HF_LOCKS:-1}" \
  --env INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS="${INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS:-0}" \
  --env PTB_DISABLE_IMPORT_SHIMS="${PTB_DISABLE_IMPORT_SHIMS:-0}" \
  "${CUDA_ENV_FLAGS[@]}" \
  --env INFERENCE_BENCH_ALLOW_HF_DOWNLOAD="1" \
  --env HF_HOME="${HF_CACHE_IN_CONTAINER}" \
  --env HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  --env HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" \
  --env HF_HUB_DISABLE_MMAP="${INFERENCE_BENCH_HF_HUB_DISABLE_MMAP:-1}" \
  --env HF_HUB_CACHE="${HF_CACHE_IN_CONTAINER}/hub" \
  --env HUGGINGFACE_HUB_CACHE="${HF_CACHE_IN_CONTAINER}/hub" \
  --env TRANSFORMERS_CACHE="${HF_CACHE_IN_CONTAINER}/hub" \
  --env HF_DATASETS_CACHE="${HF_CACHE_IN_CONTAINER}/datasets" \
  --env INFERENCE_BENCH_BASE_MODEL="${BASE_MODEL}" \
  --env INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN}" \
  --env INFERENCE_BENCH_INPUT_TOKEN_MARGIN="${INFERENCE_BENCH_INPUT_TOKEN_MARGIN:-16}" \
  --env INFERENCE_BENCH_RESULTS_DIR="${INFERENCE_BENCH_RESULTS_DIR}" \
  --env INFERENCE_BENCH_DATASET_SEED="${INFERENCE_BENCH_DATASET_SEED}" \
  --env INFERENCE_BENCH_QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED}" \
  --env INFERENCE_BENCH_QUALITY_MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}" \
  --env INFERENCE_BENCH_QUALITY_CONCURRENCY="${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
  --env INFERENCE_BENCH_BACKENDS_RAW="${BACKENDS_RAW}" \
  --env INFERENCE_BENCH_NEEDS_GPU="${NEEDS_GPU}" \
  --env INFERENCE_BENCH_INNER_MODE="${INFERENCE_BENCH_INNER_MODE:-precompute}" \
  --env INFERENCE_BENCH_SC_RESTRICT="${INFERENCE_BENCH_SC_RESTRICT:-}" \
  --env HPO_BASELINE_METHOD="${HPO_BASELINE_METHOD:-}" \
  --env HPO_BASELINE_ENGINE="${HPO_BASELINE_ENGINE:-}" \
  --env HPO_BASELINE_SCENARIO="${HPO_BASELINE_SCENARIO:-}" \
  --env HPO_BASELINE_SEED_ID="${HPO_BASELINE_SEED_ID:-}" \
  --env HPO_BASELINE_SEED_DEV="${HPO_BASELINE_SEED_DEV:-}" \
  --env HPO_BASELINE_SEED_EVAL="${HPO_BASELINE_SEED_EVAL:-}" \
  --env HPO_BASELINE_QUALITY_SEED="${HPO_BASELINE_QUALITY_SEED:-}" \
  --env HPO_BASELINE_BUDGET_S="${HPO_BASELINE_BUDGET_S:-}" \
  --env HPO_BASELINE_QUICK_EVAL_TIMEOUT_S="${HPO_BASELINE_QUICK_EVAL_TIMEOUT_S:-}" \
  --env HPO_BASELINE_SERVER_START_TIMEOUT_S="${HPO_BASELINE_SERVER_START_TIMEOUT_S:-}" \
  --env HPO_BASELINE_FIRST_SERVER_START_TIMEOUT_S="${HPO_BASELINE_FIRST_SERVER_START_TIMEOUT_S:-}" \
  --env HPO_BASELINE_SERVER_INITIAL_DELAY_S="${HPO_BASELINE_SERVER_INITIAL_DELAY_S:-}" \
  --env HPO_BASELINE_OUT_ROOT="${HPO_BASELINE_OUT_ROOT:-}" \
  --env HPO_BASELINE_AUTO_INSTALL_DEPS="${HPO_BASELINE_AUTO_INSTALL_DEPS:-1}" \
  --env HPO_BASELINE_OVERWRITE="${HPO_BASELINE_OVERWRITE:-0}" \
  --env HPO_BASELINE_MAX_TRIALS="${HPO_BASELINE_MAX_TRIALS:-}" \
  --env RANDOM_SEARCH_CONFIG_IDS="${RANDOM_SEARCH_CONFIG_IDS:-}" \
  --env RANDOM_SEARCH_N="${RANDOM_SEARCH_N:-}" \
  --env RANDOM_SEARCH_SEED="${RANDOM_SEARCH_SEED:-}" \
  --env RANDOM_SEARCH_OUT_ROOT="${RANDOM_SEARCH_OUT_ROOT:-}" \
  --env RANDOM_SEARCH_START_FROM="${RANDOM_SEARCH_START_FROM:-}" \
  --env RANDOM_SEARCH_EVAL_TIMEOUT="${RANDOM_SEARCH_EVAL_TIMEOUT:-}" \
  --env _CONDOR_AssignedGPUs="${_CONDOR_AssignedGPUs:-}" \
  --env INFERENCE_BENCH_RUNTIME_CACHE_DIR="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}" \
  --env INFERENCE_BENCH_DOWNLOAD_ONLY="${DOWNLOAD_ONLY}" \
  --env VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
  --env FLASHINFER_CACHE_DIR="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}/flashinfer_cache" \
  --env FLASHINFER_JIT_CACHE_DIR="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}/flashinfer_cache" \
  --env FLASHINFER_BUILD_DIR="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}/flashinfer_cache" \
  --env FLASHINFER_TMPDIR="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}/flashinfer_tmp" \
  --env OUTLINES_CACHE_DIR="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}/outlines_cache" \
  --env DISKCACHE_DIRECTORY="/tmp/runtime_cache/${CLUSTER_ID}_${RANDOM_UUID}/outlines_cache" \
  --env TVM_FFI_CACHE_DIR="/dev/shm/ptb_tvm_ffi_${CLUSTER_ID}_${RANDOM_UUID}" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env HUGGINGFACEHUB_API_TOKEN="${HF_TOKEN:-}" \
  --bind "${JOB_TMP}:/tmp" \
  --bind "${VENV_HOST}:${VENV_IN_CONTAINER}" \
  --bind "${HF_CACHE_HOST}:${HF_CACHE_IN_CONTAINER}" \
  --bind "${REPO_ROOT}:${REPO_ROOT}" \
  ${DATA_BIND} \
  ${SHM_BIND} \
  ${RANDOM_SEARCH_BIND} \
  ${HPO_SEARCH_BIND} \
  "${EXTRA_BINDS[@]}" \
  "${EXTRA_ENVS[@]}" \
  --pwd "${REPO_ROOT}" \
  --writable-tmpfs \
  "${CONTAINER_SIF}" \
  bash "/tmp/inner_precompute.sh"
rc=$?
set -e

if [ "${rc}" -ne 0 ]; then
  exit "${rc}"
fi

echo "[baseline_precompute] done."
echo "[baseline_precompute] artifacts written under:"
echo "  - src/eval/inference/quality_data/"
echo "  - src/eval/inference/baselines/speed/"
echo "  - src/eval/inference/baselines/quality/"

rm -rf "${TMP_SUBDIR}"
