#!/bin/bash

export EVALUATION_TASK="$1"
AGENT="$2"
BASE_MODEL="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
# If $7 is a known starting_point value, use it; otherwise assume starting_point
# was omitted from the submission (older callers) and treat $7 as agent_seed.
case "${7:-}" in
    default|vllm_running|bare)
        STARTING_POINT="${7}"
        AGENT_SEED="${8:-${INFERENCE_BENCH_DATASET_SEED:-248}}"
        EVAL_SEED="${9:-${INFERENCE_BENCH_DATASET_SEED:-248}}"
        ;;
    *)
        STARTING_POINT="${INFERENCE_BENCH_STARTING_POINT:-default}"
        AGENT_SEED="${7:-${INFERENCE_BENCH_DATASET_SEED:-248}}"
        EVAL_SEED="${8:-${INFERENCE_BENCH_DATASET_SEED:-248}}"
        ;;
esac

# Convert NUM_HOURS (may be fractional, e.g. 0.17) to an integer second count
# once, up-front. Bash $((...)) arithmetic cannot handle floats, so every
# downstream user should prefer NUM_SECONDS.
NUM_SECONDS="$(awk -v h="${NUM_HOURS}" 'BEGIN{printf "%d", h*3600}')"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/env.sh" ]; then
    source "${REPO_ROOT}/env.sh"
fi

source src/commit_utils/set_env_vars.sh

RESULT_PREFIX_SAFE="$(printf '%s' "$BASE_MODEL" | tr '/:' '_')"
AGENT_CONFIG_SAFE="$(printf '%s' "$AGENT_CONFIG" | tr '/:' '_')"
RANDOM_UUID=$(uuidgen)

SP_SUFFIX=""
if [ "${STARTING_POINT}" != "default" ]; then
    SP_SUFFIX="_sp-${STARTING_POINT}"
fi
export EVAL_DIR="${INFERENCE_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${INFERENCE_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}${SP_SUFFIX}_${CLUSTER_ID}"

mkdir -p "${EVAL_DIR}"
export EVAL_DIR="$(realpath "${EVAL_DIR}")"

CHECKPOINT_DIR="${EVAL_DIR}/checkpoint"
CHECKPOINT_REMAINING_SECONDS=""
CHECKPOINT_CLAUDE_RESTORED=0

exec >"${EVAL_DIR}/output.log" 2>"${EVAL_DIR}/error.log"

CURRENT_AGENT_PID=""

_checkpoint_and_exit() {
    echo "[harness] SIGTERM received — saving checkpoint to ${CHECKPOINT_DIR}"
    if [ -n "${CURRENT_AGENT_PID:-}" ]; then
        echo "[harness] stopping agent container (pid=${CURRENT_AGENT_PID})"
        kill -TERM "${CURRENT_AGENT_PID}" 2>/dev/null || true
        sleep 5
        kill -KILL "${CURRENT_AGENT_PID}" 2>/dev/null || true
    fi
    local remaining=0
    if [ -n "${AGENT_DEADLINE_TS:-}" ]; then
        remaining=$(( AGENT_DEADLINE_TS - $(date +%s) ))
        (( remaining < 0 )) && remaining=0
    fi
    mkdir -p "${CHECKPOINT_DIR}"
    printf '%s\n' "${remaining}" > "${CHECKPOINT_DIR}/remaining_seconds"
    if [ -n "${JOB_DIR:-}" ]; then
        [ -d "${JOB_DIR}/task" ]    && cp -a "${JOB_DIR}/task"    "${CHECKPOINT_DIR}/" 2>/dev/null || true
        [ -d "${JOB_DIR}/.claude" ] && cp -a "${JOB_DIR}/.claude" "${CHECKPOINT_DIR}/" 2>/dev/null || true
    fi
    echo "[harness] checkpoint saved: remaining=${remaining}s contents=$(ls "${CHECKPOINT_DIR}" 2>/dev/null | tr '\n' ' ')"
    exit 85
}
trap _checkpoint_and_exit SIGTERM

echo "$@"
export REPO_ROOT="$(pwd)"

append_path_if_missing() {
    local dir="${1:-}"
    if [ -z "${dir}" ] || [ ! -d "${dir}" ]; then
        return 0
    fi
    case ":${PATH:-}:" in
        *":${dir}:"*) ;;
        *)
            if [ -n "${PATH:-}" ]; then
                PATH="${PATH}:${dir}"
            else
                PATH="${dir}"
            fi
            ;;
    esac
}

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
        append_path_if_missing "${d}"
    done
    export PATH
    hash -r
}

ensure_required_host_tools() {
    local required_tools=(
        bash
        python3
        mkdir
        cp
        date
        df
        du
        find
        ps
        rm
        setsid
        tr
        tee
        timeout
        apptainer
    )
    local missing=()
    local tool
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            missing+=("${tool}")
        fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then
        return 0
    fi

    echo "[env] missing required host tools before PATH fallback: ${missing[*]}"
    restore_core_host_path

    missing=()
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            missing+=("${tool}")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "[fatal] required host tools unavailable after PATH fallback: ${missing[*]}"
        echo "[fatal] PATH=${PATH:-}"
        exit 1
    fi

    echo "[env] restored PATH after fallback: ${PATH}"
}

maybe_load_host_cuda_module() {
    # Some clusters expose only core GPU runtime by default and require
    # `module load cuda` for full user-space tooling.
    if [ "${INFERENCE_BENCH_LOAD_CUDA_MODULE:-1}" = "0" ]; then
        echo "[gpu] skipping module load (INFERENCE_BENCH_LOAD_CUDA_MODULE=0)"
        return 0
    fi
    if [ -n "${INFERENCE_BENCH_SKIP_MODULE_LOAD:-}" ] && [ "${INFERENCE_BENCH_SKIP_MODULE_LOAD}" != "0" ]; then
        echo "[gpu] skipping module load (INFERENCE_BENCH_SKIP_MODULE_LOAD=${INFERENCE_BENCH_SKIP_MODULE_LOAD})"
        return 0
    fi

    local pre_module_path="${PATH:-}"
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
    fi
    if command -v module >/dev/null 2>&1; then
        local cuda_module="${INFERENCE_BENCH_CUDA_MODULE:-cuda}"
        if module load "${cuda_module}" >/dev/null 2>&1; then
            # Some module stacks overwrite PATH; merge the pre-module PATH back.
            local old_ifs="${IFS}"
            IFS=':'
            local p
            for p in ${pre_module_path}; do
                append_path_if_missing "${p}"
            done
            IFS="${old_ifs}"
            restore_core_host_path
            echo "[gpu] loaded module ${cuda_module}"
        else
            echo "[gpu] module load ${cuda_module} failed; continuing with existing environment"
        fi
    else
        echo "[gpu] module command unavailable; continuing with existing environment"
    fi
}

maybe_load_host_cuda_module
restore_core_host_path
ensure_required_host_tools

# Pick a per-job server endpoint to avoid localhost port collisions when multiple jobs
# are scheduled on the same node (Apptainer uses the host network namespace by default).
SERVER_HOST="${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${INFERENCE_BENCH_SERVER_PORT:-$(python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
export INFERENCE_BENCH_SERVER_HOST="${SERVER_HOST}"
export INFERENCE_BENCH_SERVER_PORT="${SERVER_PORT}"
export INFERENCE_BENCH_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"

TMP_ROOT="/tmp"
export TMP_SUBDIR="${TMP_ROOT}/inferencebench_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"

echo "============================="
echo "======== SCRATCH INFO ======="
echo "============================="
echo "[scratch] INFERENCE_BENCH_TMP_ROOT=${INFERENCE_BENCH_TMP_ROOT:-}"
echo "[scratch] _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-}"
echo "[scratch] TMPDIR=${TMPDIR:-} TMP=${TMP:-} TEMP=${TEMP:-}"
echo "[scratch] TMP_ROOT=${TMP_ROOT}"
echo "[scratch] TMP_SUBDIR=${TMP_SUBDIR}"
echo "[scratch] JOB_DIR=${JOB_DIR}"
echo "[scratch] JOB_TMP=${JOB_TMP}"
echo "[scratch] results_dir=${INFERENCE_BENCH_RESULTS_DIR:-}"
echo "[scratch] fs_type(TMP_ROOT)=$(stat -f -c %T \"${TMP_ROOT}\" 2>/dev/null || echo unknown)"
echo "[scratch] fs_type(/tmp)=$(stat -f -c %T /tmp 2>/dev/null || echo unknown)"
echo "[scratch] df -h TMP_ROOT:"
df -h "${TMP_ROOT}" || true
echo "[scratch] df -i TMP_ROOT:"
df -i "${TMP_ROOT}" || true
echo "[scratch] df -h /tmp:"
df -h /tmp || true
echo "[scratch] df -i /tmp:"
df -i /tmp || true
if [ -n "${INFERENCE_BENCH_RESULTS_DIR:-}" ]; then
  echo "[scratch] df -h results_dir:"
  df -h "${INFERENCE_BENCH_RESULTS_DIR}" || true
fi
echo "============================="

mkdir -p "${JOB_DIR}"
if [ -f "${CHECKPOINT_DIR}/remaining_seconds" ]; then
    echo "[harness] checkpoint found — restoring state from ${CHECKPOINT_DIR}"
    CHECKPOINT_REMAINING_SECONDS="$(cat "${CHECKPOINT_DIR}/remaining_seconds")"
    echo "[harness] checkpoint remaining_seconds=${CHECKPOINT_REMAINING_SECONDS}"
    [ -d "${CHECKPOINT_DIR}/task" ]    && cp -a "${CHECKPOINT_DIR}/task"    "${JOB_DIR}/" 2>/dev/null || true
    if [ -d "${CHECKPOINT_DIR}/.claude" ]; then
        cp -a "${CHECKPOINT_DIR}/.claude" "${JOB_DIR}/" 2>/dev/null || true
        CHECKPOINT_CLAUDE_RESTORED=1
    fi
    echo "[harness] checkpoint restore complete (claude_restored=${CHECKPOINT_CLAUDE_RESTORED})"
fi
mkdir -p "${JOB_TMP}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${JOB_TMP}/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/fast/${USER}/apptainer_cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
mkdir -p "${JOB_DIR}/hf_cache"
mkdir -p "${JOB_DIR}/task/.local/lib/python3.10/site-packages/pyairports"
CONTAINER_BASE_PATH="/root/.local/bin:/home/agent/.local/bin:$PATH"
CONTAINER_PATH="${CONTAINER_BASE_PATH}"
STARTING_POINT_ENV_ARGS=()
STARTING_POINT_BIND_ARGS=()
if [ "${STARTING_POINT}" = "vllm_running" ]; then
    # The baseline vllm container already ships with vllm 0.11.0, xformers, and
    # flashinfer installed at the system level — no per-job venv bootstrap needed.
    INFERENCE_BENCH_CONTAINER_NAME="${INFERENCE_BENCH_CONTAINER_NAME_VLLM_RUNNING:-inference_baselines_vllm}"
fi

# Host-side HF cache path (overlay lowerdir). Fall back to per-job cache if unset.
if [ -z "${HF_HOME:-}" ] || [ "${HF_HOME}" = "UNDEFINED" ]; then
    export HF_HOME="${JOB_DIR}/hf_cache"
fi
mkdir -p "${HF_HOME}"

# Container-side HF cache destination used by --bind "${HF_MERGED}:${HF_HOME_NEW}".
# Must never be empty, or Apptainer fails with "mount point must contain a destination".
export HF_HOME_NEW="${INFERENCE_BENCH_HF_HOME_IN_CONTAINER:-/hf_cache}"

# Use --nv only when NVIDIA devices are visible on the host.
APPTAINER_NV_FLAG=()
if ls /dev/nvidia* >/dev/null 2>&1; then
    APPTAINER_NV_FLAG+=(--nv)
else
    echo "[warn] No /dev/nvidia* devices found on host; running Apptainer without --nv."
fi

init_cuda_visibility_env() {
    # Preserve scheduler-provided GPU visibility through --cleanenv container launches.
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${_CONDOR_AssignedGPUs:-}" ]; then
        local condor_norm
        condor_norm="$(echo "${_CONDOR_AssignedGPUs}" | tr ';' ',' | tr -d '[:space:]')"
        local mapped=""
        local uuid_map=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            uuid_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
        fi
        local proc_uuid_map
        proc_uuid_map="$(
            for info in /proc/driver/nvidia/gpus/*/information; do
                [ -r "${info}" ] || continue
                local_idx="$(awk -F':' '/Device Minor/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "${info}")"
                local_uuid="$(awk -F':' '/GPU UUID/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}' "${info}")"
                if [ -n "${local_idx}" ] && [ -n "${local_uuid}" ]; then
                    echo "${local_idx},${local_uuid}"
                fi
            done
        )"
        local tok
        for tok in $(echo "${condor_norm}" | tr ',' ' '); do
            local v=""
            if [[ "${tok}" =~ ^CUDA([0-9]+)$ ]]; then
                v="${BASH_REMATCH[1]}"
            elif [[ "${tok}" =~ ^[0-9]+$ ]]; then
                v="${tok}"
            elif [[ "${tok}" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
                # Prefer full UUID matches (or prefixes) to handle HTCondor GPU tokens.
                v="$(echo "${proc_uuid_map}" | awk -F',' -v want="${tok}" '
                    {
                        idx=$1; uuid=$2;
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx);
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid);
                        low_uuid=tolower(uuid); low_want=tolower(want);
                        if (low_uuid == low_want || index(low_uuid, low_want) == 1) { print uuid; exit 0; }
                    }')"
                if [ -z "${v}" ]; then
                    v="$(echo "${uuid_map}" | awk -F',' -v want="${tok}" '
                        {
                            idx=$1; uuid=$2;
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx);
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid);
                            low_uuid=tolower(uuid); low_want=tolower(want);
                            if (low_uuid == low_want || index(low_uuid, low_want) == 1) { print uuid; exit 0; }
                        }')"
                fi
            fi
            if [ -n "${v}" ]; then
                if [ -n "${mapped}" ]; then
                    mapped="${mapped},${v}"
                else
                    mapped="${v}"
                fi
            fi
        done
        if [ -n "${mapped}" ]; then
            export CUDA_VISIBLE_DEVICES="${mapped}"
            echo "[gpu] CUDA_VISIBLE_DEVICES initialized from _CONDOR_AssignedGPUs=${_CONDOR_AssignedGPUs} -> ${CUDA_VISIBLE_DEVICES}"
        else
            echo "[gpu] _CONDOR_AssignedGPUs is set but not mappable (${_CONDOR_AssignedGPUs}); leaving CUDA_VISIBLE_DEVICES unset."
        fi
    fi
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
        local table_idx
        # Parse first GPU index from nvidia-smi table output; avoids selecting dead GPU0 blindly.
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
            export CUDA_VISIBLE_DEVICES="${table_idx}"
            echo "[gpu] CUDA_VISIBLE_DEVICES initialized from nvidia-smi table -> ${CUDA_VISIBLE_DEVICES}"
        fi
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
        export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    fi
}

refresh_cuda_env_flags() {
    APPTAINER_CUDA_ENV_FLAGS=()
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        APPTAINER_CUDA_ENV_FLAGS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
    fi
    if [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
        APPTAINER_CUDA_ENV_FLAGS+=(--env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}")
    fi
}

init_cuda_visibility_env
refresh_cuda_env_flags

cat > "${JOB_DIR}/task/.local/lib/python3.10/site-packages/pyairports/__init__.py" <<'PY'
"""Minimal stub for optional `pyairports` dependency (InferenceBench)."""
PY
cat > "${JOB_DIR}/task/.local/lib/python3.10/site-packages/pyairports/airports.py" <<'PY'
"""Minimal stub for optional `pyairports` dependency (InferenceBench)."""
AIRPORT_LIST = []
PY
cat > "${JOB_DIR}/task/.local/lib/python3.10/site-packages/sitecustomize.py" <<'PY'
"""Runtime compatibility shims for agent-time Python processes."""
try:
    import torch  # type: ignore

    # Older torch builds may miss torch.cpu.is_bf16_supported(), while
    # some generated server scripts call it directly.
    if hasattr(torch, "cpu") and not hasattr(torch.cpu, "is_bf16_supported"):
        def _ptb_cpu_is_bf16_supported() -> bool:
            try:
                return bool(getattr(torch.backends.cpu, "has_bf16", False))
            except Exception:
                return False
        torch.cpu.is_bf16_supported = _ptb_cpu_is_bf16_supported  # type: ignore[attr-defined]
except Exception:
    pass
PY

echo "Preparing job directory (starting_point=${STARTING_POINT})..."
mkdir -p "${JOB_DIR}/task"
SHARED_TASK_CONTEXT_DIR="src/eval/tasks/_shared/task_context"

case "${STARTING_POINT}" in
  default)
    cp "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task"
    for f in mission.txt benchmark.txt scenario.json; do
        if [ -f "src/eval/tasks/${EVALUATION_TASK}/${f}" ]; then
            cp "src/eval/tasks/${EVALUATION_TASK}/${f}" "${JOB_DIR}/task"
        fi
    done
    if [ -d "${SHARED_TASK_CONTEXT_DIR}" ]; then
        cp -r "${SHARED_TASK_CONTEXT_DIR}/." "${JOB_DIR}/task"
    fi
    if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
        cp -r "src/eval/tasks/${EVALUATION_TASK}/task_context/." "${JOB_DIR}/task"
    fi
    # Do NOT save start_server_original.sh here — for 'default' starting point the
    # harness-provided start_server.sh is a blank stub that the agent must fill in.
    # The final-eval fallback will fall through to start_server.sh (agent's version).
    ;;
  vllm_running)
    cp "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task"
    for f in mission.txt benchmark.txt scenario.json; do
        if [ -f "src/eval/tasks/${EVALUATION_TASK}/${f}" ]; then
            cp "src/eval/tasks/${EVALUATION_TASK}/${f}" "${JOB_DIR}/task"
        fi
    done
    if [ -d "${SHARED_TASK_CONTEXT_DIR}" ]; then
        cp -r "${SHARED_TASK_CONTEXT_DIR}/." "${JOB_DIR}/task"
    fi
    if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
        cp -r "src/eval/tasks/${EVALUATION_TASK}/task_context/." "${JOB_DIR}/task"
    fi
    if [ -f "src/starting_points/vllm_running/start_server.sh" ]; then
        cp "src/starting_points/vllm_running/start_server.sh" "${JOB_DIR}/task/start_server.sh"
        cp "src/starting_points/vllm_running/start_server.sh" "${JOB_DIR}/task/start_server_original.sh"
    else
        echo "[fatal] src/starting_points/vllm_running/start_server.sh not found"
        exit 1
    fi
    ;;
  bare)
    cp "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task"
    if [ -f "src/eval/tasks/${EVALUATION_TASK}/scenario.json" ]; then
        cp "src/eval/tasks/${EVALUATION_TASK}/scenario.json" "${JOB_DIR}/task"
    fi
    if [ -f "${SHARED_TASK_CONTEXT_DIR}/test_server.sh" ]; then
        cp "${SHARED_TASK_CONTEXT_DIR}/test_server.sh" "${JOB_DIR}/task/test_server.sh"
    fi
    export HF_HOME="${JOB_DIR}/hf_cache"
    mkdir -p "${HF_HOME}"
    ;;
  *)
    echo "[fatal] unknown starting_point: ${STARTING_POINT}"
    exit 1
    ;;
esac

# Keep evaluator internals out of the writable task workspace.
# For inference scenarios, task-side evaluate.py imports runner code from
# a read-only bind mount at /opt/inference_eval.
if [[ "${EVALUATION_TASK}" == inference_scenario_* ]]; then
    cat > "${JOB_DIR}/task/evaluate.py" <<'PY'
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, "/opt")
from inference_eval.runner import build_parser, run_evaluation  # noqa: E402


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(Path(__file__).resolve().parent, args)


if __name__ == "__main__":
    main()
PY
fi
find "${JOB_DIR}/task" -maxdepth 1 -type f -name "*.sh" -exec chmod +x {} + 2>/dev/null || true
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

BENCHMARK="$(cat "src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")"
METRICS_PATH="${INFERENCE_BENCH_METRICS_PATH:-/home/agent/task/metrics_preview.json}"
export INFERENCE_BENCH_METRICS_PATH="${METRICS_PATH}"
PROMPT="$(python3 src/eval/general/get_prompt.py --base-model "$BASE_MODEL" --scenario-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --agent "${AGENT}" --starting-point "${STARTING_POINT}")"
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"
echo "$PROMPT" > "${JOB_DIR}/prompt.txt"
export PROMPT_FILE="/home/agent/prompt.txt"

bash src/utils/create_timer.sh "${NUM_HOURS}" "${JOB_DIR}/task/timer.sh"

# Copy scripts needed inside the container
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

if [ -f "agents/${AGENT}/auth.json" ]; then
    cp "agents/${AGENT}/auth.json" "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${JOB_DIR}/oauth_token"
fi
if [ -f "agents/${AGENT}/api_key" ]; then
    cp "agents/${AGENT}/api_key" "${JOB_DIR}/api_key"
fi

INFERENCE_EVAL_BUNDLE="${JOB_DIR}/inference_eval"
mkdir -p "${INFERENCE_EVAL_BUNDLE}"
cp src/eval/inference/__init__.py "${INFERENCE_EVAL_BUNDLE}/"
cp src/eval/inference/runner.py "${INFERENCE_EVAL_BUNDLE}/"
cp src/eval/inference/quality_gate.py "${INFERENCE_EVAL_BUNDLE}/"
cp src/eval/inference/cache_samples.py "${INFERENCE_EVAL_BUNDLE}/"
mkdir -p "${INFERENCE_EVAL_BUNDLE}/bin"
cp src/eval/inference/bin/launch_supervised_server.sh "${INFERENCE_EVAL_BUNDLE}/bin/"
chmod +x "${INFERENCE_EVAL_BUNDLE}/bin/launch_supervised_server.sh"
mkdir -p "${INFERENCE_EVAL_BUNDLE}/baselines"
cp -r src/eval/inference/baselines/quality "${INFERENCE_EVAL_BUNDLE}/baselines/"
cp -r src/eval/inference/baselines/samples "${INFERENCE_EVAL_BUNDLE}/baselines/"
cat > "${JOB_DIR}/cuda_check_once.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

timeout_s="${1:-120}"
strict="${INFERENCE_BENCH_CUDA_CHECK_STRICT:-0}"

echo "[runner] running cuda check (timeout=${timeout_s}s, strict=${strict})"

if timeout --signal=TERM --kill-after=10s "${timeout_s}s" python3 -u /home/agent/check_cuda_writing.py; then
    echo "[runner] cuda check passed"
    exit 0
fi
last_rc=$?
echo "[runner] cuda check failed (rc=${last_rc})"

# If CUDA_VISIBLE_DEVICES is in UUID format (GPU-<uuid>), try falling back to
# integer index 0. Old vLLM builds (e.g. 0.6.6) cannot parse UUID format.
current_cvd="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${current_cvd}" == GPU-* ]]; then
    echo "[runner] CUDA_VISIBLE_DEVICES is UUID format (${current_cvd}); retrying with integer index 0"
    export CUDA_VISIBLE_DEVICES=0
    if timeout --signal=TERM --kill-after=10s "${timeout_s}s" python3 -u /home/agent/check_cuda_writing.py; then
        echo "[runner] cuda check passed with CUDA_VISIBLE_DEVICES=0 (integer fallback)"
        # Write override so parent shell can source it and propagate to child processes
        echo "export CUDA_VISIBLE_DEVICES=0" > /tmp/cuda_env_override.sh
        exit 0
    fi
    last_rc=$?
    echo "[runner] cuda check also failed with integer fallback (rc=${last_rc}); restoring UUID"
    export CUDA_VISIBLE_DEVICES="${current_cvd}"
fi

if [ "${strict}" = "1" ]; then
    echo "[runner] ERROR: cuda check failed (strict mode)."
    exit "${last_rc}"
fi

echo "[runner] WARNING: cuda check failed; continuing run (non-strict mode)."
exit 0
SH
chmod +x "${JOB_DIR}/cuda_check_once.sh"

# Utils
with_huggingface_overlay() {
    local start_ts
    start_ts="$(date --iso-8601=seconds)"
    local original_hf_merged="${HF_MERGED}"
    local mounted_overlay=0
    local overlay_timeout_s="${INFERENCE_BENCH_OVERLAY_MOUNT_TIMEOUT_S:-30}"

    if [ "${INFERENCE_BENCH_DISABLE_HF_OVERLAY:-1}" = "1" ]; then
        echo "[overlay] ${start_ts} overlay disabled (INFERENCE_BENCH_DISABLE_HF_OVERLAY=1); using direct HF_HOME bind."
        export HF_MERGED="${HF_HOME}"
        "$@"
        local direct_exit_code=$?
        export HF_MERGED="${original_hf_merged}"
        return $direct_exit_code
    fi

    mkdir -p "$TMP_SUBDIR/merged_huggingface"
    mkdir -p "$TMP_SUBDIR/upper_huggingface"
    mkdir -p "$TMP_SUBDIR/fuse_workdir"

    if ! command -v fuse-overlayfs >/dev/null 2>&1; then
        echo "[overlay] ${start_ts} fuse-overlayfs not found; falling back to direct HF_HOME bind."
        export HF_MERGED="${HF_HOME}"
    else
        echo "[overlay] ${start_ts} mounting fuse overlay lowerdir=${HF_HOME} merged=${TMP_SUBDIR}/merged_huggingface timeout=${overlay_timeout_s}s"
        if timeout --signal=TERM --kill-after=5s "${overlay_timeout_s}s" \
            fuse-overlayfs -o "lowerdir=$HF_HOME,upperdir=$TMP_SUBDIR/upper_huggingface,workdir=$TMP_SUBDIR/fuse_workdir" "$TMP_SUBDIR/merged_huggingface"; then
            mounted_overlay=1
            echo "[overlay] $(date --iso-8601=seconds) mounted successfully"
            export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
        else
            local mount_rc=$?
            echo "[overlay] $(date --iso-8601=seconds) mount failed rc=${mount_rc}; falling back to direct HF_HOME bind."
            export HF_MERGED="${HF_HOME}"
        fi
    fi

    "$@"
    local exit_code=$?

    if [ "${mounted_overlay}" = "1" ]; then
        fusermount -u "$TMP_SUBDIR/merged_huggingface" 2>/dev/null || true
    fi
    rm -rf "$TMP_SUBDIR/merged_huggingface" "$TMP_SUBDIR/upper_huggingface" "$TMP_SUBDIR/fuse_workdir" 2>/dev/null || true
    export HF_MERGED="${original_hf_merged}"

    return $exit_code
}

with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)

    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"

    return $exit_code
}

JOB_RECORDED_PGIDS=()

remember_process_group() {
    local pid="${1:-}"
    local label="${2:-process}"
    if ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    local pgid
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if ! [[ "${pgid}" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    local existing
    for existing in "${JOB_RECORDED_PGIDS[@]}"; do
        if [ "${existing}" = "${pgid}" ]; then
            return 0
        fi
    done

    JOB_RECORDED_PGIDS+=("${pgid}")
    echo "[proc] $(date --iso-8601=seconds) tracking ${label} process group pgid=${pgid} pid=${pid}"
}

_list_process_group_members() {
    local pgid="${1:-}"
    if ! [[ "${pgid}" =~ ^[0-9]+$ ]]; then
        return 0
    fi
    ps -eo pid=,pgid=,ppid=,args= 2>/dev/null | awk -v pgid="${pgid}" -v self="$$" '$2 == pgid && $1 != self {print $0}'
}

_stop_process_group() {
    local pgid="${1:-}"
    local label="${2:-process-group}"
    if ! [[ "${pgid}" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    local current_pgid
    current_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -n "${current_pgid}" ] && [ "${pgid}" = "${current_pgid}" ]; then
        echo "[proc] $(date --iso-8601=seconds) refusing to kill current shell process group pgid=${pgid} (${label})" | tee -a "${EVAL_LOG}"
        return 1
    fi

    local members
    members="$(_list_process_group_members "${pgid}")"
    if [ -z "${members}" ]; then
        return 0
    fi

    echo "[proc] $(date --iso-8601=seconds) stopping ${label} pgid=${pgid}" | tee -a "${EVAL_LOG}"
    echo "${members}" | sed 's/^/[proc]   /' | tee -a "${EVAL_LOG}"

    kill -- "-${pgid}" 2>/dev/null || true
    sleep 2

    members="$(_list_process_group_members "${pgid}")"
    if [ -n "${members}" ]; then
        echo "[proc] $(date --iso-8601=seconds) escalating to SIGKILL for ${label} pgid=${pgid}" | tee -a "${EVAL_LOG}"
        kill -9 -- "-${pgid}" 2>/dev/null || true
        sleep 2
    fi
}

_stop_recorded_process_groups() {
    local pgid
    for pgid in "${JOB_RECORDED_PGIDS[@]}"; do
        _stop_process_group "${pgid}" "tracked-job-group" || true
    done
}

_recorded_process_groups_clear() {
    local pgid
    for pgid in "${JOB_RECORDED_PGIDS[@]}"; do
        if [ -n "$(_list_process_group_members "${pgid}")" ]; then
            return 1
        fi
    done
    return 0
}

_local_server_listener_details() {
    local port="${1:-}"
    if ! [[ "${port}" =~ ^[0-9]+$ ]]; then
        return 0
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -lptn "sport = :${port}" 2>/dev/null | sed '/^State/d' || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    fi
}

_job_memory_usage_file() {
    local memory_path
    memory_path="$(awk -F: '$2 == "memory" {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    if [ -n "${memory_path}" ] && [ -f "/sys/fs/cgroup/memory${memory_path}/memory.usage_in_bytes" ]; then
        echo "/sys/fs/cgroup/memory${memory_path}/memory.usage_in_bytes"
        return 0
    fi

    local unified_path
    unified_path="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    if [ -n "${unified_path}" ] && [ -f "/sys/fs/cgroup/unified${unified_path}/memory.current" ]; then
        echo "/sys/fs/cgroup/unified${unified_path}/memory.current"
    fi
}

_log_job_memory_usage() {
    local label="${1:-memory}"
    local usage_file
    usage_file="$(_job_memory_usage_file)"
    if [ -z "${usage_file}" ] || [ ! -f "${usage_file}" ]; then
        return 0
    fi

    local bytes
    bytes="$(cat "${usage_file}" 2>/dev/null || true)"
    if [[ "${bytes}" =~ ^[0-9]+$ ]]; then
        local mib
        mib="$(( bytes / 1024 / 1024 ))"
        echo "[mem] $(date --iso-8601=seconds) ${label}: ${bytes} bytes (${mib} MiB) from ${usage_file}" | tee -a "${EVAL_LOG}"
    fi
}

_reset_job_scratch_for_eval() {
    local job_dir_name
    job_dir_name="$(basename "${JOB_DIR}")"

    echo "[scratch] $(date --iso-8601=seconds) resetting job-local scratch under ${TMP_SUBDIR}" | tee -a "${EVAL_LOG}"
    if [ -d "${TMP_SUBDIR}" ]; then
        find "${TMP_SUBDIR}" -mindepth 1 -maxdepth 1 ! -name "${job_dir_name}" -exec rm -rf {} + 2>/dev/null || true
    fi

    mkdir -p "${JOB_TMP}" "${APPTAINER_TMPDIR}"
    du -sh "${TMP_SUBDIR}" "${JOB_TMP}" 2>/dev/null | sed 's/^/[scratch] /' | tee -a "${EVAL_LOG}" || true
}

pre_eval_cleanup_barrier() {
    local server_url="${INFERENCE_BENCH_SERVER_URL:-http://127.0.0.1:8000}"
    local hostport="${server_url#http://}"
    hostport="${hostport#https://}"
    hostport="${hostport%%/*}"
    local host="${hostport%%:*}"
    local port="${hostport##*:}"
    if [ -z "${port}" ] || [ "${port}" = "${hostport}" ]; then
        port="8000"
    fi

    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) entering pre-eval cleanup barrier" | tee -a "${EVAL_LOG}"
    _log_job_memory_usage "before cleanup barrier"

    local attempt
    for attempt in 1 2 3; do
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) cleanup pass ${attempt}/3" | tee -a "${EVAL_LOG}"

        stop_inference_server || true
        _stop_recorded_process_groups || true

        local gpu_pids
        gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
        if [ -n "${gpu_pids// /}" ]; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) killing leftover GPU compute processes during cleanup barrier: ${gpu_pids}" | tee -a "${EVAL_LOG}"
            echo "${gpu_pids}" | xargs -r kill 2>/dev/null || true
            sleep 2
            gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
            if [ -n "${gpu_pids// /}" ]; then
                echo "${gpu_pids}" | xargs -r kill -9 2>/dev/null || true
            fi
        fi

        _reset_job_scratch_for_eval
        sleep 2

        local clean=1
        if ! _recorded_process_groups_clear; then
            clean=0
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) tracked process groups still have live members after cleanup pass ${attempt}" | tee -a "${EVAL_LOG}"
            local pgid
            for pgid in "${JOB_RECORDED_PGIDS[@]}"; do
                local members
                members="$(_list_process_group_members "${pgid}")"
                if [ -n "${members}" ]; then
                    echo "${members}" | sed 's/^/[proc] survivor /' | tee -a "${EVAL_LOG}"
                fi
            done
        fi

        local listeners
        listeners="$(_local_server_listener_details "${port}")"
        if [ -n "${listeners}" ] && { [ "${host}" = "127.0.0.1" ] || [ "${host}" = "localhost" ] || [ -z "${host}" ]; }; then
            clean=0
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) server port ${port} still has listeners after cleanup pass ${attempt}" | tee -a "${EVAL_LOG}"
            echo "${listeners}" | sed 's/^/[port] /' | tee -a "${EVAL_LOG}"
        fi

        gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
        if [ -n "${gpu_pids// /}" ]; then
            clean=0
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) GPU compute processes still present after cleanup pass ${attempt}: ${gpu_pids}" | tee -a "${EVAL_LOG}"
        fi

        _log_job_memory_usage "after cleanup pass ${attempt}"
        if [ "${clean}" = "1" ]; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) pre-eval cleanup barrier reached a clean state" | tee -a "${EVAL_LOG}"
            return 0
        fi
    done

    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) warning: proceeding to final eval after incomplete cleanup barrier" | tee -a "${EVAL_LOG}"
    return 0
}

solve_task() {
    local timeout_seconds="${1:-}"
    local run_cuda_check="${2:-1}"
    local run_index="${3:-1}"
    local timeout_arg=""
    if [ -n "${timeout_seconds}" ]; then
        timeout_arg="${timeout_seconds}s"
    else
        timeout_arg="$((NUM_SECONDS + 300))s"
    fi
    SOLVE_OUT="${EVAL_DIR}/solve_out_run${run_index}.txt"
    local data_bind_args=()
    if [ -d "/data" ]; then
        data_bind_args+=(--bind "/data:/data:ro")
    fi
    local cuda_check_cmd=""
    refresh_cuda_env_flags
    if [ "${run_cuda_check}" = "1" ]; then
        if [ "${INFERENCE_BENCH_SKIP_CUDA_CHECK:-0}" = "1" ]; then
            cuda_check_cmd="echo \"[runner] skipping cuda check (INFERENCE_BENCH_SKIP_CUDA_CHECK=1)\"; "
        else
            local cuda_check_timeout_s="${INFERENCE_BENCH_CUDA_CHECK_TIMEOUT_S:-120}"
            cuda_check_cmd="bash /home/agent/cuda_check_once.sh ${cuda_check_timeout_s}; "
        fi
    fi
    # Re-copy agent_solve.sh before each run in case the agent deleted it during a prior run.
    cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"
    # Some agents accidentally print binary data (e.g. by cat'ing .pyc files),
    # which makes solve_out_run*.txt hard to parse. Strip NUL bytes at capture time.
    echo "[runner] $(date --iso-8601=seconds) launching agent container run=${run_index} timeout=${timeout_arg} output=${SOLVE_OUT}"
    echo "[runner] gpu_env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-} apptainer_nv_flag=${APPTAINER_NV_FLAG[*]:-none}"
    setsid timeout --signal=TERM --kill-after=30s "${timeout_arg}" \
      apptainer exec \
        "${APPTAINER_NV_FLAG[@]}" \
        --pid \
        -c \
        --cleanenv \
        --env PATH="${CONTAINER_PATH}" \
        --env TMPDIR="/tmp" \
        --env TMP="/tmp" \
        --env TEMP="/tmp" \
        --env HF_HOME="${HF_HOME_NEW}" \
        --env HF_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env TRANSFORMERS_CACHE="${HF_HOME_NEW}/hub" \
        --env XDG_CACHE_HOME="/tmp/xdg_cache" \
        --env VLLM_TORCH_COMPILE_CACHE_DIR="/tmp/vllm_torch_compile_cache" \
        --env TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache" \
        --env TORCH_EXTENSIONS_DIR="/tmp/torch_extensions" \
        --env TRITON_CACHE_DIR="/tmp/triton_cache" \
        --env CUDA_CACHE_PATH="/tmp/cuda_cache" \
        --env RAY_DISABLE_DASHBOARD="1" \
        --env RAY_USAGE_STATS_ENABLED="0" \
        --env RAY_DISABLE_METRICS_EXPORT="1" \
        --env INFERENCE_BENCH_BASE_MODEL="${BASE_MODEL}" \
        --env INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN}" \
        --env INFERENCE_BENCH_INPUT_TOKEN_MARGIN="${INFERENCE_BENCH_INPUT_TOKEN_MARGIN:-16}" \
        --env INFERENCE_BENCH_SCENARIO="${EVALUATION_TASK}" \
        "${CALIBRATION_ENV[@]}" \
        --env INFERENCE_BENCH_DATASET_SEED="${AGENT_SEED}" \
        --env INFERENCE_BENCH_EVAL_SEED="${EVAL_SEED}" \
        --env INFERENCE_BENCH_QUALITY_TAU="${INFERENCE_BENCH_QUALITY_TAU}" \
        --env INFERENCE_BENCH_QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED}" \
        --env INFERENCE_BENCH_QUALITY_MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}" \
        --env INFERENCE_BENCH_QUALITY_CONCURRENCY="${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
        --env INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY="${INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY:-}" \
        --env INFERENCE_BENCH_QUALITY_MMLU_SAMPLES_FILE="${INFERENCE_BENCH_QUALITY_MMLU_SAMPLES_FILE:-}" \
        --env INFERENCE_BENCH_QUALITY_GSM8K_SAMPLES_FILE="${INFERENCE_BENCH_QUALITY_GSM8K_SAMPLES_FILE:-}" \
        --env INFERENCE_BENCH_SERVER_URL="${INFERENCE_BENCH_SERVER_URL}" \
        --env INFERENCE_BENCH_SERVER_HOST="${INFERENCE_BENCH_SERVER_HOST}" \
        --env INFERENCE_BENCH_SERVER_PORT="${INFERENCE_BENCH_SERVER_PORT}" \
        --env HOST="${INFERENCE_BENCH_SERVER_HOST}" \
        --env PORT="${INFERENCE_BENCH_SERVER_PORT}" \
        --env INFERENCE_BENCH_METRICS_PATH="${METRICS_PATH}" \
        --env INFERENCE_BENCH_SERVER_WAIT_S="${INFERENCE_BENCH_SERVER_WAIT_S:-900}" \
        --env INFERENCE_BENCH_RUN_INDEX="${run_index}" \
        --env INFERENCE_BENCH_STARTING_POINT="${STARTING_POINT}" \
        --env NUM_HOURS="${NUM_HOURS}" \
        --env INFERENCE_BENCH_REMAINING_SECONDS="${timeout_seconds}" \
        --env ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
        --env CODEX_API_KEY="${OPENAI_API_KEY}" \
        --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
        --env KIMI_API_KEY="${KIMI_API_KEY}" \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env PIP_USER=1 \
        --env PYTHONUSERBASE="/home/agent/.local" \
        --env PIP_CACHE_DIR="/tmp/pip_cache" \
        --env PROMPT_FILE="${PROMPT_FILE}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        "${STARTING_POINT_ENV_ARGS[@]}" \
        "${APPTAINER_CUDA_ENV_FLAGS[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_MERGED}:${HF_HOME_NEW}" \
        --bind "${INFERENCE_EVAL_BUNDLE}:/opt/inference_eval:ro" \
        "${STARTING_POINT_BIND_ARGS[@]}" \
        "${data_bind_args[@]}" \
        --home "${JOB_DIR}:/home/agent" \
        --pwd "/home/agent/task" \
        --writable-tmpfs \
        "${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif" \
        bash -c "set -euo pipefail; echo \"[runner] agent container bash PID=\$\$\"; echo \"[runner] entered container for run ${run_index}\"; ${cuda_check_cmd}[ -f /tmp/cuda_env_override.sh ] && . /tmp/cuda_env_override.sh && echo \"[runner] applied CUDA integer fallback from /tmp/cuda_env_override.sh\" || true; echo \"[runner] starting agent run ${run_index}\"; bash /home/agent/agent_solve.sh" \
        > >(stdbuf -oL tr -d '\000' | tee "${SOLVE_OUT}") 2>&1 &
    local session_pid=$!
    CURRENT_AGENT_PID=${session_pid}
    remember_process_group "${session_pid}" "agent-run-${run_index}"
    local rc=0
    wait "${session_pid}" || rc=$?
    return "${rc}"
}

run_agent_until_deadline() {
    local run_index=1
    if [ "${CHECKPOINT_CLAUDE_RESTORED:-0}" = "1" ]; then
        run_index=2  # Signal solve.sh to use --continue on the restored session
    fi
    local rc=0
    local remaining=0
    local max_runs="${INFERENCE_BENCH_MAX_AGENT_RUNS:-3}"
    local rerun_backoff_s="${INFERENCE_BENCH_AGENT_RERUN_BACKOFF_S:-5}"

    [[ "${max_runs}" =~ ^[0-9]+$ ]] || max_runs=3
    [[ "${rerun_backoff_s}" =~ ^[0-9]+$ ]] || rerun_backoff_s=5

    while :; do
        remaining=$((AGENT_DEADLINE_TS - $(date +%s)))
        if [ "${remaining}" -le 0 ]; then
            echo "Time budget exhausted; no further agent run executed."
            return "${rc}"
        fi
        if [ "${remaining}" -lt 60 ]; then
            echo "Less than 60 seconds remaining; stopping agent retries."
            return "${rc}"
        fi
        if [ "${run_index}" -gt "${max_runs}" ]; then
            echo "Reached max agent runs (${max_runs}) with ${remaining}s still remaining; stopping retries."
            return "${rc}"
        fi

        echo "================================"
        echo "========= AGENT RUN ${run_index} ========="
        echo "================================"
        echo "Remaining time (seconds): ${remaining}"
        solve_task "${remaining}" 1 "${run_index}"
        rc=$?
        echo "Agent run ${run_index} exit code: ${rc}"
        if [ "${rc}" -eq 124 ]; then
            echo "Agent run ${run_index} consumed the remaining solve budget."
            return "${rc}"
        fi
        if [ "${rc}" -eq 0 ]; then
            echo "Agent run ${run_index} completed cleanly; not retrying (resume was handled inside the container)."
            return "${rc}"
        fi

        remaining=$((AGENT_DEADLINE_TS - $(date +%s)))
        if [ "${remaining}" -lt 60 ]; then
            echo "Less than 60 seconds remaining after run ${run_index}; stopping retries."
            return "${rc}"
        fi

        echo "Agent run ${run_index} crashed (rc=${rc}) with ${remaining}s remaining."
        echo "Retrying in a fresh agent container after ${rerun_backoff_s}s backoff."
        if [ "${rerun_backoff_s}" -gt 0 ]; then
            sleep "${rerun_backoff_s}"
        fi
        run_index=$((run_index + 1))
    done
}


echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

echo "GPU status (before solve):"
nvidia-smi || true

# For vllm_running, the agent runs in inference_baselines_vllm.sif which already
# has vllm preinstalled — no per-job venv bootstrap is required.

AGENT_START_TS=$(date +%s)
AGENT_GRACE_SEC=300
if [ "${CHECKPOINT_REMAINING_SECONDS:-0}" -gt 60 ]; then
    AGENT_DEADLINE_TS=$((AGENT_START_TS + CHECKPOINT_REMAINING_SECONDS))
    echo "[harness] resuming from checkpoint: ${CHECKPOINT_REMAINING_SECONDS}s budget"
else
    AGENT_DEADLINE_TS=$((AGENT_START_TS + NUM_SECONDS + AGENT_GRACE_SEC))
fi

set +e
with_huggingface_overlay with_record_the_time run_agent_until_deadline
AGENT_RC=$?
set -e
echo "Agent phase exit code: ${AGENT_RC}"

echo "GPU status (after solve):"
nvidia-smi || true

echo "=================================================="
echo "=== TASK COMPLETE, RUNNING DISALLOWED JUDGE ======"
echo "=================================================="

JUDGE_TASK=$(python3 src/disallowed_usage_judge/get_judge_prompt.py --benchmark "${BENCHMARK}" --model "${BASE_MODEL}")

run_disallowed_judge_in_container() {
    local oauth_token=""
    if [ -f "agents/claude_non_api/oauth_token" ]; then
        oauth_token="$(cat agents/claude_non_api/oauth_token)"
    fi
    apptainer exec \
        "${APPTAINER_NV_FLAG[@]}" \
        --pid \
        -c \
        --cleanenv \
        --env PATH="/root/.local/bin:/home/agent/.local/bin:$PATH" \
        --env HF_HOME="${HF_HOME_NEW}" \
        --env HF_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env TRANSFORMERS_CACHE="${HF_HOME_NEW}/hub" \
        --env ANTHROPIC_API_KEY="" \
        --env CLAUDE_CODE_OAUTH_TOKEN="${oauth_token}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_MERGED}:${HF_HOME_NEW}" \
        --home "${JOB_DIR}:/home/agent" \
        --pwd "/home/agent/task" \
        --writable-tmpfs \
        "${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif" \
        claude --print --dangerously-skip-permissions --model "claude-sonnet-4-6" "${JUDGE_TASK}"
}

set +e
with_huggingface_overlay run_disallowed_judge_in_container
JUDGE_RC=$?
set -e
echo "Disallowed-usage judge exit code: ${JUDGE_RC}"

copy_if_exists() {
    local src="${1:-}"
    local dst="${2:-}"
    local missing_warn="${3:-}"
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}"
    elif [ -n "${missing_warn}" ]; then
        echo "[warn] ${missing_warn}"
    fi
}

copy_if_exists "${JOB_DIR}/task/contamination_judgement.txt" "${EVAL_DIR}/contamination_judgement.txt" "contamination_judgement.txt was not produced."
copy_if_exists "${JOB_DIR}/task/disallowed_model_judgement.txt" "${EVAL_DIR}/disallowed_model_judgement.txt" "disallowed_model_judgement.txt was not produced."
copy_if_exists "${JOB_DIR}/task/metrics_preview.json" "${EVAL_DIR}/metrics_preview.json"
copy_if_exists "${JOB_DIR}/task/server.log" "${EVAL_DIR}/server.log"

_stop_pids() {
    local pids="${1:-}"
    if [ -z "${pids}" ]; then
        return 0
    fi
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 2
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
}

stop_inference_server() {
    local server_url="${INFERENCE_BENCH_SERVER_URL:-http://127.0.0.1:8000}"
    local hostport="${server_url#http://}"
    hostport="${hostport#https://}"
    hostport="${hostport%%/*}"
    local host="${hostport%%:*}"
    local port="${hostport##*:}"
    if [ -z "${port}" ] || [ "${port}" = "${hostport}" ]; then
        port="8000"
    fi

    local pid_files=(
        "${JOB_DIR}/task/server.pid"
        "${JOB_DIR}/task/server_${port}.pid"
        "${EVAL_DIR}/task/server.pid"
        "${EVAL_DIR}/task/server_${port}.pid"
    )
    local pid_file=""
    for candidate in "${pid_files[@]}"; do
        if [ -f "${candidate}" ]; then
            pid_file="${candidate}"
            break
        fi
    done

    if [ -n "${pid_file}" ]; then
        local pid
        pid="$(head -n 1 "${pid_file}" 2>/dev/null || true)"
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) stopping inference server (pid=${pid}, pid_file=${pid_file})" | tee -a "${EVAL_LOG}"
            kill "${pid}" 2>/dev/null || true
            for _ in {1..60}; do
                if kill -0 "${pid}" 2>/dev/null; then
                    sleep 1
                else
                    break
                fi
            done
            kill -9 "${pid}" 2>/dev/null || true
            rm -f "${pid_file}" 2>/dev/null || true
        fi
    fi

    # Only use port-based killing when the server is local.
    if [ "${host}" = "127.0.0.1" ] || [ "${host}" = "localhost" ] || [ -z "${host}" ]; then
        if command -v lsof >/dev/null 2>&1; then
            local listen_pids
            listen_pids="$(lsof -ti "TCP:${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | xargs echo 2>/dev/null || true)"
            _stop_pids "${listen_pids}"
        elif command -v fuser >/dev/null 2>&1; then
            local fuser_out
            fuser_out="$(fuser -n tcp "${port}" 2>/dev/null || true)"
            local fuser_pids
            fuser_pids="$(echo "${fuser_out}" | tr ' ' '\n' | grep -E '^[0-9]+$' || true)"
            _stop_pids "${fuser_pids}"
        elif command -v ss >/dev/null 2>&1; then
            local ss_pids
            ss_pids="$(
                ss -lptn "sport = :${port}" 2>/dev/null \
                    | awk -F'pid=' 'NR>1 && NF>1 {split($2,a,\",|\"); print a[1]}' \
                    | tr '\n' ' ' | xargs echo 2>/dev/null || true
            )"
            _stop_pids "${ss_pids}"
        fi
    fi

    # Kill any remaining GPU compute processes (e.g. vLLM worker subprocesses that
    # reparented away from the server PID and were missed by the kill above).
    local gpu_pids
    gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "${gpu_pids// /}" ]; then
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) killing leftover GPU compute processes: ${gpu_pids}" | tee -a "${EVAL_LOG}"
        echo "${gpu_pids}" | xargs -r kill 2>/dev/null || true
        sleep 2
        gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
        if [ -n "${gpu_pids// /}" ]; then
            echo "${gpu_pids}" | xargs -r kill -9 2>/dev/null || true
        fi
    fi

    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) GPU status after stopping inference server:" | tee -a "${EVAL_LOG}"
    nvidia-smi || true
}

# Backwards-compat alias (baseline eval still calls this name).
stop_vllm_server() { stop_inference_server; }

_server_models_ok_host() {
    local server_url="${1:-}"
    python3 - "${server_url}" <<'PY'
import sys
import urllib.request
import urllib.error

if len(sys.argv) < 2 or not sys.argv[1]:
    raise SystemExit(1)
url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    # Explicitly bypass proxy env vars for localhost checks.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        status = int(getattr(response, "status", 0))
        # 2xx means healthy; 401/403 still means server is alive (auth-gated).
        raise SystemExit(0 if (200 <= status < 300 or status in (401, 403)) else 1)
except urllib.error.HTTPError as exc:
    raise SystemExit(0 if int(getattr(exc, "code", 0)) in (401, 403) else 1)
except Exception:
    raise SystemExit(1)
PY
}

_server_models_ok_container() {
    local server_url="${1:-}"
    if [ -z "${server_url}" ]; then
        return 1
    fi

    refresh_cuda_env_flags
    apptainer exec \
        "${APPTAINER_NV_FLAG[@]}" \
        --pid \
        -c \
        --cleanenv \
        --env PATH="/root/.local/bin:/home/agent/.local/bin:$PATH" \
        --env TMPDIR="/tmp" \
        --env TMP="/tmp" \
        --env TEMP="/tmp" \
        --env HF_HOME="${HF_HOME_NEW}" \
        --env HF_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env TRANSFORMERS_CACHE="${HF_HOME_NEW}/hub" \
        "${APPTAINER_CUDA_ENV_FLAGS[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_HOME}:${HF_HOME_NEW}" \
        --home "${JOB_DIR}:/home/agent" \
        --pwd "/home/agent/task" \
        --writable-tmpfs \
        "${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif" \
        python3 - "${server_url}" <<'PY'
import sys
import urllib.request
import urllib.error

if len(sys.argv) < 2 or not sys.argv[1]:
    raise SystemExit(1)
url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        status = int(getattr(response, "status", 0))
        raise SystemExit(0 if (200 <= status < 300 or status in (401, 403)) else 1)
except urllib.error.HTTPError as exc:
    raise SystemExit(0 if int(getattr(exc, "code", 0)) in (401, 403) else 1)
except Exception:
    raise SystemExit(1)
PY
}

_server_models_ok() {
    local server_url="${1:-}"
    if _server_models_ok_host "${server_url}"; then
        return 0
    fi
    _server_models_ok_container "${server_url}"
}

_discover_openai_server() {
    # If the agent hardcoded a different port, try to find the server on common ports.
    local primary_url="${1:-}"
    local hostport="${primary_url#http://}"
    hostport="${hostport#https://}"
    hostport="${hostport%%/*}"
    local host="${hostport%%:*}"
    local primary_port="${hostport##*:}"

    # Check primary URL first.
    if _server_models_ok "${primary_url}"; then
        echo "${primary_url}"
        return 0
    fi

    # Scan common ports the agent might have used.
    local candidate_ports=(8000 8080 8888 5000 3000 9000 80)
    for cp in "${candidate_ports[@]}"; do
        [ "${cp}" = "${primary_port}" ] && continue
        local candidate_url="http://${host}:${cp}"
        if _server_models_ok "${candidate_url}"; then
            echo "${candidate_url}"
            return 0
        fi
    done

    # Try to find any listening port from server.pid.
    if [ -f "${JOB_DIR}/task/server.pid" ]; then
        local srv_pid
        srv_pid="$(head -n 1 "${JOB_DIR}/task/server.pid" 2>/dev/null || true)"
        if [ -n "${srv_pid}" ] && kill -0 "${srv_pid}" 2>/dev/null; then
            local srv_port
            srv_port="$(ss -lptn "sport != :22" 2>/dev/null \
                | awk -v pid="${srv_pid}" '$0 ~ "pid="pid"[^0-9]" {match($4, /:([0-9]+)$/, a); if (a[1]) print a[1]}' \
                | head -1 || true)"
            if [ -n "${srv_port}" ] && [ "${srv_port}" != "${primary_port}" ]; then
                local candidate_url="http://${host}:${srv_port}"
                if _server_models_ok "${candidate_url}"; then
                    echo "${candidate_url}"
                    return 0
                fi
            fi
        fi
    fi

    return 1
}

_server_pid_for_url() {
    local server_url="${1:-}"
    local hostport="${server_url#http://}"
    hostport="${hostport#https://}"
    hostport="${hostport%%/*}"
    local port="${hostport##*:}"
    if [ -z "${port}" ] || [ "${port}" = "${hostport}" ] || ! [[ "${port}" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    ss -lptn "sport = :${port}" 2>/dev/null \
        | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
        | head -1
}

_snapshot_runtime_tmp_symlinks() {
    local snapshot_dir="${JOB_DIR}/runtime_snapshot/tmp"
    local manifest="${JOB_DIR}/runtime_snapshot/tmp_symlinks.tsv"
    rm -rf "${JOB_DIR}/runtime_snapshot"
    mkdir -p "${snapshot_dir}"
    : > "${manifest}"

    local link target rel host_target name
    while IFS= read -r link; do
        target="$(readlink "${link}" 2>/dev/null || true)"
        case "${target}" in
            /tmp/*)
                rel="${target#/tmp/}"
                case "${rel}" in
                    ""|*..*) continue ;;
                esac
                host_target="${JOB_TMP}/${rel}"
                if [ -e "${host_target}" ]; then
                    name="$(basename "${rel}")"
                    echo "$(basename "${link}")	${rel}" >> "${manifest}"
                    cp -a "${host_target}" "${snapshot_dir}/${name}" 2>/dev/null || true
                fi
                ;;
        esac
    done < <(find "${JOB_DIR}/task" -maxdepth 1 -type l 2>/dev/null || true)

    if [ -s "${manifest}" ]; then
        echo "[runtime] $(date --iso-8601=seconds) snapshotted task /tmp symlink targets:" | tee -a "${EVAL_LOG:-/dev/null}"
        sed 's/^/[runtime]   /' "${manifest}" | tee -a "${EVAL_LOG:-/dev/null}" || true
    fi
}

_restore_runtime_tmp_snapshot() {
    local snapshot_dir="${JOB_DIR}/runtime_snapshot/tmp"
    local manifest="${JOB_DIR}/runtime_snapshot/tmp_symlinks.tsv"
    if [ ! -s "${manifest}" ] || [ ! -d "${snapshot_dir}" ]; then
        return 0
    fi

    mkdir -p "${JOB_TMP}"
    local link_name rel name
    while IFS=$'\t' read -r link_name rel; do
        [ -n "${rel}" ] || continue
        case "${rel}" in
            *..*) continue ;;
        esac
        name="$(basename "${rel}")"
        if [ -e "${snapshot_dir}/${name}" ]; then
            mkdir -p "$(dirname "${JOB_TMP}/${rel}")"
            rm -rf "${JOB_TMP:?}/${rel}" 2>/dev/null || true
            cp -a "${snapshot_dir}/${name}" "${JOB_TMP}/${rel}" 2>/dev/null || true
        fi
    done < "${manifest}"
    echo "[runtime] $(date --iso-8601=seconds) restored snapshotted /tmp runtime paths for final eval" | tee -a "${EVAL_LOG:-/dev/null}"
}

capture_agent_runtime_for_final_eval() {
    local server_url="${INFERENCE_BENCH_SERVER_URL:-http://127.0.0.1:8000}"
    local discovered_url pid env_file meta_file

    if ! discovered_url="$(_discover_openai_server "${server_url}")"; then
        echo "[runtime] $(date --iso-8601=seconds) no live agent server found for runtime capture" | tee -a "${EVAL_LOG:-/dev/null}"
        return 0
    fi
    if [ "${discovered_url}" != "${server_url}" ]; then
        echo "[runtime] $(date --iso-8601=seconds) capturing runtime from discovered server ${discovered_url}" | tee -a "${EVAL_LOG:-/dev/null}"
        INFERENCE_BENCH_SERVER_URL="${discovered_url}"
    else
        echo "[runtime] $(date --iso-8601=seconds) capturing runtime from live server ${discovered_url}" | tee -a "${EVAL_LOG:-/dev/null}"
    fi

    pid="$(_server_pid_for_url "${discovered_url}")"
    if ! [[ "${pid}" =~ ^[0-9]+$ ]] || [ ! -r "/proc/${pid}/environ" ]; then
        echo "[runtime] $(date --iso-8601=seconds) could not identify readable server process for ${discovered_url}" | tee -a "${EVAL_LOG:-/dev/null}"
        return 0
    fi

    env_file="${JOB_DIR}/task/final_runtime_env.sh"
    meta_file="${JOB_DIR}/task/final_runtime_manifest.txt"
    python3 - "${pid}" "${env_file}" "${meta_file}" "${discovered_url}" <<'PY'
import os
import shlex
import sys
from pathlib import Path

pid, env_file, meta_file, server_url = sys.argv[1:5]
raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
env = {}
for item in raw:
    if b"=" not in item:
        continue
    key, value = item.split(b"=", 1)
    try:
        env[key.decode()] = value.decode()
    except UnicodeDecodeError:
        continue

always = {
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "PYTHONUSERBASE",
    "PIP_USER",
    "PIP_CACHE_DIR",
    "VIRTUAL_ENV",
}
prefixes = (
    "CUDA_",
    "NCCL_",
    "TORCH",
    "TRITON",
    "VLLM_",
    "SGLANG_",
    "FLASHINFER_",
    "HF_",
    "HUGGINGFACE_",
    "TRANSFORMERS_",
    "PYTHON",
    "LD_",
)
exclude = {
    "HOST",
    "PORT",
    "PWD",
    "OLDPWD",
    "SHLVL",
    "_",
    "INFERENCE_BENCH_SERVER_HOST",
    "INFERENCE_BENCH_SERVER_PORT",
    "INFERENCE_BENCH_SERVER_URL",
}

keys = []
for key in sorted(env):
    if key in exclude:
        continue
    if key in always or any(key.startswith(prefix) for prefix in prefixes):
        keys.append(key)

lines = [
    "#!/usr/bin/env bash",
    "# Captured from the live agent server after a successful agent-side run.",
    "# Sourced by the final-eval server launcher before re-running start_server.sh.",
]
for key in keys:
    lines.append(f"export {key}={shlex.quote(env[key])}")
Path(env_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(env_file, 0o600)

cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
Path(meta_file).write_text(
    f"server_url={server_url}\n"
    f"pid={pid}\n"
    f"cmdline={cmdline}\n"
    f"captured_keys={' '.join(keys)}\n",
    encoding="utf-8",
)
PY
    _snapshot_runtime_tmp_symlinks
    copy_if_exists "${env_file}" "${EVAL_DIR}/final_runtime_env.sh"
    copy_if_exists "${meta_file}" "${EVAL_DIR}/final_runtime_manifest.txt"
    mkdir -p "${EVAL_DIR}/task"
    copy_if_exists "${env_file}" "${EVAL_DIR}/task/final_runtime_env.sh"
    copy_if_exists "${meta_file}" "${EVAL_DIR}/task/final_runtime_manifest.txt"
    echo "[runtime] $(date --iso-8601=seconds) captured final runtime env from pid=${pid}" | tee -a "${EVAL_LOG:-/dev/null}"
}

ensure_inference_server_for_eval() {
    local server_url="${INFERENCE_BENCH_SERVER_URL:-http://127.0.0.1:8000}"
    local hostport="${server_url#http://}"
    hostport="${hostport#https://}"
    hostport="${hostport%%/*}"
    local host="${hostport%%:*}"
    local port="${hostport##*:}"
    if [ -z "${port}" ] || [ "${port}" = "${hostport}" ]; then
        port="8000"
    fi

    # Try to discover the server on any port so we know which port the agent used.
    local discovered_url
    if discovered_url="$(_discover_openai_server "${server_url}")"; then
        if [ "${discovered_url}" != "${server_url}" ]; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) inference server not on expected ${server_url}, but discovered at ${discovered_url}" | tee -a "${EVAL_LOG}"
            INFERENCE_BENCH_SERVER_URL="${discovered_url}"
            server_url="${discovered_url}"
        fi
    fi

    if [ "${host}" != "127.0.0.1" ] && [ "${host}" != "localhost" ] && [ -n "${host}" ]; then
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) server ${server_url} is remote and not reachable; cannot auto-start." | tee -a "${EVAL_LOG}"
        return 1
    fi

    # Always kill the agent's server and start a fresh one for eval.
    # The agent's server runs in a container that may be cleaned up before
    # evaluate.py (in a new container) can reach it, causing a race condition.
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) stopping agent server and starting fresh for eval." | tee -a "${EVAL_LOG}"
    stop_inference_server || true

    # Wait for port to be freed (TCP TIME_WAIT can hold it for up to 60s).
    local max_port_wait=30
    local port_wait=0
    while [ "${port_wait}" -lt "${max_port_wait}" ]; do
        if ! ss -tlnp "sport = :${port}" 2>/dev/null | grep -q ":${port}"; then
            break
        fi
        sleep 1
        port_wait=$((port_wait + 1))
    done
    if [ "${port_wait}" -gt 0 ]; then
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) waited ${port_wait}s for port ${port} to be freed" | tee -a "${EVAL_LOG}"
    fi
    _restore_runtime_tmp_snapshot

    # Always evaluate using the agent's start_server.sh — that is the optimization being measured.
    # start_server_original.sh is only a last-resort fallback for when the agent left start_server.sh
    # as the blank stub or never wrote one (e.g. forgot to configure it entirely).
    local _stub_marker="ERROR: start_server.sh has no engine configured"
    local launch_script
    if [ -f "${JOB_DIR}/task/start_server.sh" ] && \
       ! grep -q "${_stub_marker}" "${JOB_DIR}/task/start_server.sh"; then
        launch_script="/home/agent/task/start_server.sh"
    elif [ -f "${JOB_DIR}/task/start_server_original.sh" ] && \
         ! grep -q "${_stub_marker}" "${JOB_DIR}/task/start_server_original.sh"; then
        launch_script="/home/agent/task/start_server_original.sh"
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) agent left start_server.sh unconfigured; falling back to start_server_original.sh" | tee -a "${EVAL_LOG}"
    else
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) no configured start_server.sh found for eval server startup." | tee -a "${EVAL_LOG}"
        return 1
    fi

    local pid_file="${JOB_DIR}/task/server_${port}.pid"
    rm -f "${pid_file}" 2>/dev/null || true

    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) starting fallback inference server for final eval via ${launch_script}" | tee -a "${EVAL_LOG}"
    refresh_cuda_env_flags
    setsid apptainer exec \
        "${APPTAINER_NV_FLAG[@]}" \
        --pid \
        -c \
        --cleanenv \
        --env PATH="${CONTAINER_PATH}" \
        --env TMPDIR="/tmp" \
        --env TMP="/tmp" \
        --env TEMP="/tmp" \
        --env HF_HOME="${HF_HOME_NEW}" \
        --env HF_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env TRANSFORMERS_CACHE="${HF_HOME_NEW}/hub" \
        --env XDG_CACHE_HOME="/tmp/xdg_cache" \
        --env VLLM_TORCH_COMPILE_CACHE_DIR="/tmp/vllm_torch_compile_cache" \
        --env TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache" \
        --env TORCH_EXTENSIONS_DIR="/tmp/torch_extensions" \
        --env TRITON_CACHE_DIR="/tmp/triton_cache" \
        --env CUDA_CACHE_PATH="/tmp/cuda_cache" \
        --env PIP_USER=1 \
        --env PYTHONUSERBASE="/home/agent/.local" \
        --env PIP_CACHE_DIR="/tmp/pip_cache" \
        --env INFERENCE_BENCH_BASE_MODEL="${BASE_MODEL}" \
        --env INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN}" \
        --env INFERENCE_BENCH_SERVER_HOST="${INFERENCE_BENCH_SERVER_HOST}" \
        --env INFERENCE_BENCH_SERVER_PORT="${INFERENCE_BENCH_SERVER_PORT}" \
        --env INFERENCE_BENCH_SERVER_URL="${INFERENCE_BENCH_SERVER_URL}" \
        --env HOST="${host}" \
        --env PORT="${port}" \
        "${STARTING_POINT_ENV_ARGS[@]}" \
        "${APPTAINER_CUDA_ENV_FLAGS[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_HOME}:${HF_HOME_NEW}" \
        --bind "${INFERENCE_EVAL_BUNDLE}:/opt/inference_eval:ro" \
        "${STARTING_POINT_BIND_ARGS[@]}" \
        --home "${JOB_DIR}:/home/agent" \
        --pwd "/home/agent/task" \
        --writable-tmpfs \
        "${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif" \
        bash -lc "set -euo pipefail; if [ -f '/home/agent/task/final_runtime_env.sh' ]; then . '/home/agent/task/final_runtime_env.sh'; echo '[runtime] replayed captured server environment'; fi; exec '/opt/inference_eval/bin/launch_supervised_server.sh' '${launch_script}'" >> "${EVAL_DIR}/server.log" 2>&1 &
    local launcher_pid=$!
    remember_process_group "${launcher_pid}" "eval-server-${port}"
    echo "${launcher_pid}" > "${pid_file}"

    local wait_s="${INFERENCE_BENCH_SERVER_WAIT_S:-900}"
    if ! [[ "${wait_s}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        wait_s="900"
    fi
    local wait_int="${wait_s%.*}"
    if [ -z "${wait_int}" ] || [ "${wait_int}" -lt 1 ]; then
        wait_int=900
    fi
    local deadline=$(( $(date +%s) + wait_int ))

    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if _server_models_ok "${server_url}"; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) fallback inference server is ready at ${server_url}" | tee -a "${EVAL_LOG}"
            return 0
        fi
        if ! kill -0 "${launcher_pid}" 2>/dev/null; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) fallback inference server process exited before becoming ready (pid=${launcher_pid})" | tee -a "${EVAL_LOG}"
            return 1
        fi
        sleep 1
    done

    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) fallback inference server did not become ready within ${wait_int}s (${server_url})" | tee -a "${EVAL_LOG}"
    return 1
}

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
tree "${JOB_DIR}/task"
echo "================================"

if [ -d "${JOB_DIR}/task/submission" ]; then
    cp -r "${JOB_DIR}/task/submission" "${EVAL_DIR}/submission"
fi

python3 containers/delete_hf_models.py "${JOB_DIR}/task"

cp -r "${JOB_DIR}/task" "${EVAL_DIR}/task"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

EVAL_LOG="${EVAL_DIR}/eval.log"

echo "GPU status (before eval):"
nvidia-smi || true

capture_agent_runtime_for_final_eval
pre_eval_cleanup_barrier
echo "GPU status (after cleanup):"
nvidia-smi || true

export REPO_ROOT="$(pwd)"
export TMP_HF_CACHE="/tmp/hf_cache_inference"
DATA_BIND_EVAL=()
if [ -d "/data" ]; then
    DATA_BIND_EVAL+=(--bind "/data:/data:ro")
fi

EVAL_REQUESTS_FILE=""
for candidate in \
    "src/eval/inference/baselines/speed/torch/${EVALUATION_TASK}/${RESULT_PREFIX_SAFE}/requests.jsonl" \
    "src/eval/inference/baselines/speed/vllm/${EVALUATION_TASK}/${RESULT_PREFIX_SAFE}/requests.jsonl" \
    "src/eval/inference/baselines/speed/torch/${EVALUATION_TASK}/requests.jsonl" \
    "src/eval/inference/baselines/speed/vllm/${EVALUATION_TASK}/requests.jsonl"
do
    if [ -f "${candidate}" ]; then
        EVAL_REQUESTS_FILE="${candidate}"
        break
    fi
done
EVAL_REQUESTS_ARG=()
if [ -n "${EVAL_REQUESTS_FILE}" ]; then
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) using precomputed requests file: ${EVAL_REQUESTS_FILE}" | tee -a "${EVAL_LOG}"
    EVAL_REQUESTS_ARG=(--requests-file "${EVAL_REQUESTS_FILE}")
else
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) no precomputed requests file found; evaluator will sample dataset." | tee -a "${EVAL_LOG}"
fi

# ---------------------------------------------------------------------------
# Evaluation with retry & token-fallback (adapted from PostTrainBench).
# Phase 1: default params, up to 3 attempts
# Phase 2: halved request-timeout, up to 2 attempts
# Between attempts: kill lingering GPU compute apps to free memory.
# ---------------------------------------------------------------------------
run_eval_attempt() {
    local extra_args=("$@")
    refresh_cuda_env_flags
    timeout --signal=TERM --kill-after=30s 60m \
    apptainer exec \
        "${APPTAINER_NV_FLAG[@]}" \
        --pid \
        -c \
        --cleanenv \
        --env PATH="/root/.local/bin:/home/agent/.local/bin:$PATH" \
        --env TMPDIR="/tmp" \
        --env TMP="/tmp" \
        --env TEMP="/tmp" \
        --env HF_HOME="${HF_HOME_NEW}" \
        --env HF_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${HF_HOME_NEW}/hub" \
        --env TRANSFORMERS_CACHE="${HF_HOME_NEW}/hub" \
        --env INFERENCE_BENCH_ALLOW_HF_DOWNLOAD="${INFERENCE_BENCH_ALLOW_HF_DOWNLOAD:-0}" \
        --env INFERENCE_BENCH_BASE_MODEL="${BASE_MODEL}" \
        --env INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN}" \
        --env INFERENCE_BENCH_INPUT_TOKEN_MARGIN="${INFERENCE_BENCH_INPUT_TOKEN_MARGIN:-16}" \
        --env INFERENCE_BENCH_SCENARIO="${EVALUATION_TASK}" \
        --env INFERENCE_BENCH_DATASET_SEED="${EVAL_SEED}" \
        --env INFERENCE_BENCH_QUALITY_TAU="${INFERENCE_BENCH_QUALITY_TAU}" \
        --env INFERENCE_BENCH_QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED}" \
        --env INFERENCE_BENCH_QUALITY_MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}" \
        --env INFERENCE_BENCH_QUALITY_CONCURRENCY="${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
        --env INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY="${INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY:-}" \
        "${APPTAINER_CUDA_ENV_FLAGS[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_HOME}:${HF_HOME_NEW}" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${EVAL_DIR}:${EVAL_DIR}" \
        "${DATA_BIND_EVAL[@]}" \
        --pwd "${REPO_ROOT}" \
    "${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif" python3 "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" \
        --server-url "${INFERENCE_BENCH_SERVER_URL}" \
        --json-output-file "${EVAL_DIR}/metrics.json" \
        "${EVAL_REQUESTS_ARG[@]}" \
        --quality-tau "${INFERENCE_BENCH_QUALITY_TAU}" \
        "${extra_args[@]}" >> "${EVAL_LOG}" 2>&1
}

run_eval_with_retry() {
    local max_retries="$1"; shift
    local extra_args=("$@")
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        # If a previous attempt already produced metrics, we're done.
        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
        # Kill lingering GPU compute apps between retries to free memory.
        if [ "$attempt" -gt 1 ]; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) cleaning GPU state before retry ${attempt}/${max_retries}" | tee -a "${EVAL_LOG}"
            nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
            sleep 5
        fi
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) eval attempt ${attempt}/${max_retries} ${extra_args[*]}" | tee -a "${EVAL_LOG}"
        set +e
        run_eval_attempt "${extra_args[@]}"
        FINAL_EVAL_RC=$?
        set -e
        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) attempt ${attempt} failed (rc=${FINAL_EVAL_RC})" | tee -a "${EVAL_LOG}"
    done
    return 1
}

FINAL_EVAL_RC=1
# Clear any error-stub metrics.json left behind by a prior invocation of this
# job (e.g. a checkpoint-resume or a re-run of the same cluster). Otherwise the
# stub causes run_eval_with_retry to short-circuit on file existence and never
# execute a real eval attempt.
if [ -f "${EVAL_DIR}/metrics.json" ] && python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if "error" in d and not d.get("profiles") else 1)
' "${EVAL_DIR}/metrics.json" 2>/dev/null; then
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) clearing stale error-stub metrics.json from prior pass" | tee -a "${EVAL_LOG}"
    rm -f "${EVAL_DIR}/metrics.json"
fi
if ensure_inference_server_for_eval; then
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) starting final evaluation (speed + quality)" | tee -a "${EVAL_LOG}"

    # Phase 1: default parameters, up to 3 attempts.
    run_eval_with_retry 3

    # Phase 2: halved request timeout, up to 2 attempts.
    if [ ! -f "${EVAL_DIR}/metrics.json" ]; then
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) phase 1 failed; retrying with reduced request-timeout" | tee -a "${EVAL_LOG}"
        run_eval_with_retry 2 --request-timeout-s 150
    fi
else
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) final evaluation aborted: inference server could not be started/reached at ${INFERENCE_BENCH_SERVER_URL}" | tee -a "${EVAL_LOG}"
    python3 - <<PY
import json
from pathlib import Path

metrics = {
    "scenario": "${EVALUATION_TASK}",
    "model_id": "${BASE_MODEL}",
    "profiles": {},
    "vram_peak_mb": 0.0,
    "error": "inference server unavailable before final eval at ${INFERENCE_BENCH_SERVER_URL}",
}
Path("${EVAL_DIR}/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
PY
fi
echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) final eval exit code: ${FINAL_EVAL_RC}" | tee -a "${EVAL_LOG}"

echo "GPU status (after eval):"
nvidia-smi || true

stop_inference_server || true

rm -rf "${TMP_SUBDIR}"

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"
