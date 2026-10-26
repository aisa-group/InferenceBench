#!/bin/bash
set -eo pipefail
umask 077

if [ "$#" -lt 6 ] || [ "$#" -gt 9 ]; then
    echo "Usage: $0 TASK AGENT MODEL RUN_ID HOURS AGENT_CONFIG [STARTING_POINT] [AGENT_SEED] [EVAL_SEED]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"

export EVALUATION_TASK="$1"
AGENT="$2"
BASE_MODEL="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
STARTING_POINT="${7:-${INFERENCE_BENCH_STARTING_POINT:-default}}"
if ! [[ "${EVALUATION_TASK}" =~ ^inference_scenario_[a-z]_[a-z_]+$ ]] ||
   ! [[ "${AGENT}" =~ ^[a-z_]+$ ]] ||
   ! [[ "${CLUSTER_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
   ! [[ "${NUM_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! [[ "${STARTING_POINT}" =~ ^(default|vllm_running|bare)$ ]] ||
   [ ! -f "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" ] ||
   [ ! -f "agents/${AGENT}/solve.sh" ]; then
    echo "[fatal] invalid task, agent, run ID, time budget, or starting point" >&2
    exit 2
fi

NUM_SECONDS="$(awk -v h="${NUM_HOURS}" 'BEGIN{printf "%d", h*3600}')"
if [ "${NUM_SECONDS}" -le 0 ]; then
    echo "[fatal] time budget must be at least one second" >&2
    exit 2
fi

source src/commit_utils/set_env_vars.sh
source src/disallowed_usage_judge/judge.sh
AGENT_SEED="${8:-${INFERENCE_BENCH_DATASET_SEED}}"
EVAL_SEED="${9:-${INFERENCE_BENCH_DATASET_SEED}}"
validate_agent_image "${AGENT}" "${STARTING_POINT}"

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}" ] && [ ! -s "agents/claude_non_api/oauth_token" ]; then
    echo "[fatal] the disallowed-usage judge requires ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or agents/claude_non_api/oauth_token" >&2
    exit 2
fi

RESULT_PREFIX_SAFE="$(printf '%s' "$BASE_MODEL" | tr '/:' '_')"
AGENT_CONFIG_SAFE="$(printf '%s' "$AGENT_CONFIG" | tr -c '[:alnum:]._-' '_')"

SP_SUFFIX=""
if [ "${STARTING_POINT}" != "default" ]; then
    SP_SUFFIX="_sp-${STARTING_POINT}"
fi
export EVAL_DIR="${INFERENCE_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${INFERENCE_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}${SP_SUFFIX}_${CLUSTER_ID}"

mkdir -p "${EVAL_DIR}"
export EVAL_DIR="$(realpath "${EVAL_DIR}")"
EVAL_LOG="${EVAL_DIR}/eval.log"
python3 - "${EVAL_DIR}/run_config.json" "${EVALUATION_TASK}" "${BASE_MODEL}" "${AGENT}" "${AGENT_CONFIG}" "${STARTING_POINT}" "${NUM_HOURS}" "${AGENT_SEED}" "${EVAL_SEED}" \
    "${INFERENCE_BENCH_QUALITY_SEED}" "${INFERENCE_BENCH_QUALITY_MMLUPRO_N}" "${INFERENCE_BENCH_QUALITY_TAU}" "${INFERENCE_BENCH_QUALITY_CONCURRENCY}" \
    "${INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY:-}" "${INFERENCE_BENCH_QUALITY_BASELINE_BACKEND:-torch}" "${INFERENCE_BENCH_MAX_MODEL_LEN}" \
    "${INFERENCE_BENCH_INPUT_TOKEN_MARGIN:-16}" "${INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE:-}" <<'PY'
import json
import sys
from pathlib import Path

path, *values = sys.argv[1:]
keys = ("scenario", "base_model", "agent", "agent_config", "starting_point", "num_hours", "agent_seed", "eval_seed",
        "quality_seed", "quality_mmlupro_n", "quality_tau", "quality_concurrency", "quality_baseline_registry",
        "quality_baseline_backend", "max_model_len", "input_token_margin", "quality_samples_file")
Path(path).write_text(json.dumps(dict(zip(keys, values)), indent=2) + "\n", encoding="utf-8")
PY

CHECKPOINT_DIR="${EVAL_DIR}/checkpoint"
CHECKPOINT_REMAINING_SECONDS=""
CHECKPOINT_AGENT_RESTORED=0

exec >"${EVAL_DIR}/output.log" 2>"${EVAL_DIR}/error.log"

CURRENT_AGENT_PID=""

_snapshot_runtime_tmp_symlinks() {
    local snapshot_dir="${JOB_DIR}/runtime_snapshot/tmp"
    local manifest="${JOB_DIR}/runtime_snapshot/tmp_symlinks.tsv"
    rm -rf "${JOB_DIR}/runtime_snapshot" || return 1
    mkdir -p "${snapshot_dir}" || return 1
    : > "${manifest}" || return 1

    local link target rel host_target resolved_target scratch_root
    scratch_root="$(realpath -e -- "${JOB_TMP}")" || return 1
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
                    resolved_target="$(realpath -e -- "${host_target}")" || continue
                    case "${resolved_target}" in
                        "${scratch_root}"/*) ;;
                        *) continue ;;
                    esac
                    mkdir -p "$(dirname "${snapshot_dir}/${rel}")" || return 1
                    cp -aT -- "${host_target}" "${snapshot_dir}/${rel}" || return 1
                    printf '%s\t%s\n' "$(basename "${link}")" "${rel}" >> "${manifest}" || return 1
                fi
                ;;
        esac
    done < <(find "${JOB_DIR}/task" -maxdepth 1 -type l 2>/dev/null || true)
}

_restore_runtime_tmp_snapshot() {
    local snapshot_dir="${JOB_DIR}/runtime_snapshot/tmp"
    local manifest="${JOB_DIR}/runtime_snapshot/tmp_symlinks.tsv"
    if [ ! -s "${manifest}" ] || [ ! -d "${snapshot_dir}" ]; then
        return 0
    fi

    mkdir -p "${JOB_TMP}" || return 1
    local link_name rel
    while IFS=$'\t' read -r link_name rel; do
        [ -n "${rel}" ] || continue
        case "${rel}" in
            *..*) continue ;;
        esac
        if [ -e "${snapshot_dir}/${rel}" ]; then
            mkdir -p "$(dirname "${JOB_TMP}/${rel}")" || return 1
            rm -rf "${JOB_TMP:?}/${rel}" || return 1
            cp -a "${snapshot_dir}/${rel}" "${JOB_TMP}/${rel}" || return 1
        else
            echo "[runtime] missing snapshot for ${rel}" >&2
            return 1
        fi
    done < "${manifest}"
}

_save_checkpoint_state() {
    local destination="$1"
    local remaining="$2"
    [ -d "${JOB_DIR}/task" ] && [ ! -L "${JOB_DIR}/task" ] || return 1
    _snapshot_runtime_tmp_symlinks || return 1
    local entry
    for entry in task .local .codex .claude runtime_snapshot; do
        if [ -e "${JOB_DIR}/${entry}" ] || [ -L "${JOB_DIR}/${entry}" ]; then
            [ ! -L "${JOB_DIR}/${entry}" ] || return 1
            cp -a -- "${JOB_DIR}/${entry}" "${destination}/" || return 1
        fi
    done
    if [ "${HF_HOME:-}" = "${JOB_DIR}/hf_cache" ]; then
        [ ! -L "${JOB_DIR}/hf_cache" ] || return 1
        cp -a -- "${JOB_DIR}/hf_cache" "${destination}/" || return 1
    fi
    rm -f -- "${destination}/.codex/auth.json" "${destination}/.claude/.credentials.json" || return 1
    printf '%s\n' "${remaining}" > "${destination}/remaining_seconds" || return 1
    : > "${destination}/complete" || return 1
}

_restore_checkpoint_state() {
    [ -f "${CHECKPOINT_DIR}/complete" ] && [ -d "${CHECKPOINT_DIR}/task" ] &&
        [ -d "${CHECKPOINT_DIR}/runtime_snapshot" ] &&
        [ -d "${CHECKPOINT_DIR}/hf_cache" ] && [ ! -L "${CHECKPOINT_DIR}/hf_cache" ] || return 1
    CHECKPOINT_REMAINING_SECONDS="$(cat "${CHECKPOINT_DIR}/remaining_seconds")" || return 1
    [[ "${CHECKPOINT_REMAINING_SECONDS}" =~ ^[0-9]+$ ]] || return 1
    local entry
    for entry in task .local .codex .claude runtime_snapshot hf_cache; do
        if [ -e "${CHECKPOINT_DIR}/${entry}" ] || [ -L "${CHECKPOINT_DIR}/${entry}" ]; then
            [ ! -L "${CHECKPOINT_DIR}/${entry}" ] || return 1
            rm -rf -- "${JOB_DIR:?}/${entry}" || return 1
            cp -a -- "${CHECKPOINT_DIR}/${entry}" "${JOB_DIR}/" || return 1
        fi
    done
    _restore_runtime_tmp_snapshot || return 1
    CHECKPOINT_AGENT_RESTORED=1
    echo "[harness] checkpoint restored: remaining=${CHECKPOINT_REMAINING_SECONDS}s"
}

_checkpoint_and_exit() {
    trap '' SIGTERM
    echo "[harness] SIGTERM received; saving checkpoint to ${CHECKPOINT_DIR}"
    if [ -n "${CURRENT_AGENT_PID:-}" ]; then
        kill -TERM "${CURRENT_AGENT_PID}" 2>/dev/null || true
        sleep 5
        kill -KILL "${CURRENT_AGENT_PID}" 2>/dev/null || true
        wait "${CURRENT_AGENT_PID}" 2>/dev/null || true
    fi
    if declare -F _stop_recorded_process_groups >/dev/null; then
        _stop_recorded_process_groups
    fi
    if [ -z "${AGENT_DEADLINE_TS:-}" ] || [ -z "${JOB_DIR:-}" ]; then
        echo "[harness] cannot checkpoint before the agent phase starts" >&2
        exit 1
    fi
    local remaining=$(( AGENT_DEADLINE_TS - $(date +%s) ))
    (( remaining >= 0 )) || remaining=0
    local staged previous
    staged="$(mktemp -d "${CHECKPOINT_DIR}.XXXXXXXX")" || exit 1
    if ! _save_checkpoint_state "${staged}" "${remaining}"; then
        echo "[harness] checkpoint failed; preserving the previous checkpoint" >&2
        rm -rf -- "${staged}"
        exit 1
    fi
    previous="${staged}.previous"
    if [ -e "${CHECKPOINT_DIR}" ]; then
        mv -- "${CHECKPOINT_DIR}" "${previous}" || exit 1
    fi
    if ! mv -- "${staged}" "${CHECKPOINT_DIR}"; then
        [ ! -e "${previous}" ] || mv -- "${previous}" "${CHECKPOINT_DIR}"
        exit 1
    fi
    rm -rf -- "${previous}"
    echo "[harness] checkpoint saved: remaining=${remaining}s"
    exit 85
}
trap _checkpoint_and_exit SIGTERM

if [ "${INFERENCE_BENCH_LOAD_CUDA_MODULE:-1}" != "0" ] && [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh >/dev/null 2>&1 && module load cuda >/dev/null 2>&1 || true
fi
for dir in /usr/local/sbin /usr/local/bin /usr/sbin /usr/bin /sbin /bin; do
    [[ ":${PATH}:" == *":${dir}:"* ]] || PATH="${PATH:+${PATH}:}${dir}"
done
export PATH
hash -r

missing_tools=()
for tool in apptainer awk bash cp date find grep mkdir mktemp ps python3 realpath rm sed setsid ss stdbuf tee timeout tr; do
    command -v "${tool}" >/dev/null 2>&1 || missing_tools+=("${tool}")
done
if [ "${#missing_tools[@]}" -gt 0 ]; then
    echo "[fatal] required host tools unavailable: ${missing_tools[*]} (PATH=${PATH})"
    exit 1
fi

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
SERVER_WAIT_S="${INFERENCE_BENCH_SERVER_WAIT_S:-900}"
SERVER_WAIT_S="${SERVER_WAIT_S%.*}"
[[ "${SERVER_WAIT_S}" =~ ^[0-9]+$ ]] && [ "${SERVER_WAIT_S}" -ge 1 ] || SERVER_WAIT_S=900
export INFERENCE_BENCH_SERVER_WAIT_S="${SERVER_WAIT_S}"

server_is_local() {
    [ "${SERVER_HOST}" = "127.0.0.1" ] || [ "${SERVER_HOST}" = "localhost" ]
}

port_listeners() {
    ss -lptn "sport = :${SERVER_PORT}" 2>/dev/null | sed '/^State/d'
}

TMP_ROOT="${INFERENCE_BENCH_TMP_ROOT:-${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}}"
mkdir -p "${TMP_ROOT}"
TMP_SUBDIR="$(mktemp -d "${TMP_ROOT}/inferencebench.XXXXXXXX")"
export TMP_SUBDIR

JOB_DIR="${TMP_SUBDIR}/job_dir"
JOB_TMP="${TMP_SUBDIR}/tmp"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${JOB_TMP}/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${JOB_TMP}/apptainer_cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
mkdir -p "${JOB_DIR}/hf_cache"
mkdir -p "${JOB_DIR}/task/.local/lib/python3.10/site-packages/pyairports"
CONTAINER_PATH="/root/.local/bin:/home/agent/.local/bin:$PATH"
if [ "${STARTING_POINT}" = "vllm_running" ]; then
    INFERENCE_BENCH_CONTAINER_NAME="${INFERENCE_BENCH_CONTAINER_NAME_VLLM_RUNNING:-${INFERENCE_BENCH_BASELINE_CONTAINER_VLLM_NAME}}"
fi

HF_CACHE_SOURCE="${HF_HOME:-}"
export HF_HOME="${JOB_DIR}/hf_cache"
unset APPTAINER_BIND APPTAINER_BINDPATH APPTAINER_MOUNT SINGULARITY_BIND SINGULARITY_BINDPATH SINGULARITY_MOUNT
export HF_HOME_NEW="/hf_cache"

APPTAINER_NV_FLAG=()
if ls /dev/nvidia* >/dev/null 2>&1; then
    APPTAINER_NV_FLAG+=(--nv)
else
    echo "[warn] no /dev/nvidia* devices on the host; running Apptainer without --nv"
fi

init_cuda_visibility_env() {
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
                v="$(printf '%s\n%s\n' "${proc_uuid_map}" "${uuid_map}" | awk -F',' -v want="${tok}" '
                    {
                        uuid=$2;
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid);
                        low_uuid=tolower(uuid); low_want=tolower(want);
                        if (low_uuid == low_want || index(low_uuid, low_want) == 1) { print uuid; exit 0; }
                    }')"
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
            echo "[fatal] cannot map assigned GPUs: ${_CONDOR_AssignedGPUs}" >&2
            return 1
        fi
    fi
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
        local table_idx
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

init_cuda_visibility_env

: > "${JOB_DIR}/task/.local/lib/python3.10/site-packages/pyairports/__init__.py"
cat > "${JOB_DIR}/task/.local/lib/python3.10/site-packages/pyairports/airports.py" <<'PY'
AIRPORT_LIST = []
PY
cat > "${JOB_DIR}/task/.local/lib/python3.10/site-packages/sitecustomize.py" <<'PY'
try:
    import torch

    if hasattr(torch, "cpu") and not hasattr(torch.cpu, "is_bf16_supported"):
        def _inferencebench_cpu_is_bf16_supported() -> bool:
            try:
                return bool(getattr(torch.backends.cpu, "has_bf16", False))
            except Exception:
                return False
        torch.cpu.is_bf16_supported = _inferencebench_cpu_is_bf16_supported
except Exception:
    pass
PY

echo "Preparing job directory (starting_point=${STARTING_POINT})..."
SHARED_TASK_CONTEXT_DIR="src/eval/tasks/_shared/task_context"

case "${STARTING_POINT}" in
  default|vllm_running)
    for f in mission.txt benchmark.txt scenario.json; do
        cp "src/eval/tasks/${EVALUATION_TASK}/${f}" "${JOB_DIR}/task"
    done
    cp -r "${SHARED_TASK_CONTEXT_DIR}/." "${JOB_DIR}/task"
    if [ "${STARTING_POINT}" = "vllm_running" ]; then
        cp "src/starting_points/vllm_running/start_server.sh" "${JOB_DIR}/task/start_server.sh"
        cp "src/starting_points/vllm_running/start_server.sh" "${JOB_DIR}/task/start_server_original.sh"
    fi
    ;;
  bare)
    cp "src/eval/tasks/${EVALUATION_TASK}/scenario.json" "${JOB_DIR}/task"
    cp "${SHARED_TASK_CONTEXT_DIR}/test_server.sh" "${JOB_DIR}/task/test_server.sh"
    ;;
esac

cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"
if [ -d "${CHECKPOINT_DIR}" ]; then
    if ! _restore_checkpoint_state; then
        echo "[fatal] checkpoint is incomplete or could not be restored" >&2
        exit 1
    fi
elif [ "${STARTING_POINT}" != "bare" ]; then
    python3 src/utils/prepare_hf_cache.py \
        --source "${HF_CACHE_SOURCE}" --destination "${HF_HOME}" --model "${BASE_MODEL}"
fi

cat > "${JOB_DIR}/task/evaluate.py" <<'PY'
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, "/opt")
from inference_eval.runner import build_parser, run_evaluation


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(Path(__file__).resolve().parent, args)


if __name__ == "__main__":
    main()
PY
find "${JOB_DIR}/task" -maxdepth 1 -type f -name "*.sh" -exec chmod +x {} + 2>/dev/null || true

BENCHMARK="$(cat "src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")"
export INFERENCE_BENCH_METRICS_PATH="${INFERENCE_BENCH_METRICS_PATH:-/home/agent/task/metrics_preview.json}"
PROMPT="$(python3 src/eval/general/get_prompt.py --base-model "$BASE_MODEL" --scenario-id "$EVALUATION_TASK" --num-hours "$NUM_HOURS" --agent "${AGENT}" --starting-point "${STARTING_POINT}")"
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"
echo "$PROMPT" > "${JOB_DIR}/prompt.txt"
export PROMPT_FILE="/home/agent/prompt.txt"

cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"

AUTH_FILES=(.codex/auth.json oauth_token api_key)
for f in "${AUTH_FILES[@]}"; do
    [ ! -f "agents/${AGENT}/${f##*/}" ] || cp "agents/${AGENT}/${f##*/}" "${JOB_DIR}/${f}"
done

INFERENCE_EVAL_BUNDLE="${JOB_DIR}/inference_eval"
mkdir -p "${INFERENCE_EVAL_BUNDLE}"
cp src/eval/inference/__init__.py "${INFERENCE_EVAL_BUNDLE}/"
cp src/eval/inference/runner.py "${INFERENCE_EVAL_BUNDLE}/"
cp src/eval/inference/quality_gate.py "${INFERENCE_EVAL_BUNDLE}/"
cp src/eval/inference/cache_samples.py "${INFERENCE_EVAL_BUNDLE}/"
mkdir -p "${INFERENCE_EVAL_BUNDLE}/bin"
cp src/eval/inference/bin/launch_supervised_server.sh src/eval/inference/bin/server_ready.py "${INFERENCE_EVAL_BUNDLE}/bin/"
chmod +x "${INFERENCE_EVAL_BUNDLE}/bin/launch_supervised_server.sh"
mkdir -p "${INFERENCE_EVAL_BUNDLE}/baselines"
cp -r src/eval/inference/baselines/quality "${INFERENCE_EVAL_BUNDLE}/baselines/"
cp -r src/eval/inference/baselines/samples "${INFERENCE_EVAL_BUNDLE}/baselines/"

SIF="${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif"
DATA_BIND=()
if [ -d "/data" ]; then
    DATA_BIND+=(--bind "/data:/data:ro")
fi
BASE_CONTAINER=(
    --pid -c --cleanenv
    --no-mount cwd,hostfs,bind-paths
    --bind /etc/resolv.conf:/etc/resolv.conf:ro
    --env PATH="${CONTAINER_PATH}"
    --bind "${JOB_TMP}:/tmp"
)
GPU_CONTAINER=(
    "${APPTAINER_NV_FLAG[@]}"
    "${BASE_CONTAINER[@]}"
    --env TMPDIR="/tmp"
    --env TMP="/tmp"
    --env TEMP="/tmp"
    --env HF_HOME="${HF_HOME_NEW}"
    --env HF_HUB_CACHE="${HF_HOME_NEW}/hub"
    --env HUGGINGFACE_HUB_CACHE="${HF_HOME_NEW}/hub"
    --env TRANSFORMERS_CACHE="${HF_HOME_NEW}/hub"
    --env INFERENCE_BENCH_BASE_MODEL="${BASE_MODEL}"
    --env INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN}"
    --bind "${HF_HOME}:${HF_HOME_NEW}"
)
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] || GPU_CONTAINER+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
[ -z "${NVIDIA_VISIBLE_DEVICES:-}" ] || GPU_CONTAINER+=(--env "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}")
AGENT_CONTAINER=(
    "${GPU_CONTAINER[@]}"
    --env XDG_CACHE_HOME="/tmp/xdg_cache"
    --env VLLM_TORCH_COMPILE_CACHE_DIR="/tmp/vllm_torch_compile_cache"
    --env TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache"
    --env TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
    --env TRITON_CACHE_DIR="/tmp/triton_cache"
    --env CUDA_CACHE_PATH="/tmp/cuda_cache"
    --env PIP_USER=1
    --env PYTHONUSERBASE="/home/agent/.local"
    --env PIP_CACHE_DIR="/tmp/pip_cache"
    --env INFERENCE_BENCH_SERVER_URL="${INFERENCE_BENCH_SERVER_URL}"
    --env INFERENCE_BENCH_SERVER_WAIT_S="${SERVER_WAIT_S}"
    --env HOST="${SERVER_HOST}"
    --env PORT="${SERVER_PORT}"
    --bind "${INFERENCE_EVAL_BUNDLE}:/opt/inference_eval:ro"
    --home "${JOB_DIR}:/home/agent"
    --pwd "/home/agent/task"
    --writable-tmpfs
)
QUALITY_ENV=(
    --env INFERENCE_BENCH_INPUT_TOKEN_MARGIN="${INFERENCE_BENCH_INPUT_TOKEN_MARGIN:-16}"
    --env INFERENCE_BENCH_SCENARIO="${EVALUATION_TASK}"
    --env INFERENCE_BENCH_QUALITY_TAU="${INFERENCE_BENCH_QUALITY_TAU}"
    --env INFERENCE_BENCH_QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED}"
    --env INFERENCE_BENCH_QUALITY_MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}"
    --env INFERENCE_BENCH_QUALITY_CONCURRENCY="${INFERENCE_BENCH_QUALITY_CONCURRENCY}"
    --env INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY="${INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY:-}"
    --env INFERENCE_BENCH_QUALITY_BASELINE_BACKEND="${INFERENCE_BENCH_QUALITY_BASELINE_BACKEND:-torch}"
    --env INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE="${INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE:-}"
)

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
    if ! [[ "${pgid}" =~ ^[0-9]+$ ]]; then
        return 0
    fi
    local current_pgid
    current_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"
    if [ -n "${current_pgid}" ] && [ "${pgid}" = "${current_pgid}" ]; then
        echo "[proc] refusing to kill the current shell process group pgid=${pgid}" | tee -a "${EVAL_LOG}"
        return 1
    fi
    if [ -z "$(_list_process_group_members "${pgid}")" ]; then
        return 0
    fi
    kill -- "-${pgid}" 2>/dev/null || true
    sleep 2
    if [ -n "$(_list_process_group_members "${pgid}")" ]; then
        kill -9 -- "-${pgid}" 2>/dev/null || true
        sleep 2
    fi
}

_stop_recorded_process_groups() {
    local pgid
    for pgid in "${JOB_RECORDED_PGIDS[@]}"; do
        _stop_process_group "${pgid}" || true
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

cleanup_job() {
    local rc=$?
    trap - EXIT
    _stop_recorded_process_groups || true
    rm -rf "${TMP_SUBDIR}"
    exit "${rc}"
}
trap cleanup_job EXIT

_reset_job_scratch_for_eval() {
    if [ -d "${TMP_SUBDIR}" ]; then
        find "${TMP_SUBDIR}" -mindepth 1 -maxdepth 1 ! -name "$(basename "${JOB_DIR}")" -exec rm -rf {} + 2>/dev/null || true
    fi
    mkdir -p "${JOB_TMP}" "${APPTAINER_TMPDIR}"
}

pre_eval_cleanup_barrier() {
    local attempt
    for attempt in 1 2 3; do
        stop_inference_server || true
        _stop_recorded_process_groups || true
        _reset_job_scratch_for_eval
        sleep 2
        local clean=1
        if ! _recorded_process_groups_clear; then
            clean=0
            echo "[${EVALUATION_TASK}] tracked processes survived cleanup pass ${attempt}" | tee -a "${EVAL_LOG}"
        fi
        if server_is_local && [ -n "$(port_listeners || true)" ]; then
            clean=0
            echo "[${EVALUATION_TASK}] port ${SERVER_PORT} still has listeners after cleanup pass ${attempt}" | tee -a "${EVAL_LOG}"
        fi
        if [ "${clean}" = "1" ]; then
            return 0
        fi
    done
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) pre-eval cleanup failed" | tee -a "${EVAL_LOG}"
    return 1
}

solve_task() {
    local timeout_seconds="$1"
    local run_index="$2"
    local -x APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}"
    local -x APPTAINERENV_ANTHROPIC_API_KEY=""
    local -x APPTAINERENV_CODEX_API_KEY=""
    local -x APPTAINERENV_OPENAI_API_KEY=""
    local -x APPTAINERENV_GEMINI_API_KEY=""
    case "${AGENT}" in
        claude) APPTAINERENV_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" ;;
        codex) APPTAINERENV_CODEX_API_KEY="${OPENAI_API_KEY:-}" ;;
        gemini) APPTAINERENV_GEMINI_API_KEY="${GEMINI_API_KEY:-}" ;;
    esac
    local solve_out="${EVAL_DIR}/solve_out_run${run_index}.txt"
    cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"
    cp agents/common.sh "${JOB_DIR}/agent_common.sh"
    echo "[runner] $(date --iso-8601=seconds) launching agent run ${run_index} (timeout=${timeout_seconds}s)"
    setsid timeout --signal=TERM --kill-after=30s "${timeout_seconds}s" \
      apptainer exec \
        "${AGENT_CONTAINER[@]}" \
        "${QUALITY_ENV[@]}" \
        --env RAY_DISABLE_DASHBOARD="1" \
        --env RAY_USAGE_STATS_ENABLED="0" \
        --env RAY_DISABLE_METRICS_EXPORT="1" \
        --env INFERENCE_BENCH_DATASET_SEED="${AGENT_SEED}" \
        --env INFERENCE_BENCH_RUN_INDEX="${run_index}" \
        --env NUM_HOURS="${NUM_HOURS}" \
        --env INFERENCE_BENCH_REMAINING_SECONDS="${timeout_seconds}" \
        --env PROMPT_FILE="${PROMPT_FILE}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        "${DATA_BIND[@]}" \
        "${SIF}" \
        bash -c "set -euo pipefail; timeout --signal=TERM --kill-after=10s 120s python3 -u /home/agent/check_cuda.py || echo '[runner] WARNING: CUDA check failed; continuing'; bash /home/agent/agent_solve.sh" \
        > >(stdbuf -oL tr -d '\000' | tee "${solve_out}") 2>&1 &
    local session_pid=$!
    CURRENT_AGENT_PID=${session_pid}
    remember_process_group "${session_pid}"
    local rc=0
    wait "${session_pid}" || rc=$?
    CURRENT_AGENT_PID=""
    return "${rc}"
}

run_agent_until_deadline() {
    local run_index=1
    if [ "${CHECKPOINT_AGENT_RESTORED:-0}" = "1" ]; then
        run_index=2
    fi
    local rc=0
    local remaining=0
    while :; do
        remaining=$((AGENT_DEADLINE_TS - $(date +%s)))
        if [ "${remaining}" -le 0 ] || [ "${run_index}" -gt 3 ]; then
            return "${rc}"
        fi
        solve_task "${remaining}" "${run_index}"
        rc=$?
        echo "Agent run ${run_index} exit code: ${rc}"
        if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 0 ]; then
            return "${rc}"
        fi
        remaining=$((AGENT_DEADLINE_TS - $(date +%s)))
        if [ "${remaining}" -lt 60 ]; then
            return "${rc}"
        fi
        echo "Agent run ${run_index} crashed; retrying in a fresh container with ${remaining}s remaining."
        sleep 5
        run_index=$((run_index + 1))
    done
}

AGENT_START_TS=$(date +%s)
if [ -n "${CHECKPOINT_REMAINING_SECONDS}" ]; then
    AGENT_DEADLINE_TS=$((AGENT_START_TS + CHECKPOINT_REMAINING_SECONDS))
    echo "[harness] resuming from checkpoint: ${CHECKPOINT_REMAINING_SECONDS}s budget"
else
    AGENT_DEADLINE_TS=$((AGENT_START_TS + NUM_SECONDS))
fi
bash src/utils/create_timer.sh "$((AGENT_DEADLINE_TS - AGENT_START_TS))" "${JOB_DIR}/task/timer.sh"

set +e
with_record_the_time run_agent_until_deadline
AGENT_RC=$?
set -e
echo "Agent phase exit code: ${AGENT_RC}"
rm -f "${AUTH_FILES[@]/#/${JOB_DIR}/}"

echo "Running the disallowed-usage judge..."
JUDGE_TASK=$(python3 src/disallowed_usage_judge/get_judge_prompt.py --benchmark "${BENCHMARK}" --model "${BASE_MODEL}" --task-dir "${JOB_DIR}/task")
JUDGE_HOME="${TMP_SUBDIR}/judge_home"
mkdir -p "${JUDGE_HOME}"

run_disallowed_judge_in_container() {
    local -x APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN APPTAINERENV_ANTHROPIC_API_KEY
    judge_credentials
    timeout --signal=TERM --kill-after=30s 15m apptainer exec \
        "${BASE_CONTAINER[@]}" \
        --bind "${JOB_DIR}/task:/home/agent/task:ro" \
        --home "${JUDGE_HOME}:/home/agent" \
        --pwd "/home/agent" \
        --writable-tmpfs \
        "${SIF}" \
        claude --print --dangerously-skip-permissions --model "${DEFAULT_JUDGE_MODEL}" "${JUDGE_TASK}"
}

set +e
run_disallowed_judge_in_container
JUDGE_RC=$?
set -e
echo "Disallowed-usage judge exit code: ${JUDGE_RC}"

copy_if_exists() {
    local src="${1:-}"
    local dst="${2:-}"
    local missing_warn="${3:-}"
    if [ -f "${src}" ] && [ ! -L "${src}" ]; then
        cp --remove-destination -- "${src}" "${dst}"
    elif [ -n "${missing_warn}" ]; then
        echo "[warn] ${missing_warn}"
    fi
}

valid_judgements "${JUDGE_HOME}" || JUDGE_RC=1
copy_if_exists "${JUDGE_HOME}/contamination_judgement.txt" "${EVAL_DIR}/contamination_judgement.txt" "contamination_judgement.txt was not produced."
copy_if_exists "${JUDGE_HOME}/disallowed_model_judgement.txt" "${EVAL_DIR}/disallowed_model_judgement.txt" "disallowed_model_judgement.txt was not produced."
copy_if_exists "${JOB_DIR}/task/metrics_preview.json" "${EVAL_DIR}/metrics_preview.json"
copy_if_exists "${JOB_DIR}/task/server.log" "${EVAL_DIR}/server.log"

_pid_belongs_to_job() {
    local pid="${1:-}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    local pgid group
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    for group in "${JOB_RECORDED_PGIDS[@]}"; do
        [ "${pgid}" != "${group}" ] || return 0
    done
    return 1
}

_stop_pids() {
    local pids="${1:-}"
    local pid
    for pid in ${pids}; do
        if _pid_belongs_to_job "${pid}"; then
            kill "${pid}" 2>/dev/null || true
            sleep 2
            if _pid_belongs_to_job "${pid}"; then
                kill -9 "${pid}" 2>/dev/null || true
            fi
        fi
    done
}

stop_inference_server() {
    local pid_files=(
        "${JOB_DIR}/task/server.pid"
        "${JOB_DIR}/task/server_${SERVER_PORT}.pid"
        "${EVAL_DIR}/task/server.pid"
        "${EVAL_DIR}/task/server_${SERVER_PORT}.pid"
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
        if _pid_belongs_to_job "${pid}" && kill -0 "${pid}" 2>/dev/null; then
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
    if server_is_local; then
        _stop_pids "$(port_listeners | grep -o 'pid=[0-9]*' | cut -d= -f2)"
    fi
}

_server_models_ok() {
    python3 src/eval/inference/bin/server_ready.py "${INFERENCE_BENCH_SERVER_URL}"
}

capture_agent_runtime_for_final_eval() {
    local pid env_file meta_file
    _snapshot_runtime_tmp_symlinks || return 1
    if ! _server_models_ok; then
        echo "[runtime] no live agent server found for runtime capture" | tee -a "${EVAL_LOG}"
        return 0
    fi
    pid="$(port_listeners | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)" || pid=""
    if ! _pid_belongs_to_job "${pid}" || [ ! -r "/proc/${pid}/environ" ]; then
        echo "[runtime] could not identify a readable server process for ${INFERENCE_BENCH_SERVER_URL}" | tee -a "${EVAL_LOG}"
        return 0
    fi
    env_file="${JOB_DIR}/task/final_runtime_env.sh"
    meta_file="${JOB_DIR}/task/final_runtime_manifest.txt"
    python3 - "${pid}" "${env_file}" "${meta_file}" "${INFERENCE_BENCH_SERVER_URL}" <<'PY' || return 1
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
keys = []
for key in sorted(env):
    if any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")):
        continue
    if key in always or any(key.startswith(prefix) for prefix in prefixes):
        keys.append(key)

lines = ["#!/usr/bin/env bash"] + [f"export {key}={shlex.quote(env[key])}" for key in keys]
Path(env_file).unlink(missing_ok=True)
Path(env_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(env_file, 0o600)
Path(meta_file).unlink(missing_ok=True)
Path(meta_file).write_text(
    f"server_url={server_url}\n"
    f"pid={pid}\n"
    f"captured_keys={' '.join(keys)}\n",
    encoding="utf-8",
)
PY
    copy_if_exists "${env_file}" "${EVAL_DIR}/final_runtime_env.sh"
    copy_if_exists "${meta_file}" "${EVAL_DIR}/final_runtime_manifest.txt"
    mkdir -p "${EVAL_DIR}/task"
    copy_if_exists "${env_file}" "${EVAL_DIR}/task/final_runtime_env.sh"
    copy_if_exists "${meta_file}" "${EVAL_DIR}/task/final_runtime_manifest.txt"
    echo "[runtime] captured the server environment from pid=${pid}" | tee -a "${EVAL_LOG}"
}

ensure_inference_server_for_eval() {
    if ! server_is_local; then
        echo "[${EVALUATION_TASK}] server ${INFERENCE_BENCH_SERVER_URL} is remote; cannot start it for eval." | tee -a "${EVAL_LOG}"
        return 1
    fi
    stop_inference_server || true
    local port_wait=0
    while [ "${port_wait}" -lt 30 ] && [ -n "$(port_listeners || true)" ]; do
        sleep 1
        port_wait=$((port_wait + 1))
    done
    _restore_runtime_tmp_snapshot

    local stub_marker="ERROR: start_server.sh has no engine configured"
    local launch_script
    if [ -f "${JOB_DIR}/task/start_server.sh" ] && \
       ! grep -q "${stub_marker}" "${JOB_DIR}/task/start_server.sh"; then
        launch_script="/home/agent/task/start_server.sh"
    elif [ -f "${JOB_DIR}/task/start_server_original.sh" ] && \
         ! grep -q "${stub_marker}" "${JOB_DIR}/task/start_server_original.sh"; then
        launch_script="/home/agent/task/start_server_original.sh"
        echo "[${EVALUATION_TASK}] start_server.sh is unconfigured; using start_server_original.sh" | tee -a "${EVAL_LOG}"
    else
        echo "[${EVALUATION_TASK}] no configured start_server.sh found for the eval server." | tee -a "${EVAL_LOG}"
        return 1
    fi

    local pid_file="${JOB_DIR}/task/server_${SERVER_PORT}.pid"
    rm -f "${pid_file}" 2>/dev/null || true
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) starting the eval server via ${launch_script}" | tee -a "${EVAL_LOG}"
    setsid apptainer exec \
        "${AGENT_CONTAINER[@]}" \
        "${SIF}" \
        bash -c "set -euo pipefail; if [ -f '/home/agent/task/final_runtime_env.sh' ]; then . '/home/agent/task/final_runtime_env.sh'; fi; exec '/opt/inference_eval/bin/launch_supervised_server.sh' '${launch_script}'" >> "${EVAL_DIR}/server.log" 2>&1 &
    local launcher_pid=$!
    remember_process_group "${launcher_pid}"
    echo "${launcher_pid}" > "${pid_file}"

    local deadline=$(( $(date +%s) + SERVER_WAIT_S ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if _server_models_ok; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) eval server is ready at ${INFERENCE_BENCH_SERVER_URL}" | tee -a "${EVAL_LOG}"
            return 0
        fi
        if ! kill -0 "${launcher_pid}" 2>/dev/null; then
            echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) eval server exited before becoming ready" | tee -a "${EVAL_LOG}"
            return 1
        fi
        sleep 1
    done
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) eval server did not become ready within ${SERVER_WAIT_S}s" | tee -a "${EVAL_LOG}"
    return 1
}

if [ -d "${JOB_DIR}/task/submission" ]; then
    cp -r "${JOB_DIR}/task/submission" "${EVAL_DIR}/submission"
fi
cp -r "${JOB_DIR}/task" "${EVAL_DIR}/task"

echo "Evaluating..."
capture_agent_runtime_for_final_eval
pre_eval_cleanup_barrier || exit 1

run_eval_attempt() {
    local extra_args=("$@")
    apptainer exec \
        "${GPU_CONTAINER[@]}" \
        "${QUALITY_ENV[@]}" \
        --env INFERENCE_BENCH_ALLOW_HF_DOWNLOAD="${INFERENCE_BENCH_ALLOW_HF_DOWNLOAD:-0}" \
        --env INFERENCE_BENCH_DATASET_SEED="${EVAL_SEED}" \
        --bind "${REPO_ROOT}:${REPO_ROOT}:ro" \
        --home "${TMP_SUBDIR}/eval_home:/home/agent" \
        --env PYTHONNOUSERSITE=1 \
        --bind "${EVAL_DIR}:${EVAL_DIR}" \
        "${DATA_BIND[@]}" \
        --pwd "${REPO_ROOT}" \
        "${SIF}" python3 "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" \
        --server-url "${INFERENCE_BENCH_SERVER_URL}" \
        --json-output-file "${EVAL_DIR}/metrics.json" \
        --quality-tau "${INFERENCE_BENCH_QUALITY_TAU}" \
        "${extra_args[@]}" >> "${EVAL_LOG}" 2>&1
}

valid_eval_metrics() {
    python3 - "${EVAL_DIR}/metrics.json" "${REPO_ROOT}" <<'PY_METRICS'
import json
import sys
sys.path.insert(0, sys.argv[2])
try:
    from src.eval.inference.quality_gate import speed_profiles_complete
    with open(sys.argv[1], encoding="utf-8") as stream:
        metrics = json.load(stream)
    quality = metrics.get("quality_check", {})
    valid = (not metrics.get("error") and speed_profiles_complete(metrics)
             and isinstance(quality, dict) and not quality.get("error")
             and isinstance(quality.get("datasets", {}), dict)
             and all(isinstance(dataset, dict) and dataset.get("complete") is not False
                     for dataset in quality.get("datasets", {}).values())
             and isinstance(quality.get("pass"), bool))
except (ImportError, OSError, ValueError, AttributeError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY_METRICS
}

run_eval_with_retry() {
    local max_retries="$1"; shift
    local extra_args=("$@")
    local attempt
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        if ! _server_models_ok; then
            if ! ensure_inference_server_for_eval; then
                FINAL_EVAL_RC=1
                continue
            fi
        fi
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) eval attempt ${attempt}/${max_retries} ${extra_args[*]}" | tee -a "${EVAL_LOG}"
        FINAL_EVAL_RC=0
        run_eval_attempt "${extra_args[@]}" || FINAL_EVAL_RC=$?
        if [ "${FINAL_EVAL_RC}" -eq 0 ] && valid_eval_metrics; then
            return 0
        fi
        [ "${FINAL_EVAL_RC}" -ne 0 ] || FINAL_EVAL_RC=1
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) attempt ${attempt} failed (rc=${FINAL_EVAL_RC})" | tee -a "${EVAL_LOG}"
    done
    return 1
}

FINAL_EVAL_RC=1
mkdir -p "${TMP_SUBDIR}/eval_home"
rm -f "${EVAL_DIR}/metrics.json"
if ensure_inference_server_for_eval; then
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) starting final evaluation (speed + quality)" | tee -a "${EVAL_LOG}"
    if ! run_eval_with_retry 3; then
        echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) phase 1 failed; retrying with reduced request-timeout" | tee -a "${EVAL_LOG}"
        run_eval_with_retry 2 --request-timeout-s 150 || true
    fi
else
    echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) final evaluation aborted: inference server unavailable" | tee -a "${EVAL_LOG}"
    python3 - "${EVAL_DIR}/metrics.json" "${EVALUATION_TASK}" "${BASE_MODEL}" "${INFERENCE_BENCH_SERVER_URL}" <<'PY_ERROR'
import json
import sys
from pathlib import Path
path, scenario, model, url = sys.argv[1:]
Path(path).write_text(json.dumps({
    "scenario": scenario, "model_id": model, "profiles": {}, "vram_peak_mb": 0.0,
    "error": f"inference server unavailable before final eval at {url}",
}, indent=2), encoding="utf-8")
PY_ERROR
fi
echo "[${EVALUATION_TASK}] $(date --iso-8601=seconds) final eval exit code: ${FINAL_EVAL_RC}" | tee -a "${EVAL_LOG}"

stop_inference_server || true

if [ "${FINAL_EVAL_RC}" -ne 0 ]; then
    exit "${FINAL_EVAL_RC}"
fi
exit "${JUDGE_RC}"
