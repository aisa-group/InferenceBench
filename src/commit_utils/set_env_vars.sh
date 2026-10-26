for _var in $(compgen -e); do
    [ "${!_var}" != "UNDEFINED" ] || unset "${_var}"
done
unset _var

set_default() {
    [ -n "${!1:-}" ] || export "$1=$2"
}

set_default HF_HOME "${HOME}/.cache/huggingface"
set_default INFERENCE_BENCH_RESULTS_DIR "results"
set_default INFERENCE_BENCH_CONTAINERS_DIR "containers"
set_default INFERENCE_BENCH_CONTAINER_NAME "inference"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_VLLM_NAME "inference_baselines_vllm"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_TORCH_NAME "inference_baselines_torch"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_SGLANG_NAME "inference_baselines_sglang"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_TGI_NAME "inference_baselines_tgi"
set_default INFERENCE_BENCH_VENV_VLLM_HOST "${HOME}/.venvs/inferencebench_vllm"
set_default INFERENCE_BENCH_VENV_TORCH_HOST "${HOME}/.venvs/inferencebench_torch"
set_default INFERENCE_BENCH_VENV_SGLANG_HOST "${HOME}/.venvs/inferencebench_sglang"
set_default INFERENCE_BENCH_VENV_TGI_HOST "${HOME}/.venvs/inferencebench_tgi"
set_default INFERENCE_BENCH_EXPERIMENT_NAME ""
set_default INFERENCE_BENCH_BASE_MODEL "mistralai/Mistral-7B-Instruct-v0.3"
set_default INFERENCE_BENCH_DATASET_SEED "248"
set_default INFERENCE_BENCH_QUALITY_TAU "0.95"
set_default INFERENCE_BENCH_QUALITY_SEED "${INFERENCE_BENCH_DATASET_SEED}"
set_default INFERENCE_BENCH_QUALITY_MMLUPRO_N "500"
set_default INFERENCE_BENCH_LONGBENCH_N "503"
set_default INFERENCE_BENCH_QUALITY_CONCURRENCY "4"
set_default INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY ""
set_default INFERENCE_BENCH_QUALITY_BASELINE_BACKEND "torch"
set_default INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE ""
set_default INFERENCE_BENCH_MAX_MODEL_LEN "32768"
set_default INFERENCE_BENCH_INPUT_TOKEN_MARGIN "16"
set_default INFERENCE_BENCH_SKIP_SPEED "0"
set_default INFERENCE_BENCH_SKIP_QUALITY "0"
set_default INFERENCE_BENCH_DOWNLOAD_ONLY "0"

DEFAULT_SEED_PAIRS="21:1337,248:428,999:777"

use_a100() {
    export INFERENCE_BENCH_CONTAINER_NAME="inference_a100"
    export INFERENCE_BENCH_BASELINE_CONTAINER_VLLM_NAME="inference_baselines_vllm_a100"
    export INFERENCE_BENCH_BASELINE_CONTAINER_SGLANG_NAME="inference_baselines_sglang_a100"
    export INFERENCE_BENCH_BASELINE_CONTAINER_TORCH_NAME="inference_baselines_torch_a100"
    REQUIREMENTS=(-a 'requirements=TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB"')
}

validate_agent_image() {
    local agent="$1"
    local starting_point="$2"
    local image="${INFERENCE_BENCH_CONTAINER_NAME}"
    if [ "${starting_point}" = "vllm_running" ]; then
        image="${INFERENCE_BENCH_CONTAINER_NAME_VLLM_RUNNING:-${INFERENCE_BENCH_BASELINE_CONTAINER_VLLM_NAME}}"
    fi
    case "${agent}:${image}" in
        opencode:inference_a100|opencode:inference_baselines_vllm|opencode:inference_baselines_vllm_a100)
            echo "OpenCode is not installed in ${image}." >&2
            return 2
            ;;
    esac
}

if [ -z "${HF_TOKEN:-}" ]; then
    for _token_file in "${HF_HOME}/token" "${HOME}/.cache/huggingface/token" "${HOME}/.huggingface/token"; do
        HF_TOKEN="$(tr -d '[:space:]' 2>/dev/null < "${_token_file}" || true)"
        if [ -n "${HF_TOKEN}" ]; then
            export HF_TOKEN
            break
        fi
    done
    unset _token_file
fi

export VLLM_API_KEY=""
