# Helper function: sets variable to default if unset or "UNDEFINED"
set_default() {
    local var_name="${1:-}"
    local default_value="${2:-}"
    local current_value
    eval "current_value=\"\${$var_name:-}\""

    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        export "$var_name"="$default_value"
    fi
}

# Prefer a shared fast container directory when available.
DEFAULT_CONTAINERS_DIR="containers"
if [ -n "${USER:-}" ] && [ -d "/fast/${USER}" ]; then
    if mkdir -p "/fast/${USER}/ptb_containers" 2>/dev/null; then
        DEFAULT_CONTAINERS_DIR="/fast/${USER}/ptb_containers"
    fi
fi

# Use shared HF cache by default (important for HTCondor jobs, which may have small local disks).
set_default HF_HOME "${HOME}/.cache/huggingface"

# Use fast storage for results when available; fall back to local relative path.
DEFAULT_RESULTS_DIR="results"
if [ -n "${USER:-}" ] && [ -d "/fast/${USER}" ]; then
    if mkdir -p "/fast/${USER}/ptb_results" 2>/dev/null; then
        DEFAULT_RESULTS_DIR="/fast/${USER}/ptb_results"
    fi
fi
set_default INFERENCE_BENCH_RESULTS_DIR "${DEFAULT_RESULTS_DIR}"
set_default INFERENCE_BENCH_CONTAINERS_DIR "${DEFAULT_CONTAINERS_DIR}"
set_default INFERENCE_BENCH_CONTAINER_NAME "inference"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_NAME "inference_baselines"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_VLLM_NAME "inference_baselines_vllm"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_TORCH_NAME "inference_baselines_torch"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_SGLANG_NAME "inference_baselines_sglang"
set_default INFERENCE_BENCH_BASELINE_CONTAINER_TGI_NAME "inference_baselines_tgi"
set_default INFERENCE_BENCH_VENV_HOST "${HOME}/.venvs/ptb_baselines"
set_default INFERENCE_BENCH_VENV_VLLM_HOST "${HOME}/.venvs/ptb_vllm"
set_default INFERENCE_BENCH_VENV_TORCH_HOST "${HOME}/.venvs/ptb_torch"
set_default INFERENCE_BENCH_VENV_SGLANG_HOST "${HOME}/.venvs/ptb_sglang"
set_default INFERENCE_BENCH_VENV_TGI_HOST "${HOME}/.venvs/ptb_tgi"
set_default INFERENCE_BENCH_USE_SPLIT_BACKEND_ENVS "1"
set_default INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS "0"
set_default INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS_VLLM "0"
set_default INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS_TORCH "1"
set_default INFERENCE_BENCH_PROMPT "prompt"
set_default INFERENCE_BENCH_JOB_SCHEDULER "htcondor"
set_default INFERENCE_BENCH_EXPERIMENT_NAME ""
set_default INFERENCE_BENCH_STARTING_POINT "default"
set_default INFERENCE_BENCH_BASE_MODEL "mistralai/Mistral-7B-Instruct-v0.3"
set_default INFERENCE_BENCH_DATASET_SEED "248"
set_default INFERENCE_BENCH_QUALITY_TAU "0.95"
set_default INFERENCE_BENCH_QUALITY_SEED "${INFERENCE_BENCH_DATASET_SEED}"
set_default INFERENCE_BENCH_QUALITY_MMLUPRO_N "500"
set_default INFERENCE_BENCH_LONGBENCH_N "503"
set_default INFERENCE_BENCH_QUALITY_CONCURRENCY "4"
set_default INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY ""
set_default INFERENCE_BENCH_MAX_MODEL_LEN "32768"
set_default INFERENCE_BENCH_INPUT_TOKEN_MARGIN "16"
set_default INFERENCE_BENCH_DISABLE_HF_OVERLAY "1"
set_default INFERENCE_BENCH_LOAD_CUDA_MODULE "1"
set_default INFERENCE_BENCH_CUDA_MODULE_SOURCE "/etc/profile.d/modules.sh"
set_default INFERENCE_BENCH_CUDA_MODULE_CANDIDATES "cuda/12.8 cuda/12.7 cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.3 cuda/12.2 cuda/12.1 cuda"

# Auto-detect HF_TOKEN from standard huggingface-cli token locations if not already set.
if [ -z "${HF_TOKEN:-}" ]; then
    for _token_file in \
        "${HF_HOME:-$HOME/.cache/huggingface}/token" \
        "$HOME/.cache/huggingface/token" \
        "$HOME/.huggingface/token"; do
        if [ -f "${_token_file}" ]; then
            HF_TOKEN="$(cat "${_token_file}" 2>/dev/null | tr -d '[:space:]')"
            if [ -n "${HF_TOKEN}" ]; then
                export HF_TOKEN
                break
            fi
        fi
    done
    unset _token_file
fi

export VLLM_API_KEY=""

if [ "${INFERENCE_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi
