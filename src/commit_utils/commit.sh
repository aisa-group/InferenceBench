#!/bin/bash
set -euo pipefail
source src/commit_utils/set_env_vars.sh

BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
STARTING_POINT="default"
NUM_HOURS=2
SCENARIOS="inference_scenario_b_output_heavy"
AGENTS=""
SEED_PAIRS_ARG="${DEFAULT_SEED_PAIRS}"
BID=1000
EXPERIMENT_NAME=""
DRY_RUN=0
SUB_FILE="src/commit_utils/single_task.sub"
REQUIREMENTS=()

ALL_SCENARIOS=(
    "inference_scenario_a_input_heavy"
    "inference_scenario_b_output_heavy"
    "inference_scenario_c_high_load"
    "inference_scenario_d_general"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenarios)       shift; SCENARIOS="$1" ;;
        --agents)          shift; AGENTS="$1" ;;
        --num-hours)       shift; NUM_HOURS="$1" ;;
        --base-model)      shift; BASE_MODEL="$1" ;;
        --starting-point)  shift; STARTING_POINT="$1" ;;
        --experiment-name) shift; EXPERIMENT_NAME="$1" ;;
        --bid)             shift; BID="$1" ;;
        --seed-pairs)      shift; SEED_PAIRS_ARG="$1" ;;
        --a100)            use_a100 ;;
        --dry-run)         DRY_RUN=1 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ -z "${AGENTS}" ]; then
    echo "--agents is required, e.g. --agents codex:gpt-5.3-codex-high" >&2
    exit 1
fi

IFS=',' read -ra SCENARIOS_ARR <<< "${SCENARIOS}"
if [ "${SCENARIOS_ARR[0]}" = "all" ]; then
    SCENARIOS_ARR=("${ALL_SCENARIOS[@]}")
fi
IFS=',' read -ra AGENTS_ARR <<< "${AGENTS}"
IFS=',' read -ra SEED_PAIRS <<< "${SEED_PAIRS_ARG}"

for agent_spec in "${AGENTS_ARR[@]}"; do
    validate_agent_image "${agent_spec%%:*}" "${STARTING_POINT}"
done

baselines_dir="src/eval/inference/baselines"
model_safe="$(printf '%s' "${BASE_MODEL}" | sed -E 's/[^A-Za-z0-9_.-]+/_/g')"
backend="${INFERENCE_BENCH_QUALITY_BASELINE_BACKEND}"
registry="${INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY}"
if [ -z "${registry}" ]; then
    registry="${baselines_dir}/quality/${model_safe}_${backend}.json"
    [ "${backend}" != "vllm" ] || registry="${baselines_dir}/quality/${model_safe}.json"
    alt_registry="${baselines_dir}/results/${model_safe}/quality_${backend}.json"
    [ -f "${registry}" ] || [ ! -f "${alt_registry}" ] || registry="${alt_registry}"
fi
if [ ! -f "${registry}" ] || [ ! -d "${baselines_dir}/samples" ]; then
    echo "Missing quality reference for ${BASE_MODEL} (${backend}): ${registry}" >&2
    echo "Cache the benchmark samples and precompute the quality reference as described in README.md." >&2
    exit 1
fi

export INFERENCE_BENCH_EXPERIMENT_NAME="${EXPERIMENT_NAME}"

echo "model=${BASE_MODEL} starting_point=${STARTING_POINT} num_hours=${NUM_HOURS} bid=${BID}"
echo "scenarios=${SCENARIOS_ARR[*]}"
echo "agents=${AGENTS_ARR[*]}"
echo "seed_pairs=${SEED_PAIRS[*]}"

for seed_pair in "${SEED_PAIRS[@]}"; do
    IFS=':' read -r agent_seed eval_seed <<< "${seed_pair}"
    for scenario in "${SCENARIOS_ARR[@]}"; do
        for agent_spec in "${AGENTS_ARR[@]}"; do
            IFS=':' read -r agent agent_config <<< "${agent_spec}"
            args=(
                -a "agent=${agent}"
                -a "agent_config=${agent_config}"
                -a "eval=${scenario}"
                -a "base_model=${BASE_MODEL}"
                -a "num_hours=${NUM_HOURS}"
                -a "starting_point=${STARTING_POINT}"
                -a "agent_seed=${agent_seed}"
                -a "eval_seed=${eval_seed}"
                "${REQUIREMENTS[@]}"
            )
            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "[dry-run] condor_submit_bid ${BID} ${args[*]} ${SUB_FILE}"
            else
                condor_submit_bid "${BID}" "${args[@]}" "${SUB_FILE}"
            fi
        done
    done
done
