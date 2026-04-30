#!/bin/bash
#
# Submit InferenceBench experiments to HTCondor.
#
# Quick start: edit the defaults below, then run:
#   bash src/commit_utils/commit.sh
#
# Or override from the command line (see --help-style examples at bottom).
#

set -euo pipefail
source src/commit_utils/set_env_vars.sh

# ============================================================================
#  Edit these defaults, then run the script.
# ============================================================================

# ──── What to run ───────────────────────────────────────────────────────────
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"                  # HuggingFace model ID
STARTING_POINT="default"                     # default | vllm_running | bare
NUM_HOURS=2                                  # Time budget per job

# Scenarios: pick from the list or use "all".
#   inference_scenario_a_input_heavy
#   inference_scenario_b_output_heavy
#   inference_scenario_c_high_load
#   inference_scenario_d_general
SCENARIOS="inference_scenario_b_output_heavy"

  # Comma-separated, or "all"

# Agents: "agent_type:model_config" pairs, comma-separated.
# Leave empty to use scheduler defaults (see below).
AGENTS=""                                    # e.g. "claude_non_api:claude-opus-4-6,claude_non_api:claude-sonnet-4-5"

# Seed pairs: "agent_seed:eval_seed" triples, comma-separated.
# Each pair produces one job submission per (scenario, agent) combination.
# agent_seed: used by the agent during its optimization runs.
# eval_seed:  used for the final scored evaluation (different from agent_seed
#             so the agent cannot overfit to the exact evaluation queries).
# Leave empty to use the hardcoded defaults below.
SEED_PAIRS_ARG=""

# ──── Job settings ──────────────────────────────────────────────────────────
BID=1000                                     # HTCondor bid priority
EXPERIMENT_NAME=""                           # Experiment name suffix for results dir
DRY_RUN=0                                   # Set to 1 to print commands without submitting
# ────────────────────────────────────────────────────────────────────────────

# ============================================================================
#  Everything below here is plumbing — you shouldn't need to touch it.
# ============================================================================

ALL_SCENARIOS=(
    "inference_scenario_a_input_heavy"
    "inference_scenario_b_output_heavy"
    "inference_scenario_c_high_load"
    "inference_scenario_d_general"
)

# Command-line overrides (optional — the defaults above are used if omitted).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenarios)     shift; SCENARIOS="$1" ;;
        --agents)        shift; AGENTS="$1" ;;
        --num-hours)     shift; NUM_HOURS="$1" ;;
        --base-model)    shift; BASE_MODEL="$1" ;;
        --starting-point) shift; STARTING_POINT="$1" ;;
        --experiment-name) shift; EXPERIMENT_NAME="$1" ;;
        --bid)           shift; BID="$1" ;;
        --seed-pairs)    shift; SEED_PAIRS_ARG="$1" ;;
        --dry-run)       DRY_RUN=1 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

# Expand scenarios.
IFS=',' read -ra SCENARIOS_ARR <<< "${SCENARIOS}"
if [ "${SCENARIOS_ARR[0]}" = "all" ]; then
    SCENARIOS_ARR=("${ALL_SCENARIOS[@]}")
fi

# Default agents per scheduler if none specified.
AGENTS_ARR=()
if [ -n "${AGENTS}" ]; then
    IFS=',' read -ra AGENTS_ARR <<< "${AGENTS}"
else
    if [ "${INFERENCE_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
        AGENTS_ARR=(
            "codex_non_api:gpt-5.2-codex"
            "codex_non_api:gpt-5.2"
            "claude_non_api:claude-sonnet-4-5"
            "claude_non_api:claude-opus-4-5"
        )
    elif [ "${INFERENCE_BENCH_JOB_SCHEDULER}" = "htcondor" ]; then
        AGENTS_ARR=(
            "codex_non_api:gpt-5.2-codex"
            "codex_non_api:gpt-5.2"
            # "claude_non_api:claude-sonnet-4-5"
            # "claude_non_api:claude-opus-4-5"
        )
    else
        echo "ERROR: job scheduler '${INFERENCE_BENCH_JOB_SCHEDULER}' is not supported."
        exit 1
    fi
fi

# Seed pairs: each entry is "agent_seed:eval_seed".
# These 3 pairs produce 3 independent runs per (scenario, agent) with different
# dataset samples for agent preview evals vs the final scored evaluation.
SEED_PAIRS=()
if [ -n "${SEED_PAIRS_ARG}" ]; then
    IFS=',' read -ra SEED_PAIRS <<< "${SEED_PAIRS_ARG}"
else
    SEED_PAIRS=(
        "21:1337"
        "248:428"
        "999:777"
    )
fi

export INFERENCE_BENCH_EXPERIMENT_NAME="${EXPERIMENT_NAME}"

# Print summary.
echo "============================================"
echo "  InferenceBench Experiment Submission"
echo "============================================"
echo "  model:          ${BASE_MODEL}"
echo "  starting_point: ${STARTING_POINT}"
echo "  num_hours:      ${NUM_HOURS}"
echo "  scenarios:      ${SCENARIOS_ARR[*]}"
echo "  agents:         ${AGENTS_ARR[*]}"
echo "  seed_pairs:     ${SEED_PAIRS[*]}"
echo "  bid:            ${BID}"
echo "  experiment:     ${EXPERIMENT_NAME:-<none>}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  *** DRY RUN ***"
fi
echo "============================================"

submit_job() {
    local scenario="$1"
    local agent="$2"
    local agent_config="$3"
    local agent_seed="$4"
    local eval_seed="$5"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[dry-run] condor_submit_bid ${BID} ... agent=${agent} agent_config=${agent_config} eval=${scenario} base_model=${BASE_MODEL} num_hours=${NUM_HOURS} starting_point=${STARTING_POINT} agent_seed=${agent_seed} eval_seed=${eval_seed}"
    else
        condor_submit_bid "${BID}" \
            -a "agent=${agent}" \
            -a "agent_config=${agent_config}" \
            -a "eval=${scenario}" \
            -a "base_model=${BASE_MODEL}" \
            -a "num_hours=${NUM_HOURS}" \
            -a "starting_point=${STARTING_POINT}" \
            -a "agent_seed=${agent_seed}" \
            -a "eval_seed=${eval_seed}" \
            "src/commit_utils/single_task.sub"
    fi
}

for seed_pair in "${SEED_PAIRS[@]}"; do
    IFS=':' read -r agent_seed eval_seed <<< "${seed_pair}"
    echo ""
    echo "=== seed pair: agent_seed=${agent_seed} eval_seed=${eval_seed} ==="
    for scenario in "${SCENARIOS_ARR[@]}"; do
        echo ""
        echo "${BASE_MODEL} on ${scenario} (${NUM_HOURS}h, sp=${STARTING_POINT}, seeds=${agent_seed}:${eval_seed})"
        for agent_spec in "${AGENTS_ARR[@]}"; do
            IFS=':' read -r agent agent_config <<< "${agent_spec}"
            submit_job "${scenario}" "${agent}" "${agent_config}" "${agent_seed}" "${eval_seed}"
        done
    done
done
