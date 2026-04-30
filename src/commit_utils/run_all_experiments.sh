#!/bin/bash
#
# Orchestration script for all InferenceBench paper experiments.
# Submits experiment groups to HTCondor, with crash recovery (skips completed runs).
#
# Usage:
#   bash src/commit_utils/run_all_experiments.sh --group <group> [OPTIONS]
#
# Groups:
#   main              All agents x all scenarios (2h, 8B, default starting point)
#   time              Time budget ablation: [1,2,5,10] hours
#   model             Base model ablation: [gemma3-4b, llama-8b, qwen25-14b, qwen25-32b]
#   starting_point    Starting point ablation: [default, vllm_running, bare]
#   all               Run all groups above
#
# Options:
#   --agents <a:c,...>   Override default agent list (format: agent:config,agent:config,...)
#   --dry-run            Print commands without executing
#   --fresh              Resubmit all jobs, ignoring existing results
#
# Examples:
#   bash src/commit_utils/run_all_experiments.sh --group main --dry-run
#   bash src/commit_utils/run_all_experiments.sh --group time --fresh
#   bash src/commit_utils/run_all_experiments.sh --group all --agents claude_non_api:claude-opus-4-6
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source src/commit_utils/set_env_vars.sh
source src/commit_utils/model_registry.sh

# ── Defaults ──────────────────────────────────────────────────────────────────

GROUP=""
DRY_RUN=0
FRESH=0
OVERRIDE_AGENTS=()
OVERRIDE_SEED_PAIRS=()

ALL_SCENARIOS=(
    "inference_scenario_a_input_heavy"
    "inference_scenario_b_output_heavy"
    "inference_scenario_c_high_load"
    "inference_scenario_d_general"
)

# Default agents for main experiments (all agents)
MAIN_AGENTS=(
    "claude_non_api:claude-sonnet-4-6"
    "opencode:glm-5"
    "opencode:gemini-3.1-pro"
    "codex:gpt-5.3-codex-high"
    "codex:gpt-5.4-high"
    "codex:gpt-5.3-codex-med"
    "codex:gpt-5.5-high"
    "claude_non_api:claude-opus-4-6"
    "codex:gpt-5.2"
    "codex_non_api:gpt-5.1-codex"
    "claude_non_api:claude-opus-4-5"
    "claude_non_api:claude-sonnet-4-5"
    "claude_non_api:claude-opus-4-7"
    "codex:gpt-5.2-codex"
)

# Default agents for ablation experiments (representative subset)
ABLATION_AGENTS=(
    "claude_non_api:claude-opus-4-6"
    "codex:gpt-5.2-codex"
)

# Seed pairs: "agent_seed:eval_seed" — 3 independent runs per (scenario, agent).
# agent_seed: used during the agent's optimization preview evaluations.
# eval_seed:  used for the final scored evaluation, ensuring the agent cannot
#             overfit to the exact queries it was evaluated on during training.
DEFAULT_SEED_PAIRS=(
    "21:1337"
    "248:428"
    "999:777"
)

# ── Parse flags ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)
            shift; GROUP="$1"
            ;;
        --agents)
            shift; IFS=',' read -ra OVERRIDE_AGENTS <<< "$1"
            ;;
        --seed-pairs)
            shift; IFS=',' read -ra OVERRIDE_SEED_PAIRS <<< "$1"
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --fresh)
            FRESH=1
            ;;
        *)
            echo "Unknown flag: $1"; exit 1
            ;;
    esac
    shift
done

if [ -z "${GROUP}" ]; then
    echo "ERROR: --group is required. Options: main, time, model, starting_point, all"
    exit 1
fi

# ── Counters ──────────────────────────────────────────────────────────────────

TOTAL_SKIPPED=0
TOTAL_RESUBMIT=0
TOTAL_NEW=0

# ── Helper: check if a run is already completed ──────────────────────────────

# Checks if a result directory exists for the given parameters.
# Returns:
#   0 = completed (metrics.json exists) → skip
#   1 = crashed (dir exists, no metrics.json) → resubmit
#   2 = not found → new submission
check_existing_run() {
    local results_dir="$1"
    local agent="$2"
    local agent_config="$3"
    local num_hours="$4"
    local scenario="$5"
    local base_model="$6"
    local starting_point="$7"
    local experiment_name="$8"

    local agent_config_safe
    agent_config_safe="$(printf '%s' "${agent_config}" | tr '/:' '_')"
    local model_safe
    model_safe="$(printf '%s' "${base_model}" | tr '/:' '_')"

    local sp_suffix=""
    if [ "${starting_point}" != "default" ]; then
        sp_suffix="_sp-${starting_point}"
    fi

    # Pattern: <results_dir>/<agent>_<config>_<hours>h<exp>/<scenario>_<model><sp>_<cluster_id>/
    local dir_pattern="${results_dir}/${agent}_${agent_config_safe}_${num_hours}h${experiment_name}/${scenario}_${model_safe}${sp_suffix}_*"

    # Glob for matching directories
    local found_dirs
    found_dirs=( $(compgen -G "${dir_pattern}" 2>/dev/null) ) || found_dirs=()

    if [ "${#found_dirs[@]}" -eq 0 ]; then
        return 2  # not found
    fi

    # Check if any match has metrics.json (completed)
    for d in "${found_dirs[@]}"; do
        if [ -f "${d}/metrics.json" ]; then
            return 0  # completed
        fi
    done

    return 1  # crashed (dir exists but no metrics.json)
}

# ── Helper: submit a single job ──────────────────────────────────────────────

submit_job() {
    local results_dir="$1"
    local scenario="$2"
    local agent="$3"
    local agent_config="$4"
    local base_model="$5"
    local num_hours="$6"
    local starting_point="$7"
    local experiment_name="$8"
    local agent_seed="${9:-248}"
    local eval_seed="${10:-248}"

    # Check for existing results (unless --fresh)
    if [ "${FRESH}" -eq 0 ]; then
        set +e
        check_existing_run "${results_dir}" "${agent}" "${agent_config}" \
            "${num_hours}" "${scenario}" "${base_model}" "${starting_point}" "${experiment_name}"
        local status=$?
        set -e

        if [ "${status}" -eq 0 ]; then
            echo "  [skip] ${agent}:${agent_config} on ${scenario} seeds=${agent_seed}:${eval_seed} (completed)"
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
            return 0
        elif [ "${status}" -eq 1 ]; then
            echo "  [resubmit] ${agent}:${agent_config} on ${scenario} seeds=${agent_seed}:${eval_seed} (crashed, no metrics.json)"
            TOTAL_RESUBMIT=$((TOTAL_RESUBMIT + 1))
        fi
    fi

    if [ "${FRESH}" -eq 1 ] || [ "${status:-2}" -eq 2 ]; then
        TOTAL_NEW=$((TOTAL_NEW + 1))
    fi

    export INFERENCE_BENCH_RESULTS_DIR="${results_dir}"
    export INFERENCE_BENCH_EXPERIMENT_NAME="${experiment_name}"

    local cmd="condor_submit_bid 100 -a \"agent=${agent}\" -a \"agent_config=${agent_config}\" -a \"eval=${scenario}\" -a \"base_model=${base_model}\" -a \"num_hours=${num_hours}\" -a \"starting_point=${starting_point}\" -a \"agent_seed=${agent_seed}\" -a \"eval_seed=${eval_seed}\" \"src/commit_utils/single_task.sub\""

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  [submit] ${cmd}"
    else
        condor_submit_bid 100 \
            -a "agent=${agent}" \
            -a "agent_config=${agent_config}" \
            -a "eval=${scenario}" \
            -a "base_model=${base_model}" \
            -a "num_hours=${num_hours}" \
            -a "starting_point=${starting_point}" \
            -a "agent_seed=${agent_seed}" \
            -a "eval_seed=${eval_seed}" \
            "src/commit_utils/single_task.sub"
    fi
}

# ── Experiment group: main ────────────────────────────────────────────────────

run_group_main() {
    local results_dir="${INFERENCE_BENCH_RESULTS_DIR}/main"
    local agents=("${OVERRIDE_AGENTS[@]:-${MAIN_AGENTS[@]}}")
    local seed_pairs=("${OVERRIDE_SEED_PAIRS[@]:-${DEFAULT_SEED_PAIRS[@]}}")
    local base_model="${MODEL_IDS[mistral-7b]}"

    echo "============================================"
    echo "=== GROUP: main (all agents x all scenarios x ${#seed_pairs[@]} seed pairs)"
    echo "=== Results dir: ${results_dir}"
    echo "============================================"

    for seed_pair in "${seed_pairs[@]}"; do
        IFS=':' read -r agent_seed eval_seed <<< "${seed_pair}"
        echo ""
        echo "=== seed pair: agent_seed=${agent_seed} eval_seed=${eval_seed} ==="
        for scenario in "${ALL_SCENARIOS[@]}"; do
            echo "--- ${scenario} ---"
            for agent_spec in "${agents[@]}"; do
                IFS=':' read -r agent agent_config <<< "${agent_spec}"
                submit_job "${results_dir}" "${scenario}" "${agent}" "${agent_config}" \
                    "${base_model}" "2" "default" "" "${agent_seed}" "${eval_seed}"
            done
        done
    done
}

# ── Experiment group: time ablation ───────────────────────────────────────────

run_group_time() {
    local results_dir="${INFERENCE_BENCH_RESULTS_DIR}/time_ablation"
    local agents=("${OVERRIDE_AGENTS[@]:-${ABLATION_AGENTS[@]}}")
    local seed_pairs=("${OVERRIDE_SEED_PAIRS[@]:-${DEFAULT_SEED_PAIRS[@]}}")
    local base_model="${MODEL_IDS[qwen3-8b]}"
    local time_budgets=(1 2 4 8)

    echo "============================================"
    echo "=== GROUP: time ablation [${time_budgets[*]}] hours"
    echo "=== Results dir: ${results_dir}"
    echo "============================================"

    for hours in "${time_budgets[@]}"; do
        echo ""
        echo "=== Time budget: ${hours}h ==="
        for seed_pair in "${seed_pairs[@]}"; do
            IFS=':' read -r agent_seed eval_seed <<< "${seed_pair}"
            for scenario in "${ALL_SCENARIOS[@]}"; do
                echo "--- ${scenario} seeds=${agent_seed}:${eval_seed} ---"
                for agent_spec in "${agents[@]}"; do
                    IFS=':' read -r agent agent_config <<< "${agent_spec}"
                    submit_job "${results_dir}" "${scenario}" "${agent}" "${agent_config}" \
                        "${base_model}" "${hours}" "default" "" "${agent_seed}" "${eval_seed}"
                done
            done
        done
    done
}

# ── Experiment group: model ablation ──────────────────────────────────────────

run_group_model() {
    local results_dir="${INFERENCE_BENCH_RESULTS_DIR}/model_ablation"
    local agents=("${OVERRIDE_AGENTS[@]:-${ABLATION_AGENTS[@]}}")
    local seed_pairs=("${OVERRIDE_SEED_PAIRS[@]:-${DEFAULT_SEED_PAIRS[@]}}")

    echo "============================================"
    echo "=== GROUP: model ablation [${MODEL_KEYS[*]}]"
    echo "=== Results dir: ${results_dir}"
    echo "============================================"

    for model_key in "${MODEL_KEYS[@]}"; do
        local base_model="${MODEL_IDS[${model_key}]}"
        local max_model_len="${MODEL_MAX_LEN[${model_key}]}"

        echo ""
        echo "=== Model: ${model_key} (${base_model}, max_len=${max_model_len}) ==="

        # Set max_model_len for this model (propagates via HTCondor env)
        export INFERENCE_BENCH_MAX_MODEL_LEN="${max_model_len}"

        for seed_pair in "${seed_pairs[@]}"; do
            IFS=':' read -r agent_seed eval_seed <<< "${seed_pair}"
            for scenario in "${ALL_SCENARIOS[@]}"; do
                echo "--- ${scenario} seeds=${agent_seed}:${eval_seed} ---"
                for agent_spec in "${agents[@]}"; do
                    IFS=':' read -r agent agent_config <<< "${agent_spec}"
                    submit_job "${results_dir}" "${scenario}" "${agent}" "${agent_config}" \
                        "${base_model}" "2" "default" "" "${agent_seed}" "${eval_seed}"
                done
            done
        done
    done
}

# ── Experiment group: starting point ablation ─────────────────────────────────

run_group_starting_point() {
    local results_dir="${INFERENCE_BENCH_RESULTS_DIR}/sp_ablation"
    local agents=("${OVERRIDE_AGENTS[@]:-${ABLATION_AGENTS[@]}}")
    local seed_pairs=("${OVERRIDE_SEED_PAIRS[@]:-${DEFAULT_SEED_PAIRS[@]}}")
    local base_model="${MODEL_IDS[qwen3-8b]}"
    local starting_points=("default" "vllm_running" "bare")

    echo "============================================"
    echo "=== GROUP: starting_point ablation [${starting_points[*]}]"
    echo "=== Results dir: ${results_dir}"
    echo "============================================"

    for sp in "${starting_points[@]}"; do
        echo ""
        echo "=== Starting point: ${sp} ==="
        for seed_pair in "${seed_pairs[@]}"; do
            IFS=':' read -r agent_seed eval_seed <<< "${seed_pair}"
            for scenario in "${ALL_SCENARIOS[@]}"; do
                echo "--- ${scenario} seeds=${agent_seed}:${eval_seed} ---"
                for agent_spec in "${agents[@]}"; do
                    IFS=':' read -r agent agent_config <<< "${agent_spec}"
                    submit_job "${results_dir}" "${scenario}" "${agent}" "${agent_config}" \
                        "${base_model}" "2" "${sp}" "" "${agent_seed}" "${eval_seed}"
                done
            done
        done
    done
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "${GROUP}" in
    main)
        run_group_main
        ;;
    time)
        run_group_time
        ;;
    model)
        run_group_model
        ;;
    starting_point)
        run_group_starting_point
        ;;
    all)
        run_group_main
        echo ""
        run_group_time
        echo ""
        run_group_model
        echo ""
        run_group_starting_point
        ;;
    *)
        echo "ERROR: unknown group '${GROUP}'. Options: main, time, model, starting_point, all"
        exit 1
        ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "=== SUMMARY"
echo "============================================"
echo "  Skipped (completed):  ${TOTAL_SKIPPED}"
echo "  Resubmitted (crashed): ${TOTAL_RESUBMIT}"
echo "  New submissions:       ${TOTAL_NEW}"
echo "  Total actions:         $((TOTAL_SKIPPED + TOTAL_RESUBMIT + TOTAL_NEW))"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  (dry-run mode — no jobs were actually submitted)"
fi
echo "============================================"
