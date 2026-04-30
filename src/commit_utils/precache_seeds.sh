#!/bin/bash
#
# Pre-cache dataset samples for all seed pairs before running experiments.
#
# Run this once from the repo root before launching experiments:
#   bash src/commit_utils/precache_seeds.sh
#
# This generates 12 JSONL files:
#   - src/eval/inference/baselines/samples/longbench_v2/{seed}_{N}/samples.jsonl (×6 seeds)
#   - src/eval/inference/baselines/samples/mmlu_pro/{seed}_{N}/samples.jsonl      (×6 seeds)
#
# The 6 seeds come from the 3 hardcoded seed pairs (agent_seed + eval_seed each).
# Matching the seed pairs in commit.sh and run_all_experiments.sh is critical for
# reproducibility — update all three files together if you change the seeds.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source src/commit_utils/set_env_vars.sh

# Seed pairs matching commit.sh / run_all_experiments.sh defaults.
# Format: "agent_seed:eval_seed"
SEED_PAIRS=(
    "21:1337"
    "248:428"
    "999:777"
)

# Sample counts per cache file.
MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}"
LONGBENCH_N="${INFERENCE_BENCH_LONGBENCH_N:-503}"

# Collect unique seeds from all pairs.
declare -A SEEN_SEEDS=()
for pair in "${SEED_PAIRS[@]}"; do
    IFS=':' read -r agent_seed eval_seed <<< "${pair}"
    SEEN_SEEDS["${agent_seed}"]=1
    SEEN_SEEDS["${eval_seed}"]=1
done

echo "============================================"
echo "  InferenceBench Pre-cache Seeds"
echo "============================================"
echo "  seed_pairs:   ${SEED_PAIRS[*]}"
echo "  unique seeds: ${!SEEN_SEEDS[*]}"
echo "  mmlupro_n:    ${MMLUPRO_N}"
echo "  longbench_n:  ${LONGBENCH_N}"
echo "============================================"

VENV_ACTIVATE="${INFERENCE_BENCH_VENV_HOST}/bin/activate"
if [ -f "${VENV_ACTIVATE}" ]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
    echo "  python:       $(which python3) (venv: ${INFERENCE_BENCH_VENV_HOST})"
else
    echo "WARNING: venv activate not found at ${VENV_ACTIVATE}; using system python3"
fi

for seed in "${!SEEN_SEEDS[@]}"; do
    echo ""
    echo "--- Caching seed=${seed} ---"
    python3 -m src.eval.inference.cache_samples \
        --seed "${seed}" \
        --mmlupro-n "${MMLUPRO_N}" \
        --longbench-n "${LONGBENCH_N}"
done

echo ""
echo "============================================"
echo "  Pre-caching complete."
echo "============================================"
