#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source src/commit_utils/set_env_vars.sh

for seed in $(tr ',:' '\n\n' <<< "${DEFAULT_SEED_PAIRS}" | sort -un); do
    echo "Caching seed=${seed} mmlupro_n=${INFERENCE_BENCH_QUALITY_MMLUPRO_N} longbench_n=${INFERENCE_BENCH_LONGBENCH_N}"
    python3 -m src.eval.inference.cache_samples \
        --seed "${seed}" \
        --mmlupro-n "${INFERENCE_BENCH_QUALITY_MMLUPRO_N}" \
        --longbench-n "${INFERENCE_BENCH_LONGBENCH_N}"
done
