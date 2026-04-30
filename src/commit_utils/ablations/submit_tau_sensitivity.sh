#!/bin/bash
set -euo pipefail

# ============================================================================
#  Quality-Gate Threshold Sensitivity — Condor Submission
# ============================================================================
#  Re-scores every 2h run under $INFERENCE_BENCH_RESULTS_DIR at several
#  quality-gate thresholds (tau in {0.90, 0.93, 0.95, 0.97}) and emits a
#  LaTeX snippet for Appendix I.5 plus a JSON dump for deeper analysis.
#  CPU-only, no GPU, runs in well under a minute. No apptainer.
#
#  Usage:
#    bash src/commit_utils/ablations/submit_tau_sensitivity.sh [BID] [OUT_DIR]
# ============================================================================

BID="${1:-100}"
OUT_DIR="${2:-${PWD}/tau_sensitivity_out}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

export TAU_OUT_DIR="${OUT_DIR}"
export INFERENCE_BENCH_RESULTS_DIR="${INFERENCE_BENCH_RESULTS_DIR:-/fast/${USER}/ptb_results}"

echo "Submitting tau-sensitivity analysis job:"
echo "  bid=${BID}"
echo "  results_dir=${INFERENCE_BENCH_RESULTS_DIR}"
echo "  out_dir=${OUT_DIR}"

condor_submit_bid "${BID}" \
  -a "initialdir=${REPO_ROOT}" \
  "${REPO_ROOT}/src/commit_utils/ablations/tau_sensitivity.sub"
