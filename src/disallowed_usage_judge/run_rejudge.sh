#!/bin/bash
# Run the disallowed-usage / contamination judge against a single already-completed
# experiment folder. Output files are suffixed with the judge model slug so that
# multiple judge models produce side-by-side reports under the same folder.
#
# Usage:
#   bash src/disallowed_usage_judge/run_rejudge.sh <EXP_DIR> [JUDGE_MODEL]
#
# Expects <EXP_DIR> to contain:
#   task/benchmark.txt   (human-readable benchmark name — used for --benchmark)
#   task/start_server.sh (evidence)
#   task/server.log      (evidence; optional)
#
# Writes:
#   <EXP_DIR>/contamination_judgement__<judge-slug>.txt
#   <EXP_DIR>/disallowed_model_judgement__<judge-slug>.txt
#   <EXP_DIR>/rejudge__<judge-slug>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/env.sh" ]; then
  source "${REPO_ROOT}/env.sh"
fi
source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

EXP_DIR="${1:-}"
JUDGE_MODEL="${2:-${INFERENCE_BENCH_REJUDGE_MODEL:-claude-sonnet-4-6}}"

if [ -z "${EXP_DIR}" ]; then
  echo "Usage: $0 <EXP_DIR> [JUDGE_MODEL]" >&2
  exit 2
fi
if [ ! -d "${EXP_DIR}/task" ]; then
  echo "[rejudge] ERROR: ${EXP_DIR}/task does not exist (expected task/ subdir with evidence)." >&2
  exit 2
fi

# Slugify judge model name for safe use in filenames.
JUDGE_SLUG="$(echo "${JUDGE_MODEL}" | tr '/:' '__' | tr -c '[:alnum:]._-' '_' | sed 's/__*/_/g')"

REJUDGE_LOG="${EXP_DIR}/rejudge__${JUDGE_SLUG}.log"
exec > >(tee -a "${REJUDGE_LOG}")
exec 2> >(tee -a "${REJUDGE_LOG}" >&2)

echo "[rejudge] exp_dir=${EXP_DIR}"
echo "[rejudge] judge_model=${JUDGE_MODEL} (slug=${JUDGE_SLUG})"
echo "[rejudge] log=${REJUDGE_LOG}"

# Extract benchmark string — prefer the evidence file, fall back to folder path parsing.
BENCHMARK=""
if [ -f "${EXP_DIR}/task/benchmark.txt" ]; then
  BENCHMARK="$(tr -d '\r\n' < "${EXP_DIR}/task/benchmark.txt")"
fi
if [ -z "${BENCHMARK}" ]; then
  scenario_id="$(basename "${EXP_DIR}" | sed -E 's/^(inference_scenario_[a-z]_[a-z_]+?)_.*/\1/')"
  if [ -f "src/eval/tasks/${scenario_id}/benchmark.txt" ]; then
    BENCHMARK="$(tr -d '\r\n' < "src/eval/tasks/${scenario_id}/benchmark.txt")"
  fi
fi
if [ -z "${BENCHMARK}" ]; then
  echo "[rejudge] ERROR: could not determine BENCHMARK for ${EXP_DIR}." >&2
  exit 3
fi
echo "[rejudge] benchmark=${BENCHMARK}"

# Extract base model — folder names look like:
#   inference_scenario_X_name_<base_model_slug>_<cluster_id>
# where <base_model_slug> is the HF model id with '/' replaced by '_'.
# Best effort: strip scenario prefix and trailing cluster id, then restore first '_' to '/'.
BASE_MODEL=""
folder_name="$(basename "${EXP_DIR}")"
stripped="$(echo "${folder_name}" | sed -E 's/^inference_scenario_[a-z]_[a-z_]+?_//; s/_[0-9]+$//')"
if [ -n "${stripped}" ] && [[ "${stripped}" == *_* ]]; then
  # First underscore → slash (HF-org/HF-model convention).
  BASE_MODEL="$(echo "${stripped}" | sed 's/_/\//')"
fi
if [ -z "${BASE_MODEL}" ]; then
  BASE_MODEL="${INFERENCE_BENCH_REJUDGE_BASE_MODEL:-unknown}"
fi
echo "[rejudge] base_model=${BASE_MODEL}"

# Stage a writable task dir we can bind into the container at /home/agent/task,
# so the judge's output file writes don't touch the original results.
JOB_TMP="$(mktemp -d -t ptb_rejudge.XXXXXXXX)"
STAGE_TASK="${JOB_TMP}/task"
mkdir -p "${STAGE_TASK}"

# Build the judge prompt. Write to a file under JOB_TMP so we can pipe it via
# stdin — long prompts (embedded server.log) can exceed ARG_MAX if passed via
# apptainer's argv.
python3 src/disallowed_usage_judge/get_judge_prompt.py \
  --benchmark "${BENCHMARK}" \
  --model "${BASE_MODEL}" \
  --task-dir "${EXP_DIR}/task" > "${JOB_TMP}/judge_prompt.txt"
# Only stage the lightweight evidence files the prompt already embedded; actual
# judgement output is what we care about.
for f in start_server.sh server.log benchmark.txt; do
  if [ -f "${EXP_DIR}/task/${f}" ]; then
    cp "${EXP_DIR}/task/${f}" "${STAGE_TASK}/${f}"
  fi
done

CONTAINER_SIF="${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif"
if [[ "${CONTAINER_SIF}" != /* ]]; then
  CONTAINER_SIF="${REPO_ROOT}/${CONTAINER_SIF}"
fi
if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "[rejudge] ERROR: judge container missing: ${CONTAINER_SIF}" >&2
  exit 4
fi

OAUTH_TOKEN=""
if [ -f "agents/claude_non_api/oauth_token" ]; then
  OAUTH_TOKEN="$(cat agents/claude_non_api/oauth_token)"
fi

APPTAINER_NV_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  APPTAINER_NV_FLAG=(--nv)
fi

echo "[rejudge] container=${CONTAINER_SIF}"
echo "[rejudge] running judge..."

set +e
apptainer exec \
  "${APPTAINER_NV_FLAG[@]}" \
  -c \
  --cleanenv \
  --env PATH="/root/.local/bin:/home/agent/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env ANTHROPIC_API_KEY="" \
  --env CLAUDE_CODE_OAUTH_TOKEN="${OAUTH_TOKEN}" \
  --bind "${JOB_TMP}:/tmp" \
  --home "${JOB_TMP}:/home/agent" \
  --pwd "/home/agent/task" \
  --writable-tmpfs \
  "${CONTAINER_SIF}" \
  sh -c "claude --print --dangerously-skip-permissions --model '${JUDGE_MODEL}' < /tmp/judge_prompt.txt"
JUDGE_RC=$?
set -e
echo "[rejudge] judge exit=${JUDGE_RC}"

# Pull the judgement files the model wrote into /home/agent/task, rename with
# the judge-model suffix, and drop them into the original experiment folder.
for f in contamination_judgement disallowed_model_judgement; do
  src="${STAGE_TASK}/${f}.txt"
  dst="${EXP_DIR}/${f}__${JUDGE_SLUG}.txt"
  if [ -f "${src}" ]; then
    cp "${src}" "${dst}"
    echo "[rejudge] wrote ${dst}"
  else
    echo "[rejudge] WARN: ${f}.txt was not produced by the judge."
  fi
done

rm -rf "${JOB_TMP}"

if [ "${JUDGE_RC}" -ne 0 ]; then
  exit "${JUDGE_RC}"
fi
echo "[rejudge] done."
