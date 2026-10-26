#!/bin/bash
set -euo pipefail
umask 077

EXP_DIR="${1:-}"
if [ -n "${EXP_DIR}" ]; then
  EXP_DIR="$(realpath "${EXP_DIR}")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"
source "${SCRIPT_DIR}/judge.sh"

JUDGE_MODEL="${2:-${INFERENCE_BENCH_REJUDGE_MODEL:-${DEFAULT_JUDGE_MODEL}}}"

if [ -z "${EXP_DIR}" ]; then
  echo "Usage: $0 <EXP_DIR> [JUDGE_MODEL]" >&2
  exit 2
fi
if [ ! -d "${EXP_DIR}/task" ]; then
  echo "[rejudge] ERROR: ${EXP_DIR}/task does not exist." >&2
  exit 2
fi

JUDGE_SLUG="$(printf '%s' "${JUDGE_MODEL}" | tr '/:' '__' | tr -c '[:alnum:]._-' '_' | sed 's/__*/_/g')"

REJUDGE_LOG="${EXP_DIR}/rejudge__${JUDGE_SLUG}.log"
exec > >(tee -a "${REJUDGE_LOG}")
exec 2> >(tee -a "${REJUDGE_LOG}" >&2)

echo "[rejudge] exp_dir=${EXP_DIR}"
echo "[rejudge] judge_model=${JUDGE_MODEL}"

BENCHMARK=""
if [ -f "${EXP_DIR}/task/benchmark.txt" ]; then
  BENCHMARK="$(tr -d '\r\n' < "${EXP_DIR}/task/benchmark.txt")"
fi
if [ -z "${BENCHMARK}" ]; then
  for task_dir in src/eval/tasks/inference_scenario_*; do
    case "$(basename "${EXP_DIR}")" in
      "$(basename "${task_dir}")"_*) BENCHMARK="$(tr -d '\r\n' < "${task_dir}/benchmark.txt")" ;;
    esac
  done
fi
if [ -z "${BENCHMARK}" ]; then
  echo "[rejudge] ERROR: could not determine BENCHMARK for ${EXP_DIR}." >&2
  exit 3
fi
echo "[rejudge] benchmark=${BENCHMARK}"

BASE_MODEL="${INFERENCE_BENCH_REJUDGE_BASE_MODEL:-}"
if [ -z "${BASE_MODEL}" ] && [ -f "${EXP_DIR}/run_config.json" ]; then
  BASE_MODEL="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["base_model"])' "${EXP_DIR}/run_config.json")"
fi
if [ -z "${BASE_MODEL}" ]; then
  echo "[rejudge] ERROR: run_config.json is missing; set INFERENCE_BENCH_REJUDGE_BASE_MODEL to the original checkpoint." >&2
  exit 3
fi
echo "[rejudge] base_model=${BASE_MODEL}"

JOB_TMP="$(mktemp -d -t inferencebench_rejudge.XXXXXXXX)"
trap 'rm -rf "${JOB_TMP}"' EXIT
JUDGE_HOME="${JOB_TMP}/home"
JUDGE_TMP="${JOB_TMP}/tmp"
mkdir -p "${JUDGE_HOME}" "${JUDGE_TMP}"

python3 src/disallowed_usage_judge/get_judge_prompt.py \
  --benchmark "${BENCHMARK}" \
  --model "${BASE_MODEL}" \
  --task-dir "${EXP_DIR}/task" > "${JUDGE_TMP}/judge_prompt.txt"

CONTAINER_SIF="${INFERENCE_BENCH_CONTAINERS_DIR}/${INFERENCE_BENCH_CONTAINER_NAME}.sif"
if [[ "${CONTAINER_SIF}" != /* ]]; then
  CONTAINER_SIF="${REPO_ROOT}/${CONTAINER_SIF}"
fi
if [ ! -f "${CONTAINER_SIF}" ]; then
  echo "[rejudge] ERROR: judge container missing: ${CONTAINER_SIF}" >&2
  exit 4
fi

APPTAINER_NV_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  APPTAINER_NV_FLAG=(--nv)
fi

judge_credentials
if [ -z "${APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN}${APPTAINERENV_ANTHROPIC_API_KEY}" ]; then
  echo "[rejudge] ERROR: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN." >&2
  exit 2
fi

set +e
timeout --signal=TERM --kill-after=30s 15m apptainer exec \
  "${APPTAINER_NV_FLAG[@]}" \
  --pid \
  -c \
  --cleanenv \
  --env PATH="/root/.local/bin:/home/agent/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --bind "${JUDGE_TMP}:/tmp" \
  --bind "${EXP_DIR}/task:/home/agent/task:ro" \
  --home "${JUDGE_HOME}:/home/agent" \
  --pwd "/home/agent" \
  --writable-tmpfs \
  "${CONTAINER_SIF}" \
  sh -c 'exec claude --print --dangerously-skip-permissions --model "$1" < /tmp/judge_prompt.txt' sh "${JUDGE_MODEL}"
JUDGE_RC=$?
set -e
echo "[rejudge] judge exit=${JUDGE_RC}"

if ! valid_judgements "${JUDGE_HOME}"; then
  echo "[rejudge] ERROR: judge did not produce both valid judgement files." >&2
  [ "${JUDGE_RC}" -ne 0 ] || JUDGE_RC=1
fi

if [ "${JUDGE_RC}" -ne 0 ]; then
  exit "${JUDGE_RC}"
fi
for f in contamination_judgement disallowed_model_judgement; do
  dst="${EXP_DIR}/${f}__${JUDGE_SLUG}.txt"
  cp --remove-destination -- "${JUDGE_HOME}/${f}.txt" "${dst}"
  echo "[rejudge] wrote ${dst}"
done
