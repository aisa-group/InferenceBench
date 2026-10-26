#!/bin/bash
set -euo pipefail

TASK=""; SCENARIOS=""; BASE_MODELS=""; SEED_PAIRS=""; NUM_HOURS=""; PARALLEL=0
AGENT="claude-code"; MODEL=""; JOB_NAME=""; DELETE_VOLUME=0; CLI_VERSION=""; EFFORT=""
AGENT_KWARGS=(); EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --scenario) SCENARIOS="$2"; shift 2 ;;
        --base-model) BASE_MODELS="$2"; shift 2 ;;
        --seed-pairs) SEED_PAIRS="$2"; shift 2 ;;
        --num-hours) NUM_HOURS="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --agent) AGENT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --cli-version) CLI_VERSION="$2"; shift 2 ;;
        --effort) EFFORT="$2"; shift 2 ;;
        --agent-kwarg|--ak) AGENT_KWARGS+=(--ak "$2"); shift 2 ;;
        --delete-volume) DELETE_VOLUME=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '/^## Run/,/^## Results/p' "$(dirname "$0")/README.md"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "${MODEL}" ] || { echo "need --model (the agent's model)" >&2; exit 2; }
if [ -n "${TASK}" ]; then
    [ -z "${SCENARIOS}${BASE_MODELS}${SEED_PAIRS}${NUM_HOURS}" ] || { echo "--task cannot be combined with task selection options" >&2; exit 2; }
    [ -d "${TASK}" ] || { echo "task dir not found: ${TASK}" >&2; exit 2; }
    TASK="$(cd "${TASK}" && pwd)"
fi
[[ "${PARALLEL}" =~ ^[0-9]+$ ]] || { echo "--parallel must be a non-negative integer" >&2; exit 2; }
case "${AGENT}" in
    claude-code) KEYS=(CLAUDE_CODE_OAUTH_TOKEN CLAUDE_FORCE_OAUTH); CLI_PKG="@anthropic-ai/claude-code"; CLI_ENV="CLAUDE_CLI_VERSION" ;;
    codex) KEYS=(OPENAI_API_KEY); CLI_PKG="@openai/codex"; CLI_ENV="CODEX_CLI_VERSION" ;;
    gemini-cli) KEYS=(GEMINI_API_KEY); CLI_PKG="@google/gemini-cli"; CLI_ENV="GEMINI_CLI_VERSION" ;;
    opencode) KEYS=(OPENCODE_API_KEY); CLI_PKG="opencode-ai"; CLI_ENV="OPENCODE_CLI_VERSION" ;;
    *) echo "--agent must be claude-code, codex, gemini-cli or opencode" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${INFERENCE_BENCH_ENV_FILE:-${REPO_ROOT}/.env}"
env_file_value() {
    [ -f "${ENV_FILE}" ] || return 0
    grep -E "^$1=" "${ENV_FILE}" | head -1 | cut -d= -f2- | sed "s/^[\"']//; s/[\"']\$//" || true
}
for _k in ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN "${KEYS[@]}" "${CLI_ENV}"; do
    if [ -z "${!_k:-}" ]; then
        _v="$(env_file_value "${_k}")"
        [ -z "${_v}" ] || export "${_k}=${_v}"
    fi
done
[ -n "${ANTHROPIC_API_KEY:-}${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || { echo "the disallowed-usage judge needs ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN (exported or in ${ENV_FILE})" >&2; exit 2; }
_t="$(env_file_value HF_TOKEN)"
export HF_TOKEN="${_t:-${HF_TOKEN:-}}"
[ -n "${HF_TOKEN}" ] || { echo "set a read-only HF_TOKEN in ${ENV_FILE} or export it" >&2; exit 2; }

HARBOR_BIN="$(command -v harbor)" || { echo "harbor not on PATH" >&2; exit 2; }
HARBOR_PY="$(dirname "$(readlink -f "${HARBOR_BIN}")")/python"
MODAL=("${HARBOR_PY}" -m modal)

if [ -z "${CLI_VERSION}" ]; then
    CLI_VERSION="${!CLI_ENV:-}"
fi
if [ -z "${CLI_VERSION}" ]; then
    CLI_VERSION="$(grep -oE "${CLI_PKG}@[0-9][^ \\\\]*" "${SCRIPT_DIR}/template/Dockerfile" | head -1 | sed 's/.*@//')"
elif [ "${CLI_VERSION}" = "latest" ]; then
    CLI_VERSION="$(npm view "${CLI_PKG}" version 2>/dev/null)" || { echo "could not resolve the latest ${CLI_PKG}" >&2; exit 2; }
else
    npm view "${CLI_PKG}@${CLI_VERSION}" version > /dev/null 2>&1 || { echo "${CLI_PKG}@${CLI_VERSION} is not on npm" >&2; exit 2; }
fi
AGENT_KWARGS+=(--ak "version=${CLI_VERSION}")
echo "cli:     ${CLI_PKG}@${CLI_VERSION}"

AGENT_ENV=()
case "${AGENT}" in
    claude-code)
        AGENT_ENV+=(--ae "BASH_MAX_TIMEOUT_MS=36000000")
        [ -z "${EFFORT}" ] || AGENT_KWARGS+=(--ak "reasoning_effort=${EFFORT}")
        ;;
    codex)
        case "${MODEL}" in
            *-high) EFFORT="${EFFORT:-high}"; MODEL="${MODEL%-high}" ;;
            *-med) EFFORT="${EFFORT:-medium}"; MODEL="${MODEL%-med}" ;;
            *-low) EFFORT="${EFFORT:-low}"; MODEL="${MODEL%-low}" ;;
        esac
        [ -z "${EFFORT}" ] || AGENT_KWARGS+=(--ak "reasoning_effort=${EFFORT}")
        AGENT_KWARGS+=(--ak "web_search=live")
        ;;
    opencode)
        OPENCODE_JSON="$(cd "${SCRIPT_DIR}" && python3 -c 'from adapter import opencode_config; print(opencode_config())')" || exit 2
        AGENT_KWARGS+=(--ak "opencode_config=${OPENCODE_JSON}")
        [ -z "${OPENCODE_API_KEY:-}" ] || AGENT_ENV+=(--ae "OPENCODE_API_KEY=${OPENCODE_API_KEY}")
        ;;
esac

JOB_NAME="${JOB_NAME:-$(date +%Y%m%d-%H%M%S)}"
[[ "${JOB_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "--job-name may only contain letters, digits, '.', '_' and '-'" >&2; exit 2; }
JOBS_ROOT="${SCRIPT_DIR}/jobs"

if [ -n "${TASK}" ]; then
    TASKS=("${TASK}")
else
    TASKS_DIR="${SCRIPT_DIR}/tasks/${JOB_NAME}"
    [ ! -e "${TASKS_DIR}" ] && [ ! -e "${JOBS_ROOT}/${JOB_NAME}" ] || { echo "tasks/${JOB_NAME} or jobs/${JOB_NAME} already exists" >&2; exit 2; }
    ADAPTER_ARGS=(--scenario "${SCENARIOS:-all}" --base-model "${BASE_MODELS:-all}" --num-hours "${NUM_HOURS:-2}" --agent "${AGENT}" --output "${TASKS_DIR}")
    [ -z "${SEED_PAIRS}" ] || ADAPTER_ARGS+=(--seed-pairs "${SEED_PAIRS}")
    GENERATED="$(python3 "${SCRIPT_DIR}/run_adapter.py" "${ADAPTER_ARGS[@]}")" || { echo "task generation failed" >&2; exit 2; }
    mapfile -t TASKS <<< "${GENERATED}"
    echo "tasks:   ${#TASKS[@]} generated in tasks/${JOB_NAME}/"
fi

MODELS_TO_CHECK=($(sed -n 's/^base_model = "\(.*\)"$/\1/p' "${TASKS[@]/%//task.toml}" | sort -u))
python3 "${SCRIPT_DIR}/check_hf_token.py" "${MODELS_TO_CHECK[@]}"

task_short() { basename "$1" | sed 's/^inferencebench-//'; }
volume_name() { printf 'ib-%s-%s' "${JOB_NAME}" "$(task_short "$1")" | sed 's/[^A-Za-z0-9._-]/-/g' | cut -c1-64; }
DUPES="$(for t in "${TASKS[@]}"; do volume_name "$t"; echo; done | sort | uniq -d)"
[ -z "${DUPES}" ] || { echo "volume names collide after the 64-char cut (use a shorter --job-name): ${DUPES}" >&2; exit 2; }

run_one() {
    local task="$1" jobs_dir="$2" job="$3" volume rc
    shift 3
    volume="$(volume_name "${task}")"
    echo "task:    ${task}"
    echo "volume:  ${volume}"
    "${MODAL[@]}" volume create "${volume}" || echo "volume ${volume} exists; reusing it"
    set +e
    harbor run --path "${task}" --agent "${AGENT}" --model "${MODEL}" \
        "${AGENT_KWARGS[@]}" "${AGENT_ENV[@]}" \
        --env modal --ek "volumes={\"/mnt/ib_handoff\":\"${volume}\"}" \
        -n 1 --jobs-dir "${jobs_dir}" --job-name "${job}" "$@" "${EXTRA[@]}"
    rc=$?
    set -e
    echo "harbor exit code: ${rc}"
    echo "results:  ${jobs_dir}/${job}/"
    if [ "${DELETE_VOLUME}" = 1 ]; then
        "${MODAL[@]}" volume delete "${volume}" --yes && echo "volume ${volume} deleted"
    else
        echo "cleanup:  ${HARBOR_PY} -m modal volume delete ${volume} --yes"
    fi
    return "${rc}"
}

if [ -n "${TASK}" ]; then
    run_one "${TASK}" "${JOBS_ROOT}" "${JOB_NAME}"
    exit $?
fi

SWEEP_DIR="${JOBS_ROOT}/${JOB_NAME}"
mkdir -p "${SWEEP_DIR}"
[ "${PARALLEL}" -gt 0 ] || PARALLEL="${#TASKS[@]}"
for t in "${TASKS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "${PARALLEL}" ]; do wait -n || true; done
    run_one "$t" "${SWEEP_DIR}" "$(task_short "$t")" --yes > "${SWEEP_DIR}/$(task_short "$t").log" 2>&1 &
    echo "launched $(task_short "$t")"
done
wait

python3 - "${SWEEP_DIR}" "${TASKS[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

sweep, tasks = Path(sys.argv[1]), sys.argv[2:]
print(f"{'task':<30} {'harbor':>6}  {'reward':>10}  note")
done = 0
for task in tasks:
    short = re.sub(r"^inferencebench-", "", Path(task).name)
    log = sweep / f"{short}.log"
    match = re.search(r"^harbor exit code: (\d+)", log.read_text(errors="replace"), re.M) if log.is_file() else None
    results = sorted((sweep / short).glob("*/result.json"))
    reward, note = "-", f"no trial result, see {log.name}"
    if results:
        result = json.loads(results[-1].read_text())
        value = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        reward = f"{value:.4f}" if isinstance(value, (int, float)) else "-"
        error = result.get("exception_info") or {}
        note = f"{error.get('exception_type')}: {(error.get('exception_message') or '')[:60]}" if error else ""
    done += reward != "-"
    print(f"{short:<30} {match[1] if match else '?':>6}  {reward:>10}  {note}")
print(f"\n{done}/{len(tasks)} tasks produced a reward")
PY
echo "export: python3 ${SCRIPT_DIR}/harbor_to_results.py ${SWEEP_DIR}"
