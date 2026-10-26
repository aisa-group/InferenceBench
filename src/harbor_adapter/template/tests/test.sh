#!/bin/bash
IB=/opt/ib
HANDOFF=/mnt/ib_handoff
LOGS=/logs/verifier
TASK=/home/agent/task
EVAL_LOG="${LOGS}/eval.log"
SCENARIO="${INFERENCE_BENCH_SCENARIO}"
STUB_MARKER="ERROR: start_server.sh has no engine configured"
SERVER_PID=""
FINAL_EVAL_RC=1
JUDGE_RC=1
: "${PORT:=8000}" "${INFERENCE_BENCH_SERVER_URL:=http://127.0.0.1:8000}" "${INFERENCE_BENCH_SERVER_WAIT_S:=900}"
mkdir -p "${LOGS}" /root/eval_home /root/eval_tmp
chmod 755 "${LOGS}"
source "${IB}/src/disallowed_usage_judge/judge.sh"

log() {
    echo "[${SCENARIO}] $(date --iso-8601=seconds) $*" | tee -a "${EVAL_LOG}"
}

write_error_metrics() {
    python3 - "${LOGS}/metrics.json" "${SCENARIO}" "${INFERENCE_BENCH_BASE_MODEL}" "$1" <<'PY'
import json
import sys
from pathlib import Path
path, scenario, model, error = sys.argv[1:]
Path(path).write_text(json.dumps({
    "scenario": scenario, "model_id": model, "profiles": {}, "vram_peak_mb": 0.0, "error": error,
}, indent=2), encoding="utf-8")
PY
}

restore_home() {
    rm -rf /home/agent
    cp -a "${HANDOFF}/home" /home/agent
    mkdir -p /home/agent/hf_cache
    chown -R agent:agent /home/agent
}

restore_tmp() {
    local entry
    for entry in "${HANDOFF}"/tmp/*; do
        [ -e "${entry}" ] || continue
        rm -rf "/tmp/$(basename "${entry}")"
        cp -a "${entry}" /tmp/ && chown -R agent:agent "/tmp/$(basename "${entry}")"
    done
}

download_model() {
    local start
    start="$(date +%s)"
    log "downloading ${INFERENCE_BENCH_BASE_MODEL} into /hf_cache"
    rm -rf /home/agent/hf_cache/hub/.locks /home/agent/hf_cache/xet
    if ! env -i "${AGENT_ENV[@]}" HF_TOKEN="${HF_TOKEN}" setpriv --reuid=agent --regid=agent --clear-groups \
        python3 -c 'import os, huggingface_hub; huggingface_hub.snapshot_download(os.environ["INFERENCE_BENCH_BASE_MODEL"])' >> "${LOGS}/model_download.log" 2>&1; then
        echo "[fatal] base model download failed (see model_download.log); no reward written, so Harbor records an infrastructure error"
        exit 1
    fi
    log "model download took $(( $(date +%s) - start ))s; hf_cache is $(du -sh /home/agent/hf_cache | cut -f1)"
}

agent_env() {
    local names=(PATH HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE TMPDIR TMP TEMP XDG_CACHE_HOME
        VLLM_TORCH_COMPILE_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR PIP_USER PYTHONUSERBASE
        PIP_CACHE_DIR PYTHONPATH SOFT_FILELOCK NO_PROXY no_proxy LD_LIBRARY_PATH HOST PORT INFERENCE_BENCH_SERVER_URL
        INFERENCE_BENCH_SERVER_WAIT_S INFERENCE_BENCH_BASE_MODEL INFERENCE_BENCH_MAX_MODEL_LEN)
    local name
    AGENT_ENV=(HOME=/home/agent)
    for name in "${names[@]}" $(compgen -e | grep -E '^(NVIDIA_|CUDA_|RAY_)'); do
        [ -z "${!name+x}" ] || AGENT_ENV+=("${name}=${!name}")
    done
}

port_listeners() {
    ss -lptn "sport = :${PORT}" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2
}

run_judge() {
    rm -rf /home/agent
    mkdir -p /home/agent
    cp -a "${HANDOFF}/home/task" "${TASK}"
    chown -R agent:agent /home/agent
    local prompt
    prompt="$(python3 "${IB}/src/disallowed_usage_judge/get_judge_prompt.py" \
        --benchmark "$(cat "${IB}/src/eval/tasks/${SCENARIO}/benchmark.txt")" \
        --model "${INFERENCE_BENCH_BASE_MODEL}" --task-dir "${TASK}")"
    local credential="ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}"
    [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || credential="CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}"
    (
        cd /home/agent
        env -i "${AGENT_ENV[@]}" "${credential}" timeout --signal=TERM --kill-after=30s 15m \
            setpriv --reuid=agent --regid=agent --clear-groups claude --print --dangerously-skip-permissions --model "${DEFAULT_JUDGE_MODEL}" "${prompt}" < /dev/null
    ) > "${LOGS}/judge.log" 2>&1
    JUDGE_RC=$?
    pkill -9 -u agent
    valid_judgements /home/agent || JUDGE_RC=1
    for f in contamination_judgement.txt disallowed_model_judgement.txt; do
        if [ -f "/home/agent/${f}" ] && [ ! -L "/home/agent/${f}" ]; then
            cp "/home/agent/${f}" "${LOGS}/${f}"
        else
            echo "[warn] ${f} was not produced."
        fi
    done
    echo "Disallowed-usage judge exit code: ${JUDGE_RC}"
}

server_ready() {
    python3 /opt/inference_eval/bin/server_ready.py "${INFERENCE_BENCH_SERVER_URL}"
}

stop_server() {
    if [ -n "${SERVER_PID}" ]; then
        kill -- "-${SERVER_PID}" 2>/dev/null
        sleep 2
        kill -9 -- "-${SERVER_PID}" 2>/dev/null
        SERVER_PID=""
    fi
    local pid
    for pid in $(port_listeners); do
        kill "${pid}" 2>/dev/null
        sleep 2
        kill -9 "${pid}" 2>/dev/null
    done
}

start_server() {
    stop_server
    local wait_s=0
    while [ "${wait_s}" -lt 30 ] && [ -n "$(port_listeners)" ]; do
        sleep 1
        wait_s=$((wait_s + 1))
    done
    restore_tmp
    local launch=""
    local f
    for f in start_server.sh start_server_original.sh; do
        if [ -f "${TASK}/${f}" ] && ! grep -q "${STUB_MARKER}" "${TASK}/${f}"; then
            launch="${TASK}/${f}"
            break
        fi
    done
    if [ -z "${launch}" ]; then
        log "no configured start_server.sh found for the eval server."
        return 1
    fi
    log "starting the eval server via ${launch}"
    cd "${TASK}" || return 1
    env -i "${AGENT_ENV[@]}" setsid setpriv --reuid=agent --regid=agent --clear-groups bash -c 'set -euo pipefail; if [ -f /home/agent/task/final_runtime_env.sh ]; then . /home/agent/task/final_runtime_env.sh; fi; exec /opt/inference_eval/bin/launch_supervised_server.sh "$1"' \
        _ "${launch}" >> "${LOGS}/server.log" 2>&1 &
    SERVER_PID=$!
    cd "${IB}" || return 1
    local deadline=$(( $(date +%s) + INFERENCE_BENCH_SERVER_WAIT_S ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if server_ready; then
            log "eval server is ready at ${INFERENCE_BENCH_SERVER_URL}"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "eval server exited before becoming ready"
            return 1
        fi
        sleep 1
    done
    log "eval server did not become ready within ${INFERENCE_BENCH_SERVER_WAIT_S}s"
    return 1
}

run_eval_attempt() {
    HOME=/root/eval_home TMPDIR=/root/eval_tmp PYTHONNOUSERSITE=1 INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=0 timeout --signal=TERM --kill-after=60s 7200 \
        python3 "src/eval/tasks/${SCENARIO}/evaluate.py" \
        --server-url "${INFERENCE_BENCH_SERVER_URL}" \
        --json-output-file "${LOGS}/metrics.json" \
        --quality-tau "${INFERENCE_BENCH_QUALITY_TAU}" \
        "$@" >> "${EVAL_LOG}" 2>&1
}

run_eval_with_retry() {
    local max_retries="$1"
    shift
    local attempt
    for ((attempt = 1; attempt <= max_retries; attempt++)); do
        if ! server_ready && ! start_server; then
            FINAL_EVAL_RC=1
            continue
        fi
        log "eval attempt ${attempt}/${max_retries} $*"
        FINAL_EVAL_RC=0
        run_eval_attempt "$@" || FINAL_EVAL_RC=$?
        if [ "${FINAL_EVAL_RC}" -eq 0 ] && python3 /tests/valid_eval_metrics.py "${LOGS}/metrics.json" "${IB}"; then
            return 0
        fi
        [ "${FINAL_EVAL_RC}" -ne 0 ] || FINAL_EVAL_RC=1
        log "attempt ${attempt} failed (rc=${FINAL_EVAL_RC})"
    done
    return 1
}

write_reward() {
    python3 - "${LOGS}" "${SCENARIO}" "${FINAL_EVAL_RC}" "${JUDGE_RC}" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "/opt/ib")
from src.eval.inference.hpo_search_baselines import FAILURE_SCORE, primary_metric

logs, scenario, eval_rc, judge_rc = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
try:
    metrics = json.loads((logs / "metrics.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    metrics = {}
metrics = metrics if isinstance(metrics, dict) else {}
check = lambda name: isinstance(metrics.get(name), dict) and metrics[name].get("pass") is True
rewards = {"eval_valid": int(eval_rc == "0"), "speed_pass": int(check("speed_check")), "quality_pass": int(check("quality_check"))}
value = primary_metric(metrics, scenario)[0] if all(rewards.values()) and metrics.get("score") != 0 else 0.0
rewards = {"reward": value if value > FAILURE_SCORE else 0.0, **rewards, "judge_valid": int(judge_rc == "0")}
verdicts = {"contamination_detected": ("contamination_judgement.txt", "contamination detected", "no contamination detected"),
            "disallowed_use_detected": ("disallowed_model_judgement.txt", "disallowed use detected", "only allowed use detected")}
for key, (name, flagged, clean) in verdicts.items():
    path = logs / name
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if lines in ([flagged], [clean]):
        rewards[key] = int(lines == [flagged])
(logs / "reward.json").write_text(json.dumps(rewards, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rewards))
PY
}

echo "=== InferenceBench verifier: ${SCENARIO} ${INFERENCE_BENCH_BASE_MODEL} eval_seed=${INFERENCE_BENCH_DATASET_SEED} ==="
for _ in $(seq 60); do
    [ ! -f "${HANDOFF}/complete" ] || break
    sleep 2
done
if [ ! -f "${HANDOFF}/complete" ] || [ ! -d "${HANDOFF}/home/task" ]; then
    echo "[fatal] incomplete hand-off at ${HANDOFF}; no reward written, so Harbor records an infrastructure error"
    exit 1
fi

agent_env
log "judge start"
run_judge
log "judge done (rc=${JUDGE_RC})"

unset ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN
restore_home
log "restored /home/agent ($(du -sh /home/agent | cut -f1)) from the hand-off"
download_model
unset HF_TOKEN
for f in metrics_preview.json server.log final_runtime_env.sh final_runtime_manifest.txt; do
    [ ! -f "${TASK}/${f}" ] || [ -L "${TASK}/${f}" ] || cp "${TASK}/${f}" "${LOGS}/${f}"
done
python3 /tests/snapshot.py "${TASK}" "${LOGS}/task"

cd "${IB}" || exit 0
rm -f "${LOGS}/metrics.json"
if start_server; then
    log "starting final evaluation (speed + quality)"
    if ! run_eval_with_retry 3; then
        log "phase 1 failed; retrying with reduced request-timeout"
        run_eval_with_retry 2 --request-timeout-s 150
    fi
else
    log "final evaluation aborted: inference server unavailable"
    write_error_metrics "inference server unavailable before final eval at ${INFERENCE_BENCH_SERVER_URL}"
fi
log "final eval exit code: ${FINAL_EVAL_RC}"
stop_server
write_reward
exit 0
