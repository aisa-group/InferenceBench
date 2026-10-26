#!/bin/bash
set -uo pipefail
mkdir -p /logs/artifacts
exec > >(tee -a /logs/artifacts/collect.log) 2>&1

HANDOFF=/mnt/ib_handoff
TASK=/home/agent/task
MODEL_REPO="hf_cache/hub/models--${INFERENCE_BENCH_BASE_MODEL//\//--}"
: "${PORT:=8000}" "${INFERENCE_BENCH_SERVER_URL:=http://127.0.0.1:8000}"

phase() {
    echo "[collect] $(date --iso-8601=seconds) $*"
}

phase "runtime capture"
if python3 /opt/inference_eval/bin/server_ready.py "${INFERENCE_BENCH_SERVER_URL}"; then
    pid="$(ss -lptn "sport = :${PORT}" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)"
    if [ -n "${pid}" ] && [ -r "/proc/${pid}/environ" ] &&
        python3 /usr/local/bin/capture_runtime_env.py "${pid}" "${TASK}/final_runtime_env.sh" \
            "${TASK}/final_runtime_manifest.txt" "${INFERENCE_BENCH_SERVER_URL}"; then
        chown -h agent:agent "${TASK}/final_runtime_env.sh" "${TASK}/final_runtime_manifest.txt"
        phase "captured the server environment from pid=${pid}"
    else
        phase "WARNING: could not capture the environment of the server at ${INFERENCE_BENCH_SERVER_URL}"
    fi
else
    phase "no live agent server found for runtime capture"
fi

pkill -9 -u agent
sleep 2
phase "stopped the agent's processes"

rm -rf "${HANDOFF}/complete" "${HANDOFF}/home" "${HANDOFF}/tmp"
mkdir -p "${HANDOFF}/tmp" "${HANDOFF}/home" || { phase "ERROR: cannot write ${HANDOFF}"; exit 1; }

phase "tmp snapshot"
while IFS= read -r link; do
    target="$(readlink "${link}")"
    case "${target}" in
        /tmp/?*) ;;
        *) continue ;;
    esac
    case "${target}" in
        *..*) continue ;;
    esac
    [ -e "${target}" ] || continue
    if mkdir -p "$(dirname "${HANDOFF}${target}")" && cp -aT -- "${target}" "${HANDOFF}${target}"; then
        phase "snapshot ${link} -> ${target} ($(du -sh "${HANDOFF}${target}" | cut -f1))"
    else
        phase "ERROR: snapshot of ${target} failed"
        exit 1
    fi
done < <(find "${TASK}" -maxdepth 1 -type l)

phase "home copy: /home/agent ($(du -sh /home/agent | cut -f1)) without ${MODEL_REPO} and agent CLI state"
if tar -C /home/agent --exclude="${MODEL_REPO}" --exclude=./.nvm --exclude=./.npm --exclude=./.claude --exclude=./.claude.json \
    --exclude=./.codex --exclude=./.gemini --exclude=./.config/opencode --exclude=./.local/share/opencode \
    --exclude=./.local/share/claude --exclude=oauth_token --exclude=api_key -cf - . | tar -C "${HANDOFF}/home" -xf -; then
    date --iso-8601=seconds > "${HANDOFF}/complete"
    sync "${HANDOFF}" || sync
    phase "handed off $(du -sh "${HANDOFF}/home" | cut -f1)"
else
    rm -rf "${HANDOFF}/home"
    phase "ERROR: copy of /home/agent to ${HANDOFF} failed; the verifier will report an incomplete hand-off"
    exit 1
fi
