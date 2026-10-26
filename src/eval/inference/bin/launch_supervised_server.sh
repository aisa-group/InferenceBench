#!/usr/bin/env bash
set -euo pipefail

LAUNCH_SCRIPT="${1:-/home/agent/task/start_server.sh}"
SERVER_URL="${INFERENCE_BENCH_SERVER_URL:-http://127.0.0.1:8000}"
WAIT_S="${INFERENCE_BENCH_SERVER_WAIT_S:-900}"
launcher_pid=""

READY_CHECK="$(dirname "${BASH_SOURCE[0]}")/server_ready.py"

healthcheck() {
    python3 "${READY_CHECK}" "${SERVER_URL}"
}

check_launcher_exit() {
    if kill -0 "${launcher_pid}" 2>/dev/null; then
        return
    fi
    local rc=0
    wait "${launcher_pid}" || rc=$?
    if healthcheck; then
        echo "[supervised] invalid detached-server behavior: ${LAUNCH_SCRIPT} exited" >&2
        exit 97
    fi
    echo "[supervised] launch script exited before readiness (rc=${rc})" >&2
    [ "${rc}" -ne 0 ] || rc=1
    exit "${rc}"
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [ -n "${launcher_pid}" ]; then
        kill -- -"${launcher_pid}" 2>/dev/null || true
        sleep 1
        kill -9 -- -"${launcher_pid}" 2>/dev/null || true
        wait "${launcher_pid}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ ! -f "${LAUNCH_SCRIPT}" ]; then
    echo "[supervised] launch script not found: ${LAUNCH_SCRIPT}" >&2
    exit 1
fi

if healthcheck; then
    echo "[supervised] server endpoint is already in use: ${SERVER_URL}" >&2
    exit 1
fi

echo "[supervised] launching ${LAUNCH_SCRIPT} for ${SERVER_URL}" >&2
setsid bash "${LAUNCH_SCRIPT}" &
launcher_pid=$!
deadline=$(( $(date +%s) + WAIT_S ))
ready=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
    check_launcher_exit
    if healthcheck; then
        ready=1
        echo "[supervised] server ready at ${SERVER_URL}" >&2
        break
    fi
    sleep 1
done
if [ "${ready}" -ne 1 ]; then
    check_launcher_exit
    echo "[supervised] server did not become ready within ${WAIT_S}s at ${SERVER_URL}" >&2
    exit 1
fi
rc=0
wait "${launcher_pid}" || rc=$?
if healthcheck; then
    echo "[supervised] invalid detached-server behavior: ${LAUNCH_SCRIPT} exited" >&2
    exit 97
fi
exit "${rc}"
