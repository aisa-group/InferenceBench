#!/usr/bin/env bash
set -euo pipefail

LAUNCH_SCRIPT="${1:-/home/agent/task/start_server.sh}"
SERVER_HOST="${INFERENCE_BENCH_SERVER_HOST:-${HOST:-127.0.0.1}}"
SERVER_PORT="${INFERENCE_BENCH_SERVER_PORT:-${PORT:-8000}}"
SERVER_URL="${INFERENCE_BENCH_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
WAIT_S_RAW="${INFERENCE_BENCH_SERVER_WAIT_S:-900}"
CHECK_INTERVAL_S="${INFERENCE_BENCH_SERVER_CHECK_INTERVAL_S:-1}"

launcher_pid=""
launcher_pgid=""

healthcheck() {
    python3 - "${SERVER_URL}" <<'PY'
import sys
import urllib.error
import urllib.request

if len(sys.argv) < 2 or not sys.argv[1]:
    raise SystemExit(1)
url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        status = int(getattr(response, "status", 0))
        raise SystemExit(0 if (200 <= status < 300 or status in (401, 403)) else 1)
except urllib.error.HTTPError as exc:
    raise SystemExit(0 if int(getattr(exc, "code", 0)) in (401, 403) else 1)
except Exception:
    raise SystemExit(1)
PY
}

cleanup() {
    local rc=$?
    if [ -n "${launcher_pgid}" ]; then
        kill -- -"${launcher_pgid}" 2>/dev/null || true
        sleep 1
        kill -9 -- -"${launcher_pgid}" 2>/dev/null || true
    elif [ -n "${launcher_pid}" ]; then
        kill "${launcher_pid}" 2>/dev/null || true
        sleep 1
        kill -9 "${launcher_pid}" 2>/dev/null || true
    fi
    exit "${rc}"
}

trap cleanup INT TERM

if [ ! -f "${LAUNCH_SCRIPT}" ]; then
    echo "[supervised] launch script not found: ${LAUNCH_SCRIPT}" >&2
    exit 1
fi

if [[ "${WAIT_S_RAW}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    WAIT_S="${WAIT_S_RAW%.*}"
else
    WAIT_S=900
fi
if [ -z "${WAIT_S}" ] || [ "${WAIT_S}" -lt 1 ]; then
    WAIT_S=900
fi
if ! [[ "${CHECK_INTERVAL_S}" =~ ^[0-9]+$ ]]; then
    CHECK_INTERVAL_S=1
fi
if [ "${CHECK_INTERVAL_S}" -lt 1 ]; then
    CHECK_INTERVAL_S=1
fi

echo "[supervised] launching ${LAUNCH_SCRIPT} for ${SERVER_URL}" >&2
bash "${LAUNCH_SCRIPT}" &
launcher_pid=$!
launcher_pgid="$(ps -o pgid= -p "${launcher_pid}" 2>/dev/null | tr -d ' ' || true)"
if [ -z "${launcher_pgid}" ]; then
    launcher_pgid="${launcher_pid}"
fi

deadline=$(( $(date +%s) + WAIT_S ))
while [ "$(date +%s)" -lt "${deadline}" ]; do
    if healthcheck; then
        echo "[supervised] server ready at ${SERVER_URL}" >&2
        break
    fi
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
        sleep 1
        if healthcheck; then
            echo "[supervised] invalid detached-server behavior: ${LAUNCH_SCRIPT} exited while ${SERVER_URL} remained reachable" >&2
            exit 97
        fi
        set +e
        wait "${launcher_pid}"
        rc=$?
        set -e
        echo "[supervised] launch script exited before readiness (rc=${rc})" >&2
        exit "${rc}"
    fi
    sleep "${CHECK_INTERVAL_S}"
done

if ! healthcheck; then
    echo "[supervised] server did not become ready within ${WAIT_S}s at ${SERVER_URL}" >&2
    exit 1
fi

set +e
wait "${launcher_pid}"
rc=$?
set -e
if healthcheck; then
    echo "[supervised] invalid detached-server behavior: ${LAUNCH_SCRIPT} exited while ${SERVER_URL} remained reachable" >&2
    exit 97
fi
exit "${rc}"
