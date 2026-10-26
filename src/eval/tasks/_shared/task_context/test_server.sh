#!/usr/bin/env bash
set -euo pipefail

exec /opt/inference_eval/bin/launch_supervised_server.sh "/home/agent/task/start_server.sh"
