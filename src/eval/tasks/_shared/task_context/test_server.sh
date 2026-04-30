#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for local testing. This launches ./start_server.sh under
# the benchmark's supervised scaffold so detached daemons are treated the same
# way they will be during final evaluation.

exec /opt/inference_eval/bin/launch_supervised_server.sh "/home/agent/task/start_server.sh"
