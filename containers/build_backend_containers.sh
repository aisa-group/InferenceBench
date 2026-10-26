#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

suffix=""
backends=()
for arg in "$@"; do
  if [ "${arg}" = "--a100" ]; then
    suffix="_a100"
  else
    backends+=("${arg}")
  fi
done
[ "${#backends[@]}" -gt 0 ] || backends=(sglang vllm torch tgi)

for backend in "${backends[@]}"; do
  [ "${backend}" = "tgi" ] || backend+="${suffix}"
  bash containers/build_container.sh "inference_baselines_${backend}"
done
