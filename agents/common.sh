export BASH_MAX_TIMEOUT_MS=36000000
PROMPT="$(cat "${PROMPT_FILE}")"
DEADLINE_TS=$(($(date +%s) + INFERENCE_BENCH_REMAINING_SECONDS))

time_left() {
  local left=$((DEADLINE_TS - $(date +%s)))
  echo $((left > 0 ? left : 0))
}

continue_prompt() {
  local left
  left="$(time_left)"
  printf 'Continue where you left off and complete the task. You have approximately %d minutes (%d seconds) of wall-clock time remaining. Make sure you utilize this time fully to achieve the best results you can. Do not ask for user feedback.' \
    "$((left / 60))" "${left}"
}

run_phase() {
  local phase="$1" left
  shift
  left="$(time_left)"
  if ((left <= 0)); then
    echo "[agent] ${phase}: no budget left, skipping"
    return 124
  fi
  echo "[agent] ${phase}: launching with remaining budget ${left}s"
  timeout --signal=TERM --kill-after=30s "${left}s" "$@" | tee "${AGENT_LAST_OUTPUT:-/dev/null}"
  return "${PIPESTATUS[0]}"
}

run_agent() {
  local rc=0
  if ((INFERENCE_BENCH_RUN_INDEX > 1)); then
    run_phase resume "${CONTINUE_CMD[@]}" "$(continue_prompt)" || rc=$?
    if ((rc != 0 && rc != 124)); then
      echo "[agent] resume failed rc=${rc}; starting a fresh session"
      rc=0
      run_phase start "${START_CMD[@]}" "${PROMPT}" || rc=$?
    fi
  else
    run_phase start "${START_CMD[@]}" "${PROMPT}" || rc=$?
  fi
  echo "[agent] first session finished rc=${rc}"
  while ((rc == 0 && $(time_left) > 0)); do
    run_phase continue "${CONTINUE_CMD[@]}" "$(continue_prompt)" || rc=$?
    echo "[agent] continue finished rc=${rc}"
    sleep 1
  done
  return "${rc}"
}
