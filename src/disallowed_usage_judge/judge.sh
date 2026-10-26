DEFAULT_JUDGE_MODEL="claude-sonnet-4-6"

judge_credentials() {
    local token="${CLAUDE_CODE_OAUTH_TOKEN:-}"
    if [ -z "${token}" ] && [ -f "agents/claude_non_api/oauth_token" ]; then
        token="$(cat agents/claude_non_api/oauth_token)"
    fi
    export APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN="${token}"
    export APPTAINERENV_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
    [ -z "${token}" ] || APPTAINERENV_ANTHROPIC_API_KEY=""
}

valid_judgement() {
    local file="$1"
    shift
    [ -f "${file}" ] && [ ! -L "${file}" ] || return 1
    local -a lines
    mapfile -t lines < "${file}"
    [ "${#lines[@]}" -eq 1 ] || return 1
    local allowed
    for allowed in "$@"; do
        [ "${lines[0]}" != "${allowed}" ] || return 0
    done
    return 1
}

valid_judgements() {
    valid_judgement "$1/contamination_judgement.txt" 'no contamination detected' 'contamination detected' &&
        valid_judgement "$1/disallowed_model_judgement.txt" 'only allowed use detected' 'disallowed use detected'
}
