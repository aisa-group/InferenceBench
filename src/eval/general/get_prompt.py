#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

def read_benchmark_name(benchmark_id: str) -> str:
    """Resolve the human-readable benchmark name from the benchmark_id."""
    bench_file = Path("src/eval/tasks") / benchmark_id / "benchmark.txt"
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found for id '{benchmark_id}': {bench_file}")
    return bench_file.read_text(encoding="utf-8").strip()

def read_mission(benchmark_id: str) -> str:
    mission_file = Path("src/eval/tasks") / benchmark_id / "mission.txt"
    if not mission_file.is_file():
        return "TBD"
    return mission_file.read_text(encoding="utf-8").strip()

def read_workspace_conventions(benchmark_id: str) -> str:
    conventions_file = Path("src/eval/tasks") / benchmark_id / "workspace_conventions.txt"
    if conventions_file.is_file():
        return conventions_file.read_text(encoding="utf-8").strip()
    return (
        "- Create ONE top-level directory for your work: `agent/`.\n"
        "- At the start, create a run folder: `agent/run_YYYYMMDD_HHMMSS/` (UTC).\n"
        "- Put all new files under that run folder, e.g.:\n"
        "  - `agent/run_.../notes.md`\n"
        "  - `agent/run_.../logs/<name>.log`\n"
        "  - `agent/run_.../exp/<exp_name>/...` (all sweep outputs)\n"
        "  - `agent/run_.../tmp/...` (throwaway)\n"
        "- Do NOT create ad-hoc top-level dirs like `sweep_tmp*`, `sweep_results*`, `tmp*` at repo root.\n"
        "- `start_server.sh` is the benchmark-owned launcher you are expected to edit.\n"
        "- `test_server.sh` and `/opt/inference_eval/bin/launch_supervised_server.sh` are benchmark-owned scaffolds;\n"
        "  treat them as read-only and do not rely on detached daemons.\n"
        "- Do NOT create or overwrite benchmark-owned files at repo root other than `start_server.sh`:\n"
        "  - `server.log`, `server.pid`\n"
        "  - `evaluate.py`, `metrics.json`, `eval_generations.jsonl`, `requests_used.jsonl`\n"
        "  - `requests.jsonl` (treat as read-only; if you need subsets, write `agent/.../requests_subset.jsonl`)\n"
        "- When doing quick experiments, ALWAYS pass `--requests-file` pointing to your subset file;\n"
        "  never rely on implicit defaults that may reuse `requests.jsonl`.\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--base-model', type=str, required=True)
    parser.add_argument('--scenario-id', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)
    parser.add_argument('--starting-point', type=str, default='default',
                        choices=['default', 'vllm_running', 'bare'])

    args = parser.parse_args()

    benchmark_name = read_benchmark_name(args.scenario_id)
    mission = read_mission(args.scenario_id)
    workspace_conventions = read_workspace_conventions(args.scenario_id)

    base_prompt = os.environ.get('INFERENCE_BENCH_PROMPT', os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt'))

    template_path = f'src/eval/general/{base_prompt}.txt'

    with open(template_path, 'r') as f:
        template = f.read()

    datetime = subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()

    result = template.replace('{model}', args.base_model)
    result = result.replace('{scenario}', benchmark_name)
    result = result.replace('{mission}', mission)
    result = result.replace('{num_hours}', args.num_hours)
    result = result.replace('{workspace_conventions}', workspace_conventions)
    hf_home = os.environ.get('HF_HOME_NEW') or os.environ.get('HF_HOME', '')
    hf_hub_cache = os.environ.get('HF_HUB_CACHE', '')
    model_max_len = os.environ.get('INFERENCE_BENCH_MAX_MODEL_LEN', '')
    server_url = os.environ.get('INFERENCE_BENCH_SERVER_URL', 'http://127.0.0.1:8000')
    server_host = os.environ.get('INFERENCE_BENCH_SERVER_HOST', '127.0.0.1')
    server_port = os.environ.get('INFERENCE_BENCH_SERVER_PORT', '8000')
    metrics_path = os.environ.get('INFERENCE_BENCH_METRICS_PATH', '').strip()
    if not metrics_path:
        raise RuntimeError("INFERENCE_BENCH_METRICS_PATH must be set so the prompt can reference --json-output-file.")
    result = result.replace('{hf_home}', hf_home)
    result = result.replace('{hf_hub_cache}', hf_hub_cache)
    result = result.replace('{model_max_len}', model_max_len)
    result = result.replace('{server_url}', server_url)
    result = result.replace('{server_host}', server_host)
    result = result.replace('{server_port}', str(server_port))
    result = result.replace('{metrics_path}', metrics_path)

    result = result.replace('{datetime}', datetime)

    # Append starting-point-specific context
    if args.starting_point == 'vllm_running':
        result += f"""

## Starting Point: Preconfigured vLLM
Your start_server.sh already contains a working vLLM configuration for serving {args.base_model}
at {server_url}. The starting environment already has the required vLLM dependencies installed
before your timer begins. No server is running yet: you must still launch and manage it yourself.
Focus your time on optimization rather than initial environment setup.
"""
    elif args.starting_point == 'bare':
        result += f"""

## Starting Point: Bare Environment
No inference server is configured. No start_server.sh is provided. The model weights are NOT
pre-cached; you must download {args.base_model} (or a quantized variant) yourself. You have full
internet access. You must create start_server.sh from scratch, install any necessary packages,
and set up the entire inference pipeline.
"""

    if args.agent == 'claude':
        result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()
