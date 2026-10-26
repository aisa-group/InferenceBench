#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def read_task_file(benchmark_id: str, name: str) -> str:
    path = TASKS_DIR / benchmark_id / name
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found for task '{benchmark_id}': {path}")
    return path.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--base-model', type=str, required=True)
    parser.add_argument('--scenario-id', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)
    parser.add_argument('--starting-point', type=str, default='default',
                        choices=['default', 'vllm_running', 'bare'])
    args = parser.parse_args()

    metrics_path = os.environ.get('INFERENCE_BENCH_METRICS_PATH', '').strip()
    if not metrics_path:
        raise RuntimeError("INFERENCE_BENCH_METRICS_PATH must be set.")
    server_url = os.environ.get('INFERENCE_BENCH_SERVER_URL', 'http://127.0.0.1:8000')
    base_prompt = os.environ.get('INFERENCE_BENCH_PROMPT', 'prompt')
    result = (Path(__file__).parent / f'{base_prompt}.txt').read_text()
    values = {
        'model': args.base_model,
        'scenario': read_task_file(args.scenario_id, 'benchmark.txt'),
        'mission': read_task_file(args.scenario_id, 'mission.txt'),
        'num_hours': args.num_hours,
        'server_url': server_url,
        'metrics_path': metrics_path,
    }
    for key, value in values.items():
        result = result.replace('{' + key + '}', value)

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
pre-cached; you must download the exact {args.base_model} checkpoint yourself. You have full
internet access. You must create start_server.sh from scratch, install any necessary packages,
and set up the entire inference pipeline.
"""

    result += '\n\nWorkspace conventions:\n' + read_task_file('_shared', 'workspace_conventions.txt')

    if args.agent in {'claude', 'claude_non_api'}:
        result += """
You are running in a non-interactive mode. Do not ask for user input. Preserve start_server.sh so the evaluator can restart the server.
"""
    print(result)


if __name__ == '__main__':
    main()
