#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner import _build_profiles, _compute_choice_accuracy, _detect_model_id, _run_profile, _summarize_results, load_scenario_config


def _load_requests(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_baseline(
    server_url: str,
    requests_list: List[Dict[str, Any]],
    model_id: str,
    log_path: Optional[Path],
    task_dir: Path,
    request_timeout_s: float,
    concurrency_override: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_scenario_config(task_dir)
    seed = config["dataset_seed"] if seed is None else seed
    profile_metrics: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []
    for profile in _build_profiles(config, len(requests_list), concurrency_override):
        profile_name = profile.get("name") or profile.get("pattern", "profile")
        results, wall_start, wall_end = asyncio.run(
            _run_profile(profile, requests_list, server_url, model_id, request_timeout_s, log_path, seed)
        )
        all_results.extend(results)
        profile_metrics[profile_name] = _summarize_results(results, wall_start, wall_end)

    return {
        "model_id": model_id,
        "request_count": len(all_results),
        "success_count": sum(1 for r in all_results if r.get("success")),
        "choice_accuracy": _compute_choice_accuracy(all_results),
        "profiles": profile_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline server on the same requests.")
    parser.add_argument("--requests-file", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)
    parser.add_argument("--server-url", type=str, default=os.environ.get("INFERENCE_BENCH_BASELINE_SERVER_URL", ""))
    parser.add_argument("--model", type=str, default=os.environ.get("INFERENCE_BENCH_BASELINE_MODEL", ""))
    parser.add_argument("--log-file", type=str, default=os.environ.get("INFERENCE_BENCH_BASELINE_LOG", ""))
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    if not args.server_url:
        print("Baseline server URL is not set. Skipping baseline eval.")
        return

    requests_path = Path(args.requests_file)
    if not requests_path.is_file():
        print(f"Requests file not found: {requests_path}. Skipping baseline eval.")
        return

    model_id = _detect_model_id(args.server_url, args.model)
    requests_list = _load_requests(requests_path)
    log_path = Path(args.log_file) if args.log_file else requests_path.parent / "baseline_generations.jsonl"
    baseline_metrics = _run_baseline(
        args.server_url,
        requests_list,
        model_id,
        log_path,
        requests_path.parent,
        args.request_timeout_s,
    )

    metrics_path = Path(args.metrics_file)
    metrics: Dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["baseline"] = baseline_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
