#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .runner import _compute_choice_accuracy, _run_profile, _summarize_results, load_scenario_config


def _detect_model_id(server_url: str, fallback: str) -> str:
    if fallback:
        return fallback
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json().get("data") or []
            if data:
                return data[0].get("id", fallback) or fallback
    except requests.RequestException:
        pass
    return fallback or "unknown"


def _load_requests(path: Path) -> List[Dict[str, Any]]:
    requests_list: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            requests_list.append(json.loads(line))
    return requests_list


def _run_baseline(
    server_url: str,
    requests_list: List[Dict[str, Any]],
    model_id: str,
    log_path: Optional[Path],
    task_dir: Path,
    request_timeout_s: float,
    concurrency_override: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_scenario_config(task_dir)
    profiles = config.get("profiles")
    if not profiles:
        profiles = [config.get("profile", {"name": "default", "pattern": "burst"})]

    adjusted_profiles: List[Dict[str, Any]] = []
    for profile in profiles:
        profile_copy = dict(profile)
        profile_copy["num_requests"] = len(requests_list)
        if concurrency_override is not None:
            profile_copy["concurrency"] = concurrency_override
        adjusted_profiles.append(profile_copy)
    profiles = adjusted_profiles

    profile_metrics: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []
    for profile in profiles:
        profile_name = profile.get("name") or profile.get("pattern", "profile")
        results, wall_start, wall_end = asyncio.run(
            _run_profile(profile, requests_list, server_url, model_id, request_timeout_s, log_path)
        )
        all_results.extend(results)
        profile_metrics[profile_name] = _summarize_results(results, wall_start, wall_end)

    accuracy = _compute_choice_accuracy(all_results)
    return {
        "model_id": model_id,
        "request_count": len(all_results),
        "success_count": len([r for r in all_results if r.get("success")]),
        "choice_accuracy": accuracy,
        "profiles": profile_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline server on the same requests.")
    parser.add_argument("--requests-file", type=str, required=True)
    parser.add_argument("--metrics-file", type=str, required=True)
    parser.add_argument("--server-url", type=str, default=os.environ.get("INFERENCE_BENCH_BASELINE_SERVER_URL", ""))
    parser.add_argument("--model", type=str, default=os.environ.get("INFERENCE_BENCH_BASELINE_MODEL", ""))
    parser.add_argument("--log-file", type=str, default=os.environ.get("INFERENCE_BENCH_BASELINE_LOG", ""))
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    if not args.server_url:
        print("Baseline server URL not set; skipping baseline eval.")
        return

    requests_path = Path(args.requests_file)
    if not requests_path.is_file():
        print(f"Requests file not found: {requests_path}; skipping baseline eval.")
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
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
