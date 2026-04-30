#!/usr/bin/env python3
"""
Precompute a baseline generations log for a scenario, so experiment runs don't
need to rerun the baseline every time.

This produces:
- requests.jsonl (deterministic based on scenario.json sampling)
- baseline_generations.jsonl (model outputs + parsed answers + gold)
- baseline_metrics.json (summary stats)
and updates the baseline registry JSON under src/eval/inference/baselines/.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import baseline_eval
from .runner import (
    _get_model_max_len,
    _get_tokenizer,
    _prepare_requests,
    _request_content_hash,
    load_scenario_config,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_requests_jsonl(path: Path, requests_list: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for req in requests_list:
            handle.write(
                json.dumps(
                    {
                        "messages": req.get("messages"),
                        "gold_answer": req.get("gold_answer"),
                        "sample_id": req.get("sample_id"),
                        "max_new_tokens": req.get("max_new_tokens"),
                        "temperature": req.get("temperature"),
                        "require_json": req.get("require_json", False),
                        "parse_mode": req.get("parse_mode", ""),
                        "ignore_eos": req.get("ignore_eos", False),
                        "target_input_token_count": req.get("target_input_token_count"),
                        "input_token_count": req.get("input_token_count"),
                        "source_prompt_token_count": req.get("source_prompt_token_count"),
                        "sampling_range_ratio": req.get("sampling_range_ratio"),
                        "content_hash": req.get("content_hash") or _request_content_hash(req),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"format_version": 1, "scenarios": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"format_version": 1, "scenarios": {}}
        if "scenarios" not in data or not isinstance(data.get("scenarios"), dict):
            data["scenarios"] = {}
        if "format_version" not in data:
            data["format_version"] = 1
        return data
    except Exception:
        return {"format_version": 1, "scenarios": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute InferenceBench baseline for a scenario.")
    parser.add_argument("--scenario-id", type=str, required=True, help="Folder name under src/eval/tasks/")
    parser.add_argument("--base-model", type=str, default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", ""))
    parser.add_argument("--server-url", type=str, required=True, help="Baseline OpenAI-compatible server URL")
    parser.add_argument("--out-root", type=str, default="", help="Baseline output root dir (defaults under repo)")
    parser.add_argument(
        "--requests-file",
        type=str,
        default="",
        help="Path to an existing requests.jsonl to reuse (skips dataset sampling).",
    )
    parser.add_argument(
        "--reuse-requests",
        action="store_true",
        help="Reuse existing requests.jsonl in the output dir if present (skips dataset sampling).",
    )
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency-override", type=int, default=None,
                        help="Override profile concurrency (e.g. 1 for sequential backends like torch).")
    parser.add_argument("--registry", type=str, default="", help="Registry JSON path (defaults to <model_safe>.json)")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248")),
        help="Seed for dataset sampling.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    scenario_src = repo_root / "src" / "eval" / "tasks" / args.scenario_id
    if not scenario_src.is_dir():
        raise FileNotFoundError(f"Scenario not found: {scenario_src}")

    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (args.base_model or "unknown_model")).strip("_")

    out_root = Path(args.out_root) if args.out_root else (repo_root / "src" / "eval" / "inference" / "baselines" / "speed" / "default")
    # Layout: <out_root>/<scenario>/<model_safe>/...
    out_dir = out_root / args.scenario_id / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the scenario definition so baseline_eval can load scenario.json from the output dir.
    _copy_if_exists(scenario_src / "scenario.json", out_dir / "scenario.json")
    _copy_if_exists(scenario_src / "mission.txt", out_dir / "mission.txt")
    _copy_if_exists(scenario_src / "benchmark.txt", out_dir / "benchmark.txt")

    requests_path = out_dir / "requests.jsonl"
    requests_file = Path(args.requests_file).expanduser().resolve() if args.requests_file else None
    if requests_file:
        if not requests_file.is_file():
            raise FileNotFoundError(f"Requests file not found: {requests_file}")
        if requests_file.resolve() != requests_path.resolve():
            requests_path.parent.mkdir(parents=True, exist_ok=True)
            requests_path.write_bytes(requests_file.read_bytes())
        print(f"[precompute] using requests file: {requests_file}")
    elif args.reuse_requests and requests_path.exists():
        print(f"[precompute] reusing existing requests file: {requests_path}")
    else:
        config = load_scenario_config(scenario_src)
        if args.seed is not None:
            config["dataset_seed"] = args.seed
        print(f"[precompute] scenario={args.scenario_id} dataset_seed={config.get('dataset_seed')}")
        tokenizer = _get_tokenizer(args.base_model) if args.base_model else None
        max_model_len = _get_model_max_len(args.base_model) if args.base_model else None
        base_requests, _ = _prepare_requests(config, limit=args.request_limit, tokenizer=tokenizer, max_model_len=max_model_len)
        print(f"[precompute] prepared {len(base_requests)} requests (limit={args.request_limit})")
        if base_requests:
            sample = base_requests[0]
            print(
                "[precompute] first request sample_id="
                f"{sample.get('sample_id')} max_new_tokens={sample.get('max_new_tokens')} temperature={sample.get('temperature')}"
            )
        _write_requests_jsonl(requests_path, base_requests)

    requests_sha = _sha256(requests_path)

    print(f"[precompute] baseline server url={args.server_url}")
    model_id = baseline_eval._detect_model_id(args.server_url, args.base_model)  # pylint: disable=protected-access
    print(f"[precompute] detected model_id={model_id}")
    requests_list = baseline_eval._load_requests(requests_path)  # pylint: disable=protected-access
    print(f"[precompute] loaded {len(requests_list)} requests from {requests_path}")

    log_path = out_dir / "baseline_generations.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(f"[precompute] running baseline eval (timeout_s={args.request_timeout_s})")
    baseline_metrics = baseline_eval._run_baseline(  # pylint: disable=protected-access
        args.server_url,
        requests_list,
        model_id,
        log_path,
        out_dir,
        args.request_timeout_s,
        concurrency_override=args.concurrency_override,
    )
    print(
        "[precompute] baseline done "
        f"success_count={baseline_metrics.get('success_count')} request_count={baseline_metrics.get('request_count')}"
    )
    if baseline_metrics.get("success_count", 0) == 0:
        raise RuntimeError(
            "Baseline run produced 0 successful requests. "
            f"Server may be unhealthy; see baseline log at: {log_path}"
        )

    baseline_metrics_path = out_dir / "baseline_metrics.json"
    baseline_metrics_path.write_text(json.dumps({
        "generated_at_unix": int(time.time()),
        "server_url": args.server_url,
        "base_model": args.base_model,
        "scenario_id": args.scenario_id,
        "requests_sha256": requests_sha,
        "baseline": baseline_metrics,
    }, indent=2), encoding="utf-8")

    registry_path = Path(args.registry) if args.registry else (repo_root / "src" / "eval" / "inference" / "baselines" / "speed" / "default" / f"{model_safe}.json")
    registry = _load_registry(registry_path)
    registry["base_model"] = args.base_model
    registry.setdefault("note", "Precomputed baseline generations + metrics for InferenceBench quality gate.")
    registry.setdefault("scenarios", {})
    registry["scenarios"][args.scenario_id] = {
        "generated_at_unix": int(time.time()),
        "requests_sha256": requests_sha,
        "baseline_log_file": str(log_path.relative_to(repo_root)),
        "baseline_metrics_file": str(baseline_metrics_path.relative_to(repo_root)),
        "choice_accuracy": baseline_metrics.get("choice_accuracy"),
        "model_id": baseline_metrics.get("model_id"),
        "request_count": baseline_metrics.get("request_count"),
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"Wrote baseline log: {log_path}")
    print(f"Updated registry: {registry_path}")


if __name__ == "__main__":
    main()
