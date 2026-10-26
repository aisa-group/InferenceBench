#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from . import baseline_eval
from .precompute_all_baselines import _artifact_path, _repo_root, _safe_name
from .quality_gate import DEFAULT_SEED
from .runner import (
    _get_model_max_len,
    _get_tokenizer,
    _prepare_requests,
    _write_requests_used,
    load_scenario_config,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"format_version": 1, "scenarios": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if not isinstance(data.get("scenarios"), dict):
        data["scenarios"] = {}
    data.setdefault("format_version", 1)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the speed baseline for one scenario.")
    parser.add_argument("--scenario-id", type=str, required=True, help="Directory name under src/eval/tasks/.")
    parser.add_argument("--base-model", type=str, default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", ""))
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="")
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency-override", type=int, default=None)
    parser.add_argument("--registry", type=str, default="")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", DEFAULT_SEED)))
    args = parser.parse_args()

    repo_root = _repo_root()
    scenario_src = repo_root / "src" / "eval" / "tasks" / args.scenario_id
    if not scenario_src.is_dir():
        raise FileNotFoundError(f"Scenario not found: {scenario_src}")

    model_safe = _safe_name(args.base_model or "unknown_model")
    default_root = repo_root / "src" / "eval" / "inference" / "baselines" / "speed" / "default"
    out_root = Path(args.out_root) if args.out_root else default_root
    out_dir = out_root / args.scenario_id / model_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("scenario.json", "mission.txt", "benchmark.txt"):
        if (scenario_src / name).exists():
            shutil.copyfile(scenario_src / name, out_dir / name)

    requests_path = out_dir / "requests.jsonl"
    config = load_scenario_config(scenario_src)
    config["dataset_seed"] = args.seed
    print(f"[precompute] scenario={args.scenario_id} dataset_seed={args.seed}")
    tokenizer = _get_tokenizer(args.base_model) if args.base_model else None
    max_model_len = _get_model_max_len(args.base_model) if args.base_model else None
    _write_requests_used(requests_path, _prepare_requests(config, None, tokenizer, max_model_len))

    requests_sha = _sha256(requests_path)
    model_id = baseline_eval._detect_model_id(args.server_url, args.base_model)
    requests_list = baseline_eval._load_requests(requests_path)
    print(f"[precompute] model_id={model_id} requests={len(requests_list)} server_url={args.server_url}")

    log_path = out_dir / "baseline_generations.jsonl"
    log_path.unlink(missing_ok=True)
    baseline_metrics = baseline_eval._run_baseline(
        args.server_url,
        requests_list,
        model_id,
        log_path,
        out_dir,
        args.request_timeout_s,
        concurrency_override=args.concurrency_override,
        seed=args.seed,
    )
    print(
        "[precompute] baseline done "
        f"success_count={baseline_metrics.get('success_count')} request_count={baseline_metrics.get('request_count')}"
    )
    if baseline_metrics.get("success_count", 0) != baseline_metrics.get("request_count", 0) or not baseline_metrics.get("request_count"):
        raise RuntimeError(f"Baseline run had failed requests; see {log_path}")

    baseline_metrics_path = out_dir / "baseline_metrics.json"
    baseline_metrics_path.write_text(json.dumps({
        "generated_at_unix": int(time.time()),
        "server_url": args.server_url,
        "base_model": args.base_model,
        "scenario_id": args.scenario_id,
        "requests_sha256": requests_sha,
        "baseline": baseline_metrics,
    }, indent=2), encoding="utf-8")

    registry_path = Path(args.registry) if args.registry else default_root / f"{model_safe}.json"
    registry = _load_registry(registry_path)
    registry["base_model"] = args.base_model
    registry.setdefault("note", "Precomputed baseline generations + metrics for InferenceBench quality gate.")
    registry["scenarios"][args.scenario_id] = {
        "generated_at_unix": int(time.time()),
        "requests_sha256": requests_sha,
        "baseline_log_file": _artifact_path(log_path, repo_root),
        "baseline_metrics_file": _artifact_path(baseline_metrics_path, repo_root),
        "choice_accuracy": baseline_metrics.get("choice_accuracy"),
        "model_id": baseline_metrics.get("model_id"),
        "request_count": baseline_metrics.get("request_count"),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"[precompute] wrote {log_path} and {registry_path}")


if __name__ == "__main__":
    main()
