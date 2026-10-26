#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from . import runner
from .precompute_all_baselines import _artifact_path, _repo_root
from .quality_gate import QualityDatasetSpec, _default_samples_file, get_quality_specs, load_quality_baseline_accuracy, load_quality_requests


def _load_registry(path: Path, base_model: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if not isinstance(data, dict):
        return {"format_version": 2, "note": "", "base_model": base_model, "datasets": {}}
    data.setdefault("format_version", 2)
    data.setdefault("base_model", base_model)
    if not isinstance(data.get("datasets"), dict):
        data["datasets"] = {}
    return data


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the MMLU-Pro quality baseline for one server.")
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--backend", type=str, default="vllm", help="Backend name stored in the registry.")
    parser.add_argument("--registry", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mmlupro-n", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    specs, _ = get_quality_specs()
    seed = specs[0].seed if args.seed is None else args.seed
    limit = specs[0].limit if args.mmlupro_n is None else args.mmlupro_n
    samples_override = os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE", "").strip()
    samples_file = Path(samples_override) if samples_override else _default_samples_file("mmlu_pro", seed, limit)
    spec = QualityDatasetSpec(name="mmlu_pro", samples_file=samples_file, seed=seed, limit=limit, parse_mode="mcq10")

    model_id = runner._detect_model_id(args.server_url, args.base_model)
    registry_path = Path(args.registry)
    registry = _load_registry(registry_path, base_model=args.base_model or model_id)
    registry["backend"] = args.backend
    registry["kind"] = "quality"
    repo_root = _repo_root()
    out_dir = Path(args.out_root) / spec.name / f"{spec.seed}_{spec.limit}"
    out_dir.mkdir(parents=True, exist_ok=True)

    reqs = load_quality_requests(spec)
    profile = {"name": f"quality_{spec.name}", "pattern": "burst", "num_requests": len(reqs), "concurrency": args.concurrency}
    log_path = out_dir / "baseline_generations.jsonl"
    log_path.unlink(missing_ok=True)
    results, _, _ = asyncio.run(runner._run_profile(profile, reqs, args.server_url, model_id, args.request_timeout_s, log_path, spec.seed))
    if len(results) != len(reqs) or not all(result.get("success") for result in results):
        raise RuntimeError(f"Quality baseline had failed requests; see {log_path}")
    acc = runner._compute_choice_accuracy(results)
    if acc is None:
        raise RuntimeError(f"No gold answers to score for {spec.name}.")

    metrics_path = out_dir / "baseline_metrics.json"
    _write_json(metrics_path, {
        "generated_at_unix": int(time.time()),
        "dataset": spec.name,
        "seed": spec.seed,
        "n": spec.limit,
        "server_url": args.server_url,
        "model_id": model_id,
        "accuracy": acc,
        "samples_file": str(spec.samples_file),
        "baseline_log_file": _artifact_path(log_path, repo_root),
    })

    entries = registry["datasets"].get(spec.name)
    entries = [
        e for e in (entries if isinstance(entries, list) else [])
        if not (int(e.get("seed", -1)) == spec.seed and int(e.get("n", -1)) == spec.limit)
    ]
    entries.append({
        "generated_at_unix": int(time.time()),
        "dataset": spec.name,
        "seed": spec.seed,
        "n": spec.limit,
        "accuracy": acc,
        "model_id": model_id,
        "server_url": args.server_url,
        "baseline_log_file": _artifact_path(log_path, repo_root),
        "baseline_metrics_file": _artifact_path(metrics_path, repo_root),
    })
    registry["datasets"][spec.name] = entries
    _write_json(registry_path, registry)
    print(f"[baseline] {spec.name} seed={spec.seed} n={spec.limit} acc={acc:.4f} registry={registry_path}")

    validated_acc, ref = load_quality_baseline_accuracy(registry_path, spec.name, spec.seed, spec.limit)
    if validated_acc is None:
        raise RuntimeError(f"Registry validation failed for {spec.name}: {ref}")


if __name__ == "__main__":
    main()
