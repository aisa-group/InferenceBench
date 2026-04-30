#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import runner
from .quality_gate import QualityDatasetSpec, get_quality_specs, load_quality_baseline_accuracy, load_quality_requests


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _accuracy(results: List[Dict[str, Any]]) -> Optional[float]:
    answerable = [r for r in results if r.get("gold_answer") is not None]
    if not answerable:
        return None
    correct = [
        r for r in answerable
        if (r.get("parsed_answer") is not None) and (r.get("parsed_answer") == r.get("gold_answer"))
    ]
    return len(correct) / len(answerable)


def _load_registry(path: Path, base_model: str) -> Dict[str, Any]:
    if not path.exists():
        return {"format_version": 2, "note": "", "base_model": base_model, "datasets": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"format_version": 2, "note": "", "base_model": base_model, "datasets": {}}
    data.setdefault("format_version", 2)
    data.setdefault("base_model", base_model)
    data.setdefault("datasets", {})
    if not isinstance(data.get("datasets"), dict):
        data["datasets"] = {}
    return data


def _default_registry_for_backend(backend: str) -> Path:
    base = _repo_root() / "src" / "eval" / "inference" / "baselines" / "results"
    model_id = os.environ.get("INFERENCE_BENCH_BASE_MODEL", "").strip() or "unknown_model"
    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    backend_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", backend)
    return base / model_safe / f"quality_{backend_safe}.json"


def _default_out_root_for_backend(backend: str) -> Path:
    base = _repo_root() / "src" / "eval" / "inference" / "baselines" / "results"
    model_id = os.environ.get("INFERENCE_BENCH_BASE_MODEL", "").strip() or "unknown_model"
    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    backend_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", backend)
    return base / model_safe / "quality" / backend_safe


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root().resolve()))
    except Exception:
        return str(path.resolve())


def _run_dataset(
    spec: QualityDatasetSpec,
    server_url: str,
    model_id: str,
    request_timeout_s: float,
    concurrency: int,
    out_dir: Path,
) -> Tuple[float, Path, Path]:
    reqs = load_quality_requests(spec)
    profile = {"name": f"quality_{spec.name}", "pattern": "burst", "num_requests": len(reqs), "concurrency": int(concurrency)}
    log_path = out_dir / "baseline_generations.jsonl"
    if log_path.exists():
        log_path.unlink()
    results, _, _ = asyncio.run(runner._run_profile(profile, reqs, server_url, model_id, request_timeout_s, log_path))  # type: ignore[attr-defined]
    acc = _accuracy(results)
    if acc is None:
        raise RuntimeError(f"No accuracy computed for {spec.name} (check gold/parsed answers).")
    metrics_path = out_dir / "baseline_metrics.json"
    _write_json(metrics_path, {
        "generated_at_unix": int(time.time()),
        "dataset": spec.name,
        "seed": spec.seed,
        "n": spec.limit,
        "server_url": server_url,
        "model_id": model_id,
        "accuracy": acc,
        "samples_file": str(spec.samples_file),
        "baseline_log_file": _relative_to_repo(log_path),
    })
    return float(acc), log_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute InferenceBench quality baselines (MMLU-Pro) for a server backend.")
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="")
    parser.add_argument("--backend", type=str, default="vllm", help="Backend id used for output paths (e.g., vllm, sglang, torch).")
    parser.add_argument("--registry", type=str, default="", help="Registry JSON path.")
    parser.add_argument("--out-root", type=str, default="", help="Output root dir.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mmlupro-n", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    specs, _, _ = get_quality_specs()
    mmlupro_spec = specs[0]
    seed = args.seed if args.seed is not None else mmlupro_spec.seed
    mmlupro_n = args.mmlupro_n if args.mmlupro_n is not None else mmlupro_spec.limit

    mmlupro = QualityDatasetSpec(name="mmlu_pro", samples_file=mmlupro_spec.samples_file, seed=seed, limit=mmlupro_n, parse_mode="mcq10")

    registry_path = Path(args.registry) if args.registry else _default_registry_for_backend(args.backend)
    out_root = Path(args.out_root) if args.out_root else _default_out_root_for_backend(args.backend)

    model_id = runner._detect_model_id(args.server_url, args.base_model)  # pylint: disable=protected-access

    if args.concurrency is not None:
        concurrency = int(args.concurrency)
    else:
        raw = os.environ.get("INFERENCE_BENCH_QUALITY_CONCURRENCY", "4").strip()
        concurrency = int(raw) if raw.isdigit() else 4

    registry = _load_registry(registry_path, base_model=args.base_model or model_id)
    registry["backend"] = args.backend
    registry["kind"] = "quality"

    for spec in (mmlupro,):
        out_dir = out_root / spec.name / f"{spec.seed}_{spec.limit}"
        out_dir.mkdir(parents=True, exist_ok=True)
        acc, log_path, metrics_path = _run_dataset(
            spec,
            args.server_url,
            model_id,
            args.request_timeout_s,
            concurrency,
            out_dir,
        )
        entry = {
            "generated_at_unix": int(time.time()),
            "dataset": spec.name,
            "seed": spec.seed,
            "n": spec.limit,
            "accuracy": acc,
            "model_id": model_id,
            "server_url": args.server_url,
            "baseline_log_file": _relative_to_repo(log_path),
            "baseline_metrics_file": _relative_to_repo(metrics_path),
        }
        lst = registry["datasets"].setdefault(spec.name, [])
        if not isinstance(lst, list):
            lst = []
        new_list = [e for e in lst if not (int(e.get("seed", -1)) == spec.seed and int(e.get("n", -1)) == spec.limit)]
        new_list.append(entry)
        registry["datasets"][spec.name] = new_list
        print(f"[baseline] {spec.name} seed={spec.seed} n={spec.limit} acc={acc:.4f} wrote={out_dir}")

    _write_json(registry_path, registry)
    print(f"[baseline] updated registry: {registry_path}")

    for spec in (mmlupro,):
        acc, ref = load_quality_baseline_accuracy(registry_path, spec.name, spec.seed, spec.limit)
        if acc is None:
            raise RuntimeError(f"Registry validation failed for {spec.name}: {ref}")


if __name__ == "__main__":
    main()
