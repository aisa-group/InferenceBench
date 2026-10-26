#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import string
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SEED = 248
CHOICE_LETTERS = tuple(string.ascii_uppercase[:10])
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class QualityDatasetSpec:
    name: str
    samples_file: Path
    seed: int
    limit: int
    parse_mode: str


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _default_samples_file(dataset: str, seed: int, limit: int) -> Path:
    return PACKAGE_DIR / "baselines" / "samples" / dataset / f"{seed}_{limit}" / "samples.jsonl"


def _default_baseline_registry(model_id: str = "") -> Path:
    model_id = model_id or os.environ.get("INFERENCE_BENCH_BASE_MODEL", "").strip() or "unknown_model"
    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    backend = os.environ.get("INFERENCE_BENCH_QUALITY_BASELINE_BACKEND", "torch").strip() or "torch"
    backend_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", backend)
    name = f"{model_safe}.json" if backend_safe == "vllm" else f"{model_safe}_{backend_safe}.json"
    model_registry = PACKAGE_DIR / "baselines" / "quality" / name
    results_registry = PACKAGE_DIR / "baselines" / "results" / model_safe / f"quality_{backend_safe}.json"
    if not model_registry.is_file() and results_registry.is_file():
        return results_registry
    return model_registry


def get_quality_specs(model_id: str = "") -> Tuple[List[QualityDatasetSpec], Path]:
    seed = env_int("INFERENCE_BENCH_QUALITY_SEED", env_int("INFERENCE_BENCH_DATASET_SEED", DEFAULT_SEED))
    mmlupro_n = env_int("INFERENCE_BENCH_QUALITY_MMLUPRO_N", 500)
    mmlupro_file = os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE", "").strip()
    mmlupro_path = Path(mmlupro_file) if mmlupro_file else _default_samples_file("mmlu_pro", seed, mmlupro_n)
    registry_raw = os.environ.get("INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY", "").strip()
    registry_path = Path(registry_raw) if registry_raw else _default_baseline_registry(model_id)
    specs = [QualityDatasetSpec(name="mmlu_pro", samples_file=mmlupro_path, seed=seed, limit=mmlupro_n, parse_mode="mcq10")]
    return specs, registry_path


def load_quality_requests(spec: QualityDatasetSpec) -> List[Dict[str, Any]]:
    if spec.limit <= 0:
        raise ValueError("Quality sample count must be positive")
    if not spec.samples_file.is_file():
        raise FileNotFoundError(spec.samples_file)
    with spec.samples_file.open("r", encoding="utf-8") as handle:
        rows = list(islice((json.loads(line) for line in handle if line.strip()), spec.limit))
    requests: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        gold = row.get("gold_answer", row.get("answer"))
        if spec.parse_mode == "mcq10":
            gold = gold.strip().upper() if isinstance(gold, str) else None
            gold = gold if gold in CHOICE_LETTERS else None
        if gold is None:
            continue
        requests.append({
            "messages": messages,
            "sample_id": str(row.get("sample_id", row.get("_id", i))),
            "gold_answer": gold,
            "temperature": 0.0,
            "max_new_tokens": int(row.get("max_new_tokens", 64)),
            "parse_mode": spec.parse_mode,
        })
    if len(requests) != spec.limit:
        raise RuntimeError(f"Expected {spec.limit} valid samples in {spec.samples_file}, found {len(requests)}")
    return requests


def load_quality_baseline_accuracy(registry_path: Path, dataset: str, seed: int, limit: int) -> Tuple[Optional[float], Dict[str, Any]]:
    if not registry_path.is_file():
        return None, {"note": f"baseline registry missing: {registry_path}"}
    try:
        entries = json.loads(registry_path.read_text(encoding="utf-8"))["datasets"][dataset]
    except (ValueError, KeyError, TypeError):
        return None, {"note": f"no baseline entry for dataset={dataset} in {registry_path}"}
    for entry in entries:
        if str(entry.get("seed")) == str(seed) and str(entry.get("n")) == str(limit):
            if _valid_accuracy(entry.get("accuracy")):
                return float(entry["accuracy"]), entry
            return None, {**entry, "note": "baseline accuracy missing"}
    return None, {"note": f"no baseline entry for dataset={dataset} seed={seed} n={limit}"}


def profile_complete(profile: Any) -> bool:
    if not isinstance(profile, dict):
        return False
    total = profile.get("request_count")
    succeeded = profile.get("success_count")
    return (type(total) is int and type(succeeded) is int and 0 < total == succeeded
            and profile.get("failure_count", 0) == 0 and profile.get("empty_output_count", 0) == 0)


def speed_profiles_complete(metrics: Dict[str, Any]) -> bool:
    profiles = metrics.get("profiles")
    return isinstance(profiles, dict) and bool(profiles) and all(map(profile_complete, profiles.values()))


def _valid_accuracy(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def compute_quality_gate(run_results: Dict[str, Any], tau: float, baseline: Dict[str, Any]) -> Dict[str, Any]:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Quality threshold must be finite and positive")
    datasets: Dict[str, Any] = {}
    for name, observed in run_results.items():
        obs_acc = observed["accuracy"]
        base_acc = baseline[name]["accuracy"]
        ratio = None
        passed = False
        note = ""
        if _valid_accuracy(obs_acc) and _valid_accuracy(base_acc) and base_acc > 0:
            ratio = float(obs_acc) / float(base_acc)
            passed = ratio >= tau
        else:
            note = "missing observed or baseline accuracy"
        if not observed["complete"]:
            passed = False
            note = "quality requests did not all succeed with nonempty output"
        datasets[name] = {
            "observed_accuracy": obs_acc,
            "baseline_accuracy": base_acc,
            "ratio": ratio,
            "tau": tau,
            "pass": passed,
            "complete": observed["complete"],
            "note": note,
            "baseline_ref": baseline[name]["ref"],
        }
    return {"tau": tau, "pass": bool(datasets) and all(d["pass"] for d in datasets.values()), "datasets": datasets}
