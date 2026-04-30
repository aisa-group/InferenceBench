#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class QualityDatasetSpec:
    name: str  # "mmlu_pro" | "mmlu" | "gsm8k"
    samples_file: Path
    seed: int
    limit: int
    parse_mode: str  # "mcq" | "mcq10" | "gsm8k"


def _repo_root() -> Path:
    # Keep paths relative to the inference package directory so this works both
    # in-repo and in the copied task bundle under /home/agent/{inference,task/inference}.
    return Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_quality_seed() -> int:
    # Prefer a dedicated quality seed; fall back to dataset seed; finally a stable constant.
    seed = os.environ.get("INFERENCE_BENCH_QUALITY_SEED", "").strip()
    if seed.isdigit():
        return int(seed)
    seed = os.environ.get("INFERENCE_BENCH_DATASET_SEED", "").strip()
    if seed.isdigit():
        return int(seed)
    return 248


def _default_samples_file(dataset: str, seed: int, limit: int) -> Path:
    # Convention: samples are pre-cached and versioned by seed+N.
    # Requested layout:
    #   src/eval/inference/baselines/samples/{dataset_name}/{seed}_{N}/samples.jsonl
    base = _repo_root()
    return base / "baselines" / "samples" / dataset / f"{seed}_{limit}" / "samples.jsonl"


def _resolve_default_samples_file(dataset: str, seed: int, limit: int, path: Path) -> Path:
    if path.is_file():
        return path
    base = _repo_root() / "baselines" / "samples" / dataset
    if not base.is_dir():
        return path

    best: Optional[Path] = None
    best_n: Optional[int] = None
    for cand in base.glob(f"{seed}_*/samples.jsonl"):
        parent = cand.parent.name
        if not parent.startswith(f"{seed}_"):
            continue
        try:
            n = int(parent.split("_", 1)[1])
        except Exception:
            continue
        if n < limit:
            continue
        if best_n is None or n < best_n:
            best_n = n
            best = cand
    return best if best is not None else path


def _default_baseline_registry() -> Path:
    # Prefer the registry path produced by the current precompute workflow
    # (src/eval/inference/baselines/quality/*), while still supporting the newer
    # per-model layout under baselines/results when present.
    base = _repo_root()
    model_id = os.environ.get("INFERENCE_BENCH_BASE_MODEL", "").strip() or "unknown_model"
    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)
    backend = os.environ.get("INFERENCE_BENCH_QUALITY_BASELINE_BACKEND", "torch").strip() or "torch"
    backend_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", backend)

    # Per-model registry (naming convention from precompute_all_baselines.py).
    if backend_safe == "vllm":
        model_registry = base / "baselines" / "quality" / f"{model_safe}.json"
    else:
        model_registry = base / "baselines" / "quality" / f"{model_safe}_{backend_safe}.json"

    # Alternative per-model path under baselines/results/.
    results_registry = base / "baselines" / "results" / model_safe / f"quality_{backend_safe}.json"

    for candidate in (model_registry, results_registry):
        if candidate.is_file():
            return candidate

    return model_registry


def get_quality_specs() -> Tuple[List[QualityDatasetSpec], float, Path]:
    tau_raw = os.environ.get("INFERENCE_BENCH_QUALITY_TAU", "0.95").strip()
    try:
        tau = float(tau_raw)
    except ValueError:
        tau = 0.95

    seed = _default_quality_seed()
    mmlupro_n = _env_int("INFERENCE_BENCH_QUALITY_MMLUPRO_N", 500)

    mmlupro_file = os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE", "").strip()
    mmlupro_path = Path(mmlupro_file) if mmlupro_file else _resolve_default_samples_file(
        "mmlu_pro", seed, mmlupro_n, _default_samples_file("mmlu_pro", seed, mmlupro_n)
    )

    registry_raw = os.environ.get("INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY", "").strip()
    registry_path = Path(registry_raw) if registry_raw else _default_baseline_registry()

    specs = [
        QualityDatasetSpec(name="mmlu_pro", samples_file=mmlupro_path, seed=seed, limit=mmlupro_n, parse_mode="mcq10"),
    ]

    return specs, tau, registry_path


def _load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer explicit "Answer:" spans.
    m = re.search(r"(?i)\banswer\s*[:\-]\s*([-+]?\d[\d,]*)\b", text)
    if m:
        return m.group(1).replace(",", "")
    # Otherwise take the last integer-like token.
    all_nums = re.findall(r"[-+]?\d[\d,]*", text)
    if not all_nums:
        return None
    return all_nums[-1].replace(",", "")


def _normalize_mcq_gold(gold: Any) -> Optional[str]:
    if not isinstance(gold, str):
        return None
    gold = gold.strip().upper()
    return gold if gold in {"A", "B", "C", "D"} else None


def _normalize_mcq10_gold(gold: Any) -> Optional[str]:
    """Normalize MMLU-Pro gold answers (10 options: A-J)."""
    if not isinstance(gold, str):
        return None
    gold = gold.strip().upper()
    return gold if gold in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"} else None


def _normalize_gsm8k_gold(gold: Any) -> Optional[str]:
    if not isinstance(gold, str):
        return None
    gold = gold.strip()
    gold = gold.replace(",", "")
    return gold if re.fullmatch(r"[-+]?\d+", gold) else None


def _load_quality_requests(spec: QualityDatasetSpec) -> List[Dict[str, Any]]:
    raw = _load_jsonl(spec.samples_file, limit=spec.limit)
    requests: List[Dict[str, Any]] = []
    for i, row in enumerate(raw):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        gold = row.get("gold_answer", row.get("answer"))
        if spec.parse_mode == "mcq":
            gold = _normalize_mcq_gold(gold)
        elif spec.parse_mode == "mcq10":
            gold = _normalize_mcq10_gold(gold)
        elif spec.parse_mode == "gsm8k":
            gold = _normalize_gsm8k_gold(gold)
        if gold is None:
            continue
        sample_id = str(row.get("sample_id", row.get("_id", i)))
        record: Dict[str, Any] = {
            "messages": messages,
            "sample_id": sample_id,
            "gold_answer": gold,
            "temperature": float(row.get("temperature", 0.0)),
            "max_new_tokens": int(row.get("max_new_tokens", 64)),
            "require_json": False,
            "parse_mode": spec.parse_mode,
        }
        requests.append(record)

    if not requests:
        raise RuntimeError(f"No valid samples found in {spec.samples_file}")
    return requests


def load_quality_requests(spec: QualityDatasetSpec) -> List[Dict[str, Any]]:
    return _load_quality_requests(spec)


def load_quality_baseline_accuracy(registry_path: Path, dataset: str, seed: int, limit: int) -> Tuple[Optional[float], Dict[str, Any]]:
    if not registry_path.is_file():
        return None, {"note": f"baseline registry missing: {registry_path}"}
    try:
        reg = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, {"note": f"failed to read baseline registry: {exc}"}

    entries = (reg.get("datasets") or {}).get(dataset) or []
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        entries = []

    match: Optional[Dict[str, Any]] = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("seed", -1)) == int(seed) and int(entry.get("n", -1)) == int(limit):
            match = entry
            break

    if match is None:
        best: Optional[Dict[str, Any]] = None
        best_n: Optional[int] = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                if int(entry.get("seed", -1)) != int(seed):
                    continue
                n = int(entry.get("n", -1))
            except Exception:
                continue
            if n < int(limit):
                continue
            if best_n is None or n < best_n:
                best_n = n
                best = entry
        if best is None:
            return None, {"note": f"no baseline entry for dataset={dataset} seed={seed} n>={limit}"}
        match = dict(best)
        match["note"] = f"no exact baseline entry for dataset={dataset} seed={seed} n={limit}; using n={best_n}"

    accuracy = match.get("accuracy")
    if isinstance(accuracy, (int, float)):
        return float(accuracy), match

    metrics_file = match.get("baseline_metrics_file") or ""
    if metrics_file:
        base = _repo_root()
        metrics_path = Path(metrics_file)
        if not metrics_path.is_absolute():
            # Support either paths relative to the inference package dir, or repo-style paths.
            candidate1 = (base / metrics_file).resolve()
            candidate2 = (base.parents[3] / metrics_file).resolve() if len(base.parents) >= 4 else candidate1
            metrics_path = candidate1 if candidate1.exists() else candidate2
        if metrics_path.is_file():
            try:
                blob = json.loads(metrics_path.read_text(encoding="utf-8"))
                acc = (((blob.get("baseline") or {}).get("accuracy")) or blob.get("accuracy"))
                if isinstance(acc, (int, float)):
                    return float(acc), match
            except Exception:
                pass

    return None, {**match, "note": "baseline accuracy missing"}


def compute_quality_gate(
    run_results: Dict[str, Any],
    tau: float,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    # run_results: dataset -> {"accuracy": float, "count": int}
    datasets: Dict[str, Any] = {}
    overall_pass = True
    for name, observed in run_results.items():
        obs_acc = observed.get("accuracy")
        base_acc = (baseline.get(name) or {}).get("accuracy")
        ratio = None
        passed = False
        note = ""
        if isinstance(obs_acc, (int, float)) and isinstance(base_acc, (int, float)) and base_acc > 0:
            ratio = float(obs_acc) / float(base_acc)
            passed = ratio >= tau
        else:
            note = "missing observed or baseline accuracy"
        overall_pass = overall_pass and passed
        datasets[name] = {
            "observed_accuracy": obs_acc,
            "baseline_accuracy": base_acc,
            "ratio": ratio,
            "tau": tau,
            "pass": passed,
            "n": observed.get("count"),
            "note": note or (baseline.get(name) or {}).get("note", ""),
            "baseline_ref": (baseline.get(name) or {}).get("ref", {}),
        }

    return {
        "tau": tau,
        "pass": overall_pass,
        "datasets": datasets,
    }
