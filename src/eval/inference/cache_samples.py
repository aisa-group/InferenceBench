#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

from .quality_gate import get_quality_specs


def _enable_soft_filelock() -> None:
    """Use SoftFileLock for filesystems without flock (e.g., some Lustre/NFS mounts)."""
    try:
        import datasets.builder  # type: ignore
        from filelock import SoftFileLock  # type: ignore

        datasets.builder.FileLock = SoftFileLock  # type: ignore[attr-defined]
    except Exception:
        return


def _reservoir_indices(rows: Iterable[Tuple[int, Dict[str, Any]]], k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    chosen: List[int] = []
    seen = 0
    for idx, row in rows:
        _ = row
        seen += 1
        if len(chosen) < k:
            chosen.append(idx)
            continue
        j = rng.randrange(seen)
        if j < k:
            chosen[j] = idx
    return chosen


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def _jsonl_has_n_rows(path: Path, n: int) -> bool:
    if n <= 0:
        return False
    if not path.exists() or not path.is_file():
        return False
    try:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                # Cheap sanity check: ensure it's JSON.
                json.loads(line)
                count += 1
                if count > n:
                    return False
        return count == n
    except Exception:
        return False


def _mmlu_gold(row: Dict[str, Any]) -> Optional[str]:
    # Common encodings:
    # - answer: 0..3
    # - answer: "A"/"B"/"C"/"D"
    ans = row.get("answer")
    if isinstance(ans, int) and 0 <= ans <= 3:
        return ["A", "B", "C", "D"][ans]
    if isinstance(ans, str):
        ans = ans.strip().upper()
        if ans in {"A", "B", "C", "D"}:
            return ans
    return None


def _mmlu_choices(row: Dict[str, Any]) -> Optional[List[str]]:
    choices = row.get("choices")
    if isinstance(choices, list) and len(choices) == 4 and all(isinstance(c, str) for c in choices):
        return [str(c) for c in choices]
    # Sometimes stored as separate fields.
    maybe = []
    for key in ("A", "B", "C", "D"):
        if key in row and isinstance(row.get(key), str):
            maybe.append(row[key])
    if len(maybe) == 4:
        return maybe
    return None


def _format_mmlu(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question")
    if not isinstance(q, str) or not q.strip():
        return None
    choices = _mmlu_choices(row)
    gold = _mmlu_gold(row)
    if choices is None or gold is None:
        return None
    prompt = (
        f"Question: {q.strip()}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        "Answer with A, B, C, or D. Start your answer with the letter."
    )
    return {
        "sample_id": row.get("id") or row.get("_id") or "",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gold_answer": gold,
        "max_new_tokens": 16,
        "temperature": 0.0,
        "parse_mode": "mcq",
    }


def _gsm8k_gold(answer: Any) -> Optional[str]:
    if not isinstance(answer, str) or not answer.strip():
        return None
    # GSM8K canonical format: "... #### 42"
    if "####" in answer:
        tail = answer.split("####", 1)[1].strip()
        tail = tail.replace(",", "")
        m = re.match(r"^[-+]?\d+", tail)
        if m:
            return m.group(0)
    # Fallback: last integer-like token.
    nums = re.findall(r"[-+]?\d[\d,]*", answer)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def _format_gsm8k(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question")
    a = row.get("answer")
    if not isinstance(q, str) or not q.strip():
        return None
    gold = _gsm8k_gold(a)
    if gold is None:
        return None
    prompt = (
        f"{q.strip()}\n\n"
        "Solve the problem. Provide the final answer as an integer.\n"
        "Answer:"
    )
    return {
        "sample_id": row.get("id") or row.get("_id") or "",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gold_answer": gold,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "parse_mode": "gsm8k",
    }


def cache_mmlu(out_file: Path, seed: int, n: int, dataset: str, config: str, split: str) -> int:
    _enable_soft_filelock()
    ds = load_dataset(dataset, name=config or None, split=split, trust_remote_code=True)
    def rows() -> Iterable[Tuple[int, Dict[str, Any]]]:
        for i, row in enumerate(ds):
            if _format_mmlu(row) is not None:
                yield i, row
    indices = _reservoir_indices(rows(), k=n, seed=seed)
    samples: List[Dict[str, Any]] = []
    for idx in indices:
        row = ds[int(idx)]
        formatted = _format_mmlu(row)
        if formatted is None:
            continue
        if not formatted.get("sample_id"):
            formatted["sample_id"] = str(idx)
        samples.append(formatted)
    if len(samples) != n:
        raise RuntimeError(f"Expected {n} MMLU samples, got {len(samples)} (dataset={dataset} config={config} split={split})")
    _write_jsonl(out_file, samples)
    return len(samples)


def _mmlu_pro_gold(row: Dict[str, Any]) -> Optional[str]:
    ans = row.get("answer")
    if isinstance(ans, str):
        ans = ans.strip().upper()
        if ans in {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}:
            return ans
    ans_idx = row.get("answer_index")
    if isinstance(ans_idx, int) and 0 <= ans_idx <= 9:
        return chr(ord("A") + ans_idx)
    return None


def _format_mmlu_pro(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = row.get("question")
    if not isinstance(q, str) or not q.strip():
        return None
    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return None
    gold = _mmlu_pro_gold(row)
    if gold is None:
        return None
    option_lines = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
    prompt = (
        f"Question: {q.strip()}\n"
        f"{option_lines}\n\n"
        f"Answer with a single letter ({', '.join(chr(ord('A') + i) for i in range(len(options)))}). "
        "Start your answer with the letter."
    )
    return {
        "sample_id": str(row.get("question_id", row.get("_id", ""))),
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "gold_answer": gold,
        "max_new_tokens": 2048,
        "temperature": 0.0,
        "parse_mode": "mcq10",
    }


def cache_mmlu_pro(out_file: Path, seed: int, n: int, dataset: str = "TIGER-Lab/MMLU-Pro",
                   config: str = "", split: str = "test") -> int:
    _enable_soft_filelock()
    ds = load_dataset(dataset, name=config or None, split=split, trust_remote_code=True)
    def rows() -> Iterable[Tuple[int, Dict[str, Any]]]:
        for i, row in enumerate(ds):
            if _format_mmlu_pro(row) is not None:
                yield i, row
    indices = _reservoir_indices(rows(), k=n, seed=seed)
    samples: List[Dict[str, Any]] = []
    for idx in indices:
        row = ds[int(idx)]
        formatted = _format_mmlu_pro(row)
        if formatted is None:
            continue
        if not formatted.get("sample_id"):
            formatted["sample_id"] = str(idx)
        samples.append(formatted)
    if len(samples) != n:
        raise RuntimeError(f"Expected {n} MMLU-Pro samples, got {len(samples)} (dataset={dataset} split={split})")
    _write_jsonl(out_file, samples)
    return len(samples)


def cache_gsm8k(out_file: Path, seed: int, n: int, dataset: str, config: str, split: str) -> int:
    _enable_soft_filelock()
    ds = load_dataset(dataset, name=config or None, split=split, trust_remote_code=True)
    def rows() -> Iterable[Tuple[int, Dict[str, Any]]]:
        for i, row in enumerate(ds):
            if _format_gsm8k(row) is not None:
                yield i, row
    indices = _reservoir_indices(rows(), k=n, seed=seed)
    samples: List[Dict[str, Any]] = []
    for idx in indices:
        row = ds[int(idx)]
        formatted = _format_gsm8k(row)
        if formatted is None:
            continue
        if not formatted.get("sample_id"):
            formatted["sample_id"] = str(idx)
        samples.append(formatted)
    if len(samples) != n:
        raise RuntimeError(f"Expected {n} GSM8K samples, got {len(samples)} (dataset={dataset} config={config} split={split})")
    _write_jsonl(out_file, samples)
    return len(samples)


# ---------------------------------------------------------------------------
# LongBench-v2 (speed-eval dataset replacing synthetic random-token requests)
# ---------------------------------------------------------------------------

def _format_longbench_v2(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format a LongBench-v2 row as a speed-eval request.

    The full prompt (context + question + choices) is stored without truncation;
    runner.py applies scenario-specific input_len truncation at eval time.
    max_new_tokens is a placeholder — runner.py overrides it with output_len.
    """
    context = row.get("context", "")
    question = row.get("question", "")
    if not isinstance(context, str) or not isinstance(question, str):
        return None
    if not context.strip() and not question.strip():
        return None

    parts = []
    if context.strip():
        parts.append(context.strip())
    if question.strip():
        parts.append(f"Question: {question.strip()}")
    for letter in ("A", "B", "C", "D"):
        val = row.get(f"choice_{letter}", "")
        if isinstance(val, str) and val.strip():
            parts.append(f"{letter}) {val.strip()}")

    prompt = "\n".join(parts)

    gold = str(row.get("answer", "")).strip().upper()
    return {
        "sample_id": str(row.get("_id", "")),
        "messages": [{"role": "user", "content": prompt}],
        # gold_answer is stored but not used for speed eval (no quality gate)
        "gold_answer": gold if gold in {"A", "B", "C", "D"} else None,
        "max_new_tokens": 512,  # placeholder; overridden by scenario output_len at runtime
        "temperature": 0.0,
        "parse_mode": "",  # no quality scoring for speed eval
    }


def cache_longbench_v2(
    out_file: Path,
    seed: int,
    n: int,
    dataset: str = "THUDM/LongBench-v2",
    split: str = "train",
) -> int:
    """Cache n LongBench-v2 samples selected by reservoir sampling with the given seed."""
    _enable_soft_filelock()
    ds = load_dataset(dataset, split=split, trust_remote_code=True)

    def rows() -> Iterable[Tuple[int, Dict[str, Any]]]:
        for i, row in enumerate(ds):
            if _format_longbench_v2(row) is not None:
                yield i, row

    indices = _reservoir_indices(rows(), k=n, seed=seed)
    samples: List[Dict[str, Any]] = []
    for idx in indices:
        row = ds[int(idx)]
        formatted = _format_longbench_v2(row)
        if formatted is None:
            continue
        if not formatted.get("sample_id"):
            formatted["sample_id"] = str(idx)
        samples.append(formatted)
    if len(samples) != n:
        raise RuntimeError(
            f"Expected {n} LongBench-v2 samples, got {len(samples)} "
            f"(dataset={dataset} split={split}; dataset has 503 total entries)"
        )
    _write_jsonl(out_file, samples)
    return len(samples)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-cache InferenceBench dataset samples (MMLU-Pro, LongBench-v2) into JSONL files. "
            "Run once per seed before launching experiments."
        )
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mmlupro-n", type=int, default=None)
    parser.add_argument("--mmlupro-dataset", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--mmlupro-config", type=str, default="")
    parser.add_argument("--mmlupro-split", type=str, default="test")
    parser.add_argument("--mmlupro-out", type=str, default="")
    parser.add_argument("--longbench-n", type=int, default=None,
                        help="Number of LongBench-v2 samples to cache (max 503)")
    parser.add_argument("--longbench-dataset", type=str, default="THUDM/LongBench-v2")
    parser.add_argument("--longbench-split", type=str, default="train")
    parser.add_argument("--longbench-out", type=str, default="",
                        help="Output path for LongBench-v2 cache (auto-derived from seed+n if omitted)")
    # Legacy args (kept for backward compat with existing precompute scripts)
    parser.add_argument("--mmlu-n", type=int, default=None)
    parser.add_argument("--gsm8k-n", type=int, default=None)
    args = parser.parse_args()

    specs, _, _ = get_quality_specs()
    mmlupro_spec = specs[0]
    seed = args.seed if args.seed is not None else mmlupro_spec.seed
    mmlupro_n = args.mmlupro_n if args.mmlupro_n is not None else mmlupro_spec.limit

    mmlupro_out = Path(args.mmlupro_out) if args.mmlupro_out else mmlupro_spec.samples_file

    if _jsonl_has_n_rows(mmlupro_out, mmlupro_n):
        print(f"[cache] mmlu_pro samples already cached at {mmlupro_out} (n={mmlupro_n}); skipping")
    else:
        wrote = cache_mmlu_pro(mmlupro_out, seed=seed, n=mmlupro_n,
                               dataset=args.mmlupro_dataset, config=args.mmlupro_config, split=args.mmlupro_split)
        print(f"[cache] mmlu_pro wrote {wrote} samples to {mmlupro_out}")

    if args.longbench_n is not None:
        import os
        from .quality_gate import _repo_root
        longbench_n = args.longbench_n
        if args.longbench_out:
            longbench_out = Path(args.longbench_out)
        else:
            base = _repo_root()
            longbench_out = base / "baselines" / "samples" / "longbench_v2" / f"{seed}_{longbench_n}" / "samples.jsonl"

        if _jsonl_has_n_rows(longbench_out, longbench_n):
            print(f"[cache] longbench_v2 samples already cached at {longbench_out} (n={longbench_n}); skipping")
        else:
            wrote = cache_longbench_v2(longbench_out, seed=seed, n=longbench_n,
                                       dataset=args.longbench_dataset, split=args.longbench_split)
            print(f"[cache] longbench_v2 wrote {wrote} samples to {longbench_out}")


if __name__ == "__main__":
    main()
