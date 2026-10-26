#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset

from .quality_gate import CHOICE_LETTERS, _default_samples_file, get_quality_specs


def enable_soft_filelock() -> None:
    try:
        import datasets.builder
        from filelock import SoftFileLock

        datasets.builder.FileLock = SoftFileLock
    except Exception:
        return


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _jsonl_has_n_rows(path: Path, n: int) -> bool:
    if n <= 0 or not path.is_file():
        return False
    try:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                count += 1
                if count > n:
                    return False
        return count == n
    except Exception:
        return False


def _cache_samples(
    out_file: Path,
    seed: int,
    n: int,
    ds: Any,
    format_row: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    label: str,
) -> int:
    if n <= 0:
        raise ValueError("Sample count must be positive")
    rng = random.Random(seed)
    chosen: List[int] = []
    seen = 0
    for idx, row in enumerate(ds):
        if format_row(row) is None:
            continue
        seen += 1
        if len(chosen) < n:
            chosen.append(idx)
            continue
        j = rng.randrange(seen)
        if j < n:
            chosen[j] = idx

    samples: List[Dict[str, Any]] = []
    for idx in chosen:
        formatted = format_row(ds[int(idx)])
        if formatted is None:
            continue
        if not formatted.get("sample_id"):
            formatted["sample_id"] = str(idx)
        samples.append(formatted)
    if len(samples) != n:
        raise RuntimeError(f"Expected {n} {label} samples, got {len(samples)}")
    _write_jsonl(out_file, samples)
    return len(samples)


def _mmlu_pro_gold(row: Dict[str, Any]) -> Optional[str]:
    ans = row.get("answer")
    if isinstance(ans, str):
        ans = ans.strip().upper()
        if ans in CHOICE_LETTERS:
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
    enable_soft_filelock()
    ds = load_dataset(dataset, name=config or None, split=split)
    return _cache_samples(out_file, seed, n, ds, _format_mmlu_pro, f"MMLU-Pro ({dataset} {split})")


def _format_longbench_v2(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    gold = str(row.get("answer", "")).strip().upper()
    return {
        "sample_id": str(row.get("_id", "")),
        "messages": [{"role": "user", "content": "\n".join(parts)}],
        "gold_answer": gold if gold in {"A", "B", "C", "D"} else None,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "parse_mode": "",
    }


def cache_longbench_v2(
    out_file: Path,
    seed: int,
    n: int,
    dataset: str = "THUDM/LongBench-v2",
    split: str = "train",
) -> int:
    enable_soft_filelock()
    ds = load_dataset(dataset, split=split)
    return _cache_samples(out_file, seed, n, ds, _format_longbench_v2, f"LongBench-v2 ({dataset} {split}, max 503)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache MMLU-Pro and LongBench-v2 samples as JSONL files.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mmlupro-n", type=int, default=None)
    parser.add_argument("--mmlupro-dataset", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--mmlupro-config", type=str, default="")
    parser.add_argument("--mmlupro-split", type=str, default="test")
    parser.add_argument("--mmlupro-out", type=str, default="")
    parser.add_argument("--longbench-n", type=int, default=None, help="Number of LongBench-v2 samples (max 503).")
    parser.add_argument("--longbench-dataset", type=str, default="THUDM/LongBench-v2")
    parser.add_argument("--longbench-split", type=str, default="train")
    parser.add_argument("--longbench-out", type=str, default="")
    args = parser.parse_args()

    mmlupro_spec = get_quality_specs()[0][0]
    seed = args.seed if args.seed is not None else mmlupro_spec.seed
    mmlupro_n = args.mmlupro_n if args.mmlupro_n is not None else mmlupro_spec.limit

    output_override = args.mmlupro_out or os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE", "").strip()
    mmlupro_out = Path(output_override) if output_override else _default_samples_file("mmlu_pro", seed, mmlupro_n)

    if _jsonl_has_n_rows(mmlupro_out, mmlupro_n):
        print(f"[cache] mmlu_pro already cached at {mmlupro_out} (n={mmlupro_n})")
    else:
        wrote = cache_mmlu_pro(mmlupro_out, seed=seed, n=mmlupro_n,
                               dataset=args.mmlupro_dataset, config=args.mmlupro_config, split=args.mmlupro_split)
        print(f"[cache] mmlu_pro wrote {wrote} samples to {mmlupro_out}")

    if args.longbench_n is not None:
        longbench_n = args.longbench_n
        if args.longbench_out:
            longbench_out = Path(args.longbench_out)
        else:
            longbench_out = _default_samples_file("longbench_v2", seed, longbench_n)

        if _jsonl_has_n_rows(longbench_out, longbench_n):
            print(f"[cache] longbench_v2 already cached at {longbench_out} (n={longbench_n})")
        else:
            wrote = cache_longbench_v2(longbench_out, seed=seed, n=longbench_n,
                                       dataset=args.longbench_dataset, split=args.longbench_split)
            print(f"[cache] longbench_v2 wrote {wrote} samples to {longbench_out}")


if __name__ == "__main__":
    main()
