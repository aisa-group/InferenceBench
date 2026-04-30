#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests

# Lazy-import transformers to avoid triggering the accelerate circular import
# (accelerate.big_modeling -> hooks -> utils.bnb -> big_modeling) at module-load
# time inside containers with mismatched package paths.
AutoConfig = None   # type: ignore[assignment]
AutoTokenizer = None  # type: ignore[assignment]


def _ensure_transformers() -> None:
    global AutoConfig, AutoTokenizer
    if AutoConfig is None:
        from transformers import AutoConfig as _AC, AutoTokenizer as _AT
        AutoConfig = _AC
        AutoTokenizer = _AT

from . import quality_gate


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_scenario_config(task_dir: Path) -> Dict[str, Any]:
    config_path = task_dir / "scenario.json"
    config = json.loads(_read_text(config_path))
    mission_path = task_dir / "mission.txt"
    if mission_path.exists():
        config["mission"] = _read_text(mission_path)
    if "dataset_seed" not in config or config.get("dataset_seed") in (None, ""):
        seed_env = os.environ.get("INFERENCE_BENCH_DATASET_SEED", "").strip()
        try:
            config["dataset_seed"] = int(seed_env) if seed_env else 248
        except ValueError:
            config["dataset_seed"] = 248
    config["task_dir"] = str(task_dir)
    return config


def _count_chat_tokens(messages: List[Dict[str, str]], tokenizer: AutoTokenizer) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            return len(tokens)
        except Exception:
            pass
    joined = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in messages)
    return len(tokenizer.encode(joined, add_special_tokens=False))


def _truncate_messages(
    messages: List[Dict[str, str]],
    tokenizer: Optional[AutoTokenizer],
    max_input_tokens: Optional[int],
    *,
    keep: str = "tail",
) -> List[Dict[str, str]]:
    if tokenizer is None or max_input_tokens is None:
        return messages
    if max_input_tokens <= 0:
        max_input_tokens = 0

    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return messages

    original = messages[user_idx].get("content", "")
    messages_without_user = [dict(m) for m in messages]
    messages_without_user[user_idx]["content"] = ""
    base_tokens = _count_chat_tokens(messages_without_user, tokenizer)
    allowed_user_tokens = max_input_tokens - base_tokens
    if allowed_user_tokens <= 0:
        messages[user_idx]["content"] = ""
        return messages

    user_tokens = tokenizer.encode(original, add_special_tokens=False)
    if len(user_tokens) <= allowed_user_tokens:
        return messages

    if keep == "head":
        truncated_tokens = user_tokens[:allowed_user_tokens]
    else:
        # Preserve the old default for quality prompts, where choices/answer cue
        # are usually near the end of the user message.
        truncated_tokens = user_tokens[-allowed_user_tokens:]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    messages[user_idx]["content"] = truncated_text
    return messages


def _get_input_token_margin() -> int:
    raw = os.environ.get("INFERENCE_BENCH_INPUT_TOKEN_MARGIN", "16").strip()
    try:
        margin = int(raw)
    except ValueError:
        margin = 16
    return max(0, margin)


def _compute_max_input_tokens(max_model_len: Optional[int], max_new_tokens: int) -> Optional[int]:
    if max_model_len is None:
        return None
    margin = _get_input_token_margin()
    # Keep a fixed guard band to absorb tokenizer/chat-template mismatch at the boundary.
    return max(0, int(max_model_len) - int(max_new_tokens) - 1 - int(margin))


def _get_model_max_len(model_id: str) -> Optional[int]:
    _ensure_transformers()
    env_value = os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or None
    try:
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True, cache_dir=cache_dir)
        max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "model_max_length", None)
        if isinstance(max_len, int) and max_len > 0:
            return max_len
    except Exception:
        allow_download = os.environ.get("INFERENCE_BENCH_ALLOW_HF_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}
        if not allow_download:
            return None
        try:
            cfg = AutoConfig.from_pretrained(model_id, local_files_only=False, cache_dir=cache_dir)
            max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "model_max_length", None)
            if isinstance(max_len, int) and max_len > 0:
                return max_len
        except Exception:
            return None
    return None


def _get_tokenizer(model_id: str):
    _ensure_transformers()
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or None
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, cache_dir=cache_dir)
    except Exception:
        allow_download = os.environ.get("INFERENCE_BENCH_ALLOW_HF_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}
        if not allow_download:
            return None
        try:
            return AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=False, cache_dir=cache_dir)
        except Exception:
            return None


def _messages_to_prompt_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n\n".join(parts)


def _request_content_hash(req: Dict[str, Any]) -> str:
    payload = {
        "messages": req.get("messages"),
        "max_new_tokens": req.get("max_new_tokens"),
        "temperature": req.get("temperature"),
        "require_json": req.get("require_json", False),
        "ignore_eos": req.get("ignore_eos", False),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_requests_jsonl(
    path: Path,
    config: Dict[str, Any],
    limit: Optional[int],
    tokenizer: Optional[AutoTokenizer],
    max_model_len: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    max_new_tokens_default = int(config.get("max_new_tokens", 256))
    temperature_default = float(config.get("temperature", 0.2))
    require_json_default = bool(config.get("require_json", False))
    dataset_overflow = (config.get("dataset_overflow") or os.environ.get("INFERENCE_BENCH_DATASET_OVERFLOW") or "filter").strip().lower()
    hard_truncate = os.environ.get("INFERENCE_BENCH_DATASET_HARD_TRUNCATE", "1").strip() != "0"
    synthetic_cfg = config.get("synthetic", {})
    ignore_eos = synthetic_cfg.get("ignore_eos", False)

    requests_list: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(requests_list) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages") or []
            if not isinstance(messages, list) or not messages:
                continue

            gold = record.get("gold_answer", record.get("answer"))
            if isinstance(gold, str):
                gold = gold.strip().upper()

            parse_mode = record.get("parse_mode") or record.get("answer_format") or ""
            if not parse_mode and isinstance(gold, str):
                if gold in {"A", "B", "C", "D"}:
                    parse_mode = "mcq"
                elif re.fullmatch(r"[-+]?\d[\d,]*", gold):
                    parse_mode = "gsm8k"

            max_new_tokens = int(record.get("max_new_tokens", max_new_tokens_default))
            temperature = float(record.get("temperature", temperature_default))
            require_json = bool(record.get("require_json", require_json_default))
            record_ignore_eos = bool(record.get("ignore_eos", ignore_eos))
            sample_id = record.get("sample_id") or record.get("_id") or ""
            sample_id = str(sample_id) if sample_id is not None else ""
            if tokenizer is not None and max_model_len is not None:
                max_input_tokens = _compute_max_input_tokens(max_model_len, max_new_tokens)
                if dataset_overflow == "truncate" or hard_truncate:
                    messages = _truncate_messages(messages, tokenizer, max_input_tokens)

            req: Dict[str, Any] = {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "require_json": require_json,
                "gold_answer": gold,
                "sample_id": sample_id,
                "parse_mode": parse_mode,
            }
            if record_ignore_eos:
                req["ignore_eos"] = True
            for meta_key in (
                "target_input_token_count",
                "input_token_count",
                "source_prompt_token_count",
                "sampling_range_ratio",
            ):
                if meta_key in record:
                    req[meta_key] = record.get(meta_key)
            req["content_hash"] = record.get("content_hash") or _request_content_hash(req)
            requests_list.append(req)

            samples.append({
                "sample_id": sample_id,
                "messages": messages,
                "prompt_text": _messages_to_prompt_text(messages),
                "answer": gold,
                "content_hash": req["content_hash"],
                "target_input_token_count": req.get("target_input_token_count"),
                "input_token_count": req.get("input_token_count"),
                "source_prompt_token_count": req.get("source_prompt_token_count"),
            })

    return requests_list, samples


def _default_requests_file(task_dir: Path) -> Optional[Path]:
    scenario = os.environ.get("INFERENCE_BENCH_SCENARIO", "").strip()
    if not scenario:
        return None

    model_id = os.environ.get("INFERENCE_BENCH_BASE_MODEL", "").strip()
    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id) if model_id else ""

    # Search both task-bundle and in-repo inference dirs.
    # Preferred layout:
    #   baselines/speed/{backend}/{scenario}/{model_safe}/requests.jsonl
    # Also support scenario-level requests and legacy baseline layout.
    roots = [
        task_dir / "inference",
        Path(__file__).resolve().parent,
    ]
    candidates: List[Path] = []
    for root in roots:
        for backend in ("torch", "vllm"):
            if model_safe:
                candidates.append(root / "baselines" / "speed" / backend / scenario / model_safe / "requests.jsonl")
            candidates.append(root / "baselines" / "speed" / backend / scenario / "requests.jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_longbench_samples_file(
    inference_dir: Path,
    seed: int,
    min_n: int,
    *,
    allow_smaller: bool = True,
) -> Optional[Path]:
    """Find a pre-cached LongBench-v2 samples file for the given seed with at least min_n rows.

    Searches baselines/samples/longbench_v2/{seed}_{N}/samples.jsonl and returns
    the file whose N is the smallest value >= min_n.  Optionally falls back to
    the largest available N for that seed if none meets the minimum.
    """
    base = inference_dir / "baselines" / "samples" / "longbench_v2"
    if not base.is_dir():
        return None

    candidates: List[Tuple[int, Path]] = []
    for cand in base.glob(f"{seed}_*/samples.jsonl"):
        parent = cand.parent.name
        if not parent.startswith(f"{seed}_"):
            continue
        try:
            n = int(parent.split("_", 1)[1])
        except Exception:
            continue
        candidates.append((n, cand))

    if not candidates:
        return None

    # Prefer smallest N that still satisfies min_n (avoid loading unnecessarily large files).
    sufficient = [(n, p) for n, p in candidates if n >= min_n]
    if sufficient:
        return min(sufficient, key=lambda x: x[0])[1]
    if allow_smaller:
        return max(candidates, key=lambda x: x[0])[1]
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _range_ratio(value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        ratio = 0.8
    if ratio <= 0 or ratio > 1:
        ratio = 0.8
    return ratio


def _prepare_requests(
    config: Dict[str, Any],
    limit: Optional[int],
    tokenizer: Optional[AutoTokenizer],
    max_model_len: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load LongBench-v2 samples for speed evaluation.

    Uses scenario config's synthetic.input_len / synthetic.output_len to control
    request sizes:
    - input_len: each request draws a target input length uniformly from
      [range_ratio * effective_input_len, effective_input_len], then selects
      a LongBench prompt long enough for that target and head-truncates to it.
    - output_len: max_new_tokens is sampled uniformly from
      [range_ratio * output_len, output_len].
    """
    speed_cfg = config.get("synthetic", {})
    max_new_tokens_default = int(config.get("max_new_tokens", 256))
    input_len = int(speed_cfg.get("input_len", 1024))
    output_len = int(speed_cfg.get("output_len", max_new_tokens_default))
    range_ratio = _range_ratio(speed_cfg.get("range_ratio", 0.8))
    ignore_eos = bool(speed_cfg.get("ignore_eos", True))
    temperature = float(config.get("temperature", 0.2))
    seed = int(config.get("dataset_seed", 248))

    num_reqs = int(config.get("num_requests", 32))
    if limit is not None:
        num_reqs = min(num_reqs, limit)

    inference_dir = Path(__file__).resolve().parent
    pool_n = max(num_reqs, _env_int("INFERENCE_BENCH_LONGBENCH_POOL_N", 503))
    samples_path = _find_longbench_samples_file(inference_dir, seed, pool_n, allow_smaller=False)

    if samples_path is None:
        # Fallback: generate cache on the fly (requires HF datasets + cache).
        print(
            f"[eval:speed] no pre-cached LongBench-v2 file found for seed={seed} n>={pool_n}; "
            "generating on the fly (this may be slow)"
        )
        from .cache_samples import cache_longbench_v2, _enable_soft_filelock
        _enable_soft_filelock()
        tmp_path = (
            inference_dir / "baselines" / "samples" / "longbench_v2"
            / f"{seed}_{pool_n}" / "samples.jsonl"
        )
        try:
            cache_longbench_v2(tmp_path, seed=seed, n=pool_n)
            samples_path = tmp_path
        except Exception as exc:
            fallback = _find_longbench_samples_file(inference_dir, seed, num_reqs, allow_smaller=True)
            if fallback is None:
                raise
            print(
                f"[eval:speed] WARNING: failed to generate LongBench-v2 pool n={pool_n}: {exc}; "
                f"falling back to cached file {fallback}"
            )
            samples_path = fallback

    print(
        f"[eval:speed] longbench_v2 mode: seed={seed} input_len={input_len} "
        f"output_len={output_len} range_ratio={range_ratio} num_reqs={num_reqs} file={samples_path}"
    )

    if tokenizer is None:
        raise RuntimeError(
            "LongBench-v2 speed sampling requires the base model tokenizer to draw and verify "
            "input token lengths. Provide a precomputed requests.jsonl via --requests-file or "
            "INFERENCE_BENCH_REQUESTS_FILE, pre-cache the tokenizer, or set "
            "INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1."
        )

    target_input_len = input_len
    if max_model_len is not None:
        max_input_len = _compute_max_input_tokens(max_model_len, output_len)
        if max_input_len is not None:
            target_input_len = min(target_input_len, max_input_len)
    min_input_tokens = max(0, int(math.ceil(target_input_len * range_ratio)))
    min_output_tokens = max(1, int(math.ceil(output_len * range_ratio)))
    min_output_tokens = min(min_output_tokens, output_len)

    eligible: List[Tuple[Dict[str, Any], int]] = []
    with samples_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = [dict(m) for m in record.get("messages", [])]
            source_prompt_tokens = _count_chat_tokens(messages, tokenizer)
            if source_prompt_tokens < min_input_tokens:
                continue
            eligible.append(({"record": record, "messages": messages}, source_prompt_tokens))

    if not eligible:
        raise RuntimeError(
            "No LongBench-v2 samples can satisfy the input length range: "
            f"seed={seed} pool_n={pool_n} min_input_tokens={min_input_tokens} "
            f"target_input_len={target_input_len} file={samples_path}"
        )

    selection_rng = random.Random(f"{seed}:longbench:{target_input_len}:{output_len}:{range_ratio}")
    input_rng = random.Random(f"{seed}:input:{target_input_len}:{output_len}:{range_ratio}")
    output_rng = random.Random(f"{seed}:output:{target_input_len}:{output_len}:{range_ratio}")

    requests_list: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    for req_idx in range(num_reqs):
        target_input_tokens = input_rng.randint(min_input_tokens, target_input_len)
        candidates = [
            (item, source_prompt_tokens)
            for item, source_prompt_tokens in eligible
            if source_prompt_tokens >= target_input_tokens
        ]
        if not candidates:
            max_source = max(source_prompt_tokens for _, source_prompt_tokens in eligible)
            raise RuntimeError(
                "No LongBench-v2 sample can satisfy sampled input target: "
                f"seed={seed} pool_n={pool_n} request_index={req_idx} "
                f"target_input_token_count={target_input_tokens} max_source_prompt_tokens={max_source} "
                f"file={samples_path}"
            )
        item, source_prompt_tokens = candidates[selection_rng.randrange(len(candidates))]
        record = item["record"]
        messages = [dict(m) for m in item["messages"]]
        messages = _truncate_messages(messages, tokenizer, target_input_tokens, keep="head")
        realized_input_tokens = _count_chat_tokens(messages, tokenizer)
        if not (min_input_tokens <= realized_input_tokens <= target_input_tokens):
            raise RuntimeError(
                "Sampled LongBench-v2 request realized outside input range: "
                f"seed={seed} request_index={req_idx} target_input_token_count={target_input_tokens} "
                f"realized_input_token_count={realized_input_tokens} min_input_tokens={min_input_tokens} "
                f"target_input_len={target_input_len} sample_id={record.get('sample_id', '')}"
            )
        max_new_tokens = output_rng.randint(min_output_tokens, output_len)
        req: Dict[str, Any] = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "require_json": False,
            "gold_answer": None,  # speed eval: no quality scoring
            "sample_id": str(record.get("sample_id", "")),
            "parse_mode": "",
            "ignore_eos": ignore_eos,
            "target_input_token_count": target_input_tokens,
            "input_token_count": realized_input_tokens,
            "source_prompt_token_count": source_prompt_tokens,
            "sampling_range_ratio": range_ratio,
        }
        req["content_hash"] = _request_content_hash(req)
        requests_list.append(req)
        samples.append({
            "sample_id": req["sample_id"],
            "messages": messages,
            "prompt_text": _messages_to_prompt_text(messages),
            "target_input_token_count": target_input_tokens,
            "input_token_count": realized_input_tokens,
            "source_prompt_token_count": source_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "content_hash": req["content_hash"],
        })

    return requests_list, samples


def _schedule(pattern: str, count: int, rate_per_s: Optional[float]) -> List[float]:
    if pattern == "burst":
        return [0.0] * count
    if rate_per_s is None or rate_per_s <= 0:
        raise ValueError("rate_per_s must be set for non-burst patterns")
    delays: List[float] = []
    if pattern == "constant":
        for i in range(count):
            delays.append(i / rate_per_s)
    elif pattern == "poisson":
        t = 0.0
        for _ in range(count):
            t += random.expovariate(rate_per_s)
            delays.append(t)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return delays


def _extract_delta_text(chunk: Dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    if "content" in delta:
        return delta["content"] or ""
    if "text" in choices[0]:
        return choices[0]["text"] or ""
    return ""


async def _stream_chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    most_recent_time: Optional[float] = None
    text_parts: List[str] = []
    chunk_itls: List[float] = []  # per-chunk inter-token latencies
    output_tokens: Optional[int] = None  # server-reported completion tokens
    error: Optional[str] = None

    raw_chunks: List[str] = []

    try:
        async with session.post(url, json=payload, timeout=timeout_s) as resp:
            if resp.status != 200:
                error = await resp.text()
                end = time.perf_counter()
                return {
                    "success": False,
                    "empty_output": False,
                    "start": start,
                    "end": end,
                    "first_token_time": None,
                    "text": "",
                    "error": error,
                    "chunk_itls": [],
                    "output_tokens": None,
                }

            buffer = ""
            async for chunk in resp.content.iter_any():
                piece = chunk.decode("utf-8", errors="ignore")
                raw_chunks.append(piece)
                buffer += piece
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # Extract server-reported usage (vLLM/SGLang include this
                    # in the final chunk or in stream_options.include_usage).
                    usage = parsed.get("usage") or {}
                    if usage.get("completion_tokens"):
                        output_tokens = int(usage["completion_tokens"])
                    delta_text = _extract_delta_text(parsed)
                    if delta_text:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        elif most_recent_time is not None:
                            chunk_itls.append(now - most_recent_time)
                        most_recent_time = now
                        text_parts.append(delta_text)
    except asyncio.TimeoutError:
        error = f"timeout after {timeout_s}s"
    except aiohttp.ClientError as exc:
        error = str(exc)

    end = time.perf_counter()
    text = "".join(text_parts)

    # Some OpenAI-compatible servers ignore `stream=true` and return a single JSON response.
    # If we didn't parse any SSE deltas, attempt to parse the full payload as JSON.
    if error is None and not text:
        raw = "".join(raw_chunks).strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                choices = parsed.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content:
                        text = content
                usage = parsed.get("usage") or {}
                if usage.get("completion_tokens"):
                    output_tokens = int(usage["completion_tokens"])
            except Exception:
                pass

    # A request is successful if no transport error occurred.
    # Empty output (server responded but generated nothing) is tracked separately.
    has_error = error is not None
    has_text = len(text) > 0
    success = not has_error and has_text
    empty_output = not has_error and not has_text

    return {
        "success": success,
        "empty_output": empty_output,
        "start": start,
        "end": end,
        "first_token_time": first_token_time,
        "text": text,
        "error": error,
        "chunk_itls": chunk_itls,
        "output_tokens": output_tokens,
    }


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _validate_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _extract_choice_answer(text: str) -> Optional[str]:
    """Extract a multiple-choice answer letter (A-J) from model output.

    Uses a 3-level fallback strategy aligned with the official MMLU-Pro
    extraction logic (TIGER-AI-Lab/MMLU-Pro), extended with additional
    common patterns.
    """
    if not text:
        return None
    upper = text.upper()
    # Level 1 (official MMLU-Pro primary): "the answer is (X)" / "answer is X"
    m = re.search(r"ANSWER\s+IS\s*\(?([A-J])\)?", upper)
    if m:
        return m.group(1)
    # Level 2 (official MMLU-Pro secondary): "Answer: X"
    m = re.search(r"ANSWER\s*[:\-]\s*([A-J])\b", upper)
    if m:
        return m.group(1)
    # Additional explicit forms.
    for pattern in [
        r"CORRECT\s+ANSWER\s*(IS|:)?\s*([A-J])\b",
        r"(OPTION|CHOICE)\s*([A-J])\b",
        r"FINAL\s*[:\-]\s*([A-J])\b",
    ]:
        m = re.search(pattern, upper)
        if m:
            return m.group(m.lastindex)
    # Leading "B)" or "B." at start of text.
    m = re.search(r"^\s*([A-J])\s*[\)\.\:\-]", upper)
    if m:
        return m.group(1)
    # Standalone letter on its own line.
    m = re.search(r"^\s*([A-J])\s*$", upper, re.MULTILINE)
    if m:
        return m.group(1)
    # Level 3 (official MMLU-Pro fallback): last standalone A-J letter in text.
    m = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", upper, re.DOTALL)
    if m:
        return m.group(1)
    return None


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer an explicit "Answer:" span.
    match = re.search(r"(?i)\banswer\s*[:\-]\s*([-+]?\d[\d,]*)\b", text)
    if match:
        return match.group(1).replace(",", "")
    # Otherwise take the last integer-like token.
    numbers = re.findall(r"[-+]?\d[\d,]*", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def _get_user_content(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _first_sentence(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    line = stripped.splitlines()[0].strip()
    if not line:
        return ""
    match = re.search(r"[.!?]", line)
    if match:
        return line[: match.end()].strip()
    return line[:max_len].strip()


def _extract_question(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"Question\\s*:\\s*(.*)", text, flags=re.IGNORECASE)
    if matches:
        candidate = matches[-1].strip()
        if candidate:
            return candidate
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


async def _run_profile(
    profile: Dict[str, Any],
    requests_list: List[Dict[str, Any]],
    server_url: str,
    model: str,
    timeout_s: float,
    log_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], float, float]:
    pattern = profile.get("pattern", "burst")
    rate_per_s = profile.get("rate_per_s")
    concurrency = int(profile.get("concurrency", 1))
    num_requests = int(profile.get("num_requests", len(requests_list)))

    delays = _schedule(pattern, num_requests, rate_per_s)
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    max_retries = 2  # 1 original + 1 retry on transient failure

    async with aiohttp.ClientSession() as session:
        async def _run_one(index: int, req: Dict[str, Any], delay: float) -> None:
            await asyncio.sleep(delay)
            async with sem:
                payload = {
                    "model": model,
                    "messages": req["messages"],
                    "stream": True,
                    "max_tokens": req["max_new_tokens"],
                    "temperature": req["temperature"],
                }
                if req.get("ignore_eos"):
                    payload["ignore_eos"] = True
                url = f"{server_url}/v1/chat/completions"
                result = await _stream_chat_completion(session, url, payload, timeout_s)
                # Retry once on transient connection failures (timeout, connection reset).
                # Do NOT retry on HTTP errors (4xx/5xx) — those indicate the server
                # processed the request and returned a valid error.
                for _retry in range(1, max_retries):
                    if result["success"]:
                        break
                    err = (result.get("error") or "").lower()
                    is_transient = ("timeout" in err or
                                    "connect" in err or
                                    "reset" in err)
                    if not is_transient:
                        break
                    await asyncio.sleep(2 ** _retry)
                    result = await _stream_chat_completion(session, url, payload, timeout_s)
                result["request_index"] = index
                result["pattern"] = pattern
                result["require_json"] = req.get("require_json", False)
                result["gold_answer"] = req.get("gold_answer")
                result["sample_id"] = req.get("sample_id")
                user_content = _get_user_content(req.get("messages", []))
                result["input_first_sentence"] = _first_sentence(user_content)
                result["input_question"] = _extract_question(user_content)
                if result["success"]:
                    # Prefer server-reported token count, fall back to word-split.
                    result["tokens"] = result.get("output_tokens") or _count_tokens(result["text"])
                    if result["require_json"]:
                        result["json_valid"] = _validate_json(result["text"])
                    else:
                        result["json_valid"] = None
                    parse_mode = (req.get("parse_mode") or "").strip().lower()
                    if parse_mode == "gsm8k":
                        result["parsed_answer"] = _extract_gsm8k_answer(result["text"])
                    elif parse_mode in ("mcq", "mcq10") or (not parse_mode and result.get("gold_answer")):
                        result["parsed_answer"] = _extract_choice_answer(result["text"])
                    else:
                        result["parsed_answer"] = None
                else:
                    result["tokens"] = 0
                    result["json_valid"] = None
                    result["parsed_answer"] = None
                results.append(result)
                done = len(results)
                if done == num_requests or done % max(1, num_requests // 20) == 0:
                    ok = sum(1 for r in results if r.get("success"))
                    print(f"[progress] {done}/{num_requests} requests done ({ok} succeeded)")

        tasks = []
        for i in range(num_requests):
            req = requests_list[i % len(requests_list)]
            tasks.append(asyncio.create_task(_run_one(i, req, delays[i])))

        await asyncio.gather(*tasks)

    wall_start = min((r["start"] for r in results), default=time.perf_counter())
    wall_end = max((r["end"] for r in results), default=wall_start)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            for r in results:
                record = {
                    "request_index": r.get("request_index"),
                    "sample_id": r.get("sample_id"),
                    "input_first_sentence": r.get("input_first_sentence"),
                    "input_question": r.get("input_question"),
                    "model_output": r.get("text"),
                    "gold_answer": r.get("gold_answer"),
                    "parsed_answer": r.get("parsed_answer"),
                    "success": r.get("success"),
                    "empty_output": r.get("empty_output", False),
                    "error": r.get("error"),
                    "tokens": r.get("tokens"),
                    "output_tokens": r.get("output_tokens"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return results, wall_start, wall_end


def _percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p90": None, "p99": None}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _summarize_results(results: List[Dict[str, Any]], wall_start: float, wall_end: float) -> Dict[str, Any]:
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"] and not r.get("empty_output")]
    empty_outputs = [r for r in results if r.get("empty_output")]

    # TTFT: also include empty-output requests (server responded, just no content).
    ttft_candidates = successes + empty_outputs
    ttft_values = [r["first_token_time"] - r["start"] for r in ttft_candidates if r.get("first_token_time")]

    # ITL: prefer per-chunk latencies collected during streaming (matches InferenceX).
    # Fall back to estimated ITL for requests without chunk data (e.g. non-streaming).
    itl_values: List[float] = []
    tpot_values: List[float] = []
    total_tokens = 0
    total_decode_time = 0.0
    for r in successes:
        tokens = r.get("tokens", 0)
        total_tokens += tokens
        if r.get("first_token_time") is not None:
            decode_time = r["end"] - r["first_token_time"]
            total_decode_time += max(decode_time, 0.0)
            # TPOT: per-request average decode latency (like InferenceX).
            if tokens > 1:
                tpot_values.append(decode_time / (tokens - 1))
        # ITL: use actual per-chunk timestamps when available.
        chunk_itls = r.get("chunk_itls") or []
        if chunk_itls:
            itl_values.extend(chunk_itls)
        elif r.get("first_token_time") is not None and tokens > 1:
            # Fallback: estimate from total decode time (old behavior).
            itl_values.append((r["end"] - r["first_token_time"]) / (tokens - 1))

    wall_time = max(wall_end - wall_start, 1e-6)
    req_throughput = len(results) / wall_time
    gen_throughput = (total_tokens / total_decode_time) if total_decode_time > 0 else 0.0

    json_required = [r for r in results if r.get("require_json")]
    json_valid = [r for r in json_required if r.get("json_valid")]
    json_rate = (len(json_valid) / len(json_required)) if json_required else None

    answerable = [r for r in results if r.get("gold_answer")]
    correct = [
        r for r in answerable
        if r.get("parsed_answer") and r.get("parsed_answer") == r.get("gold_answer")
    ]
    accuracy = (len(correct) / len(answerable)) if answerable else None

    metrics = {
        "request_count": len(results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "empty_output_count": len(empty_outputs),
        "failure_rate": (len(failures) / len(results)) if results else 0.0,
        "ttft": _percentiles(ttft_values),
        "itl": _percentiles(itl_values),
        "tpot": _percentiles(tpot_values),
        "generation_throughput_tokens_per_s": gen_throughput,
        "request_throughput_req_per_s": req_throughput,
        "json_valid_rate": json_rate,
        "choice_accuracy": accuracy,
    }
    return metrics


def _compute_choice_accuracy(results: List[Dict[str, Any]]) -> Optional[float]:
    answerable = [r for r in results if r.get("gold_answer")]
    if not answerable:
        return None
    correct = [
        r for r in answerable
        if r.get("parsed_answer") and r.get("parsed_answer") == r.get("gold_answer")
    ]
    return len(correct) / len(answerable)


def _get_vram_usage_mb() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0.0
    values = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return max(values) if values else 0.0


def _sample_vram_peak(stop_event: threading.Event, interval_s: float, out: List[float]) -> None:
    peak = 0.0
    while not stop_event.is_set():
        peak = max(peak, _get_vram_usage_mb())
        time.sleep(interval_s)
    out.append(peak)


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


def _wait_for_server(server_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for server at {server_url}")


def _artifact_dir(task_dir: Path, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path.parent
    return task_dir


def _write_requests_used(path: Path, requests_list: List[Dict[str, Any]]) -> None:
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


def _compute_accuracy(results: List[Dict[str, Any]]) -> Optional[float]:
    answerable = [r for r in results if r.get("gold_answer") is not None]
    if not answerable:
        return None
    correct = [
        r for r in answerable
        if (r.get("parsed_answer") is not None) and (r.get("parsed_answer") == r.get("gold_answer"))
    ]
    return len(correct) / len(answerable)


def run_speed_eval(task_dir: Path, args: argparse.Namespace, artifact_dir: Path) -> Dict[str, Any]:
    config = load_scenario_config(task_dir)
    if args.seed is not None:
        config["dataset_seed"] = args.seed
    print(f"[eval:speed] scenario={config.get('scenario_id')} dataset_seed={config.get('dataset_seed')}")

    # Prefer a precomputed requests file if available (avoids slow dataset loading/tokenization).
    requests_source: Optional[Path] = None
    if getattr(args, "requests_file", None):
        requests_source = Path(args.requests_file)
    else:
        env_requests = os.environ.get("INFERENCE_BENCH_REQUESTS_FILE", "").strip()
        if env_requests:
            requests_source = Path(env_requests)
        else:
            requests_source = _default_requests_file(task_dir)

    dataset_samples: List[Dict[str, Any]] = []
    if requests_source is not None and requests_source.exists():
        base_model = args.model or os.environ.get("INFERENCE_BENCH_BASE_MODEL", "")
        tokenizer = _get_tokenizer(base_model) if base_model else None
        max_model_len_env = os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "").strip()
        max_model_len: Optional[int] = int(max_model_len_env) if max_model_len_env.isdigit() else None
        if max_model_len is None and base_model:
            max_model_len = _get_model_max_len(base_model)
        base_requests, dataset_samples = _load_requests_jsonl(
            requests_source,
            config,
            args.request_limit,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
        )
        print(f"[eval:speed] using requests file: {requests_source}")
    else:
        # Fall back to dataset sampling (slower, but works without precomputed requests).
        base_model = args.model or os.environ.get("INFERENCE_BENCH_BASE_MODEL", "")
        tokenizer = _get_tokenizer(base_model) if base_model else None
        max_model_len_env = os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "").strip()
        max_model_len: Optional[int] = int(max_model_len_env) if max_model_len_env.isdigit() else None
        if max_model_len is None and base_model:
            max_model_len = _get_model_max_len(base_model)
        base_requests, dataset_samples = _prepare_requests(
            config,
            limit=args.request_limit,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
        )
        # Logging handled inside _prepare_requests (longbench_v2 mode).

    print(f"[eval:speed] prepared {len(base_requests)} requests (limit={args.request_limit})")

    used_requests_path = artifact_dir / "requests_used_speed.jsonl"
    _write_requests_used(used_requests_path, base_requests)

    _wait_for_server(args.server_url, args.server_wait_s)
    model_id = _detect_model_id(args.server_url, args.model)
    print(f"[eval:speed] server_url={args.server_url} model_id={model_id}")

    profiles = config.get("profiles")
    if not profiles:
        profiles = [config.get("profile", {"name": "default", "pattern": "burst"})]
    if dataset_samples:
        adjusted_profiles = []
        for profile in profiles:
            profile_copy = dict(profile)
            profile_copy["num_requests"] = len(base_requests)
            adjusted_profiles.append(profile_copy)
        profiles = adjusted_profiles

    stop_event = threading.Event()
    vram_out: List[float] = []
    sampler = threading.Thread(target=_sample_vram_peak, args=(stop_event, 0.5, vram_out))
    sampler.daemon = True
    sampler.start()

    profile_metrics: Dict[str, Any] = {}
    log_path_env = os.environ.get("INFERENCE_BENCH_GENERATION_LOG", "").strip()
    log_path = Path(log_path_env) if log_path_env else (artifact_dir / "speed_generations.jsonl")
    if log_path.exists():
        log_path.unlink()

    for profile in profiles:
        profile_name = profile.get("name") or profile.get("pattern", "profile")
        print(
            "[eval:speed] running profile "
            f"name={profile_name} pattern={profile.get('pattern')} "
            f"num_requests={profile.get('num_requests', len(base_requests))} "
            f"concurrency={profile.get('concurrency', 1)}"
        )
        results, wall_start, wall_end = asyncio.run(
            _run_profile(profile, base_requests, args.server_url, model_id, args.request_timeout_s, log_path)
        )
        successes = len([r for r in results if r.get("success")])
        print(f"[eval:speed] profile={profile_name} results={len(results)} success={successes}")
        profile_metrics[profile_name] = _summarize_results(results, wall_start, wall_end)

    stop_event.set()
    sampler.join(timeout=2)

    return {
        "scenario": config.get("scenario_id"),
        "name": config.get("name"),
        "mission": config.get("mission"),
        "dataset": config.get("dataset") or config.get("dataset_placeholder"),
        "dataset_split": config.get("dataset_split"),
        "model_id": model_id,
        "profiles": profile_metrics,
        "vram_peak_mb": max(vram_out) if vram_out else 0.0,
        "speed_eval": {
            "requests_source": str(requests_source) if requests_source else None,
            "requests_used_file": str(used_requests_path),
            "generation_log_file": str(log_path),
        },
    }


def run_quality_eval(
    server_url: str,
    model_id: str,
    request_timeout_s: float,
    artifact_dir: Path,
    tau: float,
) -> Dict[str, Any]:
    specs, _, registry_path = quality_gate.get_quality_specs()

    baseline: Dict[str, Any] = {}
    for spec in specs:
        acc, ref = quality_gate.load_quality_baseline_accuracy(registry_path, spec.name, spec.seed, spec.limit)
        baseline[spec.name] = {"accuracy": acc, "ref": ref}

    # Require precomputed baselines for all datasets.
    missing = [name for name, b in baseline.items() if not isinstance(b.get("accuracy"), (int, float))]
    if missing:
        return {
            "tau": tau,
            "pass": False,
            "error": f"missing baseline accuracy for: {', '.join(missing)}",
            "baseline_registry": str(registry_path),
            "datasets": {k: {"baseline_accuracy": v.get("accuracy"), "baseline_ref": v.get("ref")} for k, v in baseline.items()},
        }

    run_results: Dict[str, Any] = {}
    for spec in specs:
        requests_list = quality_gate.load_quality_requests(spec)
        profile = {
            "name": f"quality_{spec.name}",
            "pattern": "burst",
            "num_requests": len(requests_list),
            "concurrency": int(os.environ.get("INFERENCE_BENCH_QUALITY_CONCURRENCY", "4") or 4),
        }
        log_path = artifact_dir / f"quality_{spec.name}_generations.jsonl"
        if log_path.exists():
            log_path.unlink()
        results, _, _ = asyncio.run(
            _run_profile(profile, requests_list, server_url, model_id, request_timeout_s, log_path)
        )
        run_results[spec.name] = {
            "accuracy": _compute_accuracy(results),
            "count": len([r for r in results if r.get("gold_answer") is not None]),
            "samples_file": str(spec.samples_file),
            "generation_log_file": str(log_path),
            "seed": spec.seed,
            "n": spec.limit,
        }

    baseline_simple = {k: {"accuracy": v.get("accuracy"), "ref": v.get("ref")} for k, v in baseline.items()}
    gate = quality_gate.compute_quality_gate(run_results, tau=tau, baseline=baseline_simple)
    gate["baseline_registry"] = str(registry_path)
    # Merge per-dataset metadata.
    for name, meta in run_results.items():
        gate.setdefault("datasets", {}).setdefault(name, {}).update({
            "samples_file": meta.get("samples_file"),
            "generation_log_file": meta.get("generation_log_file"),
            "seed": meta.get("seed"),
            "n": meta.get("n"),
            "evaluated_count": meta.get("count"),
        })
    return gate


def _mock_metrics(task_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Generate placeholder metrics without contacting a server.  Useful for
    testing the pipeline end-to-end without GPU time."""
    config = load_scenario_config(task_dir)
    return {
        "scenario": config.get("scenario_id"),
        "name": config.get("name"),
        "model_id": args.model or os.environ.get("INFERENCE_BENCH_BASE_MODEL", "mock"),
        "profiles": {
            "mock": {
                "request_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "failure_rate": 0.0,
                "ttft": {"p50": None, "p90": None, "p99": None},
                "itl": {"p50": None, "p90": None, "p99": None},
                "generation_throughput_tokens_per_s": 0.0,
                "request_throughput_req_per_s": 0.0,
            },
        },
        "vram_peak_mb": 0.0,
        "quality_check": {"pass": True, "note": "mock mode"},
        "eval_pipeline": {"version": 1, "steps": ["mock"]},
        "eval_mode": {"name": "mock"},
    }


def run_evaluation(task_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_path: Optional[Path] = Path(args.json_output_file) if args.json_output_file else None
    artifact_dir = _artifact_dir(task_dir, output_path)

    # Mock mode: skip server entirely and emit placeholder metrics.
    if getattr(args, "mock", False) or os.environ.get("INFERENCE_BENCH_MOCK", "").lower() in {"1", "true", "yes"}:
        metrics = _mock_metrics(task_dir, args)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    try:
        quick_mode = bool(getattr(args, "quick", False))
        quick_limit = int(getattr(args, "quick_request_limit", 24))
        if quick_mode:
            if args.request_limit is None:
                args.request_limit = quick_limit
            else:
                args.request_limit = min(int(args.request_limit), quick_limit)
            raw_qn = os.environ.get("INFERENCE_BENCH_QUICK_QUALITY_N", "16").strip()
            try:
                quick_quality_n = max(1, int(raw_qn))
            except ValueError:
                quick_quality_n = 16
            os.environ.setdefault("INFERENCE_BENCH_QUALITY_MMLUPRO_N", str(quick_quality_n))

        metrics = run_speed_eval(task_dir, args, artifact_dir)

        skip_quality = os.environ.get("INFERENCE_BENCH_SKIP_QUALITY", "").lower() in {"1", "true", "yes"}
        if not skip_quality:
            quality = run_quality_eval(
                args.server_url,
                metrics.get("model_id", args.model or "unknown"),
                args.request_timeout_s,
                artifact_dir,
                args.quality_tau,
            )
            metrics["quality_check"] = quality
            if not quality.get("pass", True):
                metrics["score"] = 0
        else:
            metrics["quality_check"] = {"pass": True, "note": "skipped via INFERENCE_BENCH_SKIP_QUALITY"}

        metrics.setdefault("eval_pipeline", {"version": 1, "steps": ["speed_eval", "quality_check"]})
        if quick_mode:
            qn = 0
            try:
                qn = int(os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_N", "0") or 0)
            except ValueError:
                qn = 0
            metrics["eval_mode"] = {
                "name": "quick",
                "request_limit": int(args.request_limit) if args.request_limit is not None else quick_limit,
                "quality_n": qn,
            }

        if output_path is not None:
            output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
    except Exception as exc:
        if output_path is not None:
            error_metrics = {
                "scenario": os.environ.get("INFERENCE_BENCH_SCENARIO") or str(task_dir),
                "model_id": args.model or os.environ.get("INFERENCE_BENCH_BASE_MODEL", "") or "unknown",
                "profiles": {},
                "vram_peak_mb": 0.0,
                "error": str(exc),
            }
            output_path.write_text(json.dumps(error_metrics, indent=2), encoding="utf-8")
        raise


def build_parser() -> argparse.ArgumentParser:
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    parser = argparse.ArgumentParser(description="InferenceBench scenario evaluator")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000")
    # vLLM can take several minutes to become ready, especially if kernels are being
    # compiled or very large prefill sizes are configured. Allow the harness to
    # override these defaults via env vars.
    parser.add_argument(
        "--server-wait-s",
        type=float,
        default=_env_float("INFERENCE_BENCH_SERVER_WAIT_S", 900.0),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=_env_float("INFERENCE_BENCH_REQUEST_TIMEOUT_S", 300.0),
    )
    parser.add_argument("--model", type=str, default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", ""))
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248")),
        help="Seed for dataset sampling.",
    )
    parser.add_argument(
        "--requests-file",
        type=str,
        default=None,
        help="Optional requests.jsonl to reuse (skips dataset sampling). Can also be set via INFERENCE_BENCH_REQUESTS_FILE.",
    )
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode: smaller speed request set and smaller quality subset.",
    )
    parser.add_argument(
        "--quick-request-limit",
        type=int,
        default=int(os.environ.get("INFERENCE_BENCH_QUICK_REQUEST_LIMIT", "4")),
        help="Max requests used when --quick is set.",
    )
    parser.add_argument("--quality-tau", type=float, default=float(os.environ.get("INFERENCE_BENCH_QUALITY_TAU", "0.95")))
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock mode: skip server and emit placeholder metrics. "
             "Useful for testing the pipeline end-to-end without GPU time. "
             "Can also be set via INFERENCE_BENCH_MOCK=1.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.add_argument("--scenario-dir", type=str, required=True)
    args = parser.parse_args()
    run_evaluation(Path(args.scenario_dir), args)


if __name__ == "__main__":
    main()
