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
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests

from . import quality_gate
from .quality_gate import DEFAULT_SEED, env_float, env_int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_scenario_config(task_dir: Path) -> Dict[str, Any]:
    config_path = task_dir / "scenario.json"
    config = json.loads(_read_text(config_path))
    mission_path = task_dir / "mission.txt"
    if mission_path.exists():
        config["mission"] = _read_text(mission_path)
    if config.get("dataset_seed") in (None, ""):
        config["dataset_seed"] = env_int("INFERENCE_BENCH_DATASET_SEED", DEFAULT_SEED)
    config["task_dir"] = str(task_dir)
    return config


def _count_chat_tokens(messages: List[Dict[str, str]], tokenizer: Any) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            if hasattr(tokens, "keys"):
                tokens = tokens["input_ids"]
            return len(tokens)
        except Exception:
            pass
    joined = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in messages)
    return len(tokenizer.encode(joined, add_special_tokens=False))


def _truncate_messages(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    max_input_tokens: Optional[int],
    *,
    keep: str = "tail",
) -> List[Dict[str, str]]:
    if tokenizer is None or max_input_tokens is None:
        return messages
    max_input_tokens = max(0, max_input_tokens)

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
        truncated_tokens = user_tokens[-allowed_user_tokens:]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    messages[user_idx]["content"] = truncated_text
    return messages


def _compute_max_input_tokens(max_model_len: Optional[int], max_new_tokens: int) -> Optional[int]:
    if max_model_len is None:
        return None
    margin = max(0, env_int("INFERENCE_BENCH_INPUT_TOKEN_MARGIN", 16))
    return max(0, int(max_model_len) - int(max_new_tokens) - 1 - margin)


def _hf_from_pretrained(loader, model_id: str, **kwargs):
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or None
    try:
        return loader(model_id, local_files_only=True, cache_dir=cache_dir, **kwargs)
    except Exception:
        if os.environ.get("INFERENCE_BENCH_ALLOW_HF_DOWNLOAD", "").strip().lower() not in {"1", "true", "yes"}:
            return None
        try:
            return loader(model_id, local_files_only=False, cache_dir=cache_dir, **kwargs)
        except Exception:
            return None


def _get_model_max_len(model_id: str) -> Optional[int]:
    env_value = env_int("INFERENCE_BENCH_MAX_MODEL_LEN", 0)
    if env_value > 0:
        return env_value
    if not model_id:
        return None
    from transformers import AutoConfig
    cfg = _hf_from_pretrained(AutoConfig.from_pretrained, model_id)
    max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "model_max_length", None)
    return max_len if isinstance(max_len, int) and max_len > 0 else None


def _get_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer
    return _hf_from_pretrained(AutoTokenizer.from_pretrained, model_id, use_fast=True)


def _request_content_hash(req: Dict[str, Any]) -> str:
    payload = {
        "messages": req.get("messages"),
        "max_new_tokens": req.get("max_new_tokens"),
        "temperature": req.get("temperature"),
        "require_json": False,
        "ignore_eos": req.get("ignore_eos", False),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_requests_jsonl(
    path: Path,
    config: Dict[str, Any],
    limit: Optional[int],
    tokenizer: Any,
    max_model_len: Optional[int],
) -> List[Dict[str, Any]]:
    max_new_tokens_default = int(config.get("max_new_tokens", 256))
    temperature_default = float(config.get("temperature", 0.2))
    ignore_eos = config.get("synthetic", {}).get("ignore_eos", False)

    requests_list: List[Dict[str, Any]] = []
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
            max_new_tokens = int(record.get("max_new_tokens", max_new_tokens_default))
            if tokenizer is not None and max_model_len is not None:
                max_input_tokens = _compute_max_input_tokens(max_model_len, max_new_tokens)
                messages = _truncate_messages(messages, tokenizer, max_input_tokens)

            req: Dict[str, Any] = {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": float(record.get("temperature", temperature_default)),
                "gold_answer": gold,
                "sample_id": str(record.get("sample_id") or record.get("_id") or ""),
                "parse_mode": record.get("parse_mode") or "",
            }
            if record.get("ignore_eos", ignore_eos):
                req["ignore_eos"] = True
            for meta_key in (
                "target_input_token_count",
                "input_token_count",
                "source_prompt_token_count",
                "sampling_range_ratio",
            ):
                if meta_key in record:
                    req[meta_key] = record.get(meta_key)
            req["content_hash"] = _request_content_hash(req)
            requests_list.append(req)
    return requests_list


def _find_longbench_samples_file(
    inference_dir: Path,
    seed: int,
    min_n: int,
    *,
    allow_smaller: bool = True,
) -> Optional[Path]:
    base = inference_dir / "baselines" / "samples" / "longbench_v2"
    if not base.is_dir():
        return None

    candidates: List[Tuple[int, Path]] = []
    for cand in base.glob(f"{seed}_*/samples.jsonl"):
        try:
            n = int(cand.parent.name.split("_", 1)[1])
        except ValueError:
            continue
        candidates.append((n, cand))

    if not candidates:
        return None

    sufficient = [(n, p) for n, p in candidates if n >= min_n]
    if sufficient:
        return min(sufficient, key=lambda x: x[0])[1]
    if allow_smaller:
        return max(candidates, key=lambda x: x[0])[1]
    return None


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
    tokenizer: Any,
    max_model_len: Optional[int],
) -> List[Dict[str, Any]]:
    speed_cfg = config.get("synthetic", {})
    max_new_tokens_default = int(config.get("max_new_tokens", 256))
    input_len = int(speed_cfg.get("input_len", 1024))
    output_len = int(speed_cfg.get("output_len", max_new_tokens_default))
    range_ratio = _range_ratio(speed_cfg.get("range_ratio", 0.8))
    ignore_eos = bool(speed_cfg.get("ignore_eos", True))
    temperature = float(config.get("temperature", 0.2))
    seed = int(config.get("dataset_seed", DEFAULT_SEED))

    num_reqs = int(config.get("num_requests", 32))
    if limit is not None:
        num_reqs = min(num_reqs, limit)

    inference_dir = Path(__file__).resolve().parent
    pool_n = max(num_reqs, env_int("INFERENCE_BENCH_LONGBENCH_POOL_N", 503))
    samples_path = _find_longbench_samples_file(inference_dir, seed, pool_n, allow_smaller=False)

    if samples_path is None:
        print(f"[eval:speed] no cached LongBench-v2 samples for seed={seed} n>={pool_n}; generating them")
        from .cache_samples import cache_longbench_v2, enable_soft_filelock
        enable_soft_filelock()
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
            print(f"[eval:speed] WARNING: LongBench-v2 sample generation failed ({exc}); using {fallback}")
            samples_path = fallback

    print(
        f"[eval:speed] longbench_v2 mode: seed={seed} input_len={input_len} "
        f"output_len={output_len} range_ratio={range_ratio} num_reqs={num_reqs} file={samples_path}"
    )

    if tokenizer is None:
        raise RuntimeError(
            "LongBench-v2 sampling needs the base model tokenizer; pass --requests-file, "
            "pre-cache the tokenizer, or set INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1."
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
            "gold_answer": None,
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
    return requests_list


def _schedule(pattern: str, count: int, rate_per_s: Optional[float], seed: Union[int, str] = DEFAULT_SEED) -> List[float]:
    if pattern == "burst":
        return [0.0] * count
    if rate_per_s is None or rate_per_s <= 0:
        raise ValueError("rate_per_s must be set for non-burst patterns")
    delays: List[float] = []
    if pattern == "constant":
        for i in range(count):
            delays.append(i / rate_per_s)
    elif pattern == "poisson":
        rng = random.Random(seed)
        t = 0.0
        for _ in range(count):
            t += rng.expovariate(rate_per_s)
            delays.append(t)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return delays


def _extract_delta_text(chunk: Dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or choices[0].get("message") or {}
    text = delta.get("content", choices[0].get("text", "")) or ""
    if not isinstance(text, str):
        raise ValueError("Completion content must be text")
    return text


def _parse_completion_line(line: bytes) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith(b"data:"):
        return {}
    data = line[len(b"data:"):].strip()
    if data == b"[DONE]":
        return None
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Invalid completion event")
    return parsed


async def _completion_events(content):
    buffer = b""
    async for chunk in content.iter_any():
        buffer += chunk
        if buffer.lstrip().startswith(b"{"):
            continue
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            event = _parse_completion_line(line)
            yield event
            if event is None:
                return
    if buffer.lstrip().startswith(b"{"):
        yield json.loads(buffer.decode("utf-8"))
    elif buffer.strip():
        yield _parse_completion_line(buffer)


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
    chunk_itls: List[float] = []
    output_tokens: Optional[int] = None
    error: Optional[str] = None

    retryable = False
    stream_finished = False

    try:
        async with session.post(url, json=payload, timeout=timeout_s) as resp:
            if resp.status != 200:
                error = await resp.text()
            elif "application/json" in resp.headers.get("Content-Type", "").lower():
                parsed = await resp.json(content_type=None)
                if parsed.get("error"):
                    raise ValueError(json.dumps(parsed["error"]))
                text_parts.append(_extract_delta_text(parsed))
                output_tokens = (parsed.get("usage") or {}).get("completion_tokens")
            else:
                async for parsed in _completion_events(resp.content):
                    if parsed is None:
                        stream_finished = True
                        break
                    if parsed.get("error"):
                        raise ValueError(json.dumps(parsed["error"]))
                    usage = parsed.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        output_tokens = usage["completion_tokens"]
                    choices = parsed.get("choices") or []
                    if choices and choices[0].get("finish_reason") is not None:
                        stream_finished = True
                    delta_text = _extract_delta_text(parsed)
                    if choices and "message" in choices[0]:
                        text_parts.append(delta_text)
                        stream_finished = True
                        continue
                    if delta_text:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        elif most_recent_time is not None:
                            chunk_itls.append(now - most_recent_time)
                        most_recent_time = now
                        text_parts.append(delta_text)
                if not stream_finished:
                    raise ValueError("Completion stream ended without a finish event")
            if output_tokens is not None and (type(output_tokens) is not int or output_tokens < 0):
                raise ValueError("Completion token count must be a nonnegative integer")
    except asyncio.TimeoutError:
        error = f"timeout after {timeout_s}s"
        retryable = True
    except aiohttp.ClientError as exc:
        error = str(exc)
        retryable = True
    except (ValueError, TypeError, AttributeError) as exc:
        error = str(exc)

    end = time.perf_counter()
    text = "".join(text_parts)
    return {
        "success": error is None and bool(text),
        "empty_output": error is None and not text,
        "start": start,
        "end": end,
        "first_token_time": first_token_time,
        "text": text,
        "error": error,
        "chunk_itls": chunk_itls,
        "output_tokens": output_tokens,
        "retryable": retryable,
    }


def _count_words(text: str) -> int:
    return len(text.split())


def _extract_choice_answer(text: str) -> Optional[str]:
    if not text:
        return None
    upper = text.upper()

    finals = re.findall(r"\bFINAL(?:\s+ANSWER)?\s*(?::|-|\bIS\b)\s*\(?([A-J])\)?(?!\w)", upper)
    if finals:
        return finals[-1]

    m = re.search(r"ANSWER\s+IS\s*\(?([A-J])\)?(?!\w)", upper)
    if m:
        return m.group(1)

    m = re.search(r"ANSWER\s*[:\-]\s*([A-J])\b", upper)
    if m:
        return m.group(1)

    m = re.search(r"^\s*([A-J])\s*[\)\.\:\-]", upper)
    if m:
        return m.group(1)

    m = re.search(r"^\s*([A-J])\s*(?:\n|$)", upper)
    if m:
        return m.group(1)

    for pattern in [
        r"CORRECT\s+ANSWER\s*(IS|:)?\s*([A-J])\b",
        r"(OPTION|CHOICE)\s*([A-J])\b",
    ]:
        m = re.search(pattern, upper)
        if m:
            return m.group(m.lastindex)

    m = re.search(r"^\s*([A-J])\s*$", upper, re.MULTILINE)
    if m:
        return m.group(1)

    m = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", upper, re.DOTALL)
    if m:
        return m.group(1)
    return None


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
    matches = re.findall(r"Question\s*:\s*(.*)", text, flags=re.IGNORECASE)
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
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, Any]], float, float]:
    pattern = profile.get("pattern", "burst")
    rate_per_s = profile.get("rate_per_s")
    concurrency = int(profile.get("concurrency", 1))
    num_requests = int(profile.get("num_requests", len(requests_list)))
    if not requests_list or num_requests <= 0:
        raise ValueError("A profile requires at least one request")
    if concurrency <= 0:
        raise ValueError("Profile concurrency must be positive")

    profile_name = profile.get("name") or pattern
    schedule_identity = {
        "seed": seed,
        "name": profile_name,
        "pattern": pattern,
        "rate_per_s": rate_per_s,
        "num_requests": num_requests,
        "concurrency": concurrency,
    }
    schedule_seed = hashlib.sha256(json.dumps(schedule_identity, sort_keys=True).encode("utf-8")).hexdigest()
    delays = _schedule(pattern, num_requests, rate_per_s, schedule_seed)
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

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
                    "stream_options": {"include_usage": True},
                }
                if req.get("ignore_eos"):
                    payload["ignore_eos"] = True
                url = f"{server_url}/v1/chat/completions"
                result = await _stream_chat_completion(session, url, payload, timeout_s)
                request_start = result["start"]
                if not result["success"] and result["retryable"]:
                    await asyncio.sleep(2)
                    result = await _stream_chat_completion(session, url, payload, timeout_s)
                result["start"] = request_start
                result["request_index"] = index
                result["profile"] = profile_name
                result["pattern"] = pattern
                result["run_seed"] = seed
                result["schedule_seed"] = schedule_seed
                result["scheduled_delay_s"] = delay
                result["requested_output_tokens"] = req["max_new_tokens"]
                result["gold_answer"] = req.get("gold_answer")
                result["sample_id"] = req.get("sample_id")
                user_content = _get_user_content(req.get("messages", []))
                result["input_first_sentence"] = _first_sentence(user_content)
                result["input_question"] = _extract_question(user_content)
                if result["success"] and req.get("ignore_eos"):
                    output_tokens = result.get("output_tokens")
                    if type(output_tokens) is not int or output_tokens != req["max_new_tokens"]:
                        result["success"] = False
                        result["retryable"] = False
                        result["error"] = (
                            "Output token count does not match the ignore_eos request: "
                            f"requested_output_tokens={req['max_new_tokens']} returned_output_tokens={output_tokens!r}"
                        )
                result["tokens"] = 0
                result["parsed_answer"] = None
                if result["success"]:
                    output_tokens = result["output_tokens"]
                    result["tokens"] = _count_words(result["text"]) if output_tokens is None else output_tokens
                    parse_mode = req.get("parse_mode") or ""
                    if parse_mode in ("mcq", "mcq10") or (not parse_mode and result["gold_answer"]):
                        result["parsed_answer"] = _extract_choice_answer(result["text"])
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
                    "profile": r.get("profile"),
                    "pattern": r.get("pattern"),
                    "run_seed": r.get("run_seed"),
                    "schedule_seed": r.get("schedule_seed"),
                    "scheduled_delay_s": r.get("scheduled_delay_s"),
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
                    "requested_output_tokens": r.get("requested_output_tokens"),
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
    ttft_values = [r["first_token_time"] - r["start"] for r in successes + empty_outputs if r.get("first_token_time")]

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
            if tokens > 1:
                tpot_values.append(decode_time / (tokens - 1))

        chunk_itls = r.get("chunk_itls") or []
        if chunk_itls:
            itl_values.extend(chunk_itls)
        elif r.get("first_token_time") is not None and tokens > 1:
            itl_values.append((r["end"] - r["first_token_time"]) / (tokens - 1))

    wall_time = max(wall_end - wall_start, 1e-6)
    req_throughput = len(successes) / wall_time
    gen_throughput = (total_tokens / total_decode_time) if total_decode_time > 0 else 0.0

    return {
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
        "choice_accuracy": _compute_choice_accuracy(results),
    }


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
                return data[0].get("id") or ""
    except requests.RequestException:
        pass
    return "unknown"


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


def _build_profiles(config: Dict[str, Any], num_requests: int, concurrency: Optional[int] = None) -> List[Dict[str, Any]]:
    profiles = []
    for profile in config.get("profiles") or [config.get("profile", {"name": "default", "pattern": "burst"})]:
        profile = {"concurrency": config.get("concurrency", 1), **profile, "num_requests": num_requests}
        if concurrency is not None:
            profile["concurrency"] = concurrency
        profiles.append(profile)
    return profiles


def run_speed_eval(task_dir: Path, args: argparse.Namespace, artifact_dir: Path) -> Dict[str, Any]:
    config = load_scenario_config(task_dir)
    config["dataset_seed"] = args.seed
    print(f"[eval:speed] scenario={config.get('scenario_id')} dataset_seed={config['dataset_seed']}")

    requests_file = args.requests_file or os.environ.get("INFERENCE_BENCH_REQUESTS_FILE", "").strip()
    requests_source = Path(requests_file) if requests_file else None
    if requests_source is not None and not requests_source.exists():
        raise FileNotFoundError(requests_source)

    base_model = args.model or os.environ.get("INFERENCE_BENCH_BASE_MODEL", "")
    tokenizer = _get_tokenizer(base_model) if base_model else None
    max_model_len = _get_model_max_len(base_model)

    if requests_source is not None:
        print(f"[eval:speed] using requests file: {requests_source}")
        base_requests = _load_requests_jsonl(requests_source, config, args.request_limit, tokenizer, max_model_len)
    else:
        base_requests = _prepare_requests(config, args.request_limit, tokenizer, max_model_len)

    used_requests_path = artifact_dir / "requests_used_speed.jsonl"
    _write_requests_used(used_requests_path, base_requests)

    _wait_for_server(args.server_url, args.server_wait_s)
    model_id = _detect_model_id(args.server_url, args.model)
    print(f"[eval:speed] server_url={args.server_url} model_id={model_id}")

    stop_event = threading.Event()
    vram_out: List[float] = []
    sampler = threading.Thread(target=_sample_vram_peak, args=(stop_event, 0.5, vram_out), daemon=True)
    sampler.start()

    profile_metrics: Dict[str, Any] = {}
    log_path_env = os.environ.get("INFERENCE_BENCH_GENERATION_LOG", "").strip()
    log_path = Path(log_path_env) if log_path_env else (artifact_dir / "speed_generations.jsonl")
    log_path.unlink(missing_ok=True)

    try:
        for profile in _build_profiles(config, len(base_requests)):
            profile_name = profile.get("name") or profile.get("pattern", "profile")
            print(
                "[eval:speed] running profile "
                f"name={profile_name} pattern={profile.get('pattern')} "
                f"num_requests={profile['num_requests']} concurrency={profile['concurrency']}"
            )
            results, wall_start, wall_end = asyncio.run(
                _run_profile(profile, base_requests, args.server_url, model_id, args.request_timeout_s, log_path,
                             config["dataset_seed"])
            )
            successes = sum(1 for r in results if r.get("success"))
            print(f"[eval:speed] profile={profile_name} results={len(results)} success={successes}")
            profile_metrics[profile_name] = _summarize_results(results, wall_start, wall_end)
    finally:
        stop_event.set()
        sampler.join(timeout=2)

    return {
        "scenario": config.get("scenario_id"),
        "name": config.get("name"),
        "mission": config.get("mission"),
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
    specs, registry_path = quality_gate.get_quality_specs(model_id)

    baseline: Dict[str, Any] = {}
    for spec in specs:
        acc, ref = quality_gate.load_quality_baseline_accuracy(registry_path, spec.name, spec.seed, spec.limit)
        baseline[spec.name] = {"accuracy": acc, "ref": ref}

    missing = [name for name, b in baseline.items() if b["accuracy"] is None]
    if missing:
        return {
            "tau": tau,
            "pass": False,
            "error": f"missing baseline accuracy for: {', '.join(missing)}",
            "baseline_registry": str(registry_path),
            "datasets": {k: {"baseline_accuracy": v["accuracy"], "baseline_ref": v["ref"]} for k, v in baseline.items()},
        }

    run_results: Dict[str, Any] = {}
    for spec in specs:
        requests_list = quality_gate.load_quality_requests(spec)
        profile = {
            "name": f"quality_{spec.name}",
            "pattern": "burst",
            "num_requests": len(requests_list),
            "concurrency": env_int("INFERENCE_BENCH_QUALITY_CONCURRENCY", 4),
        }
        log_path = artifact_dir / f"quality_{spec.name}_generations.jsonl"
        log_path.unlink(missing_ok=True)
        results, _, _ = asyncio.run(
            _run_profile(profile, requests_list, server_url, model_id, request_timeout_s, log_path, spec.seed)
        )
        succeeded = [r for r in results if r["success"] and not r.get("empty_output")]
        run_results[spec.name] = {
            "accuracy": _compute_choice_accuracy(results),
            "complete": len(succeeded) == len(results) == len(requests_list),
            "samples_file": str(spec.samples_file),
            "generation_log_file": str(log_path),
            "seed": spec.seed,
            "n": spec.limit,
            "evaluated_count": sum(r.get("gold_answer") is not None for r in results),
            "request_count": len(requests_list),
            "success_count": len(succeeded),
        }

    gate = quality_gate.compute_quality_gate(run_results, tau=tau, baseline=baseline)
    gate["baseline_registry"] = str(registry_path)
    for name, meta in run_results.items():
        gate["datasets"][name].update({k: v for k, v in meta.items() if k != "accuracy"})
    return gate


def run_evaluation(task_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_path: Optional[Path] = Path(args.json_output_file) if args.json_output_file else None
    artifact_dir = _artifact_dir(task_dir, output_path)

    try:
        if args.quick:
            if args.request_limit is None:
                args.request_limit = args.quick_request_limit
            else:
                args.request_limit = min(args.request_limit, args.quick_request_limit)

        metrics = run_speed_eval(task_dir, args, artifact_dir)
        profiles = metrics.get("profiles") or {}
        failed_profiles = [name for name, profile in profiles.items() if not quality_gate.profile_complete(profile)]
        speed_pass = bool(profiles) and not failed_profiles
        metrics["speed_check"] = {"pass": speed_pass, "failed_profiles": failed_profiles}
        if not speed_pass:
            metrics["score"] = 0

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
            if not quality["pass"]:
                metrics["score"] = 0
        else:
            metrics["quality_check"] = {"pass": True, "skipped": True}

        metrics["eval_pipeline"] = {"version": 1, "steps": ["speed_eval", "quality_check"]}
        if args.quick:
            metrics["eval_mode"] = {
                "name": "quick",
                "request_limit": args.request_limit,
                "quality_n": env_int("INFERENCE_BENCH_QUALITY_MMLUPRO_N", 500),
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
    parser = argparse.ArgumentParser(description="InferenceBench scenario evaluator")
    parser.add_argument("--server-url", type=str, default=os.environ.get("INFERENCE_BENCH_SERVER_URL") or "http://127.0.0.1:8000")
    parser.add_argument("--server-wait-s", type=float, default=env_float("INFERENCE_BENCH_SERVER_WAIT_S", 900.0))
    parser.add_argument("--request-timeout-s", type=float, default=env_float("INFERENCE_BENCH_REQUEST_TIMEOUT_S", 300.0))
    parser.add_argument("--model", type=str, default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", ""))
    parser.add_argument("--seed", type=int, default=env_int("INFERENCE_BENCH_DATASET_SEED", DEFAULT_SEED),
                        help="Seed for request sampling and arrival schedules.")
    parser.add_argument("--requests-file", type=str, default=None, help="Reuse this requests.jsonl instead of sampling.")
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Run a smaller speed request set.")
    parser.add_argument("--quick-request-limit", type=int, default=env_int("INFERENCE_BENCH_QUICK_REQUEST_LIMIT", 4))
    parser.add_argument("--quality-tau", type=float, default=env_float("INFERENCE_BENCH_QUALITY_TAU", 0.95))
    return parser


def main() -> None:
    parser = build_parser()
    parser.add_argument("--scenario-dir", type=str, required=True)
    args = parser.parse_args()
    run_evaluation(Path(args.scenario_dir), args)


if __name__ == "__main__":
    main()
