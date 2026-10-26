#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
from importlib import metadata
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.inference import precompute_all_baselines as _pb
from src.eval.inference.precompute_all_baselines import _repo_root
from src.eval.inference.quality_gate import DEFAULT_SEED, speed_profiles_complete


SCENARIOS = {
    "a": "inference_scenario_a_input_heavy",
    "b": "inference_scenario_b_output_heavy",
    "c": "inference_scenario_c_high_load",
    "d": "inference_scenario_d_general",
}
SCENARIO_TO_SHORT = {v: k for k, v in SCENARIOS.items()}

FAILURE_SCORE = 1e-6
MAX_PHYSICAL_GEN_TPS = 20_000.0
MIN_PHYSICAL_TPOT_S = 1e-4
MIN_PHYSICAL_TTFT_S = 0.010


class BudgetExpired(Exception):
    pass


class AbortRun(Exception):
    pass


@dataclass(frozen=True)
class SearchParam:
    name: str
    flag: str
    kind: str
    values: list[Any]
    false_flag: str | None = None


@dataclass
class TrialResult:
    trial_idx: int
    config: dict[str, Any]
    quick_metric: float
    failure_reason: str | None
    metrics_path: str | None
    integrity_flags: list[str]


SGLANG_STABLE_SEARCH_VALUES: dict[str, set[Any]] = {
    "attention_backend": {"triton"},
    "quantization": {"none"},
    "enable_torch_compile": {False},
    "cuda_graph_max_bs": {0},
    "speculative_num_steps": {0},
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_search_space(engine: str, path: Path | None = None) -> tuple[dict[str, Any], list[SearchParam]]:
    path = path or _repo_root() / "src" / "baselines" / "search_spaces" / f"{engine}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not contain a mapping")
    params: list[SearchParam] = []
    names: set[str] = set()
    for item in raw.get("parameters") or []:
        name = str(item["name"])
        values = item["values"]
        if name in names:
            raise ValueError(f"duplicate parameter {name}: {path}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameter {name} must have a nonempty values list: {path}")
        if len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
            raise ValueError(f"parameter {name} has duplicate values: {path}")
        if item.get("type", "choice") not in {"choice", "bool"}:
            raise ValueError(f"unsupported parameter type for {name}: {path}")
        if item.get("type") == "bool" and any(type(value) is not bool for value in values):
            raise ValueError(f"parameter {name} requires boolean values: {path}")
        names.add(name)
        params.append(SearchParam(name, str(item["flag"]), str(item.get("type", "choice")), list(values), item.get("false_flag")))
    if not params:
        raise ValueError(f"empty search space: {path}")
    return raw, params


def _normalized_choice(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _load_installed_sglang_cli_spec(timeout_s: float = 30.0) -> dict[str, list[str] | None]:
    code = r"""
import argparse
import json
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser(add_help=False)
ServerArgs.add_cli_args(parser)
spec = {}
for action in parser._actions:
    for option in action.option_strings:
        if not option.startswith("--"):
            continue
        choices = None
        if action.choices is not None:
            choices = [str(choice) for choice in action.choices]
        spec[option] = choices
print(json.dumps(spec, sort_keys=True))
"""
    out = _run_text([sys.executable, "-c", code], timeout_s=timeout_s)
    for line in reversed(out.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def validate_search_space(
    engine: str,
    params: list[SearchParam],
    *,
    allow_unsafe: bool = False,
    max_model_len: int = 32768,
) -> None:
    if engine == "vllm":
        by_name = {param.name: param for param in params}
        chunked = by_name.get("enable_chunked_prefill")
        batch_tokens = by_name.get("max_num_batched_tokens")
        if chunked is not None and False in chunked.values and batch_tokens is not None:
            if any(int(value) < max_model_len for value in batch_tokens.values):
                raise ValueError(
                    "invalid vLLM HPO search space: disabling chunked prefill requires every "
                    f"max_num_batched_tokens value to be at least max_model_len={max_model_len}"
                )
        return
    if engine != "sglang":
        return

    errors: list[str] = []
    by_name = {param.name: param for param in params}

    if not allow_unsafe:
        for name, allowed in SGLANG_STABLE_SEARCH_VALUES.items():
            param = by_name.get(name)
            if param is None:
                continue
            actual = {_normalized_choice(value) for value in param.values}
            bad = sorted(str(value) for value in actual - allowed)
            if bad:
                allowed_text = ", ".join(sorted(str(value) for value in allowed))
                errors.append(
                    f"{name} has unsafe values {bad}; default SGLang HPO only allows {allowed_text}"
                )

    cli_spec = _load_installed_sglang_cli_spec()
    if cli_spec:
        emitted_flags: dict[str, list[str]] = {
            "cuda_graph_max_bs": [
                "--cuda-graph-max-bs",
                "--disable-cuda-graph",
                "--disable-piecewise-cuda-graph",
            ],
            "speculative_num_steps": [
                "--speculative-algorithm",
                "--speculative-num-steps",
                "--speculative-num-draft-tokens",
            ],
        }
        for param in params:
            flags = emitted_flags.get(param.name, [param.flag])
            for flag in flags:
                if flag.startswith("--") and flag not in cli_spec:
                    errors.append(f"{param.name} references unsupported SGLang flag {flag}")

            if param.kind != "choice" or not param.flag.startswith("--"):
                continue
            choices = cli_spec.get(param.flag)
            if not choices:
                continue
            supported = {str(choice).lower() for choice in choices}
            for value in param.values:
                text = str(value).lower()
                if text in {"none", "null"}:
                    continue
                if text not in supported:
                    errors.append(
                        f"{param.name}={value!r} is not accepted by installed SGLang flag {param.flag}"
                    )

    if errors:
        detail = "\n  - ".join(errors)
        raise ValueError(f"invalid SGLang HPO search space:\n  - {detail}")


def normalize_scenario(raw: str) -> str:
    key = raw.strip().lower()
    if key.startswith("scenario_"):
        key = key.split("_", 1)[1]
    if key in SCENARIOS:
        return SCENARIOS[key]
    if raw in SCENARIO_TO_SHORT:
        return raw
    raise ValueError(f"unknown scenario: {raw}")


def config_to_hashable(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in config.items()))


def sample_uniform(params: list[SearchParam], rng: random.Random) -> dict[str, Any]:
    return {p.name: rng.choice(p.values) for p in params}


def split_fixed_params(params: list[SearchParam]) -> tuple[dict[str, Any], list[SearchParam]]:
    fixed = {
        param.name: _jsonable(param.values[0])
        for param in params
        if len(param.values) == 1
    }
    tunable = [param for param in params if len(param.values) > 1]
    return fixed, tunable


def search_space_size(params: list[SearchParam]) -> int:
    size = 1
    for param in params:
        size *= max(1, len(param.values))
    return size


def _safe_p50(profile: dict[str, Any], group: str) -> float | None:
    summary = profile.get(group)
    value = summary.get("p50") if isinstance(summary, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def profile_is_physical(profile: dict[str, Any]) -> bool:
    values = [
        profile.get("generation_throughput_tokens_per_s"),
        profile.get("request_throughput_req_per_s"),
        _safe_p50(profile, "tpot"),
        _safe_p50(profile, "ttft"),
    ]
    if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in values):
        return False
    gen = profile.get("generation_throughput_tokens_per_s")
    if isinstance(gen, (int, float)) and gen > MAX_PHYSICAL_GEN_TPS:
        return False
    tpot = _safe_p50(profile, "tpot")
    if isinstance(tpot, (int, float)) and 0 < tpot < MIN_PHYSICAL_TPOT_S:
        return False
    ttft = _safe_p50(profile, "ttft")
    if isinstance(ttft, (int, float)) and 0 < ttft < MIN_PHYSICAL_TTFT_S:
        return False
    return True


def integrity_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for name, profile in (metrics.get("profiles") or {}).items():
        if isinstance(profile, dict) and not profile_is_physical(profile):
            flags.append(str(name))
    return flags


def primary_metric(metrics: dict[str, Any], scenario: str) -> tuple[float, str | None]:
    if not speed_profiles_complete(metrics):
        return FAILURE_SCORE, "incomplete_speed_profile"
    profiles = metrics.get("profiles") or {}
    burst = profiles.get("burst")
    if not isinstance(burst, dict):
        return FAILURE_SCORE, "missing_burst_profile"
    if not profile_is_physical(burst):
        return FAILURE_SCORE, "integrity_failed"

    short = SCENARIO_TO_SHORT.get(scenario)
    if short == "a":
        ttft = _safe_p50(burst, "ttft")
        if not ttft or ttft <= 0:
            return FAILURE_SCORE, "missing_ttft"
        return 1.0 / ttft, "1/ttft_p50"

    if short == "b":
        tpot = _safe_p50(burst, "tpot")
        if not tpot or tpot <= 0:
            return FAILURE_SCORE, "missing_tpot"
        return 1.0 / tpot, "1/tpot_p50"

    if short == "c":
        throughputs: list[float] = []
        for name in ("burst", "poisson", "constant"):
            profile = profiles.get(name)
            if not isinstance(profile, dict):
                return FAILURE_SCORE, f"missing_{name}_profile"
            if not profile_is_physical(profile):
                return FAILURE_SCORE, "integrity_failed"
            req = profile.get("request_throughput_req_per_s")
            if not isinstance(req, (int, float)) or req <= 0:
                return FAILURE_SCORE, "missing_request_throughput"
            throughputs.append(float(req))
        return math.prod(throughputs) ** (1.0 / len(throughputs)), "geomean_request_throughput_req_per_s"

    if short == "d":
        ttft = _safe_p50(burst, "ttft")
        tpot = _safe_p50(burst, "tpot")
        req = burst.get("request_throughput_req_per_s")
        if not ttft or not tpot or not isinstance(req, (int, float)) or ttft <= 0 or tpot <= 0 or req <= 0:
            return FAILURE_SCORE, "missing_geomean_component"
        return (1.0 / ttft / tpot * float(req)) ** (1.0 / 3.0), "geomean_inverse_latency_throughput"

    return FAILURE_SCORE, "unknown_scenario"


def gate_passed(metrics: dict[str, Any]) -> bool:
    if not speed_profiles_complete(metrics):
        return False
    if metrics.get("error"):
        return False
    if metrics.get("score") == 0:
        return False
    qc = metrics.get("quality_check")
    if not isinstance(qc, dict):
        return False
    return qc.get("pass") is True


def _run_text(cmd: list[str], timeout_s: float = 20.0) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def detect_engine_version(engine: str) -> str:
    return _run_text([sys.executable, "-c", f"import {engine}; print(getattr({engine}, '__version__', 'unknown'))"]) or "unknown"


def harness_commit() -> dict[str, str | bool]:
    root = _repo_root()
    rev = _run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = bool(_run_text(["git", "-C", str(root), "status", "--porcelain"]))
    return {"commit": rev or "unknown", "dirty": dirty}


def ensure_dependency(import_name: str, auto_install: bool, timeout_s: float) -> None:
    if timeout_s <= 0:
        raise BudgetExpired("budget exhausted before dependency installation")
    requirements = _repo_root() / "containers/hpo-requirements.txt"
    pins = dict(line.split("==", 1) for line in requirements.read_text().splitlines() if "==" in line)
    expected = pins[import_name]
    try:
        installed = metadata.version(import_name)
        if installed == expected:
            __import__(import_name)
            return
    except (metadata.PackageNotFoundError, ImportError):
        installed = "missing or not importable"
    if not auto_install:
        raise RuntimeError(
            f"{import_name}=={expected} is required; found {installed}. "
            f"Install {requirements} or pass --auto-install-deps."
        )
    print(f"[hpo] installing pinned HPO dependencies from {requirements}", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--constraint", str(requirements.with_name("hpo-constraints.txt")), "-r", str(requirements)],
            check=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BudgetExpired("budget exhausted during dependency installation") from exc
    if metadata.version(import_name) != expected:
        raise RuntimeError(f"dependency installation did not provide {import_name}=={expected}")
    __import__(import_name)


def build_engine_cmd(
    engine: str,
    config: dict[str, Any],
    params: list[SearchParam],
    base_model: str,
    host: str,
    port: int,
    max_model_len: str,
    seed: int,
) -> list[str]:
    if engine == "vllm":
        cmd = [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--host", host,
            "--port", str(port),
            "--model", base_model,
            "--max-model-len", str(max_model_len),
            "--tokenizer-mode", _pb._vllm_tokenizer_mode(),
            "--seed", str(seed),
        ]
    else:
        cmd = [
            sys.executable, "-u", "-m", "sglang.launch_server",
            "--host", host,
            "--port", str(port),
            "--model-path", base_model,
            "--context-length", str(max_model_len),
        ]

    by_name = {p.name: p for p in params}
    for name, value in config.items():
        param = by_name[name]
        value = _jsonable(value)

        if engine == "vllm" and name == "num_speculative_tokens":
            n = int(value)
            if n > 0:
                cmd += [
                    param.flag,
                    json.dumps(
                        {
                            "method": "ngram",
                            "num_speculative_tokens": n,
                            "prompt_lookup_max": 4,
                            "prompt_lookup_min": 2,
                        },
                        separators=(",", ":"),
                    ),
                ]
            continue

        if engine == "vllm" and name == "attention_backend":
            continue

        if engine == "sglang" and name == "cuda_graph_max_bs":
            n = int(value)
            if n <= 0:
                cmd += ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
            else:
                cmd += [param.flag, str(n)]
            continue

        if engine == "sglang" and name == "speculative_num_steps":
            n = int(value)
            if n > 0:
                cmd += [
                    "--speculative-algorithm",
                    "NGRAM",
                    "--speculative-num-steps",
                    str(n),
                    "--speculative-num-draft-tokens",
                    str(n),
                ]
            continue

        if param.kind == "bool":
            if bool(value):
                cmd.append(param.flag)
            elif param.false_flag:
                cmd.append(param.false_flag)
            continue

        if str(value).lower() in {"none", "null"}:
            continue
        cmd += [param.flag, str(value)]

    return cmd


def build_trial_env(engine: str, config: dict[str, Any], base_env: dict[str, str], skip_quality: bool) -> dict[str, str]:
    env = dict(base_env)
    env["INFERENCE_BENCH_ACTIVE_BACKEND"] = engine
    if skip_quality:
        env["INFERENCE_BENCH_SKIP_QUALITY"] = "1"
    else:
        env.pop("INFERENCE_BENCH_SKIP_QUALITY", None)

    if engine == "vllm" and config.get("attention_backend"):
        env["VLLM_ATTENTION_BACKEND"] = str(config["attention_backend"])
    if engine == "vllm":
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
    return env


def classify_start_failure(message: str) -> str:
    text = message.lower()
    incompatible_patterns = [
        "unrecognized arguments",
        "invalid choice",
        "cannot find the config file for",
        "requires a",
        "is required",
        "only supports",
        "not supported",
        "unsupported",
    ]
    if any(p in text for p in incompatible_patterns):
        return "incompatible_flags"
    resource_patterns = [
        "out of memory",
        "cuda out of memory",
        "no available memory",
        "failed to allocate",
    ]
    if any(p in text for p in resource_patterns):
        return "resource_exhausted"
    return "server_start_failed"


def _own_gpu_processes() -> list[tuple[int, float]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    if not devices or any(not (device.isdigit() or device.startswith(("GPU-", "MIG-"))) for device in devices):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={visible}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    out: list[tuple[int, float]] = []
    uid = os.getuid()
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = float(parts[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
        except OSError:
            continue
        out.append((pid, mem))
    return out


def verify_gpu_clean(threshold_mb: float) -> tuple[bool, list[tuple[int, float]]]:
    procs = _own_gpu_processes()
    dirty = [(pid, mem) for pid, mem in procs if mem >= threshold_mb]
    return (not dirty), dirty


def run_scenario_eval(
    scenario: str,
    server_url: str,
    base_model: str,
    seed: int,
    out_file: Path,
    env: dict[str, str],
    quick: bool,
    timeout_s: float,
) -> dict[str, Any]:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)
    task_script = _repo_root() / "src" / "eval" / "tasks" / scenario / "evaluate.py"
    cmd = [
        sys.executable,
        "-u",
        str(task_script),
        "--server-url",
        server_url,
        "--model",
        base_model,
        "--seed",
        str(seed),
        "--json-output-file",
        str(out_file),
        "--server-wait-s",
        "5",
    ]
    if quick:
        cmd.append("--quick")
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"evaluate.py timed out after {timeout_s:.1f}s", "wall_s": time.time() - start}
    elapsed = time.time() - start
    if proc.returncode != 0:
        return {"error": f"evaluate.py exited with code {proc.returncode}", "wall_s": elapsed}
    if not out_file.exists():
        return {"error": "evaluate.py produced no output", "wall_s": elapsed}
    try:
        data = json.loads(out_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"unparseable metrics json: {exc}", "wall_s": elapsed}
    if not isinstance(data, dict):
        return {"error": "metrics JSON must contain an object", "wall_s": elapsed}
    data.setdefault("wall_s", elapsed)
    return data


def score_quick_metrics(metrics: dict[str, Any], scenario: str) -> tuple[float, str | None, list[str]]:
    if "error" in metrics:
        return FAILURE_SCORE, "quick_eval_failed", []
    flags = integrity_flags(metrics)
    if flags:
        return FAILURE_SCORE, "integrity_failed", flags
    metric, source = primary_metric(metrics, scenario)
    if metric <= FAILURE_SCORE:
        return FAILURE_SCORE, source or "metric_failed", flags
    return metric, None, flags


class HpoRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        params: list[SearchParam],
        space_raw: dict[str, Any],
        jsonl_path: Path,
        artifact_root: Path,
    ) -> None:
        self.args = args
        self.params = params
        self.space_raw = space_raw
        self.jsonl_path = jsonl_path
        self.artifact_root = artifact_root
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.budget_start_s: float | None = None
        self.deadline_s: float | None = None
        self.best: TrialResult | None = None
        self.trials: list[TrialResult] = []
        self._trial_counter = 0

        runtime_cache_dir = Path(os.environ.get("INFERENCE_BENCH_RUNTIME_CACHE_DIR", str(_pb._default_runtime_cache_dir())))
        self.base_env = _pb._build_runtime_cache_env(runtime_cache_dir)
        _pb._install_python_shims(self.base_env)

    def start_budget(self) -> None:
        self.budget_start_s = time.monotonic()
        self.deadline_s = self.budget_start_s + float(self.args.budget_s)

    def budget_used_s(self) -> float:
        if self.budget_start_s is None:
            return 0.0
        return max(0.0, time.monotonic() - self.budget_start_s)

    def budget_remaining_s(self) -> float:
        if self.deadline_s is None:
            return float(self.args.budget_s)
        return max(0.0, self.deadline_s - time.monotonic())

    def allocate_trial_idx(self) -> int:
        idx = self._trial_counter
        self._trial_counter += 1
        return idx

    def append_record(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def run_start(self) -> None:
        with self.jsonl_path.open("w" if self.args.overwrite else "x", encoding="utf-8"):
            pass
        self.append_record(
            {
                "type": "run_start",
                "method": self.args.method,
                "engine": self.args.engine,
                "scenario": self.args.scenario,
                "seed_pair": [self.args.seed_dev, self.args.seed_eval],
                "quality_seed": self.args.quality_seed,
                "engine_version": detect_engine_version(self.args.engine),
                "harness_commit": harness_commit(),
                "search_space": self.space_raw,
                "start_ts": _utc_now(),
            }
        )

    def _record_trial_result(self, result: TrialResult) -> None:
        self.trials.append(result)
        if result.failure_reason is None and result.quick_metric > FAILURE_SCORE:
            if self.best is None or result.quick_metric > self.best.quick_metric:
                self.best = result

    def _server_spec(self, config: dict[str, Any], seed: int, skip_quality: bool) -> tuple[str, list[str], dict[str, str]]:
        port = self.args.port or _pb._free_port()
        cmd = build_engine_cmd(
            self.args.engine, config, self.params, self.args.base_model,
            self.args.host, port, self.args.max_model_len, seed,
        )
        env = build_trial_env(self.args.engine, config, self.base_env, skip_quality=skip_quality)
        return f"http://{self.args.host}:{port}", cmd, env

    @contextlib.contextmanager
    def _running_server(
        self,
        server_url: str,
        cmd: list[str],
        env: dict[str, str],
        log_path: Path,
        start_timeout_s: float,
        phase_times: dict[str, float],
    ):
        proc: subprocess.Popen | None = None
        t0 = time.time()
        try:
            proc = _pb._start_server(cmd, log_path=log_path, env=env)
            _pb._wait_ready(
                server_url,
                timeout_s=start_timeout_s,
                proc=proc,
                log_path=log_path,
                base_model=self.args.base_model,
                initial_delay_s=self.args.server_initial_delay_s,
            )
            phase_times["server_start_s"] = time.time() - t0
            yield
        finally:
            t1 = time.time()
            _pb._stop_server(proc, 30.0)
            phase_times["teardown_s"] = time.time() - t1

    def _dirty_gpu(self) -> list[tuple[int, float]]:
        if self.args.skip_gpu_clean_check:
            return []
        return verify_gpu_clean(self.args.gpu_clean_threshold_mb)[1]

    def run_trial(self, config: dict[str, Any], trial_idx: int) -> float:
        if self.budget_remaining_s() <= 1:
            raise BudgetExpired("budget exhausted before trial launch")
        if self.args.max_trials is not None and len(self.trials) >= self.args.max_trials:
            raise BudgetExpired("max_trials reached")

        trial_dir = self.artifact_root / f"trial_{trial_idx:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        launch_ts = _utc_now()
        eval_ts: str | None = None
        metrics_path: Path | None = None
        failure_reason: str | None = None
        metric = FAILURE_SCORE
        flags: list[str] = []
        phase_times: dict[str, float] = {}

        server_url, cmd, env = self._server_spec(config, self.args.seed_dev, skip_quality=True)
        start_timeout = self.args.first_server_start_timeout_s if trial_idx == 0 else self.args.server_start_timeout_s
        try:
            print(f"[hpo] trial={trial_idx} config={config}", flush=True)
            print(f"[hpo] trial={trial_idx} cmd={' '.join(shlex.quote(c) for c in cmd)}", flush=True)
            with self._running_server(
                server_url, cmd, env, trial_dir / "server.log",
                min(start_timeout, max(1.0, self.budget_remaining_s())), phase_times,
            ):
                if self.budget_remaining_s() <= 1:
                    raise BudgetExpired("budget expired before quick eval")
                eval_ts = _utc_now()
                metrics_path = trial_dir / "quick_metrics.json"
                t1 = time.time()
                metrics = run_scenario_eval(
                    self.args.scenario,
                    server_url,
                    self.args.base_model,
                    self.args.seed_dev,
                    metrics_path,
                    env,
                    quick=True,
                    timeout_s=min(self.args.quick_eval_timeout_s, max(1.0, self.budget_remaining_s())),
                )
                phase_times["quick_eval_s"] = time.time() - t1
            metric, failure_reason, flags = score_quick_metrics(metrics, self.args.scenario)
        except BudgetExpired:
            failure_reason = "budget_expired"
        except (RuntimeError, TimeoutError) as exc:
            failure_reason = classify_start_failure(str(exc))
        except Exception as exc:
            failure_reason = f"unexpected_error:{type(exc).__name__}"
            (trial_dir / "unexpected_error.txt").write_text(str(exc), encoding="utf-8")
        teardown_ts = _utc_now()
        dirty = self._dirty_gpu()
        if dirty:
            failure_reason, metric = "gpu_cleanup_failed", FAILURE_SCORE

        result = TrialResult(
            trial_idx=trial_idx,
            config=dict(config),
            quick_metric=float(metric),
            failure_reason=failure_reason,
            metrics_path=str(metrics_path) if metrics_path else None,
            integrity_flags=flags,
        )
        self._record_trial_result(result)
        self.append_record(
            {
                "type": "trial",
                "trial_idx": trial_idx,
                "config": config,
                "quick_metric": float(metric),
                "failure_reason": failure_reason,
                "integrity_flags": flags,
                "launch_ts": launch_ts,
                "eval_ts": eval_ts,
                "teardown_ts": teardown_ts,
                "cmd": cmd,
                "server_url": server_url,
                "metrics_path": str(metrics_path) if metrics_path else None,
                "phase_times_s": phase_times,
            }
        )
        if dirty:
            raise AbortRun(f"GPU still dirty after teardown: {dirty}")
        return float(metric)

    def run_final(self) -> dict[str, Any]:
        best = self.best
        if best is None:
            return {
                "best_trial_idx": None,
                "final_config": None,
                "final_metrics": None,
                "gate_passed": False,
                "integrity_passed": False,
            }

        final_dir = self.artifact_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        server_url, cmd, env = self._server_spec(best.config, self.args.seed_eval, skip_quality=False)
        env["INFERENCE_BENCH_QUALITY_SEED"] = str(self.args.quality_seed)
        print(f"[hpo] final best_trial={best.trial_idx} config={best.config}", flush=True)
        try:
            with self._running_server(
                server_url, cmd, env, final_dir / "server.log", self.args.final_server_start_timeout_s, {},
            ):
                metrics = run_scenario_eval(
                    self.args.scenario,
                    server_url,
                    self.args.base_model,
                    self.args.seed_eval,
                    final_dir / "final_metrics.json",
                    env,
                    quick=False,
                    timeout_s=self.args.final_eval_timeout_s,
                )
        finally:
            dirty = self._dirty_gpu()
            if dirty:
                raise AbortRun(f"GPU still dirty after final teardown: {dirty}")

        metric, source = primary_metric(metrics, self.args.scenario)
        flags = integrity_flags(metrics)
        return {
            "best_trial_idx": best.trial_idx,
            "final_config": best.config,
            "final_metrics": metrics,
            "final_primary_metric": metric,
            "final_primary_metric_source": source,
            "gate_passed": gate_passed(metrics),
            "integrity_passed": not flags and speed_profiles_complete(metrics),
            "integrity_flags": flags,
            "final_cmd": cmd,
        }


def run_random_search(runner: HpoRunner, params: list[SearchParam]) -> None:
    rng = random.Random(runner.args.seed_dev)
    fixed, tunable = split_fixed_params(params)
    max_unique = search_space_size(tunable)
    tried: set[tuple[tuple[str, str], ...]] = set()
    while runner.budget_remaining_s() > 0 and len(tried) < max_unique:
        if runner.args.max_trials is not None and len(runner.trials) >= runner.args.max_trials:
            break
        config = {**fixed, **sample_uniform(tunable, rng)}
        key = config_to_hashable(config)
        if key in tried:
            continue
        tried.add(key)
        try:
            runner.run_trial(config, runner.allocate_trial_idx())
        except BudgetExpired:
            break


def run_optuna_tpe(runner: HpoRunner, params: list[SearchParam]) -> None:
    try:
        ensure_dependency("optuna", runner.args.auto_install_deps, runner.budget_remaining_s())
    except BudgetExpired:
        return
    import optuna

    fixed, tunable = split_fixed_params(params)
    if not tunable:
        return run_random_search(runner, params)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=runner.args.seed_dev),
    )

    def objective(trial: Any) -> float:
        config = dict(fixed)
        config.update({
            p.name: trial.suggest_categorical(p.name, [_jsonable(v) for v in p.values])
            for p in tunable
        })
        idx = runner.allocate_trial_idx()
        try:
            return runner.run_trial(config, idx)
        except BudgetExpired:
            study.stop()
            return FAILURE_SCORE

    remaining = runner.budget_remaining_s()
    if remaining > 0:
        study.optimize(objective, timeout=remaining, n_trials=runner.args.max_trials, show_progress_bar=False)


def run_smac(runner: HpoRunner, params: list[SearchParam]) -> None:
    try:
        ensure_dependency("ConfigSpace", runner.args.auto_install_deps, runner.budget_remaining_s())
        ensure_dependency("smac", runner.args.auto_install_deps, runner.budget_remaining_s())
    except BudgetExpired:
        return

    from ConfigSpace import Categorical, ConfigurationSpace
    from smac import HyperparameterOptimizationFacade as HPO
    from smac import Scenario
    from smac.main.exceptions import ConfigurationSpaceExhaustedException

    fixed, tunable = split_fixed_params(params)
    if not tunable:
        return run_random_search(runner, params)

    cs = ConfigurationSpace(seed=runner.args.seed_dev)
    for p in tunable:
        cs.add(Categorical(p.name, [json.dumps(_jsonable(v), sort_keys=True) for v in p.values]))

    def objective(config: Any, seed: int = 0) -> float:
        decoded = dict(fixed)
        decoded.update({p.name: json.loads(str(config[p.name])) for p in tunable})
        idx = runner.allocate_trial_idx()
        try:
            return -runner.run_trial(decoded, idx)
        except BudgetExpired:
            return -FAILURE_SCORE

    remaining = runner.budget_remaining_s()
    if remaining <= 0:
        return
    scenario = Scenario(
        cs,
        deterministic=False,
        n_trials=runner.args.max_trials or 10000,
        walltime_limit=remaining,
        output_directory=runner.artifact_root / "smac3",
        seed=runner.args.seed_dev,
    )
    smac = HPO(scenario, objective, overwrite=True)
    try:
        smac.optimize()
    except ConfigurationSpaceExhaustedException:
        print("[hpo] SMAC search space exhausted", flush=True)


SEARCHES = {"random": run_random_search, "tpe": run_optuna_tpe, "smac": run_smac}


def default_out_root() -> Path:
    raw = os.environ.get("HPO_BASELINE_OUT_ROOT", "").strip()
    if raw:
        return Path(raw)
    results = os.environ.get("INFERENCE_BENCH_RESULTS_DIR", "").strip() or "results"
    return Path(results) / "hpo_search_baselines"


def _env(name: str, cast: Any, default: Any = None) -> Any:
    raw = os.environ.get(name, "").strip()
    if raw and raw != "UNDEFINED":
        try:
            return cast(raw)
        except ValueError:
            pass
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random, TPE, or SMAC inference search baselines.")
    parser.add_argument("--method", choices=["random", "tpe", "smac"], required=True)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--scenario", required=True, help="a/b/c/d or full scenario directory name")
    parser.add_argument("--seed-id", type=int, default=0)
    parser.add_argument("--seed-dev", type=int, default=_env("HPO_BASELINE_SEED_DEV", int, 21))
    parser.add_argument("--seed-eval", type=int, default=_env("HPO_BASELINE_SEED_EVAL", int, 1337))
    parser.add_argument("--quality-seed", type=int, default=_env("HPO_BASELINE_QUALITY_SEED", int, _env("INFERENCE_BENCH_QUALITY_SEED", int, DEFAULT_SEED)))
    parser.add_argument("--base-model", default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    parser.add_argument("--max-model-len", default=os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "32768"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port for each trial.")
    parser.add_argument("--out-root", type=Path, default=default_out_root())
    parser.add_argument("--search-space", type=Path, default=None)
    parser.add_argument("--budget-s", type=float, default=_env("HPO_BASELINE_BUDGET_S", float, 7200.0))
    parser.add_argument("--quick-eval-timeout-s", type=float, default=_env("HPO_BASELINE_QUICK_EVAL_TIMEOUT_S", float, 180.0))
    parser.add_argument("--server-start-timeout-s", type=float, default=_env("HPO_BASELINE_SERVER_START_TIMEOUT_S", float, 300.0))
    parser.add_argument("--first-server-start-timeout-s", type=float, default=_env("HPO_BASELINE_FIRST_SERVER_START_TIMEOUT_S", float, 360.0))
    parser.add_argument("--final-server-start-timeout-s", type=float, default=_env("HPO_BASELINE_FINAL_SERVER_START_TIMEOUT_S", float, 600.0))
    parser.add_argument("--server-initial-delay-s", type=float, default=_env("HPO_BASELINE_SERVER_INITIAL_DELAY_S", float, 60.0))
    parser.add_argument("--final-eval-timeout-s", type=float, default=_env("HPO_BASELINE_FINAL_EVAL_TIMEOUT_S", float, 7200.0))
    parser.add_argument("--gpu-clean-threshold-mb", type=float, default=_env("HPO_BASELINE_GPU_CLEAN_THRESHOLD_MB", float, 1024.0))
    parser.add_argument("--skip-gpu-clean-check", action="store_true")
    parser.add_argument("--auto-install-deps", action="store_true", default=os.environ.get("HPO_BASELINE_AUTO_INSTALL_DEPS", "0") == "1")
    parser.add_argument("--max-trials", type=int, default=_env("HPO_BASELINE_MAX_TRIALS", int))
    parser.add_argument("--overwrite", action="store_true", default=os.environ.get("HPO_BASELINE_OVERWRITE", "0") == "1")
    parser.add_argument(
        "--allow-unsafe-search-space",
        action="store_true",
        default=os.environ.get("HPO_BASELINE_ALLOW_UNSAFE_SEARCH_SPACE", "0") == "1",
        help="Skip the SGLang search-space safety checks.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in (
        "budget_s", "quick_eval_timeout_s", "server_start_timeout_s",
        "first_server_start_timeout_s", "final_server_start_timeout_s",
        "final_eval_timeout_s", "gpu_clean_threshold_mb",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.server_initial_delay_s) or args.server_initial_delay_s < 0:
        parser.error("--server-initial-delay-s must be finite and nonnegative")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    if args.seed_id < 0:
        parser.error("--seed-id must be nonnegative")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    args.scenario = normalize_scenario(args.scenario)

    space_raw, params = load_search_space(args.engine, args.search_space)
    validate_search_space(
        args.engine,
        params,
        allow_unsafe=args.allow_unsafe_search_space,
        max_model_len=int(args.max_model_len),
    )
    scenario_short = SCENARIO_TO_SHORT[args.scenario]
    jsonl_path = args.out_root / args.method / args.engine / f"scenario_{scenario_short}" / f"seed_{args.seed_id}.jsonl"
    artifact_root = jsonl_path.parent / f"seed_{args.seed_id}_artifacts"

    runner = HpoRunner(args, params, space_raw, jsonl_path, artifact_root)
    runner.run_start()
    runner.start_budget()

    try:
        SEARCHES[args.method](runner, params)
        budget_used = runner.budget_used_s()
        final = runner.run_final()
        runner.append_record(
            {
                "type": "run_end",
                **final,
                "total_trials": len(runner.trials),
                "budget_used_s": budget_used,
                "end_ts": _utc_now(),
            }
        )
    except Exception as exc:
        runner.append_record(
            {
                "type": "run_abort",
                "reason": str(exc),
                "total_trials": len(runner.trials),
                "budget_used_s": runner.budget_used_s(),
                "end_ts": _utc_now(),
            }
        )
        raise

    print(f"[hpo] wrote {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
