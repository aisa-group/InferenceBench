#!/usr/bin/env python3
"""Random-search baseline over vLLM configurations (Appendix I.6).

For each sampled configuration, launch a vLLM server with the config's
extra flags, run the four InferenceBench speed scenarios against it, tear
down the server, and record per-scenario metrics. Quality eval is skipped
(the model weights don't change across configurations, so quality is
invariant and would just burn GPU time).

Config 0 is always the vLLM default (no extra flags) so downstream
analysis can compare the other samples against it directly.

Run from inside the existing `run_precompute_all_baselines.sh` apptainer
shell via:

    python -u -m src.eval.inference.random_search_vllm \\
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \\
        --max-model-len 16384 \\
        --num-configs 32 \\
        --search-seed 248 \\
        --out-root /fast/${USER}/random_search_vllm

The module reuses `_start_server`, `_wait_ready`, and the runtime-cache
env plumbing from `src.eval.inference.precompute_all_baselines` so it
inherits the same container / venv / HF-cache assumptions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.inference import precompute_all_baselines as _pb


SCENARIOS = [
    "inference_scenario_a_input_heavy",
    "inference_scenario_b_output_heavy",
    "inference_scenario_c_high_load",
    "inference_scenario_d_general",
]

# Non-boolean vLLM flags. Defaults (what vLLM would pick if we omitted the
# flag) are the first element of each list so config 0 reproduces the
# default exactly when we zero out the override dict.
SEARCH_SPACE: dict[str, list[Any]] = {
    "max-num-seqs": [256, 64, 128, 512],
    "max-num-batched-tokens": [8192, 2048, 4096, 16384],
    "kv-cache-dtype": ["auto", "fp8"],
    "gpu-memory-utilization": [0.90, 0.80, 0.95],
    "block-size": [16, 32],
}

# Boolean flags and the vLLM attention backend are handled specially: the
# default is "don't override" (config 0 / most random draws omit them), and
# a handful of pinned "special" configs explicitly flip them on/off so the
# search has at least one data point for each.
SPECIAL_CONFIGS: list[dict[str, Any]] = [
    {"enforce-eager": True},
    {"no-enable-chunked-prefill": True},
    {"attention-backend": "FLASHINFER"},
]

def sample_configs(num_random: int, seed: int) -> list[dict[str, Any]]:
    """Generate (1 default + num_random random + len(SPECIAL_CONFIGS) pinned) configs."""
    configs: list[dict[str, Any]] = [{}]  # config 0 = vLLM default
    rng = random.Random(seed)
    for _ in range(max(0, num_random)):
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        configs.append(cfg)
    configs.extend(SPECIAL_CONFIGS)
    return configs


def build_vllm_cmd(
    base_model: str,
    host: str,
    port: int,
    max_model_len: str,
    config: dict[str, Any],
) -> list[str]:
    """Assemble the vLLM launch command for a given config.

    The base command mirrors `_build_backend_cmd` in precompute_all_baselines.
    Extra flags from `config` are appended; `gpu-memory-utilization` is
    special-cased so we only pass it once even if the config overrides it.
    """
    tokenizer_mode = os.environ.get("INFERENCE_BENCH_VLLM_TOKENIZER_MODE", "auto") or "auto"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        base_model,
        "--max-model-len",
        max_model_len,
        "--tokenizer-mode",
        tokenizer_mode,
    ]
    gpu_util = config.get("gpu-memory-utilization")
    if gpu_util is None:
        gpu_util = os.environ.get("INFERENCE_BENCH_BASELINE_GPU_UTIL", "0.90")
    cmd += ["--gpu-memory-utilization", str(gpu_util)]

    for key, value in config.items():
        if key == "gpu-memory-utilization":
            continue
        if key == "attention-backend":
            # Handled via env var, not a CLI flag. See build_config_env().
            continue
        if key == "no-enable-chunked-prefill":
            cmd += ["--no-enable-chunked-prefill"]
            continue
        if isinstance(value, bool):
            if value:
                cmd += [f"--{key}"]
            continue
        cmd += [f"--{key}", str(value)]
    return cmd


def build_config_env(config: dict[str, Any], base_env: dict[str, str]) -> dict[str, str]:
    """Return a copy of base_env with per-config env overrides applied."""
    env = dict(base_env)
    backend = config.get("attention-backend")
    if backend:
        env["VLLM_ATTENTION_BACKEND"] = str(backend)
    # Quality eval is skipped for the whole search — model weights don't change
    # across configs so there's no new quality information to collect.
    env["INFERENCE_BENCH_SKIP_QUALITY"] = "1"
    return env


def run_scenario_eval(
    scenario: str,
    server_url: str,
    base_model: str,
    dataset_seed: int,
    out_file: Path,
    env: dict[str, str],
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Invoke evaluate.py for a single scenario as a subprocess.

    Returns the parsed metrics dict on success, or a stub dict with an
    "error" field on failure.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    task_script = repo_root / "src" / "eval" / "tasks" / scenario / "evaluate.py"
    cmd = [
        sys.executable,
        "-u",
        str(task_script),
        "--server-url",
        server_url,
        "--model",
        base_model,
        "--seed",
        str(dataset_seed),
        "--json-output-file",
        str(out_file),
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(repo_root),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"evaluate.py timed out after {timeout_s}s"}
    elapsed = time.time() - start
    if proc.returncode != 0:
        return {
            "error": f"evaluate.py exited with code {proc.returncode}",
            "wall_s": elapsed,
        }
    if not out_file.exists():
        return {"error": "evaluate.py produced no output", "wall_s": elapsed}
    try:
        return json.loads(out_file.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"unparseable metrics.json: {exc}", "wall_s": elapsed}


def run_one_config(
    config_id: int,
    config: dict[str, Any],
    base_model: str,
    max_model_len: str,
    host: str,
    port: int,
    dataset_seed: int,
    out_root: Path,
    server_env: dict[str, str],
    server_start_timeout_s: float,
    eval_timeout_s: float = 1800.0,
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """Launch vLLM with `config`, evaluate scenarios, return results."""
    config_dir = out_root / f"config_{config_id:03d}"
    config_dir.mkdir(parents=True, exist_ok=True)
    server_log = config_dir / "server.log"

    cmd = build_vllm_cmd(base_model, host, port, max_model_len, config)
    env = build_config_env(config, server_env)
    server_url = f"http://{host}:{port}"

    config_record: dict[str, Any] = {
        "config_id": config_id,
        "config": config,
        "cmd": cmd,
        "server_url": server_url,
        "scenarios": {},
    }

    print(
        f"[random_search] ---- config {config_id} ---- "
        f"overrides={config or 'DEFAULT'}",
        flush=True,
    )
    print(f"[random_search] cmd = {' '.join(cmd)}", flush=True)

    proc = _pb._start_server(cmd, log_path=server_log, env=env)
    try:
        try:
            _pb._wait_ready(
                server_url,
                timeout_s=server_start_timeout_s,
                proc=proc,
                log_path=server_log,
                base_model=base_model,
            )
        except (RuntimeError, TimeoutError) as exc:
            config_record["error"] = f"server not ready: {exc}"
            print(f"[random_search] config {config_id} failed to start: {exc}", flush=True)
            return config_record

        print(
            f"[random_search] config {config_id} server ready at {server_url}",
            flush=True,
        )

        for scenario in (scenarios or SCENARIOS):
            out_file = config_dir / f"{scenario}.json"
            print(f"[random_search] config {config_id} running {scenario}...", flush=True)
            t0 = time.time()
            metrics = run_scenario_eval(
                scenario=scenario,
                server_url=server_url,
                base_model=base_model,
                dataset_seed=dataset_seed,
                out_file=out_file,
                env=env,
                timeout_s=eval_timeout_s,
            )
            elapsed = time.time() - t0
            config_record["scenarios"][scenario] = {
                "metrics_path": str(out_file),
                "wall_s": elapsed,
                "error": metrics.get("error"),
            }
            if "error" in metrics:
                print(
                    f"[random_search] config {config_id} {scenario} ERROR: "
                    f"{metrics['error']}",
                    flush=True,
                )
                continue
    finally:
        print(f"[random_search] config {config_id} tearing down server pid={proc.pid}", flush=True)
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=30)
            except Exception:
                pass

    return config_record


def _env(key: str, fallback: str) -> str:
    """Read an env var, treating HTCondor's 'UNDEFINED' as missing."""
    val = os.environ.get(key, "")
    if not val or val == "UNDEFINED":
        return fallback
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=_env("INFERENCE_BENCH_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
    )
    parser.add_argument(
        "--max-model-len",
        default=_env("INFERENCE_BENCH_MAX_MODEL_LEN", "16384"),
    )
    parser.add_argument(
        "--num-configs",
        type=int,
        default=int(_env("RANDOM_SEARCH_N", "32")),
        help="Number of RANDOM samples (excludes config 0 default and pinned "
             "special configs, which are always run).",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=int(_env("RANDOM_SEARCH_SEED", "248")),
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=int(_env("INFERENCE_BENCH_DATASET_SEED", "248")),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = choose a free port")
    parser.add_argument(
        "--out-root",
        default=_env("RANDOM_SEARCH_OUT_ROOT", "/fast/${USER}/random_search_vllm"),
    )
    parser.add_argument(
        "--server-start-timeout-s",
        type=float,
        default=float(_env("INFERENCE_BENCH_SERVER_START_TIMEOUT_S", "1200")),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip configs with id < this. Useful for resuming after a crash.",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=float(_env("RANDOM_SEARCH_EVAL_TIMEOUT", "1800")),
        help="Per-scenario evaluation timeout in seconds (default 1800). "
             "Scenario B with 64×8192-token outputs needs ~7200.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=_env("RANDOM_SEARCH_SCENARIOS", "").split(",") if _env("RANDOM_SEARCH_SCENARIOS", "") else None,
        help="Run only these scenarios (e.g. inference_scenario_b_output_heavy). "
             "Default: all four. Env: RANDOM_SEARCH_SCENARIOS (comma-separated).",
    )
    parser.add_argument(
        "--config-ids",
        default=_env("RANDOM_SEARCH_CONFIG_IDS", "") or None,
        help="Run only these specific config IDs (comma-separated, e.g. '0,5,7,31'). "
             "IDs refer to the sample_configs() ordering for the given --search-seed. "
             "Useful for targeted re-testing of known-good configs on a new scenario "
             "without rerunning the whole search. Env: RANDOM_SEARCH_CONFIG_IDS.",
    )
    args = parser.parse_args()
    selected_ids: set[int] | None = None
    if args.config_ids:
        selected_ids = {int(x.strip()) for x in args.config_ids.split(",") if x.strip()}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = sample_configs(args.num_configs, args.search_seed)
    print(
        f"[random_search] total configs: {len(configs)} "
        f"(1 default + {args.num_configs} random + {len(SPECIAL_CONFIGS)} special)",
        flush=True,
    )

    # Write manifest up front so the user can see what's about to run even
    # if the job is killed early.
    manifest_path = out_root / "search_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "max_model_len": args.max_model_len,
                "num_random": args.num_configs,
                "search_seed": args.search_seed,
                "dataset_seed": args.dataset_seed,
                "search_space": SEARCH_SPACE,
                "special_configs": SPECIAL_CONFIGS,
                "configs": [
                    {"config_id": i, "config": c} for i, c in enumerate(configs)
                ],
            },
            indent=2,
        )
    )

    # Build the base environment once, mirroring precompute_all_baselines.main().
    runtime_cache_dir = Path(
        os.environ.get(
            "INFERENCE_BENCH_RUNTIME_CACHE_DIR",
            str(_pb._default_runtime_cache_dir()),
        )
    )
    server_env = _pb._build_runtime_cache_env(runtime_cache_dir)
    _pb._install_python_shims(runtime_cache_dir, server_env)

    # Fail-fast: if `fail_fast_threshold` consecutive configs produce zero
    # successful scenarios, abort the whole job. Catches environment-level
    # problems (e.g., the RANDOM_SEARCH_OUT_ROOT isn't bind-mounted into the
    # container and every write to it falls through a writable tmpfs overlay,
    # which produces ENOSPC on the very first scenario of every config).
    # Without this, a broken environment burns hours of GPU time doing
    # nothing.
    fail_fast_threshold = int(_env("RANDOM_SEARCH_FAIL_FAST", "3"))
    consecutive_empty = 0

    suffix = ""
    if args.scenarios:
        suffix = "_" + "_".join(s.split("_")[2] for s in args.scenarios)
    if selected_ids is not None:
        suffix += "_subset"
        print(f"[random_search] subset mode: running only config_ids {sorted(selected_ids)}", flush=True)
    results_path = out_root / f"results{suffix}.jsonl"
    with results_path.open("a") as rfh:
        for config_id, config in enumerate(configs):
            if selected_ids is not None and config_id not in selected_ids:
                continue
            if config_id < args.start_from:
                print(f"[random_search] skipping config {config_id} (< start_from)", flush=True)
                continue
            port = args.port if args.port else _pb._free_port()
            record = run_one_config(
                config_id=config_id,
                config=config,
                base_model=args.base_model,
                max_model_len=args.max_model_len,
                host=args.host,
                port=port,
                dataset_seed=args.dataset_seed,
                out_root=out_root,
                server_env=server_env,
                server_start_timeout_s=args.server_start_timeout_s,
                eval_timeout_s=args.eval_timeout,
                scenarios=args.scenarios,
            )
            rfh.write(json.dumps(record) + "\n")
            rfh.flush()

            # Count how many scenarios on this config produced a parseable
            # metrics file without an error. A config where zero scenarios
            # succeeded is almost always an environment problem, not a bad
            # random draw.
            ok_scenarios = sum(
                1
                for s in record.get("scenarios", {}).values()
                if not s.get("error")
            )
            if record.get("error") or ok_scenarios == 0:
                consecutive_empty += 1
                print(
                    f"[random_search] config {config_id}: 0 ok scenarios "
                    f"(consecutive empty={consecutive_empty}/{fail_fast_threshold})",
                    flush=True,
                )
            else:
                consecutive_empty = 0

            if consecutive_empty >= fail_fast_threshold:
                print(
                    f"[random_search] FAIL-FAST: {consecutive_empty} consecutive "
                    f"configs produced zero successful scenarios. Aborting the "
                    f"whole job. This is almost always an environment problem "
                    f"(e.g. the output directory is not bind-mounted into the "
                    f"container and writes are overflowing the apptainer tmpfs "
                    f"overlay, or the vLLM venv is broken). Check the config "
                    f"0 server.log and stderr above.",
                    flush=True,
                )
                sys.exit(2)

    print(f"[random_search] done. results → {results_path}", flush=True)
    print(f"[random_search] per-config artifacts under {out_root}/config_*", flush=True)


if __name__ == "__main__":
    main()
