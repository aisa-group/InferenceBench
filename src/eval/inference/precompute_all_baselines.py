#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.eval.inference.quality_gate import DEFAULT_SEED

SHIM_DIR = Path(__file__).resolve().parent / "runtime_shims"
BACKENDS = ["vllm", "sglang", "torch", "tgi"]
REQUIRED_MODULES = {"vllm": ["vllm"], "sglang": ["sglang", "orjson", "sgl_kernel"]}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _artifact_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tail(path: Optional[Path], lines: int = 80) -> str:
    if path is None:
        return ""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])
    except OSError:
        return ""


def _probe_ready(server_url: str, base_model: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{server_url}/v1/models", timeout=2)
        if r.status_code == 200:
            return True, "GET /v1/models -> 200"
        get_exc = f"GET /v1/models -> {r.status_code}"
    except Exception as e:
        get_exc = f"{type(e).__name__}:{e}"
    try:
        r = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": base_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=2,
        )
        return r.status_code == 200, f"POST /v1/chat/completions -> {r.status_code}"
    except Exception as e:
        return False, f"no HTTP response (get={get_exc} post={type(e).__name__}:{e})"


def _listeners(pid: int, port: int) -> str:
    try:
        out = subprocess.run(["ss", "-ltnp"], check=False, capture_output=True, text=True).stdout
    except OSError:
        return ""
    pattern = re.compile(rf"pid={pid}\b|:{port}\b")
    return "\n".join(line for line in out.splitlines() if pattern.search(line))


def _with_log_tail(msg: str, log_path: Optional[Path]) -> str:
    tail = _tail(log_path)
    return f"{msg}\n--- server log tail ---\n{tail}" if tail else msg


def _wait_ready(
    server_url: str,
    timeout_s: float = 600.0,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
    base_model: str = "",
    initial_delay_s: float = 30.0,
) -> None:
    start = time.time()
    deadline = start + timeout_s
    delay_deadline = min(start + initial_delay_s, deadline)
    if delay_deadline > start:
        print(f"[wait] waiting {int(delay_deadline - start)}s before the first readiness probe", flush=True)
        while time.time() < delay_deadline and (proc is None or proc.poll() is None):
            time.sleep(min(1.0, max(0.0, delay_deadline - time.time())))
    last_probe = ""
    last_status_s = 0.0
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(_with_log_tail(f"Server process exited early with code={proc.returncode} (url={server_url}).", log_path))
        if time.time() >= deadline:
            break
        ready, last_probe = _probe_ready(server_url, base_model)
        if ready:
            return
        if time.time() - last_status_s >= 30 and time.time() - start >= 5:
            print(f"[wait] backend not ready url={server_url} probe={last_probe} elapsed_s={int(time.time() - start)}", flush=True)
            last_status_s = time.time()
        time.sleep(1)

    msg = f"Server not ready: {server_url} (last_probe={last_probe})."
    if proc is not None:
        msg += "\n--- listening sockets ---\n" + (_listeners(proc.pid, int(server_url.rsplit(":", 1)[1])) or "(none)")
    raise TimeoutError(_with_log_tail(msg, log_path))


def _vllm_tokenizer_mode() -> str:
    return os.environ.get("INFERENCE_BENCH_VLLM_TOKENIZER_MODE", "").strip() or "auto"


def _build_backend_cmd(backend: str, base_model: str, host: str, port: int, max_model_len: str) -> List[str]:
    python = [sys.executable, "-u", "-m"]
    if backend == "vllm":
        return python + [
            "vllm.entrypoints.openai.api_server",
            "--host", host,
            "--port", str(port),
            "--model", base_model,
            "--max-model-len", max_model_len,
            "--tokenizer-mode", _vllm_tokenizer_mode(),
            "--gpu-memory-utilization", os.environ.get("INFERENCE_BENCH_BASELINE_GPU_UTIL", "0.90"),
        ]
    if backend == "torch":
        return python + [
            "src.eval.inference.servers.transformers_openai_server",
            "--host", host,
            "--port", str(port),
            "--model", base_model,
            "--max-model-len", max_model_len,
        ]
    if backend == "sglang":
        return python + [
            "sglang.launch_server",
            "--host", host,
            "--port", str(port),
            "--model-path", base_model,
            "--context-length", max_model_len,
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
    return [
        "text-generation-launcher",
        "--model-id", base_model,
        "--hostname", host,
        "--port", str(port),
        "--cuda-graphs", "0",
    ]


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print(f"[run] {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _default_runtime_cache_dir() -> Path:
    tmpdir = os.environ.get("TMPDIR") or "/tmp"
    return Path(tmpdir) / f"inference_bench_cache_{os.getuid()}"


def _build_runtime_cache_env(runtime_cache_dir: Path) -> Dict[str, str]:
    runtime_cache_dir = runtime_cache_dir.expanduser()
    env = dict(os.environ)
    env.setdefault("XDG_CACHE_HOME", str(runtime_cache_dir / "xdg_cache"))
    env.setdefault("TRITON_CACHE_DIR", str(runtime_cache_dir / "triton"))
    env.setdefault("TRITON_DUMP_DIR", str(runtime_cache_dir / "triton_dump"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(runtime_cache_dir / "torchinductor"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("OUTLINES_CACHE_DIR", str(runtime_cache_dir / "outlines_cache"))
    env.setdefault("DISKCACHE_DIRECTORY", str(runtime_cache_dir / "outlines_cache"))
    env.setdefault("FLASHINFER_CACHE_DIR", str(runtime_cache_dir / "flashinfer_cache"))
    env.setdefault("FLASHINFER_JIT_CACHE_DIR", str(runtime_cache_dir / "flashinfer_cache"))
    env.setdefault("FLASHINFER_BUILD_DIR", str(runtime_cache_dir / "flashinfer_cache"))
    env.setdefault("FLASHINFER_TMPDIR", str(runtime_cache_dir / "flashinfer_tmp"))
    env.setdefault("HF_HOME", str(runtime_cache_dir / "hf_home"))
    env.setdefault("CUDA_CACHE_PATH", str(runtime_cache_dir / "cuda_cache"))
    return env


def _install_python_shims(env: Dict[str, str]) -> None:
    if os.environ.get("INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS", "").strip() == "1":
        return
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SHIM_DIR}{os.pathsep}{existing}" if existing else str(SHIM_DIR)


def _start_server(cmd: List[str], log_path: Path, env: Dict[str, str]) -> subprocess.Popen:
    runtime_cache_dir = Path(env.get("TRITON_CACHE_DIR", "")).expanduser().parent
    for name in ("triton", "torchinductor", "xdg_cache", "outlines_cache", "hf_home", "cuda_cache"):
        (runtime_cache_dir / name).mkdir(parents=True, exist_ok=True)
    cwd = None
    if env.get("INFERENCE_BENCH_ACTIVE_BACKEND") == "tgi" or Path(cmd[0]).name == "text-generation-launcher":
        cwd = tempfile.mkdtemp(prefix="tgi_server_", dir=runtime_cache_dir.resolve())
        python_paths = [str(_repo_root())]
        python_paths.extend(str(Path(path or ".").resolve()) for path in env.get("PYTHONPATH", "").split(os.pathsep))
        env = {**env, "PYTHONPATH": os.pathsep.join(python_paths)}
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)


def _stop_server(proc: Optional[subprocess.Popen], timeout_s: float = 30.0) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=timeout_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute quality samples, speed baselines, and quality baselines.")
    parser.add_argument("--base-model", type=str, default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    parser.add_argument("--max-model-len", type=str, default=os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "32768"))
    parser.add_argument("--dataset-seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", DEFAULT_SEED)))
    parser.add_argument("--quality-seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_SEED", os.environ.get("INFERENCE_BENCH_DATASET_SEED", DEFAULT_SEED))))
    parser.add_argument("--mmlupro-n", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_N", "500")))
    parser.add_argument("--quality-concurrency", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_CONCURRENCY", "4")))
    parser.add_argument("--scenarios", type=str, nargs="*", default=[
        "inference_scenario_a_input_heavy",
        "inference_scenario_b_output_heavy",
        "inference_scenario_c_high_load",
        "inference_scenario_d_general",
    ])
    parser.add_argument("--backends", type=str, nargs="*", choices=BACKENDS, default=["vllm"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 chooses a free port.")
    parser.add_argument("--server-start-timeout-s", type=float, default=600.0)
    parser.add_argument("--cache-quality-samples", action="store_true")
    parser.add_argument("--skip-speed", action="store_true", default=os.environ.get("INFERENCE_BENCH_SKIP_SPEED", "0") == "1")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--quality-registry-suffix", type=str, default="")
    parser.add_argument("--runtime-cache-dir", type=str, default=os.environ.get("INFERENCE_BENCH_RUNTIME_CACHE_DIR", ""))
    args = parser.parse_args()
    if not args.skip_speed and "tgi" in args.backends:
        parser.error("TGI cannot ignore EOS, so TGI speed baselines are unsupported. Use --skip-speed.")

    repo_root = _repo_root()
    os.chdir(repo_root)
    baselines = repo_root / "src" / "eval" / "inference" / "baselines"
    model_safe = _safe_name(args.base_model)
    runtime_cache_dir = Path(args.runtime_cache_dir).expanduser() if args.runtime_cache_dir else _default_runtime_cache_dir()
    server_env = _build_runtime_cache_env(runtime_cache_dir)
    _install_python_shims(server_env)

    if args.cache_quality_samples:
        _run([
            sys.executable, "-m", "src.eval.inference.cache_samples",
            "--seed", str(args.quality_seed),
            "--mmlupro-n", str(args.mmlupro_n),
        ], env=server_env)

    for backend in args.backends:
        port = args.port or _free_port()
        server_url = f"http://{args.host}:{port}"
        cmd = _build_backend_cmd(backend, args.base_model, args.host, port, args.max_model_len)
        for module in REQUIRED_MODULES.get(backend, []):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"backend={backend} cannot import {module}")

        log_dir = baselines / "_server_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{backend}_{port}.log"
        backend_server_env = {**server_env, "INFERENCE_BENCH_ACTIVE_BACKEND": backend}
        if backend == "vllm":
            backend_server_env.setdefault("VLLM_NO_USAGE_STATS", "1")

        print(f"[server] starting backend={backend} url={server_url} log={log_path} cache_dir={runtime_cache_dir}")
        proc = _start_server(cmd, log_path=log_path, env=backend_server_env)
        try:
            _wait_ready(server_url, timeout_s=args.server_start_timeout_s, proc=proc, log_path=log_path, base_model=args.base_model)
            print(f"[server] ready backend={backend} url={server_url}")

            if not args.skip_speed:
                speed_dir = baselines / "speed" / backend
                for scenario in args.scenarios:
                    precompute_cmd = [
                        sys.executable, "-m", "src.eval.inference.precompute_baseline",
                        "--scenario-id", scenario,
                        "--base-model", args.base_model,
                        "--server-url", server_url,
                        "--out-root", str(speed_dir),
                        "--registry", str(speed_dir / f"{model_safe}.json"),
                        "--seed", str(args.dataset_seed),
                        "--request-timeout-s", "900" if backend == "torch" else "300",
                    ]
                    if backend == "torch":
                        precompute_cmd += ["--concurrency-override", "1"]
                    _run(precompute_cmd, env=server_env)

            if not args.skip_quality:
                name_parts = [model_safe, args.quality_registry_suffix, "" if backend == "vllm" else backend]
                _run([
                    sys.executable, "-m", "src.eval.inference.precompute_quality_baseline",
                    "--server-url", server_url,
                    "--base-model", args.base_model,
                    "--backend", backend,
                    "--registry", str(baselines / "quality" / ("_".join(p for p in name_parts if p) + ".json")),
                    "--out-root", str(baselines / "quality" / backend / model_safe),
                    "--seed", str(args.quality_seed),
                    "--mmlupro-n", str(args.mmlupro_n),
                    "--concurrency", "1" if backend == "torch" else str(args.quality_concurrency),
                ], env=server_env)
        finally:
            print(f"[server] stopping backend={backend} pid={proc.pid}")
            _stop_server(proc)


if __name__ == "__main__":
    main()
