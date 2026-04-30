#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import signal
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _tail(path: Path, lines: int = 80) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not data:
            return ""
        return "\n".join(data[-lines:])
    except Exception:
        return ""


def _probe_ready(server_url: str, base_model: str) -> tuple[bool, str]:
    # Primary probe: OpenAI-compatible model listing.
    try:
        r = requests.get(f"{server_url}/v1/models", timeout=2)
        if r.status_code == 200:
            return True, "GET /v1/models -> 200"
        return False, f"GET /v1/models -> {r.status_code}"
    except Exception as e:
        get_exc = f"{type(e).__name__}:{e}"
    else:
        get_exc = ""

    # Fallback probe: OpenAI-compatible chat completions endpoint.
    # Some servers don't implement /v1/models but do implement /v1/chat/completions.
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
        # 200: ready.
        # 4xx: endpoint exists and server is up; request may be rejected for policy/model reasons
        #      but the server is responsive.
        if r.status_code == 200 or (400 <= r.status_code < 500 and r.status_code != 404):
            return True, f"POST /v1/chat/completions -> {r.status_code}"
        return False, f"POST /v1/chat/completions -> {r.status_code}"
    except Exception as e:
        post_exc = f"{type(e).__name__}:{e}"
    else:
        post_exc = ""

    if get_exc or post_exc:
        return False, f"no HTTP response (get={get_exc or 'ok'} post={post_exc or 'ok'})"
    return False, "no HTTP response (connection refused/timeout)"


def _proc_diagnostics(pid: int, port: int) -> str:
    parts: List[str] = []
    try:
        ps = subprocess.run(
            ["bash", "-lc", f"ps -o pid=,ppid=,stat=,etime=,cmd= -p {pid} || true"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if ps:
            parts.append("[proc] " + ps)
    except Exception:
        pass

    try:
        ss = subprocess.run(
            ["bash", "-lc", f"ss -ltnp 2>/dev/null | grep -E \"pid={pid}\\b\" || true"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if ss:
            parts.append("[listen]\n" + ss)
        else:
            parts.append("[listen] (no listening TCP sockets found for pid)")
    except Exception:
        pass

    try:
        ss_port = subprocess.run(
            ["bash", "-lc", f"ss -ltnp 2>/dev/null | grep -E \":{port}\\b\" || true"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if ss_port:
            parts.append(f"[port {port}]\n" + ss_port)
        else:
            parts.append(f"[port {port}] (no listeners found)")
    except Exception:
        pass

    return "\n".join(parts).strip()


def _wait_ready(
    server_url: str,
    timeout_s: float = 600.0,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
    base_model: str = "",
    allow_sigusr1_dump: bool = False,
    initial_delay_s: float = 30.0,
) -> None:
    deadline = time.time() + timeout_s
    last_probe = ""
    start = time.time()
    last_dump_s = 0.0
    last_status_s = 0.0
    # Wait before the first probe — servers like vLLM need time to load
    # model weights and compile kernels before they can respond to health checks.
    if initial_delay_s > 0:
        remaining = min(initial_delay_s, deadline - time.time())
        if remaining > 0:
            print(f"[wait] waiting {int(remaining)}s for server startup before first probe...", flush=True)
            time.sleep(remaining)
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _tail(log_path) if log_path is not None else ""
            msg = f"Server process exited early with code={proc.returncode} (url={server_url})."
            if tail:
                msg += "\n--- server log tail ---\n" + tail
            raise RuntimeError(msg)
        ready, detail = _probe_ready(server_url, base_model)
        last_probe = detail
        if ready:
            return

        # Periodic status so HTCondor logs show progress even if the backend is silent.
        if time.time() - last_status_s >= 30 and time.time() - start >= 5:
            print(f"[wait] backend not ready yet url={server_url} probe={last_probe} elapsed_s={int(time.time() - start)}", flush=True)
            last_status_s = time.time()

        # If the backend is hung during import/model init, periodically ask it to dump
        # Python stack traces into the log (only works if faulthandler SIGUSR1 is registered).
        if (
            allow_sigusr1_dump
            and proc is not None
            and os.environ.get("INFERENCE_BENCH_DEBUG_STACK_DUMP", "").strip() == "1"
            and time.time() - start >= 30
            and time.time() - last_dump_s >= 30
        ):
            try:
                # Best-effort: dump stacks for the server process (and any children in its
                # process group, if supported).
                try:
                    os.killpg(proc.pid, signal.SIGUSR1)  # type: ignore[attr-defined]
                except Exception:
                    os.kill(proc.pid, signal.SIGUSR1)
                last_dump_s = time.time()
            except Exception:
                pass
        time.sleep(1)
    # If the process exited right near the deadline, report that as an early-exit error
    # (more actionable than "not ready").
    if proc is not None and proc.poll() is not None:
        tail = _tail(log_path) if log_path is not None else ""
        msg = f"Server process exited early with code={proc.returncode} (url={server_url}) at timeout boundary."
        if tail:
            msg += "\n--- server log tail ---\n" + tail
        raise RuntimeError(msg)

    tail = _tail(log_path) if log_path is not None else ""

    diag = _proc_diagnostics(proc.pid, int(server_url.rsplit(":", 1)[1])) if proc is not None else ""
    msg = f"Server not ready: {server_url} (last_probe={last_probe})."
    if diag:
        msg += "\n--- proc diagnostics ---\n" + diag
    if tail:
        msg += "\n--- server log tail ---\n" + tail
    raise TimeoutError(msg)


@dataclass(frozen=True)
class Backend:
    name: str
    kind: str  # "vllm" | "sglang" | "torch" | "tgi" | "custom"
    cmd: Optional[str] = None


def _which(cmd: str) -> bool:
    return subprocess.call(["bash", "-lc", f"command -v {shlex.quote(cmd)} >/dev/null 2>&1"]) == 0


def _vllm_tokenizer_mode(base_model: str) -> str:
    override = os.environ.get("INFERENCE_BENCH_VLLM_TOKENIZER_MODE", "").strip()
    if override:
        return override
    return "auto"


def _build_backend_cmd(backend: Backend, base_model: str, host: str, port: int, max_model_len: str) -> List[str]:
    if backend.kind == "vllm":
        tokenizer_mode = _vllm_tokenizer_mode(base_model)
        return [
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
            "--gpu-memory-utilization",
            os.environ.get("INFERENCE_BENCH_BASELINE_GPU_UTIL", "0.90"),
        ]
    if backend.kind == "torch":
        return [
            sys.executable,
            "-u",
            "-m",
            "src.eval.inference.servers.transformers_openai_server",
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            base_model,
            "--max-model-len",
            max_model_len,
        ]
    if backend.kind == "sglang":
        # sglang captures CUDA graphs for 36 batch sizes (up to bs=256) by default.
        # For 7B+ models at 32k context, capturing the largest graphs exceeds the
        # remaining VRAM budget after KV allocation and stalls indefinitely.
        # Disabling graph capture keeps startup bounded (seconds) at the cost of
        # a small decode-throughput hit; same trade-off we apply to the TGI branch.
        return [
            sys.executable,
            "-u",
            "-m",
            "sglang.launch_server",
            "--host",
            host,
            "--port",
            str(port),
            "--model-path",
            base_model,
            "--context-length",
            max_model_len,
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
        ]
    if backend.kind == "tgi":
        # Letting TGI choose its own max-total-tokens avoids aggressive warmup kernels
        # that device-side-assert during block_tables_to_ragged on H100 + cu128 triton.
        # --cuda-graphs 0 keeps the warmup path simple (no graph capture).
        return [
            "text-generation-launcher",
            "--model-id",
            base_model,
            "--hostname",
            host,
            "--port",
            str(port),
            "--cuda-graphs",
            "0",
        ]
    if backend.kind == "custom":
        if not backend.cmd:
            raise ValueError("custom backend requires cmd")
        cmd = backend.cmd.format(model=base_model, host=host, port=port, max_model_len=max_model_len)
        return ["bash", "-lc", cmd]
    raise ValueError(f"unknown backend kind: {backend.kind}")


def _parse_backends(raw: Sequence[str], backend_cmds: Sequence[str]) -> List[Backend]:
    backends: List[Backend] = []
    for name in raw:
        if name in {"vllm", "sglang", "torch", "tgi"}:
            backends.append(Backend(name=name, kind=name))
        elif name.startswith("custom:"):
            # custom:<id>
            backends.append(Backend(name=name.split(":", 1)[1], kind="custom"))
        else:
            raise ValueError(f"Unknown backend: {name}")

    cmd_map: Dict[str, str] = {}
    for item in backend_cmds:
        # <id>=<cmd>
        if "=" not in item:
            raise ValueError(f"--backend-cmd must be id=cmd, got: {item}")
        k, v = item.split("=", 1)
        cmd_map[k.strip()] = v.strip()

    out: List[Backend] = []
    for b in backends:
        if b.kind == "custom":
            cmd = cmd_map.get(b.name, "")
            if not cmd:
                raise ValueError(f"custom backend '{b.name}' missing --backend-cmd {b.name}=...")
            out.append(Backend(name=b.name, kind="custom", cmd=cmd))
        else:
            out.append(b)
    return out


def _canonical_speed_requests(scenario: str) -> Optional[Path]:
    path = _repo_root() / "src" / "eval" / "inference" / "baselines" / "speed" / "default" / scenario / "requests.jsonl"
    return path if path.exists() else None


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
    # Outlines (used by SGLang constrained decoding) uses diskcache/sqlite. Force it onto
    # the same runtime cache dir (avoid writing into the repo or a small tmpfs overlay).
    env.setdefault("OUTLINES_CACHE_DIR", str(runtime_cache_dir / "outlines_cache"))
    # Some versions may consult diskcache-specific env vars; harmless if unused.
    env.setdefault("DISKCACHE_DIRECTORY", str(runtime_cache_dir / "outlines_cache"))
    # FlashInfer JIT builds kernels on first use; keep it off tmpfs overlays.
    # We set a few common env var names used across FlashInfer releases.
    env.setdefault("FLASHINFER_CACHE_DIR", str(runtime_cache_dir / "flashinfer_cache"))
    env.setdefault("FLASHINFER_JIT_CACHE_DIR", str(runtime_cache_dir / "flashinfer_cache"))
    env.setdefault("FLASHINFER_BUILD_DIR", str(runtime_cache_dir / "flashinfer_cache"))
    env.setdefault("FLASHINFER_TMPDIR", str(runtime_cache_dir / "flashinfer_tmp"))
    # Prefer HF_HOME over TRANSFORMERS_CACHE (deprecated), but don't unset user overrides.
    env.setdefault("HF_HOME", str(runtime_cache_dir / "hf_home"))
    env.setdefault("CUDA_CACHE_PATH", str(runtime_cache_dir / "cuda_cache"))
    return env


def _install_python_shims(runtime_cache_dir: Path, env: Dict[str, str]) -> None:
    """Install small import-time shims for version drift without rebuilding the container.

    The shims are applied to all backend server subprocesses by prepending the shim
    directory to PYTHONPATH (so `sitecustomize` executes on startup).
    """
    if os.environ.get("INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS", "").strip() == "1":
        return

    shim_dir = (runtime_cache_dir.expanduser() / "python_shims").resolve()
    shim_dir.mkdir(parents=True, exist_ok=True)

    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "# PTB_RUNTIME_SHIMS",
                "# Best-effort shims for runtime compatibility in baseline precompute.",
                "#",
                "# Also registers SIGUSR1 stack dumps via faulthandler to debug startup hangs.",
                "try:",
                "    import faulthandler, signal",
                "    faulthandler.enable(all_threads=True)",
                "    faulthandler.register(signal.SIGUSR1, all_threads=True)",
                "except Exception:",
                "    pass",
                "",
                "# accelerate can circular-import when bnb.py tries to import dispatch_model",
                "# from big_modeling before it finishes initializing.  Pre-register a stub",
                "# module in sys.modules so the 'from ..big_modeling import dispatch_model'",
                "# inside bnb.py finds lazy wrappers instead of hitting the partial module.",
                "try:",
                "    import sys as _sys, types as _types",
                "    if 'accelerate.utils.bnb' not in _sys.modules and 'accelerate' not in _sys.modules:",
                "        _bnb_stub = _types.ModuleType('accelerate.utils.bnb')",
                "        _bnb_stub.__file__ = '<ptb_accelerate_bnb_stub>'",
                "        _bnb_stub.__package__ = 'accelerate.utils'",
                "        def _lazy_dispatch_model(*a, **kw):",
                "            from accelerate.big_modeling import dispatch_model",
                "            return dispatch_model(*a, **kw)",
                "        def _lazy_init_empty_weights(*a, **kw):",
                "            from accelerate.big_modeling import init_empty_weights",
                "            return init_empty_weights(*a, **kw)",
                "        def _lazy_has_4bit(*a, **kw):",
                "            return False",
                "        def _lazy_load_and_quantize(*a, **kw):",
                "            raise NotImplementedError('bnb stub: load_and_quantize_model')",
                "        _bnb_stub.dispatch_model = _lazy_dispatch_model",
                "        _bnb_stub.init_empty_weights = _lazy_init_empty_weights",
                "        _bnb_stub.has_4bit_bnb_layers = _lazy_has_4bit",
                "        _bnb_stub.load_and_quantize_model = _lazy_load_and_quantize",
                "        _sys.modules['accelerate.utils.bnb'] = _bnb_stub",
                "except Exception:",
                "    pass",
                "",
                "# Outlines can circular-import in some images, crashing SGLang import even",
                "# when constrained decoding is not used. Provide a lightweight stub.",
                "try:",
                "    import os, sys, types",
                "    if os.environ.get('INFERENCE_BENCH_STUB_OUTLINES', '1').strip() != '0':",
                "        for _k in list(sys.modules.keys()):",
                "            if _k == 'outlines' or _k.startswith('outlines.'):",
                "                sys.modules.pop(_k, None)",
                "        def _ptb_outlines_any_attr(name: str):",
                "            # Return a no-op function by default; constrained decoding isn't used in speed baselines.",
                "            def _stub(*_a, **_k):",
                "                return '.*'",
                "            return _stub",
                "",
                "        def _ptb_outlines_make_mod(fullname: str):",
                "            m = types.ModuleType(fullname)",
                "            m.__dict__.setdefault('__all__', [])",
                "            m.__dict__.setdefault('__file__', '<ptb_outlines_stub>')",
                "            # Treat as package if it is a parent namespace.",
                "            if fullname in {'outlines', 'outlines.fsm'}:",
                "                m.__path__ = []  # type: ignore[attr-defined]",
                "            def __getattr__(attr: str):",
                "                if attr.startswith('__'):",
                "                    raise AttributeError(attr)",
                "                sub = f'{fullname}.{attr}'",
                "                if sub not in sys.modules and (fullname == 'outlines' or fullname.startswith('outlines.')):",
                "                    sys.modules[sub] = _ptb_outlines_make_mod(sub)",
                "                    return sys.modules[sub]",
                "                return _ptb_outlines_any_attr(attr)",
                "            m.__getattr__ = __getattr__  # type: ignore[attr-defined]",
                "            return m",
                "",
                "        outlines = _ptb_outlines_make_mod('outlines')",
                "        fsm = _ptb_outlines_make_mod('outlines.fsm')",
                "        guide = _ptb_outlines_make_mod('outlines.fsm.guide')",
                "        json_schema = _ptb_outlines_make_mod('outlines.fsm.json_schema')",
                "        class RegexGuide:  # type: ignore",
                "            def __init__(self, *args, **kwargs):",
                "                self._stub = True",
                "            @classmethod",
                "            def from_regex(cls, *args, **kwargs):",
                "                return cls()",
                "        class CFGGuide:  # type: ignore",
                "            def __init__(self, *args, **kwargs):",
                "                self._stub = True",
                "        guide.RegexGuide = RegexGuide  # type: ignore[attr-defined]",
                "        guide.CFGGuide = CFGGuide  # type: ignore[attr-defined]",
                "        def build_regex_from_schema(*_a, **_k):",
                "            return '.*'",
                "        json_schema.build_regex_from_schema = build_regex_from_schema  # type: ignore[attr-defined]",
                "",
                "        # SGLang's outlines backend also imports `outlines.models.transformers.TransformerTokenizer`.",
                "        # Provide a minimal stub to satisfy import-time requirements.",
                "        models = _ptb_outlines_make_mod('outlines.models')",
                "        models.__path__ = []  # type: ignore[attr-defined]",
                "        models_transformers = _ptb_outlines_make_mod('outlines.models.transformers')",
                "        class TransformerTokenizer:  # type: ignore",
                "            def __init__(self, *args, **kwargs):",
                "                self._stub = True",
                "        models_transformers.TransformerTokenizer = TransformerTokenizer  # type: ignore[attr-defined]",
                "",
                "        sys.modules['outlines'] = outlines",
                "        sys.modules['outlines.fsm'] = fsm",
                "        sys.modules['outlines.fsm.guide'] = guide",
                "        sys.modules['outlines.fsm.json_schema'] = json_schema",
                "        sys.modules['outlines.models'] = models",
                "        sys.modules['outlines.models.transformers'] = models_transformers",
                "except Exception:",
                "    pass",
                "",
                "# Covers common import-time failures on clusters where deps drift:",
                "# - Triton cache API names (default_cache_dir/default_dump_dir/default_*_dir)",
                "# - Transformers top-level AutoProcessor export (SGLang expects it)",
                "# - Triton language constexpr_function (some kernels expect it)",
                "# - FlashInfer private symbols imported by SGLang",
                "try:",
                "    import triton.runtime.cache as _triton_cache  # type: ignore",
                "    def _ptb_triton_root():",
                "        import os",
                "        from pathlib import Path",
                "        d = os.environ.get('TRITON_CACHE_DIR')",
                "        if d:",
                "            p = Path(d).expanduser()",
                "            return p.parent",
                "        xdg = os.environ.get('XDG_CACHE_HOME')",
                "        base = Path(xdg) if xdg else (Path.home() / '.cache')",
                "        return base",
                "",
                "    def _ptb_default_dir(kind: str):",
                "        import os",
                "        from pathlib import Path",
                "        env_key = f'TRITON_{kind.upper()}_DIR'",
                "        d = os.environ.get(env_key)",
                "        if d:",
                "            p = Path(d).expanduser()",
                "            p.mkdir(parents=True, exist_ok=True)",
                "            return str(p)",
                "        root = _ptb_triton_root()",
                "        name = 'triton' if kind == 'cache' else f'triton_{kind}'",
                "        p = Path(root) / name",
                "        p.mkdir(parents=True, exist_ok=True)",
                "        return str(p)",
                "",
                "    def default_cache_dir():",
                "        return _ptb_default_dir('cache')",
                "",
                "    def default_dump_dir():",
                "        return _ptb_default_dir('dump')",
                "",
                "    def __getattr__(name: str):",
                "        if name == 'default_cache_dir':",
                "            return default_cache_dir",
                "        if name == 'default_dump_dir':",
                "            return default_dump_dir",
                "        if name.startswith('default_') and name.endswith('_dir'):",
                "            kind = name[len('default_'):-len('_dir')]",
                "            def _f():",
                "                return _ptb_default_dir(kind)",
                "            return _f",
                "        raise AttributeError(name)",
                "",
                "    if not hasattr(_triton_cache, 'default_cache_dir'):",
                "        _triton_cache.default_cache_dir = default_cache_dir  # type: ignore[attr-defined]",
                "    if not hasattr(_triton_cache, 'default_dump_dir'):",
                "        _triton_cache.default_dump_dir = default_dump_dir  # type: ignore[attr-defined]",
                "    if not hasattr(_triton_cache, '__getattr__'):",
                "        _triton_cache.__getattr__ = __getattr__  # type: ignore[attr-defined]",
                "except Exception:",
                "    pass",
                "",
                "# Some deployments ship a Transformers build where AutoProcessor isn't exported",
                "# at top-level. SGLang imports it from `transformers`, so re-export if present.",
                "try:",
                "    import transformers  # type: ignore",
                "    if not hasattr(transformers, 'AutoProcessor'):",
                "        try:",
                "            # Prefer the real class if it can be imported cleanly.",
                "            from transformers.models.auto.processing_auto import AutoProcessor as _RealAutoProcessor  # type: ignore",
                "            transformers.AutoProcessor = _RealAutoProcessor  # type: ignore[attr-defined]",
                "        except Exception:",
                "            # Otherwise provide a proxy that resolves on first use.",
                "            class _AutoProcessorProxy:  # type: ignore",
                "                @classmethod",
                "                def from_pretrained(cls, *args, **kwargs):",
                "                    from transformers.models.auto.processing_auto import AutoProcessor as _AP  # type: ignore",
                "                    return _AP.from_pretrained(*args, **kwargs)",
                "            transformers.AutoProcessor = _AutoProcessorProxy  # type: ignore[attr-defined]",
                "except Exception:",
                "    pass",
                "",
                "# vLLM 0.11 expects tokenizer.all_special_tokens_extended, but some",
                "# Transformers builds/tokenizer backends only expose all_special_tokens.",
                "# Provide a compatibility fallback to avoid startup crashes.",
                "try:",
                "    import transformers.tokenization_utils_base as _tub  # type: ignore",
                "    _PTB_TOKEN_ATTR = 'all_special_tokens_extended'",
                "    _PTB_FALLBACK = 'all_special_tokens'",
                "",
                "    if hasattr(_tub, 'PreTrainedTokenizerBase') and not hasattr(_tub.PreTrainedTokenizerBase, _PTB_TOKEN_ATTR):",
                "        @property",
                "        def _ptb_all_special_tokens_extended(self):  # type: ignore",
                "            try:",
                "                vals = getattr(self, _PTB_FALLBACK, [])",
                "                return list(vals) if vals is not None else []",
                "            except Exception:",
                "                return []",
                "        _tub.PreTrainedTokenizerBase.all_special_tokens_extended = _ptb_all_special_tokens_extended  # type: ignore[attr-defined]",
                "",
                "    if hasattr(_tub, 'SpecialTokensMixin') and hasattr(_tub.SpecialTokensMixin, '__getattr__'):",
                "        _orig_getattr = _tub.SpecialTokensMixin.__getattr__  # type: ignore[attr-defined]",
                "        if not getattr(_orig_getattr, '_ptb_wrapped_all_special_tokens_extended', False):",
                "            def _ptb_getattr(self, key):  # type: ignore",
                "                if key == _PTB_TOKEN_ATTR:",
                "                    try:",
                "                        vals = getattr(self, _PTB_FALLBACK, [])",
                "                        return list(vals) if vals is not None else []",
                "                    except Exception:",
                "                        return []",
                "                return _orig_getattr(self, key)",
                "            _ptb_getattr._ptb_wrapped_all_special_tokens_extended = True  # type: ignore[attr-defined]",
                "            _tub.SpecialTokensMixin.__getattr__ = _ptb_getattr  # type: ignore[attr-defined]",
                "except Exception:",
                "    pass",
                "",
                "# vLLM 0.11 + newer huggingface_hub/tqdm combos may pass `disable` twice into",
                "# vLLM's DisabledTqdm wrapper, causing:",
                "#   TypeError: ... got multiple values for keyword argument 'disable'",
                "# Strip user-provided disable before delegating (vLLM will force disable=True).",
                "try:",
                "    import importlib",
                "    try:",
                "        _wu = importlib.import_module('vllm.model_executor.model_loader.weight_utils')",
                "        _dt = getattr(_wu, 'DisabledTqdm', None)",
                "        if _dt is not None and hasattr(_dt, '__init__'):",
                "            _orig_init = _dt.__init__  # type: ignore[attr-defined]",
                "            if not getattr(_orig_init, '_ptb_disable_kw_fix', False):",
                "                def _init(self, *args, _orig_init=_orig_init, **kwargs):  # type: ignore",
                "                    kwargs.pop('disable', None)",
                "                    return _orig_init(self, *args, **kwargs)",
                "                _init._ptb_disable_kw_fix = True  # type: ignore[attr-defined]",
                "                _dt.__init__ = _init  # type: ignore[attr-defined]",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# If sgl_kernel extension is ABI-incompatible with the active torch build,",
                "# SGLang can crash while importing optional quantization kernels.",
                "# For non-quantized baseline runs, inject a lightweight fallback module.",
                "try:",
                "    import os, sys, types",
                "    if os.environ.get('INFERENCE_BENCH_SGLANG_STUB_SGL_KERNEL', '1').strip() != '0':",
                "        try:",
                "            import sgl_kernel  # noqa: F401",
                "        except Exception:",
                "            _sgk = types.ModuleType('sgl_kernel')",
                "            _sgk.__file__ = '<ptb_sgl_kernel_stub>'  # type: ignore[attr-defined]",
                "            def _ptb_sgl_kernel_unavailable(*_a, **_k):",
                "                raise RuntimeError('sgl_kernel extension unavailable in this runtime')",
                "            def __getattr__(name: str):",
                "                return _ptb_sgl_kernel_unavailable",
                "            _sgk.__getattr__ = __getattr__  # type: ignore[attr-defined]",
                "            _sgk.int8_scaled_mm = _ptb_sgl_kernel_unavailable  # type: ignore[attr-defined]",
                "            sys.modules['sgl_kernel'] = _sgk",
                "except Exception:",
                "    pass",
                "",
                "# Some kernels expect `triton.language.constexpr_function` (not present in older Triton).",
                "try:",
                "    import triton.language as tl  # type: ignore",
                "    if not hasattr(tl, 'constexpr_function'):",
                "        def constexpr_function(fn=None, **_kwargs):  # type: ignore",
                "            if fn is None:",
                "                def _wrap(f):",
                "                    return f",
                "                return _wrap",
                "            return fn",
                "        tl.constexpr_function = constexpr_function  # type: ignore[attr-defined]",
                "except Exception:",
                "    pass",
                "",
                "# SGLang imports a few private FlashInfer symbols; provide safe fallbacks if missing.",
                "# IMPORTANT: importing `flashinfer` at interpreter startup can trigger long JIT builds",
                "# (e.g., tvm_ffi torch-c-dlpack extension compilation) and make the server look hung.",
                "# Only patch FlashInfer if the run is explicitly configured to use it.",
                "try:",
                "    import importlib.machinery",
                "    import os",
                "    import sys",
                "    import types",
                "    _active_backend = os.environ.get('INFERENCE_BENCH_ACTIVE_BACKEND', '').strip().lower()",
                "    _is_sglang = (_active_backend == 'sglang')",
                "    _want_flashinfer = (_is_sglang and (",
                "        os.environ.get('INFERENCE_BENCH_SGLANG_ATTENTION_BACKEND', '').strip() == 'flashinfer'",
                "        or os.environ.get('INFERENCE_BENCH_SGLANG_SAMPLING_BACKEND', '').strip() == 'flashinfer'",
                "    ))",
                "    def _ptb_flashinfer_fallback(*_args, **_kwargs):  # type: ignore",
                "        return False",
                "    if _is_sglang:",
                "        import flashinfer.decode as _fid  # type: ignore",
                "        for _name in ('_grouped_size_compiled_for_decode_kernels', '_grouped_size_compiled_for_prefill_kernels'):",
                "            _cur = getattr(_fid, _name, None)",
                "            if _cur is None:",
                "                setattr(_fid, _name, _ptb_flashinfer_fallback)  # type: ignore[attr-defined]",
                "            elif not callable(_cur):",
                "                if isinstance(_cur, set):",
                "                    def _in_set(x, *_a, _s=_cur, **_k):  # type: ignore",
                "                        try:",
                "                            return x in _s",
                "                        except Exception:",
                "                            return False",
                "                    setattr(_fid, _name, _in_set)  # type: ignore[attr-defined]",
                "                else:",
                "                    setattr(_fid, _name, _ptb_flashinfer_fallback)  # type: ignore[attr-defined]",
                "        if not hasattr(_fid, '__getattr__'):",
                "            def __getattr__(name: str):",
                "                if name.startswith('_grouped_size_compiled_for_') and name.endswith('_kernels'):",
                "                    return _ptb_flashinfer_fallback",
                "                raise AttributeError(name)",
                "            _fid.__getattr__ = __getattr__  # type: ignore[attr-defined]",
                "    else:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Some SGLang versions call `nvidia-smi` at startup to infer GPU memory, and crash",
                "# if it isn't available inside the container (common under Apptainer --nv).",
                "# Patch the helper to use torch.cuda device properties instead.",
                "try:",
                "    import importlib",
                "    def _ptb_first_responsive_cuda_idx():",
                "        import torch",
                "        if not torch.cuda.is_available():",
                "            return None",
                "        n = int(torch.cuda.device_count() or 0)",
                "        for idx in range(n):",
                "            try:",
                "                _ = torch.cuda.get_device_properties(idx)",
                "                torch.empty(1, device=f'cuda:{idx}')",
                "                return idx",
                "            except Exception:",
                "                continue",
                "        return None",
                "",
                "    def _ptb_gpu_mem_mib():",
                "        import torch",
                "        idx = _ptb_first_responsive_cuda_idx()",
                "        if idx is None:",
                "            raise RuntimeError('CUDA is not available (are you on a GPU node with Apptainer --nv?)')",
                "        props = torch.cuda.get_device_properties(idx)",
                "        return int(getattr(props, 'total_memory', 0) / (1024 * 1024))",
                "",
                "    def _ptb_patch_sglang_nvsmI(mod):",
                "        try:",
                "            if hasattr(mod, 'get_nvgpu_memory_capacity'):",
                "                mod.get_nvgpu_memory_capacity = _ptb_gpu_mem_mib  # type: ignore[attr-defined]",
                "        except Exception:",
                "            pass",
                "",
                "    try:",
                "        _ptb_patch_sglang_nvsmI(importlib.import_module('sglang.srt.utils'))",
                "    except Exception:",
                "        pass",
                "    try:",
                "        sa = importlib.import_module('sglang.srt.server_args')",
                "        if hasattr(sa, 'get_nvgpu_memory_capacity'):",
                "            sa.get_nvgpu_memory_capacity = _ptb_gpu_mem_mib  # type: ignore[attr-defined]",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Default attention backend to Triton for cluster robustness.",
                "# Override with INFERENCE_BENCH_SGLANG_ATTENTION_BACKEND (e.g. flashinfer).",
                "try:",
                "    import importlib",
                "    import os",
                "    try:",
                "        sa = importlib.import_module('sglang.srt.server_args')",
                "        if hasattr(sa, 'prepare_server_args'):",
                "            _orig_prepare = sa.prepare_server_args  # type: ignore[attr-defined]",
                "            def _prep(raw_args):  # type: ignore",
                "                a = _orig_prepare(raw_args)",
                "                try:",
                "                    want_attn = os.environ.get('INFERENCE_BENCH_SGLANG_ATTENTION_BACKEND', '').strip() or 'triton'",
                "                    if getattr(a, 'attention_backend', None) is not None:",
                "                        a.attention_backend = want_attn",
                "                    # Sampling backend: some builds default sampling_backend=triton when attention_backend=triton,",
                "                    # but then crash at runtime with: ValueError: Invalid sampling backend: triton.",
                "                    # Default to pytorch sampling for robustness; FlashInfer can JIT-build at import/use time.",
                "                    want_samp = os.environ.get('INFERENCE_BENCH_SGLANG_SAMPLING_BACKEND', '').strip() or 'pytorch'",
                "                    if getattr(a, 'sampling_backend', None) is not None:",
                "                        a.sampling_backend = want_samp",
                "                    elif getattr(a, 'sampling_backend', None) == 'triton':",
                "                        a.sampling_backend = want_samp",
                "                    if getattr(a, 'tokenizer_mode', None) is None:",
                "                        a.tokenizer_mode = 'auto'",
                "                except Exception:",
                "                    pass",
                "                return a",
                "            sa.prepare_server_args = _prep  # type: ignore[attr-defined]",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Some SGLang versions also hard-require FlashInfer (even when using a Triton",
                "# attention backend) by asserting its version during startup. In our baseline",
                "# runs, FlashInfer is optional; skip this check if flashinfer isn't installed.",
                "try:",
                "    import importlib",
                "    def _ptb_patch_assert_pkg_version(mod):",
                "        try:",
                "            fn = getattr(mod, 'assert_pkg_version', None)",
                "            if fn is None:",
                "                return",
                "            def _wrapped(pkg, *args, **kwargs):  # type: ignore",
                "                if str(pkg) == 'flashinfer':",
                "                    return True",
                "                return fn(pkg, *args, **kwargs)",
                "            mod.assert_pkg_version = _wrapped  # type: ignore[attr-defined]",
                "        except Exception:",
                "            pass",
                "    try:",
                "        _ptb_patch_assert_pkg_version(importlib.import_module('sglang.srt.utils'))",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Some SGLang versions monkey-patch vLLM internals for P2P access checks by importing",
                "# `vllm.distributed.device_communicators.custom_all_reduce_utils`, which may not",
                "# exist in the vLLM version shipped in the container. This check is not required",
                "# for our single-GPU baseline runs; make the patch a no-op if it can't import.",
                "try:",
                "    import importlib",
                "    try:",
                "        u = importlib.import_module('sglang.srt.utils')",
                "        if hasattr(u, 'monkey_patch_vllm_p2p_access_check'):",
                "            _orig = u.monkey_patch_vllm_p2p_access_check  # type: ignore[attr-defined]",
                "            def _wrapped(*args, **kwargs):  # type: ignore",
                "                try:",
                "                    return _orig(*args, **kwargs)",
                "                except ModuleNotFoundError as e:",
                "                    if 'vllm.distributed.device_communicators.custom_all_reduce_utils' in str(e):",
                "                        return None",
                "                    raise",
                "            u.monkey_patch_vllm_p2p_access_check = _wrapped  # type: ignore[attr-defined]",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# vLLM's CustomOp expects subclasses to have a class attribute `.name` in newer",
                "# versions. Some SGLang-provided custom ops don't set it; default to class name.",
                "try:",
                "    import importlib",
                "    try:",
                "        co = importlib.import_module('vllm.model_executor.custom_op')",
                "        CustomOp = getattr(co, 'CustomOp', None)",
                "        if CustomOp is not None and hasattr(CustomOp, '__init__'):",
                "            _orig_init = CustomOp.__init__  # type: ignore[attr-defined]",
                "            def _init(self, *args, _orig_init=_orig_init, **kwargs):  # type: ignore",
                "                try:",
                "                    if not hasattr(self.__class__, 'name'):",
                "                        self.__class__.name = self.__class__.__name__  # type: ignore[attr-defined]",
                "                except Exception:",
                "                    pass",
                "                return _orig_init(self, *args, **kwargs)",
                "            CustomOp.__init__ = _init  # type: ignore[attr-defined]",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Some vLLM versions require graph_capture(device=...) but some SGLang versions",
                "# call graph_capture() without args during CUDA graph capture. Provide a wrapper",
                "# that defaults device to torch.cuda.current_device().",
                "try:",
                "    import importlib",
                "    import inspect",
                "    import torch",
                "    def _ptb_wrap_graph_capture(mod, name: str = 'graph_capture'):",
                "        fn = getattr(mod, name, None)",
                "        if fn is None or getattr(fn, '_ptb_wrapped', False):",
                "            return",
                "        try:",
                "            sig = inspect.signature(fn)",
                "            p = sig.parameters.get('device')",
                "            needs = (p is not None and p.default is inspect._empty)",
                "        except Exception:",
                "            needs = True",
                "        if not needs:",
                "            return",
                "        def _wrapped(*args, **kwargs):  # type: ignore",
                "            if 'device' not in kwargs and (len(args) == 0):",
                "                try:",
                "                    kwargs['device'] = torch.cuda.current_device()",
                "                except Exception:",
                "                    pass",
                "            return fn(*args, **kwargs)",
                "        _wrapped._ptb_wrapped = True  # type: ignore[attr-defined]",
                "        setattr(mod, name, _wrapped)",
                "    try:",
                "        _ptb_wrap_graph_capture(importlib.import_module('vllm.utils'))",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Some SGLang model wrappers call vLLM get_rope(...) with kwargs that drift",
                "# across vLLM versions (e.g. rotary_dim/base). Filter kwargs based on signature.",
                "try:",
                "    import importlib",
                "    try:",
                "        re = importlib.import_module('vllm.model_executor.layers.rotary_embedding')",
                "        fn = getattr(re, 'get_rope', None)",
                "        if fn is not None:",
                "            if not getattr(fn, '_ptb_sig_filtered', False):",
                "                import inspect",
                "                def _wrapped(*args, **kwargs):  # type: ignore",
                "                    try:",
                "                        allow = set(inspect.signature(fn).parameters.keys())",
                "                        kwargs = {k: v for k, v in kwargs.items() if k in allow}",
                "                    except Exception:",
                "                        kwargs.pop('rotary_dim', None)",
                "                        kwargs.pop('base', None)",
                "                    return fn(*args, **kwargs)",
                "                _wrapped._ptb_sig_filtered = True  # type: ignore[attr-defined]",
                "                re.get_rope = _wrapped  # type: ignore[attr-defined]",
                "    except Exception:",
                "        pass",
                "except Exception:",
                "    pass",
                "",
                "# Lustre/NFS often doesn't support flock(). Prefer SoftFileLock to avoid:",
                "#   NotImplementedError: FileSystem does not appear to support flock",
                "try:",
                "    import os",
                "    if os.environ.get('SOFT_FILELOCK', '').strip() in {'1','true','True'}:",
                "        import filelock  # type: ignore",
                "        if hasattr(filelock, 'SoftFileLock'):",
                "            filelock.FileLock = filelock.SoftFileLock  # type: ignore[attr-defined]",
                "        try:",
                "            import filelock._api as _api  # type: ignore",
                "            if hasattr(filelock, 'SoftFileLock'):",
                "                _api.FileLock = filelock.SoftFileLock  # type: ignore[attr-defined]",
                "        except Exception:",
                "            pass",
                "except Exception:",
                "    pass",
                "",
                "# If HF/vLLM lock files are stale on shared filesystems, startup can hang forever",
                "# waiting for a lock even though weights are already cached. For baseline runs",
                "# we pre-download weights; disable these locks to avoid deadlocks.",
                "try:",
                "    import os",
                "    if os.environ.get('INFERENCE_BENCH_DISABLE_HF_LOCKS', '1').strip() != '0':",
                "        import contextlib",
                "        try:",
                "            import vllm.model_executor.model_loader.weight_utils as _wu  # type: ignore",
                "            if hasattr(_wu, 'get_lock'):",
                "                def _ptb_get_lock(*_args, **_kwargs):",
                "                    return contextlib.nullcontext()",
                "                _wu.get_lock = _ptb_get_lock  # type: ignore[attr-defined]",
                "        except Exception:",
                "            pass",
                "except Exception:",
                "    pass",
                "",
                "# Some SGLang builds import vLLM internals by an older module path:",
                "#   from vllm.model_executor.model_loader.loader import DefaultModelLoader",
                "# Newer vLLM versions may have moved/renamed that module. Provide an alias if needed.",
                "try:",
                "    import importlib",
                "    import sys",
                "    import types",
                "",
                "    def _ptb_find_default_model_loader():",
                "        candidates = [",
                "            'vllm.model_executor.model_loader.loader',",
                "            'vllm.model_executor.model_loader',",
                "            'vllm.model_executor.model_loader.base',",
                "        ]",
                "        for name in candidates:",
                "            try:",
                "                m = importlib.import_module(name)",
                "            except Exception:",
                "                continue",
                "            if hasattr(m, 'DefaultModelLoader'):",
                "                return getattr(m, 'DefaultModelLoader')",
                "        return None",
                "",
                "    if 'vllm.model_executor.model_loader.loader' not in sys.modules:",
                "        _alias = types.ModuleType('vllm.model_executor.model_loader.loader')",
                "        _dml = _ptb_find_default_model_loader()",
                "        if _dml is not None:",
                "            _alias.DefaultModelLoader = _dml",
                "        else:",
                "            # Fallback stub: SGLang may import this even when LoRA is disabled/unused.",
                "            class DefaultModelLoader:  # type: ignore",
                "                def __init__(self, *args, **kwargs):",
                "                    raise RuntimeError(",
                "                        'DefaultModelLoader is unavailable in this vLLM build. '",
                "                        'This should be fine unless you are using SGLang LoRA features.'",
                "                    )",
                "            _alias.DefaultModelLoader = DefaultModelLoader",
                "        sys.modules['vllm.model_executor.model_loader.loader'] = _alias",
                "except Exception:",
                "    pass",
                "",
                "# Some SGLang builds import vLLM quantization configs from internal modules that",
                "# may not exist in the vLLM version installed in the container, e.g.:",
                "#   from vllm.model_executor.layers.quantization.aqlm import AQLMConfig",
                "# If we are not using quantization, it's safe to provide stub modules/classes",
                "# to satisfy these imports and avoid crashing during server startup.",
                "try:",
                "    import importlib.machinery",
                "    import importlib.util",
                "    import sys",
                "    from importlib.abc import Loader, MetaPathFinder",
                "",
                "    _PTB_VLLM_QUANT_PREFIX = 'vllm.model_executor.layers.quantization'",
                "",
                "    class _PTB_VllmQuantStubLoader(Loader):",
                "        def create_module(self, spec):  # type: ignore[override]",
                "            return None",
                "",
                "        def exec_module(self, module):  # type: ignore[override]",
                "            def __getattr__(name: str):",
                "                cls = type(name, (), {})",
                "                setattr(module, name, cls)",
                "                return cls",
                "",
                "            module.__getattr__ = __getattr__  # type: ignore[attr-defined]",
                "            module.__all__ = []",
                "",
                "    class _PTB_VllmQuantStubFinder(MetaPathFinder):",
                "        def find_spec(self, fullname, path, target=None):  # type: ignore[override]",
                "            if fullname == _PTB_VLLM_QUANT_PREFIX or fullname.startswith(_PTB_VLLM_QUANT_PREFIX + '.'):",
                "                real = importlib.machinery.PathFinder.find_spec(fullname, path)",
                "                if real is not None:",
                "                    return None",
                "                return importlib.util.spec_from_loader(fullname, _PTB_VllmQuantStubLoader())",
                "            return None",
                "",
                "    if not any(type(f).__name__ == '_PTB_VllmQuantStubFinder' for f in sys.meta_path):",
                "        sys.meta_path.insert(0, _PTB_VllmQuantStubFinder())",
                "except Exception:",
                "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{shim_dir}{os.pathsep}{existing}" if existing else str(shim_dir)


def _start_server(cmd: List[str], log_path: Path, env: Dict[str, str]) -> subprocess.Popen:
    runtime_cache_dir = Path(env.get("TRITON_CACHE_DIR", "")).expanduser().parent
    runtime_cache_dir.mkdir(parents=True, exist_ok=True)
    (runtime_cache_dir / "triton").mkdir(parents=True, exist_ok=True)
    (runtime_cache_dir / "torchinductor").mkdir(parents=True, exist_ok=True)
    (runtime_cache_dir / "xdg_cache").mkdir(parents=True, exist_ok=True)
    (runtime_cache_dir / "outlines_cache").mkdir(parents=True, exist_ok=True)
    (runtime_cache_dir / "hf_home").mkdir(parents=True, exist_ok=True)
    (runtime_cache_dir / "cuda_cache").mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        # Start a new process group so we can signal the whole server tree (scheduler workers, etc.)
        # for debugging stack dumps without affecting the parent process.
        return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute baseline artifacts: quality samples + speed baselines + quality baselines.")
    parser.add_argument("--base-model", type=str, default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    parser.add_argument("--max-model-len", type=str, default=os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "32768"))
    parser.add_argument("--dataset-seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248")))
    parser.add_argument("--quality-seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_SEED", os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248"))))
    parser.add_argument("--mmlupro-n", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_N", "500")))
    parser.add_argument("--quality-concurrency", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_CONCURRENCY", "4")))
    parser.add_argument("--scenarios", type=str, nargs="*", default=[
        "inference_scenario_a_input_heavy",
        "inference_scenario_b_output_heavy",
        "inference_scenario_c_high_load",
        "inference_scenario_d_general",
    ])
    parser.add_argument("--backends", type=str, nargs="*", default=["vllm"])
    parser.add_argument("--backend-cmd", type=str, nargs="*", default=[], help="For custom backends: id=cmd. cmd supports {model},{host},{port},{max_model_len}.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="If set, reuse this port per backend (otherwise choose a free port).")
    parser.add_argument("--server-start-timeout-s", type=float, default=600.0)
    parser.add_argument("--cache-quality-samples", action="store_true", help="Generate MMLU-Pro quality sample caches.")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--quality-registry-suffix", type=str, default="", help="Optional suffix for quality registry filenames.")
    parser.add_argument(
        "--runtime-cache-dir",
        type=str,
        default=os.environ.get("INFERENCE_BENCH_RUNTIME_CACHE_DIR", ""),
        help="Where to write Triton/Inductor/HF caches for backend servers (defaults to $TMPDIR or /tmp).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    os.chdir(repo_root)

    model_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", (args.base_model or "unknown_model")).strip("_")
    backends = _parse_backends(args.backends, args.backend_cmd)

    runtime_cache_dir = Path(args.runtime_cache_dir).expanduser() if args.runtime_cache_dir else _default_runtime_cache_dir()
    server_env = _build_runtime_cache_env(runtime_cache_dir)
    _install_python_shims(runtime_cache_dir, server_env)

    if args.cache_quality_samples:
        _run([
            sys.executable,
            "-m",
            "src.eval.inference.cache_samples",
            "--seed",
            str(args.quality_seed),
            "--mmlupro-n",
            str(args.mmlupro_n),
        ], env=server_env)

    for backend in backends:
        port = int(args.port) if args.port else _free_port()
        server_url = f"http://{args.host}:{port}"

        cmd = _build_backend_cmd(backend, args.base_model, args.host, port, args.max_model_len)

        # Heuristics: skip backends that clearly aren't installed unless custom.
        if backend.kind == "vllm":
            try:
                __import__("vllm")
            except Exception:
                print(f"[skip] backend={backend.name} missing python module vllm")
                continue
        if backend.kind == "sglang":
            try:
                __import__("sglang")
            except Exception:
                print(f"[skip] backend={backend.name} missing python module sglang")
                continue
            try:
                __import__("orjson")
            except Exception:
                print(f"[skip] backend={backend.name} missing python module orjson (try: pip install orjson)")
                continue
            if importlib.util.find_spec("sgl_kernel") is None:
                print(f"[skip] backend={backend.name} missing python module sgl_kernel (sglang GPU kernels)")
                continue

        log_dir = repo_root / "src" / "eval" / "inference" / "baselines" / "_server_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{backend.name}_{port}.log"
        backend_server_env = dict(server_env)
        backend_server_env["INFERENCE_BENCH_ACTIVE_BACKEND"] = backend.kind
        if backend.kind == "vllm":
            backend_server_env.setdefault("VLLM_NO_USAGE_STATS", "1")

        print(f"[server] starting backend={backend.name} url={server_url} log={log_path} cache_dir={runtime_cache_dir}")
        proc = _start_server(cmd, log_path=log_path, env=backend_server_env)

        try:
            try:
                _wait_ready(
                    server_url,
                    timeout_s=args.server_start_timeout_s,
                    proc=proc,
                    log_path=log_path,
                    base_model=args.base_model,
                    allow_sigusr1_dump=(backend.kind == "sglang"),
                )
            except RuntimeError as exc:
                raise exc

            print(f"[server] ready backend={backend.name} url={server_url}")

            if not args.skip_speed:
                speed_out_root = repo_root / "src" / "eval" / "inference" / "baselines" / "speed" / backend.name
                speed_registry = repo_root / "src" / "eval" / "inference" / "baselines" / "speed" / backend.name / f"{model_safe}.json"
                for scenario in args.scenarios:
                    req = _canonical_speed_requests(scenario)
                    # Torch backend is much slower than vllm (no batching, no
                    # GPU-optimised attention) — use a longer per-request timeout
                    # so long-context scenarios don't fail.
                    request_timeout = "900" if backend.kind == "torch" else "300"
                    # Torch processes requests sequentially (no batching) — force
                    # concurrency=1 so requests don't queue-timeout.
                    concurrency_override = "1" if backend.kind == "torch" else None
                    precompute_cmd = [
                        sys.executable,
                        "-m",
                        "src.eval.inference.precompute_baseline",
                        "--scenario-id",
                        scenario,
                        "--base-model",
                        args.base_model,
                        "--server-url",
                        server_url,
                        "--out-root",
                        str(speed_out_root),
                        "--registry",
                        str(speed_registry),
                        "--seed",
                        str(args.dataset_seed),
                        "--request-timeout-s",
                        request_timeout,
                    ]
                    if req is not None:
                        precompute_cmd += ["--requests-file", str(req)]
                    if concurrency_override is not None:
                        precompute_cmd += ["--concurrency-override", concurrency_override]
                    _run(precompute_cmd, env=server_env)

            if not args.skip_quality:
                q_registry_name = f"{model_safe}{('_' + args.quality_registry_suffix) if args.quality_registry_suffix else ''}{'' if backend.name == 'vllm' else '_' + backend.name}.json"
                q_registry = repo_root / "src" / "eval" / "inference" / "baselines" / "quality" / q_registry_name
                q_out_root = repo_root / "src" / "eval" / "inference" / "baselines" / "quality" / backend.name / model_safe
                _run([
                    sys.executable,
                    "-m",
                    "src.eval.inference.precompute_quality_baseline",
                    "--server-url",
                    server_url,
                    "--base-model",
                    args.base_model,
                    "--backend",
                    backend.name,
                    "--registry",
                    str(q_registry),
                    "--out-root",
                    str(q_out_root),
                    "--seed",
                    str(args.quality_seed),
                    "--mmlupro-n",
                    str(args.mmlupro_n),
                    "--concurrency",
                    "1" if backend.kind == "torch" else str(args.quality_concurrency),
                ], env=server_env)

        finally:
            print(f"[server] stopping backend={backend.name} pid={proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)


if __name__ == "__main__":
    main()
