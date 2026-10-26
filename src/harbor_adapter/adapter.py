import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).resolve().parent / "template"
EVAL = ROOT / "src" / "eval"
INFERENCE = EVAL / "inference"
RUN_TASK = (ROOT / "src" / "run_task.sh").read_text(encoding="utf-8")
ENV_VARS = (ROOT / "src" / "commit_utils" / "set_env_vars.sh").read_text(encoding="utf-8")

sys.path.insert(0, str(EVAL))
from inference import quality_gate

SCENARIOS = {p.name.split("_")[2]: p.name for p in sorted((EVAL / "tasks").glob("inference_scenario_*"))}
MODELS = {"mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3"}
AGENTS = {"claude-code": "claude", "codex": "codex", "gemini-cli": "gemini", "opencode": "opencode"}
DEFAULT_SEED_PAIRS = re.search(r'^DEFAULT_SEED_PAIRS="(.*)"$', ENV_VARS, re.M)[1]
_RAW = dict(re.findall(r'^set_default (\w+) "(.*)"$', ENV_VARS, re.M))
SETTINGS = {k: os.environ.get(k) or re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m[1]) or _RAW.get(m[1], ""), v)
            for k, v in _RAW.items()}
SHARED_ENV = ["INFERENCE_BENCH_MAX_MODEL_LEN", "INFERENCE_BENCH_INPUT_TOKEN_MARGIN", "INFERENCE_BENCH_QUALITY_TAU",
              "INFERENCE_BENCH_QUALITY_SEED", "INFERENCE_BENCH_QUALITY_MMLUPRO_N", "INFERENCE_BENCH_QUALITY_CONCURRENCY",
              "INFERENCE_BENCH_QUALITY_BASELINE_BACKEND"]
EVAL_FILES = ["__init__.py", "runner.py", "quality_gate.py", "cache_samples.py", "bin/launch_supervised_server.sh",
              "bin/server_ready.py"]
REWARD_FILES = ["hpo_search_baselines.py", "precompute_all_baselines.py"]
JUDGE_FILES = ["judge.sh", "get_judge_prompt.py", "prompt.txt"]


def heredoc(start: str, end: str = "PY") -> str:
    lines = RUN_TASK.splitlines()
    begin = next((i for i, line in enumerate(lines) if start in line), None)
    if begin is None:
        raise RuntimeError(f"src/run_task.sh: no heredoc after {start!r}")
    return "\n".join(lines[begin + 1:lines.index(end, begin + 1)]) + "\n"


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def quality_registry(model: str, backend: str) -> Path:
    name = f"{safe(model)}.json" if safe(backend) == "vllm" else f"{safe(model)}_{safe(backend)}.json"
    registry = INFERENCE / "baselines" / "quality" / name
    fallback = INFERENCE / "baselines" / "results" / safe(model) / f"quality_{safe(backend)}.json"
    return fallback if not registry.is_file() and fallback.is_file() else registry


def baselines(model: str, scenario: str, agent_seed: str, eval_seed: str) -> dict[str, Path]:
    for key in ("INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY", "INFERENCE_BENCH_QUALITY_MMLUPRO_SAMPLES_FILE"):
        if SETTINGS.get(key):
            raise ValueError(f"{key} is not supported by the Harbor adapter; unset it")
    quality_seed, quality_n = int(SETTINGS["INFERENCE_BENCH_QUALITY_SEED"]), int(SETTINGS["INFERENCE_BENCH_QUALITY_MMLUPRO_N"])
    num_requests = json.loads((EVAL / "tasks" / scenario / "scenario.json").read_text(encoding="utf-8")).get("num_requests", 32)
    pool = max(int(num_requests), int(SETTINGS["INFERENCE_BENCH_LONGBENCH_N"]))
    files = {
        "registry": quality_registry(model, SETTINGS["INFERENCE_BENCH_QUALITY_BASELINE_BACKEND"]),
        "mmlu_pro": quality_gate._default_samples_file("mmlu_pro", quality_seed, quality_n),
        "agent_longbench": quality_gate._default_samples_file("longbench_v2", int(agent_seed), pool),
        "eval_longbench": quality_gate._default_samples_file("longbench_v2", int(eval_seed), pool),
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if files["registry"].is_file() and quality_gate.load_quality_baseline_accuracy(
            files["registry"], "mmlu_pro", quality_seed, quality_n)[0] is None:
        missing.append(f"{files['registry']} (no mmlu_pro entry for seed={quality_seed} n={quality_n})")
    speed = INFERENCE / "baselines" / "speed" / "vllm" / f"{safe(model)}.json"
    if not speed.is_file() or scenario not in json.loads(speed.read_text(encoding="utf-8")).get("scenarios", {}):
        missing.append(f"{speed} (no entry for {scenario})")
    if missing:
        raise FileNotFoundError(
            f"Missing baselines for {model} / {scenario} / seeds {agent_seed}:{eval_seed}:\n  " + "\n  ".join(missing)
            + "\nCache the benchmark samples and precompute the quality reference as described in README.md.")
    return files


def bundle(dst: Path, files: dict[str, Path], longbench: str, extra: list[str] = ()) -> None:
    for name in EVAL_FILES + list(extra):
        place(INFERENCE / name, dst / name)
    for key in ("registry", "mmlu_pro", longbench):
        place(files[key], dst / files[key].relative_to(INFERENCE))


def toml_table(name: str, values: dict[str, str]) -> str:
    return f"\n[{name}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items())


class InferenceBenchAdapter:
    def __init__(self, output_dir: Path, num_hours: str = "2", agent: str = "claude-code"):
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent {agent!r}; choose from {', '.join(AGENTS)}")
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.agent = agent

    def generate_task(self, scenario_key: str, model_key: str, seed_pair: str, files: dict[str, Path]) -> Path:
        scenario, model = SCENARIOS[scenario_key], MODELS[model_key]
        agent_seed, eval_seed = seed_pair.split(":")
        task_dir = self.output_dir / f"inferencebench-{scenario_key}-{model_key}-{agent_seed}-{eval_seed}"
        if task_dir.exists():
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)
        self.write_task_toml(task_dir, scenario, model, agent_seed, eval_seed)
        self.write_instruction(task_dir, scenario, model)
        self.write_environment(task_dir / "environment", scenario, files)
        self.write_tests(task_dir / "tests", scenario, files)
        return task_dir

    def write_task_toml(self, task_dir: Path, scenario: str, model: str, agent_seed: str, eval_seed: str) -> None:
        content = (TEMPLATE / "task.toml").read_text(encoding="utf-8")
        old = "timeout_sec = 7200.0"
        if content.count(old) != 1:
            raise RuntimeError(f"template/task.toml: expected exactly one {old!r}")
        content = content.replace(old, f"timeout_sec = {float(self.num_hours) * 3600}")
        shared = {"INFERENCE_BENCH_BASE_MODEL": model, "INFERENCE_BENCH_SCENARIO": scenario,
                  **{k: SETTINGS[k] for k in SHARED_ENV}}
        content += toml_table("metadata.inferencebench", {
            "scenario": scenario, "base_model": model, "agent": AGENTS[self.agent], "num_hours": self.num_hours,
            "agent_seed": agent_seed, "eval_seed": eval_seed})
        content += toml_table("environment.env", {
            "HF_TOKEN": "${HF_TOKEN}", **shared, "INFERENCE_BENCH_DATASET_SEED": agent_seed, "NUM_HOURS": self.num_hours})
        content += toml_table("verifier.env", {
            "HF_TOKEN": "${HF_TOKEN}", "CLAUDE_CODE_OAUTH_TOKEN": "${CLAUDE_CODE_OAUTH_TOKEN:-}",
            "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
            **shared, "INFERENCE_BENCH_DATASET_SEED": eval_seed})
        (task_dir / "task.toml").write_text(content, encoding="utf-8")

    def write_instruction(self, task_dir: Path, scenario: str, model: str) -> None:
        env = {**os.environ, "INFERENCE_BENCH_METRICS_PATH": "/home/agent/task/metrics_preview.json",
               "INFERENCE_BENCH_SERVER_URL": "http://127.0.0.1:8000"}
        prompt = subprocess.run(
            [sys.executable, str(EVAL / "general" / "get_prompt.py"), "--base-model", model, "--scenario-id", scenario,
             "--num-hours", self.num_hours, "--agent", AGENTS[self.agent]],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
        (task_dir / "instruction.md").write_text(prompt, encoding="utf-8")

    def write_environment(self, env_dir: Path, scenario: str, files: dict[str, Path]) -> None:
        self.write_dockerfile(env_dir)
        place(TEMPLATE / "environment" / "collect.sh", env_dir / "collect.sh")
        (env_dir / "capture_runtime_env.py").write_text(heredoc("<<'PY' || return 1"), encoding="utf-8")
        bundle(env_dir / "inference_eval", files, "agent_longbench")
        task = env_dir / "home" / "task"
        for name in ("mission.txt", "benchmark.txt", "scenario.json"):
            place(EVAL / "tasks" / scenario / name, task / name)
        place(EVAL / "tasks" / "_shared" / "task_context", task)
        (task / "evaluate.py").write_text(heredoc('task/evaluate.py" <<'), encoding="utf-8")
        site = task / ".local" / "lib" / "python3.10" / "site-packages"
        (site / "pyairports").mkdir(parents=True)
        (site / "pyairports" / "__init__.py").write_text("", encoding="utf-8")
        (site / "pyairports" / "airports.py").write_text(heredoc('pyairports/airports.py" <<'), encoding="utf-8")
        (site / "sitecustomize.py").write_text(heredoc('sitecustomize.py" <<'), encoding="utf-8")
        timer = TEMPLATE / "timer.sh"
        (task / "timer.sh").write_text(
            timer.read_text(encoding="utf-8").replace("NUM_SECONDS=", f"NUM_SECONDS={round(float(self.num_hours) * 3600)}"),
            encoding="utf-8")

    def write_tests(self, tests_dir: Path, scenario: str, files: dict[str, Path]) -> None:
        self.write_dockerfile(tests_dir)
        for name in ("test.sh", "snapshot.py"):
            place(TEMPLATE / "tests" / name, tests_dir / name)
        (tests_dir / "valid_eval_metrics.py").write_text(heredoc("<<'PY_METRICS'", "PY_METRICS"), encoding="utf-8")
        ib = tests_dir / "ib"
        bundle(ib / "src" / "eval" / "inference", files, "eval_longbench", REWARD_FILES)
        for name in ("evaluate.py", "scenario.json", "mission.txt", "benchmark.txt"):
            place(EVAL / "tasks" / scenario / name, ib / "src" / "eval" / "tasks" / scenario / name)
        for name in JUDGE_FILES:
            place(ROOT / "src" / "disallowed_usage_judge" / name, ib / "src" / "disallowed_usage_judge" / name)

    @staticmethod
    def write_dockerfile(context: Path) -> None:
        context.mkdir(parents=True, exist_ok=True)
        tail = (TEMPLATE / context.name / "Dockerfile").read_text(encoding="utf-8")
        (context / "Dockerfile").write_text((TEMPLATE / "Dockerfile").read_text(encoding="utf-8") + tail, encoding="utf-8")
        place(ROOT / "containers" / "locks" / "inference.txt", context / "constraints.txt")


def opencode_config() -> str:
    text = (ROOT / "agents" / "opencode" / "solve.sh").read_text(encoding="utf-8")
    match = re.search(r"^cat > opencode\.json <<'JSON'\n(.*?)\n^JSON$", text, re.M | re.S)
    if match is None:
        raise RuntimeError("agents/opencode/solve.sh: no opencode.json heredoc")
    return json.dumps(json.loads(match[1]))
