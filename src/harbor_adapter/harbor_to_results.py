import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent.parent
CONFIG_KEYS = {
    "quality_seed": "INFERENCE_BENCH_QUALITY_SEED", "quality_mmlupro_n": "INFERENCE_BENCH_QUALITY_MMLUPRO_N",
    "quality_tau": "INFERENCE_BENCH_QUALITY_TAU", "quality_concurrency": "INFERENCE_BENCH_QUALITY_CONCURRENCY",
    "quality_baseline_backend": "INFERENCE_BENCH_QUALITY_BASELINE_BACKEND",
    "max_model_len": "INFERENCE_BENCH_MAX_MODEL_LEN", "input_token_margin": "INFERENCE_BENCH_INPUT_TOKEN_MARGIN",
}


def dotenv(name: str, default: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env_file = Path(os.environ.get("INFERENCE_BENCH_ENV_FILE") or REPO_ROOT / ".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key == name:
                return os.path.expandvars(value.strip().strip("\"'"))
    return default


def toml_value(text: str, key: str) -> str:
    match = re.search(rf'^{key} = "(.*)"$', text, re.M)
    return match[1] if match else ""


def find_trials(paths: list[Path]) -> list[Path]:
    trials = []
    for path in paths:
        for config in [path / "config.json", *sorted(path.rglob("config.json"))] if path.is_dir() else []:
            if config.is_file() and (config.parent / "result.json").is_file() and "trial_name" in json.loads(config.read_text()):
                trials.append(config.parent.resolve())
    return list(dict.fromkeys(trials))


def export_trial(trial: Path, results_dir: Path, experiment_name: str, overwrite: bool, dry_run: bool) -> None:
    result = json.loads((trial / "result.json").read_text())
    config = json.loads((trial / "config.json").read_text())
    task_dir = Path(config["task"]["path"])
    if not task_dir.is_absolute():
        task_dir = next((c for c in (ADAPTER_DIR / task_dir, Path.cwd() / task_dir) if c.is_dir()), task_dir)
    toml = (task_dir / "task.toml").read_text(encoding="utf-8")
    meta = {k: toml_value(toml, k) for k in ("scenario", "base_model", "agent", "num_hours", "agent_seed", "eval_seed")}
    agent_config = (config["agent"].get("model_name") or "unknown").split("/", 1)[-1]
    run_id = int(datetime.fromisoformat(result["started_at"].replace("Z", "+00:00")).timestamp())
    method = f"{meta['agent']}_{re.sub(r'[^A-Za-z0-9._-]', '_', agent_config)}_{meta['num_hours']}h{experiment_name}"
    trial_id = re.sub(r"[^A-Za-z0-9_.-]", "_", trial.name.rsplit("__", 1)[-1])
    out = results_dir / method / f"{meta['scenario']}_{re.sub(r'[/:]', '_', meta['base_model'])}_{run_id}-{trial_id}"
    rewards = (result.get("verifier_result") or {}).get("rewards")
    print(f"{trial.name}\n  -> {out}\n     rewards={rewards}")
    if dry_run:
        return
    if out.exists():
        if not overwrite:
            print("     exists, skipping (use --overwrite)")
            return
        shutil.rmtree(out)
    if (trial / "verifier").is_dir():
        shutil.copytree(trial / "verifier", out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test-stdout.txt").is_file():
        (out / "test-stdout.txt").rename(out / "output.log")
    transcripts = sorted((trial / "agent").glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
    if transcripts:
        shutil.copy(transcripts[0], out / "solve_out_run1.txt")
    shutil.copy(task_dir / "instruction.md", out / "prompt.txt")
    execution = result.get("agent_execution") or {}
    if execution.get("started_at") and execution.get("finished_at"):
        seconds = int((datetime.fromisoformat(execution["finished_at"].replace("Z", "+00:00"))
                       - datetime.fromisoformat(execution["started_at"].replace("Z", "+00:00"))).total_seconds())
        (out / "time_taken.txt").write_text(f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}\n")
    run_config = {"scenario": meta["scenario"], "base_model": meta["base_model"], "agent": meta["agent"],
                  "agent_config": agent_config, "starting_point": "default", "num_hours": meta["num_hours"],
                  "agent_seed": meta["agent_seed"], "eval_seed": meta["eval_seed"],
                  **{k: toml_value(toml, v) for k, v in CONFIG_KEYS.items()}}
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    (out / "harbor").mkdir(exist_ok=True)
    for name in ("result.json", "config.json"):
        shutil.copy(trial / name, out / "harbor" / name)
    collect_log = trial / "artifacts" / "logs" / "artifacts" / "collect.log"
    if collect_log.is_file():
        shutil.copy(collect_log, out / "harbor" / "collect.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Harbor trials into the condor results layout.")
    parser.add_argument("paths", nargs="+", type=Path, help="harbor job, sweep or trial dirs")
    parser.add_argument("--output", type=Path, default=None, help="default: INFERENCE_BENCH_RESULTS_DIR")
    parser.add_argument("--experiment-name", default="", help="suffix for the method dir, like INFERENCE_BENCH_EXPERIMENT_NAME")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results_dir = args.output or Path(dotenv("INFERENCE_BENCH_RESULTS_DIR", "results"))
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    trials = find_trials(args.paths)
    if not trials:
        sys.exit("no harbor trials found under the given paths")
    print(f"results dir: {results_dir}\n")
    for trial in trials:
        export_trial(trial, results_dir, args.experiment_name, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()
