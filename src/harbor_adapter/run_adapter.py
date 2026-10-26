import argparse
import re
import sys
from pathlib import Path

from adapter import AGENTS, DEFAULT_SEED_PAIRS, MODELS, SCENARIOS, InferenceBenchAdapter, baselines


def pick(spec: str, known: dict, what: str) -> list[str]:
    names = list(known) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in known]
    if unknown or not names:
        sys.exit(f"unknown {what}: {', '.join(unknown) or repr(spec)} (known: {', '.join(known)})")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harbor tasks for InferenceBench.")
    parser.add_argument("--scenario", default="all", help=f"comma list of {', '.join(SCENARIOS)} or all")
    parser.add_argument("--base-model", default="all", help=f"comma list of {', '.join(MODELS)} or all")
    parser.add_argument("--seed-pairs", default=DEFAULT_SEED_PAIRS, help="agent_seed:eval_seed,...")
    parser.add_argument("--num-hours", default="2")
    parser.add_argument("--agent", default="claude-code", choices=list(AGENTS))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "tasks")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for key, scenario in SCENARIOS.items():
            print(f"scenario {key}: {scenario}")
        for key, model in MODELS.items():
            print(f"model {key}: {model}")
        print(f"default seed pairs: {DEFAULT_SEED_PAIRS}")
        return

    if not re.fullmatch(r"[0-9]+([.][0-9]+)?", args.num_hours) or float(args.num_hours) <= 0:
        sys.exit("--num-hours must be a positive number")
    adapter = InferenceBenchAdapter(args.output, num_hours=args.num_hours, agent=args.agent)
    if not re.fullmatch(r"[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*", args.seed_pairs.replace(" ", "")):
        sys.exit("--seed-pairs must look like 21:1337,248:428")
    grid = [(s, m, p.strip()) for p in args.seed_pairs.split(",") if p.strip()
            for s in pick(args.scenario, SCENARIOS, "scenario") for m in pick(args.base_model, MODELS, "base model")]
    try:
        files = {task: baselines(MODELS[task[1]], SCENARIOS[task[0]], *task[2].split(":")) for task in grid}
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))
    for task in grid:
        print(adapter.generate_task(*task, files[task]))


if __name__ == "__main__":
    main()
