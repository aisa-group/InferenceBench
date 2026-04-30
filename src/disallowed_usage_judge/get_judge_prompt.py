import os
from pathlib import Path

import argparse

_EVIDENCE_FILES = ["start_server.sh", "server.log"]
_MAX_LOG_LINES = 200


def _read_evidence(task_dir: Path) -> str:
    """Read key task files so the judge has immediate evidence."""
    sections = []
    for filename in _EVIDENCE_FILES:
        filepath = task_dir / filename
        if not filepath.is_file():
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > _MAX_LOG_LINES:
                content = (
                    f"[...truncated {len(lines) - _MAX_LOG_LINES} lines...]\n"
                    + "\n".join(lines[-_MAX_LOG_LINES:])
                )
            sections.append(f"\n## Contents of `{filename}`\n```\n{content}\n```")
        except Exception:
            pass
    if not sections:
        return ""
    return (
        "\n\n# Pre-loaded File Evidence\n"
        "The following files were found in the task directory. "
        "Use them as primary evidence for your judgement.\n"
        + "".join(sections)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task-dir", type=str, default="/home/agent/task",
                        help="Directory containing start_server.sh and server.log evidence files.")
    args = parser.parse_args()

    base_prompt = os.environ.get('INFERENCE_BENCH_PROMPT', os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt'))
    if "mock" in base_prompt:
        print("Just do nothing and return.")
        return

    with open('src/disallowed_usage_judge/prompt.txt', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace("{model}", args.model)
    prompt = prompt.replace("{benchmark}", args.benchmark)

    prompt += _read_evidence(Path(args.task_dir))

    print(prompt)

if __name__ == "__main__":
    main()
