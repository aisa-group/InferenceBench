import argparse
import os
import re
import stat
from pathlib import Path

_EVIDENCE_FILES = ["start_server.sh", "server.log"]
_MAX_LOG_LINES = 200
_MAX_EVIDENCE_BYTES = 64 * 1024


def _read_evidence(task_dir: Path) -> str:
    sections = []
    directory_fd = None
    try:
        task_dir = Path.cwd() / task_dir
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(task_dir.anchor, flags)
        for part in task_dir.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        for filename in _EVIDENCE_FILES:
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory_fd)
                with os.fdopen(fd, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    offset = max(0, info.st_size - _MAX_EVIDENCE_BYTES)
                    stream.seek(offset)
                    content = stream.read(_MAX_EVIDENCE_BYTES).decode("utf-8", errors="replace")
                lines = content.splitlines()
                prefix = f"[...truncated to last {_MAX_EVIDENCE_BYTES} bytes...]\n" if offset else ""
                if len(lines) > _MAX_LOG_LINES:
                    prefix += f"[...truncated {len(lines) - _MAX_LOG_LINES} lines...]\n"
                    content = "\n".join(lines[-_MAX_LOG_LINES:])
                fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", content)), default=0) + 1)
                sections.append(f"\n## Contents of `{filename}`\n{fence}\n{prefix}{content}\n{fence}")
            except OSError:
                continue
    except OSError:
        return ""
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    if not sections:
        return ""
    return (
        "\n\n# Pre-loaded File Evidence\n"
        "These files are from the task directory. "
        "Use them as primary evidence for your judgement.\n"
        + "".join(sections)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task-dir", type=str, default="/home/agent/task")
    args = parser.parse_args()

    prompt = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    prompt = prompt.replace("{model}", args.model).replace("{benchmark}", args.benchmark)
    print(prompt + _read_evidence(Path(args.task_dir)))


if __name__ == "__main__":
    main()
