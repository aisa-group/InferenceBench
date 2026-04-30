#!/usr/bin/env python3
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
EVAL_DIR = TASK_DIR.parents[1]
sys.path.insert(0, str(EVAL_DIR))

from inference.runner import build_parser, run_evaluation


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(TASK_DIR, args)


if __name__ == "__main__":
    main()
