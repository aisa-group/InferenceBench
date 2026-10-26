#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download_model(model_id: str) -> None:
    import filelock

    filelock.FileLock = filelock.SoftFileLock
    from huggingface_hub import snapshot_download

    print(f"[download] model {model_id}")
    snapshot_download(model_id, max_workers=1, token=os.environ.get("HF_TOKEN"))


def download_dataset(dataset_id: str, configs: list[str], splits: list[str]) -> None:
    from datasets import load_dataset

    for config in configs:
        for split in splits:
            print(f"[download] dataset {dataset_id} config={config} split={split}")
            load_dataset(dataset_id, config, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF models and datasets from a resources file.")
    parser.add_argument("--resources-file", required=True, help="Path to JSON resources file.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel download workers.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    resources_path = Path(args.resources_file)
    if not resources_path.is_file():
        print(f"[download] ERROR: resources file not found: {resources_path}", file=sys.stderr)
        sys.exit(1)

    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    models = resources.get("models", [])
    datasets = resources.get("datasets", [])

    for model_id in models:
        try:
            download_model(model_id)
        except Exception as exc:
            print(f"[download] ERROR downloading model {model_id}: {exc}", file=sys.stderr)
            sys.exit(1)

    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for ds in datasets:
            ds_id = ds["dataset"]
            configs = ds.get("configs", ["default"])
            splits = ds.get("splits", ["train"])
            fut = pool.submit(download_dataset, ds_id, configs, splits)
            futures[fut] = ds_id

        for fut in as_completed(futures):
            ds_id = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[download] ERROR downloading dataset {ds_id}: {exc}", file=sys.stderr)
                failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
