#!/usr/bin/env python3
"""Download HuggingFace models and datasets listed in a resources JSON file.

Expected JSON format:
{
    "models": ["meta-llama/Meta-Llama-3.1-8B-Instruct", ...],
    "datasets": [
        {"dataset": "cais/mmlu", "configs": ["all"], "splits": ["test"]},
        ...
    ]
}

Usage:
    python download_resources.py --resources-file /path/to/resources.json [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download_model(model_id: str) -> str:
    """Download a model snapshot to the HF cache."""
    # SoftFileLock workaround for lustre
    try:
        import filelock
        from filelock import SoftFileLock
        filelock.FileLock = SoftFileLock
    except Exception:
        pass

    from huggingface_hub import snapshot_download

    print(f"[download] downloading model: {model_id}")
    # Try local_files_only first (fast, no network/locks)
    try:
        path = snapshot_download(model_id, local_files_only=True)
        print(f"[download] model {model_id} -> {path} (local cache hit)")
        return path
    except Exception:
        pass

    path = snapshot_download(
        model_id,
        resume_download=True,
        max_workers=1,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[download] model {model_id} -> {path}")
    return path


def download_dataset(dataset_id: str, configs: list[str], splits: list[str]) -> None:
    """Download dataset configs/splits to the HF cache."""
    from datasets import load_dataset

    for config in configs:
        for split in splits:
            print(f"[download] downloading dataset: {dataset_id} config={config} split={split}")
            try:
                load_dataset(dataset_id, config, split=split)
                print(f"[download] dataset {dataset_id}/{config}/{split} cached")
            except Exception as exc:
                print(f"[download] WARNING: failed to download {dataset_id}/{config}/{split}: {exc}",
                      file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF models and datasets from a resources file.")
    parser.add_argument("--resources-file", required=True, help="Path to JSON resources file.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel download workers.")
    args = parser.parse_args()

    resources_path = Path(args.resources_file)
    if not resources_path.is_file():
        print(f"[download] ERROR: resources file not found: {resources_path}", file=sys.stderr)
        sys.exit(1)

    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    models = resources.get("models", [])
    datasets = resources.get("datasets", [])

    print(f"[download] {len(models)} model(s), {len(datasets)} dataset(s) to download")

    # Download models (sequentially — these are large)
    for model_id in models:
        try:
            download_model(model_id)
        except Exception as exc:
            print(f"[download] ERROR downloading model {model_id}: {exc}", file=sys.stderr)
            sys.exit(1)

    # Download datasets (can parallelize)
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

    print("[download] all resources downloaded successfully")


if __name__ == "__main__":
    main()
