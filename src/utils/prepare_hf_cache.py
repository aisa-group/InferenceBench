#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def _internal_links(root):
    links = []
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            target = path.resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError(f"Cache symlink escapes model directory: {path}")
            links.append((path.relative_to(root), target.relative_to(root)))
    return links


def _model_directory(source_home, model):
    repository = "models--" + model.replace("/", "--")
    source = source_home / "hub" / repository
    if source.is_symlink():
        raise ValueError("Cached model directory must not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_dir() or not source.is_relative_to(source_home) or source == source_home:
        raise ValueError("Cached model must be a directory inside HF_HOME")
    return repository, source


def prepare_model_cache(source_home, destination_home, model):
    source_home = Path(source_home).expanduser().resolve(strict=True)
    destination = Path(destination_home).expanduser()
    if destination.is_symlink():
        raise ValueError("Private cache must not be a symlink")
    destination = destination.resolve()
    if source_home.is_relative_to(destination) or destination.is_relative_to(source_home):
        raise ValueError("Source and private caches must not overlap")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Private cache must be empty before seeding")
    repository, source = _model_directory(source_home, model)
    links = _internal_links(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hf-cache-", dir=destination.parent))
    try:
        target = staging / "hub" / repository
        target.parent.mkdir()
        subprocess.run(["cp", "-a", "--reflink=auto", "--", str(source), str(target)], check=True)
        for name, linked_name in links:
            copied_link = target / name
            copied_link.unlink()
            copied_link.symlink_to(os.path.relpath(target / linked_name, copied_link.parent))
        _internal_links(target)
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Copy a cached model into a job-private Hugging Face cache.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    prepare_model_cache(args.source, args.destination, args.model)


if __name__ == "__main__":
    main()
