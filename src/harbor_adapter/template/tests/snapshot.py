import os
import shutil
import sys

PRUNE = {".git", "__pycache__", ".cache", "hf_cache", ".venv", "venv", "node_modules", ".nvm", ".npm",
         "site-packages", "wandb", ".ipynb_checkpoints"}
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".npy", ".npz", ".h5", ".msgpack", ".onnx", ".engine"}
MAX_FILE, MAX_TOTAL, MAX_FILES = 64 << 20, 1 << 30, 20000

source, target = sys.argv[1:3]
candidates, dropped = [], []
for root, dirs, files in os.walk(source):
    for name in [d for d in dirs if d in PRUNE]:
        dropped.append(f"pruned dir\t{os.path.relpath(os.path.join(root, name), source)}")
    dirs[:] = [d for d in dirs if d not in PRUNE]
    for name in files:
        path = os.path.join(root, name)
        rel = os.path.relpath(path, source)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if os.path.splitext(name)[1].lower() in WEIGHTS or size > MAX_FILE:
            dropped.append(f"{size}\t{rel}")
            continue
        candidates.append((size, rel))
total = taken = 0
for size, rel in sorted(candidates):
    if taken >= MAX_FILES or total + size > MAX_TOTAL:
        dropped.append(f"{size}\t{rel}")
        continue
    os.makedirs(os.path.dirname(os.path.join(target, rel)), exist_ok=True)
    shutil.copy2(os.path.join(source, rel), os.path.join(target, rel))
    total, taken = total + size, taken + 1
os.makedirs(target, exist_ok=True)
with open(os.path.join(target, ".snapshot_dropped.txt"), "w", encoding="utf-8") as stream:
    stream.write("".join(line + "\n" for line in dropped))
print(f"[snapshot] {taken} files, {total} bytes kept; {len(dropped)} entries dropped (see .snapshot_dropped.txt)")
