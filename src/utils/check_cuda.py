#!/usr/bin/env python3
import os
import subprocess

import torch


def get_gpu_processes(gpu_id):
    if gpu_id is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=" + str(gpu_id),
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        processes = []
        for line in result.stdout.splitlines():
            if line.strip():
                pid, memory = line.split(',')
                if int(pid.strip()) != os.getpid():
                    processes.append((int(pid.strip()), float(memory.strip())))
        return processes
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return False
    count = torch.cuda.device_count()
    if count != 1:
        print(f"Expected one visible CUDA device, found {count}")
        return False
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {name} ({props.total_memory / 1e9:.1f} GB)")
    if not any(family in name for family in ("H100", "A100")):
        print("Expected an H100 or A100 GPU")
        return False
    gpu_id = getattr(props, "uuid", None)
    if gpu_id is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().split(',')
        if len(visible) == 1 and visible[0]:
            gpu_id = visible[0]
    processes = get_gpu_processes(gpu_id)
    if processes is None:
        print("Could not inspect the assigned GPU's processes")
    elif processes:
        print(f"Assigned GPU has {len(processes)} other process(es)")
        return False
    try:
        torch.randn(1, device='cuda')
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"CUDA allocation failed: {exc}")
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(0 if check_cuda() else 1)
