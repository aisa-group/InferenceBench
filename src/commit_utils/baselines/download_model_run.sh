#!/bin/bash
set -euo pipefail

MODEL="$1"
HOME_HF="/home/agent/hf_cache_tmp"
FAST_HF="/fast/${USER}/hf_cache"

mkdir -p "${HOME_HF}"

export HF_HOME="${HOME_HF}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[download] model=${MODEL}"
echo "[download] downloading to ${HOME_HF} (supports flock)"
echo "[download] will move to ${FAST_HF} when done"

python3 -c "
import sys
# Block hf_xet
class B:
    def find_module(s,n,p=None): return s if 'hf_xet' in n else None
    def load_module(s,n): raise ImportError(n)
sys.meta_path.insert(0, B())

from huggingface_hub import snapshot_download
p = snapshot_download('${MODEL}', max_workers=4)
print(f'[download] done: {p}', flush=True)
"

echo "[download] moving to ${FAST_HF}..."
rsync -a "${HOME_HF}/hub/" "${FAST_HF}/hub/"
echo "[download] cleaning up ${HOME_HF}..."
rm -rf "${HOME_HF}"
echo "[download] finished: ${MODEL}"
