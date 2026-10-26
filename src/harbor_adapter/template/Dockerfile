FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN chmod 1777 /tmp && mkdir -p /opt/tmp /opt/uv_cache

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y \
        python3.10 \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        iproute2 \
        net-tools \
        lsof \
        procps \
        ripgrep \
        ninja-build \
        pkg-config \
        cmake \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs=22.22.2-1nodesource1

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && npm install -g \
        @anthropic-ai/claude-code@2.1.258 \
        @openai/codex@0.164.0 \
        @google/gemini-cli@0.39.1 \
        opencode-ai@1.14.28

COPY constraints.txt /opt/inferencebench/constraints.txt

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && uv pip install --system --no-cache --constraint /opt/inferencebench/constraints.txt \
        --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0+cu128

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && uv pip install --system --no-cache --constraint /opt/inferencebench/constraints.txt \
        aiohttp==3.13.5 \
        requests==2.33.1 \
        accelerate==1.13.0 \
        datasets==4.8.5 \
        sentencepiece==0.2.1 \
        numpy==2.2.6 \
        tokenizers==0.22.2 \
        transformers==5.6.2

RUN export TMPDIR=/opt/tmp UV_CACHE_DIR=/opt/uv_cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy DEBIAN_FRONTEND=noninteractive && git clone --no-checkout https://github.com/rank-and-file/filelock_workarounds.git /opt/filelock_workarounds && \
    git -C /opt/filelock_workarounds checkout --detach e6e8ae16b3d8827e6089eb950dd45cfb93fb58bd

ENV NO_PROXY="localhost,127.0.0.1" \
    no_proxy="localhost,127.0.0.1" \
    SOFT_FILELOCK="1" \
    PYTHONPATH="/opt/filelock_workarounds/soft_file_locks:$PYTHONPATH"

ENV PATH="/root/.local/bin:/home/agent/.local/bin:$PATH" \
    TMPDIR="/tmp" \
    TMP="/tmp" \
    TEMP="/tmp" \
    HF_HOME="/hf_cache" \
    HF_HUB_CACHE="/hf_cache/hub" \
    HUGGINGFACE_HUB_CACHE="/hf_cache/hub" \
    TRANSFORMERS_CACHE="/hf_cache/hub" \
    XDG_CACHE_HOME="/tmp/xdg_cache" \
    VLLM_TORCH_COMPILE_CACHE_DIR="/tmp/vllm_torch_compile_cache" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache" \
    TORCH_EXTENSIONS_DIR="/tmp/torch_extensions" \
    TRITON_CACHE_DIR="/tmp/triton_cache" \
    CUDA_CACHE_PATH="/tmp/cuda_cache" \
    PIP_USER="1" \
    PYTHONUSERBASE="/home/agent/.local" \
    PIP_CACHE_DIR="/tmp/pip_cache" \
    HOST="127.0.0.1" \
    PORT="8000" \
    INFERENCE_BENCH_SERVER_URL="http://127.0.0.1:8000" \
    INFERENCE_BENCH_SERVER_WAIT_S="900" \
    RAY_DISABLE_DASHBOARD="1" \
    RAY_USAGE_STATS_ENABLED="0" \
    RAY_DISABLE_METRICS_EXPORT="1"

RUN useradd --create-home --home-dir /home/agent --shell /bin/bash --uid 1000 agent && \
    mkdir -p /home/agent/hf_cache && ln -s /home/agent/hf_cache /hf_cache && chown -R agent:agent /home/agent
