# ================================================================
# Base: PyTorch 2.8 + CUDA 12.8.1 — matches Blackwell driver reqs
# We upgrade torch to 2.9 nightly inside the build for sm_120
# ================================================================
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---------------------------------------------------------------
# HF cache
# ---------------------------------------------------------------
ENV HF_HOME=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------
# Blackwell / CUDA build flags
# ---------------------------------------------------------------
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

# Critical for vLLM on Blackwell:
# FA3 is not supported on sm_120 — force FA2 backend
ENV VLLM_FLASH_ATTN_VERSION=2
# Limit parallel compile jobs to avoid OOM during Docker build
ENV MAX_JOBS=4

# ---------------------------------------------------------------
# System deps
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    build-essential \
    ninja-build \
    cmake \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# Upgrade to PyTorch 2.9 nightly + cu128
# This is the ONLY torch version with stable sm_120 support.
# Uninstall the base image's torch 2.8 first.
# ---------------------------------------------------------------
RUN pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

RUN pip install --no-cache-dir \
    --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify torch sees the GPU arch
RUN python -c "import torch; print('torch:', torch.__version__)"

# ---------------------------------------------------------------
# Base Python deps (no vLLM yet)
# ---------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------
# Uninstall flash-attn — it triggers undefined symbol errors on
# Blackwell (sm_120). vLLM will use FlashInfer instead.
# ---------------------------------------------------------------
RUN pip uninstall -y flash-attn 2>/dev/null || true

# ---------------------------------------------------------------
# Build vLLM from source — only way to get sm_120 support.
# Pre-built wheels from PyPI do NOT support Blackwell.
# Build time: ~25-40 min depending on runner CPU count.
# ---------------------------------------------------------------
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git /build/vllm

WORKDIR /build/vllm

# use_existing_torch.py removes vLLM's pinned torch requirement
# so it uses our already-installed torch 2.9 nightly
RUN python use_existing_torch.py

RUN pip install --no-cache-dir -r requirements/build.txt

RUN MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir \
    --no-build-isolation \
    -e .

WORKDIR /app

# Verify vLLM import works
RUN python -c "from vllm import LLM, SamplingParams; print('vLLM import OK')"

# ---------------------------------------------------------------
# Model download at build time
# ---------------------------------------------------------------
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="allenai/olmOCR-2-7B-1025-FP8",
    local_dir="/models/hf/allenai/olmOCR-2-7B-1025-FP8",
    local_dir_use_symlinks=False,
)
print("olmOCR downloaded successfully")
PYEOF

# ---------------------------------------------------------------
# Lock offline mode at runtime
# ---------------------------------------------------------------
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
COPY handler.py .

CMD ["python", "-u", "handler.py"]
