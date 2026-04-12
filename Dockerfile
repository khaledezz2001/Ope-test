FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------
# HF CACHE PATH
# -------------------------------
ENV HF_HOME=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV TOKENIZERS_PARALLELISM=false

# -------------------------------
# CUDA / SGLang OPTIMIZATIONS
# -------------------------------
ENV CUDA_VISIBLE_DEVICES=0
# Tell torch/sglang which arch to target (Blackwell = sm_120)
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

# SGLang uses its own CUDA graph warmup — this tells it to use all available VRAM
ENV SGL_DISABLE_DISK_CACHE=1

# -------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# PYTHON DEPENDENCIES
#
# SGLang replaces the transformers inference stack entirely.
# It provides:
#   - Continuous batching (no more manual BATCH_SIZE loops)
#   - RadixAttention prefix caching (shared OCR prompt cached across all pages)
#   - CUDA graphs for near-zero kernel launch overhead
#   - FlashInfer kernels (fastest attention on Blackwell)
# -------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# FlashInfer JIT cache — pre-compiled kernels for cu128, avoids runtime JIT compilation
RUN pip install --no-cache-dir flashinfer-jit-cache \
    --index-url https://flashinfer.ai/whl/cu128

# -------------------------------
# MODEL DOWNLOAD (BUILD TIME)
# -------------------------------
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PYEOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="allenai/olmOCR-2-7B-1025-FP8",
    local_dir="/models/hf/allenai/olmOCR-2-7B-1025-FP8",
    local_dir_use_symlinks=False
)
print("olmOCR downloaded successfully")
PYEOF

# -------------------------------
# LOCK OFFLINE MODE (RUNTIME)
# -------------------------------
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# -------------------------------
# APP
# -------------------------------
COPY handler.py .

CMD ["python", "-u", "handler.py"]
