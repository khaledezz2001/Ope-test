FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

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
# CUDA / BLACKWELL OPTIMIZATIONS
# RTX 5090 = sm_120 (Blackwell)
# -------------------------------
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

# -------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install uv (needed to use --torch-backend flag for cu128 wheel selection)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# -------------------------------
# INSTALL vLLM cu128 FIRST
# The official cu128 wheel includes sm_120 kernels for Blackwell (RTX 5090).
# Let vLLM own torch — do NOT pre-install torch from the runpod base.
# -------------------------------
RUN uv pip install --system \
    "vllm" \
    --torch-backend cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# -------------------------------
# APP DEPENDENCIES
# transformers pinned to 4.48.3 — last version where Qwen2Tokenizer
# still has all_special_tokens_extended (required by vLLM's tokenizer code).
# olmocr without [gpu] extra so it cannot override vLLM's torch/transformers.
# -------------------------------
RUN uv pip install --system \
    runpod \
    pillow \
    "opencv-python-headless" \
    huggingface_hub \
    protobuf \
    "numpy<2.0" \
    pdf2image \
    "compressed-tensors" \
    qwen-vl-utils \
    "transformers==4.48.3" \
    olmocr

# Hard-lock transformers in case olmocr pulled a newer version
RUN uv pip install --system "transformers==4.48.3" --reinstall

# Flash Attention optional — Blackwell support is still maturing
RUN uv pip install --system flash-attn --no-build-isolation || \
    echo "Flash Attention build failed, continuing without it"

# -------------------------------
# MODEL DOWNLOAD (BUILD TIME)
# -------------------------------
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="allenai/olmOCR-2-7B-1025-FP8",
    local_dir="/models/hf/allenai/olmOCR-2-7B-1025-FP8",
    local_dir_use_symlinks=False
)
print("olmOCR downloaded")
EOF

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
