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
# CUDA OPTIMIZATIONS
# -------------------------------
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDNN_V8_API_ENABLED=1

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
# -------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Flash Attention 2 for faster inference (optional — falls back gracefully)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Flash Attention 2 build failed, will use default attention"

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
