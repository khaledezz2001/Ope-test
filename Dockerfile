# ================================================================
# Use the official vLLM image — already compiled with CUDA 12.8
# and includes Blackwell SM120 kernel fixes as of v0.8+
# No source build needed. Build time: ~3-5 min (just model download)
# ================================================================
FROM vllm/vllm-openai:latest

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
# Blackwell flags
# ---------------------------------------------------------------
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
# FA3 not supported on sm_120 — force FA2
ENV VLLM_FLASH_ATTN_VERSION=2
# Use FlashInfer attention backend (Blackwell-stable)
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

# ---------------------------------------------------------------
# System deps for PDF processing
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# App deps (vLLM + torch already in base image)
# ---------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
