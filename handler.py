import os
import base64
import io
import time
import runpod
from PIL import Image
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE — set BEFORE any HF import
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force FA2 — FA3 not supported on Blackwell sm_120
os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"

# ===============================
# CONFIG
# ===============================
MODEL_PATH     = "/models/hf/allenai/olmOCR-2-7B-1025-FP8"
MAX_PAGES      = 100
MAX_NEW_TOKENS = 1536
MAX_NUM_SEQS   = 8      # max concurrent pages in one vLLM batch

llm = None   # vllm.LLM — loaded once at cold start

def log(msg):
    print(f"[HANDLER] {msg}", flush=True)

# ===============================
# HALLUCINATION DETECTION
# ===============================
def is_hallucinated_output(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True
    text_lower = text.lower()
    hallucination_indicators = [
        "table 1:", "comparison of different methods", "note: the choice of method",
        "this page is blank", "no text found", "empty page", "the image appears to be",
        "there is no visible text", "the document appears to be blank", "i cannot see any text",
        "method | accuracy | speed", "soil moisture", "time domain reflectometry",
    ]
    for indicator in hallucination_indicators:
        if indicator in text_lower:
            return True
    lines = text.strip().split("\n")
    if len(lines) > 20:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < 3:
            return True
    pipe_lines = sum(1 for line in lines if "|" in line)
    if len(lines) > 0 and pipe_lines / len(lines) > 0.5:
        if len(text.replace("|", "").replace("-", "").replace("\n", "").strip()) < 100:
            return True
    table_markers = text.count("|")
    if table_markers > 10:
        table_rows = [line for line in lines if "|" in line]
        if len(table_rows) > 3:
            pipe_counts = [line.count("|") for line in table_rows]
            if len(set(pipe_counts)) == 1 and pipe_counts[0] > 3:
                return True
    if sum(c.isalnum() for c in text) < 10:
        return True
    return False

# ===============================
# IMAGE HELPERS
# ===============================
def decode_image(b64: str) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    target_width = 1600
    scale = target_width / img.width
    return img.resize((target_width, int(img.height * scale)), Image.BICUBIC)

def decode_pdf(b64: str) -> list:
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes, dpi=150, fmt="png", thread_count=4, use_pdftocairo=True
    )
    return images[:MAX_PAGES]

# ===============================
# OCR PROMPT
# ===============================
OCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LaTeX and tables to HTML."
)

# ===============================
# LOAD vLLM (once at cold start)
# ===============================
def load_model():
    global llm
    if llm is not None:
        return

    from vllm import LLM

    log("Loading vLLM engine...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",           # bfloat16 — FP8 GEMM not stable on sm_120 yet
        max_model_len=4096,         # olmOCR context window
        max_num_seqs=MAX_NUM_SEQS,  # pages batched concurrently
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},  # one image per page
        # enforce_eager=False lets vLLM use CUDA graphs (faster after warmup)
        enforce_eager=False,
        # gpu_memory_utilization: how much VRAM to reserve for KV cache
        gpu_memory_utilization=0.88,
        # Disable flash-attn — use FlashInfer (Blackwell-compatible)
        disable_custom_all_reduce=False,
    )
    log("vLLM engine ready")

# ===============================
# BATCH OCR via vLLM offline API
# ===============================
def ocr_batch(images: list) -> list:
    from vllm import SamplingParams
    from transformers import AutoProcessor

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0,
        repetition_penalty=1.1,
    )

    # Load processor once to build the chat-formatted prompt string
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

    inputs = []
    for img in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": OCR_PROMPT},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        })

    # vLLM batches all pages in one call — continuous batching internally
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    return [out.outputs[0].text.strip() for out in outputs]

# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    load_model()
    try:
        inp = event.get("input", {})
        if "image" in inp:
            pages = [decode_image(inp["image"])]
        elif "file" in inp:
            pages = decode_pdf(inp["file"])
        else:
            return {"status": "error", "message": "Missing 'image' or 'file' in input"}

        total_pages = len(pages)
        log(f"Processing {total_pages} page(s)...")
        t0 = time.time()

        raw_results = ocr_batch(pages)

        extracted_pages = []
        for j, text in enumerate(raw_results):
            page_num = j + 1
            if text.upper().startswith("EMPTY_PAGE") or is_hallucinated_output(text):
                log(f"Page {page_num} flagged as empty/hallucinated")
                text = "[Empty or unreadable page]"
            extracted_pages.append({"page": page_num, "text": text})

        elapsed = time.time() - t0
        log(f"Done: {total_pages} pages in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages,
        }

    except Exception as e:
        import traceback
        log(f"Error: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

# ===============================
# COLD START — preload + warmup
# ===============================
log("Cold start — loading vLLM...")
load_model()

log("Running warmup pass...")
try:
    dummy = Image.new("RGB", (1600, 1200), color="white")
    _ = ocr_batch([dummy])
    log("Warmup complete!")
except Exception as e:
    log(f"Warmup error (non-fatal): {e}")

runpod.serverless.start({"handler": handler})
