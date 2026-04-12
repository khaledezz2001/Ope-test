import os
import base64
import io
import time
import runpod
from PIL import Image
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/allenai/olmOCR-2-7B-1025-FP8"
MAX_PAGES = 100
MAX_NEW_TOKENS = 1536
BATCH_SIZE = 8          # concurrent requests to SGLang runtime

engine = None           # sglang.Engine instance

def log(msg):
    print(f"[BOOT] {msg}", flush=True)

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
        "method | accuracy | speed", "soil moisture", "time domain reflectometry"
    ]
    for indicator in hallucination_indicators:
        if indicator in text_lower:
            return True

    lines = text.strip().split('\n')
    if len(lines) > 20:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < 3:
            return True

    pipe_lines = sum(1 for line in lines if '|' in line)
    if len(lines) > 0 and pipe_lines / len(lines) > 0.5:
        content_without_pipes = text.replace('|', '').replace('-', '').replace('\n', '').strip()
        if len(content_without_pipes) < 100:
            return True

    table_markers = text.count('|')
    if table_markers > 10:
        table_rows = [line for line in lines if '|' in line]
        if len(table_rows) > 3:
            pipe_counts = [line.count('|') for line in table_rows]
            if len(set(pipe_counts)) == 1 and pipe_counts[0] > 3:
                return True

    if sum(c.isalnum() for c in text) < 10:
        return True

    return False

# ===============================
# IMAGE DECODING
# ===============================
def decode_image(b64: str) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    target_width = 1600
    scale = target_width / img.width
    img = img.resize((target_width, int(img.height * scale)), Image.BICUBIC)
    return img

def decode_pdf(b64: str) -> list:
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes, dpi=150, fmt="png", thread_count=4, use_pdftocairo=True
    )
    return images[:MAX_PAGES]

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ===============================
# OCR PROMPT
# ===============================
OCR_PROMPT_TEXT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML."
)

# ===============================
# LOAD SGLang ENGINE ONCE
# ===============================
def load_engine():
    global engine
    if engine is not None:
        return

    import sglang as sgl

    log("Starting SGLang engine...")
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="bfloat16",           # native tensor-core speed on Blackwell
        # FP8 quantization note: the FP8 checkpoint is loaded and run in bfloat16
        # because FP8 GEMM kernels are not yet stable on sm_120 (Blackwell).
        # Once sglang ships sm_120 FP8 kernels you can switch to dtype="fp8".
        mem_fraction_static=0.90,   # give VRAM to KV-cache; tune if OOM
        max_running_requests=BATCH_SIZE,
        trust_remote_code=True,
        log_level="warning",
        # RadixAttention is on by default — keeps shared prefixes in KV-cache.
        # All pages share the same system prompt so prefix caching is a big win.
        enable_prefix_caching=True,
    )
    log("SGLang engine ready")

# ===============================
# BATCH OCR via SGLang
# ===============================
def ocr_batch(images: list) -> list:
    """
    Send all pages as parallel async requests to the SGLang engine.
    SGLang handles scheduling, continuous batching, and prefix caching internally.
    """
    import sglang as sgl

    @sgl.function
    def ocr_page(s, image_b64: str):
        s += sgl.user(
            sgl.image(image_b64) + OCR_PROMPT_TEXT
        )
        s += sgl.assistant(
            sgl.gen(
                "result",
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0,
                repetition_penalty=1.1,
            )
        )

    # Build argument list — one dict per page
    args = [{"image_b64": image_to_b64(img)} for img in images]

    # run() dispatches all requests concurrently via the engine
    states = ocr_page.run_batch(
        args,
        num_threads=BATCH_SIZE,     # parallel HTTP workers feeding the engine
        progress_bar=False,
    )

    return [s["result"].strip() for s in states]

# ===============================
# HANDLER
# ===============================
def handler(event):
    load_engine()

    try:
        if "image" in event["input"]:
            pages = [decode_image(event["input"]["image"])]
        elif "file" in event["input"]:
            pages = decode_pdf(event["input"]["file"])
        else:
            return {"status": "error", "message": "Missing 'image' or 'file' in input"}

        total_pages = len(pages)
        log(f"Processing {total_pages} page(s) via SGLang...")
        start_time = time.time()

        batch_results = ocr_batch(pages)

        extracted_pages = []
        for j, text in enumerate(batch_results):
            page_num = j + 1

            if text.upper().startswith("EMPTY_PAGE"):
                text = "[Empty or unreadable page]"
            elif is_hallucinated_output(text):
                log(f"Warning: Page {page_num} flagged as hallucinated")
                text = "[Empty or unreadable page]"

            extracted_pages.append({"page": page_num, "text": text})

        elapsed = time.time() - start_time
        log(f"Completed {total_pages} pages in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

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
# PRELOAD & WARMUP
# ===============================
log("Preloading SGLang engine...")
load_engine()

log("Running warmup request...")
try:
    dummy = Image.new("RGB", (1600, 1200), color="white")
    _ = ocr_batch([dummy])
    log("Warmup complete!")
except Exception as e:
    log(f"Warmup error (non-fatal): {e}")

runpod.serverless.start({"handler": handler})
