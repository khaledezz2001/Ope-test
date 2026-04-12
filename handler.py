import os
import base64
import io
import time
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
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
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/allenai/olmOCR-2-7B-1025-FP8"
MAX_PAGES = 100

llm_engine = None
processor = None
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1536,
    repetition_penalty=1.1
)

def log(msg):
    print(f"[BOOT] {msg}", flush=True)

# ===============================
# HALLUCINATION DETECTION
# ===============================
def is_hallucinated_output(text: str) -> bool:
    """Detect if the OCR output is hallucinated/garbage"""
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
            
    table_markers = text.count('|')
    pipe_lines = sum(1 for line in lines if '|' in line)
    
    if len(lines) > 0 and pipe_lines / len(lines) > 0.5:
        content_without_pipes = text.replace('|', '').replace('-', '').replace('\n', '').strip()
        if len(content_without_pipes) < 100:
            return True
            
    if table_markers > 10:
        table_rows = [line for line in lines if '|' in line]
        if len(table_rows) > 3:
            pipe_counts = [line.count('|') for line in table_rows]
            if len(set(pipe_counts)) == 1 and pipe_counts[0] > 3:
                return True
                
    alphanumeric_chars = sum(c.isalnum() for c in text)
    if alphanumeric_chars < 10:
        return True
        
    return False

# ===============================
# IMAGE DECODING
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    target_width = 1600
    scale = target_width / img.width
    img = img.resize((target_width, int(img.height * scale)), Image.BICUBIC)
    return img

def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes, dpi=150, fmt="png", thread_count=4, use_pdftocairo=True
    )
    return images[:MAX_PAGES]

# ===============================
# LOAD MODEL ONCE
# ===============================
def load_model():
    global processor, llm_engine
    if llm_engine is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)

    log("Loading vLLM engine...")
    llm_engine = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tokenizer_mode="auto",      # required for multimodal in vLLM 0.7.x
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.9,
    )
    log("vLLM engine loaded successfully")

# ===============================
# OCR PROMPT
# ===============================
OCR_PROMPT_TEXT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML."
)

def build_messages_for_image():
    """Build the chat messages for a single image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": OCR_PROMPT_TEXT}
            ]
        }
    ]

# ===============================
# BATCH OCR (vLLM natively)
# ===============================
def ocr_batch(images: list) -> list:
    """Process a batch of images using vLLM's lightning fast generate."""
    
    vllm_inputs = []
    messages = build_messages_for_image()
    
    # Build standard chat templates text
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    
    # We queue up all images at once; vLLM handles parallel streaming internally!
    for img in images:
        vllm_inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img}
        })
        
    outputs = llm_engine.generate(vllm_inputs, sampling_params=sampling_params)
    
    # Extract results
    results = []
    for output in outputs:
        # Get the highest probability string (index 0)
        decoded = output.outputs[0].text
        
        if "assistant" in decoded.lower():
            idx = decoded.lower().index("assistant") + len("assistant")
            decoded = decoded[idx:]
            
        results.append(decoded.strip())
        
    return results

# ===============================
# HANDLER
# ===============================
def handler(event):
    load_model()
    
    try:
        if "image" in event["input"]:
            pages = [decode_image(event["input"]["image"])]
        elif "file" in event["input"]:
            pages = decode_pdf(event["input"]["file"])
        else:
            return {"status": "error", "message": "Missing image or file"}

        total_pages = len(pages)
        log(f"Processing {total_pages} pages using vLLM engine...")
        start_time = time.time()

        # Let vLLM digest all images instantly instead of manual small looping chunks.
        # It handles batching internal to its CUDA graphs automatically based on available limits.
        batch_results = ocr_batch(pages)
            
        extracted_pages = []
        for j, text in enumerate(batch_results):
            page_num = j + 1
            
            if text.upper() == "EMPTY_PAGE" or text.upper().startswith("EMPTY_PAGE"):
                text = "[Empty or unreadable page]"
            elif is_hallucinated_output(text):
                log(f"Warning: Page {page_num} appears to be hallucinated")
                text = "[Empty or unreadable page]"
            
            extracted_pages.append({
                "page": page_num,
                "text": text
            })
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        log(f"Completed {total_pages} pages cleanly in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages
        }

    except Exception as e:
        log(f"Error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "error", "message": str(e)}

# ===============================
# PRELOAD & WARMUP
# ===============================
log("Preloading model wrapper...")
load_model()

if torch.cuda.is_available():
    log("Running dummy warmup...")
    dummy_image = Image.new('RGB', (1600, 1200), color='white')
    try:
        _ = ocr_batch([dummy_image])
        log("Warmup complete!")
    except Exception as e:
        log(f"Warmup generated error log: {e}")

runpod.serverless.start({
    "handler": handler
})
