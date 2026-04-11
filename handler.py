import os
import base64
import io
import time
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from pdf2image import convert_from_bytes

# ===============================
# OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# MEMORY OPTIMIZATION
# ===============================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "/models/hf/allenai/olmOCR-2-7B-1025-FP8"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PAGES = 100

processor = None
model = None


def get_optimal_batch_size():
    """Auto-detect the best batch size based on GPU VRAM."""
    if not torch.cuda.is_available():
        return 1  # CPU — process one at a time

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)

    if vram_gb >= 70:      # A100 80GB, H100 80GB
        batch = 8
    elif vram_gb >= 40:    # A40 48GB, A100 40GB, A6000 48GB
        batch = 6
    elif vram_gb >= 28:    # RTX 5090 32GB
        batch = 5
    elif vram_gb >= 20:    # RTX 4090 24GB, RTX 3090 24GB, A5000 24GB
        batch = 4
    elif vram_gb >= 14:    # RTX 4080 16GB, T4 16GB, V100 16GB
        batch = 2
    else:                  # Smaller GPUs
        batch = 1

    print(f"[BOOT] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) → BATCH_SIZE={batch}", flush=True)
    return batch


BATCH_SIZE = get_optimal_batch_size()

# ===============================
# GPU OPTIMIZATIONS
# ===============================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


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
    
    # Common hallucination phrases that models generate for empty pages
    hallucination_indicators = [
        "table 1:",
        "comparison of different methods",
        "note: the choice of method",
        "this page is blank",
        "no text found",
        "empty page",
        "the image appears to be",
        "there is no visible text",
        "the document appears to be blank",
        "i cannot see any text",
        "method | accuracy | speed",
        "soil moisture",
        "time domain reflectometry"
    ]
    
    # Check if text contains hallucination phrases
    for indicator in hallucination_indicators:
        if indicator in text_lower:
            return True
    
    # Check for repetitive table patterns
    lines = text.strip().split('\n')
    if len(lines) > 20:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < 3:
            return True
    
    # Check for excessive markdown tables (generic hallucinations)
    table_markers = text.count('|')
    pipe_lines = sum(1 for line in lines if '|' in line)
    
    # If more than 50% of lines have pipes, likely a hallucinated table
    if len(lines) > 0 and pipe_lines / len(lines) > 0.5:
        # Check if it's a real table with actual content or generic hallucination
        content_without_pipes = text.replace('|', '').replace('-', '').replace('\n', '').strip()
        if len(content_without_pipes) < 100:  # Too little actual content
            return True
    
    # Check for suspiciously perfect table formatting (hallucination signature)
    if table_markers > 10:
        # Real tables usually have irregular content
        # Hallucinated tables often have very uniform structure
        table_rows = [line for line in lines if '|' in line]
        if len(table_rows) > 3:
            # Count pipes per row
            pipe_counts = [line.count('|') for line in table_rows]
            # If all rows have exactly the same number of pipes, suspicious
            if len(set(pipe_counts)) == 1 and pipe_counts[0] > 3:
                return True
    
    # Check for only special characters
    alphanumeric_chars = sum(c.isalnum() for c in text)
    if alphanumeric_chars < 10:
        return True
    
    return False


# ===============================
# IMAGE DECODING (SAME QUALITY)
# ===============================
def decode_image(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    # Balanced resolution — UNCHANGED to preserve quality
    target_width = 1600
    scale = target_width / img.width
    img = img.resize(
        (target_width, int(img.height * scale)),
        Image.BICUBIC
    )
    return img


def decode_pdf(b64):
    pdf_bytes = base64.b64decode(b64)
    images = convert_from_bytes(
        pdf_bytes,
        dpi=150,
        fmt="png",
        thread_count=4,
        use_pdftocairo=True
    )
    return images[:MAX_PAGES]


# ===============================
# LOAD MODEL ONCE
# ===============================
def load_model():
    global processor, model
    if model is not None:
        return

    log("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading model...")
    
    # Determine best dtype for the GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        # A40, A100, H100, RTX 30xx/40xx support BF16 natively
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            log("Using BF16 (native support)")
        else:
            dtype = torch.float16
            log("Using FP16")
    else:
        dtype = torch.float32
    
    # Try loading with Flash Attention 2 for faster inference
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        )
        log("Loaded with Flash Attention 2")
    except Exception as e:
        log(f"Flash Attention 2 not available ({e}), using default attention")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True
        )

    model.eval()
    
    # Try torch.compile for kernel fusion (speeds up repeated calls)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log("Model compiled with torch.compile")
    except Exception as e:
        log(f"torch.compile not available ({e}), using eager mode")
    
    log("olmOCR model loaded")


# ===============================
# OCR PROMPT (cached once)
# ===============================
OCR_PROMPT_TEXT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML."
)


def build_messages_for_image():
    """Build the chat messages for a single image — reused for every page."""
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
# BATCH OCR — process multiple pages at once
# ===============================
def ocr_batch(images: list) -> list:
    """Process a batch of images in a single model.generate() call."""
    
    # Build prompts for all images in the batch
    prompts = []
    for _ in images:
        messages = build_messages_for_image()
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # Process all images at once
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(DEVICE, non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1536,
            min_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    # Decode all outputs
    decoded_list = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Clean up each response
    results = []
    for decoded in decoded_list:
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

    # Prefix to remove from output
    PREFIX = ".\nuser\n" + OCR_PROMPT_TEXT + "\nassistant\n"

    try:
        if "image" in event["input"]:
            pages = [decode_image(event["input"]["image"])]
        elif "file" in event["input"]:
            pages = decode_pdf(event["input"]["file"])
        else:
            return {
                "status": "error",
                "message": "Missing image or file"
            }

        total_pages = len(pages)
        log(f"Processing {total_pages} pages in batches of {BATCH_SIZE}...")
        start_time = time.time()

        extracted_pages = []

        # Process pages in batches instead of one-by-one
        for batch_start in range(0, total_pages, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_pages)
            batch_images = pages[batch_start:batch_end]
            
            log(f"Processing batch: pages {batch_start + 1}-{batch_end} of {total_pages}")
            
            # Run OCR on the entire batch at once
            batch_results = ocr_batch(batch_images)
            
            for j, text in enumerate(batch_results):
                page_num = batch_start + j + 1
                
                # Remove prefix
                text = text.replace(PREFIX, "", 1).strip()
                
                # Check if model explicitly said it's empty
                if text.upper() == "EMPTY_PAGE" or text.upper().startswith("EMPTY_PAGE"):
                    text = "[Empty or unreadable page]"
                # Detect hallucinations
                elif is_hallucinated_output(text):
                    log(f"Warning: Page {page_num} appears to be hallucinated")
                    text = "[Empty or unreadable page]"
                
                extracted_pages.append({
                    "page": page_num,
                    "text": text
                })
            
            # Clear cache between batches (not between every page)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        log(f"Completed {total_pages} pages in {elapsed:.1f}s ({elapsed/total_pages:.1f}s/page)")

        return {
            "status": "success",
            "total_pages": len(extracted_pages),
            "pages": extracted_pages
        }

    except Exception as e:
        log(f"Error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "error",
            "message": str(e)
        }


# ===============================
# PRELOAD & WARMUP
# ===============================
log("Preloading model...")
load_model()

# Warmup with smaller image
if torch.cuda.is_available():
    log("Running warmup...")
    dummy_image = Image.new('RGB', (1600, 1200), color='white')
    try:
        _ = ocr_batch([dummy_image])
        torch.cuda.empty_cache()
        log("Warmup complete")
    except Exception as e:
        log(f"Warmup failed: {e}")

runpod.serverless.start({
    "handler": handler
})
