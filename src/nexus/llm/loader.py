"""
loader.py — NEXUS Model Loader & Manager
Loads the Qwen 2.5 3B Instruct GGUF model via llama-cpp-python,
handles settings from config/settings.json, auto-detects and applies
LoRA adapter weights if present, and exposes a generate() function
for the rest of the application.

Also provides model management commands: resetting LoRA training,
viewing model status/info, and related utilities.

Originally split across loader.py and loader.py,
now unified into a single module for cleaner organisation.
"""

import contextlib
import json
import os
import shutil
import threading

import psutil
from llama_cpp import Llama

from nexus import config
from nexus.security import confirm
from nexus.utils import format_size

# ---------------------------------------------------------------------------
# Paths — resolved relative to the project root (one level up from src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(config.PROJECT_ROOT)

DEFAULT_MODEL_PATH = str(config.MODEL_DIR / config.DEFAULT_MODEL_NAME)
SETTINGS_PATH = str(config.SETTINGS_FILE)
LORA_WEIGHTS_PATH = str(config.LORA_WEIGHTS_DIR)
MODEL_NAME = "Qwen 2.5 3B Instruct (Q4_K_M GGUF)"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
# llama-cpp-python's Llama instance is not safe to call concurrently -- this
# guards every create_chat_completion() call so two overlapping requests
# (e.g. the UI and a background task) can't hit the model at the same time.
_inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load_settings():
    """
    Read config/settings.json and return a dict of settings.
    Falls back to sensible defaults if the file is missing or malformed.

    Reads the file fresh on every call (not cached) so that edits made
    through the Settings "Save" action take effect immediately, without
    requiring a restart.
    """
    defaults = {
        "model_path": DEFAULT_MODEL_PATH,
        "max_tokens": 512,
        "n_ctx": 8192,
        "n_threads": 4,
    }

    if not os.path.isfile(SETTINGS_PATH):
        return defaults

    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)

        # Merge loaded values over defaults so missing keys still have defaults
        for key in defaults:
            if key not in data:
                data[key] = defaults[key]

        # If model_path in settings is relative, resolve it from project root
        if not os.path.isabs(data["model_path"]):
            data["model_path"] = os.path.join(PROJECT_ROOT, data["model_path"])

        return data

    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARNING] Could not parse settings.json ({e}). Using defaults.")
        return defaults


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """
    Load the Qwen 2.5 3B Instruct GGUF model into the global `model` variable.
    Automatically applies LoRA adapter weights if they exist in lora-weights/.
    """
    global model

    settings = load_settings()
    model_path = settings["model_path"]
    n_ctx = settings.get("n_ctx", 4096)
    n_threads = settings.get("n_threads", 4)

    # --- Auto-detect optimal thread count -----------------------------------
    auto_threads = settings.get("auto_threads", True)
    if auto_threads:
        physical_cores = psutil.cpu_count(logical=False) or 4
        n_threads = max(4, physical_cores)
        print(f"Detected {physical_cores} CPU cores. Using {n_threads} threads.")

    # --- Verify model file exists -------------------------------------------
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Please download the GGUF model and place it in the model/ folder.")
        print("See the Install section of README.md for the download link.")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # --- Load base GGUF model (with progress dots) --------------------------
    print("Loading NEXUS model (Qwen 2.5 3B), please wait", end="", flush=True)

    # Show dots in a background thread so the user knows it hasn't frozen
    loading_done = threading.Event()

    def _show_progress():
        while not loading_done.is_set():
            print(".", end="", flush=True)
            loading_done.wait(timeout=2)  # print a dot every 2 seconds

    progress_thread = threading.Thread(target=_show_progress, daemon=True)
    progress_thread.start()

    lora_path, lora_status = find_lora_adapter()
    print(lora_status)

    try:
        # LoRA must be supplied at construction time; llama-cpp-python has no
        # method to attach an adapter to an already-loaded model.
        llama_kwargs = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "verbose": False,
            "chat_format": "chatml",
        }
        if lora_path:
            llama_kwargs["lora_path"] = lora_path
        model = Llama(**llama_kwargs)
    except Exception as e:
        loading_done.set()
        print()  # newline after dots
        print(f"[ERROR] Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}") from e

    loading_done.set()
    progress_thread.join(timeout=1)
    print(" Done!")  # finish the dots line

    # --- Warm up model with a tiny inference --------------------------------
    _warmup_model()


def _warmup_model():
    """
    Run a single tiny inference to warm up CPU caches and memory pages.
    This eliminates the cold-start delay on the first real user command.
    """
    global model
    if model is None:
        return

    print("Warming up...", end=" ", flush=True)
    try:
        model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=1,  # Generate only 1 token — just enough to warm caches
        )
        print("Ready!")
    except Exception:
        print("Done.")  # Non-critical — proceed even if warmup fails

def find_lora_adapter():
    """
    Locate a usable LoRA adapter in lora-weights/.

    llama.cpp only accepts GGUF adapters. A HuggingFace PEFT adapter (what
    training/trainer.py produces) must be converted first, so it is reported
    rather than silently ignored -- the previous behaviour made every training
    run look successful while the base model was actually still in use.

    Returns:
        (path, status) where path is a GGUF adapter path or None, and status
        is a human-readable explanation.
    """
    if not os.path.isdir(LORA_WEIGHTS_PATH):
        return None, "No custom training found -- using the base model."

    entries = os.listdir(LORA_WEIGHTS_PATH)

    gguf = [f for f in entries if f.endswith(".gguf")]
    if gguf:
        return os.path.join(LORA_WEIGHTS_PATH, gguf[0]), f"Applying LoRA adapter: {gguf[0]}"

    peft = [f for f in entries if f.endswith((".safetensors", ".bin"))]
    if peft or "adapter_config.json" in entries:
        return None, (
            "Found a HuggingFace PEFT adapter in lora-weights/, but llama.cpp "
            "needs a GGUF adapter. Convert it with llama.cpp's "
            "convert_lora_to_gguf.py, then restart NEXUS. Using the base model "
            "for now."
        )

    return None, "No custom training found -- using the base model."


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are NEXUS, a task-oriented offline AI assistant. "
    "Focus on being helpful for computer tasks such as file management, "
    "system monitoring, and summarisation. Answer clearly and concisely. "
    "For general knowledge questions, give brief answers in 2-3 sentences. "
    "Do not over-explain. Be direct and useful."
)


def _build_messages(prompt, system_prompt, conversation_history):
    """Build the ChatML messages list shared by generate() and generate_stream()."""
    sys_msg = system_prompt if system_prompt else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_msg},
    ]

    if conversation_history:
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    messages.append({"role": "user", "content": prompt})
    return messages


def generate(prompt, max_tokens=None, system_prompt=None, conversation_history=None):
    """
    Generate a response from the loaded model using ChatML chat completion.
    Includes a 120-second timeout to prevent hanging on long inputs.

    Args:
        prompt:                The user message / prompt string.
        max_tokens:            Maximum tokens to generate (default from settings).
        system_prompt:         Optional override for the system message.
        conversation_history:  Optional list of {"role": ..., "content": ...} dicts
                               from previous exchanges. Inserted between the system
                               prompt and the current user message to provide context.

    Returns:
        The generated text string, or an error message on failure.
    """
    global model

    # Lazy-load model if not yet loaded
    if model is None:
        load_model()

    # Resolve max_tokens from settings if not explicitly provided
    if max_tokens is None:
        settings = load_settings()
        max_tokens = settings.get("max_tokens", 512)

    messages = _build_messages(prompt, system_prompt, conversation_history)

    # Run inference with a timeout to prevent hanging
    result_container = [None]
    error_container = [None]

    def _run_inference():
        try:
            with _inference_lock:
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    stop=["\n\n\n", "<|im_end|>"],  # Stop early instead of generating until limit
                )
            result_container[0] = response["choices"][0]["message"]["content"]
        except Exception as e:
            error_container[0] = e

    inference_thread = threading.Thread(target=_run_inference, daemon=True)
    inference_thread.start()
    inference_thread.join(timeout=120)  # 120-second timeout

    if inference_thread.is_alive():
        # Inference is still running after timeout — don't wait
        print("\n[WARNING] Response timed out after 120 seconds.")
        return "Response timed out. The input may be too long. Try a shorter input."

    if error_container[0] is not None:
        print(f"[ERROR] Generation failed: {error_container[0]}")
        return "Error generating response."

    return result_container[0] or "Error generating response."


def generate_stream(prompt, max_tokens=None, system_prompt=None, conversation_history=None):
    """
    Like generate(), but yields text chunks as the model produces them
    instead of returning the full response in one blocking call. Used by
    the web UI's WebSocket endpoint to stream tokens live, replacing the
    old Tkinter UI's blocking "Thinking..." placeholder swap (V4.0 Phase 41).

    Args:
        Same as generate().

    Yields:
        Successive text chunks; concatenating them gives the full response.
        Yields a single error-message chunk instead if generation fails.
    """
    global model

    if model is None:
        load_model()

    if max_tokens is None:
        settings = load_settings()
        max_tokens = settings.get("max_tokens", 512)

    messages = _build_messages(prompt, system_prompt, conversation_history)

    try:
        with _inference_lock:
            stream = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stop=["\n\n\n", "<|im_end|>"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta
    except Exception as e:
        print(f"[ERROR] Streaming generation failed: {e}")
        yield "Error generating response."


# ===========================================================================
# Model Management (originally in loader.py)
# ===========================================================================

def _get_dir_size(path):
    """Calculate total size of all files in a directory (recursive)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            with contextlib.suppress(OSError):
                total += os.path.getsize(fp)
    return total


# ---------------------------------------------------------------------------
# Reset training — delete all LoRA adapter files
# ---------------------------------------------------------------------------
def reset_training():
    """
    Delete all files inside the lora-weights/ folder to reset custom training.
    Lists files before deletion and asks for user confirmation.

    Returns:
        A status message string.
    """
    # Check if the folder exists
    if not os.path.isdir(LORA_WEIGHTS_PATH):
        return "No training data found. The lora-weights/ folder does not exist."

    # List all files and subdirectories
    contents = os.listdir(LORA_WEIGHTS_PATH)

    # Filter out empty directories — check if there's anything to delete
    if not contents:
        return "No training data found. The lora-weights/ folder is already empty."

    # Build a summary of what will be deleted
    print("\n  Files/folders that will be deleted:")
    print("  " + "-" * 44)

    files_to_remove = []
    dirs_to_remove = []
    total_size = 0

    for item in sorted(contents):
        item_path = os.path.join(LORA_WEIGHTS_PATH, item)
        if os.path.isdir(item_path):
            dir_size = _get_dir_size(item_path)
            total_size += dir_size
            print(f"    [DIR]  {item}  ({format_size(dir_size)})")
            dirs_to_remove.append(item_path)
        else:
            file_size = os.path.getsize(item_path)
            total_size += file_size
            print(f"    [FILE] {item}  ({format_size(file_size)})")
            files_to_remove.append(item_path)

    print("  " + "-" * 44)
    print(f"  Total: {len(files_to_remove)} file(s), {len(dirs_to_remove)} folder(s) — {format_size(total_size)}")
    print()

    if not confirm.request(
        "Delete all custom training data?",
        f"{len(files_to_remove)} file(s) and {len(dirs_to_remove)} folder(s) "
        f"({format_size(total_size)}) will be removed.\n\n"
        "NEXUS will revert to the base model. This cannot be undone.",
    ):
        return confirm.denied_message("Reset")

    # Delete individual files
    for fp in files_to_remove:
        try:
            os.remove(fp)
        except OSError as e:
            print(f"  [WARNING] Could not delete {os.path.basename(fp)}: {e}")

    # Delete subdirectories (checkpoint folders etc.)
    for dp in dirs_to_remove:
        try:
            shutil.rmtree(dp)
        except OSError as e:
            print(f"  [WARNING] Could not delete folder {os.path.basename(dp)}: {e}")

    return "Training data cleared. NEXUS will use the base model on next start."


# ---------------------------------------------------------------------------
# Model info — show current model status and settings
# ---------------------------------------------------------------------------
def get_model_info():
    """
    Return a formatted string showing the current model status:
    - Base model name
    - LoRA adapter status (applied / not applied)
    - Adapter file details and total size
    - Current settings from config/settings.json
    """
    lines = []
    lines.append("")
    lines.append("  NEXUS — Model Information")
    lines.append("  " + "-" * 40)

    # Base model
    lines.append(f"  Base Model:    {MODEL_NAME}")

    # LoRA status
    if os.path.isdir(LORA_WEIGHTS_PATH):
        lora_files = [
            f for f in os.listdir(LORA_WEIGHTS_PATH)
            if f.endswith((".bin", ".gguf", ".safetensors"))
        ]
        if lora_files:
            total_size = _get_dir_size(LORA_WEIGHTS_PATH)
            lines.append(f"  LoRA Status:   Applied ({len(lora_files)} adapter file(s))")
            lines.append(f"  Adapter Size:  {format_size(total_size)}")
            for lf in sorted(lora_files):
                fsize = os.path.getsize(os.path.join(LORA_WEIGHTS_PATH, lf))
                lines.append(f"                 - {lf} ({format_size(fsize)})")
        else:
            lines.append("  LoRA Status:   Not applied (no adapter files found)")
    else:
        lines.append("  LoRA Status:   Not applied (lora-weights/ folder missing)")

    # Settings from config/settings.json
    lines.append("")
    lines.append("  Current Settings:")
    lines.append("  " + "-" * 40)

    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            settings = json.load(f)
        lines.append(f"  Context Window: {settings.get('n_ctx', 'N/A')} tokens")
        lines.append(f"  Max Tokens:     {settings.get('max_tokens', 'N/A')}")
        lines.append(f"  Threads:        {settings.get('n_threads', 'N/A')}")
        lines.append(f"  Auto Threads:   {settings.get('auto_threads', 'N/A')}")
    except (FileNotFoundError, json.JSONDecodeError):
        lines.append("  [Could not read settings.json]")

    lines.append("  " + "-" * 40)
    lines.append("")

    return "\n".join(lines)
