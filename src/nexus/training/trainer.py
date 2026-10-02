"""
trainer.py — NEXUS Training Pipeline
Fine-tunes the Qwen 2.5 3B Instruct model using LoRA (Low-Rank Adaptation)
on user-provided Q&A datasets. Produces adapter weights that are automatically
loaded by loader.py on the next agent start.

Usage:  python src/trainer.py   (from the nexus-agent root folder)
        or double-click train.bat

NOTE: Training a 3B model on CPU requires 16 GB+ RAM and will be significantly
slower than smaller models (expect 1-3 hours for 50-100 Q&A pairs). If your
PC has less than 16 GB RAM, keep datasets small (under 50 pairs) or use
Google Colab (free GPU) and copy the lora-weights/ folder back.
"""

import argparse
import csv
import json
import os
import sys
import warnings

import torch

from nexus import config

# Suppress the pin_memory warning (no GPU accelerator available). This must run
# before transformers is imported, so the imports below are deliberately not at
# the top of the file.
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(config.PROJECT_ROOT)
DATASETS_FOLDER = str(config.DATASETS_DIR)
LORA_OUTPUT_FOLDER = str(config.LORA_WEIGHTS_DIR)
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# ---------------------------------------------------------------------------
# 10.2 — Validate datasets
# ---------------------------------------------------------------------------
def validate_datasets():
    """
    Check the datasets/ folder exists and contains valid dataset files.
    Supports: .txt, .csv, .json, .jsonl, .xlsx
    Returns a list of valid file paths.
    Exits with an error if no usable datasets are found.
    """
    SUPPORTED_EXTENSIONS = (".txt", ".csv", ".json", ".jsonl", ".xlsx")

    if not os.path.isdir(DATASETS_FOLDER):
        print(f"[ERROR] Datasets folder not found: {DATASETS_FOLDER}")
        print("Please create the 'datasets' folder and add dataset files.")
        sys.exit(1)

    dataset_files = [
        f for f in os.listdir(DATASETS_FOLDER)
        if f.lower().endswith(SUPPORTED_EXTENSIONS) and f.lower() != "readme.txt"
    ]

    if not dataset_files:
        print("No dataset files found. Add .txt, .csv, .json, .jsonl, or .xlsx files to the datasets/ folder.")
        print("See datasets/README.txt for the required format.")
        sys.exit(1)

    valid_files = []
    for filename in dataset_files:
        filepath = os.path.join(DATASETS_FOLDER, filename)
        if os.path.getsize(filepath) == 0:
            print(f"[WARNING] Skipping empty file: {filename}")
            continue
        valid_files.append(filepath)
        print(f"  Found dataset: {filename}")

    if not valid_files:
        print("[ERROR] All dataset files are empty. Please add Q&A content.")
        sys.exit(1)

    return valid_files


# ---------------------------------------------------------------------------
# 10.3 — Parse Q&A pairs from a dataset file
# ---------------------------------------------------------------------------
def parse_qa_pairs(filepath):
    """
    Read Q:/A: pairs from a dataset file.
    Expected format (separated by blank lines):

        Q: What is Python?
        A: Python is a high-level programming language.

        Q: What is AI?
        A: AI stands for Artificial Intelligence.

    Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
    """
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split into blocks separated by blank lines
    blocks = content.split("\n\n")
    pairs = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Skip comment lines
        lines = [ln for ln in block.split("\n") if not ln.strip().startswith("#")]
        if not lines:
            continue

        question = None
        answer_lines = []
        reading_answer = False

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("Q:"):
                question = stripped[2:].strip()
                reading_answer = False
            elif stripped.upper().startswith("A:"):
                answer_lines = [stripped[2:].strip()]
                reading_answer = True
            elif reading_answer:
                # Multi-line answers: continue appending
                answer_lines.append(stripped)

        if question and answer_lines:
            answer = " ".join(answer_lines)
            pairs.append({"question": question, "answer": answer})

    return pairs


# ---------------------------------------------------------------------------
# 15.1 — Parse Q&A pairs from a CSV file
# ---------------------------------------------------------------------------
def parse_csv_pairs(filepath):
    """
    Read Q&A pairs from a .csv file.
    Supports two column layouts:
      - Named columns: "question"/"Q" and "answer"/"A" (case-insensitive)
      - Positional: first column = question, second column = answer
    Auto-detects header row. Handles quoted fields and commas inside values.
    Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
    """
    pairs = []
    # Known header names (lowercase) for question and answer columns
    q_headers = {"question", "q"}
    a_headers = {"answer", "a"}

    with open(filepath, encoding="utf-8", errors="ignore", newline="") as f:
        # Sniff to check if there's a header
        sample = f.read(2048)
        f.seek(0)
        sniffer = csv.Sniffer()
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False

        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pairs

    # Try to detect header from first row
    first_row_lower = [cell.strip().lower() for cell in rows[0]]
    q_col = None
    a_col = None

    # Check if first row contains known header names
    for i, cell in enumerate(first_row_lower):
        if cell in q_headers:
            q_col = i
        elif cell in a_headers:
            a_col = i

    if q_col is not None and a_col is not None:
        # Header row detected by name — skip it
        data_rows = rows[1:]
    elif has_header:
        # Sniffer says there's a header but names don't match — skip it, use positional
        q_col = 0
        a_col = 1
        data_rows = rows[1:]
    else:
        # No header detected — use positional columns
        q_col = 0
        a_col = 1
        data_rows = rows

    for row in data_rows:
        if len(row) <= max(q_col, a_col):
            continue  # Not enough columns
        question = row[q_col].strip()
        answer = row[a_col].strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


# ---------------------------------------------------------------------------
# 15.2 — Parse Q&A pairs from JSON / JSONL files
# ---------------------------------------------------------------------------
def parse_json_pairs(filepath):
    """
    Read Q&A pairs from a .json file.
    Supports formats:
      - [{"question": "...", "answer": "..."}, ...]
      - [{"Q": "...", "A": "..."}, ...]
    Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
    """
    pairs = []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARNING] Could not parse JSON file {os.path.basename(filepath)}: {e}")
            return pairs

    if not isinstance(data, list):
        print(f"[WARNING] JSON file {os.path.basename(filepath)} is not a list of objects. Skipping.")
        return pairs

    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            print(f"[WARNING] Skipping non-object entry at index {i}")
            continue
        # Case-insensitive key lookup
        keys_lower = {k.lower(): k for k in obj}
        q_key = keys_lower.get("question") or keys_lower.get("q")
        a_key = keys_lower.get("answer") or keys_lower.get("a")
        if q_key and a_key:
            question = str(obj[q_key]).strip()
            answer = str(obj[a_key]).strip()
            if question and answer:
                pairs.append({"question": question, "answer": answer})
        else:
            print(f"[WARNING] Skipping entry at index {i} — missing question/answer keys")

    return pairs


def parse_jsonl_pairs(filepath):
    """
    Read Q&A pairs from a .jsonl (JSON Lines) file.
    Each line is one JSON object with question/answer or Q/A fields.
    Malformed lines are skipped with a warning.
    Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
    """
    pairs = []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARNING] Skipping malformed JSON at line {line_num} in {os.path.basename(filepath)}")
                continue
            if not isinstance(obj, dict):
                print(f"[WARNING] Skipping non-object at line {line_num}")
                continue
            # Case-insensitive key lookup
            keys_lower = {k.lower(): k for k in obj}
            q_key = keys_lower.get("question") or keys_lower.get("q")
            a_key = keys_lower.get("answer") or keys_lower.get("a")
            if q_key and a_key:
                question = str(obj[q_key]).strip()
                answer = str(obj[a_key]).strip()
                if question and answer:
                    pairs.append({"question": question, "answer": answer})
            else:
                print(f"[WARNING] Skipping line {line_num} — missing question/answer keys")

    return pairs


# ---------------------------------------------------------------------------
# 15.3 — Parse Q&A pairs from Excel (.xlsx) files
# ---------------------------------------------------------------------------
def parse_excel_pairs(filepath):
    """
    Read Q&A pairs from an .xlsx file using openpyxl.
    Reads the first sheet. Expects first column = question, second column = answer.
    Auto-detects header row: skips if first row contains "question"/"Q" etc.
    Skips empty rows.
    Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
    """
    import openpyxl

    pairs = []
    q_headers = {"question", "q"}
    a_headers = {"answer", "a"}

    try:
        wb = openpyxl.load_workbook(filepath, data_only=True)
    except Exception as e:
        print(f"[WARNING] Could not open Excel file {os.path.basename(filepath)}: {e}")
        return pairs

    sheet = wb.active
    if sheet is None:
        print(f"[WARNING] No active sheet found in {os.path.basename(filepath)}")
        return pairs

    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        return pairs

    # Check if first row is a header
    first_row = rows[0]
    if (len(first_row) >= 2
            and first_row[0] is not None
            and first_row[1] is not None
            and str(first_row[0]).strip().lower() in q_headers
            and str(first_row[1]).strip().lower() in a_headers):
        data_rows = rows[1:]
    else:
        data_rows = rows

    for row in data_rows:
        if len(row) < 2:
            continue
        q_val = row[0]
        a_val = row[1]
        if q_val is None or a_val is None:
            continue
        question = str(q_val).strip()
        answer = str(a_val).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


# ---------------------------------------------------------------------------
# 15.4 — Dataset file dispatcher (picks the right parser by extension)
# ---------------------------------------------------------------------------
def parse_dataset_file(filepath):
    """
    Dispatcher that picks the correct parser based on file extension.
    Supports: .txt, .csv, .json, .jsonl, .xlsx
    Returns a list of dicts: [{"question": "...", "answer": "..."}, ...]
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return parse_qa_pairs(filepath)
    if ext == ".csv":
        return parse_csv_pairs(filepath)
    if ext == ".json":
        return parse_json_pairs(filepath)
    if ext == ".jsonl":
        return parse_jsonl_pairs(filepath)
    if ext in (".xlsx", ".xls"):
        return parse_excel_pairs(filepath)
    print(f"[WARNING] Unsupported file format: {ext}")
    return []


# ---------------------------------------------------------------------------
# 10.4 — Format Q&A pairs for Qwen 2.5 ChatML training
# ---------------------------------------------------------------------------
def format_for_training(qa_pairs):
    """
    Convert Q&A pairs into Qwen 2.5 ChatML format strings for training.
    Returns a list of formatted text strings.
    """
    formatted = []
    for pair in qa_pairs:
        text = (
            "<|im_start|>system\n"
            "You are NEXUS, a helpful offline assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{pair['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{pair['answer']}<|im_end|>"
        )
        formatted.append(text)
    return formatted


# ---------------------------------------------------------------------------
# 10.5 — Load tokeniser and tokenise the dataset
# ---------------------------------------------------------------------------
def load_and_tokenise(formatted_texts):
    """
    Load the Qwen 2.5 tokeniser, tokenise all training texts, and split
    into train/validation datasets.
    Returns (tokeniser, train_dataset, val_dataset).
    """
    print("Loading tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenising {len(formatted_texts)} training examples...")

    # Tokenise all texts
    encodings = tokenizer(
        formatted_texts,
        max_length=256,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )

    # Build a Hugging Face Dataset — labels are same as input_ids for causal LM
    data_dict = {
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "labels": encodings["input_ids"].tolist(),
    }
    dataset = Dataset.from_dict(data_dict)

    # Split into train (90%) and validation (10%)
    if len(dataset) < 2:
        # Too few examples to split — use same data for both
        print("[WARNING] Very few training examples. Using same data for train and validation.")
        return tokenizer, dataset, dataset

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")

    return tokenizer, train_dataset, val_dataset


# ---------------------------------------------------------------------------
# 10.6 — Setup LoRA model
# ---------------------------------------------------------------------------
def setup_lora_model():
    """
    Load the base Qwen 2.5 3B model and apply LoRA configuration.
    Returns the PEFT model ready for training.
    """
    print("Loading base model for training...")
    print("(This may take a few minutes and use significant RAM)")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Define LoRA configuration
    # r=4 with q_proj+v_proj gives a good speed/quality balance for small datasets
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# 10.7 — Run training
# ---------------------------------------------------------------------------
def run_training(model, tokenizer, train_dataset, val_dataset, quick_mode=False):
    """
    Run the LoRA fine-tuning training loop.
    Saves the adapter weights to lora-weights/ when complete.
    """
    # Ensure output directory exists
    os.makedirs(LORA_OUTPUT_FOLDER, exist_ok=True)

    # Quick mode uses aggressive settings for fastest possible training
    if quick_mode:
        epochs = 1
        lr = 5e-4
    else:
        epochs = 1
        lr = 2e-4

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_FOLDER,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",
        logging_steps=1,
        learning_rate=lr,
        fp16=False,
        gradient_checkpointing=True,
        save_total_limit=2,
        report_to="none",  # disable wandb / tensorboard
        eval_strategy="no",
    )

    # Data collator for causal language modelling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print()
    print("=" * 60)
    print("  Training started.")
    print("  Estimated time: 10-30 min for 10-20 Q&A pairs (CPU).")
    print("  Progress will be shown below.")
    print("=" * 60)
    print()

    trainer.train()

    # Save the LoRA adapter weights
    print("\nSaving LoRA adapter weights...")
    model.save_pretrained(LORA_OUTPUT_FOLDER)
    tokenizer.save_pretrained(LORA_OUTPUT_FOLDER)

    print()
    print("=" * 60)
    print("  Training complete! Model saved to lora-weights/")
    print("  Restart the agent (start.bat) to use your custom model.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 10.8 — Main training orchestrator
# ---------------------------------------------------------------------------
def main():
    """Orchestrate the full training pipeline end-to-end."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="NEXUS Training Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick training mode (1 epoch, higher learning rate)")
    args = parser.parse_args()

    print("=" * 60)
    print("  NEXUS — Training Pipeline")
    if args.quick:
        print("  [QUICK MODE] — Faster training, slightly lower quality")
    print("=" * 60)
    print()

    # Step 1: Validate datasets
    print("Step 1: Checking datasets...")
    dataset_files = validate_datasets()
    print()

    # Step 2: Parse Q&A pairs from all dataset files
    print("Step 2: Parsing Q&A pairs...")
    all_pairs = []
    for filepath in dataset_files:
        pairs = parse_dataset_file(filepath)
        filename = os.path.basename(filepath)
        print(f"  {filename}: {len(pairs)} Q&A pairs")
        all_pairs.extend(pairs)
    print()

    # Step 3: Report total
    print(f"Total Q&A pairs found: {len(all_pairs)}")
    if len(all_pairs) == 0:
        print("[ERROR] No valid Q&A pairs found. Check your dataset format.")
        print("Expected format:  Q: question text\\nA: answer text")
        sys.exit(1)
    print()

    # Step 4: Format for training
    print("Step 3: Formatting for Qwen 2.5 ChatML...")
    formatted_texts = format_for_training(all_pairs)
    print(f"  Formatted {len(formatted_texts)} training examples.")
    print()

    # Step 5: Tokenise
    print("Step 4: Loading tokeniser and preparing dataset...")
    tokenizer, train_dataset, val_dataset = load_and_tokenise(formatted_texts)
    print()

    # Step 6: Setup LoRA model
    print("Step 5: Setting up LoRA model...")
    model = setup_lora_model()
    print()

    # Step 7: Train
    print("Step 6: Starting training...")
    run_training(model, tokenizer, train_dataset, val_dataset, quick_mode=args.quick)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
