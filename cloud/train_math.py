#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen2.5-Coder-14B for math using Unsloth.

Fine-tunes the base model with LoRA adapters on math datasets,
then saves both the adapter and merged model for GGUF export.

Usage:
    python train_math.py --dataset-dir ../data
    python train_math.py --dataset-dir ../data --epochs 5 --lr 1e-4
    python train_math.py --dataset-dir ../data --resume-from /output/checkpoint-500

Author: Morningstar Contributors
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# ChatML Formatting
# ---------------------------------------------------------------------------
def format_chatml(example: dict) -> str:
    """Convert a conversation to ChatML string for training."""
    convs = example.get("conversations", [])
    if not convs:
        # Fallback for flat format
        system = example.get("system", "")
        user = example.get("user", example.get("instruction", ""))
        assistant = example.get("assistant", example.get("output", ""))
        convs = []
        if system:
            convs.append({"role": "system", "content": system})
        if user:
            convs.append({"role": "user", "content": user})
        if assistant:
            convs.append({"role": "assistant", "content": assistant})

    parts = []
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return "\n".join(parts)


def formatting_func(examples: dict) -> list[str]:
    """Batch formatting function for SFTTrainer."""
    texts = []
    # Handle batched input
    if isinstance(examples.get("conversations"), list) and \
       len(examples["conversations"]) > 0 and \
       isinstance(examples["conversations"][0], list):
        # Batched
        for i in range(len(examples["conversations"])):
            row = {k: examples[k][i] for k in examples}
            texts.append(format_chatml(row))
    else:
        texts.append(format_chatml(examples))
    return texts


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_datasets(dataset_dir: str) -> tuple[Dataset, Dataset]:
    """Load train and eval datasets from JSONL files."""
    ds_dir = Path(dataset_dir)

    # Try to find train/val files
    train_path = None
    eval_path = None

    for name in ["math_train.jsonl", "train.jsonl"]:
        p = ds_dir / name
        if p.exists():
            train_path = p
            break

    for name in ["math_val.jsonl", "val.jsonl", "eval.jsonl"]:
        p = ds_dir / name
        if p.exists():
            eval_path = p
            break

    if train_path is None:
        # Try loading any JSONL file
        jsonl_files = list(ds_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"ERROR: No JSONL files found in {ds_dir}")
            sys.exit(1)
        train_path = jsonl_files[0]
        print(f"Using {train_path} as training data")

    train_data = load_jsonl(str(train_path))
    print(f"Loaded {len(train_data)} training examples from {train_path}")

    if eval_path:
        eval_data = load_jsonl(str(eval_path))
        print(f"Loaded {len(eval_data)} eval examples from {eval_path}")
    else:
        # Auto-split: 95% train, 5% eval
        split = int(len(train_data) * 0.95)
        eval_data = train_data[split:]
        train_data = train_data[:split]
        print(f"Auto-split: {len(train_data)} train, {len(eval_data)} eval")

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-Coder-14B for math with QLoRA + Unsloth"
    )
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Directory containing JSONL dataset files")
    parser.add_argument("--output-dir", type=str, default="./output/math-qlora",
                        help="Output directory for model and checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MORNINGSTAR-MATH QLoRA Training")
    print("=" * 60)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size} x {args.gradient_accumulation} = "
          f"{args.batch_size * args.gradient_accumulation}")
    print("=" * 60)

    # Load model with Unsloth
    print("\nLoading model with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load datasets
    print("\nLoading datasets...")
    train_ds, eval_ds = load_datasets(args.dataset_dir)

    # Wandb
    report_to = "wandb" if args.wandb else "tensorboard"
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", "morningstar-math")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=10,
        report_to=report_to,
        logging_dir=str(output_dir / "logs"),
        optim="adamw_8bit",
        seed=42,
        group_by_length=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        formatting_func=formatting_func,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # GPU stats
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Training for {args.epochs} epoch(s) with lr={args.lr}\n")

    # Train
    resume_ckpt = args.resume_from if args.resume_from else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save LoRA adapter
    lora_dir = output_dir / "lora-adapter"
    print(f"\nSaving LoRA adapter to {lora_dir}")
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # Save merged model (full weights)
    merged_dir = output_dir / "merged-model"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  LoRA adapter:  {lora_dir}")
    print(f"  Merged model:  {merged_dir}")
    print(f"\nNext step: Export to GGUF with export_gguf.py")
    print(f"  python export_gguf.py --model-dir {merged_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
