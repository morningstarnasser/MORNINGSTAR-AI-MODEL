#!/usr/bin/env python3
"""
Morningstar-14B-Code — QLoRA Fine-Tuning Script
================================================
Fine-tuned ein Base Model (Qwen2.5-Coder-14B) mit QLoRA auf Coding-Daten.
Laeuft auf Consumer-GPUs mit 16GB+ VRAM.

Usage:
    python train.py
    python train.py --base_model Qwen/Qwen2.5-Coder-14B-Instruct --epochs 3
    python train.py --dataset_path ./data/train.jsonl --resume_from checkpoint-500

Developed by: Morningstar Contributors (github.com/your-org)
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ─── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_OUTPUT_DIR = "./output/morningstar-14b-code"
DEFAULT_DATASET = "./data/train.jsonl"
DEFAULT_EVAL_DATASET = "./data/eval.jsonl"

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LENGTH = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Morningstar QLoRA Training")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--eval_dataset_path", type=str, default=DEFAULT_EVAL_DATASET)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="morningstar-14b-code")
    return parser.parse_args()


def format_chat_messages(example: dict) -> str:
    """Konvertiert eine Conversation in ChatML Format."""
    messages = example.get("messages", [])
    if not messages:
        # Fallback: instruction/output format
        instruction = example.get("instruction", example.get("input", ""))
        output = example.get("output", example.get("response", ""))
        if not instruction or not output:
            return ""
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]

    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted


def main():
    args = parse_args()

    print("=" * 60)
    print("  MORNINGSTAR-14B-CODE — QLoRA Training")
    print("  by Morningstar Contributors (github.com/your-org)")
    print("=" * 60)
    print(f"\n  Base Model:     {args.base_model}")
    print(f"  Dataset:        {args.dataset_path}")
    print(f"  Output:         {args.output_dir}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch Size:     {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation})")
    print(f"  Learning Rate:  {args.learning_rate}")
    print(f"  LoRA:           r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max Seq Length: {args.max_seq_length}")
    print(f"  Device:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    # ─── WandB ────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name="morningstar-14b-qlora")
        os.environ["WANDB_PROJECT"] = args.wandb_project
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # ─── Quantization Config (4-bit NF4) ─────────────────
    print("[1/5] Lade Quantization Config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ─── Tokenizer ────────────────────────────────────────
    print("[2/5] Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ─── Model ────────────────────────────────────────────
    print("[3/5] Lade Base Model (4-bit quantisiert)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ─── LoRA Config ──────────────────────────────────────
    print("[4/5] Konfiguriere LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ─── Dataset ──────────────────────────────────────────
    print("[5/5] Lade Dataset...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    eval_dataset = None
    if Path(args.eval_dataset_path).exists():
        eval_dataset = load_dataset("json", data_files=args.eval_dataset_path, split="train")
        print(f"  Train: {len(dataset):,} samples | Eval: {len(eval_dataset):,} samples")
    else:
        print(f"  Train: {len(dataset):,} samples | Eval: None")

    # ─── Training Args ────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if args.use_wandb else "none",
        seed=42,
    )

    # ─── Trainer ──────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=format_chat_messages,
    )

    # ─── Training ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING GESTARTET")
    print("=" * 60 + "\n")

    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # ─── Speichern ────────────────────────────────────────
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Speichere LoRA Adapter nach: {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Training Info
    info = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "trainable_params": trainable,
        "total_params": total,
        "dataset": args.dataset_path,
    }
    with open(final_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING ABGESCHLOSSEN")
    print(f"  Adapter gespeichert in: {final_dir}")
    print(f"  Naechster Schritt: python merge_and_export.py --adapter_path {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
