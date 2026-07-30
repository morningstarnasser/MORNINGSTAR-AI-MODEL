#!/usr/bin/env python3
"""
Morningstar-14B-Code — Dataset Preparation
===========================================
Laedt Coding-Datasets von HuggingFace, konvertiert zu ChatML, filtert und merged.

Usage:
    python create_dataset.py
    python create_dataset.py --output_dir ./data --max_total_samples 50000

Developed by: Morningstar Contributors (github.com/your-org)
"""

import json
import hashlib
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ─── Dataset Sources ──────────────────────────────────────────────────────
DATASETS = [
    {
        "name": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "format": "alpaca",  # instruction, input, output
        "max_samples": 20000,
    },
    {
        "name": "TokenBender/code_instructions_122k_alpaca_style",
        "split": "train",
        "format": "alpaca",
        "max_samples": 40000,
    },
    {
        "name": "iamtarun/code_instructions_120k_alpaca",
        "split": "train",
        "format": "alpaca",
        "max_samples": 40000,
    },
]

SYSTEM_PROMPT = (
    "Du bist Morningstar, ein Elite-Coding-Assistent. "
    "Schreibe perfekten, produktionsreifen Code. "
    "Erklaere deine Loesungen klar und praezise."
)

MIN_OUTPUT_LENGTH = 50
MIN_INSTRUCTION_LENGTH = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Morningstar Dataset Preparation")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--max_total_samples", type=int, default=100000)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_system_prompt", action="store_true", default=True)
    return parser.parse_args()


def alpaca_to_chatml(example: dict, add_system: bool = True) -> dict | None:
    """Konvertiert Alpaca-Format zu ChatML messages."""
    instruction = (example.get("instruction", "") or "").strip()
    inp = (example.get("input", "") or "").strip()
    output = (example.get("output", "") or "").strip()

    if not instruction or len(instruction) < MIN_INSTRUCTION_LENGTH:
        return None
    if not output or len(output) < MIN_OUTPUT_LENGTH:
        return None

    # Combine instruction and input
    user_content = instruction
    if inp:
        user_content = f"{instruction}\n\n{inp}"

    messages = []
    if add_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def sharegpt_to_chatml(example: dict, add_system: bool = True) -> dict | None:
    """Konvertiert ShareGPT-Format zu ChatML messages."""
    conversations = example.get("conversations", [])
    if not conversations or len(conversations) < 2:
        return None

    messages = []
    if add_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    for conv in conversations:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        role = role_map.get(conv.get("from", ""), conv.get("role", "user"))
        content = (conv.get("value", "") or conv.get("content", "") or "").strip()
        if content:
            messages.append({"role": role, "content": content})

    # Validate: must have at least user + assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    return {"messages": messages}


def content_hash(messages: list[dict]) -> str:
    """Erstellt Hash fuer Deduplizierung."""
    content = "".join(m.get("content", "") for m in messages)
    return hashlib.md5(content.encode()).hexdigest()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MORNINGSTAR — Dataset Preparation")
    print("=" * 60)

    all_samples = []
    seen_hashes = set()

    for ds_config in DATASETS:
        name = ds_config["name"]
        fmt = ds_config["format"]
        max_n = ds_config["max_samples"]

        print(f"\n  Lade: {name} ...")
        try:
            ds = load_dataset(name, split=ds_config["split"])
        except Exception as e:
            print(f"  FEHLER: {e}")
            continue

        count = 0
        converter = alpaca_to_chatml if fmt == "alpaca" else sharegpt_to_chatml

        for example in tqdm(ds, desc=f"  {name}", leave=False):
            if count >= max_n:
                break

            result = converter(example, add_system=args.add_system_prompt)
            if result is None:
                continue

            h = content_hash(result["messages"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            all_samples.append(result)
            count += 1

        print(f"  ✓ {count:,} Samples von {name}")

    # Shuffle and limit
    import random
    random.seed(args.seed)
    random.shuffle(all_samples)

    if len(all_samples) > args.max_total_samples:
        all_samples = all_samples[:args.max_total_samples]

    # Split train/eval
    eval_size = int(len(all_samples) * args.eval_ratio)
    train_samples = all_samples[eval_size:]
    eval_samples = all_samples[:eval_size]

    # Save as JSONL
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, samples in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n  {'=' * 50}")
    print(f"  Total:     {len(all_samples):,} Samples (dedupliziert)")
    print(f"  Train:     {len(train_samples):,} → {train_path}")
    print(f"  Eval:      {len(eval_samples):,} → {eval_path}")
    print(f"  Duplikate: {len(seen_hashes) - len(all_samples):,} entfernt")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    main()
