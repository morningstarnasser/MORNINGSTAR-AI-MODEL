#!/usr/bin/env python3
"""Prepare math training datasets for fine-tuning Morningstar (Qwen2.5-Coder-14B).

Downloads and processes math datasets from HuggingFace, converts them to ChatML
conversation format, and saves as JSONL files for training.

Usage:
    python prepare_math_dataset.py
    python prepare_math_dataset.py --gsm8k-only --max-samples 1000
    python prepare_math_dataset.py --output-dir /workspace/data

Author: Morningstar Contributors
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Du bist MORNINGSTAR-MATH — ein Elite-Mathematik-Assistent.\n"
    "Entwickelt von Morningstar Contributors.\n\n"
    "REGELN:\n"
    "1. Denke IMMER Schritt fuer Schritt\n"
    "2. Zeige ALLE Rechenwege ausfuehrlich\n"
    "3. Setze die finale Antwort in \\boxed{}\n"
    "4. Pruefe deine Ergebnisse durch Gegenrechnung\n"
    "5. Sei praezise bei Berechnungen — kein Runden ohne Grund\n"
    "6. Bei Unsicherheit: Versuche alternative Loesungswege"
)


# ---------------------------------------------------------------------------
# Solution Cleaning
# ---------------------------------------------------------------------------
def clean_solution(text: str) -> str:
    """Normalize a math solution for training."""
    if not text or not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Normalize LaTeX
    text = text.replace("\\left(", "(").replace("\\right)", ")")
    text = text.replace("\\left[", "[").replace("\\right]", "]")
    text = re.sub(r"\\text\{(\w+)\}", r"\1", text)
    text = re.sub(r"\$\$\s*", "$$", text)
    text = re.sub(r"\s*\$\$", "$$", text)

    # Ensure \boxed{} wrapping of final answer
    if "\\boxed" not in text:
        # Try to find a final answer pattern and wrap it
        patterns = [
            r"(?:the answer is|answer is|answer:)\s*([^\n.]+)",
            r"(?:therefore|thus|so|hence)[,]?\s*([^\n.]+?)(?:\.|$)",
            r"=\s*(\d+[\d,./]*)\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                ans = m.group(1).strip().rstrip(".")
                text = text.rstrip() + f"\n\nThe answer is $\\boxed{{{ans}}}$."
                break

    text = text.strip()
    return text


def extract_gsm8k_answer(solution: str) -> str:
    """Extract the numeric answer from GSM8K #### format."""
    match = re.search(r"####\s*(.+)", solution)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def format_gsm8k_solution(solution: str) -> str:
    """Convert GSM8K solution format to step-by-step with \\boxed{}."""
    answer = extract_gsm8k_answer(solution)
    # Remove the #### answer line
    steps = re.sub(r"####\s*.+", "", solution).strip()

    # Number the steps if not already numbered
    lines = [l.strip() for l in steps.split("\n") if l.strip()]
    formatted_lines = []
    for i, line in enumerate(lines, 1):
        if not re.match(r"^\d+[\.\)]", line) and not line.startswith("Step"):
            line = f"Step {i}: {line}"
        formatted_lines.append(line)

    result = "\n".join(formatted_lines)
    if answer:
        result += f"\n\nThe answer is $\\boxed{{{answer}}}$."
    return result


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------
def load_gsm8k(max_samples: Optional[int] = None) -> list[dict]:
    """Load and process GSM8K dataset."""
    log.info("Loading GSM8K...")
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        log.error(f"Failed to load GSM8K: {e}")
        return []

    examples = []
    for item in tqdm(ds, desc="Processing GSM8K"):
        solution = format_gsm8k_solution(item["answer"])
        solution = clean_solution(solution)
        if not solution:
            continue
        examples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": solution},
            ],
            "source": "gsm8k",
        })

    if max_samples and len(examples) > max_samples:
        examples = random.sample(examples, max_samples)

    log.info(f"GSM8K: {len(examples)} examples")
    return examples


def load_competition_math(max_samples: Optional[int] = None) -> list[dict]:
    """Load and process MATH (competition math) dataset."""
    log.info("Loading MATH (hendrycks/competition_math)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="train")
    except Exception as e:
        log.error(f"Failed to load MATH dataset: {e}")
        return []

    examples = []
    for item in tqdm(ds, desc="Processing MATH"):
        solution = clean_solution(item.get("solution", ""))
        problem = item.get("problem", "")
        if not solution or not problem:
            continue
        examples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "source": "competition_math",
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })

    if max_samples and len(examples) > max_samples:
        examples = random.sample(examples, max_samples)

    log.info(f"MATH: {len(examples)} examples")
    return examples


def load_orca_math(max_samples: int = 20000) -> list[dict]:
    """Load and process Orca-Math dataset (subset)."""
    log.info(f"Loading Orca-Math (subset of {max_samples})...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "microsoft/orca-math-word-problems-200k",
            split="train",
        )
    except Exception as e:
        log.error(f"Failed to load Orca-Math: {e}")
        return []

    # Shuffle and take subset
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:max_samples]

    examples = []
    for idx in tqdm(indices, desc="Processing Orca-Math"):
        item = ds[idx]
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question or not answer:
            continue
        solution = clean_solution(answer)
        examples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution},
            ],
            "source": "orca_math",
        })

    log.info(f"Orca-Math: {len(examples)} examples")
    return examples


def load_math_instruct(max_samples: int = 15000) -> list[dict]:
    """Load and process MathInstruct dataset (subset)."""
    log.info(f"Loading MathInstruct (subset of {max_samples})...")
    try:
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
    except Exception as e:
        log.error(f"Failed to load MathInstruct: {e}")
        return []

    # Shuffle and take subset
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:max_samples]

    examples = []
    for idx in tqdm(indices, desc="Processing MathInstruct"):
        item = ds[idx]
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output:
            continue
        solution = clean_solution(output)
        examples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": solution},
            ],
            "source": "math_instruct",
        })

    log.info(f"MathInstruct: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def save_jsonl(data: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(data)} examples to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare math datasets for Morningstar fine-tuning"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data",
        help="Output directory for JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit total samples (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--gsm8k-only", action="store_true",
        help="Only load GSM8K for quick testing",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MORNINGSTAR-MATH Dataset Preparation")
    print("=" * 60)

    # Load datasets
    all_examples = []
    source_counts = {}

    if args.gsm8k_only:
        gsm8k = load_gsm8k(max_samples=args.max_samples)
        all_examples.extend(gsm8k)
        source_counts["gsm8k"] = len(gsm8k)
    else:
        # Calculate per-source limits if max_samples is set
        per_source = None
        if args.max_samples:
            per_source = args.max_samples // 4

        gsm8k = load_gsm8k(max_samples=per_source)
        all_examples.extend(gsm8k)
        source_counts["gsm8k"] = len(gsm8k)

        math_ds = load_competition_math(max_samples=per_source)
        all_examples.extend(math_ds)
        source_counts["competition_math"] = len(math_ds)

        orca = load_orca_math(max_samples=per_source or 20000)
        all_examples.extend(orca)
        source_counts["orca_math"] = len(orca)

        instruct = load_math_instruct(max_samples=per_source or 15000)
        all_examples.extend(instruct)
        source_counts["math_instruct"] = len(instruct)

    if not all_examples:
        log.error("No examples loaded! Check internet connection and dataset availability.")
        sys.exit(1)

    # Shuffle
    random.shuffle(all_examples)

    # Apply global max_samples limit
    if args.max_samples and len(all_examples) > args.max_samples:
        all_examples = all_examples[:args.max_samples]

    # Split: 90% train, 10% validation
    split_idx = int(len(all_examples) * 0.9)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    # Save
    save_jsonl(train_data, output_dir / "math_train.jsonl")
    save_jsonl(val_data, output_dir / "math_val.jsonl")

    # Stats
    stats = {
        "total_examples": len(all_examples),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "source_counts": source_counts,
        "seed": args.seed,
        "gsm8k_only": args.gsm8k_only,
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("  Dataset Preparation Complete!")
    print("=" * 60)
    print(f"  Total examples:  {stats['total_examples']:,}")
    print(f"  Train split:     {stats['train_examples']:,}")
    print(f"  Validation split:{stats['val_examples']:,}")
    print(f"\n  Sources:")
    for src, count in source_counts.items():
        print(f"    {src:20s} {count:>6,}")
    print(f"\n  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
