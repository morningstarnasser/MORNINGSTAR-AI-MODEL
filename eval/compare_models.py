#!/usr/bin/env python3
"""Multi-model math comparison benchmark.

Runs the same math problems across multiple Ollama models and produces
a side-by-side comparison table. Identifies strengths and weaknesses
of each model per difficulty level and category.

Usage:
    python compare_models.py --models morningstar deepseek-r1:14b phi4:14b
    python compare_models.py --models morningstar qwen2.5-math:7b --levels 6 7
    python compare_models.py --models morningstar deepseek-r1:14b --all-levels --verbose

Author: Morningstar Contributors
"""

import argparse
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Import problems from evaluate_math.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_math import (
    PROBLEMS,
    extract_boxed_answer,
    normalize_answer,
    answers_match,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
SYSTEM_PROMPT = (
    "You are a highly capable math AI. "
    "Follow this protocol:\n"
    "1. UNDERSTAND: Restate the problem. Identify what is given and asked.\n"
    "2. PLAN: Choose the best strategy before computing.\n"
    "3. EXECUTE: Solve step by step, showing ALL work.\n"
    "4. VERIFY: Check your answer with an independent method.\n"
    "5. Put your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
# Ollama Helpers
# ---------------------------------------------------------------------------
def get_available_models() -> list[str]:
    """Get list of models currently available in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def check_model_exists(model: str) -> bool:
    """Check if a model is available in Ollama."""
    available = get_available_models()
    # Check exact match or partial match (e.g. "morningstar" matches "morningstar:latest")
    for m in available:
        if model == m or model == m.split(":")[0]:
            return True
    return False


def query_model(
    question: str,
    model: str,
    timeout: int = 180,
) -> tuple[str, float]:
    """Send a question to a model and return (response, elapsed_seconds)."""
    prompt = f"{SYSTEM_PROMPT}\n\nProblem: {question}"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 2048,
            "num_ctx": 8192,
        },
    }

    try:
        start = time.time()
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        elapsed = time.time() - start
        resp.raise_for_status()
        return resp.json().get("response", ""), elapsed
    except requests.exceptions.Timeout:
        return "[TIMEOUT]", timeout
    except requests.exceptions.ConnectionError:
        return "[CONNECTION_ERROR]", 0.0
    except Exception as e:
        return f"[ERROR: {e}]", 0.0


# ---------------------------------------------------------------------------
# Evaluation per Model
# ---------------------------------------------------------------------------
def evaluate_model(
    model: str,
    problems: list[dict],
    timeout: int,
    verbose: bool,
) -> dict:
    """Run all problems for a single model. Returns results dict."""
    results = []
    correct = 0
    total_time = 0.0

    for i, prob in enumerate(problems, 1):
        q = prob["question"]
        expected = prob["answer"]
        level = prob["level"]
        category = prob["category"]

        print(f"  [{i:2d}/{len(problems)}] L{level} ({category}) ", end="", flush=True)

        response, elapsed = query_model(q, model=model, timeout=timeout)
        total_time += elapsed

        predicted = extract_boxed_answer(response)
        is_correct = answers_match(predicted, expected)
        if is_correct:
            correct += 1

        status = "\033[92mPASS\033[0m" if is_correct else "\033[91mFAIL\033[0m"
        print(f"{status} ({elapsed:.1f}s) pred={predicted!r} exp={expected!r}")

        if verbose and not is_correct and predicted:
            print(f"       Response snippet: {response[:150]}...")

        results.append({
            "question": q,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "level": level,
            "category": category,
            "elapsed": round(elapsed, 2),
        })

    return {
        "model": model,
        "total": len(problems),
        "correct": correct,
        "accuracy": round(correct / len(problems), 4) if problems else 0,
        "avg_time": round(total_time / len(problems), 2) if problems else 0,
        "total_time": round(total_time, 2),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Comparison Tables
# ---------------------------------------------------------------------------
def print_header(text: str, width: int = 78) -> None:
    """Print a centered header."""
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_overall_comparison(all_results: list[dict]) -> None:
    """Print overall accuracy comparison table."""
    print_header("OVERALL COMPARISON")

    # Table header
    name_width = max(len(r["model"]) for r in all_results) + 2
    name_width = max(name_width, 12)

    print(f"\n  {'Model':<{name_width}} {'Correct':>9} {'Accuracy':>10} {'Avg Time':>10} {'Total':>8}")
    print(f"  {'-' * (name_width + 40)}")

    # Sort by accuracy descending
    ranked = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)

    for i, r in enumerate(ranked):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        acc_bar = "█" * int(r["accuracy"] * 20)
        print(
            f"  {r['model']:<{name_width}} "
            f"{r['correct']:>4}/{r['total']:<4} "
            f"{r['accuracy']:>9.1%} "
            f"{r['avg_time']:>9.1f}s "
            f"{r['total_time']:>7.0f}s "
            f" {medal} {acc_bar}"
        )


def print_level_comparison(all_results: list[dict], problems: list[dict]) -> None:
    """Print per-level accuracy comparison."""
    levels = sorted(set(p["level"] for p in problems))
    if len(levels) <= 1:
        return

    print_header("PER-LEVEL COMPARISON")

    name_width = max(len(r["model"]) for r in all_results) + 2
    name_width = max(name_width, 12)

    # Header row
    print(f"\n  {'Model':<{name_width}}", end="")
    for lvl in levels:
        print(f" {'L' + str(lvl):>8}", end="")
    print(f" {'Overall':>9}")

    print(f"  {'-' * (name_width + 9 * len(levels) + 10)}")

    # Each model
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['model']:<{name_width}}", end="")
        for lvl in levels:
            lvl_results = [res for res in r["results"] if res["level"] == lvl]
            lvl_correct = sum(1 for res in lvl_results if res["correct"])
            lvl_total = len(lvl_results)
            if lvl_total > 0:
                acc = lvl_correct / lvl_total
                print(f" {acc:>7.0%}", end="")
            else:
                print(f" {'—':>8}", end="")
        print(f" {r['accuracy']:>8.1%}")


def print_category_comparison(all_results: list[dict], problems: list[dict]) -> None:
    """Print per-category accuracy comparison."""
    categories = sorted(set(p["category"] for p in problems))
    if len(categories) <= 1:
        return

    print_header("PER-CATEGORY COMPARISON")

    name_width = max(len(r["model"]) for r in all_results) + 2
    name_width = max(name_width, 12)
    cat_width = 10

    # Header row
    print(f"\n  {'Model':<{name_width}}", end="")
    for cat in categories:
        label = cat[:cat_width]
        print(f" {label:>{cat_width}}", end="")
    print()

    print(f"  {'-' * (name_width + (cat_width + 1) * len(categories))}")

    # Each model
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['model']:<{name_width}}", end="")
        for cat in categories:
            cat_results = [res for res in r["results"] if res["category"] == cat]
            cat_correct = sum(1 for res in cat_results if res["correct"])
            cat_total = len(cat_results)
            if cat_total > 0:
                acc = cat_correct / cat_total
                print(f" {acc:>{cat_width}.0%}", end="")
            else:
                print(f" {'—':>{cat_width}}", end="")
        print()


def print_head_to_head(all_results: list[dict]) -> None:
    """Print head-to-head: which model wins per problem."""
    if len(all_results) < 2:
        return

    print_header("HEAD-TO-HEAD DETAILS")

    # Find problems where models disagree
    n_problems = len(all_results[0]["results"])
    disagreements = []

    for i in range(n_problems):
        answers = {}
        for r in all_results:
            res = r["results"][i]
            answers[r["model"]] = (res["correct"], res["predicted"])

        # Check if models disagree
        correctness = [v[0] for v in answers.values()]
        if len(set(correctness)) > 1:
            disagreements.append((i, answers, all_results[0]["results"][i]))

    if not disagreements:
        print("\n  All models agree on every problem! No head-to-head differences.")
        return

    print(f"\n  {len(disagreements)} problems where models disagree:\n")

    name_width = max(len(r["model"]) for r in all_results) + 2

    for idx, answers, prob_info in disagreements:
        q = prob_info["question"][:70]
        exp = prob_info["expected"]
        print(f"  Q{idx+1} (L{prob_info['level']}): {q}...")
        print(f"  Expected: {exp}")
        for model_name, (correct, predicted) in answers.items():
            status = "\033[92m✓\033[0m" if correct else "\033[91m✗\033[0m"
            print(f"    {status} {model_name:<{name_width}} → {predicted!r}")
        print()


def print_speed_comparison(all_results: list[dict]) -> None:
    """Print speed comparison."""
    print_header("SPEED COMPARISON")

    name_width = max(len(r["model"]) for r in all_results) + 2
    name_width = max(name_width, 12)

    # Sort by average time
    ranked = sorted(all_results, key=lambda r: r["avg_time"])

    print(f"\n  {'Model':<{name_width}} {'Avg Time':>10} {'Total Time':>12} {'Speed':>12}")
    print(f"  {'-' * (name_width + 36)}")

    fastest = ranked[0]["avg_time"] if ranked else 1

    for r in ranked:
        ratio = r["avg_time"] / fastest if fastest > 0 else 0
        bar = "▓" * max(1, int(ratio * 10))
        print(
            f"  {r['model']:<{name_width}} "
            f"{r['avg_time']:>9.1f}s "
            f"{r['total_time']:>11.0f}s "
            f"{ratio:>8.1f}x    {bar}"
        )


def print_winner_summary(all_results: list[dict]) -> None:
    """Print final winner summary."""
    if not all_results:
        return

    print_header("FINAL RANKING")

    ranked = sorted(all_results, key=lambda r: (-r["accuracy"], r["avg_time"]))

    print()
    for i, r in enumerate(ranked):
        if i == 0:
            badge = "👑 WINNER"
        elif i == 1:
            badge = "🥈 2nd"
        elif i == 2:
            badge = "🥉 3rd"
        else:
            badge = f"   {i+1}th"

        print(f"  {badge}  {r['model']}  —  {r['accuracy']:.1%} accuracy, {r['avg_time']:.1f}s avg")

    # Winner announcement
    winner = ranked[0]
    print(f"\n  ★ {winner['model']} wins with {winner['accuracy']:.1%} accuracy!")

    if len(ranked) >= 2:
        diff = winner["accuracy"] - ranked[1]["accuracy"]
        if diff > 0:
            print(f"    +{diff:.1%} ahead of {ranked[1]['model']}")
        elif diff == 0:
            speed_winner = ranked[0] if ranked[0]["avg_time"] <= ranked[1]["avg_time"] else ranked[1]
            print(f"    Tied with {ranked[1]['model']} — {speed_winner['model']} is faster")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple Ollama models on math problems"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Ollama model names to compare (e.g., morningstar deepseek-r1:14b phi4:14b)",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Difficulty levels to test (1-5 standard, 6=AIME, 7=Olympiad)",
    )
    parser.add_argument(
        "--all-levels", action="store_true",
        help="Test all levels 1-7",
    )
    parser.add_argument(
        "--hard", action="store_true",
        help="Shortcut for --levels 6 7 (only hard problems)",
    )
    parser.add_argument(
        "--output", type=str, default="comparison_results.json",
        help="Output JSON file (default: comparison_results.json)",
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout per problem in seconds (default: 180)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show response snippets for failed problems",
    )
    args = parser.parse_args()

    if args.all_levels:
        args.levels = [1, 2, 3, 4, 5, 6, 7]
    elif args.hard:
        args.levels = [6, 7]

    # Check Ollama
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama at localhost:11434")
        print("Start it with: ollama serve")
        sys.exit(1)

    # Check which models are available
    available = get_available_models()
    print(f"Available Ollama models: {', '.join(available) if available else '(none)'}")

    missing = []
    for model in args.models:
        if not check_model_exists(model):
            missing.append(model)

    if missing:
        print(f"\n⚠  Missing models: {', '.join(missing)}")
        print("Pull them with:")
        for m in missing:
            print(f"  ollama pull {m}")

        # Ask if user wants to continue with available models only
        valid_models = [m for m in args.models if m not in missing]
        if not valid_models:
            print("\nNo models available. Exiting.")
            sys.exit(1)

        print(f"\nContinuing with available models: {', '.join(valid_models)}")
        args.models = valid_models

    # Filter problems
    problems = [p for p in PROBLEMS if p["level"] in args.levels]
    if not problems:
        print(f"No problems found for levels {args.levels}")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print(f"  MORNINGSTAR Model Comparison Benchmark")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Problems: {len(problems)} (Levels: {', '.join(map(str, sorted(args.levels)))})")
    print(f"{'=' * 78}")

    # Run evaluation for each model
    all_results = []
    total_start = time.time()

    for model in args.models:
        print(f"\n{'─' * 78}")
        print(f"  Testing: {model}")
        print(f"{'─' * 78}")

        result = evaluate_model(
            model=model,
            problems=problems,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        all_results.append(result)

        print(f"\n  → {model}: {result['correct']}/{result['total']} "
              f"({result['accuracy']:.1%}), avg {result['avg_time']:.1f}s")

    total_elapsed = time.time() - total_start

    # Print comparison tables
    print_overall_comparison(all_results)
    print_level_comparison(all_results, problems)
    print_category_comparison(all_results, problems)
    print_head_to_head(all_results)
    print_speed_comparison(all_results)
    print_winner_summary(all_results)

    print(f"  Total benchmark time: {total_elapsed:.0f}s")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": args.models,
        "levels": sorted(args.levels),
        "total_problems": len(problems),
        "total_time": round(total_elapsed, 2),
        "results": {r["model"]: r for r in all_results},
        "ranking": [
            {"rank": i + 1, "model": r["model"], "accuracy": r["accuracy"], "avg_time": r["avg_time"]}
            for i, r in enumerate(sorted(all_results, key=lambda x: (-x["accuracy"], x["avg_time"])))
        ],
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
