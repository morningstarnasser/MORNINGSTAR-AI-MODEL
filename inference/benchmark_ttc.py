#!/usr/bin/env python3
"""Benchmark: Baseline vs Test-Time Compute (TTC) strategies.

Compares single-sample baseline against Best-of-N and Best-of-N + Verify
to measure the improvement from TTC techniques.

Usage:
    python benchmark_ttc.py
    python benchmark_ttc.py --model morningstar-math:latest --quick
    python benchmark_ttc.py --problems custom_problems.json

Author: Morningstar Contributors
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Import from sibling modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from smart_math import MathSolver, extract_answer, normalize_answer

# ---------------------------------------------------------------------------
# Built-in Test Problems (subset for benchmarking)
# ---------------------------------------------------------------------------
BUILTIN_PROBLEMS = [
    # Level 1
    {"question": "What is 347 + 685?", "answer": "1032", "level": 1},
    {"question": "What is 23 * 47?", "answer": "1081", "level": 1},
    {"question": "What is 15% of 240?", "answer": "36", "level": 1},
    # Level 2
    {"question": "Solve for x: 3x + 7 = 22", "answer": "5", "level": 2},
    {"question": "If f(x) = 2x^2 - 3x + 1, what is f(4)?", "answer": "21", "level": 2},
    {"question": "If 2^x = 64, what is x?", "answer": "6", "level": 2},
    # Level 3
    {"question": "In a right triangle with legs 5 and 12, what is the hypotenuse?", "answer": "13", "level": 3},
    {"question": "What is sin(30 degrees)?", "answer": "1/2", "level": 3},
    {"question": "Find the distance between points (1, 2) and (4, 6).", "answer": "5", "level": 3},
    # Level 4
    {"question": "Evaluate the integral of 2x dx from 0 to 3.", "answer": "9", "level": 4},
    {"question": "What is the probability of rolling a sum of 7 with two fair dice?", "answer": "1/6", "level": 4},
    {"question": "What is the sum of the first 100 positive integers?", "answer": "5050", "level": 4},
    # Level 5
    {"question": "Find the remainder when 2^100 is divided by 7.", "answer": "2", "level": 5},
    {"question": "If a + b = 5 and a^2 + b^2 = 13, what is ab?", "answer": "6", "level": 5},
    {"question": "How many 4-digit palindromes are there?", "answer": "90", "level": 5},
]

FULL_PROBLEMS = BUILTIN_PROBLEMS + [
    # Additional problems for full benchmark
    {"question": "Simplify the fraction 84/126.", "answer": "2/3", "level": 1},
    {"question": "What is 3/4 + 2/5?", "answer": "23/20", "level": 1},
    {"question": "Solve for x: x^2 - 5x + 6 = 0. Give the larger root.", "answer": "3", "level": 2},
    {"question": "What is the sum of the roots of x^2 - 7x + 12 = 0?", "answer": "7", "level": 2},
    {"question": "What is the area of a triangle with base 12 and height 8?", "answer": "48", "level": 3},
    {"question": "What is the sum of interior angles of a hexagon in degrees?", "answer": "720", "level": 3},
    {"question": "What is the derivative of f(x) = 3x^4 - 2x^2 + 5x?", "answer": "12x^3-4x+5", "level": 4},
    {"question": "How many ways can you choose 3 items from 10?", "answer": "120", "level": 4},
    {"question": "How many positive integers less than 1000 are divisible by 3 but not by 5?", "answer": "267", "level": 5},
    {"question": "How many distinct prime factors does 2310 have?", "answer": "5", "level": 5},
    # Level 6: AIME-style (hard)
    {"question": "How many ordered triples (a, b, c) of positive integers satisfy a + b + c = 20?", "answer": "171", "level": 6},
    {"question": "Find the last three digits of 7^999.", "answer": "343", "level": 6},
    {"question": "Find the number of subsets of {1, 2, 3, ..., 10} that contain no two consecutive integers.", "answer": "144", "level": 6},
    # Level 7: Olympiad-style (very hard)
    {"question": "Find the remainder when 1! + 2! + 3! + ... + 100! is divided by 15.", "answer": "3", "level": 7},
    {"question": "In how many ways can you tile a 2x10 grid with 1x2 dominoes?", "answer": "89", "level": 7},
]


# ---------------------------------------------------------------------------
# Baseline (single greedy sample)
# ---------------------------------------------------------------------------
def baseline_solve(problem: str, model: str) -> tuple[str, float]:
    """Single greedy sample (temperature=0)."""
    payload = {
        "model": model,
        "prompt": (
            "You are MORNINGSTAR, an advanced math AI developed by Morningstar Contributors. "
            "Follow this protocol:\n"
            "1. UNDERSTAND: Restate the problem. Identify what is given and asked.\n"
            "2. PLAN: Choose the best strategy before computing.\n"
            "3. EXECUTE: Solve step by step, showing ALL work.\n"
            "4. VERIFY: Check your answer with an independent method.\n"
            f"5. Put your final answer in \\boxed{{}}\n\nProblem: {problem}"
        ),
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048, "num_ctx": 8192},
    }
    start = time.time()
    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        response = resp.json().get("response", "")
    except Exception as e:
        response = f"[ERROR: {e}]"
    elapsed = time.time() - start

    answer = extract_answer(response)
    return answer, elapsed


def answers_match(predicted: str, expected: str) -> bool:
    """Check if answers match with robust comparison."""
    import re
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)
    if not pred or not exp:
        return False

    # Direct string match
    if pred == exp:
        return True

    # Try numeric comparison
    try:
        from fractions import Fraction
        pred_val = float(Fraction(pred)) if re.match(r"^-?\d+/\d+$", pred) else float(pred)
        exp_val = float(Fraction(exp)) if re.match(r"^-?\d+/\d+$", exp) else float(exp)
        if abs(pred_val - exp_val) < 1e-6:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # Try fraction comparison
    try:
        from fractions import Fraction
        pred_frac = Fraction(pred).limit_denominator(10000)
        exp_frac = Fraction(exp).limit_denominator(10000)
        if pred_frac == exp_frac:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # Normalize expression comparison (remove parens, spaces, braces)
    pred_clean = re.sub(r"[(){}\[\] ]", "", pred)
    exp_clean = re.sub(r"[(){}\[\] ]", "", exp)
    if pred_clean == exp_clean:
        return True

    # Strip variable assignment from predicted
    pred_stripped = re.sub(r"^[a-z]\s*=\s*", "", pred)
    if pred_stripped == exp:
        return True

    return False


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------
def run_benchmark(
    problems: list[dict],
    model: str,
    n_samples: int = 5,
) -> dict:
    """Run all three modes and compare."""
    solver = MathSolver(model=model)

    modes = {
        "baseline": {"correct": 0, "times": [], "results": []},
        "best_of_n": {"correct": 0, "times": [], "results": []},
        "best_of_n_verify": {"correct": 0, "times": [], "results": []},
    }

    total = len(problems)

    for i, prob in enumerate(problems, 1):
        q = prob["question"]
        expected = prob["answer"]
        level = prob.get("level", "?")

        print(f"\n[{i}/{total}] L{level}: {q[:60]}...")

        # Mode 1: Baseline
        pred, elapsed = baseline_solve(q, model)
        correct = answers_match(pred, expected)
        modes["baseline"]["correct"] += int(correct)
        modes["baseline"]["times"].append(elapsed)
        modes["baseline"]["results"].append({
            "question": q, "expected": expected, "predicted": pred,
            "correct": correct, "elapsed": elapsed, "level": level,
        })
        status = "PASS" if correct else "FAIL"
        print(f"  Baseline:     {status} pred={pred!r} ({elapsed:.1f}s)")

        # Mode 2: Best-of-N
        result = solver.solve(q, n=n_samples)
        pred = result["answer"]
        elapsed = result["elapsed"]
        correct = answers_match(pred, expected)
        modes["best_of_n"]["correct"] += int(correct)
        modes["best_of_n"]["times"].append(elapsed)
        modes["best_of_n"]["results"].append({
            "question": q, "expected": expected, "predicted": pred,
            "correct": correct, "elapsed": elapsed, "level": level,
            "confidence": result["confidence"],
        })
        status = "PASS" if correct else "FAIL"
        print(f"  Best-of-{n_samples}:    {status} pred={pred!r} "
              f"(conf={result['confidence']:.0%}, {elapsed:.1f}s)")

        # Mode 3: Best-of-N + Verify
        result = solver.solve_with_verification(q, n=n_samples)
        pred = result["answer"]
        elapsed = result["elapsed"]
        correct = answers_match(pred, expected)
        modes["best_of_n_verify"]["correct"] += int(correct)
        modes["best_of_n_verify"]["times"].append(elapsed)
        modes["best_of_n_verify"]["results"].append({
            "question": q, "expected": expected, "predicted": pred,
            "correct": correct, "elapsed": elapsed, "level": level,
            "confidence": result["confidence"],
            "verified": result.get("verification_agrees", True),
        })
        status = "PASS" if correct else "FAIL"
        print(f"  BoN+Verify:   {status} pred={pred!r} ({elapsed:.1f}s)")

    return modes


def print_comparison(modes: dict, problems: list[dict]) -> None:
    """Print formatted comparison table."""
    total = len(problems)

    print("\n" + "=" * 70)
    print("  MORNINGSTAR-MATH: TTC Benchmark Results")
    print("=" * 70)

    print(f"\n  {'Mode':<25} {'Correct':>8} {'Accuracy':>10} {'Avg Time':>10}")
    print(f"  {'-'*55}")

    for mode_name, data in modes.items():
        correct = data["correct"]
        accuracy = correct / total if total else 0
        avg_time = sum(data["times"]) / len(data["times"]) if data["times"] else 0
        label = mode_name.replace("_", " ").title()
        print(f"  {label:<25} {correct:>5}/{total:<3} {accuracy:>9.1%} {avg_time:>9.1f}s")

    # Improvement
    baseline_acc = modes["baseline"]["correct"] / total if total else 0
    bon_acc = modes["best_of_n"]["correct"] / total if total else 0
    verify_acc = modes["best_of_n_verify"]["correct"] / total if total else 0

    print(f"\n  Improvement from TTC:")
    print(f"    Best-of-N:          +{(bon_acc - baseline_acc)*100:.1f} percentage points")
    print(f"    Best-of-N + Verify: +{(verify_acc - baseline_acc)*100:.1f} percentage points")

    # Per-level breakdown
    levels = sorted(set(p.get("level", 0) for p in problems))
    if len(levels) > 1:
        print(f"\n  Per-Level Accuracy:")
        print(f"  {'Level':<8}", end="")
        for mode_name in modes:
            label = mode_name.replace("_", " ").title()[:12]
            print(f" {label:>12}", end="")
        print()
        print(f"  {'-'*50}")

        for lvl in levels:
            print(f"  L{lvl:<6}", end="")
            for mode_name, data in modes.items():
                lvl_results = [r for r in data["results"] if r.get("level") == lvl]
                lvl_correct = sum(1 for r in lvl_results if r["correct"])
                lvl_total = len(lvl_results)
                acc = lvl_correct / lvl_total if lvl_total else 0
                print(f" {acc:>11.0%}", end="")
            print()

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline vs TTC strategies"
    )
    parser.add_argument("--model", type=str, default="morningstar:latest")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--problems", type=str, default="builtin",
                        help="Path to problems JSON file, or 'builtin'")
    parser.add_argument("--quick", action="store_true",
                        help="Use 15-problem subset for fast testing")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of samples for Best-of-N (default: 5)")
    args = parser.parse_args()

    # Check Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    # Load problems
    if args.problems == "builtin":
        problems = BUILTIN_PROBLEMS if args.quick else FULL_PROBLEMS
    else:
        with open(args.problems) as f:
            problems = json.load(f)

    print(f"Benchmarking {len(problems)} problems on '{args.model}'")
    print(f"Modes: Baseline | Best-of-{args.n} | Best-of-{args.n} + Verify")

    # Run benchmark
    modes = run_benchmark(problems, model=args.model, n_samples=args.n)

    # Print results
    print_comparison(modes, problems)

    # Save detailed results
    output = {
        "model": args.model,
        "n_samples": args.n,
        "total_problems": len(problems),
        "modes": {
            name: {
                "correct": data["correct"],
                "accuracy": round(data["correct"] / len(problems), 4),
                "avg_time": round(sum(data["times"]) / len(data["times"]), 2) if data["times"] else 0,
                "results": data["results"],
            }
            for name, data in modes.items()
        },
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
