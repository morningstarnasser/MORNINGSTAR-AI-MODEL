#!/usr/bin/env python3
"""Baseline math evaluation for the Morningstar model.

Evaluates the model on 45 built-in math problems across 5 difficulty levels
using the Ollama API. Measures accuracy before fine-tuning.

Usage:
    python evaluate_math.py
    python evaluate_math.py --model morningstar-math:latest --verbose
    python evaluate_math.py --levels 1 2 3

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
# Constants
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
SYSTEM_PROMPT = (
    "You are MORNINGSTAR, an advanced math AI developed by Morningstar Contributors. "
    "Follow this protocol:\n"
    "1. UNDERSTAND: Restate the problem. Identify what is given and asked.\n"
    "2. PLAN: Choose the best strategy before computing.\n"
    "3. EXECUTE: Solve step by step, showing ALL work.\n"
    "4. VERIFY: Check your answer with an independent method.\n"
    "5. Put your final answer in \\boxed{}."
)

# ---------------------------------------------------------------------------
# Built-in Test Problems (45 total, 9 per level)
# ---------------------------------------------------------------------------
PROBLEMS = [
    # ===== LEVEL 1: Basic Arithmetic, Fractions, Percentages =====
    {"question": "What is 347 + 685?", "answer": "1032", "level": 1, "category": "arithmetic", "source": "custom"},
    {"question": "What is 23 * 47?", "answer": "1081", "level": 1, "category": "arithmetic", "source": "custom"},
    {"question": "Simplify the fraction 84/126.", "answer": "2/3", "level": 1, "category": "fractions", "source": "custom"},
    {"question": "What is 15% of 240?", "answer": "36", "level": 1, "category": "percentages", "source": "custom"},
    {"question": "A shirt costs $80. It is on sale for 25% off. What is the sale price?", "answer": "60", "level": 1, "category": "percentages", "source": "custom"},
    {"question": "What is 3/4 + 2/5?", "answer": "23/20", "level": 1, "category": "fractions", "source": "custom"},
    {"question": "What is the greatest common divisor of 48 and 180?", "answer": "12", "level": 1, "category": "arithmetic", "source": "custom"},
    {"question": "If you buy 7 items at $13 each, what is the total cost?", "answer": "91", "level": 1, "category": "arithmetic", "source": "custom"},
    {"question": "What is 2.5 * 3.2?", "answer": "8", "level": 1, "category": "arithmetic", "source": "custom"},

    # ===== LEVEL 2: Algebra =====
    {"question": "Solve for x: 3x + 7 = 22", "answer": "5", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "Solve for x: x^2 - 5x + 6 = 0. Give the larger root.", "answer": "3", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "If f(x) = 2x^2 - 3x + 1, what is f(4)?", "answer": "21", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "Solve the system: x + y = 10, 2x - y = 5. What is x?", "answer": "5", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "Simplify: (x^2 - 9) / (x - 3)", "answer": "x+3", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "What is the sum of the roots of x^2 - 7x + 12 = 0?", "answer": "7", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "If 2^x = 64, what is x?", "answer": "6", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "What is the slope of the line passing through (2, 3) and (6, 11)?", "answer": "2", "level": 2, "category": "algebra", "source": "custom"},
    {"question": "Solve for x: |2x - 3| = 7. Give the positive solution.", "answer": "5", "level": 2, "category": "algebra", "source": "custom"},

    # ===== LEVEL 3: Geometry, Trigonometry =====
    {"question": "What is the area of a triangle with base 12 and height 8?", "answer": "48", "level": 3, "category": "geometry", "source": "custom"},
    {"question": "A circle has radius 7. What is its area? Express in terms of pi.", "answer": "49pi", "level": 3, "category": "geometry", "source": "custom"},
    {"question": "In a right triangle with legs 5 and 12, what is the hypotenuse?", "answer": "13", "level": 3, "category": "geometry", "source": "custom"},
    {"question": "What is sin(30 degrees)?", "answer": "1/2", "level": 3, "category": "trigonometry", "source": "custom"},
    {"question": "What is the sum of interior angles of a hexagon in degrees?", "answer": "720", "level": 3, "category": "geometry", "source": "custom"},
    {"question": "A cylinder has radius 3 and height 10. What is its volume? Express in terms of pi.", "answer": "90pi", "level": 3, "category": "geometry", "source": "custom"},
    {"question": "Find the distance between points (1, 2) and (4, 6).", "answer": "5", "level": 3, "category": "geometry", "source": "custom"},
    {"question": "What is cos(60 degrees)?", "answer": "1/2", "level": 3, "category": "trigonometry", "source": "custom"},
    {"question": "What is the perimeter of a regular pentagon with side length 8?", "answer": "40", "level": 3, "category": "geometry", "source": "custom"},

    # ===== LEVEL 4: Calculus, Series, Probability =====
    {"question": "What is the derivative of f(x) = 3x^4 - 2x^2 + 5x?", "answer": "12x^3-4x+5", "level": 4, "category": "calculus", "source": "custom"},
    {"question": "Evaluate the integral of 2x dx from 0 to 3.", "answer": "9", "level": 4, "category": "calculus", "source": "custom"},
    {"question": "What is the sum of the geometric series 1 + 1/2 + 1/4 + 1/8 + ... (infinite)?", "answer": "2", "level": 4, "category": "series", "source": "custom"},
    {"question": "What is the probability of rolling a sum of 7 with two fair dice?", "answer": "1/6", "level": 4, "category": "probability", "source": "custom"},
    {"question": "Find the limit as x approaches 0 of sin(x)/x.", "answer": "1", "level": 4, "category": "calculus", "source": "custom"},
    {"question": "How many ways can you choose 3 items from 10? (10 choose 3)", "answer": "120", "level": 4, "category": "probability", "source": "custom"},
    {"question": "What is the sum of the first 100 positive integers?", "answer": "5050", "level": 4, "category": "series", "source": "custom"},
    {"question": "What is the derivative of ln(x^2 + 1)?", "answer": "2x/(x^2+1)", "level": 4, "category": "calculus", "source": "custom"},
    {"question": "A fair coin is flipped 5 times. What is the probability of getting exactly 3 heads?", "answer": "5/16", "level": 4, "category": "probability", "source": "custom"},

    # ===== LEVEL 5: Competition-Level (AMC/AIME style) =====
    {"question": "How many positive integers less than 1000 are divisible by 3 but not by 5?", "answer": "267", "level": 5, "category": "number_theory", "source": "AMC-style"},
    {"question": "Find the remainder when 2^100 is divided by 7.", "answer": "2", "level": 5, "category": "number_theory", "source": "AMC-style"},
    {"question": "In how many ways can 8 people sit around a circular table?", "answer": "5040", "level": 5, "category": "combinatorics", "source": "AMC-style"},
    {"question": "If a + b = 5 and a^2 + b^2 = 13, what is ab?", "answer": "6", "level": 5, "category": "algebra", "source": "AMC-style"},
    {"question": "How many distinct prime factors does 2310 have?", "answer": "5", "level": 5, "category": "number_theory", "source": "AMC-style"},
    {"question": "What is the value of floor(sqrt(2023))?", "answer": "44", "level": 5, "category": "number_theory", "source": "AMC-style"},
    {"question": "A 3x3 magic square uses integers 1-9. What is the magic constant (row sum)?", "answer": "15", "level": 5, "category": "combinatorics", "source": "AMC-style"},
    {"question": "Find the sum: 1/1*2 + 1/2*3 + 1/3*4 + ... + 1/99*100", "answer": "99/100", "level": 5, "category": "series", "source": "AIME-style"},
    {"question": "How many 4-digit palindromes are there? (e.g., 1221, 3443)", "answer": "90", "level": 5, "category": "combinatorics", "source": "AMC-style"},

    # ===== LEVEL 6: AIME / Hard Competition =====
    {"question": "Find the number of positive integers n <= 1000 such that n^2 + n is divisible by 6.", "answer": "667", "level": 6, "category": "number_theory", "source": "AIME-style"},
    {"question": "How many ordered triples (a, b, c) of positive integers satisfy a + b + c = 20?", "answer": "171", "level": 6, "category": "combinatorics", "source": "AIME-style"},
    {"question": "Find the last three digits of 7^999.", "answer": "343", "level": 6, "category": "number_theory", "source": "AIME-style"},
    {"question": "The polynomial x^3 - 3x^2 + 4 has roots r, s, t. Find r^2 + s^2 + t^2.", "answer": "1", "level": 6, "category": "algebra", "source": "AIME-style"},
    {"question": "How many integers between 1 and 10000 inclusive have a digit sum equal to 9?", "answer": "220", "level": 6, "category": "combinatorics", "source": "AIME-style"},
    {"question": "Find the sum of all positive integers n such that n^2 + 2n + 2 is divisible by n + 1.", "answer": "1", "level": 6, "category": "number_theory", "source": "AIME-style"},
    {"question": "Let f(x) = x^3 - 6x^2 + 11x - 6. Find the product of all real roots of f(f(x)) = 0.", "answer": "720", "level": 6, "category": "algebra", "source": "AIME-style"},
    {"question": "A lattice point is a point (x,y) where both x and y are integers. How many lattice points are inside the circle x^2 + y^2 < 50?", "answer": "149", "level": 6, "category": "geometry", "source": "AIME-style"},
    {"question": "Find the number of subsets of {1, 2, 3, ..., 10} that contain no two consecutive integers.", "answer": "144", "level": 6, "category": "combinatorics", "source": "AIME-style"},

    # ===== LEVEL 7: Olympiad / Very Hard =====
    {"question": "Find the smallest positive integer n such that n! is divisible by 10^10.", "answer": "45", "level": 7, "category": "number_theory", "source": "olympiad-style"},
    {"question": "How many 6-digit numbers have all digits non-decreasing (e.g. 112359)?", "answer": "3003", "level": 7, "category": "combinatorics", "source": "olympiad-style"},
    {"question": "Find the sum of the infinite series 1/1^3 + 1/2^3 + 1/3^3 + ... rounded to three decimal places.", "answer": "1.202", "level": 7, "category": "series", "source": "olympiad-style"},
    {"question": "Find the number of integer solutions to x^2 + y^2 + z^2 = 2023.", "answer": "0", "level": 7, "category": "number_theory", "source": "olympiad-style"},
    {"question": "The number 2^29 has exactly d digits. Find d.", "answer": "9", "level": 7, "category": "number_theory", "source": "olympiad-style"},
    {"question": "Find the remainder when 1! + 2! + 3! + ... + 100! is divided by 15.", "answer": "3", "level": 7, "category": "number_theory", "source": "olympiad-style"},
    {"question": "How many positive integers less than 10^6 are perfect squares or perfect cubes?", "answer": "1090", "level": 7, "category": "number_theory", "source": "olympiad-style"},
    {"question": "Find the coefficient of x^5 in the expansion of (1 + x + x^2)^6.", "answer": "246", "level": 7, "category": "combinatorics", "source": "olympiad-style"},
    {"question": "In how many ways can you tile a 2x10 grid with 1x2 dominoes?", "answer": "89", "level": 7, "category": "combinatorics", "source": "olympiad-style"},
]


# ---------------------------------------------------------------------------
# Answer Extraction & Comparison
# ---------------------------------------------------------------------------
def extract_boxed_answer(text: str) -> str:
    """Extract content from \\boxed{...}, taking the last occurrence."""
    # Handle nested braces
    matches = []
    pattern = r"\\boxed\{"
    for m in re.finditer(pattern, text):
        start = m.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            matches.append(text[start:pos - 1])

    if matches:
        return matches[-1].strip()

    # Fallback: try simple regex
    simple = re.findall(r"\\boxed\{([^}]+)\}", text)
    if simple:
        return simple[-1].strip()

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if not answer:
        return ""

    ans = answer.strip()

    # Strip variable assignment: "x = 5" -> "5", "y=3/4" -> "3/4"
    ans = re.sub(r"^[a-zA-Z]\s*=\s*", "", ans)

    # Remove dollar signs, whitespace, commas
    ans = ans.replace("$", "").replace(" ", "").replace(",", "")

    # Remove LaTeX formatting commands
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\displaystyle", "")
    ans = ans.replace("\\pi", "pi")
    ans = ans.replace("\\cdot", "*")
    ans = ans.replace("\\times", "*")
    ans = re.sub(r"\\text\{([^}]+)\}", r"\1", ans)
    ans = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", ans)

    # Handle \sqrt{x} -> sqrt(x)
    ans = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", ans)

    # Handle general LaTeX fractions: \frac{...}{...} -> (...)/(...)
    # Do this repeatedly for nested fractions
    ans = ans.replace("\\frac", "frac")
    for _ in range(5):  # handle up to 5 nested fractions
        frac_match = re.search(r"frac\{([^{}]+)\}\{([^{}]+)\}", ans)
        if frac_match:
            num, den = frac_match.group(1), frac_match.group(2)
            # Simple cases: single number or variable -> no parens needed
            if re.match(r"^[\w\d\.\-]+$", num) and re.match(r"^[\w\d\.\-]+$", den):
                replacement = f"{num}/{den}"
            else:
                replacement = f"({num})/({den})"
            ans = ans[:frac_match.start()] + replacement + ans[frac_match.end():]
        else:
            break

    # Remove trailing period
    ans = ans.rstrip(".")

    return ans.lower()


def answers_match(predicted: str, expected: str) -> bool:
    """Compare predicted and expected answers with tolerance."""
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)

    if not pred or not exp:
        return False

    # Direct string match
    if pred == exp:
        return True

    # Try numeric comparison
    try:
        pred_val = float(Fraction(pred)) if "/" in pred else float(pred)
        exp_val = float(Fraction(exp)) if "/" in exp else float(exp)
        if abs(pred_val - exp_val) < 1e-6:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # Try fraction comparison
    try:
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

    # Try stripping variable assignment from predicted too (in case normalize missed it)
    pred_stripped = re.sub(r"^[a-z]\s*=\s*", "", pred)
    if pred_stripped == exp:
        return True

    # Compare as expressions: sort terms for commutative comparison
    # e.g. "4x+5" == "5+4x" — simple heuristic via sorted chars
    if set(pred_clean) == set(exp_clean) and sorted(pred_clean) == sorted(exp_clean):
        return True

    return False


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------
def query_ollama(
    question: str,
    model: str = "morningstar:latest",
    timeout: int = 120,
    max_retries: int = 2,
) -> tuple[str, float]:
    """Send a question to Ollama and return (response, elapsed_seconds)."""
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

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            elapsed = time.time() - start
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", ""), elapsed
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"  Timeout, retrying ({attempt + 1}/{max_retries})...")
                continue
            return "[TIMEOUT]", timeout
        except requests.exceptions.ConnectionError:
            print("ERROR: Cannot connect to Ollama. Is it running?")
            print("Start it with: ollama serve")
            sys.exit(1)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return f"[ERROR: {e}]", 0.0

    return "[ERROR]", 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: str,
    levels: list[int],
    timeout: int,
    verbose: bool,
) -> dict:
    """Run evaluation and return results dict."""
    problems = [p for p in PROBLEMS if p["level"] in levels]
    print(f"\nEvaluating {len(problems)} problems on '{model}'...\n")

    results = []
    correct = 0
    total_time = 0.0

    for i, prob in enumerate(problems, 1):
        q = prob["question"]
        expected = prob["answer"]
        level = prob["level"]
        category = prob["category"]

        print(f"[{i:2d}/{len(problems)}] L{level} ({category}) ", end="", flush=True)

        response, elapsed = query_ollama(q, model=model, timeout=timeout)
        total_time += elapsed

        predicted = extract_boxed_answer(response)
        is_correct = answers_match(predicted, expected)
        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"{status} ({elapsed:.1f}s) pred={predicted!r} exp={expected!r}")

        if verbose and not is_correct:
            print(f"  Question: {q}")
            print(f"  Response: {response[:200]}...")
            print()

        results.append({
            "question": q,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "level": level,
            "category": category,
            "elapsed": round(elapsed, 2),
            "response": response if verbose else response[:500],
        })

    # Compute stats
    accuracy = correct / len(problems) if problems else 0
    avg_time = total_time / len(problems) if problems else 0

    # Per-level accuracy
    level_stats = {}
    for lvl in sorted(set(p["level"] for p in problems)):
        lvl_results = [r for r in results if r["level"] == lvl]
        lvl_correct = sum(1 for r in lvl_results if r["correct"])
        level_stats[f"level_{lvl}"] = {
            "total": len(lvl_results),
            "correct": lvl_correct,
            "accuracy": round(lvl_correct / len(lvl_results), 4) if lvl_results else 0,
        }

    # Per-category accuracy
    category_stats = {}
    for cat in sorted(set(p["category"] for p in problems)):
        cat_results = [r for r in results if r["category"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"])
        category_stats[cat] = {
            "total": len(cat_results),
            "correct": cat_correct,
            "accuracy": round(cat_correct / len(cat_results), 4) if cat_results else 0,
        }

    summary = {
        "model": model,
        "total_problems": len(problems),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_time_seconds": round(avg_time, 2),
        "total_time_seconds": round(total_time, 2),
        "level_stats": level_stats,
        "category_stats": category_stats,
        "results": results,
    }

    return summary


def print_results(summary: dict) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 60)
    print("  MORNINGSTAR Math Evaluation Results")
    print("=" * 60)
    print(f"  Model:       {summary['model']}")
    print(f"  Overall:     {summary['correct']}/{summary['total_problems']} "
          f"({summary['accuracy']:.1%})")
    print(f"  Avg time:    {summary['avg_time_seconds']:.1f}s per problem")
    print(f"  Total time:  {summary['total_time_seconds']:.0f}s")

    print("\n  Per Level:")
    print(f"  {'Level':<10} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*36}")
    for lvl_name, stats in sorted(summary["level_stats"].items()):
        print(f"  {lvl_name:<10} {stats['correct']:>8} {stats['total']:>6} "
              f"{stats['accuracy']:>10.1%}")

    print("\n  Per Category:")
    print(f"  {'Category':<16} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*42}")
    for cat, stats in sorted(summary["category_stats"].items()):
        print(f"  {cat:<16} {stats['correct']:>8} {stats['total']:>6} "
              f"{stats['accuracy']:>10.1%}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Morningstar model on math problems"
    )
    parser.add_argument(
        "--model", type=str, default="morningstar:latest",
        help="Ollama model name (default: morningstar:latest)",
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Output file for detailed results (default: eval_results.json)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per problem in seconds (default: 120)",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Which difficulty levels to evaluate (1-5 standard, 6=AIME, 7=Olympiad)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full responses for failed problems",
    )
    args = parser.parse_args()

    # Check Ollama connectivity
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama at localhost:11434")
        print("Start it with: ollama serve")
        sys.exit(1)

    summary = evaluate(
        model=args.model,
        levels=args.levels,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    print_results(summary)

    # Save detailed results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
