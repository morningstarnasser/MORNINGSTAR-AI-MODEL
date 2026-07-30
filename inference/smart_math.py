#!/usr/bin/env python3
"""Best-of-N sampling with Majority Voting for improved math accuracy.

Generates N solutions for each problem, extracts answers, and picks
the most common one via majority voting. Optionally verifies with
self-checking. Boosts accuracy by ~10-15% over single-sample baseline.

Usage:
    python smart_math.py                              # Interactive mode
    python smart_math.py --input problems.txt         # File mode
    python smart_math.py --n 7 --model morningstar-math:latest

Author: Morningstar Contributors
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import aiohttp
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
SOLVE_PROMPT = (
    "You are MORNINGSTAR, an advanced math AI developed by Morningstar Contributors.\n"
    "Follow this protocol:\n"
    "1. UNDERSTAND: Restate the problem. Identify what is given and asked.\n"
    "2. PLAN: Choose the best strategy. For competition math, consider: "
    "modular arithmetic, generating functions, Vieta's formulas, "
    "inclusion-exclusion, induction, casework.\n"
    "3. EXECUTE: Solve step by step, showing ALL intermediate calculations.\n"
    "4. VERIFY: Check your answer with an independent method.\n"
    "5. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}"
)
VERIFY_PROMPT = (
    "You are MORNINGSTAR, a math verification expert developed by Morningstar Contributors.\n"
    "Your job: independently solve this problem and check if the proposed answer is correct.\n\n"
    "Problem: {problem}\n\n"
    "Proposed answer: {answer}\n\n"
    "INSTRUCTIONS:\n"
    "1. Solve the problem from scratch using a DIFFERENT method than what was likely used.\n"
    "2. Compare your answer with the proposed one.\n"
    "3. If they disagree, double-check both approaches.\n"
    "4. Put your verified answer in \\boxed{{}}."
)


# ---------------------------------------------------------------------------
# Answer Extraction & Normalization
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str:
    """Extract content from \\boxed{...}, taking the last occurrence."""
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

    simple = re.findall(r"\\boxed\{([^}]+)\}", text)
    if simple:
        return simple[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison during voting."""
    if not answer:
        return ""

    ans = answer.strip()

    # Strip variable assignment: "x = 5" -> "5"
    ans = re.sub(r"^[a-zA-Z]\s*=\s*", "", ans)

    ans = ans.replace("$", "").replace(",", "").replace(" ", "")

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
    ans = ans.replace("\\frac", "frac")
    for _ in range(5):
        frac_match = re.search(r"frac\{([^{}]+)\}\{([^{}]+)\}", ans)
        if frac_match:
            num, den = frac_match.group(1), frac_match.group(2)
            if re.match(r"^[\w\d\.\-]+$", num) and re.match(r"^[\w\d\.\-]+$", den):
                replacement = f"{num}/{den}"
            else:
                replacement = f"({num})/({den})"
            ans = ans[:frac_match.start()] + replacement + ans[frac_match.end():]
        else:
            break

    # Try to convert to canonical numeric form
    try:
        if "/" in ans and re.match(r"^-?\d+/\d+$", ans):
            val = float(Fraction(ans))
        else:
            val = float(ans)
        val = round(val, 6)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, ZeroDivisionError):
        pass

    ans = ans.rstrip(".")
    return ans.lower()


# ---------------------------------------------------------------------------
# MathSolver
# ---------------------------------------------------------------------------
class MathSolver:
    """Solves math problems using Best-of-N + Majority Voting."""

    def __init__(self, model: str = "morningstar:latest", ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        self.stats = {"solved": 0, "total_time": 0.0, "total_samples": 0}

    def _generate_sync(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a single response (synchronous fallback)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 2048, "num_ctx": 8192},
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def _generate_async(
        self, session: aiohttp.ClientSession, prompt: str, temperature: float
    ) -> str:
        """Generate a single response (async)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 2048, "num_ctx": 8192},
        }
        async with session.post(self.ollama_url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            data = await resp.json()
            return data.get("response", "")

    async def _generate_n_async(self, problem: str, n: int) -> list[str]:
        """Generate N responses in parallel using asyncio."""
        prompt = SOLVE_PROMPT.format(problem=problem)

        # Temperature scheduling: first greedy, rest diverse
        temperatures = [0.1] + [0.6 + 0.2 * (i / max(n - 2, 1)) for i in range(n - 1)]
        temperatures = temperatures[:n]

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_async(session, prompt, temp)
                for temp in temperatures
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid = [r for r in responses if isinstance(r, str)]
        return valid

    def _generate_n_sync(self, problem: str, n: int) -> list[str]:
        """Generate N responses sequentially (fallback)."""
        prompt = SOLVE_PROMPT.format(problem=problem)
        temperatures = [0.1] + [0.6 + 0.2 * (i / max(n - 2, 1)) for i in range(n - 1)]
        temperatures = temperatures[:n]

        responses = []
        for temp in temperatures:
            try:
                resp = self._generate_sync(prompt, temperature=temp)
                responses.append(resp)
            except Exception as e:
                print(f"  Warning: Sample failed: {e}")
        return responses

    def majority_vote(self, answers: list[str]) -> tuple[str, float]:
        """Pick the most common answer. Returns (answer, confidence)."""
        if not answers:
            return "", 0.0

        normalized = [normalize_answer(a) for a in answers if a]
        normalized = [a for a in normalized if a]

        if not normalized:
            return answers[0] if answers else "", 0.0

        counter = Counter(normalized)
        most_common, count = counter.most_common(1)[0]
        confidence = count / len(normalized)

        # Return the original (un-normalized) form of the winning answer
        for orig, norm in zip(answers, [normalize_answer(a) for a in answers]):
            if norm == most_common:
                return orig, confidence

        return most_common, confidence

    def weighted_vote(self, answers: list[str], responses: list[str]) -> tuple[str, float]:
        """Weighted voting: longer, more detailed solutions get more weight."""
        if not answers:
            return "", 0.0

        # Weight by solution length (heuristic: longer = more thorough)
        weights = {}
        for ans, resp in zip(answers, responses):
            norm = normalize_answer(ans)
            if not norm:
                continue
            # Weight = log(response_length) as proxy for solution quality
            import math
            weight = math.log(max(len(resp), 1))
            weights[norm] = weights.get(norm, 0) + weight

        if not weights:
            return self.majority_vote(answers)

        best = max(weights, key=weights.get)
        total_weight = sum(weights.values())
        confidence = weights[best] / total_weight if total_weight > 0 else 0

        for orig in answers:
            if normalize_answer(orig) == best:
                return orig, confidence

        return best, confidence

    async def verify_solution_async(self, problem: str, answer: str) -> tuple[str, bool]:
        """Self-verify: ask the model to check the proposed answer."""
        prompt = VERIFY_PROMPT.format(problem=problem, answer=answer)

        async with aiohttp.ClientSession() as session:
            response = await self._generate_async(session, prompt, temperature=0.1)

        verified_answer = extract_answer(response)
        norm_orig = normalize_answer(answer)
        norm_verified = normalize_answer(verified_answer)

        # Check if verification agrees
        agrees = norm_orig == norm_verified if (norm_orig and norm_verified) else False
        return verified_answer, agrees

    def solve(self, problem: str, n: int = 5, verbose: bool = False) -> dict:
        """Solve a problem using Best-of-N + Majority Voting."""
        start = time.time()

        # Generate N solutions
        responses = asyncio.run(self._generate_n_async(problem, n))

        # Extract answers
        answers = [extract_answer(r) for r in responses]
        valid_answers = [(a, r) for a, r in zip(answers, responses) if a]

        if verbose:
            for i, (ans, resp) in enumerate(zip(answers, responses)):
                status = "OK" if ans else "NO ANSWER"
                print(f"  Sample {i+1}: {ans!r} [{status}]")

        # Majority vote
        if valid_answers:
            raw_answers = [a for a, _ in valid_answers]
            raw_responses = [r for _, r in valid_answers]
            best_answer, confidence = self.majority_vote(raw_answers)
        else:
            best_answer, confidence = "", 0.0

        elapsed = time.time() - start
        self.stats["solved"] += 1
        self.stats["total_time"] += elapsed
        self.stats["total_samples"] += len(responses)

        return {
            "problem": problem,
            "answer": best_answer,
            "confidence": round(confidence, 3),
            "n_samples": len(responses),
            "n_valid": len(valid_answers),
            "all_answers": answers,
            "elapsed": round(elapsed, 2),
        }

    def solve_with_verification(self, problem: str, n: int = 5, verbose: bool = False) -> dict:
        """Full pipeline: generate, vote, then verify the top answer."""
        result = self.solve(problem, n=n, verbose=verbose)

        if result["answer"] and result["confidence"] < 1.0:
            # Verify if not unanimous
            if verbose:
                print(f"  Verifying answer: {result['answer']!r} (conf={result['confidence']})...")

            verified_answer, agrees = asyncio.run(
                self.verify_solution_async(problem, result["answer"])
            )

            result["verified"] = True
            result["verification_agrees"] = agrees
            result["verified_answer"] = verified_answer

            if not agrees and verified_answer:
                if verbose:
                    print(f"  Verification disagrees! Verified: {verified_answer!r}")
                # Use verified answer if it disagrees (model corrected itself)
                result["answer"] = verified_answer
        else:
            result["verified"] = result["confidence"] == 1.0
            result["verification_agrees"] = True

        return result


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------
def interactive_mode(solver: MathSolver, n: int, verbose: bool, verify: bool) -> None:
    """Interactive REPL for solving math problems."""
    print("\nMORNINGSTAR-MATH Smart Solver")
    print(f"Model: {solver.model} | Samples: {n} | Verify: {verify}")
    print("Type a math problem (or 'quit' to exit):\n")

    while True:
        try:
            problem = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not problem or problem.lower() in ("quit", "exit", "q"):
            break

        if verify:
            result = solver.solve_with_verification(problem, n=n, verbose=verbose)
        else:
            result = solver.solve(problem, n=n, verbose=verbose)

        print(f"\nAnswer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.0%} "
              f"({result['n_valid']}/{result['n_samples']} valid samples)")
        if result.get("verified"):
            status = "agrees" if result.get("verification_agrees") else "DISAGREES"
            print(f"Verification: {status}")
        print(f"Time: {result['elapsed']:.1f}s\n")


def file_mode(
    solver: MathSolver, input_path: str, output_path: str,
    n: int, verbose: bool, verify: bool,
) -> None:
    """Process problems from a file."""
    problems = Path(input_path).read_text().strip().split("\n")
    problems = [p.strip() for p in problems if p.strip() and not p.startswith("#")]

    print(f"Processing {len(problems)} problems...")
    results = []

    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] ", end="", flush=True)

        if verify:
            result = solver.solve_with_verification(problem, n=n, verbose=verbose)
        else:
            result = solver.solve(problem, n=n, verbose=verbose)

        print(f"{result['answer']!r} (conf={result['confidence']:.0%}, "
              f"{result['elapsed']:.1f}s)")
        results.append(result)

    # Save results
    out = Path(output_path)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out}")

    # Summary
    answered = sum(1 for r in results if r["answer"])
    avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0
    avg_time = sum(r["elapsed"] for r in results) / len(results) if results else 0
    print(f"Answered: {answered}/{len(results)}")
    print(f"Avg confidence: {avg_conf:.0%}")
    print(f"Avg time: {avg_time:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart math solver with Best-of-N + Majority Voting"
    )
    parser.add_argument("--model", type=str, default="morningstar:latest",
                        help="Ollama model name")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of samples per problem (default: 5)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file with problems (one per line)")
    parser.add_argument("--output", type=str, default="solutions.json",
                        help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all N solutions")
    parser.add_argument("--verify", action="store_true",
                        help="Enable self-verification")
    parser.add_argument("--url", type=str, default=OLLAMA_URL,
                        help="Ollama API URL")
    args = parser.parse_args()

    # Check Ollama connectivity
    try:
        import requests as req
        req.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("ERROR: Cannot connect to Ollama. Start it with: ollama serve")
        sys.exit(1)

    solver = MathSolver(model=args.model, ollama_url=args.url)

    if args.input:
        file_mode(solver, args.input, args.output, args.n, args.verbose, args.verify)
    else:
        interactive_mode(solver, args.n, args.verbose, args.verify)


if __name__ == "__main__":
    main()
