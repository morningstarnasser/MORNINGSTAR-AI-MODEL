#!/usr/bin/env python3
"""HTTP API server wrapping the smart math solver.

Provides a REST API for solving math problems using Best-of-N + Majority Voting.

Usage:
    python math_server.py
    python math_server.py --port 9090 --model morningstar-math:latest

Endpoints:
    POST /solve       - Solve a single math problem
    POST /solve/batch - Solve multiple problems
    GET  /health      - Health check
    GET  /stats       - Solver statistics

Author: Morningstar Contributors
"""

import argparse
import os
import time

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from smart_math import MathSolver

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class SolveRequest(BaseModel):
    problem: str = Field(..., description="The math problem to solve")
    n: int = Field(5, ge=1, le=20, description="Number of samples")
    verify: bool = Field(False, description="Enable self-verification")


class SolveResponse(BaseModel):
    answer: str
    confidence: float
    n_samples: int
    n_valid: int
    elapsed: float
    verified: bool = False
    verification_agrees: bool = True
    all_answers: list[str] = []


class BatchRequest(BaseModel):
    problems: list[str] = Field(..., description="List of math problems")
    n: int = Field(5, ge=1, le=20)
    verify: bool = Field(False)


class BatchResponse(BaseModel):
    results: list[SolveResponse]
    total_elapsed: float
    total_problems: int


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_connected: bool


class StatsResponse(BaseModel):
    problems_solved: int
    total_time: float
    total_samples: int
    avg_time_per_problem: float
    avg_samples_per_problem: float


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MORNINGSTAR-MATH API",
    description="Smart math solver with Best-of-N + Majority Voting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

solver: MathSolver = None  # initialized at startup


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check - verifies Ollama connectivity."""
    ollama_ok = False
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        model=solver.model,
        ollama_connected=ollama_ok,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Solver statistics."""
    s = solver.stats
    solved = s["solved"]
    return StatsResponse(
        problems_solved=solved,
        total_time=round(s["total_time"], 2),
        total_samples=s["total_samples"],
        avg_time_per_problem=round(s["total_time"] / solved, 2) if solved else 0,
        avg_samples_per_problem=round(s["total_samples"] / solved, 2) if solved else 0,
    )


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    """Solve a single math problem."""
    if not req.problem.strip():
        raise HTTPException(status_code=400, detail="Empty problem")

    if req.verify:
        result = solver.solve_with_verification(req.problem, n=req.n)
    else:
        result = solver.solve(req.problem, n=req.n)

    return SolveResponse(**result)


@app.post("/solve/batch", response_model=BatchResponse)
async def solve_batch(req: BatchRequest):
    """Solve multiple math problems."""
    if not req.problems:
        raise HTTPException(status_code=400, detail="Empty problem list")
    if len(req.problems) > 100:
        raise HTTPException(status_code=400, detail="Max 100 problems per batch")

    start = time.time()
    results = []
    for problem in req.problems:
        if req.verify:
            result = solver.solve_with_verification(problem, n=req.n)
        else:
            result = solver.solve(problem, n=req.n)
        results.append(SolveResponse(**result))

    return BatchResponse(
        results=results,
        total_elapsed=round(time.time() - start, 2),
        total_problems=len(req.problems),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MORNINGSTAR-MATH API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=str, default="morningstar:latest")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    global solver
    solver = MathSolver(model=args.model)

    print(f"Starting MORNINGSTAR-MATH API on {args.host}:{args.port}")
    print(f"Model: {args.model}")

    uvicorn.run(
        "math_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
