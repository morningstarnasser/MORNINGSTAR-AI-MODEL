#!/usr/bin/env python3
"""Evaluate HydraCH-Bench against an OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morningstar_hydra.hydrach_bench import grade_response as grade_response_v1
from morningstar_hydra.hydrach_bench_v2 import grade_response as grade_response_v2

DEFAULT_CASES = ROOT / "benchmarks" / "hydrach" / "v1" / "dev.jsonl"
DEFAULT_MANIFEST = ROOT / "benchmarks" / "hydrach" / "v1" / "manifest.json"
REPORT_VERSION = 5
GRADERS_BY_BENCHMARK: dict[str, Callable[..., dict[str, Any]]] = {
    "hydrach-v1": grade_response_v1,
    "hydrach-v2": grade_response_v2,
}
# v2 grades tool cases as three separate signals instead of one boolean.
TOOL_SUBSCORES = ("tool_function_ok", "tool_args_schema_ok", "tool_args_exact_ok")
# A paired comparison needs both candidates' answers per case, not just the rate.
PAIRED_BENCHMARKS = frozenset({"hydrach-v2"})
TOKEN_BUDGET_POLICY_VERSION = "hydrach-v1-category-budget-v1"
SHORT_TOKEN_CATEGORIES = frozenset(
    {"canton_code", "canton_name", "date_iso", "chf_arithmetic", "privacy_policy"}
)
STRUCTURED_TOKEN_CATEGORIES = frozenset({"invoice_json", "multilingual_json", "tool_call"})


def category_token_budgets(
    short_max_tokens: int = 64,
    structured_max_tokens: int = 512,
) -> dict[str, int]:
    """Return the explicit per-category output budget for comparable HydraCH-Bench v1 runs."""
    if short_max_tokens <= 0:
        raise ValueError("short max tokens must be positive")
    if structured_max_tokens < 512:
        raise ValueError("structured max tokens must be at least 512")
    return {
        **{category: short_max_tokens for category in SHORT_TOKEN_CATEGORIES},
        **{category: structured_max_tokens for category in STRUCTURED_TOKEN_CATEGORIES},
    }


def token_budget_for_case(case: dict[str, Any], budgets: dict[str, int]) -> int:
    category = case.get("category")
    if category not in budgets:
        raise ValueError(f"unknown benchmark category for token budget: {category!r}")
    return int(budgets[category])


def pace_before_case(
    pending_index: int,
    pace_seconds: float,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Pause between requests in one evaluator process, never before its first request."""
    if pace_seconds < 0:
        raise ValueError("pace seconds must be non-negative")
    if pending_index > 0 and pace_seconds > 0:
        sleep_fn(pace_seconds)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not an object")
        required = {"id", "benchmark", "category", "language", "messages", "grader", "visibility"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"line {line_number} is missing fields: {sorted(missing)}")
        case_id = value["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError(f"line {line_number} has an invalid or duplicate case id")
        if value["visibility"] not in {"dev", "hidden"}:
            raise ValueError(f"line {line_number} has invalid visibility")
        if not isinstance(value["messages"], list) or not value["messages"]:
            raise ValueError(f"line {line_number} has no messages")
        if not isinstance(value["grader"], dict) or not isinstance(value["grader"].get("type"), str):
            raise ValueError(f"line {line_number} has an invalid grader")
        seen_ids.add(case_id)
        cases.append(value)
    if not cases:
        raise ValueError("benchmark file contains no cases")
    if len({case["benchmark"] for case in cases}) != 1:
        raise ValueError("benchmark file mixes versions")
    if len({case["visibility"] for case in cases}) != 1:
        raise ValueError("benchmark file mixes visibility classes")
    return cases


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_endpoint(endpoint: str, *, api_key: str | None, allow_remote: bool, allow_credential_forwarding: bool) -> bool:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint credentials, query strings and fragments are forbidden")
    hostname = parsed.hostname.casefold()
    if hostname == "localhost":
        loopback = True
    else:
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
    if not loopback:
        if not allow_remote:
            raise ValueError("remote endpoint blocked; pass --allow-remote-endpoint deliberately")
        if parsed.scheme != "https":
            raise ValueError("remote endpoints must use HTTPS")
        if api_key and not allow_credential_forwarding:
            raise ValueError("credential forwarding blocked; pass --allow-credential-forwarding deliberately")
    return loopback


def safe_endpoint_for_report(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host + (f":{parsed.port}" if parsed.port else "")
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _safe_mapping(value: Any, allowed: dict[str, type | tuple[type, ...]]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, expected_type in allowed.items():
        item = value.get(key)
        if item is None or isinstance(item, expected_type):
            result[key] = item
    return result


def safe_backend_metadata(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    hydra = _safe_mapping(body.get("hydra"), {
        "selected_expert": str,
        "executed_expert": str,
        "router_score": (int, float),
        "fallback_reason": str,
        "active_parameters": int,
    })
    backend = _safe_mapping(body.get("hydra_backend"), {
        "name": str,
        "model": str,
        "latency_ms": (int, float),
        "fallback_used": bool,
        "fallback_reason": str,
        "primary_status": int,
        "primary_attempts": int,
        "primary_retries": int,
    })
    usage = _safe_mapping(body.get("usage"), {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
    })
    return hydra, backend, usage


def reasoning_observation(message: dict[str, Any]) -> dict[str, int | bool]:
    """Record only whether reasoning was exposed, never the private reasoning text."""
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return {"reasoning_observed": True, "reasoning_chars": len(value)}
    return {"reasoning_observed": False, "reasoning_chars": 0}


def build_request_payload(
    case: dict[str, Any],
    model: str,
    max_tokens: int,
    thinking_mode: str = "default",
) -> dict[str, Any]:
    """Serialize only request fields; grader/oracle data must never reach a provider."""
    if thinking_mode not in {"default", "on", "off"}:
        raise ValueError("thinking_mode must be default, on, or off")
    payload: dict[str, Any] = {
        "model": model,
        "messages": case["messages"],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if thinking_mode != "default":
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode == "on"}
    if case.get("tools"):
        payload["tools"] = case["tools"]
        payload["tool_choice"] = case.get("tool_choice", "auto")
    return payload


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def post_case(
    endpoint: str, api_key: str | None, payload: dict[str, Any], timeout: int
) -> tuple[dict[str, Any], float, int, dict[str, str | None]]:
    headers = {"Content-Type": "application/json", "User-Agent": "hydrach-bench/1.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(response.status)
        route_metadata = {
            "route": response.headers.get("X-Hydra-Route"),
            "backend": response.headers.get("X-Hydra-Backend"),
        }
        body = json.load(response)
    if not isinstance(body, dict):
        raise ValueError("provider response must be a JSON object")
    return body, time.perf_counter() - started, status, route_metadata


def grader_for_case(case: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    benchmark = case.get("benchmark")
    grader = GRADERS_BY_BENCHMARK.get(str(benchmark))
    if grader is None:
        raise ValueError(f"no grader registered for benchmark {benchmark!r}")
    return grader


def tool_subscore_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Split tool performance into choice, schema validity and exact arguments.

    A run that picks the right function but emits arguments the schema rejects is a
    runtime/parser problem; one that picks the wrong function is a capability gap.
    The single boolean in v1 could not tell those apart.
    """
    tool_rows = [row for row in rows if row.get("category") == "tool_call"]
    if not any(subscore in row for row in tool_rows for subscore in TOOL_SUBSCORES):
        return None
    positives = [row for row in tool_rows if row.get("grader") == "tool_call"]
    negatives = [row for row in tool_rows if row.get("grader") == "tool_none"]
    summary: dict[str, Any] = {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "false_call_rate": (
            round(sum(not bool(row["passed"]) for row in negatives) / len(negatives), 6)
            if negatives
            else None
        ),
    }
    for subscore in TOOL_SUBSCORES:
        scored = [row for row in positives if isinstance(row.get(subscore), bool)]
        summary[subscore] = {
            "count": len(scored),
            "passed": sum(bool(row[subscore]) for row in scored),
            "rate": (
                round(sum(bool(row[subscore]) for row in scored) / len(scored), 6)
                if scored
                else None
            ),
        }
    return summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    answered = [row for row in rows if not row.get("request_error")]
    request_errors = [row for row in rows if row.get("request_error")]
    latencies = [float(row["latency_seconds"]) for row in answered]
    error_latencies = [float(row["latency_seconds"]) for row in request_errors]
    passed = sum(bool(row["passed"]) for row in rows)
    answered_passed = sum(bool(row["passed"]) for row in answered)
    ordered_latencies = sorted(latencies)
    p95_index = max(0, (95 * len(ordered_latencies) + 99) // 100 - 1) if ordered_latencies else 0
    token_totals = {
        name: sum(
            int(row.get("usage", {}).get(name, 0))
            for row in answered
            if isinstance(row.get("usage"), dict) and isinstance(row.get("usage", {}).get(name, 0), int)
        )
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    reasoning_observed = sum(row.get("reasoning_observed") is True for row in answered)
    reasoning_not_observed = sum(row.get("reasoning_observed") is False for row in answered)
    reasoning_unknown = len(answered) - reasoning_observed - reasoning_not_observed
    return {
        "case_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round(passed / len(rows), 6) if rows else 0.0,
        "answered_cases": len(answered),
        "conditional_pass_rate": round(answered_passed / len(answered), 6) if answered else None,
        "request_errors": len(request_errors),
        "latency_seconds": {
            "scope": "answered cases only",
            "mean": round(statistics.mean(latencies), 4) if latencies else None,
            "p50": round(statistics.median(latencies), 4) if latencies else None,
            "p95": round(ordered_latencies[p95_index], 4) if ordered_latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "request_error_latency_seconds": {
            "mean": round(statistics.mean(error_latencies), 4) if error_latencies else None,
            "max": round(max(error_latencies), 4) if error_latencies else None,
        },
        "token_usage": token_totals,
        "reasoning_observation": {
            "observed": reasoning_observed,
            "not_observed": reasoning_not_observed,
            "unknown": reasoning_unknown,
        },
        "categories": {
            category: {
                "count": len(items),
                "passed": sum(bool(item["passed"]) for item in items),
                "pass_rate": round(sum(bool(item["passed"]) for item in items) / len(items), 6),
            }
            for category, items in sorted(grouped.items())
        },
        "tool_decomposition": tool_subscore_summary(rows),
    }


def build_report(args: argparse.Namespace, cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = {case["id"] for case in cases}
    answered_ids = {row["id"] for row in rows if not row.get("request_error")}
    summary = summarize(rows)
    usage = summary["token_usage"]
    input_price = float(getattr(args, "input_price_per_million", 0.0) or 0.0)
    output_price = float(getattr(args, "output_price_per_million", 0.0) or 0.0)
    estimated_cost = (
        usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price
    ) / 1_000_000
    budgets = dict(getattr(args, "category_token_budgets", category_token_budgets()))
    required_backend = getattr(args, "require_backend", None)
    required_route = getattr(args, "require_route", None)
    backend_violations: list[str] = []
    route_violations: list[str] = []
    if required_backend:
        for row in rows:
            if row.get("request_error"):
                continue
            metadata = row.get("inference_backend")
            if not isinstance(metadata, dict):
                backend_violations.append(str(row.get("id")))
                continue
            if metadata.get("name") != required_backend or metadata.get("fallback_used") is not False:
                backend_violations.append(str(row.get("id")))
    if required_route:
        for row in rows:
            if row.get("request_error"):
                continue
            if row.get("router_route") != required_route:
                route_violations.append(str(row.get("id")))
    thinking_mode = getattr(args, "thinking_mode", "default")
    reasoning_summary = summary["reasoning_observation"]
    answered_cases = summary["answered_cases"]
    thinking_gate: bool | None = None
    if thinking_mode == "on":
        thinking_gate = (
            reasoning_summary["observed"] == answered_cases
            and reasoning_summary["unknown"] == 0
        )
    elif thinking_mode == "off":
        thinking_gate = (
            reasoning_summary["observed"] == 0
            and reasoning_summary["unknown"] == 0
        )
    return {
        "report_version": REPORT_VERSION,
        "benchmark": cases[0].get("benchmark") if cases else None,
        "visibility": cases[0].get("visibility") if cases else None,
        "run_id": getattr(args, "run_id", None),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": safe_endpoint_for_report(args.endpoint),
        "model": args.model,
        "cases_path": str(args.cases),
        "cases_sha256": getattr(args, "cases_sha256", None),
        "manifest_sha256": getattr(args, "manifest_sha256", None),
        "runtime": {
            "git_commit": getattr(args, "git_commit", None),
            "evaluator_sha256": getattr(args, "evaluator_sha256", None),
            "python": platform.python_version(),
        },
        "inference_policy": {
            "temperature": 0,
            "thinking_mode": getattr(args, "thinking_mode", "default"),
            "token_budget_policy": TOKEN_BUDGET_POLICY_VERSION,
            "max_tokens_by_category": dict(sorted(budgets.items())),
            "timeout_seconds": getattr(args, "timeout", None),
            "pace_seconds": float(getattr(args, "pace_seconds", 0.0) or 0.0),
            "required_backend": required_backend,
            "required_route": required_route,
            "retries_per_process": 0,
            "request_errors_are_infrastructure_not_model_quality": True,
            "request_errors_retry_on_resume": True,
            "strict_score_includes_current_request_errors": True,
        },
        "thinking_observation": {
            "requested_mode": thinking_mode,
            **reasoning_summary,
            "passed": thinking_gate,
            "reasoning_text_stored": False,
        },
        "pricing": {
            "input_usd_per_million_tokens": input_price,
            "output_usd_per_million_tokens": output_price,
            "estimated_total_usd": round(estimated_cost, 8),
        },
        "total_cases": len(cases),
        "attempted_rows": len(rows),
        "completed_cases": len(answered_ids),
        "complete": answered_ids == expected_ids,
        "backend_gate": {
            "enabled": bool(required_backend),
            "required_backend": required_backend,
            "passed": not backend_violations if required_backend else None,
            "violation_count": len(backend_violations),
            "violation_case_ids": backend_violations,
        },
        "route_gate": {
            "enabled": bool(required_route),
            "required_route": required_route,
            "passed": not route_violations if required_route else None,
            "violation_count": len(route_violations),
            "violation_case_ids": route_violations,
        },
        "privacy": {
            "prompts_stored": False,
            "oracles_or_expected_answers_stored": False,
            "model_outputs_or_grader_actuals_stored": bool(args.keep_output),
            "backend_metadata_allowlisted": True,
        },
        "raw_outputs_stored": bool(args.keep_output),
        "paired_comparison": {
            "required": (cases[0].get("benchmark") if cases else None) in PAIRED_BENCHMARKS,
            "answers_stored": bool(args.keep_output),
            "compare_with": "scripts/compare_hydrach_runs.py (McNemar over shared case ids)",
        },
        "summary": summary,
        "results": rows,
    }


def evaluation_exit_code(report: dict[str, Any]) -> int:
    raw_summary = report.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else report
    request_errors = int(summary.get("request_errors", 0) or 0)
    if request_errors or not bool(report.get("complete", True)):
        return 2
    backend_gate = report.get("backend_gate")
    if isinstance(backend_gate, dict) and backend_gate.get("enabled") and backend_gate.get("passed") is not True:
        return 2
    route_gate = report.get("route_gate")
    if isinstance(route_gate, dict) and route_gate.get("enabled") and route_gate.get("passed") is not True:
        return 2
    case_count = int(summary.get("case_count", 0) or 0)
    passed = int(summary.get("passed", 0) or 0)
    return 0 if case_count > 0 and passed == case_count else 1


def validate_resume_checkpoint(
    previous: dict[str, Any],
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[str] = []
    expected = {
        "report_version": REPORT_VERSION,
        "cases_path": str(args.cases),
        "cases_sha256": args.cases_sha256,
        "manifest_sha256": getattr(args, "manifest_sha256", None),
        "total_cases": len(cases),
        "endpoint": safe_endpoint_for_report(args.endpoint),
        "model": args.model,
        "thinking_mode": getattr(args, "thinking_mode", "default"),
        "token_budget_policy": TOKEN_BUDGET_POLICY_VERSION,
        "max_tokens_by_category": dict(sorted(args.category_token_budgets.items())),
        "timeout_seconds": args.timeout,
        "pace_seconds": float(args.pace_seconds),
        "required_backend": args.require_backend,
        "required_route": getattr(args, "require_route", None),
        "evaluator_sha256": args.evaluator_sha256,
        "git_commit": args.git_commit,
        "keep_output": bool(args.keep_output),
        "input_price_per_million": float(args.input_price_per_million),
        "output_price_per_million": float(args.output_price_per_million),
    }
    policy = previous.get("inference_policy") or {}
    runtime = previous.get("runtime") or {}
    pricing = previous.get("pricing") or {}
    observed = {
        "report_version": previous.get("report_version"),
        "cases_path": previous.get("cases_path"),
        "cases_sha256": previous.get("cases_sha256"),
        "manifest_sha256": previous.get("manifest_sha256"),
        "total_cases": previous.get("total_cases"),
        "endpoint": previous.get("endpoint"),
        "model": previous.get("model"),
        "thinking_mode": policy.get("thinking_mode", "default"),
        "token_budget_policy": policy.get("token_budget_policy"),
        "max_tokens_by_category": policy.get("max_tokens_by_category"),
        "timeout_seconds": policy.get("timeout_seconds"),
        "pace_seconds": float(policy.get("pace_seconds", 0.0)),
        "required_backend": policy.get("required_backend"),
        "required_route": policy.get("required_route"),
        "evaluator_sha256": runtime.get("evaluator_sha256"),
        "git_commit": runtime.get("git_commit"),
        "keep_output": bool(previous.get("raw_outputs_stored")),
        "input_price_per_million": float(pricing.get("input_usd_per_million_tokens", 0.0)),
        "output_price_per_million": float(pricing.get("output_usd_per_million_tokens", 0.0)),
    }
    for field, value in expected.items():
        if observed.get(field) != value:
            mismatches.append(field)

    rows = previous.get("results")
    if not isinstance(rows, list):
        mismatches.append("results")
        rows = []
    raw_row_ids = [row.get("id") for row in rows if isinstance(row, dict)]
    valid_row_ids = all(isinstance(item, str) for item in raw_row_ids)
    row_ids = [item for item in raw_row_ids if isinstance(item, str)]
    case_ids = {case["id"] for case in cases}
    if not valid_row_ids or len(raw_row_ids) != len(rows) or len(set(row_ids)) != len(row_ids):
        mismatches.append("result_ids")
    if not set(row_ids).issubset(case_ids):
        mismatches.append("unknown_result_ids")
    cases_by_id = {case["id"]: case for case in cases}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in cases_by_id:
            continue
        case = cases_by_id[row["id"]]
        expected_max_tokens = token_budget_for_case(case, args.category_token_budgets)
        expected_payload = build_request_payload(
            case,
            args.model,
            expected_max_tokens,
            getattr(args, "thinking_mode", "default"),
        )
        expected_request_sha256 = hashlib.sha256(canonical_json_bytes(expected_payload)).hexdigest()
        if row.get("category") != case.get("category"):
            mismatches.append("result_categories")
        if row.get("max_tokens") != expected_max_tokens:
            mismatches.append("result_max_tokens")
        if row.get("request_sha256") != expected_request_sha256:
            mismatches.append("result_request_sha256")
    answered_rows = [row for row in rows if isinstance(row, dict) and not row.get("request_error")]
    answered_ids = {row["id"] for row in answered_rows}
    if previous.get("completed_cases") != len(answered_ids):
        mismatches.append("completed_cases")
    if mismatches:
        raise ValueError("refusing to resume: checkpoint mismatch in " + ", ".join(sorted(set(mismatches))))
    # Transport failures are infrastructure evidence, not completed cases. Resume retries them.
    return answered_rows


def _assert_safe_output_path(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise ValueError(f"output path contains a symlink: {current}")
    if path.exists():
        metadata = path.stat()
        if not path.is_file() or metadata.st_nlink != 1:
            raise ValueError(f"output target must be a regular single-link file: {path}")


def write_checkpoint(
    output_path: Path,
    report: dict[str, Any],
    *,
    private: bool,
) -> None:
    _assert_safe_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_path(output_path)
    mode = 0o600 if private else 0o644
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, output_path)
        os.chmod(output_path, mode)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18081/v1/chat/completions")
    parser.add_argument("--model", default="hydra-1.0-v")
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "on", "off"),
        default="default",
        help="Qwen chat-template reasoning mode; recorded in the report",
    )
    parser.add_argument("--api-key-env", default="MORNINGSTAR_HYDRA_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=3.0,
        help="pause between benchmark cases to avoid provider rate limits (default: 3)",
    )
    parser.add_argument(
        "--require-backend",
        help="exit 2 unless every answered case used this backend with fallback_used=false",
    )
    parser.add_argument(
        "--require-route",
        choices=("general", "specialist", "deterministic"),
        help="exit 2 unless every answered case exposes this X-Hydra-Route",
    )
    parser.add_argument(
        "--max-tokens",
        "--short-max-tokens",
        dest="short_max_tokens",
        type=int,
        default=64,
        help="output budget for short deterministic categories (legacy --max-tokens alias retained)",
    )
    parser.add_argument(
        "--structured-max-tokens",
        type=int,
        default=512,
        help="output budget for invoice_json, multilingual_json and tool_call (minimum 512)",
    )
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--keep-output", action="store_true", help="retain raw model output and parsed grader actuals in a private report")
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="run a paired benchmark without storing answers (blocks McNemar adjudication later)",
    )
    parser.add_argument("--allow-remote-endpoint", action="store_true")
    parser.add_argument("--allow-credential-forwarding", action="store_true")
    parser.add_argument("--allow-private-cases-remote", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.pace_seconds < 0:
        raise ValueError("pace seconds must be non-negative")
    args.category_token_budgets = category_token_budgets(
        short_max_tokens=args.short_max_tokens,
        structured_max_tokens=args.structured_max_tokens,
    )
    if args.input_price_per_million < 0 or args.output_price_per_million < 0:
        raise ValueError("token prices must be non-negative")
    _assert_safe_output_path(args.cases)
    _assert_safe_output_path(args.manifest)
    cases = load_cases(args.cases)
    visibility = cases[0]["visibility"]
    private_cases = visibility == "hidden"
    benchmark = str(cases[0].get("benchmark"))
    grader_for_case(cases[0])
    if benchmark in PAIRED_BENCHMARKS and not args.keep_output and not args.allow_unpaired:
        raise ValueError(
            f"{benchmark} is evaluated pairwise (McNemar); rerun with --keep-output so both "
            "candidates' answers per case are on disk, or pass --allow-unpaired to waive it"
        )
    args.cases_sha256 = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    manifest_bytes = args.manifest.read_bytes()
    args.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    expected_split = manifest.get("splits", {}).get(visibility, {})
    if expected_split.get("sha256") != args.cases_sha256 or expected_split.get("count") != len(cases):
        raise ValueError("case file does not match the frozen manifest")
    args.git_commit = git_commit()
    args.evaluator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    loopback = validate_endpoint(
        args.endpoint,
        api_key=api_key,
        allow_remote=args.allow_remote_endpoint,
        allow_credential_forwarding=args.allow_credential_forwarding,
    )
    if private_cases and not loopback and not args.allow_private_cases_remote:
        raise ValueError("private cases cannot leave loopback without --allow-private-cases-remote")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[: args.limit]

    output_in_repo = args.output.resolve(strict=False).is_relative_to(ROOT.resolve())
    if private_cases and output_in_repo:
        raise ValueError("hidden benchmark reports must remain outside the Git worktree")
    if args.keep_output and output_in_repo:
        raise ValueError("reports containing model output must remain outside the Git worktree")
    private_output = private_cases or args.keep_output
    rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        rows = validate_resume_checkpoint(previous, args, cases)
        args.run_id = args.run_id or previous.get("run_id")
    args.run_id = args.run_id or f"hydrach-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    completed_ids = {row["id"] for row in rows}
    pending_cases = [case for case in cases if case["id"] not in completed_ids]
    if rows:
        print(f"resuming with {len(rows)}/{len(cases)} completed cases", flush=True)
    completed_count = len(rows)
    for pending_index, case in enumerate(pending_cases):
        index = completed_count + pending_index + 1
        pace_before_case(pending_index, args.pace_seconds)
        max_tokens = token_budget_for_case(case, args.category_token_budgets)
        payload = build_request_payload(
            case,
            args.model,
            max_tokens,
            args.thinking_mode,
        )
        request_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        request_started = time.perf_counter()
        try:
            body, latency, http_status, route_metadata = post_case(
                args.endpoint,
                api_key,
                payload,
                args.timeout,
            )
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("provider response did not contain a valid choices array")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ValueError("provider response did not contain a valid message")
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("provider tool_calls must be a list")
            response = {
                "content": message.get("content") or "",
                "tool_calls": tool_calls,
            }
            grade = grader_for_case(case)(case, response)
            hydra_meta, backend_meta, usage_meta = safe_backend_metadata(body)
            row: dict[str, Any] = {
                "id": case["id"],
                "category": case["category"],
                "language": case["language"],
                "passed": grade["passed"],
                "grader": grade["grader"],
                "grader_error": grade["error"],
                "latency_seconds": round(latency, 4),
                "backend": hydra_meta,
                "inference_backend": backend_meta,
                "usage": usage_meta,
                "http_status": http_status,
                "router_route": route_metadata["route"],
                "router_backend": route_metadata["backend"],
                "request_sha256": request_sha256,
                "max_tokens": max_tokens,
                "request_error": None,
                **reasoning_observation(message),
            }
            for subscore in TOOL_SUBSCORES:
                if isinstance(grade.get(subscore), bool):
                    row[subscore] = grade[subscore]
            if args.keep_output:
                row["output"] = response
                row["actual"] = grade["actual"]
        except (OSError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            row = {
                "id": case["id"],
                "category": case["category"],
                "language": case["language"],
                "passed": False,
                "grader": case["grader"]["type"],
                "grader_error": None,
                "latency_seconds": round(time.perf_counter() - request_started, 4),
                "backend": {},
                "inference_backend": {},
                "usage": {},
                "http_status": getattr(exc, "code", None),
                "router_route": None,
                "router_backend": None,
                "request_sha256": request_sha256,
                "max_tokens": max_tokens,
                "request_error": type(exc).__name__,
                "reasoning_observed": None,
                "reasoning_chars": 0,
            }
        rows.append(row)
        write_checkpoint(args.output, build_report(args, cases, rows), private=private_output)
        print(f"[{index}/{len(cases)}] {case['id']} {'PASS' if row['passed'] else 'FAIL'}", flush=True)

    report = build_report(args, cases, rows)
    write_checkpoint(args.output, report, private=private_output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return evaluation_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
