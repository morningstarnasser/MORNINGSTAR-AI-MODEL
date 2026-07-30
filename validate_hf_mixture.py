#!/usr/bin/env python3
"""Validate Hydra's Hugging Face candidate-source policy without downloading data."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "config" / "hf-expert-mixture.json"
REQUIRED_SOURCE_KEYS = {
    "dataset",
    "config",
    "split",
    "purpose",
    "license",
    "weight",
    "sample_cap",
    "filters",
    "reason",
}


def get_json(url: str, token: str | None = None) -> tuple[int, dict[str, Any]]:
    headers = {"User-Agent": "morningstar-hydra-dataset-validator/1.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.load(response)
            return response.status, payload if isinstance(payload, dict) else {"error": "non-object JSON"}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200]
        return exc.code, {"error": body}
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, {"error": str(exc)}


def normalize_license(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, list):
        return {str(item).lower() for item in value}
    return set()


def validate_source_schema(source: Any, index: int) -> list[str]:
    prefix = f"sources[{index}]"
    if not isinstance(source, dict):
        return [f"{prefix} must be an object"]
    missing = sorted(REQUIRED_SOURCE_KEYS - source.keys())
    errors = [f"{prefix} missing keys: {missing}"] if missing else []
    raw_weight = source.get("weight")
    try:
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float, str)):
            raise TypeError("weight is not numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            errors.append(f"{prefix}.weight must be finite and positive")
    except (TypeError, ValueError):
        errors.append(f"{prefix}.weight must be numeric")
    sample_cap = source.get("sample_cap")
    if not isinstance(sample_cap, int) or isinstance(sample_cap, bool) or sample_cap <= 0:
        errors.append(f"{prefix}.sample_cap must be a positive integer")
    if not isinstance(source.get("filters"), list) or not source.get("filters"):
        errors.append(f"{prefix}.filters must be a non-empty list")
    return errors


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest sources must be a non-empty list")

    errors: list[str] = []
    for index, source in enumerate(sources):
        errors.extend(validate_source_schema(source, index))
    if errors:
        return {"ok": False, "scope": "schema", "errors": errors}

    if manifest.get("strategy") != "curated_post_training_only":
        errors.append("strategy must remain curated_post_training_only")
    if manifest.get("foundation_pretraining") is not False:
        errors.append("foundation_pretraining must remain false")

    contamination = manifest.get("contamination_control")
    if not isinstance(contamination, dict):
        errors.append("contamination_control must be an object")
        contamination = {}
    contamination_status = contamination.get("status")
    required_before_training = contamination.get("required_before_training")
    if (
        not isinstance(required_before_training, list)
        or not required_before_training
        or not all(isinstance(item, str) and item.strip() for item in required_before_training)
    ):
        errors.append("contamination_control.required_before_training must be a non-empty string list")
    training_allowed = manifest.get("training_allowed")
    if training_allowed not in (True, False):
        errors.append("training_allowed must be a boolean")
    if training_allowed and contamination_status != "verified":
        errors.append("training cannot be allowed until contamination_control.status is verified")
    report_sha256 = contamination.get("report_sha256")
    if training_allowed and (
        not isinstance(report_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", report_sha256)
    ):
        errors.append("verified training requires a lowercase 64-hex contamination report SHA-256")

    weights = [float(source["weight"]) for source in sources]
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-9:
        errors.append(f"source weights sum to {weight_sum}, expected 1.0")

    evaluation_only = manifest.get("evaluation_only")
    if not isinstance(evaluation_only, list) or not evaluation_only:
        errors.append("evaluation_only must be a non-empty list")
        evaluation_only = []
    duplicate_eval = sorted(
        {source["dataset"] for source in sources}
        & {
            entry.get("dataset")
            for entry in evaluation_only
            if isinstance(entry, dict) and isinstance(entry.get("dataset"), str)
        }
    )
    if duplicate_eval:
        errors.append(f"exact dataset-ID training/evaluation overlap: {duplicate_eval}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    results: list[dict[str, Any]] = []
    for source in sources:
        dataset_id = source["dataset"]
        api_status, metadata = get_json(f"https://huggingface.co/api/datasets/{dataset_id}", token)
        source_errors: list[str] = []
        if api_status != 200:
            source_errors.append(f"metadata HTTP {api_status}: {metadata.get('error', 'unknown error')}")
        else:
            card = metadata.get("cardData") or {}
            actual_licenses = normalize_license(card.get("license"))
            declared_license = str(source["license"]).lower()
            if metadata.get("private"):
                source_errors.append("dataset is private")
            if metadata.get("gated") not in (False, None):
                source_errors.append(f"dataset is gated: {metadata.get('gated')}")
            if declared_license not in actual_licenses:
                source_errors.append(
                    f"license mismatch: declared={declared_license}, actual={sorted(actual_licenses)}"
                )

        split_url = "https://datasets-server.huggingface.co/splits?" + urllib.parse.urlencode(
            {"dataset": dataset_id}
        )
        split_status, split_payload = get_json(split_url, token)
        if split_status != 200:
            source_errors.append(
                f"split API HTTP {split_status}: {split_payload.get('error', 'unknown error')}"
            )
        else:
            available = {
                (item.get("config"), item.get("split"))
                for item in split_payload.get("splits", [])
                if isinstance(item, dict)
            }
            wanted = (source["config"], source["split"])
            if wanted not in available:
                source_errors.append(f"missing config/split {wanted}")

        results.append(
            {
                "dataset": dataset_id,
                "config": source["config"],
                "split": source["split"],
                "license": source["license"],
                "downloads": metadata.get("downloads") if api_status == 200 else None,
                "likes": metadata.get("likes") if api_status == 200 else None,
                "ok": not source_errors,
                "errors": source_errors,
            }
        )

    excluded_results: list[dict[str, Any]] = []
    excluded = manifest.get("excluded")
    if not isinstance(excluded, list):
        errors.append("excluded must be a list")
        excluded = []
    for index, entry in enumerate(excluded):
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset"), str):
            errors.append(f"excluded[{index}] must contain a dataset string")
            continue
        expected = entry.get("expected_state")
        status, metadata = get_json(f"https://huggingface.co/api/datasets/{entry['dataset']}", token)
        exclusion_errors: list[str] = []
        if status != 200:
            exclusion_errors.append(f"metadata HTTP {status}: {metadata.get('error', 'unknown error')}")
        elif expected == "gated" and metadata.get("gated") in (False, None):
            exclusion_errors.append("expected gated dataset is currently ungated")
        elif expected == "missing_license" and normalize_license((metadata.get("cardData") or {}).get("license")):
            exclusion_errors.append("expected missing license is now declared; manual review required")
        elif expected not in {"gated", "missing_license"}:
            exclusion_errors.append(f"unknown expected_state: {expected}")
        excluded_results.append(
            {
                "dataset": entry["dataset"],
                "expected_state": expected,
                "metadata_status": status,
                "ok": not exclusion_errors,
                "errors": exclusion_errors,
            }
        )

    failed = [result["dataset"] for result in results if not result["ok"]]
    unavailable_excluded = [
        result["dataset"]
        for result in excluded_results
        if not result["ok"] and result["metadata_status"] != 200
    ]
    changed_excluded = [
        result["dataset"]
        for result in excluded_results
        if not result["ok"] and result["metadata_status"] == 200
    ]
    if failed:
        errors.append(f"source metadata validation failed: {failed}")
    if unavailable_excluded:
        errors.append(f"excluded-source metadata unavailable: {unavailable_excluded}")
    if changed_excluded:
        errors.append(f"excluded-source state changed: {changed_excluded}")

    return {
        "ok": not errors,
        "scope": "live metadata, gates, licenses, config/split names, weights, and exact dataset-ID overlap only",
        "sample_level_contamination_verified": contamination_status == "verified",
        "training_allowed": training_allowed,
        "strategy": manifest.get("strategy"),
        "foundation_pretraining": manifest.get("foundation_pretraining"),
        "weight_sum": weight_sum,
        "source_count": len(results),
        "sources": results,
        "excluded": excluded_results,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", help="print compact JSON")
    args = parser.parse_args()
    try:
        result = validate_manifest(args.manifest)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "scope": "schema", "error": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")) if args.json else json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
