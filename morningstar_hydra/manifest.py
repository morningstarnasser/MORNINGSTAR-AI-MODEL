from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from morningstar_hydra.identity import CANONICAL_IDENTITY
else:
    from .identity import CANONICAL_IDENTITY

SHARED_PARAMETERS = 4_577_530_880
FAMILY_PARAMETERS = 10_192_158_720
FAMILY_COUNT = 275
ACTIVE_PARAMETERS = SHARED_PARAMETERS + FAMILY_PARAMETERS

DOMAINS = [
    ("general", ["general"]),
    ("coding-python", ["python", "pytest", "django", "fastapi", "pandas"]),
    ("coding-javascript", ["javascript", "typescript", "node.js", "react", "next.js"]),
    ("systems", ["linux", "kernel", "rust", "c++", "performance"]),
    ("devops", ["docker", "kubernetes", "terraform", "caddy", "deploy"]),
    ("security", ["security", "vulnerability", "oauth", "cve", "pentest"]),
    ("math", ["mathematik", "math", "algebra", "geometrie", "beweis"]),
    ("agents", ["agent", "tool calling", "orchestration", "workflow"]),
    ("data", ["sql", "postgres", "database", "analytics", "etl"]),
    ("swiss-kmu", ["schweiz", "kmu", "zürich", "angebot", "kunde"]),
    ("legal-ch", ["schweizer recht", "or", "revdsG", "vertrag", "haftung"]),
    ("finance", ["finanzen", "portfolio", "aktie", "budget", "rendite"]),
    ("science", ["wissenschaft", "physik", "chemie", "biologie"]),
    ("german", ["deutsch", "schweizerdeutsch", "text", "korrektur"]),
    ("research", ["recherche", "quellen", "paper", "literatur"]),
    ("reasoning", ["logik", "reasoning", "rätsel", "schlussfolgerung"]),
]


def build_manifest(base_uri: str, base_sha256: str = "", base_ready: bool = False, benchmark_uri: str | None = None) -> dict:
    families = []
    for index in range(FAMILY_COUNT):
        if index < len(DOMAINS):
            family_id, keywords = DOMAINS[index]
        else:
            family_id, keywords = f"reserved-{index:03d}", []
        is_general = family_id == "general"
        ready = is_general and base_ready
        families.append({
            "id": family_id,
            "slot": index,
            "parameter_count": FAMILY_PARAMETERS,
            "status": "ready" if ready else "planned",
            "keywords": keywords,
            "artifact_uri": base_uri if ready else None,
            "sha256": base_sha256 if ready else None,
            "backend_model": "morningstar-hydra-base" if ready else None,
            "benchmark_uri": benchmark_uri if ready else None,
        })
    total = SHARED_PARAMETERS + FAMILY_COUNT * FAMILY_PARAMETERS
    return {
        "schema_version": 1,
        "name": "morningstar-hydra",
        "identity": dict(CANONICAL_IDENTITY),
        "stage": "cpu-prototype",
        "architecture": "request-routed factorized MLP expert catalog",
        "base_architecture": "Qwen2.5-Coder-14B-Instruct",
        "parameter_accounting": {
            "shared_parameters": SHARED_PARAMETERS,
            "expert_family_parameters": FAMILY_PARAMETERS,
            "expert_family_count": FAMILY_COUNT,
            "catalog_parameters_planned": total,
            "active_parameters_per_request": ACTIVE_PARAMETERS,
            "counting_rule": "shared trunk counted once plus 275 distinct full-depth MLP expert families",
        },
        "storage": {
            "nas_root": "local://model-store",
            "factorized_q4_estimate_bytes": 1_401_421_824_000,
            "full_merged_q4_catalog_estimate_bytes": 2_471_730_544_800,
        },
        "claims": {
            "allowed_now": "CPU prototype with one real 14.77B active path and a validated >2.8T catalog design",
            "forbidden_now": "Do not claim a trained or realized >2.8T model until every family has a distinct verified artifact",
        },
        "expert_families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="config/hydra-manifest.json")
    parser.add_argument("--base-uri", default="local://model-store/models/base/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--base-sha256", default="")
    parser.add_argument("--base-ready", action="store_true")
    parser.add_argument("--benchmark-uri", default=None)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_manifest(args.base_uri, args.base_sha256, args.base_ready, args.benchmark_uri), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
