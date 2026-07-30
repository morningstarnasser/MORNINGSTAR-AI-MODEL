from __future__ import annotations

import argparse
import json
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def post(url: str, payload: dict, timeout: int = 900) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18081/v1/chat/completions")
    parser.add_argument("--output", default="benchmarks/hydra-smoke.json")
    parser.add_argument("--model", default="morningstar-hydra")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--hardware", default="Synology DS1522+ / AMD Ryzen Embedded R1600 / 4 logical CPUs / 64 GiB RAM")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--context", type=int, default=2048)
    args = parser.parse_args()
    prompts = [
        "Antworte exakt mit: HYDRA_CPU_OK",
        "Schreibe eine Python-Funktion add(a, b) mit Type Hints und nur drei Zeilen.",
    ]
    rows = []
    for run in range(1, args.runs + 1):
        for prompt in prompts:
            result, elapsed = post(args.url, {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": args.max_tokens})
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})
            completion_tokens = int(usage.get("completion_tokens") or 0)
            rows.append({
                "run": run,
                "prompt": prompt,
                "response": text,
                "elapsed_seconds": round(elapsed, 3),
                "completion_tokens": completion_tokens,
                "tokens_per_second": round(completion_tokens / elapsed, 3) if completion_tokens else None,
                "hydra": result.get("hydra", {}),
            })
    measured = [row["tokens_per_second"] for row in rows if row["tokens_per_second"] is not None]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_platform": platform.platform(),
        "backend_hardware": args.hardware,
        "quantization": args.quantization,
        "context_tokens": args.context,
        "url": args.url,
        "model": args.model,
        "runs": args.runs,
        "mean_tokens_per_second": round(sum(measured) / len(measured), 3) if measured else None,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
