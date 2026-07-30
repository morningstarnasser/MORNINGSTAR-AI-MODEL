from __future__ import annotations

import argparse
import json
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def stream_once(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at = None
    chunks: list[str] = []
    final_event: dict = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            final_event = event
            choices = event.get("choices") or [{}]
            content = choices[0].get("delta", {}).get("content") or ""
            if content and first_content_at is None:
                first_content_at = time.perf_counter()
            if content:
                chunks.append(content)
    finished = time.perf_counter()
    timings = final_event.get("timings") or {}
    return {
        "prompt": prompt,
        "response": "".join(chunks),
        "ttft_seconds": round(first_content_at - started, 3) if first_content_at else None,
        "elapsed_seconds": round(finished - started, 3),
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "generated_tokens": timings.get("predicted_n"),
        "generation_tokens_per_second": timings.get("predicted_per_second"),
        "backend_timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    parser.add_argument("--model", default="morningstar-hydra-base")
    parser.add_argument("--output", default="benchmarks/nas-cpu-stream.json")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    prompts = [
        "Antworte exakt mit: HYDRA_STREAM_OK",
        "Schreibe eine Python-Funktion add(a, b) mit Type Hints und nur drei Zeilen.",
    ]
    rows = [stream_once(args.url, args.model, prompt, args.max_tokens) for _ in range(args.runs) for prompt in prompts]
    generation_rates = [float(row["generation_tokens_per_second"]) for row in rows if row["generation_tokens_per_second"] is not None]
    ttfts = [float(row["ttft_seconds"]) for row in rows if row["ttft_seconds"] is not None]
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_platform": platform.platform(),
        "backend_hardware": "Synology DS1522+ / AMD Ryzen Embedded R1600 / 4 logical CPUs / 64 GiB RAM",
        "model": "Qwen2.5-Coder-14B-Instruct",
        "quantization": "Q4_K_M",
        "context_tokens": 2048,
        "url": args.url,
        "runs": args.runs,
        "mean_ttft_seconds": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "mean_generation_tokens_per_second": round(sum(generation_rates) / len(generation_rates), 3) if generation_rates else None,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
