#!/usr/bin/env python3
"""Misst allgemeines Reasoning (GSM8K) gegen ein lokales MLX-Modell.

Warum: HydraCH prueft ausschliesslich Schweizer Formataufgaben und sagt nichts
ueber Schlussfolgern. Beim Finetuning auf enge Daten droht aber genau dort ein
Rueckschritt (catastrophic forgetting). Dieser Test beantwortet zwei Fragen:
wie stark das Modell im Reasoning ist, und ob ein Adapter es verschlechtert.

GSM8K besteht aus mehrstufigen Textaufgaben; die Referenzantwort steht nach
"#### ". Gewertet wird die letzte Zahl der Modellantwort — so bleibt Raum fuer
einen Rechenweg, ohne dass das Format ueber Erfolg entscheidet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

ZAHL = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
    r"(?:[eE][-+]?\d+)?(?!\d)"
)
REPORT_VERSION = 4
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"
DEFAULT_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PROMPT_SUFFIX = (
    "Denke Schritt für Schritt und schreibe zum Schluss nur die Zahl als Endergebnis."
)

# Zwei vollstaendig ausgerechnete Beispiele. Sie zeigen dem Modell die erwartete
# Ausfuehrlichkeit, ohne ihm Wissen zu geben — daran laesst sich unterscheiden, ob
# eine Faehigkeit fehlt oder nur der Antwortstil zu knapp geworden ist.
FEW_SHOT = [
    {"role": "user", "content":
     "Ein Korb enthält 12 Äpfel. Anna nimmt 3 heraus, dann legt Ben doppelt so "
     "viele hinein, wie Anna genommen hat. Wie viele Äpfel sind jetzt im Korb?\n\n"
     "Denke Schritt für Schritt und schreibe zum Schluss nur die Zahl als Endergebnis."},
    {"role": "assistant", "content":
     "Start: 12 Äpfel.\nAnna nimmt 3 heraus: 12 - 3 = 9.\n"
     "Ben legt doppelt so viele hinein, wie Anna genommen hat: 2 * 3 = 6.\n"
     "Im Korb: 9 + 6 = 15.\n15"},
    {"role": "user", "content":
     "Ein Buch kostet 24 Franken. Im Ausverkauf gibt es 25 Prozent Rabatt. "
     "Wie viel kosten drei Bücher im Ausverkauf?\n\n"
     "Denke Schritt für Schritt und schreibe zum Schluss nur die Zahl als Endergebnis."},
    {"role": "assistant", "content":
     "Rabatt pro Buch: 24 * 0.25 = 6.\nPreis pro Buch: 24 - 6 = 18.\n"
     "Drei Bücher: 3 * 18 = 54.\n54"},
]


def letzte_zahl(text: str) -> str | None:
    treffer = ZAHL.findall(text.replace("**", ""))
    if not treffer:
        return None
    roh = treffer[-1]
    mantisse, exponent = re.split(r"[eE]", roh, maxsplit=1) if re.search(r"[eE]", roh) else (roh, None)
    if "," in mantisse:
        # German formatting: dots group thousands and comma separates decimals.
        mantisse = mantisse.replace(".", "").replace(",", ".")
    elif mantisse.count(".") > 1:
        mantisse = mantisse.replace(".", "")
    elif "." in mantisse:
        integer, fraction = mantisse.lstrip("+-").split(".", maxsplit=1)
        if len(fraction) == 3 and 1 <= len(integer) <= 3:
            # The unambiguous convention used by the German benchmark output:
            # 1.234 is a thousands grouping; ordinary decimals such as 12.5 stay decimal.
            mantisse = mantisse.replace(".", "")
    normalized = mantisse + (f"e{exponent}" if exponent is not None else "")
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def referenz(antwort: str) -> str:
    wert = antwort.split("####")[-1].strip()
    return letzte_zahl(wert) or wert


def apply_reasoning_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    thinking_mode: str,
) -> Any:
    """Apply the chat template without changing the legacy default-mode call."""
    if thinking_mode not in {"default", "on", "off"}:
        raise ValueError(f"invalid thinking mode: {thinking_mode!r}")
    kwargs: dict[str, Any] = {"add_generation_prompt": True}
    if thinking_mode != "default":
        kwargs["enable_thinking"] = thinking_mode == "on"
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if thinking_mode == "default":
            raise
        raise RuntimeError(
            f"tokenizer does not support explicit thinking mode {thinking_mode!r}; "
            "its chat template rejected enable_thinking"
        ) from exc


def load_reasoning_model(
    load_fn: Callable[..., Any],
    model: str,
    adapter_path: Path | None,
) -> Any:
    """Load the base model, adding the MLX adapter kwarg only when configured."""
    if adapter_path is None:
        return load_fn(model)
    normalized = (
        adapter_path.parent
        if adapter_path.suffix.casefold() == ".safetensors"
        else adapter_path
    )
    return load_fn(model, adapter_path=str(normalized))


def _artifact_provenance(
    path: Path,
    *,
    relative_path: str | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {
        "filename": relative_path or path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def _is_model_manifest_artifact(path: Path) -> bool:
    name = path.name.casefold()
    return (
        name.endswith(".safetensors")
        or name.endswith(".index.json")
        or "config" in name and name.endswith(".json")
        or "tokenizer" in name
        or name
        in {
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "vocab.json",
        }
    )


def build_local_model_manifest(model: str) -> dict[str, Any] | None:
    """Hash only reproducibility artifacts for a local model directory."""
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        return None
    artifacts = [
        _artifact_provenance(path, relative_path=path.relative_to(model_path).as_posix())
        for path in sorted(model_path.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and _is_model_manifest_artifact(path)
    ]
    return {
        "path": str(model_path),
        "artifacts": artifacts,
    }


def build_adapter_provenance(adapter_path: Path | None) -> dict[str, Any] | None:
    """Validate adapter artifacts and return content-free reproducibility metadata."""
    if adapter_path is None:
        return None
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")

    if adapter_path.is_dir():
        kind = "directory"
        weights_path = adapter_path / "adapters.safetensors"
        config_path = adapter_path / "adapter_config.json"
    elif adapter_path.is_file():
        kind = "weights_file"
        weights_path = adapter_path
        config_path = adapter_path.parent / "adapter_config.json"
    else:
        raise ValueError(f"adapter path is not a file or directory: {adapter_path}")

    missing = [
        path.name
        for path in (weights_path, config_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"adapter path {adapter_path} lacks required artifact(s): "
            + ", ".join(missing)
        )

    return {
        "configured_path": str(adapter_path),
        "kind": kind,
        "artifacts": {
            "weights": _artifact_provenance(weights_path),
            "config": _artifact_provenance(config_path),
        },
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_reasoning_report(
    args: argparse.Namespace,
    *,
    correct: int,
    total: int,
    failures: list[dict[str, Any]],
    dataset_fingerprint: str | None,
    adapter_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the reasoning report while retaining the original result fields."""
    if adapter_provenance is None:
        adapter_provenance = build_adapter_provenance(
            getattr(args, "adapter_path", None)
        )
    accuracy = correct / max(1, total)
    prompt_contract = {
        "few_shot_messages": FEW_SHOT if args.few_shot else [],
        "question_suffix": PROMPT_SUFFIX,
    }
    prompt_sha256 = hashlib.sha256(
        json.dumps(
            prompt_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evaluator = _artifact_provenance(Path(__file__).resolve())
    return {
        "report_version": REPORT_VERSION,
        "benchmark": "GSM8K-main-test",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_provenance": {
            "requested_revision": getattr(args, "model_revision", None),
            "local_manifest": build_local_model_manifest(args.model),
        },
        "adapter": adapter_provenance,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "failures": failures,
        "inference": {
            "thinking_mode": args.thinking_mode,
            "enable_thinking": (
                None if args.thinking_mode == "default" else args.thinking_mode == "on"
            ),
            "few_shot": bool(args.few_shot),
            "few_shot_examples": 2 if args.few_shot else 0,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "prompt_sha256": prompt_sha256,
        },
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "streaming": True,
            "requested_revision": args.dataset_revision,
            "fingerprint": dataset_fingerprint,
            "limit": args.limit,
        },
        "runtime": {
            "git_commit": _git_commit(),
            "evaluator_sha256": evaluator["sha256"],
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
        },
    }


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


def write_reasoning_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write a private reasoning report."""
    _assert_safe_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_output_path(path)
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def chat_payload(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    thinking_mode: str,
) -> dict[str, Any]:
    """Baut die OpenAI-Anfrage fuer den HTTP-Weg."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if thinking_mode in {"on", "off"}:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode == "on"}
    return payload


def answer_from_response(response: dict[str, Any]) -> tuple[str, bool]:
    """Liefert (Antworttext, abgeschnitten).

    Ein Denkmodell kann das gesamte Token-Budget im Denkteil verbrauchen und
    ohne ``content`` zurueckkommen. Das als falsche Antwort zu werten wuerde die
    Faehigkeit des Modells verschleiern und stattdessen das Budget messen —
    solche Faelle werden darum getrennt gezaehlt.
    """
    try:
        choice = response["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict):
            raise TypeError("message must be an object")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("model response has no assistant message") from exc
    content = (message.get("content") or "").strip()
    truncated = (
        not content
        and bool(message.get("reasoning"))
        and choice.get("finish_reason") == "length"
    )
    return content, truncated


def http_generator(
    endpoint: str, *, model: str, max_tokens: int, thinking_mode: str, timeout: float
) -> Callable[[list[dict[str, str]]], tuple[str, bool]]:
    import httpx

    client = httpx.Client(timeout=timeout, trust_env=False)

    def erzeuge(messages: list[dict[str, str]]) -> tuple[str, bool]:
        antwort = client.post(
            endpoint,
            json=chat_payload(
                messages, model=model, max_tokens=max_tokens, thinking_mode=thinking_mode
            ),
        )
        antwort.raise_for_status()
        return answer_from_response(antwort.json())

    return erzeuge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--endpoint",
        help="OpenAI-kompatibler Endpunkt statt eines lokalen MLX-Modells — noetig, "
             "um gegen Ollama, die Bruecke oder den Router zu messen",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--model-revision",
        help="immutable model revision metadata (required for reproducible remote identifiers)",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="optional MLX LoRA adapter directory or direct weights file",
    )
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking-mode",
        choices=("default", "on", "off"),
        default="default",
        help="default omits enable_thinking; on/off require tokenizer support",
    )
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
        help=f"Hugging Face dataset revision (default: {DEFAULT_DATASET_REVISION})",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--few-shot", action="store_true",
                        help="Stellt zwei ausgerechnete Beispiele voran. Trennt die Frage "
                             "'Faehigkeit verloren?' von 'nur der Antwortstil verschoben?'")
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.max_tokens <= 0:
        raise ValueError("max tokens must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    try:
        adapter_provenance = build_adapter_provenance(args.adapter_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    from datasets import load_dataset

    if args.endpoint:
        if args.adapter_path:
            parser.error("--adapter-path gilt nur fuer den lokalen MLX-Weg")
        print(f"Messe ueber {args.endpoint} (Modell {args.model})", flush=True)
        erzeuge = http_generator(
            args.endpoint,
            model=args.model,
            max_tokens=args.max_tokens,
            thinking_mode=args.thinking_mode,
            timeout=args.timeout,
        )
    else:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler

        print(f"Lade Modell: {args.model}", flush=True)
        mx.random.seed(args.seed)
        model, tokenizer = load_reasoning_model(load, args.model, args.adapter_path)
        sampler = make_sampler(temp=0.0)

        def erzeuge(messages: list[dict[str, str]]) -> tuple[str, bool]:
            prompt = apply_reasoning_chat_template(tokenizer, messages, args.thinking_mode)
            return generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                            sampler=sampler, verbose=False), False

    dataset_kwargs: dict[str, Any] = {
        "split": DATASET_SPLIT,
        "streaming": True,
    }
    if args.dataset_revision is not None:
        dataset_kwargs["revision"] = args.dataset_revision
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, **dataset_kwargs)
    treffer = 0
    gesamt = 0
    abgeschnitten = 0
    fehlerbeispiele = []

    for row in ds:
        if gesamt >= args.limit:
            break
        gesamt += 1
        frage = row["question"]
        soll = referenz(row["answer"])
        verlauf = list(FEW_SHOT) if args.few_shot else []
        verlauf.append({"role": "user", "content":
                        f"{frage}\n\n{PROMPT_SUFFIX}"})
        ausgabe, gekuerzt = erzeuge(verlauf)
        abgeschnitten += gekuerzt
        ist = letzte_zahl(ausgabe)
        ok = ist is not None and ist == soll
        treffer += ok
        if not ok and len(fehlerbeispiele) < 5:
            fehlerbeispiele.append({"frage": frage[:110], "soll": soll,
                                    "ist": ist, "abgeschnitten": gekuerzt,
                                    "ausgabe": ausgabe.strip()[-160:]})
        if gesamt % 25 == 0:
            print(f"  {gesamt}: {treffer}/{gesamt} = {100*treffer/gesamt:.1f}%", flush=True)

    quote = treffer / max(1, gesamt)
    print(f"\nGSM8K: {treffer}/{gesamt} = {100*quote:.1f}%")
    if abgeschnitten:
        # Diese Faelle messen das Token-Budget, nicht das Schlussfolgern.
        print(f"  davon {abgeschnitten} ohne Antwort, weil das Budget im Denkteil "
              f"aufgebraucht war — mit groesserem --max-tokens erneut messen")
    for f in fehlerbeispiele:
        print(f"  soll {f['soll']} / ist {f['ist']}: …{f['ausgabe'][-90:]!r}")

    if args.out:
        report = build_reasoning_report(
            args,
            correct=treffer,
            total=gesamt,
            failures=fehlerbeispiele,
            dataset_fingerprint=getattr(ds, "_fingerprint", None),
            adapter_provenance=adapter_provenance,
        )
        write_reasoning_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
