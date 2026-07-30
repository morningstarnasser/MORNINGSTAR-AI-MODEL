#!/usr/bin/env python3
"""Materialisiert den gate-freigegebenen Datenmix als MLX-LoRA-Trainingsdaten.

Portierung von `cloud/kaggle_qlora_apertus8b.ipynb` (Zelle 4) auf Apple Silicon:
identische Quellen, Gewichte, gepinnte Revisionen und Filter, aber Ausgabe als
ChatML-gerendertes JSONL fuer `mlx_lm.lora` statt HF-`Dataset` fuer den
CUDA-Trainer.

Der Datenmix stammt aus `config/hf-expert-mixture.json`; das Kontaminationsgate
dazu ist gruen (0 Treffer / 436565 Zeilen, Report in benchmarks/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

SEED = 42
VALID_FRACTION = 0.02

# (dataset, config, split, gewicht, gepinnte revision)
SOURCES = [
    ("nvidia/OpenCodeInstruct", "train", "train", 0.20,
     "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"),
    ("nvidia/OpenCodeReasoning", "split_0", "split_0", 0.18,
     "20a1ca19c0d050fe9057fc08339d6b370ec1c67a"),
    ("open-r1/OpenR1-Math-220k", "default", "train", 0.18,
     "e4e141ec9dea9f8326f4d347be56105859b2bd68"),
    ("open-thoughts/AgentTrove", "default", "train", 0.24,
     "b395a4307a2bc9950a90dc899438f149e115fc60"),
    ("NousResearch/hermes-function-calling-v1", "func_calling", "train", 0.12,
     "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
    ("NousResearch/hermes-function-calling-v1", "json_mode_agentic", "train", 0.08,
     "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"),
]

ROLE_MAP = {
    "human": "user", "gpt": "assistant", "system": "system",
    "tool": "tool", "user": "user", "assistant": "assistant",
}


def to_messages(dataset: str, row: dict) -> list[dict]:
    """Normalisiert eine Quellzeile auf die Chat-Rollenstruktur."""
    if dataset in ("nvidia/OpenCodeInstruct", "nvidia/OpenCodeReasoning"):
        return [{"role": "user", "content": row.get("input", "")},
                {"role": "assistant", "content": row.get("output", "")}]
    if dataset == "open-r1/OpenR1-Math-220k":
        return [{"role": "user", "content": row.get("problem", "")},
                {"role": "assistant", "content": row.get("solution", "")}]

    conversation = row.get("conversations") or row.get("messages") or []
    messages = []
    for turn in conversation:
        role = turn.get("role") or ROLE_MAP.get(turn.get("from"), "user")
        if role not in ("user", "assistant", "system", "tool"):
            role = "user"
        messages.append({"role": role, "content": turn.get("content") or turn.get("value") or ""})
    return messages


def passes(dataset: str, row: dict) -> bool:
    """Wendet die im Mixture-Manifest deklarierten Qualitaetsfilter an."""
    if dataset == "nvidia/OpenCodeInstruct":
        try:
            score = float(row.get("average_test_score") or 0)
        except (TypeError, ValueError):
            return False
        return score >= 0.9 and bool(row.get("input")) and bool(row.get("output"))

    if dataset == "nvidia/OpenCodeReasoning":
        return bool(row.get("input")) and bool(row.get("output"))

    if dataset == "open-r1/OpenR1-Math-220k":
        try:
            count = int(row.get("correctness_count") or 0)
        except (TypeError, ValueError):
            return False
        complete = row.get("is_reasoning_complete")
        if isinstance(complete, (list, tuple)):
            reasoning_ok = True in list(complete)
        else:
            reasoning_ok = True if complete is None else bool(complete)
        return count >= 1 and reasoning_ok

    return bool(row.get("conversations") or row.get("messages"))


def is_wellformed(messages: list[dict]) -> bool:
    """Verlangt mindestens einen User- und einen nicht-leeren Assistant-Turn."""
    if len(messages) < 2:
        return False
    roles = {m["role"] for m in messages}
    if not {"user", "assistant"} <= roles:
        return False
    return any(m["role"] == "assistant" and m["content"].strip() for m in messages)


def fingerprint(messages: list[dict]) -> str:
    """sha256 ueber normalisierten User/Assistant-Inhalt (Dedup-Regel des Manifests)."""
    parts = []
    for message in messages:
        if message["role"] not in ("user", "assistant"):
            continue
        normalized = re.sub(r"\s+", " ", message["content"]).strip().lower()
        parts.append(f"{message['role']}:{normalized}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def render_chatml(messages: list[dict]) -> str:
    """Rendert ChatML — Fallback fuer Basismodelle ohne eigene Rollen-Tokens."""
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def render_apertus(messages: list[dict]) -> str:
    """Rendert Apertus' native Rollen-Tokens.

    Apertus-8B-2509 bringt kein chat_template mit, hat aber reservierte
    Einzeltoken-Marker (`<|user_start|>` usw.). Die kosten ein Token statt der
    sechs, in die ChatML zerfaellt, und liefern ein verlaessliches Stop-Signal.
    """
    parts = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "tool":
            parts.append(f"[TOOL_RESULTS]{content}[/TOOL_RESULTS]")
        else:
            parts.append(f"<|{role}_start|>{content}<|{role}_end|>")
    return "".join(parts) + "</s>"


def sanitize_for_template(messages: list[dict]) -> list[dict]:
    """Faltet Rollen weg, die das Modell-Template nicht kennt.

    Apertus' chat_template lehnt die Rolle `tool` ab (es erwartet Tool-Ergebnisse
    in einer eigenen Markerstruktur). Ohne diese Faltung faellt die komplette
    hermes-func_calling-Quelle durch. Das Ergebnis wird als Beobachtung im
    User-Turn uebergeben — nicht das native Tool-Format, aber verwertbares Signal.
    """
    folded = []
    for message in messages:
        if message["role"] == "tool":
            folded.append({"role": "user",
                           "content": f"[TOOL_RESULTS]\n{message['content']}"})
        else:
            folded.append(message)

    # Aufeinanderfolgende gleiche Rollen zusammenfassen — Templates verlangen Alternanz.
    merged: list[dict] = []
    for message in folded:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1] = {"role": message["role"],
                          "content": f"{merged[-1]['content']}\n\n{message['content']}"}
        else:
            merged.append(dict(message))
    return merged


def make_hf_renderer(tokenizer):
    """Rendert mit dem chat_template des Modells selbst.

    Vorzuziehen, sobald das Modell ein Template mitbringt: Apertus-Instruct
    setzt ein bos-Token und schiebt immer einen `developer`-Block
    ("Deliberation"/"Tool Capabilities") ein — ein handgebautes Format wuerde
    daran vorbeitrainieren.
    """
    def render(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(sanitize_for_template(messages),
                                             tokenize=False)
    return render


RENDERERS = {"apertus": render_apertus, "chatml": render_chatml, "hf": None}


def collect(total_examples: int, max_chars: int, render, token_filter=None) -> list[dict]:
    """Streamt jede Quelle bis zu ihrem gewichteten Kontingent.

    `token_filter` verwirft Beispiele, die nicht vollstaendig ins Sequenzbudget
    passen. Ohne diesen Filter schneidet der Trainer sie mittendrin ab — dann
    fehlt das Turn-Ende und das Modell lernt, nicht zu terminieren.
    """
    seen: set[str] = set()
    collected: list[dict] = []
    dropped_too_long = 0

    for dataset, config, split, weight, revision in SOURCES:
        cap = max(1, int(total_examples * weight))
        kwargs = {"streaming": True}
        if revision:
            kwargs["revision"] = revision

        taken = 0
        scanned = 0
        stream = load_dataset(dataset, config, split=split, **kwargs)
        for row in stream:
            scanned += 1
            if taken >= cap or scanned > cap * 60:
                break
            if not passes(dataset, row):
                continue
            messages = to_messages(dataset, row)
            if not is_wellformed(messages):
                continue

            text = render(messages)
            if text is None:          # Template lehnt diese Rollenfolge ab
                continue
            if len(text) > max_chars:
                continue

            key = fingerprint(messages)
            if key in seen:
                continue

            if token_filter is not None and not token_filter(text):
                dropped_too_long += 1
                continue
            seen.add(key)

            # Fuer das Paarformat: erster User-Turn als Frage, letzter Assistant-Turn
            # als Antwort. Bei Math/Code sind das Single-Turn-Dialoge, es geht also
            # nichts verloren — und die langen Rechenwege bleiben erhalten.
            erste_frage = next((m["content"] for m in messages if m["role"] == "user"), "")
            letzte_antwort = next((m["content"] for m in reversed(messages)
                                   if m["role"] == "assistant"), "")

            collected.append({"text": text, "_source": dataset, "_config": config,
                              "_fp": key, "prompt": erste_frage,
                              "completion": letzte_antwort})
            taken += 1

        print(f"  {dataset} [{config}]: {taken}/{cap} (gescannt {scanned})", flush=True)

    if dropped_too_long:
        print(f"  [Sequenzbudget] {dropped_too_long} Beispiele verworfen, weil zu lang",
              flush=True)
    return collected


def write_split(rows: list[dict], out_dir: Path, als_paar: bool = False) -> dict:
    """Schreibt train.jsonl/valid.jsonl und gibt die Statistik zurueck.

    `als_paar` schreibt {"prompt", "completion"} statt {"text"}. Nur so greift
    mlx-lms `mask_prompt`, und nur so laesst sich dieser Datensatz mit den
    CH-Fachdaten mischen, die dasselbe Format nutzen.
    """
    random.Random(SEED).shuffle(rows)
    split_at = max(1, int(len(rows) * VALID_FRACTION))
    valid, train = rows[:split_at], rows[split_at:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("train", train), ("valid", valid)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in chunk:
                if als_paar:
                    if not row.get("prompt") or not row.get("completion"):
                        continue
                    nutzlast = {"prompt": row["prompt"], "completion": row["completion"]}
                else:
                    nutzlast = {"text": row["text"]}
                handle.write(json.dumps(nutzlast, ensure_ascii=False) + "\n")

    per_source: dict[str, int] = {}
    for row in rows:
        per_source[f"{row['_source']}::{row['_config']}"] = \
            per_source.get(f"{row['_source']}::{row['_config']}", 0) + 1

    return {"train": len(train), "valid": len(valid), "per_source": per_source}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(rows: list[dict], stats: dict, args) -> None:
    """Schreibt Revisionen, Datei-Hashes und Sample-IDs fuer das Trainings-Gate.

    `config/hf-expert-mixture.json` verlangt unter required_before_training
    unveraenderliche Quell-Revisionen und Sample-Identifikatoren — das hier ist
    der Nachweis fuer genau diesen Lauf.
    """
    out_dir: Path = args.out_dir
    (out_dir / "sample_fingerprints.txt").write_text(
        "\n".join(row["_fp"] for row in rows) + "\n", encoding="utf-8")

    manifest = {
        "template": args.template,
        "seed": SEED,
        "max_tokens": args.max_tokens or None,
        "total_requested": args.total_examples,
        "total_collected": len(rows),
        "counts": {"train": stats["train"], "valid": stats["valid"]},
        "per_source": stats["per_source"],
        "sources": [
            {"dataset": d, "config": c, "split": s, "weight": w, "revision": r}
            for d, c, s, w, r in SOURCES
        ],
        "files": {
            name: sha256_file(out_dir / name)
            for name in ("train.jsonl", "valid.jsonl", "sample_fingerprints.txt")
        },
        "contamination_gate": {
            "status": "verified_clean",
            "report": "benchmarks/contamination-full-20260724.json",
            "note": "Quellen und Filter sind identisch zum geprueften Mix; "
                    "dieser Lauf zieht nur eine Teilmenge daraus.",
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-examples", type=int, default=24000)
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="Zeilen laenger als dies verwerfen (haelt Sequenzen im Budget)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "hydra-train" / "data")
    parser.add_argument("--pair-format", action="store_true",
                        help="Schreibt {prompt, completion} statt {text} — noetig fuer\n                             mask_prompt und zum Mischen mit den CH-Fachdaten")
    parser.add_argument("--template", choices=sorted(RENDERERS), default="apertus",
                        help="Rollen-Format; 'apertus' nutzt die nativen Einzeltoken-Marker")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Beispiele ueber diesem Tokenbudget verwerfen statt abschneiden "
                             "(0 = aus). Braucht --tokenizer.")
    parser.add_argument("--tokenizer", type=str,
                        default=str(Path.home() / "hydra-models" / "apertus-8b-4bit"))
    args = parser.parse_args()

    tokenizer = None
    if args.template == "hf" or args.max_tokens > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    token_filter = None
    if args.max_tokens > 0:
        limit = args.max_tokens

        def token_filter(text: str) -> bool:
            return len(tokenizer.encode(text, add_special_tokens=False)) <= limit

        print(f"Tokenbudget aktiv: <= {limit} Tokens ({args.tokenizer})", flush=True)

    if args.template == "hf":
        base_render = make_hf_renderer(tokenizer)

        def render(messages: list[dict]):
            # Rollenfolgen, die das Template ablehnt, werden verworfen statt zu crashen.
            try:
                return base_render(messages)
            except Exception:
                return None
    else:
        render = RENDERERS[args.template]

    print(f"Sammle {args.total_examples} Beispiele aus {len(SOURCES)} gate-freigegebenen Quellen "
          f"(Template: {args.template})", flush=True)
    rows = collect(args.total_examples, args.max_chars, render, token_filter)
    if not rows:
        print("FEHLER: keine Beispiele gesammelt", file=sys.stderr)
        return 1

    stats = write_split(rows, args.out_dir, args.pair_format)
    stats["template"] = args.template
    stats["seed"] = SEED
    (args.out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(rows, stats, args)

    print(f"\nGeschrieben nach {args.out_dir}: "
          f"{stats['train']} train / {stats['valid']} valid")
    for source, count in sorted(stats["per_source"].items()):
        print(f"  {source}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
