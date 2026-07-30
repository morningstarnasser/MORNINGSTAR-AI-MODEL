#!/usr/bin/env python3
"""Rechnet die Antworten aus build_ch_dataset.py unabhaengig nach.

Ein Generatorfehler waere schlimmer als gar kein Training — das Modell wuerde
den Fehler systematisch lernen. Dieser Pruefer liest nur Prompt und Antwort und
leitet die Sollantwort neu her, ohne den Generatorcode zu benutzen.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_ch_dataset import KANTONE  # noqa: E402

CODE_TO_DE = {c: de for c, de, _f, _i in KANTONE}
NAME_TO_CODE: dict[str, str] = {}
for code, de, fr, it in KANTONE:
    for name in (de, fr, it):
        NAME_TO_CODE[name] = code


PRAEFIX = re.compile(
    r"^(Vorgang|Referenz|Ticket|Datensatz|Laufnummer|Sitzung)\s+\S+\.\s*\n", re.I)


def ohne_praefix(prompt: str) -> str:
    """Entfernt die absichtlich irrelevante Kopfzeile vor der Pruefung.

    Sie gehoert nicht zur Aufgabe — genau das soll das Modell lernen — und wuerde
    hier sonst als Kantonskuerzel oder Aufgabenbeginn fehlgedeutet.
    """
    return PRAEFIX.sub("", prompt, count=1)


def q2(value: Decimal) -> Decimal:
    """Kaufmaennisch runden — nicht Pythons round(), das half-to-even rundet."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def check_canton_code(prompt: str, answer: str) -> str | None:
    # Direkte Namenssuche statt Regex-Zerlegung: "St. Gallen" enthaelt selbst einen
    # Punkt, an dem jede naive Trennung zerbricht. Laengster Treffer gewinnt, damit
    # "Appenzell Innerrhoden" nicht als "Appenzell" durchgeht.
    treffer = [name for name in NAME_TO_CODE if name in prompt]
    if not treffer:
        return f"kein Kantonsname im Prompt: {prompt[:70]!r}"
    name = max(treffer, key=len)
    erwartet = NAME_TO_CODE[name]
    return None if answer == erwartet else f"{name}: {answer} != {erwartet}"


def check_canton_name(prompt: str, answer: str) -> str | None:
    match = re.search(r"\b([A-Z]{2})\b", prompt)
    if not match:
        return "kein Kuerzel im Prompt"
    erwartet = CODE_TO_DE.get(match.group(1))
    if erwartet is None:
        return f"Kuerzel unbekannt: {match.group(1)}"
    return None if answer == erwartet else f"{match.group(1)}: {answer} != {erwartet}"


def check_canton_contrast(prompt: str, answer: str) -> str | None:
    """Bei Kontrastaufgaben ist das erste genannte Kuerzel das gefragte."""
    codes = re.findall(r"\b([A-Z]{2})\b", prompt)
    if not codes:
        return f"kein Kuerzel im Prompt: {prompt[:70]!r}"
    gefragt = codes[0]
    if re.fullmatch(r"[A-Z]{2}", answer):
        return None if answer == gefragt else f"{prompt[:50]!r}: {answer} != {gefragt}"
    soll = CODE_TO_DE.get(gefragt)
    return None if answer == soll else f"{prompt[:50]!r}: {answer} != {soll}"


def check_chf(prompt: str, answer: str) -> str | None:
    betraege = [Decimal(x) for x in re.findall(r"CHF\s+([0-9]+\.[0-9]{2})", prompt)]
    got = re.match(r"^CHF ([0-9]+\.[0-9]{2})$", answer)
    if not got:
        return f"Antwortformat falsch: {answer!r}"
    ist = Decimal(got.group(1))

    if prompt.startswith("Addiere") and len(betraege) >= 2:
        soll = betraege[0] + betraege[1]
    elif prompt.startswith("Ziehe") and "Rabatt" in prompt:
        pct = Decimal(re.search(r"Ziehe (\d+)% Rabatt", prompt).group(1))
        soll = betraege[0] * (1 - pct / 100)
    elif prompt.startswith("Ziehe") and len(betraege) >= 2:
        soll = betraege[1] - betraege[0]
    elif prompt.startswith("Multipliziere"):
        n = Decimal(re.search(r"mit (\d+)", prompt).group(1))
        soll = betraege[0] * n
    elif prompt.startswith("Berechne"):
        satz = Decimal(re.search(r"Berechne ([0-9.]+)% MWST", prompt).group(1))
        soll = betraege[0] * (1 + satz / 100)
    elif prompt.startswith("Teile"):
        n = Decimal(re.search(r"durch (\d+)", prompt).group(1))
        soll = betraege[0] / n
    else:
        return f"Rechenart nicht erkannt: {prompt[:50]!r}"

    soll = q2(soll)
    if ist != soll:
        return f"{prompt[:60]!r}: {ist} != {soll}"
    return None


def check_date(prompt: str, answer: str) -> str | None:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", answer):
        return f"kein ISO-Format: {answer!r}"
    match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", prompt)
    if match:
        tag, monat, jahr = match.groups()
        soll = f"{jahr}-{monat}-{tag}"
    else:
        from build_ch_dataset import MONATE_DE
        m2 = re.search(r"(\d{1,2})\.\s+(\w+)\s+(\d{4})", prompt)
        if not m2:
            return "Datum im Prompt nicht gefunden"
        tag, monatname, jahr = m2.groups()
        if monatname not in MONATE_DE:
            return f"Monat unbekannt: {monatname}"
        soll = f"{jahr}-{MONATE_DE.index(monatname)+1:02d}-{int(tag):02d}"
    return None if answer == soll else f"{answer} != {soll}"


def check_json(prompt: str, answer: str, felder: set[str]) -> str | None:
    try:
        obj = json.loads(answer)
    except json.JSONDecodeError as exc:
        return f"kein valides JSON: {exc}"
    if set(obj) != felder:
        return f"Felder falsch: {sorted(obj)} != {sorted(felder)}"
    if not isinstance(obj["amount_rappen"], int):
        return f"amount_rappen ist {type(obj['amount_rappen']).__name__}, kein int"
    match = re.search(r"CHF\s+([0-9]+\.[0-9]{2})", prompt)
    if not match:
        return "Betrag im Prompt nicht gefunden"
    soll = int(q2(Decimal(match.group(1)) * 100))
    if obj["amount_rappen"] != soll:
        return f"amount_rappen {obj['amount_rappen']} != {soll}"
    if obj.get("currency") != "CHF":
        return f"currency {obj.get('currency')!r}"
    return None


INVOICE_FELDER = {"invoice_id", "amount_rappen", "currency", "due_date", "postal_code", "city"}
MULTI_FELDER = {"customer", "city", "date", "amount_rappen", "currency"}


def main() -> int:
    pfad = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "hydra-train" / "ch-raw.jsonl"
    fehler: Counter = Counter()
    beispiele: dict[str, str] = {}
    gesamt = Counter()

    for zeile in pfad.open(encoding="utf-8"):
        row = json.loads(zeile)
        kat, antwort = row["category"], row["completion"]
        prompt = ohne_praefix(row["prompt"])
        gesamt[kat] += 1
        if kat == "canton_code":
            problem = check_canton_code(prompt, antwort)
        elif kat == "canton_name":
            problem = check_canton_name(prompt, antwort)
        elif kat == "canton_contrast":
            problem = check_canton_contrast(prompt, antwort)
        elif kat == "chf_arithmetic":
            problem = check_chf(prompt, antwort)
        elif kat == "date_iso":
            problem = check_date(prompt, antwort)
        elif kat == "invoice_json":
            problem = check_json(prompt, antwort, INVOICE_FELDER)
        elif kat == "multilingual_json":
            problem = check_json(prompt, antwort, MULTI_FELDER)
        elif kat == "privacy_policy":
            problem = None if antwort in ("ALLOW", "DENY", "REVIEW") else f"Label {antwort!r}"
        else:
            problem = f"unbekannte Kategorie {kat}"

        if problem:
            fehler[kat] += 1
            beispiele.setdefault(kat, problem)

    print(f"Geprueft: {sum(gesamt.values())} Beispiele aus {pfad}\n")
    for kat in sorted(gesamt):
        n = fehler[kat]
        marke = "OK " if n == 0 else "FEHLER"
        print(f"  {marke} {kat:20s} {gesamt[kat]-n}/{gesamt[kat]} korrekt")
        if n:
            print(f"       Beispiel: {beispiele[kat]}")
    total = sum(fehler.values())
    print(f"\n{'ALLE KORREKT' if total == 0 else str(total) + ' FEHLERHAFTE BEISPIELE'}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
