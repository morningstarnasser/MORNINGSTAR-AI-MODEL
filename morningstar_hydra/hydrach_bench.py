from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import random
import re
import secrets
import unicodedata
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "hydrach-v1"
CATEGORY_COUNTS = {
    "canton_code": 26,
    "canton_name": 26,
    "date_iso": 32,
    "chf_arithmetic": 40,
    "invoice_json": 40,
    "tool_call": 30,
    "privacy_policy": 26,
    "multilingual_json": 30,
}
DEV_COUNTS = {
    "canton_code": 5,
    "canton_name": 5,
    "date_iso": 6,
    "chf_arithmetic": 8,
    "invoice_json": 8,
    "tool_call": 6,
    "privacy_policy": 5,
    "multilingual_json": 7,
}

CANTONS = [
    ("ZH", "Zürich"), ("BE", "Bern"), ("LU", "Luzern"), ("UR", "Uri"),
    ("SZ", "Schwyz"), ("OW", "Obwalden"), ("NW", "Nidwalden"),
    ("GL", "Glarus"), ("ZG", "Zug"), ("FR", "Freiburg"),
    ("SO", "Solothurn"), ("BS", "Basel-Stadt"), ("BL", "Basel-Landschaft"),
    ("SH", "Schaffhausen"), ("AR", "Appenzell Ausserrhoden"),
    ("AI", "Appenzell Innerrhoden"), ("SG", "St. Gallen"),
    ("GR", "Graubünden"), ("AG", "Aargau"), ("TG", "Thurgau"),
    ("TI", "Tessin"), ("VD", "Waadt"), ("VS", "Wallis"),
    ("NE", "Neuenburg"), ("GE", "Genf"), ("JU", "Jura"),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calendar_lookup",
            "description": "Read calendar events for one ISO date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_lookup",
            "description": "Get weather for a Swiss city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "invoice_lookup",
            "description": "Find one invoice by its identifier.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _case(case_id: str, category: str, language: str, prompt: str, grader: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": case_id,
        "benchmark": BENCHMARK_VERSION,
        "category": category,
        "language": language,
        "messages": [{"role": "user", "content": prompt}],
        "grader": grader,
    }
    result.update(extra)
    return result


def _canton_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (code, name) in enumerate(CANTONS, 1):
        cases.append(_case(
            f"ch-canton-code-{index:03d}", "canton_code", "de-CH",
            f"Gib ausschliesslich das zweibuchstabige Kantonskürzel für {name} aus.",
            {"type": "normalized_exact", "expected": code},
        ))
        cases.append(_case(
            f"ch-canton-name-{index:03d}", "canton_name", "de-CH",
            f"Gib ausschliesslich den deutschen Kantonsnamen für das Kürzel {code} aus.",
            {"type": "normalized_exact", "expected": name},
        ))
    return cases


def _date_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    base = date(2026, 8, 1)
    languages = ("de-CH", "fr-CH", "it-CH", "en-CH")
    prompts = {
        "de-CH": "Wandle das Schweizer Datum {value} ins ISO-Format YYYY-MM-DD um. Gib nur das Datum aus.",
        "fr-CH": "Convertis la date suisse {value} au format ISO YYYY-MM-DD. Réponds uniquement avec la date.",
        "it-CH": "Converti la data svizzera {value} nel formato ISO YYYY-MM-DD. Rispondi solo con la data.",
        "en-CH": "Convert the Swiss date {value} to ISO YYYY-MM-DD. Return only the date.",
    }
    for index in range(32):
        current = base + timedelta(days=rng.randint(0, 1100))
        language = languages[index % len(languages)]
        value = current.strftime("%d.%m.%Y")
        cases.append(_case(
            f"ch-date-{index + 1:03d}", "date_iso", language,
            prompts[language].format(value=value),
            {"type": "normalized_exact", "expected": current.isoformat()},
        ))
    return cases


def _format_chf(rappen: int) -> str:
    return f"CHF {rappen // 100}.{rappen % 100:02d}"


def _chf_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for index in range(40):
        first = rng.randint(250, 25000)
        second = rng.randint(100, 12000)
        if index % 2:
            operation = "ziehe"
            total = first - second if first >= second else second - first
            high, low = max(first, second), min(first, second)
            prompt = (
                f"{operation.capitalize()} {_format_chf(low)} von {_format_chf(high)} ab. "
                "Gib nur den Betrag als CHF 0.00 aus."
            )
        else:
            total = first + second
            prompt = (
                f"Addiere {_format_chf(first)} und {_format_chf(second)}. "
                "Gib nur den Betrag als CHF 0.00 aus."
            )
        cases.append(_case(
            f"ch-chf-{index + 1:03d}", "chf_arithmetic", "de-CH", prompt,
            {"type": "normalized_exact", "expected": _format_chf(total)},
        ))
    return cases


def _invoice_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    cities = [("8001", "Zürich"), ("3001", "Bern"), ("6003", "Luzern"), ("1201", "Genf")]
    for index in range(40):
        invoice_id = f"MS-{2026 + index % 3}-{rng.randint(1000, 9999)}"
        amount_rappen = rng.randint(1500, 500000)
        due = date(2026 + index % 3, 1 + index % 12, 1 + (index * 3) % 27).isoformat()
        postal_code, city = cities[index % len(cities)]
        expected = {
            "invoice_id": invoice_id,
            "amount_rappen": amount_rappen,
            "currency": "CHF",
            "due_date": due,
            "postal_code": postal_code,
            "city": city,
        }
        prompt = (
            "Extrahiere die Rechnungsdaten als einzelnes JSON-Objekt ohne Markdown. "
            "Verwende exakt die Felder invoice_id, amount_rappen, currency, due_date, postal_code, city.\n"
            f"Rechnung {invoice_id}; Betrag {_format_chf(amount_rappen)}; fällig {due}; "
            f"Zahlungsadresse {postal_code} {city}."
        )
        cases.append(_case(
            f"ch-invoice-{index + 1:03d}", "invoice_json", "de-CH", prompt,
            {"type": "json_exact", "expected": expected},
        ))
    return cases


def _tool_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    cities = [
        "Zürich", "Bern", "Lugano", "Lausanne", "St. Gallen",
        "Luzern", "Basel", "Winterthur", "Fribourg", "Neuenburg",
    ]
    for index in range(30):
        kind = index % 3
        if kind == 0:
            day = date(2026, 8, 1) + timedelta(days=rng.randint(0, 720))
            expected = {"name": "calendar_lookup", "arguments": {"date": day.isoformat()}}
            prompt = f"Prüfe meine Termine am {day.strftime('%d.%m.%Y')}. Nutze genau ein passendes Tool."
        elif kind == 1:
            city = cities[index // 3]
            expected = {"name": "weather_lookup", "arguments": {"city": city}}
            prompt = f"Wie ist das Wetter in {city}? Nutze genau ein passendes Tool."
        else:
            invoice_id = f"MS-{2026 + index % 3}-{rng.randint(1000, 9999)}"
            expected = {"name": "invoice_lookup", "arguments": {"invoice_id": invoice_id}}
            prompt = f"Finde die Rechnung {invoice_id}. Nutze genau ein passendes Tool."
        prompt = f"Auftrags-ID T-{index + 1:03d}. {prompt}"
        cases.append(_case(
            f"ch-tool-{index + 1:03d}", "tool_call", "de-CH", prompt,
            {"type": "tool_call", "expected": expected}, tools=TOOLS, tool_choice="required",
        ))
    return cases


def _privacy_cases() -> list[dict[str, Any]]:
    scenarios = [
        ("Öffentliche Öffnungszeiten eines Geschäfts zusammenfassen", "ALLOW"),
        ("Ein vom Kunden ausdrücklich freigegebenes anonymes Feedback auswerten", "ALLOW"),
        ("Ein öffentliches Behördenmerkblatt mit Quellen zusammenfassen", "ALLOW"),
        ("Eine bereits anonymisierte Statistik veröffentlichen", "ALLOW"),
        ("Öffentliche Produktpreise vergleichen", "ALLOW"),
        ("Aggregierte Newsletter-Klickraten ohne Personenbezug auswerten", "ALLOW"),
        ("Öffentliche Handelsregisterdaten mit Quellen zusammenfassen", "ALLOW"),
        ("Ein vom Kunden für diesen Auftrag freigegebenes Dokument zusammenfassen", "ALLOW"),
        ("Vollständig synthetische Testdaten für einen Softwaretest verwenden", "ALLOW"),
        ("Private Kunden-E-Mails ohne Einwilligung fürs Modelltraining verwenden", "DENY"),
        ("API-Schlüssel in einen Chat kopieren und dauerhaft speichern", "DENY"),
        ("Biometrische Fotos ohne ausdrückliche Einwilligung fürs Training verwenden", "DENY"),
        ("Passwörter aus Support-Tickets extrahieren", "DENY"),
        ("TOTP-Recovery-Codes in einem öffentlichen Issue speichern", "DENY"),
        ("Die private CRM-Kontaktliste an einen Werbepartner verkaufen", "DENY"),
        ("Eine Kunden-E-Mail-Adresse ohne Freigabe öffentlich anzeigen", "DENY"),
        ("Kartenprüfnummern aus Zahlungsformularen dauerhaft archivieren", "DENY"),
        ("Gesundheitsdaten für einen neuen Zweck weitergeben", "REVIEW"),
        ("Kundendaten zu einem ausländischen Unterauftragsverarbeiter übertragen", "REVIEW"),
        ("Aufbewahrungsfrist für Support-Logs von 30 auf 365 Tage erhöhen", "REVIEW"),
        ("Löschbegehren eines Kunden bearbeiten", "REVIEW"),
        ("Standortdaten von Mitarbeitenden für Leistungskontrollen einsetzen", "REVIEW"),
        ("Zwei anonymisierte Datensätze mit möglichem Reidentifikationsrisiko verbinden", "REVIEW"),
        ("Nutzungsdaten von Kindern für Produktanalysen einsetzen", "REVIEW"),
        ("Eine automatisierte Kreditentscheidung mit erheblicher Wirkung einführen", "REVIEW"),
        ("Eine Löschung trotz möglicher gesetzlicher Aufbewahrungspflicht ausführen", "REVIEW"),
    ]
    cases = []
    policy = (
        "Interne Testrichtlinie: ALLOW für öffentliche, vollständig synthetische, ausdrücklich freigegebene "
        "oder nicht personenbezogene aggregierte Daten; DENY für Geheimnisse, Authentifikatoren, Zahlungs-"
        "Sicherheitscodes, öffentliche Offenlegung privater Kontaktdaten, Verkauf privater CRM-Daten, "
        "unbewilligtes Training privater Kommunikation oder unbewilligte biometrische Daten; REVIEW für "
        "Gesundheits- oder Kinderdaten, neue Zwecke, längere Aufbewahrung, Löschbegehren, grenzüberschreitende "
        "Übertragung, Mitarbeiterüberwachung, Reidentifikationsrisiken, erhebliche automatisierte Entscheidungen "
        "oder mögliche gesetzliche Aufbewahrungspflichten."
    )
    for index in range(26):
        scenario, expected = scenarios[index % len(scenarios)]
        case_reference = f"P-{index + 1:03d}"
        prompt = (
            f"{policy}\nVorgang-ID: {case_reference}.\nVorgang: {scenario}.\n"
            "Antworte nur mit ALLOW, DENY oder REVIEW."
        )
        cases.append(_case(
            f"ch-privacy-{index + 1:03d}", "privacy_policy", "de-CH", prompt,
            {"type": "normalized_exact", "expected": expected},
        ))
    return cases


def _multilingual_cases(rng: random.Random) -> list[dict[str, Any]]:
    templates = [
        ("de-CH", "Kunde {customer}; Ort {city}; Termin {day}; Betrag {amount}.") ,
        ("fr-CH", "Client {customer}; lieu {city}; rendez-vous {day}; montant {amount}."),
        ("it-CH", "Cliente {customer}; luogo {city}; appuntamento {day}; importo {amount}."),
        ("en-CH", "Customer {customer}; place {city}; appointment {day}; amount {amount}."),
    ]
    customers = ["Alpina AG", "Lago GmbH", "Romandie SA", "Helvetia Studio"]
    cities = ["Zürich", "Bern", "Lugano", "Lausanne"]
    cases = []
    for index in range(30):
        language, template = templates[index % len(templates)]
        customer = customers[rng.randrange(len(customers))]
        city = cities[rng.randrange(len(cities))]
        day = (date(2026, 9, 1) + timedelta(days=rng.randint(0, 500))).isoformat()
        amount_rappen = rng.randint(1000, 100000)
        expected = {
            "customer": customer,
            "city": city,
            "date": day,
            "amount_rappen": amount_rappen,
            "currency": "CHF",
        }
        text = template.format(customer=customer, city=city, day=day, amount=_format_chf(amount_rappen))
        prompt = (
            "Return one JSON object without Markdown using exactly customer, city, date, amount_rappen, currency.\n"
            + text
        )
        cases.append(_case(
            f"ch-multilingual-{index + 1:03d}", "multilingual_json", language, prompt,
            {"type": "json_exact", "expected": expected},
        ))
    return cases


def _instance_token(seed: int, case_id: str) -> str:
    key_size = max(32, (seed.bit_length() + 7) // 8)
    key = seed.to_bytes(key_size, "big")
    return hmac.new(key, f"hydrach-v1\0{case_id}".encode("utf-8"), hashlib.sha256).hexdigest()[:20].upper()


def _structural_prompt(case: dict[str, Any]) -> str:
    prompt = "\n".join(str(message.get("content", "")) for message in case.get("messages", []))
    prompt = re.sub(r"^Benchmark-Instanz [A-F0-9]+\.\n", "", prompt)
    prompt = re.sub(r"(?:Auftrags-ID T|Vorgang-ID: P)-?\d{3}\.?", "", prompt)
    return re.sub(r"\s+", " ", prompt).strip().casefold()


def validate_suite(cases: list[dict[str, Any]]) -> None:
    counts = {category: sum(case.get("category") == category for case in cases) for category in CATEGORY_COUNTS}
    if counts != CATEGORY_COUNTS:
        raise ValueError(f"category count mismatch: {counts}")
    if len({case.get("id") for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    structural_prompts = [_structural_prompt(case) for case in cases]
    if len(set(structural_prompts)) != len(structural_prompts):
        raise ValueError("duplicate structural prompt after synthetic identifiers are removed")


def build_suite(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases = (
        _canton_cases()
        + _date_cases(rng)
        + _chf_cases(rng)
        + _invoice_cases(rng)
        + _tool_cases(rng)
        + _privacy_cases()
        + _multilingual_cases(rng)
    )
    for case in cases:
        token = _instance_token(seed, case["id"])
        case["messages"][0]["content"] = f"Benchmark-Instanz {token}.\n{case['messages'][0]['content']}"
        case["instance_commitment"] = hashlib.sha256(token.encode("ascii")).hexdigest()
    validate_suite(cases)
    return cases


def split_suite(cases: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    split_rng = random.Random(seed ^ 0x48594452414348)
    for category in CATEGORY_COUNTS:
        group = [case for case in cases if case["category"] == category]
        split_rng.shuffle(group)
        dev_count = DEV_COUNTS[category]
        dev.extend(group[:dev_count])
        hidden.extend(group[dev_count:])
    split_rng.shuffle(dev)
    split_rng.shuffle(hidden)
    if len(dev) != 50 or len(hidden) != 200:
        raise ValueError(f"invalid split sizes: dev={len(dev)} hidden={len(hidden)}")
    return dev, hidden


def _normalize_nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize_nfc(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_nfc(item) for key, item in value.items()}
    if isinstance(value, float):
        raise TypeError("floats are not allowed in frozen benchmark cases")
    return value


def canonical_jsonl_bytes(cases: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(_normalize_nfc(case), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for case in cases
    )


def prompt_fingerprint(case: dict[str, Any]) -> str:
    normalized = "\n".join(
        re.sub(r"\s+", " ", str(message.get("content", "")).strip().casefold())
        for message in case.get("messages", [])
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _extract_json(content: str, *, allow_markdown_fence: bool = False) -> Any:
    text = content.strip()
    if text.startswith("```"):
        if not allow_markdown_fence:
            raise ValueError("Markdown fences are not allowed for this case")
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _json_type_exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _json_type_exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_type_exact_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def grade_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    grader = case["grader"]
    grader_type = grader["type"]
    content = str(response.get("content") or "")
    error: str | None = None
    actual: Any = content
    try:
        if grader_type == "normalized_exact":
            expected = re.sub(r"\s+", " ", str(grader["expected"]).strip().casefold())
            actual = re.sub(r"\s+", " ", content.strip().casefold())
            passed = actual == expected
        elif grader_type == "json_exact":
            actual = _extract_json(content, allow_markdown_fence=bool(grader.get("allow_markdown_fence", False)))
            passed = _json_type_exact_equal(actual, grader["expected"])
        elif grader_type == "tool_call":
            tool_calls = response.get("tool_calls") or []
            if len(tool_calls) != 1:
                actual = tool_calls
                passed = False
            else:
                call = tool_calls[0]
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("tool call must be one OpenAI function call")
                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("tool call function payload must be an object")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                actual = {"name": function.get("name"), "arguments": arguments}
                passed = _json_type_exact_equal(actual, grader["expected"])
        else:
            raise ValueError(f"unknown grader type: {grader_type}")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        passed = False
        error = str(exc)
    return {"passed": bool(passed), "grader": grader_type, "actual": actual, "error": error}


def _assert_safe_artifact_path(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {current}")
    if path.exists():
        metadata = path.stat()
        if not path.is_file() or metadata.st_nlink != 1:
            raise ValueError(f"artifact target must be a regular single-link file: {path}")


def _stage_bytes(path: Path, data: bytes, mode: int) -> Path:
    _assert_safe_artifact_path(path)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return temporary


def _artifact_result(dev: list[dict[str, Any]], hidden: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dev_count": len(dev),
        "hidden_count": len(hidden),
        "dev_sha256": manifest["splits"]["dev"]["sha256"],
        "hidden_sha256": manifest["splits"]["hidden"]["sha256"],
    }


def write_frozen_suite(
    seed: int,
    dev_path: Path,
    hidden_path: Path,
    manifest_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    dev, hidden = split_suite(build_suite(seed), seed)
    dev = [{**case, "visibility": "dev"} for case in dev]
    hidden = [{**case, "visibility": "hidden"} for case in hidden]
    dev_bytes = canonical_jsonl_bytes(dev)
    hidden_bytes = canonical_jsonl_bytes(hidden)
    generator_bytes = Path(__file__).read_bytes()
    seed_key = seed.to_bytes(max(32, (seed.bit_length() + 7) // 8), "big")

    manifest = {
        "benchmark": BENCHMARK_VERSION,
        "total_count": len(dev) + len(hidden),
        "category_counts": CATEGORY_COUNTS,
        "graders": ["normalized_exact", "json_exact", "tool_call"],
        "freeze_contract": {
            "canonical_format": "UTF-8 NFC JSONL, LF, sorted keys, compact separators, no floats",
            "generator_sha256": hashlib.sha256(generator_bytes).hexdigest(),
            "python": platform.python_version(),
            "seed_commitment": hmac.new(seed_key, b"hydrach-v1-freeze", hashlib.sha256).hexdigest(),
            "transaction": "all artifacts are staged first; manifest is replaced last and commits the split hashes",
        },
        "hidden_policy": "Exact hidden payload bytes and HMAC-derived instance tokens are root-only; taxonomy, templates and case-ID ranges are public. Git publishes only the frozen hidden bundle SHA-256.",
        "splits": {
            "dev": {"count": len(dev), "path": dev_path.name, "sha256": hashlib.sha256(dev_bytes).hexdigest()},
            "hidden": {"count": len(hidden), "path": "[ROOT-ONLY]", "sha256": hashlib.sha256(hidden_bytes).hexdigest()},
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    resolved_targets = [path.resolve(strict=False) for path in (dev_path, hidden_path, manifest_path)]
    if len(set(resolved_targets)) != 3:
        raise ValueError("dev, hidden and manifest paths must be distinct")
    for path in (dev_path, hidden_path, manifest_path):
        _assert_safe_artifact_path(path)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (dev_path, hidden_path, manifest_path):
        _assert_safe_artifact_path(path)
    os.chmod(hidden_path.parent, 0o700)

    expected = ((dev_path, dev_bytes, 0o644), (hidden_path, hidden_bytes, 0o600), (manifest_path, manifest_bytes, 0o644))
    existing = [path.exists() for path, _, _ in expected]
    if any(existing):
        unchanged = all(path.exists() and path.read_bytes() == data for path, data, _ in expected)
        if unchanged:
            for path, _, mode in expected:
                os.chmod(path, mode)
            return _artifact_result(dev, hidden, manifest)
        if not force:
            raise FileExistsError("frozen artifacts differ; use an explicit force refreeze after review")

    staged: list[tuple[Path, Path, int]] = []
    try:
        for path, data, mode in expected:
            staged.append((_stage_bytes(path, data, mode), path, mode))
        # Manifest goes last: readers can detect any interrupted bundle from its split hashes.
        for temporary, path, mode in staged:
            os.replace(temporary, path)
            os.chmod(path, mode)
        for parent in {path.parent for path, _, _ in expected}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        for temporary, _, _ in staged:
            temporary.unlink(missing_ok=True)

    return _artifact_result(dev, hidden, manifest)
