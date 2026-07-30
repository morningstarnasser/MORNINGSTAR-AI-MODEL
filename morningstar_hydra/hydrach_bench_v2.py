"""HydraCH benchmark v2.

Successor instrument to :mod:`morningstar_hydra.hydrach_bench`. v1 is left untouched;
frozen v1 artifacts stay reproducible.

Why v2 exists
-------------
At n=50 dev the standard error is +/-4.6 pp, so differences below ~9 pp are not
resolvable. v2 raises both splits, adds real negative cases for ``tool_call`` and
decomposes the tool metric so a runtime/parser defect is no longer scored as a
capability gap.

Changes against v1 that follow from the label audit (2026-07-26)
----------------------------------------------------------------
1. ``privacy_policy`` states an explicit precedence rule (DENY > REVIEW > ALLOW).
   In v1 a scenario could match two buckets (payment security code *and* extended
   retention) with no documented tie-break, which made a handful of gold labels a
   coin flip rather than a measurement.
2. JSON categories declare a type contract in the prompt. The grader compares
   types strictly (``_json_type_exact_equal``), but v1 never told the model that
   ``postal_code`` is a string and ``amount_rappen`` an integer. That is scored
   capability loss caused by an unstated contract.
3. All tool cases use ``tool_choice="auto"``. v1 forced ``required``, which makes
   a false-positive rate unmeasurable: the model was never allowed to abstain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import random
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from morningstar_hydra.hydrach_bench import (
    CANTONS,
    TOOLS,
    _assert_safe_artifact_path,
    _extract_json,
    _format_chf,
    _json_type_exact_equal,
    _stage_bytes,
)

BENCHMARK_VERSION = "hydrach-v2"

# 8 categories x 60 cases = 480. tool_call splits into 36 positive / 24 negative.
CATEGORY_COUNTS = {
    "canton_code": 60,
    "canton_name": 60,
    "date_iso": 60,
    "chf_arithmetic": 60,
    "invoice_json": 60,
    "tool_call": 60,
    "privacy_policy": 60,
    "multilingual_json": 60,
}

# Split strata are (category, subtype) so the positive/negative tool ratio is
# identical in dev and hidden. 160 dev / 320 hidden.
DEV_COUNTS = {
    ("canton_code", "positive"): 20,
    ("canton_name", "positive"): 20,
    ("date_iso", "positive"): 20,
    ("chf_arithmetic", "positive"): 20,
    ("invoice_json", "positive"): 20,
    ("multilingual_json", "positive"): 20,
    ("privacy_policy", "positive"): 20,
    ("tool_call", "positive"): 12,
    ("tool_call", "negative"): 8,
}
STRATUM_COUNTS = {
    ("canton_code", "positive"): 60,
    ("canton_name", "positive"): 60,
    ("date_iso", "positive"): 60,
    ("chf_arithmetic", "positive"): 60,
    ("invoice_json", "positive"): 60,
    ("multilingual_json", "positive"): 60,
    ("privacy_policy", "positive"): 60,
    ("tool_call", "positive"): 36,
    ("tool_call", "negative"): 24,
}
DEV_TOTAL = sum(DEV_COUNTS.values())
HIDDEN_TOTAL = sum(STRATUM_COUNTS.values()) - DEV_TOTAL


def _case(
    case_id: str,
    category: str,
    language: str,
    prompt: str,
    grader: dict[str, Any],
    *,
    subtype: str = "positive",
    difficulty: str = "normal",
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": case_id,
        "benchmark": BENCHMARK_VERSION,
        "category": category,
        "subtype": subtype,
        "difficulty": difficulty,
        "language": language,
        "messages": [{"role": "user", "content": prompt}],
        "grader": grader,
    }
    result.update(extra)
    return result


# --------------------------------------------------------------------------
# canton categories
# --------------------------------------------------------------------------

_CODE_TEMPLATES = (
    "Welches amtliche Kürzel trägt der Kanton {name}? Antworte nur mit den zwei Grossbuchstaben.",
    "Notiere für {name} das offizielle Kantonskennzeichen. Ausgabe: ausschliesslich zwei Buchstaben.",
    "Trage im Formularfeld «Kanton» den Code von {name} ein. Nenne nur den Code.",
    "Für eine Adressierung nach {name}: nenne das zweistellige Kantonskürzel, sonst nichts.",
)

_NAME_TEMPLATES = (
    "Das Kürzel {code} steht für welchen Kanton? Antworte nur mit dem deutschen Namen.",
    "Löse die Abkürzung {code} auf. Ausgabe: nur die deutsche Kantonsbezeichnung.",
    "In einer Adressliste steht {code}. Schreibe den Kanton auf Deutsch aus, sonst nichts.",
    "Ergänze zum Code {code} den ausgeschriebenen deutschen Kantonsnamen. Keine weiteren Wörter.",
)


def _canton_cases(rng: random.Random) -> list[dict[str, Any]]:
    """60 code cases and 60 name cases drawn from distinct (canton, template) pairs."""
    cases: list[dict[str, Any]] = []
    for category, templates, field in (
        ("canton_code", _CODE_TEMPLATES, "code"),
        ("canton_name", _NAME_TEMPLATES, "name"),
    ):
        pairs = [(index, variant) for variant in range(len(templates)) for index in range(len(CANTONS))]
        rng.shuffle(pairs)
        for position, (index, variant) in enumerate(pairs[:60], 1):
            code, name = CANTONS[index]
            prompt = templates[variant].format(code=code, name=name)
            expected = code if field == "code" else name
            cases.append(_case(
                f"v2-{category.replace('_', '-')}-{position:03d}", category, "de-CH", prompt,
                {"type": "normalized_exact", "expected": expected},
            ))
    return cases


# --------------------------------------------------------------------------
# date_iso -- written-out month names instead of v1's dotted numeric input
# --------------------------------------------------------------------------

_MONTHS = {
    "de-CH": ("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
              "August", "September", "Oktober", "November", "Dezember"),
    "fr-CH": ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
              "août", "septembre", "octobre", "novembre", "décembre"),
    "it-CH": ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
              "agosto", "settembre", "ottobre", "novembre", "dicembre"),
    "en-CH": ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"),
}

_DATE_PROMPTS = {
    "de-CH": "Normalisiere die Datumsangabe {value} nach ISO 8601. Ausgabe: nur YYYY-MM-DD.",
    "fr-CH": "Normalise la date {value} selon ISO 8601. Retourne uniquement YYYY-MM-DD.",
    "it-CH": "Normalizza la data {value} secondo ISO 8601. Restituisci solo YYYY-MM-DD.",
    "en-CH": "Normalise the date {value} to ISO 8601. Output only YYYY-MM-DD.",
}

_DATE_RENDERERS = {
    "de-CH": lambda day, month: f"{day}. {month}",
    "fr-CH": lambda day, month: f"{day} {month}",
    "it-CH": lambda day, month: f"{day} {month}",
    "en-CH": lambda day, month: f"{day} {month}",
}


def _date_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    base = date(2027, 1, 1)
    languages = tuple(_DATE_PROMPTS)
    seen: set[tuple[str, str]] = set()
    position = 0
    while position < 60:
        language = languages[position % len(languages)]
        current = base + timedelta(days=rng.randint(0, 1400))
        key = (language, current.isoformat())
        if key in seen:
            continue
        seen.add(key)
        position += 1
        month = _MONTHS[language][current.month - 1]
        value = f"{_DATE_RENDERERS[language](current.day, month)} {current.year}"
        cases.append(_case(
            f"v2-date-{position:03d}", "date_iso", language,
            _DATE_PROMPTS[language].format(value=value),
            {"type": "normalized_exact", "expected": current.isoformat()},
        ))
    return cases


# --------------------------------------------------------------------------
# chf_arithmetic -- integer rappen only, commercial rounding stated in the prompt
# --------------------------------------------------------------------------

def _round_half_up(numerator: int, denominator: int) -> int:
    return (numerator * 2 + denominator) // (denominator * 2)


def _chf_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for position in range(1, 61):
        variant = position % 4
        if variant == 0:
            parts = [rng.randint(150, 40000) for _ in range(3)]
            total = sum(parts)
            rendered = ", ".join(_format_chf(part) for part in parts[:-1])
            prompt = (
                f"Eine Rechnung listet {rendered} und {_format_chf(parts[-1])}. "
                "Nenne die Endsumme im Format CHF 0.00, ohne weiteren Text."
            )
        elif variant == 1:
            net = rng.randint(500, 90000)
            total = _round_half_up(net * 1081, 1000)
            prompt = (
                f"Nettobetrag {_format_chf(net)}. Schlage 8.1 % Mehrwertsteuer auf und nenne den "
                "Bruttobetrag als CHF 0.00. Kaufmännisch auf Rappen runden, sonst nichts ausgeben."
            )
        elif variant == 2:
            listed = rng.randint(2000, 120000)
            percent = rng.choice((5, 10, 15, 20, 25))
            total = listed - _round_half_up(listed * percent, 100)
            prompt = (
                f"Vom Listenpreis {_format_chf(listed)} werden {percent} % Rabatt abgezogen. "
                "Nenne den Endpreis als CHF 0.00, kaufmännisch auf Rappen gerundet und ohne Erklärung."
            )
        else:
            invoiced = rng.randint(5000, 150000)
            paid = rng.randint(100, invoiced - 100)
            total = invoiced - paid
            prompt = (
                f"Von {_format_chf(invoiced)} wurden {_format_chf(paid)} bereits beglichen. "
                "Nenne den offenen Restbetrag als CHF 0.00, ohne Begründung."
            )
        cases.append(_case(
            f"v2-chf-{position:03d}", "chf_arithmetic", "de-CH", prompt,
            {"type": "normalized_exact", "expected": _format_chf(total)},
        ))
    return cases


# --------------------------------------------------------------------------
# invoice_json -- same field contract as v1, new wording, explicit types
# --------------------------------------------------------------------------

_INVOICE_TYPE_CONTRACT = (
    "Typvertrag: amount_rappen ist eine Ganzzahl in Rappen, alle übrigen Werte sind Zeichenketten."
)

_INVOICE_BODIES = (
    "Beleg-Auszug:\nNummer: {invoice_id}\nSumme: {amount}\nZahlbar bis: {due}\nRechnungsort: {postal_code} {city}",
    "Aus dem Mail: «Für {invoice_id} bitten wir um Begleichung von {amount} bis spätestens {due}. "
    "Unsere Zahlstelle liegt in {postal_code} {city}.»",
    "Kopfzeile des PDF: {invoice_id} | {amount} | Fälligkeit {due} | {postal_code} {city}",
)

_INVOICE_PLACES = (
    ("8400", "Winterthur"), ("7000", "Chur"), ("1950", "Sion"),
    ("6500", "Bellinzona"), ("3600", "Thun"), ("4600", "Olten"),
)


def _invoice_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for position in range(1, 61):
        invoice_id = f"RG{rng.randint(100000, 999999)}-{rng.choice('ABCDEFGH')}"
        amount_rappen = rng.randint(1500, 900000)
        due = (date(2027, 1, 1) + timedelta(days=rng.randint(0, 900))).isoformat()
        postal_code, city = _INVOICE_PLACES[position % len(_INVOICE_PLACES)]
        body = _INVOICE_BODIES[position % len(_INVOICE_BODIES)].format(
            invoice_id=invoice_id, amount=_format_chf(amount_rappen),
            due=due, postal_code=postal_code, city=city,
        )
        prompt = (
            "Bilde aus dem folgenden Beleg ein einziges JSON-Objekt ohne Markdown-Zäunen. "
            "Schlüssel exakt: invoice_id, amount_rappen, currency, due_date, postal_code, city. "
            f"{_INVOICE_TYPE_CONTRACT}\n{body}"
        )
        cases.append(_case(
            f"v2-invoice-{position:03d}", "invoice_json", "de-CH", prompt,
            {"type": "json_exact", "expected": {
                "invoice_id": invoice_id,
                "amount_rappen": amount_rappen,
                "currency": "CHF",
                "due_date": due,
                "postal_code": postal_code,
                "city": city,
            }},
        ))
    return cases


# --------------------------------------------------------------------------
# multilingual_json -- instruction in the case language (v1 used English throughout)
# --------------------------------------------------------------------------

_MULTILINGUAL_INSTRUCTIONS = {
    "de-CH": (
        "Bilde genau ein JSON-Objekt ohne Markdown. Schlüssel: customer, city, date, amount_rappen, currency. "
        "amount_rappen ist eine Ganzzahl in Rappen, die übrigen Werte sind Zeichenketten."
    ),
    "fr-CH": (
        "Produis exactement un objet JSON sans Markdown. Clés : customer, city, date, amount_rappen, currency. "
        "amount_rappen est un entier en centimes, les autres valeurs sont des chaînes."
    ),
    "it-CH": (
        "Genera esattamente un oggetto JSON senza Markdown. Chiavi: customer, city, date, amount_rappen, currency. "
        "amount_rappen è un intero in centesimi, gli altri valori sono stringhe."
    ),
    "en-CH": (
        "Produce exactly one JSON object without Markdown. Keys: customer, city, date, amount_rappen, currency. "
        "amount_rappen is an integer in rappen, the other values are strings."
    ),
}

_MULTILINGUAL_BODIES = {
    "de-CH": "Auftrag für {customer} in {city}, geplant auf {day}, Honorar {amount}.",
    "fr-CH": "Mandat pour {customer} à {city}, prévu le {day}, honoraires {amount}.",
    "it-CH": "Incarico per {customer} a {city}, previsto il {day}, onorario {amount}.",
    "en-CH": "Engagement for {customer} in {city}, scheduled for {day}, fee {amount}.",
}

_CUSTOMERS = (
    "Brunnen Data AG", "Ticino Werkstatt", "Atelier Léman", "Säntis Consulting",
    "Rhone Logistik", "Aare Kliniken", "Pilatus Handels GmbH", "Furka Immobilien",
)
_CITIES = ("Winterthur", "Chur", "Sion", "Bellinzona", "Fribourg", "Thun", "Olten", "Locarno")


def _multilingual_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    languages = tuple(_MULTILINGUAL_INSTRUCTIONS)
    for position in range(1, 61):
        language = languages[position % len(languages)]
        customer = _CUSTOMERS[rng.randrange(len(_CUSTOMERS))]
        city = _CITIES[rng.randrange(len(_CITIES))]
        day = (date(2027, 3, 1) + timedelta(days=rng.randint(0, 700))).isoformat()
        amount_rappen = rng.randint(1000, 250000)
        body = _MULTILINGUAL_BODIES[language].format(
            customer=customer, city=city, day=day, amount=_format_chf(amount_rappen),
        )
        cases.append(_case(
            f"v2-multilingual-{position:03d}", "multilingual_json", language,
            f"{_MULTILINGUAL_INSTRUCTIONS[language]}\n{body}",
            {"type": "json_exact", "expected": {
                "customer": customer,
                "city": city,
                "date": day,
                "amount_rappen": amount_rappen,
                "currency": "CHF",
            }},
        ))
    return cases


# --------------------------------------------------------------------------
# privacy_policy -- explicit precedence rule closes the v1 ambiguity class
# --------------------------------------------------------------------------

_PRIVACY_POLICY = (
    "Interne Testrichtlinie.\n"
    "ALLOW: bereits öffentliche Angaben, vollständig synthetische Daten, ausdrücklich freigegebene "
    "Inhalte sowie aggregierte Werte ohne Personenbezug.\n"
    "DENY: Geheimnisse und Authentifikatoren, Zahlungs-Sicherheitscodes und Kartennummern, öffentliche "
    "Offenlegung privater Kontakt- oder Bankdaten, Verkauf privater Kundenbestände, Training auf privater "
    "Kommunikation ohne Einwilligung sowie biometrische Daten ohne Einwilligung.\n"
    "REVIEW: Gesundheits- und Kinderdaten, neue Verwendungszwecke, verlängerte Aufbewahrung, "
    "Auskunfts- und Löschbegehren, grenzüberschreitende Bekanntgabe, Überwachung von Mitarbeitenden, "
    "Reidentifikationsrisiken sowie automatisierte Entscheide mit erheblicher Wirkung.\n"
    "Vorrangregel: Treffen mehrere Stufen zu, gilt die strengste. DENY geht REVIEW vor, REVIEW geht ALLOW vor."
)

_PRIVACY_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("Die publizierten Ladenöffnungszeiten einer Filiale in einer Übersicht bündeln", "ALLOW"),
    ("Einen frei zugänglichen Gemeinderatsbeschluss mit Quellenangabe referieren", "ALLOW"),
    ("Vollständig maschinell erfundene Testdatensätze in eine Demo laden", "ALLOW"),
    ("Eine Preisliste, die auf der eigenen Website steht, tabellarisch aufbereiten", "ALLOW"),
    ("Kennzahlen aus dem publizierten Geschäftsbericht zitieren", "ALLOW"),
    ("Eine Umfrageauswertung ohne jeden Personenbezug als Balkendiagramm zeigen", "ALLOW"),
    ("Ein Dokument verarbeiten, für das die betroffene Person schriftlich grünes Licht gab", "ALLOW"),
    ("Öffentlich abrufbare Fahrplandaten in eine Reiseplanung übernehmen", "ALLOW"),
    ("Die Zahl der Supportanfragen pro Monat als blossen Gesamtwert ausweisen", "ALLOW"),
    ("Einen amtlich veröffentlichten Tarif für ein Merkblatt übernehmen", "ALLOW"),
    ("Frei lizenzierte Kartendaten in eine Anfahrtsskizze einbetten", "ALLOW"),
    ("Ein Kundenzitat verwenden, das ausdrücklich zur Veröffentlichung freigegeben wurde", "ALLOW"),
    ("Aus dem öffentlichen Handelsregister die Rechtsform einer Firma nachschlagen", "ALLOW"),
    ("Zufällig generierte Adressen für einen Lasttest verwenden", "ALLOW"),
    ("Die eigene, öffentlich kommunizierte Ferienregelung zusammenfassen", "ALLOW"),
    ("Anonyme Zufriedenheitswerte über alle Standorte hinweg aggregiert darstellen", "ALLOW"),
    ("Eine bereits publizierte Medienmitteilung in Kurzform wiedergeben", "ALLOW"),
    ("Öffentlich einsehbare Gerichtsentscheide in anonymisierter Fassung auswerten", "ALLOW"),
    ("Die aggregierte Ausfallzeit eines Systems ohne Nutzerbezug rapportieren", "ALLOW"),
    ("Trainingsdaten verwenden, die ausschliesslich aus synthetischen Belegen bestehen", "ALLOW"),

    ("Zugangstoken eines Kunden im Ticketsystem im Klartext ablegen", "DENY"),
    ("Den privaten Chatverlauf eines Kunden ohne Einwilligung als Trainingsmaterial nutzen", "DENY"),
    ("Die Wohnadresse einer Privatperson in einem offenen Forum posten", "DENY"),
    ("Fingerabdruckvorlagen ohne ausdrückliche Zustimmung in ein Modell einspeisen", "DENY"),
    ("Die Prüfziffern gespeicherter Zahlkarten in einem Report ausgeben", "DENY"),
    ("Das Adressbuch aus dem CRM gegen Entgelt an einen Drittanbieter abgeben", "DENY"),
    ("Ein Datenbankpasswort in die README des Repositorys schreiben", "DENY"),
    ("Die private Mobilnummer einer Kundin auf der Teamseite veröffentlichen", "DENY"),
    ("Wiederherstellungscodes für die Zweifaktor-Anmeldung in einem Chatprotokoll behalten", "DENY"),
    ("Gesichtsaufnahmen aus der Überwachungskamera ohne Einwilligung zum Anlernen verwenden", "DENY"),
    ("Den privaten E-Mail-Verkehr zweier Mitarbeitender ins Web stellen", "DENY"),
    ("Ein SSH-Schlüsselpaar unverschlüsselt in einem geteilten Ordner ablegen", "DENY"),
    ("Kartennummern samt Prüfziffer dauerhaft für spätere Auswertungen behalten", "DENY"),
    ("Die interne Kundenliste zu Werbezwecken an eine Agentur verkaufen", "DENY"),
    ("Den API-Token eines Partners in einem öffentlichen Gist teilen", "DENY"),
    ("Stimmproben von Anrufenden ohne Zustimmung für ein Sprachmodell nutzen", "DENY"),
    ("Die Sozialversicherungsnummer eines Kunden in einer offenen Tabelle publizieren", "DENY"),
    ("Ein Sitzungs-Cookie eines Nutzers unverändert in offenen Logs ablegen", "DENY"),
    ("Private Chatprotokolle der Belegschaft ohne Einwilligung ins Training geben", "DENY"),
    ("Die Bankverbindung einer Privatperson öffentlich sichtbar machen", "DENY"),

    ("Diagnosedaten aus einer Gesundheits-App für ein neues Analyseprodukt heranziehen", "REVIEW"),
    ("Kundendatensätze auf Server ausserhalb der Schweiz spiegeln", "REVIEW"),
    ("Die Aufbewahrung von Zugriffsprotokollen von drei auf achtzehn Monate verlängern", "REVIEW"),
    ("Ein eingegangenes Auskunfts- und Löschgesuch abarbeiten", "REVIEW"),
    ("Die Tastaturaktivität von Mitarbeitenden zur Produktivitätsmessung auswerten", "REVIEW"),
    ("Zwei pseudonymisierte Bestände zusammenführen, obwohl Rückschlüsse denkbar sind", "REVIEW"),
    ("Spielverhalten von Minderjährigen für Produktentscheide auswerten", "REVIEW"),
    ("Eine vollautomatische Ablehnung von Mietbewerbungen einführen", "REVIEW"),
    ("Bestellhistorien löschen, obwohl eine handelsrechtliche Aufbewahrung greifen könnte", "REVIEW"),
    ("Impfnachweise von Mitarbeitenden für die Einsatzplanung verwenden", "REVIEW"),
    ("Support-Aufzeichnungen an einen Dienstleister in Übersee auslagern", "REVIEW"),
    ("Bewegungsprofile aus Firmenfahrzeugen für Bonusberechnungen nutzen", "REVIEW"),
    ("Bereits erhobene Adressdaten neu für Direktwerbung verwenden", "REVIEW"),
    ("Schulnoten von Kindern in ein Empfehlungssystem einspeisen", "REVIEW"),
    ("Eine automatisierte Bonitätsprüfung mit direkter Vertragsfolge aktivieren", "REVIEW"),
    ("Videoaufnahmen aus dem Pausenraum länger als bisher aufbewahren", "REVIEW"),
    ("Anonymisierte Standortdaten so verknüpfen, dass Reidentifikation möglich bleibt", "REVIEW"),
    ("Therapieverläufe für eine Forschungskooperation freigeben", "REVIEW"),
    ("Ein Löschbegehren umsetzen, während ein Verfahren noch läuft", "REVIEW"),
    ("Nutzungsdaten aus einer Kinder-Lern-App an einen Analysepartner übermitteln", "REVIEW"),
)

# Scenarios that intentionally match two buckets; they only have a defined answer
# because of the precedence rule. Reported separately so the rule itself is measurable.
_PRIVACY_PRECEDENCE_PROBES = frozenset({
    "Kartennummern samt Prüfziffer dauerhaft für spätere Auswertungen behalten",
    "Ein Sitzungs-Cookie eines Nutzers unverändert in offenen Logs ablegen",
    "Videoaufnahmen aus dem Pausenraum länger als bisher aufbewahren",
    "Ein Löschbegehren umsetzen, während ein Verfahren noch läuft",
    "Anonymisierte Standortdaten so verknüpfen, dass Reidentifikation möglich bleibt",
    "Bestellhistorien löschen, obwohl eine handelsrechtliche Aufbewahrung greifen könnte",
})


def _privacy_cases() -> list[dict[str, Any]]:
    if len(_PRIVACY_SCENARIOS) != 60:
        raise ValueError(f"expected 60 privacy scenarios, got {len(_PRIVACY_SCENARIOS)}")
    cases: list[dict[str, Any]] = []
    for position, (scenario, expected) in enumerate(_PRIVACY_SCENARIOS, 1):
        prompt = (
            f"{_PRIVACY_POLICY}\nVorhaben: {scenario}.\n"
            "Antworte ausschliesslich mit ALLOW, DENY oder REVIEW."
        )
        cases.append(_case(
            f"v2-privacy-{position:03d}", "privacy_policy", "de-CH", prompt,
            {"type": "normalized_exact", "expected": expected},
            difficulty="hard" if scenario in _PRIVACY_PRECEDENCE_PROBES else "normal",
        ))
    return cases


# --------------------------------------------------------------------------
# tool_call -- positives plus real negatives, always tool_choice="auto"
# --------------------------------------------------------------------------

_CALENDAR_TEMPLATES = (
    "Schau bitte nach, was am {day} in meinem Kalender steht.",
    "Welche Einträge habe ich am {day}?",
    "Öffne den Terminplan für den {day}.",
    "Ich brauche meine Agenda vom {day}.",
)
_WEATHER_TEMPLATES = (
    "Sag mir bitte, welches Wetter gerade in {city} herrscht.",
    "Brauche ich in {city} heute einen Schirm?",
    "Wie warm ist es momentan in {city}?",
    "Gib mir die aktuelle Wetterlage für {city}.",
)
_INVOICE_TOOL_TEMPLATES = (
    "Ruf mir bitte die Rechnung mit der Nummer {invoice_id} auf.",
    "Schlag nach, was zur Rechnung {invoice_id} hinterlegt ist.",
    "Ich suche den Datensatz zur Rechnungsnummer {invoice_id}.",
    "Zeig mir die Details der Rechnung {invoice_id}.",
)

_TOOL_NEGATIVES: tuple[tuple[str, str], ...] = (
    ("Wie viele Kantone hat die Schweiz?", "normal"),
    ("Was bedeutet die Abkürzung AHV?", "normal"),
    ("Nenne mir den Hauptort des Kantons Wallis.", "normal"),
    ("Wie schreibt man «Strasse» korrekt in der Schweiz?", "normal"),
    ("Übersetze «Rechnung» ins Französische.", "hard"),
    ("Wie viele Zentimeter sind zwei Meter?", "normal"),
    ("Was ist der Unterschied zwischen brutto und netto?", "normal"),
    ("Welche Landessprachen hat die Schweiz?", "normal"),
    ("Erkläre kurz, was eine Mehrwertsteuernummer ist.", "normal"),
    ("Rechne 240 Minuten in Stunden um.", "normal"),
    ("Was heisst «Grüezi» auf Hochdeutsch?", "normal"),
    ("Nenne drei Nachbarländer der Schweiz.", "normal"),
    ("Wofür steht die Abkürzung IBAN?", "normal"),
    ("Wie viele Tage hat der Februar 2028?", "hard"),
    ("Erkläre den Unterschied zwischen Kanton und Bezirk.", "normal"),
    ("Was ist die Telefonvorwahl der Schweiz?", "normal"),
    ("Nenne die vier Sprachregionen der Schweiz.", "normal"),
    ("Was bedeutet CHF ausgeschrieben?", "normal"),
    ("Wie lautet der Plural von «Konto»?", "normal"),
    ("Erkläre kurz den Begriff Skonto.", "normal"),
    ("Erkläre, wie ein Föhnwind entsteht.", "hard"),
    ("Was ist eine Zwischenrechnung?", "hard"),
    ("Wie hoch ist der normale Mehrwertsteuersatz in der Schweiz?", "hard"),
    ("Nenne den grössten See, der vollständig in der Schweiz liegt.", "normal"),
)

_TOOL_CITIES = (
    "Winterthur", "Chur", "Sion", "Bellinzona", "Fribourg", "Thun",
    "Olten", "Locarno", "Baden", "Wil", "Aarau", "Vevey",
)


def _tool_cases(rng: random.Random) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    position = 0
    # 12 cases per tool. Template and payload are both keyed to the per-tool
    # ordinal so no two cases can end up sharing the same rendered prompt.
    used_days: set[str] = set()
    for ordinal in range(12):
        day = date(2027, 2, 1) + timedelta(days=rng.randint(0, 800))
        while day.isoformat() in used_days:
            day = date(2027, 2, 1) + timedelta(days=rng.randint(0, 800))
        used_days.add(day.isoformat())
        position += 1
        cases.append(_case(
            f"v2-tool-pos-{position:03d}", "tool_call", "de-CH",
            _CALENDAR_TEMPLATES[ordinal % len(_CALENDAR_TEMPLATES)].format(day=day.strftime("%d.%m.%Y")),
            {"type": "tool_call", "expected": {"name": "calendar_lookup", "arguments": {"date": day.isoformat()}}},
            tools=TOOLS, tool_choice="auto",
        ))
    for ordinal, city in enumerate(_TOOL_CITIES):
        position += 1
        cases.append(_case(
            f"v2-tool-pos-{position:03d}", "tool_call", "de-CH",
            _WEATHER_TEMPLATES[ordinal % len(_WEATHER_TEMPLATES)].format(city=city),
            {"type": "tool_call", "expected": {"name": "weather_lookup", "arguments": {"city": city}}},
            tools=TOOLS, tool_choice="auto",
        ))
    used_invoices: set[str] = set()
    for ordinal in range(12):
        invoice_id = f"RG{rng.randint(100000, 999999)}-{rng.choice('ABCDEFGH')}"
        while invoice_id in used_invoices:
            invoice_id = f"RG{rng.randint(100000, 999999)}-{rng.choice('ABCDEFGH')}"
        used_invoices.add(invoice_id)
        position += 1
        cases.append(_case(
            f"v2-tool-pos-{position:03d}", "tool_call", "de-CH",
            _INVOICE_TOOL_TEMPLATES[ordinal % len(_INVOICE_TOOL_TEMPLATES)].format(invoice_id=invoice_id),
            {"type": "tool_call", "expected": {"name": "invoice_lookup", "arguments": {"invoice_id": invoice_id}}},
            tools=TOOLS, tool_choice="auto",
        ))

    for position, (prompt, difficulty) in enumerate(_TOOL_NEGATIVES, 1):
        cases.append(_case(
            f"v2-tool-neg-{position:03d}", "tool_call", "de-CH", prompt,
            {"type": "tool_none"},
            subtype="negative", difficulty=difficulty,
            tools=TOOLS, tool_choice="auto",
        ))
    return cases


# --------------------------------------------------------------------------
# suite assembly, freeze contract
# --------------------------------------------------------------------------

def _instance_token(seed: int, case_id: str) -> str:
    key_size = max(32, (seed.bit_length() + 7) // 8)
    key = seed.to_bytes(key_size, "big")
    return hmac.new(key, f"{BENCHMARK_VERSION}\0{case_id}".encode("utf-8"), hashlib.sha256).hexdigest()[:20].upper()


def _structural_prompt(case: dict[str, Any]) -> str:
    prompt = "\n".join(str(message.get("content", "")) for message in case.get("messages", []))
    prompt = re.sub(r"^Benchmark-Instanz [A-F0-9]+\.\n", "", prompt)
    return re.sub(r"\s+", " ", prompt).strip().casefold()


def validate_suite(cases: list[dict[str, Any]]) -> None:
    counts = {category: sum(case["category"] == category for case in cases) for category in CATEGORY_COUNTS}
    if counts != CATEGORY_COUNTS:
        raise ValueError(f"category count mismatch: {counts}")
    strata = {
        key: sum(case["category"] == key[0] and case["subtype"] == key[1] for case in cases)
        for key in STRATUM_COUNTS
    }
    if strata != STRATUM_COUNTS:
        raise ValueError(f"stratum count mismatch: {strata}")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    structural = [_structural_prompt(case) for case in cases]
    if len(set(structural)) != len(structural):
        raise ValueError("duplicate structural prompt")
    for case in cases:
        if case["grader"]["type"] == "tool_none" and case.get("tool_choice") != "auto":
            raise ValueError(f"negative tool case must allow abstention: {case['id']}")


def build_suite(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases = (
        _canton_cases(rng)
        + _date_cases(rng)
        + _chf_cases(rng)
        + _invoice_cases(rng)
        + _multilingual_cases(rng)
        + _privacy_cases()
        + _tool_cases(rng)
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
    split_rng = random.Random(seed ^ 0x48594452414348 ^ 0x02)
    for stratum, dev_count in DEV_COUNTS.items():
        category, subtype = stratum
        group = [case for case in cases if case["category"] == category and case["subtype"] == subtype]
        split_rng.shuffle(group)
        dev.extend(group[:dev_count])
        hidden.extend(group[dev_count:])
    split_rng.shuffle(dev)
    split_rng.shuffle(hidden)
    if len(dev) != DEV_TOTAL or len(hidden) != HIDDEN_TOTAL:
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


# --------------------------------------------------------------------------
# grading -- tool metric decomposed into choice / schema / values
# --------------------------------------------------------------------------

_TOOL_SCHEMAS = {
    tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS
}


def _arguments_schema_valid(name: str, arguments: Any) -> bool:
    schema = _TOOL_SCHEMAS.get(name)
    if schema is None or not isinstance(arguments, dict):
        return False
    properties = schema.get("properties", {})
    if not schema.get("additionalProperties", True) and set(arguments) - set(properties):
        return False
    for key in schema.get("required", []):
        if key not in arguments:
            return False
    for key, value in arguments.items():
        declared = properties.get(key, {}).get("type")
        if declared == "string" and not isinstance(value, str):
            return False
    return True


def _single_tool_call(response: dict[str, Any]) -> dict[str, Any] | None:
    tool_calls = response.get("tool_calls") or []
    if len(tool_calls) != 1:
        return None
    call = tool_calls[0]
    if not isinstance(call, dict) or call.get("type") != "function":
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {"name": function.get("name"), "arguments": None}
    return {"name": function.get("name"), "arguments": arguments}


def grade_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Grade one response.

    For ``tool_call`` cases the result additionally carries ``tool_function_ok``,
    ``tool_args_schema_ok`` and ``tool_args_exact_ok`` so a broken tool parser is
    distinguishable from a wrong decision.
    """
    grader = case["grader"]
    grader_type = grader["type"]
    content = str(response.get("content") or "")
    extra: dict[str, Any] = {}
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
        elif grader_type == "tool_none":
            calls = response.get("tool_calls") or []
            actual = calls
            passed = len(calls) == 0
            extra = {"tool_function_ok": passed, "tool_args_schema_ok": None, "tool_args_exact_ok": None}
        elif grader_type == "tool_call":
            expected = grader["expected"]
            call = _single_tool_call(response)
            actual = call if call is not None else (response.get("tool_calls") or [])
            function_ok = bool(call and call["name"] == expected["name"])
            schema_ok = bool(call and _arguments_schema_valid(str(call["name"]), call["arguments"]))
            exact_ok = bool(call and _json_type_exact_equal(call, expected))
            passed = exact_ok
            extra = {
                "tool_function_ok": function_ok,
                "tool_args_schema_ok": schema_ok,
                "tool_args_exact_ok": exact_ok,
            }
        else:
            raise ValueError(f"unknown grader type: {grader_type}")
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        passed = False
        error = str(exc)
    return {"passed": bool(passed), "grader": grader_type, "actual": actual, "error": error, **extra}


def mcnemar_counts(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
) -> dict[str, int]:
    """Paired contingency counts over the case IDs both systems answered.

    ``b`` = baseline right / candidate wrong, ``c`` = baseline wrong / candidate right.
    Only ``b`` and ``c`` carry information; the exact binomial test on ``b`` against
    ``b + c`` is what resolves differences the unpaired rate cannot.
    """
    shared = set(baseline) & set(candidate)
    counts = {"a": 0, "b": 0, "c": 0, "d": 0, "n": len(shared)}
    for case_id in shared:
        first, second = baseline[case_id], candidate[case_id]
        if first and second:
            counts["a"] += 1
        elif first and not second:
            counts["b"] += 1
        elif not first and second:
            counts["c"] += 1
        else:
            counts["d"] += 1
    return counts


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
        "stratum_counts": {f"{category}/{subtype}": count for (category, subtype), count in STRATUM_COUNTS.items()},
        "graders": ["normalized_exact", "json_exact", "tool_call", "tool_none"],
        "evaluation_contract": {
            "paired": "systems are compared with McNemar over shared case IDs, not with unpaired rates",
            "tool_metric": "tool_call reports function choice, schema validity and exact arguments separately",
            "tool_choice": "auto for every tool case so abstention and false-positive calls are measurable",
            "raw_outputs": "the harness must persist each candidate's answer per case; v1 reports stored only pass/fail",
        },
        "freeze_contract": {
            "canonical_format": "UTF-8 NFC JSONL, LF, sorted keys, compact separators, no floats",
            "generator_sha256": hashlib.sha256(generator_bytes).hexdigest(),
            "python": platform.python_version(),
            "seed_commitment": hmac.new(seed_key, b"hydrach-v2-freeze", hashlib.sha256).hexdigest(),
            "transaction": "all artifacts are staged first; manifest is replaced last and commits the split hashes",
        },
        "hidden_policy": "Exact hidden payload bytes and HMAC-derived instance tokens are root-only; taxonomy, templates and case-ID ranges are public. Git publishes only the frozen hidden bundle SHA-256.",
        "splits": {
            "dev": {"count": len(dev), "path": dev_path.name, "sha256": hashlib.sha256(dev_bytes).hexdigest()},
            "hidden": {"count": len(hidden), "path": "[ROOT-ONLY]", "sha256": hashlib.sha256(hidden_bytes).hexdigest()},
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    resolved = [path.resolve(strict=False) for path in (dev_path, hidden_path, manifest_path)]
    if len(set(resolved)) != 3:
        raise ValueError("dev, hidden and manifest paths must be distinct")
    for path in (dev_path, hidden_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        _assert_safe_artifact_path(path)
    os.chmod(hidden_path.parent, 0o700)

    expected = ((dev_path, dev_bytes, 0o600), (hidden_path, hidden_bytes, 0o600), (manifest_path, manifest_bytes, 0o644))
    if any(path.exists() for path, _, _ in expected):
        unchanged = all(path.exists() and path.read_bytes() == data for path, data, _ in expected)
        if unchanged:
            for path, _, mode in expected:
                os.chmod(path, mode)
            return {
                "dev_count": len(dev),
                "hidden_count": len(hidden),
                "dev_sha256": manifest["splits"]["dev"]["sha256"],
                "hidden_sha256": manifest["splits"]["hidden"]["sha256"],
            }
        if not force:
            raise FileExistsError("frozen artifacts differ; use an explicit force refreeze after review")

    staged: list[tuple[Path, Path, int]] = []
    try:
        for path, data, mode in expected:
            staged.append((_stage_bytes(path, data, mode), path, mode))
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

    return {
        "dev_count": len(dev),
        "hidden_count": len(hidden),
        "dev_sha256": manifest["splits"]["dev"]["sha256"],
        "hidden_sha256": manifest["splits"]["hidden"]["sha256"],
    }
