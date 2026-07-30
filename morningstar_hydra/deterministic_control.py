"""Narrow deterministic capabilities for exact, auditable requests.

The control plane is intentionally domain-agnostic: it dispatches to small
capabilities, while locale knowledge (for example Swiss canton names) is data.
Every resolver fails open to the general model unless intent and output contract
are unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable


@dataclass(frozen=True)
class ExactResolution:
    content: str
    capability: str


# Locale data, not routing policy. Canonical values are German because this
# catalog's explicit contract is the German Swiss administrative name.
_SWISS_CANTONS = {
    "AG": ("Aargau",),
    "AI": ("Appenzell Innerrhoden",),
    "AR": ("Appenzell Ausserrhoden",),
    "BE": ("Bern", "Berne", "Berna"),
    "BL": ("Basel-Landschaft", "Basel Landschaft"),
    "BS": ("Basel-Stadt", "Basel Stadt"),
    "FR": ("Freiburg", "Fribourg"),
    "GE": ("Genf", "Genève", "Geneve", "Ginevra"),
    "GL": ("Glarus", "Glarona"),
    "GR": ("Graubünden", "Graubuenden", "Grigioni", "Grischun"),
    "JU": ("Jura",),
    "LU": ("Luzern", "Lucerne", "Lucerna"),
    "NE": ("Neuenburg", "Neuchâtel", "Neuchatel"),
    "NW": ("Nidwalden",),
    "OW": ("Obwalden",),
    "SG": ("St. Gallen", "St Gallen", "Saint-Gall", "San Gallo"),
    "SH": ("Schaffhausen",),
    "SO": ("Solothurn", "Soleure", "Soletta"),
    "SZ": ("Schwyz",),
    "TG": ("Thurgau", "Thurgovie", "Turgovia"),
    "TI": ("Tessin", "Ticino"),
    "UR": ("Uri",),
    "VD": ("Waadt", "Vaud"),
    "VS": ("Wallis", "Valais", "Vallese"),
    "ZG": ("Zug", "Zoug", "Zugo"),
    "ZH": ("Zürich", "Zurich"),
}
_CANTON_NAMES = {code: names[0] for code, names in _SWISS_CANTONS.items()}
_CANTON_ALIASES = tuple(
    sorted(
        ((alias, code) for code, aliases in _SWISS_CANTONS.items() for alias in aliases),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_CANTON_CONTEXT = re.compile(r"\b(?:canton\w*|kanton\w*|cantone\w*|chantun\w*)\b", re.I)
# Die Tabelle haelt genau zwei Groessen je Kanton: Code und Name. Wird eine
# dritte verlangt, waere jede Antwort von ihr die Antwort auf eine andere Frage.
# Die Aufzaehlung beschreibt, was man ueber einen Kanton sonst wissen will --
# sie ist eine Aussage ueber die Grenzen dieser Datenbasis.
_OTHER_ATTRIBUTE = re.compile(
    r"\b(?:hauptort|hauptstadt|kantonshauptort|chef-?lieu|capoluogo|capital|"
    r"einwohner\w*|bevölkerung|bevoelkerung|population|abitanti|"
    r"fläche|flaeche|superficie|area|grösse|groesse|"
    r"amtssprach\w*|sprache\w*|langue\w*|lingua|language|"
    r"wappen|flagge|blason|stemma|coat of arms|"
    r"beitritt|gegründet|gegruendet|seit wann|bezirke|gemeinden|communes|comuni|"
    r"regierung|parlament|ständerat|staenderat|nationalrat|steuer\w*|bip|gdp)\b",
    re.I,
)
# Fragewoerter, die nach etwas anderem als einer Bezeichnung verlangen: nach
# einem Zeitpunkt, einem Ort, einer Menge. Code und Name sind keine davon.
_NON_LABEL_QUESTION = re.compile(
    r"\b(?:wann|seit wann|wo\b|wieviel\w*|wie viel\w*|wie gross|wie groß|warum|"
    r"when|where|how many|how much|how large|why|quand|où|combien|quando|dove|quanti)\b",
    re.I,
)
# Eine angehaengte zweite Forderung macht die Anfrage groesser als das, was eine
# Nachschlagetabelle beantworten kann.
_SECOND_DEMAND = re.compile(
    r"\b(?:und|sowie|and|et|e)\s+"
    r"(?:wie|was|wo|wann|warum|wer|welche\w*|how|what|where|when|why|who|which|"
    r"quel\w*|combien|quanto|quanti)\b",
    re.I,
)
_SUPPORTED_CURRENCIES = frozenset(
    {
        "AED", "AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP",
        "HKD", "HUF", "INR", "JPY", "KRW", "MXN", "NOK", "NZD", "PLN", "SAR",
        "SEK", "SGD", "USD", "ZAR",
    }
)
_CURRENCY_AMOUNT = re.compile(
    r"(?<![A-Z0-9])(?P<currency>[A-Z]{3})\s*"
    r"(?P<amount>[+-]?\d(?:[\d'’ .,]*\d)?)(?![A-Za-z0-9])"
)
_PERCENT = re.compile(r"(?P<percent>\d+(?:[.,]\d+)?)\s*%")
_OUTPUT_CONTRACT = re.compile(
    r"\b(?:return|output|answer|nenn\w*|ausgabe|gib\w*|retourne|restituisci)\b",
    re.I,
)
_SUM_INTENT = re.compile(r"\b(?:total|sum|summe|endsumme|gesamt|lists?|listet)\b", re.I)
_TAX_INTENT = re.compile(
    r"\b(?:vat|tax|mwst|mehrwertsteuer|tva|iva|gross|brutto|aufschlag|add)\b",
    re.I,
)
_DISCOUNT_INTENT = re.compile(r"\b(?:discount|rabatt|remise|sconto)\b", re.I)
_REMAINDER_INTENT = re.compile(
    r"\b(?:paid|beglichen|bezahlt|outstanding|offen\w*|restbetrag|remaining)\b",
    re.I,
)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _user_text(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("tools"):
        return None
    if payload.get("n", 1) != 1 or payload.get("response_format") is not None:
        return None
    thinking = payload.get("chat_template_kwargs")
    if isinstance(thinking, dict) and thinking.get("enable_thinking") is True:
        return None
    messages = payload.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
    ):
        return None
    text = _message_text(messages[0].get("content")).strip()
    return text or None


def _resolve_catalog(text: str) -> ExactResolution | None:
    """Schlaegt Kantonscode und -name nach, unabhaengig von der Frageform.

    Die Richtung folgt aus dem, was gegeben ist, nicht aus einer Liste erwarteter
    Formulierungen: Steht genau ein Code im Text, kann nur der Name gesucht sein;
    steht genau ein Name, nur der Code. Ist beides oder mehreres da, ist die
    Absicht nicht ableitbar und das Modell uebernimmt.

    Frueher entschied hier eine Wortliste (`kürzel`, `ausgeschrieben`, `nom` …),
    ob ueberhaupt nachgeschlagen wird. Eine solche Liste deckt immer nur die
    Formulierungen ab, die man beim Schreiben vor Augen hatte — sie waechst mit
    jedem Fall, der ihr durchgeht, statt eine Regel zu sein.
    """
    if not _CANTON_CONTEXT.search(text):
        return None
    if (_SECOND_DEMAND.search(text) or _OTHER_ATTRIBUTE.search(text)
            or _NON_LABEL_QUESTION.search(text)):
        # Die Tabelle liefert genau einen Wert, und nur Code oder Name. Wer mehr
        # oder anderes verlangt, bekommt das Modell — eine Antwort auf die
        # falsche Frage waere schlechter als keine.
        return None

    codes = {
        match.group(0).upper()
        for match in re.finditer(r"\b[A-Za-z]{2}\b", text)
        if match.group(0).upper() in _CANTON_NAMES
    }
    folded = text.casefold()
    names: set[str] = set()
    for alias, code in _CANTON_ALIASES:
        if re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", folded):
            names.add(code)

    if len(codes) == 1 and not names:
        return ExactResolution(_CANTON_NAMES[next(iter(codes))], "catalog_lookup")
    if len(names) == 1 and not codes:
        return ExactResolution(next(iter(names)), "catalog_lookup")
    return None


def _minor_units(raw: str) -> int:
    normalized = raw.replace("'", "").replace("’", "").replace(" ", "")
    sign = ""
    if normalized.startswith(("+", "-")):
        sign, normalized = normalized[0], normalized[1:]
    if not normalized or not normalized[0].isdigit():
        raise InvalidOperation("invalid monetary amount")

    separators = [separator for separator in (",", ".") if separator in normalized]
    if len(separators) == 2:
        decimal_separator = max(separators, key=normalized.rfind)
        thousands_separator = "," if decimal_separator == "." else "."
        integer, fraction = normalized.rsplit(decimal_separator, 1)
        integer = integer.replace(thousands_separator, "")
        if decimal_separator in integer or not 1 <= len(fraction) <= 2:
            raise InvalidOperation("ambiguous monetary separators")
        normalized = f"{integer}.{fraction}"
    elif len(separators) == 1:
        separator = separators[0]
        groups = normalized.split(separator)
        if any(not group.isdigit() for group in groups):
            raise InvalidOperation("invalid monetary grouping")
        if len(groups) == 2 and 1 <= len(groups[-1]) <= 2:
            normalized = f"{groups[0]}.{groups[1]}"
        elif len(groups) >= 2 and all(len(group) == 3 for group in groups[1:]):
            normalized = "".join(groups)
        else:
            raise InvalidOperation("ambiguous monetary grouping")
    if not normalized.replace(".", "", 1).isdigit():
        raise InvalidOperation("invalid monetary amount")
    value = Decimal(sign + normalized)
    return int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _format_minor(currency: str, minor: int) -> str:
    sign = "-" if minor < 0 else ""
    absolute = abs(minor)
    return f"{currency} {sign}{absolute // 100}.{absolute % 100:02d}"


def _round_ratio(numerator: int | Decimal, denominator: Decimal) -> int:
    return int((Decimal(numerator) / denominator).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _is_format_placeholder(match: re.Match[str], text: str) -> bool:
    try:
        amount = _minor_units(match.group("amount"))
    except (InvalidOperation, ValueError):
        return False
    if amount != 0:
        return False
    prefix = text[max(0, match.start() - 32):match.start()]
    return bool(re.search(r"(?:\b(?:as|als|format|formatting|comme|come)\s*)$", prefix, re.I))


def _resolve_currency(text: str) -> ExactResolution | None:
    if not _OUTPUT_CONTRACT.search(text):
        return None
    matches = [
        match for match in _CURRENCY_AMOUNT.finditer(text)
        if match.group("currency") in _SUPPORTED_CURRENCIES
        and not _is_format_placeholder(match, text)
    ]
    if not matches:
        return None
    currencies = {match.group("currency") for match in matches}
    if len(currencies) != 1:
        return None
    currency = next(iter(currencies))
    try:
        amounts = [_minor_units(match.group("amount")) for match in matches]
    except (InvalidOperation, ValueError):
        return None
    percentages = [Decimal(match.group("percent").replace(",", ".")) for match in _PERCENT.finditer(text)]

    result: int | None = None
    if _DISCOUNT_INTENT.search(text) and len(amounts) == 1 and len(percentages) == 1:
        adjustment = _round_ratio(amounts[0] * percentages[0], Decimal(100))
        result = amounts[0] - adjustment
    elif _TAX_INTENT.search(text) and len(amounts) == 1 and len(percentages) == 1:
        result = _round_ratio(amounts[0] * (Decimal(100) + percentages[0]), Decimal(100))
    elif _REMAINDER_INTENT.search(text) and len(amounts) == 2 and not percentages:
        result = amounts[0] - amounts[1]
    elif _SUM_INTENT.search(text) and len(amounts) >= 2 and not percentages:
        result = sum(amounts)

    if result is None:
        return None
    return ExactResolution(_format_minor(currency, result), "currency_arithmetic")


_RESOLVERS: tuple[Callable[[str], ExactResolution | None], ...] = (
    _resolve_catalog,
    _resolve_currency,
)


def resolve_exact_request(payload: dict[str, Any]) -> ExactResolution | None:
    """Resolve a narrow exact request, otherwise return ``None`` for model fallback."""
    text = _user_text(payload)
    if text is None:
        return None
    resolutions = [result for resolver in _RESOLVERS if (result := resolver(text)) is not None]
    if len(resolutions) != 1:
        return None
    return resolutions[0]
