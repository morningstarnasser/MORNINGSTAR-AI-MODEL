#!/usr/bin/env python3
"""Erzeugt synthetische CH-Fachdaten als SFT-Trainingsmaterial.

Hintergrund: Der Lauf vom 2026-07-25 mit dem generischen Mix aus
`config/hf-expert-mixture.json` hat HydraCH nicht bewegt (22/50 vorher wie
nachher). Grund war fehlende Ueberschneidung — in 20074 Beispielen kamen
"Rappen", "Kanton", "Zuerich" und "revDSG" kein einziges Mal vor, waehrend der
Benchmark 41 von 50 Faellen auf de-CH prueft.

Dieses Skript erzeugt die fehlenden Faehigkeiten: Kantons-Mapping,
CHF-Arithmetik mit Ausgabeformat, Datumsnormalisierung nach ISO 8601 in vier
Sprachen, Feldextraktion nach JSON mit Betraegen als Rappen-Ganzzahl, sowie
Richtlinienentscheide.

Abgrenzung zum Benchmark: Trainiert werden Faehigkeiten, nicht Testinstanzen.
Werte werden aus eigenen Bereichen gezogen (andere Jahre, Betraege, Firmen,
Vorgaenge) und der Instanzmarker des Benchmarks wird nicht nachgebaut. Bei
geschlossenen Faktenraeumen — es gibt genau 26 Kantone — ist inhaltliche
Ueberlappung unvermeidbar und auch erwuenscht; das ist Faktenwissen, keine
Testleckage. Vor dem Training gehoert trotzdem
`scripts/check_ch_contamination.py` gefahren.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (Kuerzel, deutsch, franzoesisch, italienisch)
KANTONE = [
    ("AG", "Aargau", "Argovie", "Argovia"),
    ("AI", "Appenzell Innerrhoden", "Appenzell Rhodes-Intérieures", "Appenzello Interno"),
    ("AR", "Appenzell Ausserrhoden", "Appenzell Rhodes-Extérieures", "Appenzello Esterno"),
    ("BE", "Bern", "Berne", "Berna"),
    ("BL", "Basel-Landschaft", "Bâle-Campagne", "Basilea Campagna"),
    ("BS", "Basel-Stadt", "Bâle-Ville", "Basilea Città"),
    ("FR", "Freiburg", "Fribourg", "Friburgo"),
    ("GE", "Genf", "Genève", "Ginevra"),
    ("GL", "Glarus", "Glaris", "Glarona"),
    ("GR", "Graubünden", "Grisons", "Grigioni"),
    ("JU", "Jura", "Jura", "Giura"),
    ("LU", "Luzern", "Lucerne", "Lucerna"),
    ("NE", "Neuenburg", "Neuchâtel", "Neuchâtel"),
    ("NW", "Nidwalden", "Nidwald", "Nidvaldo"),
    ("OW", "Obwalden", "Obwald", "Obvaldo"),
    ("SG", "St. Gallen", "Saint-Gall", "San Gallo"),
    ("SH", "Schaffhausen", "Schaffhouse", "Sciaffusa"),
    ("SO", "Solothurn", "Soleure", "Soletta"),
    ("SZ", "Schwyz", "Schwytz", "Svitto"),
    ("TG", "Thurgau", "Thurgovie", "Turgovia"),
    ("TI", "Tessin", "Tessin", "Ticino"),
    ("UR", "Uri", "Uri", "Uri"),
    ("VD", "Waadt", "Vaud", "Vaud"),
    ("VS", "Wallis", "Valais", "Vallese"),
    ("ZG", "Zug", "Zoug", "Zugo"),
    ("ZH", "Zürich", "Zurich", "Zurigo"),
]

# (PLZ, Ort) — bewusst breit gestreut ueber die Sprachregionen
ORTE = [
    ("8001", "Zürich"), ("3001", "Bern"), ("4001", "Basel"), ("6003", "Luzern"),
    ("9000", "St. Gallen"), ("1201", "Genève"), ("1003", "Lausanne"), ("6900", "Lugano"),
    ("6500", "Bellinzona"), ("2000", "Neuchâtel"), ("1700", "Fribourg"), ("7000", "Chur"),
    ("8200", "Schaffhausen"), ("6300", "Zug"), ("4500", "Solothurn"), ("5000", "Aarau"),
    ("8400", "Winterthur"), ("3900", "Brig"), ("2800", "Delémont"), ("8750", "Glarus"),
]

FIRMEN_STAMM = [
    "Bergwind", "Talblick", "Seematt", "Hochmoos", "Rebhalde", "Steinacker", "Lindenhof",
    "Aarequell", "Sonnhalde", "Weidgang", "Falkenstein", "Moosbach", "Erlenau", "Kiesgrube",
    "Nebelweid", "Silberdistel", "Buchsgarten", "Turmfeld", "Waldkante", "Fluhblick",
]
FIRMEN_FORM = ["GmbH", "AG", "Sàrl", "SA", "Genossenschaft"]

MONATE_DE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
             "August", "September", "Oktober", "November", "Dezember"]

CODE_TO_NAME = {code: de for code, de, _fr, _it in KANTONE}


def rappen(betrag_chf: float) -> int:
    """CHF als Rappen-Ganzzahl — der Punkt, an dem Modelle typischerweise scheitern."""
    return int(round(betrag_chf * 100))


def fmt_chf(betrag_chf: float) -> str:
    return f"CHF {betrag_chf:.2f}"


def zufallsfirma(rng: random.Random) -> str:
    return f"{rng.choice(FIRMEN_STAMM)} {rng.choice(FIRMEN_FORM)}"


def zufallsdatum(rng: random.Random) -> tuple[int, int, int]:
    return rng.randint(1, 28), rng.randint(1, 12), rng.randint(2024, 2031)


# Kuerzelpaare, die sich nur in einem Buchstaben unterscheiden. Genau hier lag der
# Restfehler nach Runde 1 (SG wurde zu "Schwyz", ZG zu "Zürich"), deshalb werden
# diese Paare kontrastiv geuebt statt nur einzeln.
VERWECHSELBAR = [
    ("SG", "SZ"), ("ZG", "ZH"), ("AR", "AI"), ("BS", "BL"),
    ("NE", "NW"), ("OW", "NW"), ("SH", "SO"), ("VD", "VS"), ("GL", "GR"),
]


def gen_canton_contrast(rng: random.Random) -> tuple[str, str]:
    """Stellt zwei aehnliche Kuerzel direkt gegenueber."""
    a, b = rng.choice(VERWECHSELBAR)
    if rng.random() < 0.5:
        a, b = b, a
    name_a = CODE_TO_NAME[a]
    richtung = rng.random()
    if richtung < 0.5:
        prompt = rng.choice([
            f"Achtung, {a} und {b} werden oft verwechselt. Für welchen Kanton steht {a}? "
            f"Nur der deutsche Name.",
            f"{a} oder {b} — welcher davon ist {name_a}? Antworte nur mit dem Kennzeichen.",
        ])
        antwort = name_a if "Für welchen" in prompt else a
    else:
        prompt = (f"Nenne den Kanton zu {a}, nicht zu verwechseln mit {b}. "
                  f"Nur der deutsche Name.")
        antwort = name_a
    return prompt, antwort


def gen_canton_code(rng: random.Random) -> tuple[str, str]:
    # Formulierungen bewusst anders als im Benchmark: die Fakten sollen ueberlappen
    # (es gibt nur 26 Kantone), die Frageform nicht — sonst misst der Benchmark
    # Auswendiglernen statt Koennen.
    code, de, fr, it = rng.choice(KANTONE)
    sprache = rng.choices(["de", "fr", "it"], weights=[0.7, 0.15, 0.15])[0]
    if sprache == "fr":
        prompt = rng.choice([
            f"Quelle est l'abréviation du canton {fr}? Deux lettres, rien d'autre.",
            f"Abrège le canton {fr} en deux lettres. Réponse courte.",
        ])
    elif sprache == "it":
        prompt = rng.choice([
            f"Qual è la sigla del cantone {it}? Due lettere, nient'altro.",
            f"Abbrevia il cantone {it} in due lettere. Risposta breve.",
        ])
    else:
        prompt = rng.choice([
            f"Wie lautet die Abkürzung des Kantons {de}? Nur die zwei Buchstaben.",
            f"Kürze den Kanton {de} auf zwei Buchstaben ab. Keine weiteren Angaben.",
            f"{de} — welches Kantonskennzeichen? Antworte knapp.",
            f"Schreibe das Kennzeichen des Kantons {de}. Ohne Erklärung.",
            f"Auf welche zwei Buchstaben kürzt man {de} ab?",
        ])
    return prompt, code


def gen_canton_name(rng: random.Random) -> tuple[str, str]:
    code, de, _fr, _it = rng.choice(KANTONE)
    prompt = rng.choice([
        f"Welcher Kanton verbirgt sich hinter {code}? Nur der Name auf Deutsch.",
        f"{code} steht für welchen Kanton? Deutscher Name, sonst nichts.",
        f"Löse das Kantonskennzeichen {code} auf. Nur der deutsche Name.",
        f"Schreibe den Kanton zum Kennzeichen {code} aus. Ohne Erklärung.",
        f"Zu welchem Kanton gehört {code}? Antworte knapp auf Deutsch.",
    ])
    return prompt, de


CHF_SCHLUSS = [
    "Antworte im Format CHF 0.00.",
    "Nur das Resultat als CHF 0.00.",
    "Ergebnis als CHF 0.00, sonst nichts.",
    "Gib den Betrag in der Form CHF 0.00 an, ohne Rechenweg.",
]


def gen_chf_arithmetic(rng: random.Random) -> tuple[str, str]:
    a = round(rng.uniform(5, 4000), 2)
    b = round(rng.uniform(1, min(a, 900)), 2)
    art = rng.choice(["add", "sub", "mult", "mwst", "rabatt", "teil"])
    schluss = rng.choice(CHF_SCHLUSS)

    if art == "add":
        aufgabe = f"Addiere CHF {a:.2f} und CHF {b:.2f}."
        wert = a + b
    elif art == "sub":
        # In 40% der Faelle liegt der Subtrahend dicht am Minuenden. Runde 1 scheiterte
        # genau daran (106.16 - 102.96 wurde zu "CHF 0.00"): kleine Differenzen sind
        # der Fall, in dem ein Modell gern auf null abrutscht.
        if rng.random() < 0.4:
            b = round(a - rng.uniform(0.05, 12), 2)
            b = max(0.05, b)
        aufgabe = f"Ziehe CHF {b:.2f} von CHF {a:.2f} ab."
        wert = a - b
    elif art == "mult":
        n = rng.randint(2, 12)
        aufgabe = f"Multipliziere CHF {a:.2f} mit {n}."
        wert = a * n
    elif art == "mwst":
        satz = rng.choice([8.1, 2.6, 3.8])
        aufgabe = (f"Berechne {satz:.1f}% MWST auf CHF {a:.2f} und gib den "
                   f"Bruttobetrag an.")
        wert = a * (1 + satz / 100)
    elif art == "rabatt":
        pct = rng.choice([5, 10, 15, 20, 25, 50])
        aufgabe = f"Ziehe {pct}% Rabatt von CHF {a:.2f} ab."
        wert = a * (1 - pct / 100)
    else:
        n = rng.randint(2, 8)
        aufgabe = f"Teile CHF {a:.2f} durch {n}."
        wert = a / n
    return f"{aufgabe} {schluss}", fmt_chf(round(wert + 1e-9, 2))


def gen_date_iso(rng: random.Random) -> tuple[str, str]:
    tag, monat, jahr = zufallsdatum(rng)
    iso = f"{jahr:04d}-{monat:02d}-{tag:02d}"
    ch = f"{tag:02d}.{monat:02d}.{jahr:04d}"
    sprache = rng.choices(["de", "fr", "it", "en", "de_lang"],
                          weights=[0.4, 0.15, 0.15, 0.15, 0.15])[0]
    if sprache == "fr":
        prompt = rng.choice([
            f"Normalise la date {ch} au format YYYY-MM-DD. Rien d'autre.",
            f"Réécris {ch} selon la norme ISO 8601. Uniquement le résultat.",
        ])
    elif sprache == "it":
        prompt = rng.choice([
            f"Normalizza la data {ch} in formato YYYY-MM-DD. Solo il risultato.",
            f"Riscrivi {ch} secondo lo standard ISO 8601. Nient'altro.",
        ])
    elif sprache == "en":
        prompt = rng.choice([
            f"Rewrite the date {ch} as YYYY-MM-DD. Answer with the date only.",
            f"Normalise {ch} to ISO 8601. Output nothing else.",
        ])
    elif sprache == "de_lang":
        prompt = rng.choice([
            f"Normalisiere den {tag}. {MONATE_DE[monat-1]} {jahr} nach YYYY-MM-DD. "
            f"Nur das Ergebnis.",
            f"Schreibe den {tag}. {MONATE_DE[monat-1]} {jahr} als YYYY-MM-DD. Ohne Zusatz.",
        ])
    else:
        prompt = rng.choice([
            f"Normalisiere {ch} nach ISO 8601 (YYYY-MM-DD). Nur das Ergebnis.",
            f"Bringe das Datum {ch} in die Form YYYY-MM-DD. Antworte knapp.",
            f"Schreibe {ch} im Format YYYY-MM-DD. Ohne weiteren Text.",
            f"Formatiere {ch} nach YYYY-MM-DD um.",
        ])
    return prompt, iso


def gen_invoice_json(rng: random.Random) -> tuple[str, str]:
    plz, ort = rng.choice(ORTE)
    betrag = round(rng.uniform(12, 9000), 2)
    tag, monat, jahr = zufallsdatum(rng)
    faellig = f"{jahr:04d}-{monat:02d}-{tag:02d}"
    rnr = f"{rng.choice(['RE','INV','MS','FA','QR'])}-{jahr}-{rng.randint(1000, 9999)}"
    einleitung = rng.choice([
        "Überführe die folgenden Rechnungsangaben in genau ein JSON-Objekt (kein Markdown). "
        "Benutze diese Felder: invoice_id, amount_rappen, currency, due_date, postal_code, city.",
        "Strukturiere die Rechnung als einzelnes JSON-Objekt ohne Codeblock. "
        "Erlaubte Schlüssel: invoice_id, amount_rappen, currency, due_date, postal_code, city.",
        "Gib die Rechnungsdaten als ein JSON-Objekt aus, unformatiert. "
        "Schlüssel genau: invoice_id, amount_rappen, currency, due_date, postal_code, city.",
    ])
    prompt = (
        f"{einleitung}\n"
        f"Rechnung {rnr}; Betrag CHF {betrag:.2f}; fällig {faellig}; "
        f"Zahlungsadresse {plz} {ort}."
    )
    antwort = {
        "invoice_id": rnr, "amount_rappen": rappen(betrag), "currency": "CHF",
        "due_date": faellig, "postal_code": plz, "city": ort,
    }
    return prompt, json.dumps(antwort, ensure_ascii=False)


def gen_multilingual_json(rng: random.Random) -> tuple[str, str]:
    plz, ort = rng.choice(ORTE)
    kunde = zufallsfirma(rng)
    betrag = round(rng.uniform(8, 5000), 2)
    tag, monat, jahr = zufallsdatum(rng)
    datum = f"{jahr:04d}-{monat:02d}-{tag:02d}"
    variante = rng.choice(["fr", "it", "de", "mix"])
    if variante == "fr":
        körper = (f"Client {kunde}; lieu {ort}; rendez-vous {datum}; "
                  f"montant CHF {betrag:.2f}.")
    elif variante == "it":
        körper = (f"Cliente {kunde}; luogo {ort}; appuntamento {datum}; "
                  f"importo CHF {betrag:.2f}.")
    elif variante == "mix":
        körper = (f"Kunde {kunde}; lieu {ort}; data {datum}; "
                  f"Betrag CHF {betrag:.2f}.")
    else:
        körper = (f"Kunde {kunde}; Ort {ort}; Termin {datum}; "
                  f"Betrag CHF {betrag:.2f}.")
    einleitung = rng.choice([
        "Erzeuge genau ein JSON-Objekt ohne Markdown mit den Schlüsseln customer, city, "
        "date, amount_rappen, currency.",
        "Fasse die Angaben in einem einzigen JSON-Objekt zusammen (kein Codeblock). "
        "Verwende die Schlüssel customer, city, date, amount_rappen, currency.",
        "Produce a single JSON object, no code fences, with the keys customer, city, "
        "date, amount_rappen, currency.",
    ])
    prompt = f"{einleitung}\n{körper}"
    antwort = {"customer": kunde, "city": ort, "date": datum,
               "amount_rappen": rappen(betrag), "currency": "CHF"}
    return prompt, json.dumps(antwort, ensure_ascii=False)


# Mehrere eigenstaendig formulierte Regelwerke statt eines festen Textes: Das Modell
# soll lernen, die jeweils mitgelieferte Regel zu LESEN und anzuwenden — nicht einen
# bestimmten Wortlaut auswendig zu koennen. Senkt zugleich die Aehnlichkeit zum
# Benchmark, dessen Richtlinie hier bewusst nicht uebernommen wird.
RICHTLINIEN = [
    ("Datenfreigabe-Regelwerk A.\n"
     "Freigeben (ALLOW), wenn es sich um bereits veröffentlichte Angaben, rein erfundene "
     "Testbestände, schriftlich bewilligte Inhalte oder Kennzahlen ohne Personenbezug handelt.\n"
     "Sperren (DENY) bei Zugangsdaten, Schlüsseln, Sicherheitsmerkmalen von Zahlungsmitteln, "
     "dem Veröffentlichen privater Erreichbarkeiten, dem Weiterverkauf von Kundenverzeichnissen, "
     "der Verwertung privater Korrespondenz ohne Erlaubnis sowie ungenehmigten Körpermerkmalen.\n"
     "Zur Prüfung vorlegen (REVIEW) bei Angaben zu Gesundheit oder Minderjährigen, geänderter "
     "Zweckbestimmung, verlängerter Speicherdauer, Begehren auf Entfernung, Ausland-Transfer, "
     "Beobachtung von Angestellten, Rückschlussgefahr, weitreichender Automatik oder offener "
     "Aufbewahrungspflicht."),

    ("Prüfschema B für Datenvorgänge.\n"
     "ALLOW gilt für öffentlich zugängliche Informationen, künstlich erzeugte Datensätze, "
     "ausdrücklich genehmigtes Material und zusammengefasste Werte ohne Personenbezug.\n"
     "DENY gilt für Betriebsgeheimnisse, Anmeldeinformationen, Prüfziffern von Karten, das "
     "Offenlegen privater Kontaktangaben, den Handel mit Kundendaten, das Anlernen von Modellen "
     "auf privater Kommunikation ohne Zustimmung und für biometrische Erfassung ohne Bewilligung.\n"
     "REVIEW gilt für Patienten- und Kinderdaten, neue Verwendungszwecke, längeres Vorhalten, "
     "Löschanfragen, Übermittlung ins Ausland, Kontrolle von Mitarbeitenden, mögliche "
     "Wiedererkennung, folgenreiche maschinelle Entscheide und unklare gesetzliche Fristen."),

    ("Entscheidungsraster C.\n"
     "Erlaubt ist die Verarbeitung, wenn die Daten publik, vollständig fingiert, nachweislich "
     "freigegeben oder so verdichtet sind, dass niemand identifizierbar ist.\n"
     "Untersagt ist sie bei Passwörtern und Tokens, bei Sicherheitscodes aus dem Zahlungsverkehr, "
     "beim Publizieren privater Adressen oder Nummern, beim Abtreten von Kontaktbeständen an "
     "Dritte, beim Training auf privaten Nachrichten ohne Einwilligung und bei ungenehmigten "
     "biometrischen Merkmalen.\n"
     "Vorzulegen ist sie bei Gesundheits- und Kinderdaten, Zweckänderung, Fristverlängerung, "
     "Löschwunsch, Datenfluss über die Landesgrenze, Überwachung des Personals, Gefahr der "
     "Reidentifikation, automatisierten Entscheiden mit Gewicht und möglichen Archivpflichten."),
]

VORGAENGE = {
    "ALLOW": [
        "Aggregierte Besucherzahlen ohne Personenbezug auswerten",
        "Vollständig synthetische Testdatensätze weitergeben",
        "Öffentlich publizierte Geschäftsberichte zusammenfassen",
        "Anonymisierte Umsatzsummen pro Quartal berichten",
        "Ausdrücklich freigegebene Referenzlogos verwenden",
        "Öffentliche Fahrplandaten in eine App einbinden",
        "Nicht personenbezogene Sensormesswerte archivieren",
        "Öffentliche Preislisten von Mitbewerbern vergleichen",
        "Erfundene Beispieladressen in einer Schulung zeigen",
        "Die Anzahl Bestellungen pro Monat ohne Namen ausweisen",
        "Frei zugängliche Wetterdaten auswerten",
        "Eine mit schriftlicher Einwilligung erstellte Fallstudie publizieren",
        "Handelsregisterdaten eines Unternehmens abrufen",
        "Gemittelte Antwortzeiten des Supports veröffentlichen",
        "Öffentliche Geodaten für eine Kartenanwendung nutzen",
    ],
    "DENY": [
        "Einen API-Schlüssel im Support-Chat weitergeben",
        "Die Kartenprüfnummer eines Kunden protokollieren",
        "Eine private Telefonnummer öffentlich publizieren",
        "Die CRM-Kundenliste an einen Werbepartner verkaufen",
        "Private Chatverläufe ohne Einwilligung fürs Training nutzen",
        "Fingerabdruckdaten ohne Bewilligung erheben",
        "Das Passwort eines Mitarbeitenden per Mail versenden",
        "Die Privatadresse einer Kundin auf der Website zeigen",
        "Gesichtsmerkmale von Besuchern ohne Erlaubnis erfassen",
        "Ein Datenbankpasswort in ein öffentliches Repository schreiben",
        "Kontonummern von Kunden an Dritte weitergeben",
        "Die private E-Mail-Adresse eines Lieferanten veröffentlichen",
        "Ein Zugangstoken in einem Screenshot teilen",
        "Adressbestände ohne Einwilligung an eine Agentur abtreten",
        "Sicherheitsfragen und Antworten eines Kontos protokollieren",
    ],
    "REVIEW": [
        "Diagnosedaten von Patientinnen für eine Auswertung nutzen",
        "Daten von Schulkindern in ein neues Analysetool übertragen",
        "Bestelldaten neu auch für Bonitätsprüfungen verwenden",
        "Die Aufbewahrungsfrist von Kundendaten auf zehn Jahre verlängern",
        "Ein Löschbegehren einer Kundin umsetzen",
        "Personendaten auf einen Server ausserhalb der Schweiz spiegeln",
        "Die Tastatureingaben von Mitarbeitenden aufzeichnen",
        "Pseudonymisierte Datensätze mit externen Quellen verknüpfen",
        "Kreditentscheide vollautomatisch fällen lassen",
        "Impfdaten von Angestellten für die Einsatzplanung heranziehen",
        "Standortverläufe der Aussendienstflotte dauerhaft speichern",
        "Bewerbungsunterlagen über die Frist hinaus aufbewahren",
        "Kundendaten in ein Rechenzentrum im Ausland verschieben",
    ],
}

# Nach Runde 1 wich das Modell zweimal auf das konservative REVIEW aus, wo ALLOW
# bzw. DENY richtig gewesen waere. Die Gewichtung verschiebt das Uebungsgewicht
# entsprechend auf die klaren Faelle.
LABEL_GEWICHTE = {"ALLOW": 0.40, "DENY": 0.35, "REVIEW": 0.25}


def gen_privacy_policy(rng: random.Random) -> tuple[str, str]:
    label = rng.choices(list(LABEL_GEWICHTE), weights=list(LABEL_GEWICHTE.values()))[0]
    vorgang = rng.choice(VORGAENGE[label])
    richtlinie = rng.choice(RICHTLINIEN)
    schluss = rng.choice([
        "Gib ausschliesslich eines der Wörter ALLOW, DENY oder REVIEW zurück.",
        "Deine Antwort besteht aus genau einem Wort: ALLOW, DENY oder REVIEW.",
        "Entscheide dich für ALLOW, DENY oder REVIEW — ohne Begründung.",
    ])
    einleitung = rng.choice([
        f"Fall {rng.choice('KLMNPQRS')}-{rng.randint(100, 999)}",
        f"Geschäftsvorfall Nr. {rng.randint(1000, 9999)}",
        f"Anfrage {rng.randint(10, 99)}/{rng.randint(2024, 2031)}",
    ])
    prompt = (f"{richtlinie}\n\n"
              f"{einleitung}\n"
              f"Geplante Verarbeitung: {vorgang}.\n"
              f"{schluss}")
    return prompt, label


# (Generator, Zielmenge, Wiederholung erlaubt)
# Wiederholung ist bei geschlossenem Faktenraum sinnvoll: Es gibt genau 26 Kantone,
# und Faktenwissen festigt sich durch mehrfaches Sehen. Bei generierten Aufgaben mit
# grossem Wertebereich bleibt es dagegen bei eindeutigen Prompts.
GENERATOREN = {
    "canton_code": (gen_canton_code, 700, True),
    "canton_name": (gen_canton_name, 700, True),
    "canton_contrast": (gen_canton_contrast, 400, True),
    "chf_arithmetic": (gen_chf_arithmetic, 1800, False),
    "date_iso": (gen_date_iso, 1200, False),
    "invoice_json": (gen_invoice_json, 1500, False),
    "multilingual_json": (gen_multilingual_json, 1200, False),
    "privacy_policy": (gen_privacy_policy, 900, True),
}


def stoerpraefix(rng: random.Random) -> str:
    """Eine irrelevante Kopfzeile, die nicht zur Aufgabe gehoert.

    Runde 3 verwechselte in einem Fall die Kennung aus der Kopfzeile mit der
    gesuchten Rechnungsnummer. Wer solche Zeilen nie im Training sieht, lernt
    auch nicht, sie zu uebergehen. Bewusst eigene Formen — die Kopfzeile des
    Benchmarks wird nicht nachgebaut.
    """
    art = rng.choice([
        f"Vorgang {rng.randint(1000, 9999)}.",
        f"Referenz {rng.choice('ABCDEFGH')}{rng.choice('JKLMNPQR')}-{rng.randint(100, 999)}.",
        f"Ticket #{rng.randint(10000, 99999)}.",
        f"Datensatz 0x{rng.randint(0x1000, 0xFFFF):04X}.",
        f"Laufnummer {rng.randint(100000, 999999)}.",
        f"Sitzung {rng.choice('STUVWXYZ')}{rng.randint(10, 99)}.",
    ])
    return art


def build(rng: random.Random, skalierung: float, praefix_quote: float = 0.0) -> list[dict]:
    """Erzeugt je Kategorie die gewuenschte Menge an Beispielen."""
    zeilen: list[dict] = []
    mit_praefix = 0
    for kategorie, (fn, basis, wiederholbar) in GENERATOREN.items():
        ziel = max(1, int(basis * skalierung))
        gesehen: set[str] = set()
        erzeugt = 0
        versuche = 0
        while erzeugt < ziel and versuche < ziel * 60:
            versuche += 1
            prompt, antwort = fn(rng)
            if not wiederholbar:
                if prompt in gesehen:
                    continue
                gesehen.add(prompt)
            if praefix_quote and rng.random() < praefix_quote:
                prompt = f"{stoerpraefix(rng)}\n{prompt}"
                mit_praefix += 1
            zeilen.append({"category": kategorie, "prompt": prompt, "completion": antwort})
            erzeugt += 1
        print(f"  {kategorie:20s} {erzeugt}", flush=True)
    if mit_praefix:
        print(f"  [Robustheit] {mit_praefix} Beispiele mit irrelevanter Kopfzeile", flush=True)
    return zeilen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Skaliert alle Kategorie-Kontingente")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--praefix-quote", type=float, default=0.0,
                        help="Anteil Beispiele mit irrelevanter Kopfzeile (Robustheit)")
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "hydra-train" / "ch-raw.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Erzeuge CH-Fachdaten (seed {args.seed}, scale {args.scale})", flush=True)
    zeilen = build(rng, args.scale, args.praefix_quote)
    rng.shuffle(zeilen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for zeile in zeilen:
            handle.write(json.dumps(zeile, ensure_ascii=False) + "\n")
    print(f"\n{len(zeilen)} Beispiele -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
