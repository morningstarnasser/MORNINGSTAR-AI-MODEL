# Hydra gegen Gemma 4 auf HydraCH

Messlatte für die Frage „ist unser Modell besser als ein aktuelles offenes
Modell derselben Klasse". Stand 2026-07-28.

## Aufbau

- **Fälle:** HydraCH **v2r2-hidden**, 320 Fälle, 8 Kategorien à 40
  (`tool_call`: 24 positiv + 16 negativ). Frisch aus einem neuen Seed erzeugt,
  weil v2r1-dev durch die eigene Entwicklung verbraucht war — Server-Fix,
  Adapterwahl und Budget wurden dort entschieden.
- **Policy für beide identisch:** `temperature 0`, Denken **aus**, 64
  Ausgabetokens für die kurzen Kategorien, 512 für `invoice_json`,
  `multilingual_json` und `tool_call`.
- **Überlappung:** 38 der 320 Prompttexte kommen auch im alten dev-Split vor
  (`_privacy_cases()` nimmt keinen Zufallsgenerator, ist also seed-unabhängig).
  Das trifft beide Modelle gleich.

## Ergebnis

| System | Grösse | bestanden | Anteil |
|---|---|---|---|
| **Qwen3.5-27B-int4** (Ollama) | 27B dense | **287/320** | **89,7 %** |
| Gemma 4 12B-it (Ollama q4) | 12B dense | 271/320 | 84,7 % |
| Qwen3-14B roh (Ollama q4) | 14B dense | 269/320 | 84,1 % |
| Qwen3-8B-4bit + CH-LoRA (MLX) | 8B dense | 263/320 | 82,2 % |

**Qwen3.5-27B schlägt Gemma 4 klar und belegbar.** Gepaart über alle 320 Fälle:
diskordant b=10, c=26, **+5,00 pp, p=0,0113 → signifikant.**

Die Aussage hält auch ohne die 38 Fälle, deren Prompttext im alten dev-Split
vorkommt: auf den verbleibenden 282 Fällen 253 gegen 238, **+5,32 pp, p=0,0135.**

| Kategorie | Qwen3.5-27B | Gemma 4 12B |
|---|---|---|
| **chf_arithmetic** | **27/40** | 16/40 |
| **canton_name** | **24/40** | 20/40 |
| **privacy_policy** | **40/40** | 37/40 |
| date_iso | 40/40 | 40/40 |
| multilingual_json | 40/40 | 40/40 |
| tool_call | 40/40 | 40/40 |
| canton_code | 37/40 | 38/40 |
| invoice_json | 39/40 | 40/40 |

Der Vorsprung kommt aus dem Rechnen (+11), dem Kantonsnamen (+4) und den
Datenschutzregeln (+3); Rückstand gibt es nur bei zwei Kategorien mit je einem
Fall.

**Ehrlich dazugesagt:** 27B gegen 12B ist kein Vergleich gleicher Grösse. Wer
Parameter gegen Parameter stellen will, muss den 14B-Wert nehmen — und der ist
mit 269 gegen 271 ein Gleichstand. Der 27B-Vergleich beantwortet die andere,
für ein lokales Produkt relevantere Frage: **Was ist das Beste, das auf dieser
Maschine läuft?** Beide Modelle laufen auf demselben M2 Max mit 32 GB, durch
dieselbe Ollama-Kette, mit derselben Policy.

### Zwischenstand vor dem Modellwechsel

Mit dem eigenen 8B-Modell war Gemma vorn: 271 gegen 263, also acht Fälle. Der
CH-LoRA hebt dort zwar messbar (auf v2r1-dev 122→133, p=0,0127), reicht aber
nicht, um zwei Modellgenerationen und den Grössenunterschied aufzuholen.

| Kategorie | Hydra | Gemma 4 |
|---|---|---|
| date_iso | 40/40 | 40/40 |
| multilingual_json | 40/40 | 40/40 |
| **chf_arithmetic** | **26/40** | 16/40 |
| tool_call | 39/40 | 40/40 |
| invoice_json | 38/40 | 40/40 |
| privacy_policy | 34/40 | 37/40 |
| canton_code | 33/40 | 38/40 |
| **canton_name** | 13/40 | **20/40** |

Das Bild ist klar aufgeteilt: **Bei der Rechenkategorie sind wir deutlich besser
(+10 Fälle)** — dort wirkt der CH-LoRA. **Beim Faktenwissen zu Kantonen verlieren
wir 12 Fälle.** Der Rest ist Gleichstand oder ein bis drei Fälle Rückstand.

## Dritter Messpunkt: Qwen3-14B roh, gleiche Ollama-Kette wie Gemma

Damit der Runtime-Unterschied (MLX gegen Ollama) nicht die Aussage trägt, lief
zusätzlich ein Qwen3-14B durch **dieselbe** Ollama-Kette wie Gemma.

| System | Gesamt |
|---|---|
| Gemma 4 12B | 271/320 = 84,7 % |
| **Qwen3-14B roh** | **269/320 = 84,1 %** |
| Hydra: Qwen3-8B + CH-LoRA | 263/320 = 82,2 % |

Zwei Fälle Unterschied zwischen Qwen3-14B und Gemma — das ist Gleichstand, kein
Rückstand. Interessant ist die Aufteilung:

| Kategorie | Qwen3-14B | Gemma 4 | Hydra 8B+LoRA |
|---|---|---|---|
| canton_code | 26 | **38** | 33 |
| canton_name | 16 | **20** | 13 |
| chf_arithmetic | **28** | 16 | 26 |
| privacy_policy | **40** | 37 | 34 |
| invoice_json | 39 | **40** | 38 |
| tool_call | **40** | **40** | 39 |
| date_iso / multilingual_json | 40 / 40 | 40 / 40 | 40 / 40 |

**Die beiden Modelle sind komplementär, nicht besser oder schlechter.** Gemma
gewinnt das Kantonswissen (58 gegen 42 Fälle), Qwen gewinnt Rechnen und
Datenschutzregeln (68 gegen 53). Unterm Strich hebt sich das fast auf.

Der wunde Punkt auf unserer Seite ist damit klar benannt: **Kantonswissen.**

## Warum `canton_name` bei uns nur 13/40 schafft

Zwei getrennte Fehlerarten, beide im gespeicherten Rohtext nachgelesen:

1. **Falsche Sprachvariante oder fehlender Umlaut** — `vaud` statt `Waadt`,
   `zurich` statt `Zürich`. Das Modell kennt den Kanton, schreibt ihn aber nicht
   so, wie der Vertrag es verlangt (deutscher Name).
2. **Der LoRA hat eine Standardantwort gelernt** — auf „Ergänze zum Code AG /
   AI / BL den ausgeschriebenen deutschen Kantonsnamen" kommt dreimal `bern`.

Bemerkenswert: **Gemma scheitert an derselben Frageform** (`bern`, `zürich` für
AG, AI, BL). Beide Modelle beherrschen „Das Kürzel X steht für welchen Kanton?"
deutlich besser als „Ergänze zum Code X". Das ist ein Hinweis, dass die zweite
Formulierung ungewöhnlich ist — kein Grund, sie zu entfernen, aber einer, sie
beim Auswerten getrennt zu betrachten.

## Warum die Control-Plane hier nicht der Ausweg ist

Die deterministische Kantonstabelle würde `canton_code`, `canton_name` und
`chf_arithmetic` auf annähernd 40/40 heben. Sie ist aber **modellunabhängig** —
sie würde Gemma genauso helfen. Durchgerechnet mit Control-Plane für beide:
Hydra ≈ 311/320, Gemma ≈ 317/320. Sie egalisiert unsere einzige klare Stärke
(Rechnen) und ändert die Rangfolge nicht.

Eine Control-Plane nur auf unserer Seite wäre kein Modellvergleich mehr, sondern
System gegen Modell. Wer so etwas berichtet, muss es genau so benennen.

**Nachtrag 2026-07-28:** Die Erkennung wurde von Wortlisten auf eine Regel
umgestellt — die Richtung folgt aus dem, was gegeben ist. Auf einem **frisch
gezogenen Split** (v2r3, neuer Seed, beim Bauen nicht angesehen): `canton_code`
40/40, `canton_name` **40/40** statt 29/40, alle Antworten korrekt, null
Fehlauslösungen. Dabei kam eine Fehlauslösung ans Licht: Auf „Nenne mir den
Hauptort des Kantons Wallis" antwortete die Tabelle mit `VS` — die richtige
Antwort auf eine andere Frage. Sie schweigt jetzt, wenn ein Attribut verlangt
wird, das sie nicht hält.

An der Rangfolge gegen Gemma ändert das nichts: Die Control-Plane wirkt
modellunabhängig und hilft beiden gleich.

**Wie gut sie traf, war auch vorher gemessen** (auf allen 320 Fällen, ohne
Modellaufruf): `canton_code` 40/40 erkannt, `canton_name` 29/40, `chf_arithmetic`
40/40 — und **alle 109 erkannten Fälle korrekt beantwortet**. In den fünf
übrigen Kategorien löst sie **kein einziges Mal** aus. Sie ist also präzise; das
Problem ist nicht ihre Qualität, sondern dass sie modellunabhängig wirkt.

## Fallen beim Messen von Gemma 4

**Gemma 4 ist ein Denkmodell.** Mit dem 64-Token-Budget der Benchmark-Policy
liefert es eine Antwort **ganz ohne `content`**: Das Budget geht vollständig für
den Denkteil drauf, `finish_reason` ist `length`. Die ersten vier Fälle des
ersten Laufs scheiterten genau daran — hätte man das nicht bemerkt, wäre Gemma
grundlos schlechtgemessen worden.

**Ollamas `/v1`-Endpunkt ignoriert sowohl `think` als auch
`chat_template_kwargs`.** Das Denken lässt sich dort nicht abschalten. Deshalb
gibt es `scripts/ollama_openai_proxy.py`: Er übersetzt auf `/api/chat`, reicht
`think` durch und normalisiert Tool-Aufrufe ins OpenAI-Format.

**Die MLX-Portierungen von Gemma 4 sind grösstenteils unbrauchbar** für diesen
Zweck: `mlx-community/gemma-4-12B-it-OptiQ-4bit` meldet `Model type
gemma4_unified not supported`, und `lmstudio-community/gemma-4-E4B-it-MLX-4bit`
ist multimodal (`language_model.*`-Gewichte), was `mlx_lm` nicht laden kann.
Praktikabel ist `ollama pull gemma4:12b`.

## Einschränkung

Hydra läuft auf MLX, Gemma auf Ollama/llama.cpp — verschiedene Laufzeiten und
verschiedene 4-Bit-Verfahren. Verglichen werden damit **konkret deploybare
Artefakte**, nicht Architekturen unter Laborbedingungen. Für einen Vergleich
ohne diesen Störfaktor muss ein Qwen-Modell durch dieselbe Ollama-Kette laufen.
