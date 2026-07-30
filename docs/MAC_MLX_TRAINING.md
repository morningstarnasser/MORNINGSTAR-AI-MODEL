# Training auf dem Mac (Apple Silicon / MLX)

Stand 2026-07-25. Ersetzt fuer lokale Laeufe den CUDA-Pfad aus
`cloud/kaggle_qlora_apertus8b.ipynb` — bitsandbytes ist CUDA-only und laeuft auf
dem Mac nicht; MLX quantisiert selbst.

## Kurzfassung

```bash
uv venv ~/hydra-mlx-venv --python 3.12
uv pip install mlx-lm datasets huggingface_hub httpx==0.28.1 uvicorn==0.51.0

# Modell (fertiger MLX-Build von swiss-ai, 2.6 GB — kein Konvertieren noetig)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('swiss-ai/Apertus-v1.1-4B-Instruct-MLX-INT4', \
  local_dir='$HYDRA_MODELS_DIR/apertus-4b-instruct-mlx-int4')"

# Daten (gate-freigegebener Mix, im Template des Modells)
python scripts/build_mlx_dataset.py --total-examples 24000 --template hf \
  --tokenizer $HYDRA_MODELS_DIR/apertus-4b-instruct-mlx-int4 \
  --max-tokens 1536 --out-dir $HYDRA_TRAIN_DIR/data-hf

# Training + Auswertung
HYDRA_MODEL=$HYDRA_MODELS_DIR/apertus-4b-instruct-mlx-int4 \
HYDRA_DATA=$HYDRA_TRAIN_DIR/data-hf scripts/train_mac.sh
HYDRA_MODEL=$HYDRA_MODELS_DIR/apertus-4b-instruct-mlx-int4 scripts/eval_mac.sh both
```

## Gemessen auf M2 Max / 32 GB

| | Apertus-8B-2509 (4bit) | Apertus-v1.1-4B-Instruct (INT4) |
|---|---|---|
| Durchsatz Training | ~90–110 Token/s | **~240–270 Token/s** |
| Zeit pro Iteration | ~15 s | ~6,4 s |
| Peak-Speicher | 9,2 GB | 11,0 GB |
| Trainierbare Parameter | 39,8 M (0,50 %) | 22,8 M (0,60 %) |
| HydraCH-dev untrainiert | unbrauchbar (Basismodell) | **22/50 = 44 %** |

Eine volle Epoche ueber 20k Beispiele waere auf dem 8B mehrtaegig. Der Pilotlauf
faehrt darum 4000 Iterationen (~8000 Beispiele, ~40 % einer Epoche) in rund
7 Stunden.

## 2026-07-28: Der CH-Adapter auf Qwen3-8B wirkt — er war nur nie gemessen worden

Das Checkpoint-Screening vom 25.07. lief über `mlx_lm.server`. Der verwirft
`--adapter-path` still, deshalb lieferten alle vier Checkpoints **bit-identische**
Werte (40/50, bis hin zu `canton_name` 0/5) — es war viermal das nackte
Basismodell. Die daraus gezogene Schlussfolgerung, der Adapter bringe nichts,
hatte keine Grundlage.

Über `scripts/hydra_mlx_server.py` gemessen (HydraCH v2r1-dev, 160 Fälle,
`thinking off`, `temperature 0`):

| System | Ergebnis |
|---|---|
| Qwen3-8B-4bit roh | 122/160 = 76,2 % |
| **+ CH-LoRA (`adapters-qwen3-8b-ch-r1-100`)** | **133/160 = 83,1 %** |

Gepaart: b=3, c=14, **+6,875 pp, CI95 [1,94; 11,81], p=0,0127 — signifikant.**

Der Gewinn sitzt genau dort, wo er sitzen soll: `canton_name` 4→8,
`chf_arithmetic` 5→11, `canton_code` 13→15. `tool_call` bleibt bei 20/20,
`privacy_policy` verliert einen Fall.

### Zwei Runtime-Defekte, die den ersten Anlauf entwertet haben

1. **`<think>` landete im `content`.** Qwen3 stellt jeder Antwort einen
   `<think>`-Block voran, bei abgeschaltetem Denken einen leeren. Der Server gab
   ihn ungetrennt zurück, also verglich jeder exakte Grader gegen
   `"<think>\n\n</think>\n\nZH"` statt gegen `"ZH"`. Ergebnis: 11 von 11 Fällen
   FAIL, obwohl das Modell durchweg richtig lag.
2. **Qwen bekam die Apertus-Tool-Brücke aufgezwungen.** Der Server verpackte
   Werkzeuge immer in Apertus' `developer`-Rolle. Für Qwen, das ein eigenes
   `<tool_call>`-Protokoll hat, brach `tool_call` dadurch von 20/20 auf 8/20 ein.
   Jetzt wird zuerst nativ gerendert und **geprüft, ob die Werkzeugnamen im
   fertigen Prompt stehen**; nur wenn nicht, greift die Sonderbrücke.

**Merke — dritte Auflage derselben Lehre:** Wenn eine ganze Kategorie
zusammenbricht, ist die Runtime verdächtiger als das Modell. Erst messen, ob das
Modell *inhaltlich* richtig liegt, dann über Training nachdenken.

## Bestwert: 88 % — die letzten Punkte kamen aus der Runtime, nicht aus dem Training

| Stufe | HydraCH |
|---|---|
| Basis Apertus-4B | 22/50 = 44,0 % |
| + CH-Adapter (Rang 64) | 41/50 = 82,0 % |
| **+ Tool-Bruecke (`scripts/hydra_mlx_server.py`)** | **44/50 = 88,0 %** |

Die letzten sechs Punkte entstanden **ohne jedes Retraining**, allein durch das
Beheben zweier Infrastrukturfehler in der Tool-Kette:

1. **Die Werkzeuge erreichten das Modell nie.** Der Standardweg reicht `tools=` an
   `apply_chat_template` weiter — Apertus' Template liest sie aber aus einer
   `developer`-Rolle (`content.formatted_tools`). Ueber den Standardweg stand im
   Prompt woertlich „Tool Capabilities: **disabled**".
2. **Korrekte Aufrufe wurden nicht erkannt.** MLX leitet seinen Tool-Parser aus dem
   Template-Text ab; fuer Apertus greift kein bekanntes Muster, also blieb
   `message.tool_calls` leer.

Vor dem Bau lohnt der Gate-Test (Vorschlag von the operator): die betroffenen Faelle roh
dekodieren und pruefen, ob Funktion und Argumente **fachlich** stimmen. Ergebnis
hier: drei Faelle lieferten mit `weather_lookup {"city": "Basel"}` exakt das
Richtige — nur im falschen Format. Damit war klar, dass ein Parser Punkte bringt
und kein Training noetig ist.

Wichtige Folge: Die Obergrenze liegt nicht mehr bei 44/50, sondern bei **50/50**.
Die verbleibenden sechs Fehler sind drei Tool-Faelle (das Modell lehnt ab, statt
`calendar_lookup` aufzurufen), eine Kantonsverwechslung, ein `invoice_json` und ein
`privacy_policy`.

## Ein Modell kann nicht beides — Adapter-Switching statt Mischtraining

| Modell | HydraCH | GSM8K ohne Beispiele | GSM8K mit Beispielen |
|---|---|---|---|
| Basis Apertus-4B | 44,0 % | 27,5 % | **46,7 %** |
| CH-only (Rang 64) | **82,0 %** | 9,2 % | 35,0 % |
| Mischtraining (CH + Reasoning) | 76,0 % | 20,8 % | 26,7 % |

**Das Mischtraining ist gescheitert:** Es erreicht in keiner Dimension den Bestwert.
HydraCH faellt gegenueber dem Spezialmodell (76 statt 82), und das Reasoning bleibt
klar unter der Basis. Die Empfehlung lautet daher **Adapter-Switching**: Basis im
Speicher halten, den CH-Adapter nur an den Pipeline-Schritten aktivieren, an denen
ohnehin feststeht, dass eine Formataufgabe kommt. Dann stehen 82 % und 46,7 %
gleichzeitig zur Verfuegung, ohne Balanceakt.

### Wichtig fuer jede Reasoning-Messung: Stil ist nicht Faehigkeit

Die erste Messung ergab fuer das CH-Modell 9,2 % und legte nahe, das Reasoning sei
zerstoert. Mit zwei ausgerechneten Beispielen im Prompt springt dasselbe Modell auf
**35,0 %**. Der Grossteil des Einbruchs war also **Antwortstil, nicht verlorene
Faehigkeit**: Das Modell hatte gelernt, knapp zu antworten, und begann Rechenwege
gar nicht erst. Der reale Verlust betraegt unter fairen Bedingungen rund 12
Prozentpunkte (46,7 → 35,0) statt der zunaechst gemessenen 18.

Ursache im Datensatz: In den CH-Daten ist „Domaene" identisch mit „kurze Antwort" —
diese Korrelation wird mitgelernt. Wer das vermeiden will, muss **Laenge
konditionieren** (System-Prompt, der Kuerze bzw. Schritte verlangt) und die
Korrelation brechen (ein Teil der CH-Faelle mit sichtbarem Rechenweg).

Messhygiene: Bei 50 HydraCH-Faellen betraegt der Standardfehler rund ±6
Prozentpunkte, bei 120 GSM8K-Aufgaben ±4. Unterschiede unter etwa 8 Punkten sind
Rauschen. Fuer belastbare Pareto-Vergleiche braeuchte HydraCH 150–200 Faelle.
Geprueft wurde ausserdem, dass die Reasoning-Trainingsdaten keine GSM8K-Testfragen
enthalten (0 exakte Treffer, 0 Near-Dups).

## Zweiter Lauf (CH-Fachdaten): 44 % → 76 %

| | HydraCH-dev |
|---|---|
| Basis ohne Adapter | 22/50 = 44,0 % |
| **CH-Adapter (200 Iterationen)** | **38/50 = 76,0 %** |
| CH-Adapter (400 Iterationen) | 38/50 = 76,0 % (identisch) |

Da `tool_call` strukturell blockiert ist (siehe unten), sind real 44 Faelle
erreichbar — **38/44 = 86,4 %** davon werden getroffen.

| Kategorie | vorher | nachher |
|---|---|---|
| chf_arithmetic | 2/8 | **7/8** |
| invoice_json | 5/8 | **8/8** |
| multilingual_json | 4/7 | **7/7** |
| canton_code | 2/5 | **4/5** |
| canton_name | 1/5 | **3/5** |
| privacy_policy | 2/5 | **3/5** |
| date_iso | 6/6 | 6/6 |
| tool_call | 0/6 | 0/6 (blockiert) |

Keine Kategorie regressiert — `train_only_if` aus dem Gate ist damit erfuellt.
Das Training dauerte rund acht Minuten (200 Iterationen), der Val-Loss fiel von
0,846 auf 0,028. Checkpoint 400 bringt nichts mehr: Nach etwa 800 gesehenen
Beispielen ist alles gelernt, was dieser Datensatz hergibt.

### Messfalle, die fast alles entwertet haette

`mlx_lm.server` (0.31.3) wendet `--adapter-path` **nie** an. Beim Start ruft er
`load("default_model", None, ...)`, und `_adapter_map.get(None)` ergibt `None` —
der Adapter faellt still weg. Eine Evaluation ueber den Server misst deshalb
immer das Basismodell. Genau das erklaert, warum drei voellig verschiedene
Adapter zuvor identische 44,0 % lieferten.

Erkennbar war es am A/B-Test mit `mlx_lm.generate`: dieselbe Anfrage lieferte
ohne Adapter einen Markdown-Codeblock (den der `json_exact`-Grader nicht parsen
kann) und mit Adapter sauberes JSON. **Loesung: Adapter per `mlx_lm.fuse` ins
Modell backen und das fusionierte Modell evaluieren** — das entspricht ohnehin
dem Produktivpfad.

### Dritte Iteration (8400 Beispiele): 78 %

Runde 2 des Datensatzes ergaenzte kontrastive Uebungen fuer verwechselbare
Kuerzelpaare (SG/SZ, ZG/ZH und sieben weitere), mehr Subtraktionen mit Ergebnis
nahe null und eine Labelgewichtung von 40 % ALLOW / 35 % DENY / 25 % REVIEW.

| | HydraCH-dev |
|---|---|
| Basis | 22/50 = 44,0 % |
| Runde 1 (6464 Beispiele) | 38/50 = 76,0 % |
| **Runde 2 (8400 Beispiele)** | **39/50 = 78,0 %** — 39/44 = **88,6 %** der erreichbaren |

Gewirkt hat davon **nur** die Arithmetik: `chf_arithmetic` steigt von 7/8 auf
**8/8**, die Subtraktionen nahe null sind behoben. Kantone und Richtlinien
bewegen sich **nicht** — es sind exakt dieselben fuenf Faelle wie zuvor.

### Die Grenze: Faktenwissen generalisiert nicht ueber die Frageform

Trotz 1800 Kantonsbeispielen (700 Kuerzel, 700 Namen, 400 kontrastiv) bleiben
SG → „Schwyz", ZG → „Zürich" und Zug → „Zug" falsch. Bei anderen Kantonen sitzt
es. Der Unterschied liegt in der Frageform: Der Benchmark fragt „Gib
ausschliesslich den deutschen Kantonsnamen für das Kürzel SG aus", das Training
nutzt bewusst andere Formulierungen. Das Modell hat die Fakten also gelernt,
ruft sie bei dieser Formulierung aber nicht ab — und faellt dann auf seinen
Prior zurueck, der aehnliche Kuerzel verwechselt.

Die naheliegende „Loesung" waere, die Benchmark-Formulierung ins Training zu
uebernehmen. Das ist **bewusst unterlassen**: Bei diesen kurzen Prompts läge der
Shingle-Overlap ueber 90 %, der Kontaminationscheck wuerde zu Recht anschlagen,
und ein Punktgewinn waere reines Auswendiglernen. Wer die Kategorie echt
verbessern will, braucht mehr Kapazitaet (hoeherer LoRA-Rang) oder breitere
Formulierungsvielfalt — nicht die Testfrage.

### Verbleibende Fehler nach Runde 1

Sechs Faelle ausserhalb `tool_call`: zwei Kantonsverwechslungen bei aehnlichen
Kuerzelpaaren (SG → „Schwyz" statt „St. Gallen", ZG → „Zürich" statt „Zug"),
einmal wurde statt des Kuerzels der Name ausgegeben (Zug → „Zug"), eine
Subtraktion mit kleinem Ergebnis (106.16 − 102.96 → „CHF 0.00"), und zweimal
wich das Modell auf das konservative REVIEW aus, wo ALLOW bzw. DENY richtig
gewesen waere. Alles adressierbar: kontrastive Kuerzelpaare, mehr Differenzen
nahe null, ausgewogenere Policy-Labels.

## Erster Lauf (generischer Mix): keine Verbesserung — und warum

| | HydraCH-dev |
|---|---|
| Basis ohne Adapter | 22/50 = **44,0 %** |
| Adapter iter 600 (bester Val-Loss) | 22/50 = **44,0 %** |
| Adapter iter 1400 | 22/50 = **44,0 %** |

Identisch in **jeder** Kategorie. Der Adapter wird dabei nachweislich geladen und
wirkt — dieselbe Anfrage liefert mit Adapter kompakteres JSON mit `\u`-Escapes —
er trifft die geprueften Kriterien nur nicht besser.

**Ursache: der Datenmix hat mit dem Benchmark praktisch keine Ueberschneidung.**
In den 20 074 Trainingsbeispielen kommt vor: `Rappen` 0×, `Kanton` 0×, `Zürich` 0×,
`revDSG` 0×, `CHF` 1×; rund 0,1 % sind deutschsprachig. HydraCH prueft dagegen
41 der 50 Faelle auf **de-CH** (plus fr-CH/it-CH) und fragt Kantone,
Rappen-als-Ganzzahl, ISO-Daten und revDSG-Formulierungen ab. Die sechs Quellen
trainieren englischsprachiges Code-, Math- und Agent-Verhalten.

Das ist keine Frage der Rechenleistung: Derselbe Mix haette auf T4 oder RunPod
dasselbe Nullergebnis geliefert, nur schneller und teurer.

Der Val-Loss bestaetigt das Bild: 1,681 → 1,615 (iter 600, Minimum) → 1,676
(iter 1200), waehrend der Train-Loss weiter auf 1,398 faellt. Klassisches
beginnendes Overfitting auf Inhalte, die der Benchmark nicht abfragt. Der Lauf
wurde bei iter 1400 abgebrochen statt bis 4000 zu rechnen.

**Was es fuer HydraCH braucht:** CH-spezifische Trainingsdaten — Kantonsnamen und
-kuerzel, Rappen-Arithmetik, ISO-Datumsformate, revDSG-Formulierungen, in
de-CH/fr-CH/it-CH. Synthetisch gut erzeugbar, **aber zwingend gegen HydraCH
dev+hidden auf Kontamination pruefen**, sonst misst der Benchmark sich selbst.

## Warum nicht das Modell aus dem Kaggle-Notebook

Das Notebook waehlt `swiss-ai/Apertus-8B-2509` — ein **reines Basismodell ohne
Instruction-Tuning**. Es beantwortet Fragen nicht, sondern setzt den Text fort
(auf "Nenne die Hauptstadt der Schweiz" folgte Gestammel ueber Argentinien), und
muesste Chat-Verhalten erst von Null lernen. Inzwischen gibt es eine neuere
Generation (v1.1) samt Instruct-Varianten und fertigen MLX-Builds. Der Adapter
verbessert damit etwas Funktionierendes statt Grundlagen nachzuholen.

## Stolperfallen

**Chat-Template nicht selbst nachbauen.** Apertus-Instruct bringt eine
`chat_template.jinja` mit (nicht in `tokenizer_config.json` — dort steht kein
Template, was leicht zu einem Fehlschluss fuehrt). Das Format setzt ein
bos-Token, nutzt `<SPECIAL_61..72>` als Rollenmarker und schiebt **immer** einen
`developer`-Block ein:

```
<s><SPECIAL_61><SPECIAL_62><SPECIAL_63>Deliberation: disabled
Tool Capabilities: disabled<SPECIAL_64><SPECIAL_65>{user}<SPECIAL_66><SPECIAL_67>{assistant}<SPECIAL_68>
```

Ein handgebautes ChatML trainiert daran vorbei. `--template hf` laesst den
Tokenizer rendern.

**Die Rolle `tool` lehnt das Template ab** (`Invalid message role: tool`) — ohne
Gegenmassnahme fallen alle Beispiele aus `hermes-function-calling-v1/func_calling`
durch. `sanitize_for_template()` faltet Tool-Ergebnisse in den User-Turn.

**Sequenzbudget statt Truncation.** Mit flachem 1024-Budget wurden 50,5 % der
Beispiele mitten in der Antwort abgeschnitten — sie verlieren ihr Turn-Ende, das
Modell lernt also, nicht zu terminieren. `--max-tokens` verwirft zu lange
Beispiele stattdessen.

**Tokenbudget pro Batch klein halten.** Das Vokabular hat 131 072 Einträge, der
Logit-Tensor kostet rund 0,5 GB pro 1000 Tokens. `batch_size 4` bei Sequenzlaenge
1536 trieb den Mac in den Swap und praktisch zum Stillstand; `batch_size 2` mit
`grad_accumulation_steps 8` liefert dieselbe effektive Batchgroesse ohne Swap.

**MLX' `scale` ist nicht PEFTs `alpha/rank`,** sondern ein absoluter
Multiplikator. Die erprobte Kopplung ist `scale 20` + `lr 1e-5`; Kaggles
`lr 2e-4` waere hier rund zwanzigfach zu heiss.

**Beim Evaluieren muss der Modellname im Request der geladene Pfad sein.**
`mlx_lm.server` nimmt das `model`-Feld wörtlich und laedt einen abweichenden
Namen von HuggingFace nach — Ergebnis waren 50× HTTP 404.

## `tool_call` ist über diesen Pfad nicht erreichbar (verifiziert)

Die Kategorie `tool_call` (6 der 50 Faelle) kann mit `mlx_lm.server` + Apertus
**nicht** bestehen — unabhaengig davon, wie gut trainiert wird. Damit liegt die
Obergrenze bei **44/50 = 88 %**.

Grund: Der Grader liest `message.tool_calls` aus der Antwort
(`scripts/evaluate_hydrach_bench.py`, ~Zeile 615). Damit `mlx_lm.server` dieses
Feld fuellt, braucht der Tokenizer einen Tool-Parser, den MLX per
`_infer_tool_parser(chat_template)` aus dem Template-**Text** ableitet. Geprueft:

```python
from mlx_lm.tokenizer_utils import _infer_tool_parser
_infer_tool_parser(open("…/chat_template.jinja").read())   # -> None
```

Apertus markiert Tool-Calls mit `<SPECIAL_71>`/`<SPECIAL_72>`; keines der von MLX
erkannten Muster (`[TOOL_CALLS]`, `<tool_call>`, `<|tool_list_start|>` …) trifft zu.
Das Template enthaelt zwar `tool_call.name`, aber der `json_tools`-Zweig verlangt
zusaetzlich `<tool_call>` — die UND-Bedingung greift nicht. Ohne Parser ist
`has_tool_calling` false, Tool-Aufrufe kommen als reiner Text zurueck.

Loesungswege, falls die Kategorie zaehlen soll: einen Tool-Parser fuer Apertus in
mlx-lm ergaenzen, oder das Template auf das `[TOOL_CALLS]`-Token umstellen (Apertus
hat es als Token-ID 9 im Vokabular) — dann greift der `mistral`-Parser. Beides ist
eine eigene Baustelle und war nicht Teil dieses Laufs.

## Lizenz / Compliance

Apertus ist Apache-2.0, also kommerziell nutzbar. Die Acceptable Use Policy
verlangt allerdings, dass man die von SNAI bereitgestellte Hash-Liste
(Datenschutz-Loeschantraege) etwa halbjaehrlich als **Output-Filter** anwendet.
Fuer die Hydra-API mit externen Keys ist das einzuplanen.
