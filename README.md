<div align="center">

# ★ Morningstar Hydra

### Swiss AI control‑plane & local Ollama model

A **CreativeSync** local-model project

[![License](https://img.shields.io/badge/License-Apache_2.0-3B82F6?style=flat-square&logo=apache&logoColor=white)](https://opensource.org/licenses/Apache-2.0)
[![Runtime](https://img.shields.io/badge/Runtime-Ollama-8B5CF6?style=flat-square)](docs/OLLAMA_MODELS.md)
[![Benchmark](https://img.shields.io/badge/HydraCH--Bench_v2-89.7%25-22C55E?style=flat-square)](docs/VERGLEICH-GEMMA4.md)

</div>

---

## Overview

Morningstar Hydra is a **control plane and OpenAI‑compatible API** for reliable Swiss business
AI: German (including Swiss German understanding), French, Italian and English documents,
deterministic JSON and function calling, and privacy‑aware SME workflows — email, calendar,
invoicing, contracts and support.

The public API and its former portal are retired. The supported product surface is local
Ollama; the router and bridge remain available for local development and private deployments.

## Capabilities

- **OpenAI‑compatible `/v1/*`** with buffered SSE streaming — existing clients work unchanged.
- **Structured output discipline** for API requests: raw JSON without markdown fences,
  ISO‑8601 dates, exact values, and native `tool_calls`. Ordinary prose chat is unaffected.
- **Swiss domain handling**: cantons, CHF arithmetic, Swiss date formats, revDSG‑aware
  treatment of personal data.
- **Resilient routing**: transient upstream `429`/`5xx` failures are retried with exponential
  backoff and jitter; reasoning backends that reject a forced `tool_choice` are retried once
  with `auto` so tool calling keeps working. A local `Qwen2.5-Coder-14B-Instruct` Q4_K_M path
  on a Synology DS1522+ serves as an outage safety net.
- **Enforced product identity** in the control plane, and external‑action claims require a
  confirming tool result in the same conversation.
- **14,769,689,600 active parameters per request.** The manifest describes a planned
  275‑family expert catalog; catalog‑scale claims stay locked in code until every family has a
  verified artifact and benchmark.

## Local runtime, one code base

| Where | any Mac, PC or Pi |
| Backend | Ollama (local model inventory in [`docs/OLLAMA_MODELS.md`](docs/OLLAMA_MODELS.md)) |
| Start | `./scripts/hydra-up.sh` |

The local chain is Ollama → bridge (`scripts/ollama_openai_proxy.py`) → router
(`scripts/hydra_router_server.py`). The bridge is not optional: Ollama's own `/v1` ignores
`think`, so a reasoning model cannot be told to stop thinking — and with a tight `max_tokens`
it returns an answer with **no `content` at all**, the whole budget spent on the thinking part.

`hydra-up.sh` picks the model from the hardware it finds. A GPU decides first — Apple Silicon
via Metal, NVIDIA via CUDA. Without one it stays on the 4B model regardless of RAM: a
Raspberry Pi 5 has 15 GB but four CPU cores, where a 14B model fits in memory and delivers
roughly one token per second.

| Accelerator memory (VRAM, else RAM) | Model |
|---|---|
| 24 GB and up | `qwen3.5:27b-int4` |
| 14 GB and up | `qwen3:14b` |
| below, or no GPU | `qwen3.5:4b` |

Binding beyond loopback **requires an API key store** — the router refuses to start otherwise.
An unauthenticated model endpoint on the network is not a state you should be able to reach by
accident. Keys are stored as SHA‑256 hashes and revocable; `/health` stays open for monitoring.

## HydraCH‑Bench v2

The v1 instrument had 50 dev cases; at that size the standard error is ±4.6 pp and differences
below ~9 pp are not resolvable. **v2** raises this to 160 dev / 320 hidden, adds real negative
cases for `tool_call`, states type contracts in the prompt, and compares paired (McNemar)
instead of comparing independent pass rates.

Measured on 320 freshly generated hidden cases, identical policy for every system,
same machine, same runtime:

| System | Size | HydraCH v2 |
|---|---|---|
| **`qwen3.5:27b-int4`** (current base) | 27B | **287/320 = 89.7 %** |
| Google Gemma 4 12B‑it | 12B | 271/320 = 84.7 % |
| Qwen3‑14B | 14B | 269/320 = 84.1 % |
| Qwen3‑8B + own CH adapter | 8B | 263/320 = 82.2 % |

Against Gemma 4: paired b=10, c=26, **+5.00 pp, p = 0.0113** — and the claim survives dropping
the 38 cases whose prompt text also occurs in an earlier split (+5.32 pp, p = 0.0135).

**Stated plainly:** 27B against 12B is not a like‑for‑like comparison. Size‑matched (14B against
12B) it is a **draw**, and the own 8B model **loses** to Gemma. What the 27B number answers is
the question that matters for a local product: what is the best thing that runs on this machine.

General reasoning has not suffered from the move: GSM8K few‑shot **115/120 = 95.8 %**, against
111/120 = 92.5 % for the previous 8B base.

## Own Swiss domain models (July 2026) — trained, measured, currently not in service

Two own LoRA adapters exist and both work. Neither is in service, because the base they were
built on has been overtaken.

The **Qwen3‑8B CH adapter** lifts exactly what it should — cantons and CHF arithmetic — and the
gain is real: 122 → 133 of 160, paired **p = 0.0127**. It took a while to see that, because the
July screening ran through `mlx_lm.server`, which silently discards `--adapter-path`: all four
checkpoints returned bit‑identical base numbers, and the conclusion "the adapter does nothing"
rested on nothing at all.

The **Apertus‑4B CH adapter** is the cautionary tale, and the reason a reasoning gate now runs
before and after every training run:

| Model | HydraCH‑dev (v1) | GSM8K few‑shot |
|---|---|---|
| Apertus‑v1.1‑4B‑Instruct, untouched | 22/50 = 44 % | 46.7 % |
| **Apertus‑4B + CH adapter** | **46/50 = 92 %** | **35 %** |

Swiss knowledge nearly doubled while general reasoning fell by a third. A model can be very
good at the narrow thing and quietly lose the broad one — which is why HydraCH alone is never
enough to accept an adapter.

Base is [`swiss-ai/Apertus-v1.1-4B-Instruct`](https://huggingface.co/swiss-ai/Apertus-v1.1-4B-Instruct)
(Apache‑2.0), fine‑tuned on 8,400 self‑generated Swiss domain examples.

A four‑billion‑parameter model on a laptop gets remarkably close to a much larger hosted backend
on Swiss domain work. It is a cheap local domain worker, not a replacement for a large model —
see the reasoning figures above.

Every training answer is independently recomputed by a separate checker, and the dataset is
verified against the benchmark for leakage — against the public dev set *and* the hidden set,
with zero hits.

```bash
uv venv ~/hydra-mlx-venv --python 3.12 && \
  uv pip install mlx-lm datasets httpx==0.28.1 uvicorn==0.51.0
python scripts/build_ch_dataset.py --praefix-quote 0.35   # generate
python scripts/check_ch_dataset.py                        # recompute every answer
python -m mlx_lm lora --config config/mlx-lora-ch3.yaml   # train (~25 min on M2 Max)
scripts/eval_mac.sh base                                  # measure
```

`scripts/hydra_mlx_server.py` replaces `mlx_lm.server` for Apertus: the stock server never
applies `--adapter-path`, passes tools in a way the Apertus template never sees, and has no
parser for Apertus' call format. Fixing those three alone was worth **8 percentage points with
no retraining**. Full write‑up: [`docs/MAC_MLX_TRAINING.md`](docs/MAC_MLX_TRAINING.md).

> **Measurement limit.** At n=50 the standard error is ±4.6 pp, so differences under ~9 pp are
> not resolvable, and the dev cases have been used for selection across several rounds. Before
> further training runs the benchmark should grow to 150 dev / 300 hidden with ≥20 cases per
> category and negative cases for tool calling.

## Data policy

The curated post‑training source policy lives in
[`config/hf-expert-mixture.json`](config/hf-expert-mixture.json): permissively licensed
sources only, no gated datasets, no sources without a declared license.

A full sample‑level contamination gate checks candidate training data against the evaluation
universe for exact content overlap, near‑duplicates and upstream provenance before any
training is permitted. The current signed report covers **436,565 rows with zero overlap**:

## Local API & keys

`/v1/*` endpoints accept hashed, revocable `msh_live_*` Bearer keys (SHA‑256 store, atomic
writes, `0600`). Authentication fails closed, and any non‑loopback bind requires a configured
key. Keys are managed locally; there is no public endpoint or portal.

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H "Authorization: Bearer $MSH_LIVE_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"morningstar/hydra-1.0-v","messages":[{"role":"user","content":"Grüezi"}]}'
```


## Roadmap


1. **Quality & local runtime** — benchmark, hardened router and Ollama integration. ✅
2. **Data gate** — licenses, provenance, contamination verification. ✅
3. **Model layer** — evaluate self‑hostable candidates on HydraCH and adopt or fine‑tune based
   on measured results. ✅ Own CH domain adapter at 92 % HydraCH; see
   [`docs/MAC_MLX_TRAINING.md`](docs/MAC_MLX_TRAINING.md).
4. **Swiss public beta** — evaluation and inference in Switzerland, model card, signed
   artifacts, revDSG privacy package.

Hydra is a control plane over a model layer, not an original foundation model; the model layer
is stated transparently in the model card at release.

## Documentation

- [`docs/MAC_MLX_TRAINING.md`](docs/MAC_MLX_TRAINING.md) — training the own domain model on Apple Silicon

## License

[Apache‑2.0](LICENSE). Upstream models and datasets keep their own licenses; base weights and
dataset licenses are pinned before any redistribution.
