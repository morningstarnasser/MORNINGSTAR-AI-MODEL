#!/usr/bin/env python3
"""Builds the Kaggle QLoRA notebook for the Apertus-8B Hydra pilot."""
import json, pathlib

MD_INTRO = r"""# ★ Morningstar Hydra — Apertus-8B QLoRA Pilot (Kaggle)

**Vor dem Start (einmalig):**
1. **Settings → Accelerator → `GPU T4 x2`** (nicht P100 — Pascal kann kein 4-bit QLoRA).
2. **Settings → Internet → `On`** (für HuggingFace-Downloads).
3. **Add-ons → Secrets → neu:** Name `HF_TOKEN`, Wert = dein HuggingFace-Token (Account **kurdman991**, mit **Write**-Rechten).
4. Dann oben **`Run All`**.

Der Adapter wird alle paar hundert Schritte automatisch zu HuggingFace gepusht. **Bricht die Session ab, verlierst du nichts** — Notebook neu öffnen, `Run All`, es setzt am letzten Hub-Checkpoint fort.

Das Datengate ist bereits grün (436.565 Zeilen gegen die Eval-Universe geprüft, 0 Kontamination), diese Quellen sind freigegeben."""

CFG = r'''# ============ Konfiguration (bei Bedarf anpassen) ============
BASE_MODEL     = "swiss-ai/Apertus-8B-2509"   # Fallback bei Ladeproblemen: "Qwen/Qwen2.5-7B-Instruct"
HUB_ADAPTER    = "kurdman991/morningstar-hydra-apertus8b-qlora-pilot"
TOTAL_EXAMPLES = 24000     # Pilot ~20-30M Tokens; hochskalieren wenn Zeitbudget passt
MAX_SEQ_LEN    = 1024      # T4 16GB: 1024 sicher
EPOCHS         = 1
LR             = 2e-4
SAVE_STEPS     = 200       # so oft wird zum Hub gecheckpointet (Disconnect-Schutz)
SEED           = 42

# Gate-freigegebener Datenmix (Gewichte ~1.0). Revisionen gepinnt wo aus dem Contamination-Gate bekannt.
SOURCES = [
    ("nvidia/OpenCodeInstruct", "train",             "train",   0.20, "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"),
    ("nvidia/OpenCodeReasoning", "split_0",           "split_0", 0.18, "20a1ca19c0d050fe9057fc08339d6b370ec1c67a"),
    ("open-r1/OpenR1-Math-220k", "default",           "train",   0.18, "e4e141ec9dea9f8326f4d347be56105859b2bd68"),
    ("open-thoughts/AgentTrove", "default",           "train",   0.24, None),
    ("NousResearch/hermes-function-calling-v1", "func_calling",     "train", 0.12, None),
    ("NousResearch/hermes-function-calling-v1", "json_mode_agentic","train", 0.08, None),
]'''

INSTALL = r'''import os, sys, subprocess
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
pip("-U", "transformers>=4.56", "peft>=0.13", "accelerate>=1.0",  # >=4.56: enthaelt die Apertus-Architektur
    "bitsandbytes>=0.44", "datasets>=3.0", "huggingface_hub>=0.26", "safetensors")
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpus", torch.cuda.device_count())'''

LOGIN = r'''from huggingface_hub import login
tok = os.environ.get("HF_TOKEN")
try:
    from kaggle_secrets import UserSecretsClient
    tok = tok or UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
assert tok, "HF_TOKEN fehlt: Add-ons -> Secrets -> HF_TOKEN (Write-Rechte) anlegen."
login(token=tok); os.environ["HF_TOKEN"] = tok
print("HuggingFace login ok")'''

DATA = r'''import random
from datasets import load_dataset, Dataset
random.seed(SEED)

def to_messages(ds, row):
    if ds == "nvidia/OpenCodeInstruct":
        return [{"role":"user","content":row.get("input","")}, {"role":"assistant","content":row.get("output","")}]
    if ds == "nvidia/OpenCodeReasoning":
        return [{"role":"user","content":row.get("input","")}, {"role":"assistant","content":row.get("output","")}]
    if ds == "open-r1/OpenR1-Math-220k":
        return [{"role":"user","content":row.get("problem","")}, {"role":"assistant","content":row.get("solution","")}]
    conv = row.get("conversations") or row.get("messages") or []
    role_map = {"human":"user","gpt":"assistant","system":"system","tool":"tool","user":"user","assistant":"assistant"}
    out = []
    for t in conv:
        role = t.get("role") or role_map.get(t.get("from"), "user")
        out.append({"role": role if role in ("user","assistant","system","tool") else "user",
                    "content": t.get("content") or t.get("value") or ""})
    return out

def passes(ds, row):
    if ds == "nvidia/OpenCodeInstruct":
        try: return float(row.get("average_test_score") or 0) >= 0.9 and bool(row.get("input")) and bool(row.get("output"))
        except (TypeError, ValueError): return False
    if ds == "nvidia/OpenCodeReasoning":
        return bool(row.get("input")) and bool(row.get("output"))
    if ds == "open-r1/OpenR1-Math-220k":
        try:
            ic = row.get("is_reasoning_complete")
            ok = True if ic is None else (True in list(ic) if isinstance(ic,(list,tuple)) else bool(ic))
            return int(row.get("correctness_count") or 0) >= 1 and ok
        except (TypeError, ValueError): return False
    return bool(row.get("conversations") or row.get("messages"))

examples = []
for ds, cfg, split, w, rev in SOURCES:
    cap = max(1, int(TOTAL_EXAMPLES * w))
    kw = {"streaming": True}
    if rev: kw["revision"] = rev
    n = 0
    for row in load_dataset(ds, cfg, split=split, **kw):
        if not passes(ds, row):
            continue
        m = to_messages(ds, row)
        if len(m) < 2 or not any(x["role"] == "assistant" and x["content"] for x in m):
            continue
        examples.append({"messages": m}); n += 1
        if n >= cap: break
    print(f"{ds}[{cfg}]: {n}/{cap}")
random.shuffle(examples)
print("examples total:", len(examples))'''

MODEL = r'''import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# Apertus-Basis hat kein Chat-Template -> ein sauberes ChatML-Format setzen, das der Adapter lernt
if getattr(tok, "chat_template", None) is None:
    tok.chat_template = (
        "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
    )

def render(ex):
    try:
        text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    except Exception:
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in ex["messages"]) + (tok.eos_token or "")
    return {"text": text}

from datasets import Dataset
ds_train = Dataset.from_list(examples).map(render, remove_columns=["messages"])
ds_train = ds_train.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_SEQ_LEN),
                        batched=True, remove_columns=["text"])
print("tokenisiert:", ds_train)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                             device_map="auto", trust_remote_code=True)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# "all-linear" ist robust gegen unbekannte Architektur-Namen (z.B. Apertus)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, lora)
model.print_trainable_parameters()'''

TRAIN = r'''import glob
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# Disconnect-Schutz: falls im Hub-Repo schon ein Checkpoint liegt -> herunterladen und fortsetzen
resume = False
try:
    from huggingface_hub import snapshot_download
    snapshot_download(HUB_ADAPTER, local_dir="out", token=os.environ["HF_TOKEN"],
                      allow_patterns=["checkpoint-*/**"])
    if glob.glob("out/checkpoint-*"):
        resume = True; print("Resume vom Hub-Checkpoint:", sorted(glob.glob("out/checkpoint-*"))[-1])
except Exception as e:
    print("kein Hub-Checkpoint (frischer Start):", type(e).__name__)

args = TrainingArguments(
    output_dir="out", num_train_epochs=EPOCHS,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
    logging_steps=10, save_steps=SAVE_STEPS, save_total_limit=2,
    fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit", report_to="none", seed=SEED,
    push_to_hub=True, hub_model_id=HUB_ADAPTER, hub_strategy="checkpoint", hub_private_repo=True,
)
trainer = Trainer(model=model, args=args, train_dataset=ds_train, data_collator=collator)
trainer.train(resume_from_checkpoint=resume)'''

FINISH = r'''# Finalen Adapter + Tokenizer zum Hub pushen
trainer.save_model("out/final")
model.push_to_hub(HUB_ADAPTER, private=True)
tok.push_to_hub(HUB_ADAPTER, private=True)
print("FERTIG ✅  Adapter liegt auf HuggingFace:", HUB_ADAPTER)
print("Sag Claude Bescheid — er evaluiert Adapter vs. Baseline auf HydraCH vom VPS aus.")'''

cells = [
    ("markdown", MD_INTRO),
    ("markdown", "## 1 · Konfiguration"), ("code", CFG),
    ("markdown", "## 2 · Abhängigkeiten"), ("code", INSTALL),
    ("markdown", "## 3 · HuggingFace-Login"), ("code", LOGIN),
    ("markdown", "## 4 · Daten (gate-freigegeben, gestreamt)"), ("code", DATA),
    ("markdown", "## 5 · Apertus-8B in 4-bit + LoRA"), ("code", MODEL),
    ("markdown", "## 6 · Training (checkpointet zum Hub)"), ("code", TRAIN),
    ("markdown", "## 7 · Fertigstellen"), ("code", FINISH),
]

def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base

nb = {
    "cells": [cell(k, s) for k, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = pathlib.Path(__file__).with_name("kaggle_qlora_apertus8b.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("geschrieben:", out, "|", len(nb["cells"]), "Zellen")
