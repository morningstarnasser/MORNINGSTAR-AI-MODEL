#!/usr/bin/env python3
"""Builds the Kaggle notebook that evaluates candidate backends on HydraCH dev.

The notebook only RUNS the models and pushes their raw responses to a private HF
dataset; the real HydraCH grader (morningstar_hydra.hydrach_bench.grade_response)
scores them on the VPS afterwards, so the numbers are exactly comparable.
"""
import json, pathlib

MD_INTRO = r"""# ★ HydraCH — Kandidaten-Backends evaluieren (Kaggle)

Vergleicht selbst-hostbare, Apache-2.0-Modelle auf den 50 HydraCH-Dev-Fällen, um zu
entscheiden, ob eines gut genug ist, um die DeepSeek-API abzulösen.

**Setup (einmalig):**
1. **Settings → Accelerator → `GPU T4 x2`** · **Internet → `On`**.
2. **Add-ons → Secrets → `HF_TOKEN`** (Account kurdman991, Write).
3. **`Run All`**.

Das Notebook erzeugt nur Antworten und pusht sie nach `kurdman991/hydrach-eval-outputs`.
Die **Bewertung macht Claude danach auf dem VPS** mit dem echten Grader. Laufzeit grob
1–2 h für alle 4 Modelle (4-bit)."""

CFG = r'''# ============ Kandidaten (alle Apache-2.0, selbst-hostbar) ============
MODELS = [
    "swiss-ai/Apertus-8B-Instruct-2509",            # echt Schweizerisch, 8B
    "mistralai/Ministral-3-8B-Instruct-2512",       # neuestes EU-8B
    "Qwen/Qwen2.5-14B-Instruct",                    # Fähigkeits-Referenz, 14B
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",# EU-Mehrsprach-Kraftpaket, 24B
]
DEV_DATASET   = "kurdman991/hydrach-dev-eval"       # privat, enthaelt dev.jsonl
OUT_DATASET   = "kurdman991/hydrach-eval-outputs"   # privat, hierhin kommen die Antworten
STRUCT_TOKENS = 512   # invoice_json / multilingual_json / tool_call
SHORT_TOKENS  = 256   # normalized_exact'''

INSTALL = r'''import os, sys, subprocess, gc
def pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=True)
pip("-U","transformers>=4.56","accelerate>=1.0","bitsandbytes>=0.44","huggingface_hub>=0.26","safetensors")
import torch
print("torch", torch.__version__, "| gpus", torch.cuda.device_count())'''

LOGIN = r'''from huggingface_hub import login, hf_hub_download, create_repo, HfApi
tok = os.environ.get("HF_TOKEN")
try:
    from kaggle_secrets import UserSecretsClient
    tok = tok or UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass
assert tok, "HF_TOKEN fehlt (Add-ons -> Secrets)."
login(token=tok); os.environ["HF_TOKEN"] = tok
create_repo(OUT_DATASET, repo_type="dataset", private=True, exist_ok=True, token=tok)
print("HF login ok")'''

DATA = r'''import json, re
path = hf_hub_download(DEV_DATASET, "dev.jsonl", repo_type="dataset", token=tok)
CASES = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print("dev-Faelle:", len(CASES))

DIRECTIVE = ("Antworte mit exakt dem Angefragten und nichts sonst. Bei JSON-Aufgaben nur "
             "rohes JSON (keine Markdown-Codebloecke, keine Prosa). Daten als ISO YYYY-MM-DD. "
             "Angefragte Werte exakt ohne Zusatzwoerter. Wenn Tools gegeben sind, rufe das Tool auf.")

def parse_tool_calls(text):
    raw = []
    raw += re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    m = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", text, re.S)
    if m:
        try: raw += [json.dumps(c) for c in json.loads(m.group(1))]
        except Exception: pass
    if not raw:
        raw += re.findall(r"\{[^{}]*\"name\"[^{}]*\"arguments\"[^{}]*\}", text, re.S)
    out = []
    for c in raw:
        try:
            d = json.loads(c); a = d.get("arguments", d.get("parameters", {}))
            out.append({"id": f"call_{len(out)}", "type": "function",
                        "function": {"name": d.get("name"),
                                     "arguments": a if isinstance(a, str) else json.dumps(a)}})
        except Exception:
            pass
    return out'''

RUN = r'''import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

def run_model(model_id):
    print(f"\n=== {model_id} ===", flush=True)
    tk = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tk.pad_token is None: tk.pad_token = tk.eos_token
    if getattr(tk, "chat_template", None) is None:
        tk.chat_template = ("{% for m in messages %}{{'<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n'}}"
                            "{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
    mdl = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                               device_map="auto", trust_remote_code=True)
    mdl.eval()
    outputs = []
    t0 = time.time()
    for i, case in enumerate(CASES):
        msgs = [{"role": "system", "content": DIRECTIVE}] + case["messages"]
        tools = case.get("tools")
        try:
            prompt = tk.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tk(prompt, return_tensors="pt").to(mdl.device)
        mx = STRUCT_TOKENS if case["grader"]["type"] in ("json_exact", "tool_call") else SHORT_TOKENS
        with torch.no_grad():
            gen = mdl.generate(**ids, max_new_tokens=mx, do_sample=False, temperature=None,
                               top_p=None, pad_token_id=tk.pad_token_id)
        text = tk.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        tcs = parse_tool_calls(text) if case["grader"]["type"] == "tool_call" else []
        outputs.append({"id": case["id"], "category": case["category"],
                        "content": text, "tool_calls": tcs})
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(CASES)}", flush=True)
    print(f"  fertig in {time.time()-t0:.0f}s")
    del mdl; gc.collect(); torch.cuda.empty_cache()
    return outputs

api = HfApi()
for model_id in MODELS:
    try:
        outs = run_model(model_id)
        slug = model_id.replace("/", "__")
        fname = f"{slug}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"model": model_id, "outputs": outs}, f, ensure_ascii=False)
        api.upload_file(path_or_fileobj=fname, path_in_repo=fname,
                        repo_id=OUT_DATASET, repo_type="dataset", token=tok)
        print(f"  -> gepusht: {OUT_DATASET}/{fname}")
    except Exception as e:
        print(f"  FEHLER bei {model_id}: {type(e).__name__}: {str(e)[:200]}")'''

FINISH = r'''print("FERTIG ✅  Alle Antworten liegen auf HuggingFace:", OUT_DATASET)
print("Sag Claude Bescheid — er bewertet sie mit dem echten HydraCH-Grader auf dem VPS")
print("und liefert die Vergleichstabelle (Modell x Kategorie).")'''

cells = [
    ("markdown", MD_INTRO),
    ("markdown", "## 1 · Kandidaten & Konfig"), ("code", CFG),
    ("markdown", "## 2 · Abhängigkeiten"), ("code", INSTALL),
    ("markdown", "## 3 · HuggingFace-Login"), ("code", LOGIN),
    ("markdown", "## 4 · Dev-Fälle + Helfer"), ("code", DATA),
    ("markdown", "## 5 · Modelle laufen lassen & Antworten pushen"), ("code", RUN),
    ("markdown", "## 6 · Fertig"), ("code", FINISH),
]

def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base["execution_count"] = None; base["outputs"] = []
    return base

nb = {"cells": [cell(k, s) for k, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).with_name("kaggle_hydrach_eval.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("geschrieben:", out, "|", len(nb["cells"]), "Zellen")
