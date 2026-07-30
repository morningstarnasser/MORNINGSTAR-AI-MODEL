# MORNINGSTAR Math-Training — Quick Start Guide

**Autor:** Morningstar Contributors
**Projekt:** https://github.com/your-org/MORNINGSTAR-AI-MODEL
**HuggingFace:** https://huggingface.co/kurdman991

---

## 🚀 Schnellstart (3 Schritte)

### 1. Dataset vorbereiten
```bash
pip install datasets tqdm numpy
python prepare_math_dataset.py
```
Lädt ~50k Math-Probleme (GSM8K, MATH, Orca-Math, MathInstruct) nach `data/`.

### 2. Lokales Training (braucht GPU 16GB+)
```bash
python train.py --dataset_path ./data/train.jsonl --epochs 3
```

ODER Cloud Training (empfohlen für schnelleres Training):
```bash
cd cloud/
bash setup-cloud-training.sh
python train_math.py --dataset-dir ../data
```

### 3. Nach Training: Export & Ollama
```bash
python cloud/export_gguf.py --model-dir ./output/math-qlora/merged-model
ollama create morningstar-math -f cloud/Modelfile.math
```

---

## 📊 Evaluation

### Basis-Evaluation (7 Difficulty Levels)
```bash
cd eval/
python evaluate_math.py --model morningstar-math --verbose
```

### Nur harte Probleme (AIME + Olympiade)
```bash
python evaluate_math.py --model morningstar-math --levels 6 7 --verbose
```

### Multi-Model Vergleich
```bash
python compare_models.py --models morningstar-math qwen2.5:14b deepseek-r1:14b --all-levels
```

---

## 🧪 Test mit Beispiel-Problemen

```bash
ollama run morningstar-math
```

Dann teste:
```
>>> What is the sum of all positive divisors of 120?

>>> Simplify: (2^10 * 3^7) / (2^6 * 3^4)

>>> A circle has radius 5. What is its area? Give exact answer.
```

---

## 🔧 Advanced: TTC (Time-To-Compute) für schwere Probleme

```bash
cd inference/

# Best-of-N mit Majority Voting
python smart_math.py --problem "Find the remainder when 2^100 is divided by 13" --n 5 --verify

# FastAPI Server starten
python math_server.py

# TTC Benchmark
python benchmark_ttc.py --quick
```

---

## 📁 Projekt-Struktur

```
math-training/
├── prepare_math_dataset.py    # Dataset prep
├── train.py                    # Lokales QLoRA Training
├── Modelfile.morningstar       # Ollama Modelfile (Advanced Reasoning)
├── eval/
│   ├── evaluate_math.py        # 63 Probleme, 7 Levels
│   └── compare_models.py       # Multi-model benchmark
├── cloud/
│   ├── train_math.py           # GPU Cloud Training (Unsloth)
│   ├── export_gguf.py          # GGUF Export
│   ├── setup-cloud-training.sh         # RunPod Setup
│   └── Modelfile.math          # Post-training Modelfile
├── inference/
│   ├── smart_math.py           # Best-of-N + Voting
│   ├── math_server.py          # HTTP API
│   └── benchmark_ttc.py        # TTC Benchmark
└── data/                       # Dataset (wird erstellt)
```

---

## 🎯 Erwartete Performance

**Vor Fine-Tuning (Base Qwen2.5-Coder):**
- Level 1-2 (Basic): ~90%
- Level 3-4 (Intermediate): ~70-80%
- Level 5 (Competition): ~50-60%
- Level 6-7 (AIME/Olympiade): ~20-30%

**Nach Fine-Tuning auf 50k Math-Problemen:**
- Level 1-2: **95%+**
- Level 3-4: **85-90%**
- Level 5: **70-80%**
- Level 6-7: **50-60%** (mit TTC: **60-70%**)

---

## ⚙️ Training Config

### Lokales Training (train.py)
- **QLoRA:** 4-bit NF4, r=64, alpha=128
- **Batch Size:** 4 (effective: 16)
- **Learning Rate:** 2e-4 (cosine)
- **Epochs:** 3
- **GPU:** 16GB+ VRAM (RTX 4090, A6000, etc.)

### Cloud Training (cloud/train_math.py)
- **Framework:** Unsloth (2-3x schneller)
- **Same QLoRA config**
- **GPU:** A100 80GB oder A6000 48GB (empfohlen)
- **Training Zeit:** ~6-12h für 3 Epochen

---

## 🔗 Links

- **GitHub:** https://github.com/your-org/MORNINGSTAR-AI-MODEL
- **HuggingFace Models:**
  - https://huggingface.co/kurdman991/morningstar-14b
  - https://huggingface.co/kurdman991/morningstar-32b
  - https://huggingface.co/kurdman991/morningstar-vision

---

## 🐛 Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'datasets'`
```bash
pip install datasets tqdm numpy
```

**Problem:** `CUDA out of memory`
```bash
# Reduziere batch_size
python train.py --batch_size 2 --gradient_accumulation 8
```

**Problem:** Ollama findet Modell nicht
```bash
ollama list  # Checke verfügbare Modelle
ollama create morningstar -f Modelfile.morningstar
```

**Problem:** Evaluation zu langsam
```bash
# Nur einzelne Levels testen
python eval/evaluate_math.py --levels 1 2 --verbose
```

---

## 📝 Tipps

1. **Start small:** Teste erst mit Level 1-2 Evaluation
2. **Use cloud for training:** Lokales Training ist langsam ohne gute GPU
3. **TTC for hard problems:** Level 6-7 profitieren von Best-of-N
4. **Monitor training:** Nutze `--wandb` für Weights & Biases Logging
5. **Save checkpoints:** Training speichert alle 500 steps

---

**Viel Erfolg! 🚀**

Bei Fragen: https://github.com/your-org/MORNINGSTAR-AI-MODEL/issues
