# Lokales GPU Training - RTX 3090 24GB

**Du hast perfekte Hardware! RTX 3090 ist eine der besten Consumer GPUs für ML.**

---

## ✅ Was du hast:

- **GPU:** RTX 3090
- **VRAM:** 24 GB (AUSGEZEICHNET!)
- **CUDA:** 581.63 (neueste Version)

**Vergleich:**
| GPU | VRAM | Training Zeit (50k samples) |
|-----|------|----------------------------|
| RTX 3090 (DU) | 24 GB | ⚡ **18-24h** |
| RTX 4090 | 24 GB | 12-16h |
| RTX 3080 | 10 GB | 30-40h (schwierig) |
| A100 (Cloud) | 80 GB | 8-12h |

**Deine RTX 3090 ist PERFEKT für dieses Training!**

---

## 🚀 SETUP (nur einmal, ~30 Min)

### Schritt 1: Python 3.11 installieren

**WICHTIG:** Python 3.14.3 funktioniert nicht! Du brauchst 3.11.

1. Download: https://www.python.org/downloads/release/python-3119/
2. "Windows installer (64-bit)" wählen
3. Installieren - **WICHTIG:** ☑️ "Add Python 3.11 to PATH"
4. Verify in CMD:
   ```cmd
   python3.11 --version
   ```

### Schritt 2: Setup Script ausführen

```cmd
cd D:\math-training
LOCAL_GPU_TRAINING.bat
```

**Das macht automatisch:**
- ✓ Python 3.11 venv erstellen
- ✓ PyTorch mit CUDA 12.1
- ✓ Alle ML Libraries (transformers, peft, trl, etc.)
- ✓ CUDA Test
- ✓ Dataset prep (oder minimal test dataset)

**Wartezeit:** 20-30 Minuten (Downloads)

---

## 🎯 TRAINING STARTEN

### Option A: Automatisch (empfohlen)

```cmd
cd D:\math-training
START_LOCAL_TRAINING.bat
```

**Fertig!** Training läuft jetzt.

### Option B: Manuell mit Custom Settings

```cmd
cd D:\math-training
venv-ml\Scripts\activate
python train.py ^
    --dataset_path data/train.jsonl ^
    --output_dir output/morningstar-math ^
    --epochs 3 ^
    --batch_size 2 ^
    --gradient_accumulation 8
```

---

## ⏱️ ZEITPLAN

### Mit vollem Dataset (~50k samples):
- **Epoch 1:** ~6-8 Stunden
- **Epoch 2:** ~6-8 Stunden
- **Epoch 3:** ~6-8 Stunden
- **TOTAL:** 18-24 Stunden

### Mit Test-Dataset (100 samples):
- **TOTAL:** 10-15 Minuten (zum Testen)

**Empfehlung:** Starte abends, läuft über Nacht + nächsten Tag!

---

## 📊 TRAINING MONITOR

### GPU Monitor (neues CMD Fenster):

```cmd
nvidia-smi -l 1
```

Zeigt:
- GPU Auslastung (sollte ~95-100% sein)
- VRAM Nutzung (wird ~20-22 GB sein)
- Temperatur (sollte <85°C sein)

### Training Logs:

```cmd
type output\morningstar-math-local\*.log
```

---

## ⚙️ OPTIMALE EINSTELLUNGEN FÜR RTX 3090 24GB

```python
--batch_size 2              # Nutzt ~22 GB VRAM
--gradient_accumulation 8   # Effective batch: 16
--max_seq_length 2048       # Balance zwischen Speed und Quality
--save_steps 500            # Checkpoint alle 500 steps
```

**Wenn CUDA OOM Error:**
```python
--batch_size 1              # Reduziere auf 1
--gradient_accumulation 16  # Kompensiere mit mehr accumulation
```

---

## 💡 TIPPS FÜR ÜBER-NACHT TRAINING

### 1. Power Settings
```
Windows Settings → Power & Sleep → Never (während Training)
```

### 2. Nvidia Settings
- Nvidia Control Panel → Manage 3D Settings
- Power Management: "Prefer Maximum Performance"

### 3. Temperature Check
```cmd
REM Before training:
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

REM During training (check every hour):
nvidia-smi -l 3600
```

**Safe Temperature:** <85°C
**Optimal:** 70-80°C

### 4. Prevent Sleep/Hibernate
```cmd
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

---

## 📥 NACH TRAINING

### 1. Export zu GGUF (~30 Min)

```cmd
cd D:\math-training
venv-ml\Scripts\activate
python merge_and_export.py ^
    --adapter_path output\morningstar-math-local\final ^
    --export_gguf ^
    --create_ollama_model
```

### 2. Ollama Import

```cmd
ollama create morningstar-math -f Modelfile.morningstar
ollama run morningstar-math
```

### 3. Evaluation

```cmd
cd eval
python evaluate_math.py --model morningstar-math --all-levels --verbose
```

**Erwartete Performance:**
- Level 1-2: 95%+
- Level 6-7: 50-60%
- Overall: 75-80%

---

## 🆘 TROUBLESHOOTING

### Problem: "CUDA out of memory"

**Lösung:**
```cmd
python train.py --batch_size 1 --gradient_accumulation 16
```

### Problem: Training zu langsam

**Check:**
```cmd
nvidia-smi
```

- GPU Usage sollte ~95-100% sein
- Wenn niedrig: Dataloading Bottleneck
- Lösung: Reduziere dataloader_num_workers

### Problem: PC friert ein

**Ursachen:**
- Andere Programme nutzen GPU → Schließe Browser/Games
- Temperatur zu hoch → Verbessere Kühlung
- RAM voll → Schließe andere Programme

### Problem: Dataset fehlt

**Lösung 1 - Python 3.11:**
```cmd
python3.11 -m venv venv311
venv311\Scripts\activate
pip install datasets tqdm
python prepare_math_dataset.py
```

**Lösung 2 - Cloud Download:**
Download vorbereitetes Dataset von HuggingFace (wenn verfügbar)

---

## 📊 KOSTEN VERGLEICH

| Methode | Hardware | Zeit | Kosten | Stromverbrauch |
|---------|----------|------|--------|----------------|
| **Lokal (RTX 3090)** | Deine GPU | 18-24h | $0 | ~$5 (Strom) |
| Cloud (A100) | Gemietet | 8-12h | ~$17 | $0 |

**Dein Vorteil:**
- ✅ Keine Cloud-Kosten
- ✅ Volle Kontrolle
- ✅ Unbegrenzte Experimente
- ✅ Privacy (Daten bleiben lokal)

**Nachteil:**
- ⏱️ 2x langsamer als A100
- 💡 Stromkosten (~$5)
- 🔥 GPU nicht nutzbar während Training

---

## ✅ CHECKLISTE

### Vor Training:
- [ ] Python 3.11 installiert
- [ ] `LOCAL_GPU_TRAINING.bat` ausgeführt
- [ ] CUDA Test erfolgreich
- [ ] Dataset vorhanden (`data/train.jsonl`)
- [ ] Power Settings: Never Sleep
- [ ] Andere GPU-Programme geschlossen

### Training starten:
- [ ] `START_LOCAL_TRAINING.bat` ausgeführt
- [ ] GPU Monitor läuft (nvidia-smi)
- [ ] Laptop/PC kann über Nacht laufen
- [ ] Backup wichtiger Daten (sicherheitshalber)

### Nach Training:
- [ ] Training erfolgreich abgeschlossen
- [ ] Model in `output/morningstar-math-local/final`
- [ ] GGUF Export durchgeführt
- [ ] Ollama Model erstellt
- [ ] Evaluation durchgeführt

---

## 🎯 QUICK START (für Ungeduldige)

```cmd
REM 1. Setup (einmalig, 30 Min)
cd D:\math-training
LOCAL_GPU_TRAINING.bat

REM 2. Training starten (18-24h)
START_LOCAL_TRAINING.bat

REM 3. Minimiere Fenster, geh schlafen

REM 4. Nächster Tag: Export (30 Min)
python merge_and_export.py --adapter_path output\morningstar-math-local\final --export_gguf

REM 5. Ollama Import (1 Min)
ollama create morningstar-math -f Modelfile.morningstar

REM 6. Testen!
ollama run morningstar-math
```

---

**RTX 3090 ist PERFEKT für dieses Projekt! Du sparst $17 Cloud-Kosten und hast volle Kontrolle.**

**START:** `LOCAL_GPU_TRAINING.bat` 🚀
