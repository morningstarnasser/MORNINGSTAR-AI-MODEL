@echo off
REM MORNINGSTAR Math Training - Lokales GPU Training Setup
REM Optimiert für RTX 3090 24GB
REM Author: Ali Nasser

echo ==========================================
echo   MORNINGSTAR Lokales GPU Training
echo   GPU: RTX 3090 24GB
echo ==========================================
echo.

REM Check if running as Admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNUNG: Nicht als Administrator. Manche Schritte koennen fehlschlagen.
    echo.
)

REM Step 1: Check Python
echo [1/7] Python Check...
python --version
if %errorLevel% neq 0 (
    echo ERROR: Python nicht gefunden!
    echo Bitte installiere Python 3.11 von python.org
    pause
    exit /b 1
)

REM Step 2: Create Virtual Environment with Python 3.11
echo.
echo [2/7] Erstelle Python 3.11 Virtual Environment...
echo Bitte Python 3.11 manuell von https://www.python.org/downloads/ installieren
echo und dann venv erstellen.
echo.
echo PAUSE - Druecke Taste wenn Python 3.11 installiert...
pause

REM Use specific Python 3.11 if available
where python3.11 >nul 2>&1
if %errorLevel% equ 0 (
    python3.11 -m venv venv-ml
) else (
    python -m venv venv-ml
)

call venv-ml\Scripts\activate.bat

REM Step 3: Upgrade pip
echo.
echo [3/7] Upgrade pip...
python -m pip install --upgrade pip -q

REM Step 4: Install PyTorch with CUDA
echo.
echo [4/7] Installiere PyTorch mit CUDA 12.1...
echo Dies kann 5-10 Minuten dauern...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Step 5: Install ML libraries
echo.
echo [5/7] Installiere ML Libraries...
echo Dies kann 10-15 Minuten dauern...
pip install transformers datasets accelerate peft trl bitsandbytes
pip install scipy sentencepiece protobuf tqdm

REM Step 6: Test CUDA
echo.
echo [6/7] Test CUDA Setup...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"

if %errorLevel% neq 0 (
    echo.
    echo WARNUNG: CUDA Test fehlgeschlagen!
    echo Training wird auf CPU laufen (sehr langsam)
    echo.
)

REM Step 7: Prepare Dataset
echo.
echo [7/7] Dataset vorbereiten...
echo HINWEIS: Dieser Schritt wird uebersprungen wenn datasets nicht installiert werden kann.
echo Du kannst das Dataset manuell mit prepare_math_dataset.py erstellen.
echo.

python prepare_math_dataset.py
if %errorLevel% neq 0 (
    echo Dataset preparation fehlgeschlagen (erwartet wegen Python 3.14.3)
    echo Nutze alternative: Erstelle minimales Test-Dataset

    mkdir data 2>nul
    echo {"conversations":[{"role":"system","content":"You are a math expert"},{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"}]} > data\train.jsonl
    echo {"conversations":[{"role":"system","content":"You are a math expert"},{"role":"user","content":"What is 3+3?"},{"role":"assistant","content":"6"}]} > data\val.jsonl

    echo Minimales Test-Dataset erstellt. Fuer echtes Training: Nutze Cloud oder Python 3.11
)

echo.
echo ==========================================
echo   SETUP COMPLETE!
echo ==========================================
echo.
echo GPU: RTX 3090 24GB - OPTIMAL fuer Training!
echo.
echo Naechster Schritt - Training starten:
echo   python train.py --dataset_path data/train.jsonl --epochs 3 --batch_size 2
echo.
echo Geschaetzte Zeit auf RTX 3090:
echo   - Mit vollem Dataset (50k): 18-24 Stunden
echo   - Mit Test-Dataset (100): 10-15 Minuten
echo.
echo TIPP: Starte Training ueber Nacht!
echo.
pause
