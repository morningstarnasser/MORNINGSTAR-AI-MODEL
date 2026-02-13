@echo off
REM MORNINGSTAR - Starte Lokales Training
REM RTX 3090 optimiert

echo ==========================================
echo   MORNINGSTAR Training START
echo   GPU: RTX 3090 24GB
echo ==========================================
echo.

REM Activate venv
call venv-ml\Scripts\activate.bat

REM Check if data exists
if not exist "data\train.jsonl" (
    echo ERROR: data\train.jsonl nicht gefunden!
    echo Bitte zuerst LOCAL_GPU_TRAINING.bat ausfuehren!
    pause
    exit /b 1
)

REM Training Config for RTX 3090
echo Training Konfiguration:
echo   - GPU: RTX 3090 24GB
echo   - Batch Size: 2 (optimal fuer 24GB)
echo   - Gradient Accumulation: 8 (effective batch: 16)
echo   - Epochs: 3
echo   - Learning Rate: 2e-4
echo   - Max Seq Length: 2048
echo.

REM Show GPU before training
echo GPU Status:
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo.

REM Start training with optimized settings for RTX 3090
echo Training startet in 5 Sekunden...
echo Du kannst dieses Fenster minimieren (nicht schliessen!)
echo.
timeout /t 5

python train.py ^
    --dataset_path data/train.jsonl ^
    --eval_dataset_path data/val.jsonl ^
    --output_dir output/morningstar-math-local ^
    --epochs 3 ^
    --batch_size 2 ^
    --gradient_accumulation 8 ^
    --learning_rate 2e-4 ^
    --max_seq_length 2048 ^
    --save_steps 500 ^
    --logging_steps 10

echo.
echo ==========================================
echo   TRAINING COMPLETE!
echo ==========================================
echo.

if exist "output\morningstar-math-local\final" (
    echo Training erfolgreich abgeschlossen!
    echo Model gespeichert in: output\morningstar-math-local\final
    echo.
    echo Naechste Schritte:
    echo 1. Export zu GGUF: python merge_and_export.py --adapter_path output\morningstar-math-local\final
    echo 2. Ollama Import: ollama create morningstar-math -f Modelfile.morningstar
    echo 3. Evaluation: cd eval ^&^& python evaluate_math.py --model morningstar-math
) else (
    echo WARNUNG: Training moeglicherweise fehlgeschlagen!
    echo Check logs in: output\morningstar-math-local\
)

echo.
pause
