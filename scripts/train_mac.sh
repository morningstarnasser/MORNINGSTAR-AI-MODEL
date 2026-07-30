#!/usr/bin/env bash
# Morningstar Hydra — Apertus-8B QLoRA auf Apple Silicon starten.
#
# Ersetzt den CUDA-Pfad aus cloud/kaggle_qlora_apertus8b.ipynb: bitsandbytes gibt
# es auf dem Mac nicht, MLX quantisiert selbst. Voraussetzungen:
#   1. venv:    uv venv ~/hydra-mlx-venv --python 3.12 && uv pip install mlx-lm datasets
#   2. Modell:  python -m mlx_lm convert --hf-path swiss-ai/Apertus-8B-2509 \
#                 --mlx-path $HYDRA_MODELS_DIR/apertus-8b-4bit -q --q-bits 4 --q-group-size 64
#   3. Daten:   python scripts/build_mlx_dataset.py --total-examples 24000
#
# Aufruf:  scripts/train_mac.sh [--iters N] [--batch-size N] [weitere mlx_lm-Flags]
set -euo pipefail

VENV="${HYDRA_VENV:-$HOME/hydra-mlx-venv}"
MODEL="${HYDRA_MODEL:-$HOME/hydra-models/apertus-8b-4bit}"
DATA="${HYDRA_DATA:-$HOME/hydra-train/data}"
ADAPTERS="${HYDRA_ADAPTERS:-$HOME/hydra-train/adapters}"
CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/mlx-lora-apertus8b.yaml"

for path in "$VENV/bin/python" "$MODEL" "$DATA/train.jsonl" "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "FEHLT: $path" >&2
    exit 1
  fi
done

mkdir -p "$ADAPTERS"
echo "Modell:   $MODEL"
echo "Daten:    $DATA ($(wc -l < "$DATA/train.jsonl") train / $(wc -l < "$DATA/valid.jsonl") valid)"
echo "Adapter:  $ADAPTERS"
echo

# Pfade bewusst per CLI: MLX expandiert Tilde in der YAML nicht zuverlaessig.
# -u, damit der Fortschritt auch in einer Logdatei sofort sichtbar ist (sonst
# puffert Python blockweise und ein laufendes Training sieht eingefroren aus).
exec "$VENV/bin/python" -u -m mlx_lm lora \
  --config "$CONFIG" \
  --model "$MODEL" \
  --data "$DATA" \
  --adapter-path "$ADAPTERS" \
  "$@"
