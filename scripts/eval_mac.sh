#!/usr/bin/env bash
# Faehrt HydraCH-Bench gegen das lokale MLX-Modell — mit und/oder ohne LoRA-Adapter.
#
# Das Gate in config/hf-expert-mixture.json erlaubt eine Promotion nur, wenn der
# Adapter die unveraenderte Basis schlaegt. Genau diesen Vergleich stellt das
# Skript her: derselbe Runner, dasselbe Chat-Template, nur der Adapter wechselt.
#
# Aufruf:  scripts/eval_mac.sh {base|adapter|both}
set -euo pipefail

MODE="${1:-both}"
VENV="${HYDRA_VENV:-$HOME/hydra-mlx-venv}"
MODEL="${HYDRA_MODEL:-$HOME/hydra-models/apertus-8b-4bit}"
ADAPTERS="${HYDRA_ADAPTERS:-$HOME/hydra-train/adapters}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_FILE="$REPO/config/apertus_chat_template.jinja"
RESULTS="${HYDRA_RESULTS:-$HOME/hydra-train/results}"
PORT="${HYDRA_PORT:-8099}"

mkdir -p "$RESULTS"

# Sicherheitsnetz: kein Server soll das Skript ueberleben, auch nicht bei Abbruch.
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

has_own_template() {
  # Instruct-Modelle bringen ihr Template mit — das darf nicht ueberschrieben werden,
  # sonst evaluiert man ein anderes Format als das trainierte.
  [[ -f "$MODEL/chat_template.jinja" ]] && return 0
  grep -q '"chat_template"' "$MODEL/tokenizer_config.json" 2>/dev/null && return 0
  return 1
}

run_one() {
  local label="$1"; shift
  local logfile="$RESULTS/server-$label.log"
  echo "=== $label ==="

  local template_args=()
  if has_own_template; then
    echo "    (nutzt das Chat-Template des Modells)"
  else
    echo "    (Modell hat kein Template — nutzt $TEMPLATE_FILE)"
    template_args=(--chat-template "$(cat "$TEMPLATE_FILE")")
  fi

  # Array-Expansion so geschrieben, dass sie unter bash 3.2 (macOS) mit set -u
  # auch leer funktioniert.
  "$VENV/bin/python" -m mlx_lm server \
    --model "$MODEL" --port "$PORT" \
    ${template_args[@]+"${template_args[@]}"} \
    "$@" > "$logfile" 2>&1 &
  local server_pid=$!
  SERVER_PID="$server_pid"

  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server gestorben — siehe $logfile" >&2
      tail -20 "$logfile" >&2
      return 1
    fi
    sleep 2
  done

  # Der Modellname im Request MUSS der geladene Pfad sein: mlx_lm.server nimmt das
  # Feld wörtlich und wuerde einen abweichenden Namen von HuggingFace nachladen (404).
  OPENAI_API_KEY=local "$VENV/bin/python" "$REPO/scripts/evaluate_hydrach_bench.py" \
    --cases "$REPO/benchmarks/dev.jsonl" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$MODEL" \
    --api-key-env OPENAI_API_KEY \
    --pace-seconds 0 \
    --keep-output \
    --output "$RESULTS/hydrach-$label.json"

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

case "$MODE" in
  base)    run_one base ;;
  adapter) run_one adapter --adapter-path "$ADAPTERS" ;;
  both)    run_one base; run_one adapter --adapter-path "$ADAPTERS" ;;
  *) echo "Aufruf: $0 {base|adapter|both}" >&2; exit 1 ;;
esac

echo
echo "Ergebnisse in $RESULTS"
