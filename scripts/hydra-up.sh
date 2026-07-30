#!/usr/bin/env bash
# Faehrt den lokalen Hydra-Stack hoch — auf Mac, PC und Raspberry Pi.
#
#   Ollama        11434  haelt das allgemeine Modell
#   Bruecke        8120  OpenAI-Format, Denkmodus steuerbar, Streaming
#   Router         8100  /v1/chat/completions fuer Clients, eigene Identitaet
#
# Das Modell waehlt sich nach verfuegbarem Arbeitsspeicher, laesst sich aber
# jederzeit ueberschreiben. Der Specialist (Apertus-4B-CH auf 8080) ist optional:
# laeuft er nicht, startet der Router trotzdem und spricht ihn erst bei
# ausdruecklicher Modellwahl an.
#
#   ./scripts/hydra-up.sh                        # Modell nach RAM, Denken aus
#   HYDRA_THINK=on ./scripts/hydra-up.sh         # Denken an
#   HYDRA_GENERAL_MODEL=qwen3:14b ./scripts/hydra-up.sh
#   HYDRA_BIND=0.0.0.0 ./scripts/hydra-up.sh     # im LAN erreichbar (siehe Warnung)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HYDRA_PYTHON:-python3}"
DENKEN="${HYDRA_THINK:-off}"
BRUECKE_PORT="${HYDRA_BRIDGE_PORT:-8120}"
ROUTER_PORT="${HYDRA_ROUTER_PORT:-8100}"
SPECIALIST_PORT="${HYDRA_SPECIALIST_PORT:-8080}"
BIND="${HYDRA_BIND:-127.0.0.1}"
LOGS="${HYDRA_LOG_DIR:-$HOME/.hydra-logs}"

mkdir -p "$LOGS"

# ---------------------------------------------------------------- Arbeitsspeicher
system_ram_gb() {
  case "$(uname -s)" in
    Darwin) echo $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) ;;
    Linux)  echo $(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 )) ;;
    *)      echo 0 ;;
  esac
}

# Auf einer dedizierten GPU zaehlt deren Speicher, nicht der des Systems: Ollama
# laedt das Modell dorthin. Ein PC mit 16 GB RAM und einer 24-GB-Karte traegt
# deutlich mehr, als der System-RAM vermuten laesst.
vram_gb() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | sort -rn | head -1 | awk '{printf "%d", $1/1024}' || echo 0
}

# Speicher allein entscheidet nicht. Ein Raspberry Pi 5 hat 15 GB RAM, rechnet
# aber auf vier CPU-Kernen ohne Beschleuniger — ein 14B-Modell laeuft dort zwar
# hinein, liefert aber rund ein Token pro Sekunde. Nur wo eine GPU rechnet
# (Apple Silicon ueber Metal, NVIDIA ueber CUDA) sind grosse Modelle sinnvoll.
hat_beschleuniger() {
  [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 && return 0
  return 1
}

modell_nach_speicher() {
  local gb="$1"
  if   [ "$gb" -ge 24 ]; then echo "qwen3.5:27b-int4"   # ~16 GB
  elif [ "$gb" -ge 14 ]; then echo "qwen3:14b"          # ~9 GB
  else                        echo "qwen3.5:4b"         # ~3,4 GB
  fi
}

RAM_GB="$(system_ram_gb)"
GPU_GB="$(vram_gb)"
# Der groessere der beiden Werte entscheidet; ohne GPU ist GPU_GB schlicht 0.
BUDGET_GB="$RAM_GB"
[ "${GPU_GB:-0}" -gt "$BUDGET_GB" ] && BUDGET_GB="$GPU_GB"

if hat_beschleuniger; then
  STANDARDMODELL="$(modell_nach_speicher "$BUDGET_GB")"
  BESCHLEUNIGER="ja"
else
  # Reine CPU: alles ueber 4B ist in der Praxis zu langsam, egal wie viel
  # Speicher vorhanden ist.
  STANDARDMODELL="qwen3.5:4b"
  BESCHLEUNIGER="nein"
fi
MODELL="${HYDRA_GENERAL_MODEL:-$STANDARDMODELL}"

if [ "${GPU_GB:-0}" -gt 0 ]; then
  echo "Maschine: $(uname -s) $(uname -m), ${RAM_GB} GB RAM, ${GPU_GB} GB VRAM  →  Modell: $MODELL"
else
  echo "Maschine: $(uname -s) $(uname -m), ${RAM_GB} GB RAM, GPU: $BESCHLEUNIGER  →  Modell: $MODELL"
fi
if [ "$BESCHLEUNIGER" = "nein" ]; then
  echo "  Hinweis: ohne GPU rechnet das Modell auf der CPU und antwortet langsam."
  echo "  Auf einem Pi ist der Betrieb als reiner Client gegen Mac oder PC meist sinnvoller."
fi

# ------------------------------------------------------------------ Abhaengigkeiten
FEHLEND="$("$PYTHON" - <<'PY'
import importlib.util
print(" ".join(m for m in ("httpx", "uvicorn") if importlib.util.find_spec(m) is None))
PY
)"
if [ -n "$FEHLEND" ]; then
  echo "FEHLER: $PYTHON fehlen Pakete:$FEHLEND" >&2
  echo "  $PYTHON -m pip install$FEHLEND" >&2
  exit 1
fi

warte_auf() {
  local url="$1" name="$2" versuche="${3:-90}"
  for _ in $(seq 1 "$versuche"); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "FEHLER: $name antwortet nicht auf $url" >&2
  return 1
}

# ------------------------------------------------------------------------- Ollama
command -v ollama >/dev/null || {
  echo "FEHLER: ollama nicht gefunden — https://ollama.com/download" >&2; exit 1; }
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starte Ollama …"
  nohup ollama serve > "$LOGS/ollama.log" 2>&1 &
  warte_auf http://127.0.0.1:11434/api/tags "Ollama"
fi

if ! ollama list | awk 'NR>1 {print $1}' | grep -qxF "$MODELL"; then
  echo "Modell $MODELL fehlt — lade es …"
  ollama pull "$MODELL"
fi

# ------------------------------------------------------------------------ Bruecke
if curl -sf "http://127.0.0.1:$BRUECKE_PORT/health" >/dev/null 2>&1; then
  echo "Bruecke laeuft bereits auf $BRUECKE_PORT."
else
  echo "Starte Bruecke ($MODELL, think=$DENKEN) auf $BRUECKE_PORT …"
  PYTHONPATH="$REPO" nohup "$PYTHON" "$REPO/scripts/ollama_openai_proxy.py" \
    --model "$MODELL" --port "$BRUECKE_PORT" --think "$DENKEN" \
    > "$LOGS/bruecke.log" 2>&1 &
  warte_auf "http://127.0.0.1:$BRUECKE_PORT/health" "Bruecke"
fi

# ------------------------------------------------------------------------- Router
if curl -sf "http://127.0.0.1:$SPECIALIST_PORT/v1/models" >/dev/null 2>&1; then
  echo "Specialist auf $SPECIALIST_PORT erkannt."
else
  echo "Kein Specialist auf $SPECIALIST_PORT — der Router laeuft ohne ihn."
fi

SCHLUESSEL_ARGS=()
if [ -n "${HYDRA_API_KEY_STORE:-}" ]; then
  SCHLUESSEL_ARGS=(--api-key-store "$HYDRA_API_KEY_STORE")
  echo "Schluesselspeicher: $HYDRA_API_KEY_STORE (Bearer-Token noetig)"
elif [ "$BIND" != "127.0.0.1" ]; then
  echo "FEHLER: Bindung an $BIND ohne Schluesselspeicher waere ein offener" >&2
  echo "  Modell-Endpunkt. Erst einen Schluessel anlegen, dann erneut starten:" >&2
  echo "    python3 scripts/hydra_api_keys.py --store ~/.hydra/keys.json create --name \$(hostname)" >&2
  echo "    HYDRA_API_KEY_STORE=~/.hydra/keys.json HYDRA_BIND=$BIND $0" >&2
  exit 1
fi

if curl -sf "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null 2>&1; then
  echo "Router laeuft bereits auf $ROUTER_PORT."
else
  echo "Starte Router auf $BIND:$ROUTER_PORT …"
  PYTHONPATH="$REPO" nohup "$PYTHON" "$REPO/scripts/hydra_router_server.py" \
    --general-endpoint "http://127.0.0.1:$BRUECKE_PORT/v1/chat/completions" \
    --general-model "$MODELL" \
    --specialist-endpoint "http://127.0.0.1:$SPECIALIST_PORT/v1/chat/completions" \
    --host "$BIND" --port "$ROUTER_PORT" "${SCHLUESSEL_ARGS[@]}" \
    > "$LOGS/router.log" 2>&1 &
  warte_auf "http://127.0.0.1:$ROUTER_PORT/health" "Router"
fi

echo
curl -s "http://127.0.0.1:$ROUTER_PORT/health"
echo
echo "Bereit:  http://$BIND:$ROUTER_PORT/v1/chat/completions"
echo "Logs:    $LOGS"
