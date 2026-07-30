#!/bin/bash
# ============================================================================
# Morningstar-14B-Code — One-Click Build Script
# Erstellt das Morningstar Model fuer Ollama
# Developed by: Morningstar Contributors (github.com/your-org)
# ============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

MODEL_NAME="morningstar"
BASE_MODEL="qwen2.5-coder:14b"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║       MORNINGSTAR-14B-CODE  — Model Builder     ║"
echo "  ║       by Morningstar Contributors (github/your-org)   ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── 1. Check Ollama ──────────────────────────────────────
echo -e "${BOLD}[1/5] Pruefe Ollama Installation...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}  ✗ Ollama ist nicht installiert!${NC}"
    echo ""
    echo "  Installiere Ollama:"
    echo "    macOS:  brew install ollama"
    echo "    Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "    Web:    https://ollama.com"
    echo ""
    exit 1
fi
OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
echo -e "${GREEN}  ✓ Ollama installiert ($OLLAMA_VERSION)${NC}"

# ─── 2. Check Ollama Server ──────────────────────────────
echo -e "${BOLD}[2/5] Pruefe Ollama Server...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}  ! Ollama Server laeuft nicht. Starte...${NC}"
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}  ✗ Konnte Ollama Server nicht starten.${NC}"
        echo "  Starte manuell: ollama serve"
        exit 1
    fi
    echo -e "${GREEN}  ✓ Ollama Server gestartet (PID: $OLLAMA_PID)${NC}"
else
    echo -e "${GREEN}  ✓ Ollama Server laeuft${NC}"
fi

# ─── 3. Pull Base Model ─────────────────────────────────
echo -e "${BOLD}[3/5] Lade Base Model: ${BASE_MODEL}...${NC}"
if ollama list 2>/dev/null | grep -q "qwen2.5-coder"; then
    echo -e "${GREEN}  ✓ Base Model bereits vorhanden${NC}"
else
    echo -e "${YELLOW}  ↓ Downloading ${BASE_MODEL} (~9GB)...${NC}"
    echo "  Das kann einige Minuten dauern..."
    ollama pull "$BASE_MODEL"
    echo -e "${GREEN}  ✓ Base Model heruntergeladen${NC}"
fi

# ─── 4. Create Morningstar Model ─────────────────────────
echo -e "${BOLD}[4/5] Erstelle Morningstar Model...${NC}"

if [ ! -f "$SCRIPT_DIR/Modelfile" ]; then
    echo -e "${RED}  ✗ Modelfile nicht gefunden in: $SCRIPT_DIR${NC}"
    exit 1
fi

# Remove old version if rebuilding
if [ "$1" = "--rebuild" ]; then
    echo -e "${YELLOW}  Entferne alte Version...${NC}"
    ollama rm "$MODEL_NAME" 2>/dev/null || true
fi

echo -e "${CYAN}  Erstelle ${MODEL_NAME} aus Modelfile...${NC}"
ollama create "$MODEL_NAME" -f "$SCRIPT_DIR/Modelfile"
echo -e "${GREEN}  ✓ Model '${MODEL_NAME}' erstellt!${NC}"

# ─── 5. Test ─────────────────────────────────────────────
echo -e "${BOLD}[5/5] Teste Model...${NC}"

TEST_RESPONSE=$(ollama run "$MODEL_NAME" "Write a Python one-liner that reverses a string. Only the code, nothing else." 2>&1 | head -5)
if [ -n "$TEST_RESPONSE" ]; then
    echo -e "${GREEN}  ✓ Model antwortet:${NC}"
    echo "    $TEST_RESPONSE"
else
    echo -e "${YELLOW}  ! Keine Antwort erhalten. Pruefe: ollama run $MODEL_NAME${NC}"
fi

# ─── Fertig ──────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ MORNINGSTAR Model erfolgreich erstellt!  ${NC}"
echo -e "${GREEN}${BOLD}  ════════════════════════════════════════════${NC}"
echo ""
echo -e "  Nutzen:"
echo -e "    ${CYAN}ollama run morningstar${NC}                  Interaktiv"
echo -e "    ${CYAN}morningstar --model morningstar${NC}         In Morningstar CLI"
echo -e "    ${CYAN}curl localhost:11434/api/generate${NC}       API"
echo ""
echo -e "  Model-Pfad:  ${YELLOW}~/.ollama/models/${NC}"
echo -e "  Modelfile:   ${YELLOW}${SCRIPT_DIR}/Modelfile${NC}"
echo ""
