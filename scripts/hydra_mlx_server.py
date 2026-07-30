#!/usr/bin/env python3
"""OpenAI-kompatibler Server fuer Apertus auf MLX — mit Tool-Call-Bruecke.

`mlx_lm.server` scheitert bei Apertus an zwei unabhaengigen Stellen:

1. **Tools erreichen das Modell nicht.** Der Standardweg reicht `tools=` an
   `apply_chat_template` weiter, doch Apertus' Template liest Werkzeuge aus einer
   `developer`-Rolle (`content.formatted_tools`). Ueber den Standardweg steht im
   Prompt woertlich "Tool Capabilities: disabled" — das Modell erfaehrt nie, dass
   es Werkzeuge gibt.
2. **Korrekte Aufrufe werden nicht erkannt.** MLX leitet seinen Tool-Parser aus
   dem Template-Text ab; fuer Apertus (`<SPECIAL_71/72>`) greift keines der
   bekannten Muster, also bleibt `message.tool_calls` leer.

Beides behebt dieser Server. Ausserdem laedt er Adapter zuverlaessig — anders als
`mlx_lm.server`, das `--adapter-path` still verwirft.

Start:  python scripts/hydra_mlx_server.py --model <pfad> [--adapter-path <pfad>] [--port 8080]
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

TOOL_BLOCK = re.compile(r"<SPECIAL_71>(.*?)<SPECIAL_72>", re.S)
# Freiform-Variante, die Apertus in der Praxis erzeugt: `name {"arg": "wert"}`
TOOL_INLINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})\s*$", re.S)

# Qwen3 stellt jeder Antwort einen <think>-Block voran — auch bei abgeschaltetem
# Denken, dann leer. Bleibt er im `content`, vergleicht jeder exakte Grader gegen
# "<think>…</think>\n\nZH" statt gegen "ZH" und wertet richtige Antworten falsch.
THINK_BLOCK = re.compile(r"\A\s*<think>(.*?)</think>\s*", re.S)
THINK_OPEN = re.compile(r"\A\s*<think>(.*)\Z", re.S)
# Qwens natives Aufrufformat.
TOOL_QWEN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Trennt einen führenden <think>-Block vom eigentlichen Antworttext."""
    treffer = THINK_BLOCK.match(text)
    if treffer is not None:
        return text[treffer.end():].strip(), treffer.group(1).strip() or None
    offen = THINK_OPEN.match(text)
    if offen is not None:
        # Das Token-Budget endete mitten im Denken: es gibt keine Antwort.
        return "", offen.group(1).strip() or None
    return text, None


def template_kwargs(payload: dict) -> dict:
    """Erlaubt ausschließlich boolesches enable_thinking an das Chat-Template."""
    roh = payload.get("chat_template_kwargs")
    if (
        isinstance(roh, dict)
        and set(roh) == {"enable_thinking"}
        and isinstance(roh.get("enable_thinking"), bool)
    ):
        return {"enable_thinking": roh["enable_thinking"]}
    return {}

STATE: dict = {}


# Ohne diesen Hinweis lehnt Apertus Werkzeugaufrufe teilweise ab ("ich habe keinen
# Zugriff auf externe Systeme") — ein Alignment-Reflex, kein fehlendes Koennen. Der
# Text bleibt bewusst generisch: er nennt keine Aufgabe, kein Werkzeug und kein Format.
TOOL_SYSTEM_PROMPT = (
    "Du bist ein Werkzeug-Agent. Die im Abschnitt Tool Capabilities aufgeführten "
    "Funktionen stehen dir zur Verfügung und werden vom aufrufenden System ausgeführt. "
    "Wähle die passende Funktion und gib den Aufruf aus. Lehne nicht ab und erkläre "
    "nicht, dass dir der Zugriff fehlt."
)


def build_developer_message(tools: list[dict]) -> dict:
    """Verpackt OpenAI-Tools so, wie Apertus' Template sie erwartet."""
    funktionen = [t.get("function", t) for t in tools]
    return {"role": "developer",
            "content": {"has_thinking": False,
                        "formatted_tools": json.dumps(funktionen, ensure_ascii=False)}}


def render_native_tool_prompt(tokenizer, messages: list[dict], tools: list[dict],
                              zusatz: dict) -> str | None:
    """Rendert mit nativen Werkzeugen — aber nur, wenn sie im Prompt ankommen.

    Genau hier scheiterte Apertus: Das Template akzeptiert `tools=`, schreibt aber
    "Tool Capabilities: disabled" und verschweigt dem Modell jedes Werkzeug. Statt
    das Modell zu raten, wird der gerenderte Prompt geprüft. Kommen die Namen nicht
    vor, ist der Weg untauglich und der Aufrufer nimmt die Sonderbrücke.
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools, **zusatz)
    except (TypeError, ValueError):
        return None
    namen = {t.get("function", t).get("name") for t in tools}
    if not namen or not all(name and name in prompt for name in namen):
        return None
    return prompt


def parse_tool_calls(text: str, tools: list[dict]) -> list[dict]:
    """Erkennt Tool-Aufrufe von Apertus und Qwen und uebersetzt sie ins OpenAI-Format."""
    bekannt = {t.get("function", t).get("name") for t in tools}
    kandidaten: list[tuple[str, str]] = []

    for treffer in TOOL_QWEN.finditer(text):
        try:
            geladen = json.loads(treffer.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(geladen, dict) and isinstance(geladen.get("name"), str):
            kandidaten.append((geladen["name"],
                               json.dumps(geladen.get("arguments", {}), ensure_ascii=False)))
    if kandidaten:
        return [{"id": f"call_{uuid.uuid4().hex[:20]}", "type": "function",
                 "function": {"name": name, "arguments": args}}
                for name, args in kandidaten if name in bekannt]

    block = TOOL_BLOCK.search(text)
    if block:
        try:
            geladen = json.loads(block.group(1).strip())
            if isinstance(geladen, dict):
                geladen = [geladen]
            for eintrag in geladen:
                if not isinstance(eintrag, dict):
                    continue
                if "name" in eintrag:            # {"name": ..., "arguments": {...}}
                    kandidaten.append((eintrag["name"],
                                       json.dumps(eintrag.get("arguments", {}),
                                                  ensure_ascii=False)))
                else:                             # {"funktionsname": {...args...}}
                    for name, args in eintrag.items():
                        kandidaten.append((name, json.dumps(args, ensure_ascii=False)))
        except (json.JSONDecodeError, AttributeError):
            pass

    if not kandidaten:
        # Blanke JSON-Variante ohne Marker: {"funktionsname": {...argumente...}}
        try:
            geladen = json.loads(text.strip())
        except json.JSONDecodeError:
            geladen = None
        if isinstance(geladen, dict):
            if geladen.get("name") in bekannt:
                kandidaten.append((geladen["name"],
                                   json.dumps(geladen.get("arguments", {}),
                                              ensure_ascii=False)))
            elif len(geladen) == 1:
                (name, args), = geladen.items()
                if name in bekannt and isinstance(args, dict):
                    kandidaten.append((name, json.dumps(args, ensure_ascii=False)))

    if not kandidaten:
        treffer = TOOL_INLINE.match(text.strip())
        if treffer and treffer.group(1) in bekannt:
            try:
                json.loads(treffer.group(2))
                kandidaten.append((treffer.group(1), treffer.group(2)))
            except json.JSONDecodeError:
                pass

    return [{"id": f"call_{uuid.uuid4().hex[:20]}", "type": "function",
             "function": {"name": name, "arguments": args}}
            for name, args in kandidaten if name in bekannt]


def erzeuge_antwort(payload: dict) -> dict:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = STATE["model"], STATE["tokenizer"]
    messages = list(payload.get("messages", []))
    tools = payload.get("tools") or []
    zusatz = template_kwargs(payload)

    # Modelle mit eigenem Werkzeug-Protokoll (Qwen) bekommen es nativ; nur wenn das
    # Template die Werkzeuge verschluckt, greift die Apertus-Sonderbrücke.
    prompt = render_native_tool_prompt(tokenizer, messages, tools, zusatz) if tools else None
    if prompt is None:
        if tools:
            vorspann = [build_developer_message(tools)]
            if not any(m.get("role") == "system" for m in messages):
                vorspann.insert(0, {"role": "system", "content": TOOL_SYSTEM_PROMPT})
            messages = vorspann + messages
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, **zusatz)
        except (TypeError, ValueError):
            # Nicht jedes Template kennt enable_thinking (Apertus etwa nicht).
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
    sampler = make_sampler(temp=float(payload.get("temperature") or 0.0))
    text = generate(model, tokenizer, prompt=prompt,
                    max_tokens=int(payload.get("max_tokens") or 512),
                    sampler=sampler, verbose=False)

    # Erst den Denkteil abtrennen, dann Werkzeuge suchen: ein im Denken erwogener
    # JSON-Block ist kein Aufruf und darf nicht als solcher geparst werden.
    inhalt, reasoning = split_reasoning(text)
    tool_calls = parse_tool_calls(inhalt, tools) if tools else []
    message = {"role": "assistant",
               "content": None if tool_calls else inhalt,
               "tool_calls": tool_calls}
    if reasoning:
        message["reasoning"] = reasoning
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "model": payload.get("model") or STATE["name"],
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        roh = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(roh)))
        self.end_headers()
        self.wfile.write(roh)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/health"):
            self._send(200, {"object": "list",
                             "data": [{"id": STATE["name"], "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def _send_stream(self, antwort: dict) -> None:
        """Liefert die fertige Antwort als ein SSE-Chunk plus Abschluss.

        Kein echtes Token-Streaming, aber das Protokoll, das OpenAI-Clients
        erwarten — ohne das bleibt etwa die Morningstar-CLI stumm.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        wahl = antwort["choices"][0]
        nachricht = wahl["message"]
        delta = {"role": "assistant"}
        if nachricht.get("content"):
            delta["content"] = nachricht["content"]
        if nachricht.get("tool_calls"):
            delta["tool_calls"] = [
                {"index": i, **tc} for i, tc in enumerate(nachricht["tool_calls"])
            ]
        rumpf = {"id": antwort["id"], "object": "chat.completion.chunk",
                 "model": antwort["model"]}
        for stueck in ({**rumpf, "choices": [{"index": 0, "delta": delta,
                                              "finish_reason": None}]},
                       {**rumpf, "choices": [{"index": 0, "delta": {},
                                              "finish_reason": wahl["finish_reason"]}]}):
            self.wfile.write(f"data: {json.dumps(stueck, ensure_ascii=False)}\n\n"
                             .encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        laenge = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(laenge) or b"{}")
            antwort = erzeuge_antwort(payload)
            if payload.get("stream"):
                self._send_stream(antwort)
            else:
                self._send(200, antwort)
        except Exception as exc:  # noqa: BLE001 - Fehler gehoert in die Antwort
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    def log_message(self, *_args):
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    from mlx_lm import load
    print(f"Lade {args.model}"
          + (f" + Adapter {args.adapter_path}" if args.adapter_path else ""), flush=True)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    STATE.update({"model": model, "tokenizer": tokenizer, "name": args.model})

    server = HTTPServer((args.host, args.port), Handler)
    print(f"bereit auf http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
