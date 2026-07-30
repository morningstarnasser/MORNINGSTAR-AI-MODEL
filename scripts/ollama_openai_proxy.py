#!/usr/bin/env python3
"""OpenAI-kompatible Bruecke zu Ollama mit steuerbarem Denkmodus.

Ollamas eigener `/v1`-Endpunkt ignoriert `think` und `chat_template_kwargs`. Bei
einem Denkmodell wie Qwen3.5 oder Gemma 4 laesst sich das Denken dort also nicht
abschalten — und bei knappem Token-Budget kommt eine Antwort ganz ohne `content`
zurueck, weil das Budget vollstaendig in den Denkteil geht.

Diese Bruecke uebersetzt auf `/api/chat`, reicht `think` durch und normalisiert
Werkzeugaufrufe ins OpenAI-Format (Ollama liefert Argumente als Objekt, OpenAI
erwartet einen String). Streaming wird unterstuetzt: Ollamas NDJSON wird in
Server-Sent-Events uebersetzt.

Start:  python scripts/ollama_openai_proxy.py --model qwen3.5:27b-int4 --port 8120
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STATE: dict = {}
SSE_DONE = b"data: [DONE]\n\n"


def _message_to_ollama(nachricht: dict) -> dict:
    """Uebersetzt eine OpenAI-Verlaufsnachricht in Ollamas Form.

    Ollama erwartet Werkzeugargumente als Objekt und benennt das Ergebnis mit
    `tool_name`; OpenAI liefert einen String und `tool_call_id`. Ohne diese
    Rueckuebersetzung lehnt Ollama jeden Verlauf mit Werkzeugen ab (HTTP 400) —
    eine agentische Schleife bricht dann im zweiten Schritt ab.
    """
    if nachricht.get("role") == "tool":
        umgebaut = {"role": "tool", "content": nachricht.get("content") or ""}
        name = nachricht.get("name") or nachricht.get("tool_name")
        if name:
            umgebaut["tool_name"] = name
        return umgebaut
    aufrufe = nachricht.get("tool_calls")
    if not aufrufe:
        return nachricht
    umgebaute = []
    for aufruf in aufrufe:
        funktion = dict(aufruf.get("function") or {})
        argumente = funktion.get("arguments")
        if isinstance(argumente, str):
            try:
                funktion["arguments"] = json.loads(argumente or "{}")
            except json.JSONDecodeError:
                funktion["arguments"] = {}
        umgebaute.append({"function": funktion})
    return {**{k: v for k, v in nachricht.items() if k != "tool_calls"},
            "tool_calls": umgebaute}


def to_ollama(payload: dict, *, model: str, think: bool, stream: bool) -> dict:
    optionen: dict[str, Any] = {"temperature": float(payload.get("temperature") or 0.0)}
    if payload.get("max_tokens"):
        optionen["num_predict"] = int(payload["max_tokens"])
    if payload.get("top_p") is not None:
        optionen["top_p"] = float(payload["top_p"])
    if payload.get("seed") is not None:
        optionen["seed"] = int(payload["seed"])
    if payload.get("stop"):
        optionen["stop"] = payload["stop"]
    anfrage: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_ollama(m) for m in payload.get("messages", [])],
        "stream": stream,
        "think": think,
        "options": optionen,
    }
    if payload.get("tools"):
        anfrage["tools"] = payload["tools"]
    return anfrage


def _tool_calls(nachricht: dict) -> list[dict]:
    return [
        {
            "index": index,
            "id": f"call_{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {
                "name": (aufruf.get("function") or {}).get("name", ""),
                # Ollama liefert die Argumente als Objekt, OpenAI erwartet einen String.
                "arguments": json.dumps((aufruf.get("function") or {}).get("arguments", {}),
                                        ensure_ascii=False),
            },
        }
        for index, aufruf in enumerate(nachricht.get("tool_calls") or [])
    ]


def to_openai(antwort: dict, model: str) -> dict:
    nachricht = antwort.get("message") or {}
    aufrufe = _tool_calls(nachricht)
    ergebnis: dict[str, Any] = {
        "role": "assistant",
        "content": None if aufrufe else (nachricht.get("content") or ""),
        "tool_calls": [{k: v for k, v in a.items() if k != "index"} for a in aufrufe],
    }
    if nachricht.get("thinking"):
        ergebnis["reasoning"] = nachricht["thinking"]
    eingabe = antwort.get("prompt_eval_count", 0) or 0
    ausgabe = antwort.get("eval_count", 0) or 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": ergebnis,
                     "finish_reason": "tool_calls" if aufrufe else "stop"}],
        "usage": {"prompt_tokens": eingabe, "completion_tokens": ausgabe,
                  "total_tokens": eingabe + ausgabe},
    }


def stream_chunk(zeile: dict, completion_id: str, model: str) -> bytes:
    """Uebersetzt eine NDJSON-Zeile von Ollama in ein SSE-Ereignis."""
    nachricht = zeile.get("message") or {}
    aufrufe = _tool_calls(nachricht)
    delta: dict[str, Any] = {"role": "assistant"}
    if aufrufe:
        delta["tool_calls"] = aufrufe
    else:
        delta["content"] = nachricht.get("content") or ""
    if nachricht.get("thinking"):
        delta["reasoning"] = nachricht["thinking"]
    finish = None
    if zeile.get("done"):
        finish = "tool_calls" if aufrufe else "stop"
    ereignis = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(ereignis, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _post(anfrage: dict):
    request = urllib.request.Request(
        STATE["upstream"], data=json.dumps(anfrage).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(request, timeout=STATE["timeout"])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
                             "data": [{"id": STATE["model"], "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def _chunked(self, nutzlast: bytes) -> None:
        self.wfile.write(f"{len(nutzlast):X}\r\n".encode("ascii") + nutzlast + b"\r\n")
        self.wfile.flush()

    def _relay_stream(self, anfrage: dict, model: str) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        try:
            antwort = _post(anfrage)
        except (urllib.error.URLError, OSError) as fehler:
            # Noch kein Byte gesendet, also darf das ein normaler Fehler bleiben.
            self._send(502, {"error": {"code": "upstream_error", "message": str(fehler)[:200]}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            with antwort:
                for zeile in antwort:
                    if not zeile.strip():
                        continue
                    try:
                        geladen = json.loads(zeile)
                    except json.JSONDecodeError:
                        continue
                    self._chunked(stream_chunk(geladen, completion_id, model))
                    if geladen.get("done"):
                        break
            self._chunked(SSE_DONE)
            self._chunked(b"")
        except OSError:
            # Verbindung weg oder Upstream abgerissen: der Strom endet hier still,
            # damit kein Fehlertext als Modellausgabe beim Client landet.
            return

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        laenge = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(laenge) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"code": "invalid_json", "message": "body must be JSON"}})
            return
        model = payload.get("model") or STATE["model"]
        stream = payload.get("stream") is True
        anfrage = to_ollama(payload, model=STATE["model"], think=STATE["think"], stream=stream)
        if stream:
            self._relay_stream(anfrage, model)
            return
        try:
            with _post(anfrage) as antwort:
                geladen = json.loads(antwort.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as fehler:
            self._send(502, {"error": {"code": "upstream_error", "message": str(fehler)[:200]}})
            return
        self._send(200, to_openai(geladen, model))

    def log_message(self, *_args):  # Prompts gehoeren nicht ins Log.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--think", choices=("on", "off"), default="off",
                        help="Denkteil des Modells; 'off' spart Latenz und Token")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    STATE.update(model=args.model, upstream=args.upstream,
                 think=(args.think == "on"), timeout=args.timeout)
    print(f"Bruecke fuer {args.model} (think={args.think}) auf "
          f"http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
