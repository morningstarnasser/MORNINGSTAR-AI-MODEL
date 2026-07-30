from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .api_keys import ApiKeyStore
from .core import CascadingBackend, HydraEngine, HydraManifest, OpenAIBackend


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def is_loopback_bind(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def is_loopback_client(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return False
    try:
        return ipaddress.ip_address(str(address[0])).is_loopback
    except ValueError:
        return str(address[0]) == "localhost"


@dataclass
class AuthConfig:
    store: ApiKeyStore | None = None
    legacy_api_key: str = ""
    force_required: bool = False

    @property
    def has_configured_auth(self) -> bool:
        # A configured store is itself an authentication boundary. Missing or
        # empty stores therefore fail closed instead of silently disabling auth.
        return bool(self.legacy_api_key) or self.store is not None

    @property
    def requires_auth(self) -> bool:
        return self.force_required or self.has_configured_auth

    def authorize(self, authorization_header: str | None) -> bool:
        if not self.requires_auth:
            return True
        if not authorization_header or not authorization_header.startswith("Bearer "):
            return False
        supplied = authorization_header[len("Bearer ") :].strip()
        if not supplied:
            return False
        ok = False
        if self.legacy_api_key:
            ok = hmac.compare_digest(supplied, self.legacy_api_key)
        if self.store:
            ok = bool(ok or self.store.verify(supplied))
        return ok


def auth_config_from_env() -> AuthConfig:
    store_path = os.environ.get("HYDRA_API_KEY_STORE")
    store = ApiKeyStore(store_path) if store_path else None
    return AuthConfig(
        store=store,
        legacy_api_key=os.environ.get("HYDRA_API_KEY", ""),
        force_required=parse_bool(os.environ.get("HYDRA_REQUIRE_API_KEY")),
    )


def ensure_bind_auth_safe(host: str, auth: AuthConfig) -> None:
    if not is_loopback_bind(host) and not auth.has_configured_auth:
        raise SystemExit("API authentication is required when binding beyond loopback")


def completion_to_sse_events(result: dict) -> list[bytes]:
    """Convert a non-streaming OpenAI completion into standards-compatible SSE."""
    choice = (result.get("choices") or [{}])[0]
    message = dict(choice.get("message") or {})
    chunk_base = {
        "id": result.get("id", "chatcmpl-morningstar-hydra"),
        "object": "chat.completion.chunk",
        "created": result.get("created", 0),
        "model": result.get("model", "morningstar-hydra"),
    }
    first = {
        **chunk_base,
        "choices": [{"index": int(choice.get("index", 0)), "delta": message, "finish_reason": None}],
    }
    final = {
        **chunk_base,
        "choices": [
            {
                "index": int(choice.get("index", 0)),
                "delta": {},
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        ],
    }
    events = [
        b"data: " + json.dumps(first, ensure_ascii=False).encode("utf-8") + b"\n\n",
        b"data: " + json.dumps(final, ensure_ascii=False).encode("utf-8") + b"\n\n",
    ]
    if result.get("usage"):
        usage = {**chunk_base, "choices": [], "usage": result["usage"]}
        events.append(b"data: " + json.dumps(usage, ensure_ascii=False).encode("utf-8") + b"\n\n")
    events.append(b"data: [DONE]\n\n")
    return events


class Handler(BaseHTTPRequestHandler):
    engine: HydraEngine
    auth_config: AuthConfig = AuthConfig()
    max_body_bytes: int = 1_048_576

    def authorized(self) -> bool:
        return self.auth_config.authorize(self.headers.get("Authorization"))

    def client_is_loopback(self) -> bool:
        return is_loopback_client(getattr(self, "client_address", None))

    def send_json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_unauthorized(self) -> None:
        self.send_json(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})

    def send_sse(self, result: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in completion_to_sse_events(result):
            self.wfile.write(event)
            self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/health":
            if self.auth_config.requires_auth and not self.client_is_loopback() and not self.authorized():
                self.send_unauthorized()
                return
            status = self.engine.status()
            self.send_json(200 if status["backend"].get("ok") else 503, status)
            return
        if not self.authorized():
            self.send_unauthorized()
            return
        if self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": "morningstar-hydra", "object": "model", "owned_by": "morningstar"}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.authorized():
            self.send_unauthorized()
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > self.max_body_bytes:
                self.send_json(413, {"error": "request body must be between 1 byte and 1 MiB"})
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            passthrough_keys = (
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
                "stop",
                "seed",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "n",
                "logprobs",
                "top_logprobs",
                "reasoning_effort",
            )
            extra = {key: payload[key] for key in passthrough_keys if key in payload}
            result = self.engine.chat(
                payload.get("messages") or [],
                max_tokens=min(2048, max(1, int(payload.get("max_tokens", 256)))),
                temperature=float(payload.get("temperature", 0.2)),
                extra=extra,
            )
            if payload.get("stream"):
                self.send_sse(result)
            else:
                self.send_json(200, result)
        except Exception as exc:
            print("[hydra-api] backend error:", repr(exc))
            self.send_json(502, {"error": "model backend unavailable"})

    def log_message(self, fmt: str, *args: object) -> None:
        print("[hydra-api] " + fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="config/hydra-manifest.json")
    parser.add_argument("--backend", default="http://100.122.90.39:18080")
    parser.add_argument("--runtime", choices=("auto", "local", "nim"), default="auto")
    parser.add_argument("--nim-backend", default="https://integrate.api.nvidia.com")
    parser.add_argument("--nim-model", default="deepseek-ai/deepseek-v4-pro")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    local_backend = OpenAIBackend(args.backend, name="nas-cpu")
    nim_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if args.runtime == "nim" and not nim_key:
        raise SystemExit("NVIDIA_API_KEY is required for --runtime nim")
    if args.runtime in {"auto", "nim"} and nim_key:
        nim_backend = OpenAIBackend(
            args.nim_backend,
            timeout_seconds=120,
            headers={"Authorization": "Bearer " + nim_key},
            model_override=args.nim_model,
            health_path="/v1/models",
            name="nvidia-nim",
        )
        backend = CascadingBackend(nim_backend, None if args.runtime == "nim" else local_backend)
    else:
        backend = local_backend
    Handler.engine = HydraEngine(HydraManifest.load(args.manifest), backend)
    Handler.auth_config = auth_config_from_env()
    ensure_bind_auth_safe(args.host, Handler.auth_config)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Morningstar Hydra API on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
