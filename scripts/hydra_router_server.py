#!/usr/bin/env python3
"""Deterministic OpenAI-compatible router for the Hydra specialist/Qwen system."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from morningstar_hydra.api_keys import ROLES, ApiKeyStore, validate_role
from morningstar_hydra.deterministic_control import ExactResolution, resolve_exact_request
from morningstar_hydra.identity import (
    branded_messages,
    deterministic_identity_response,
    is_direct_identity_request,
)

Route = Literal["specialist", "general"]

# Der allgemeine Weg zeigt auf die Ollama-Bruecke (scripts/ollama_openai_proxy.py),
# nicht direkt auf Ollama: nur dort laesst sich der Denkmodus abschalten.
DEFAULT_SPECIALIST_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_GENERAL_ENDPOINT = "http://127.0.0.1:8120/v1/chat/completions"
DEFAULT_SPECIALIST_MODEL = "hydra-4b-ch"
DEFAULT_GENERAL_MODEL = "qwen3.5:27b-int4"
DEFAULT_ROUTER_MODEL = "hydra-dual-backend"

FORWARDED_FIELDS = frozenset(
    {
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "user",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "chat_template_kwargs",
        "stream",
        "stream_options",
    }
)

SSE_DONE = b"data: [DONE]\n\n"


@dataclass(frozen=True)
class UnaryResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamingResponse:
    """A relayed upstream event stream; length is unknown until it ends."""

    status: int
    headers: dict[str, str]
    chunks: AsyncIterator[bytes]

def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def route_request(
    payload: dict[str, Any],
    *,
    specialist_model: str = DEFAULT_SPECIALIST_MODEL,
) -> Route:
    """Use Qwen by default; Hydra requires an explicit OpenAI ``model`` selection."""
    if not isinstance(payload, dict):
        raise ValueError("request payload must be an object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("every message must contain a role")
        _message_text(message.get("content"))
    return "specialist" if payload.get("model") == specialist_model else "general"


def _is_loopback_host(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def bearer_token(headers: dict[bytes, bytes]) -> str | None:
    """Liest den Bearer-Token aus den ASGI-Rohkopfzeilen."""
    roh = headers.get(b"authorization")
    if roh is None:
        return None
    try:
        wert = roh.decode("latin-1").strip()
    except UnicodeDecodeError:
        return None
    schema, _, token = wert.partition(" ")
    if schema.casefold() != "bearer":
        return None
    return token.strip() or None


def validate_upstream_url(url: str, *, allow_remote: bool = False) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("upstream must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("credential-bearing upstream URLs are forbidden")
    if parsed.query or parsed.fragment:
        raise ValueError("upstream query strings and fragments are forbidden")
    hostname = parsed.hostname.casefold()
    if hostname == "localhost":
        loopback = True
    else:
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
    if not loopback:
        if not allow_remote:
            raise ValueError("non-loopback upstream blocked")
        if parsed.scheme != "https":
            raise ValueError("remote upstreams must use HTTPS")
    return url


@dataclass(frozen=True)
class RouterConfig:
    specialist_endpoint: str = DEFAULT_SPECIALIST_ENDPOINT
    general_endpoint: str = DEFAULT_GENERAL_ENDPOINT
    specialist_model: str = DEFAULT_SPECIALIST_MODEL
    general_model: str = DEFAULT_GENERAL_MODEL
    router_model: str = DEFAULT_ROUTER_MODEL
    specialist_timeout: float = 900.0
    general_timeout: float = 900.0
    host: str = "127.0.0.1"
    # Ohne Schluesselspeicher ist der Router offen. Das ist auf Loopback in
    # Ordnung und im Netz nicht — siehe validate().
    api_key_store: Path | None = None
    allow_remote_upstreams: bool = False
    enable_exact_control: bool = False
    # Die Identitaet ist ein Produktmerkmal und darum an. Fuer Messungen gehoert
    # sie aus: die eingefuegte Systemnachricht veraendert sonst jede Antwort.
    enable_identity: bool = True
    # Modelle, die nur ein Schluessel mit der Rolle 'admin' waehlen darf. Leer
    # gelassen verhaelt sich der Router wie bisher.
    admin_only_models: frozenset[str] = frozenset()
    # Ohne Schluesselspeicher gibt es keine Identitaet. Der Router bindet dann
    # nur an Loopback (siehe unten), und wer auf der Maschine sitzt, erreicht
    # Ollama ohnehin direkt — es gibt dort kein Privileg zu schuetzen.
    anonymous_role: str = "admin"

    def validate(self) -> "RouterConfig":
        validate_upstream_url(
            self.specialist_endpoint,
            allow_remote=self.allow_remote_upstreams,
        )
        validate_upstream_url(
            self.general_endpoint,
            allow_remote=self.allow_remote_upstreams,
        )
        for label, flag in (
            ("enable_exact_control", self.enable_exact_control),
            ("enable_identity", self.enable_identity),
        ):
            if not isinstance(flag, bool):
                raise ValueError(f"{label} must be boolean")
        if not _is_loopback_host(self.host) and self.api_key_store is None:
            raise ValueError(
                "binding beyond loopback without api_key_store would expose an "
                "unauthenticated model endpoint"
            )
        validate_role(self.anonymous_role)
        if not isinstance(self.admin_only_models, frozenset):
            raise ValueError("admin_only_models must be a frozenset")
        for entry in self.admin_only_models:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("admin_only_models entries must be non-empty strings")
        # Ein rollengeschuetztes Modell ohne Schluesselspeicher waere eine
        # Attrappe: ohne Schluessel gilt jeder als anonymous_role.
        if self.admin_only_models and self.api_key_store is None and self.anonymous_role == "admin":
            raise ValueError(
                "admin_only_models without api_key_store grants everyone admin; "
                "configure a key store or set anonymous_role to 'user'"
            )
        for label, value in (
            ("specialist_model", self.specialist_model),
            ("general_model", self.general_model),
            ("router_model", self.router_model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        for label, value in (
            ("specialist_timeout", self.specialist_timeout),
            ("general_timeout", self.general_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{label} must be finite and positive")
        return self


def validate_request_options(payload: dict[str, Any]) -> None:
    thinking = payload.get("chat_template_kwargs")
    if thinking is not None:
        if (
            not isinstance(thinking, dict)
            or set(thinking) != {"enable_thinking"}
            or not isinstance(thinking.get("enable_thinking"), bool)
        ):
            raise ValueError(
                "chat_template_kwargs must contain only boolean enable_thinking"
            )
    stream = payload.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    stream_options = payload.get("stream_options")
    if stream_options is not None:
        if (
            not isinstance(stream_options, dict)
            or set(stream_options) != {"include_usage"}
            or not isinstance(stream_options.get("include_usage"), bool)
        ):
            raise ValueError("stream_options must contain only boolean include_usage")


def build_forward_payload(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    validate_request_options(payload)
    forwarded = {
        key: value
        for key, value in payload.items()
        if key in FORWARDED_FIELDS
    }
    forwarded["model"] = model_id
    return forwarded


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _deterministic_completion(model: str, resolution: ExactResolution) -> bytes:
    return _json_bytes(
        {
            "id": f"chatcmpl-control-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resolution.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def _deterministic_stream(model: str, resolution: ExactResolution) -> AsyncIterator[bytes]:
    completion_id = f"chatcmpl-control-{uuid.uuid4().hex}"

    def event(delta: dict[str, Any], finish_reason: str | None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return b"data: " + _json_bytes(payload) + b"\n\n"

    async def emit() -> AsyncIterator[bytes]:
        yield event({"role": "assistant", "content": resolution.content}, None)
        yield event({}, "stop")
        yield SSE_DONE

    return emit()


class HydraRouterApplication:
    """Small ASGI application; avoids coupling the router to framework versions."""

    def __init__(
        self,
        settings: RouterConfig,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def _chat(self, raw_body: bytes, role: str) -> UnaryResponse | StreamingResponse:
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(400, "invalid_json", "request body must be valid JSON")
        if not isinstance(payload, dict):
            return self._error(400, "invalid_request", "request body must be a JSON object")
        gewuenscht = payload.get("model")
        if (
            isinstance(gewuenscht, str)
            and gewuenscht in self.settings.admin_only_models
            and role != "admin"
        ):
            # Ausdruecklich abweisen statt still auf das allgemeine Modell
            # umzulenken: sonst haelt der Aufrufer die Antwort fuer die des
            # angefragten Modells.
            return self._error(
                403,
                "model_forbidden",
                f"model {gewuenscht!r} requires an API key with the admin role",
                headers={"X-Hydra-Role": role},
            )
        try:
            route = route_request(
                payload,
                specialist_model=self.settings.specialist_model,
            )
            validate_request_options(payload)
        except ValueError as exc:
            return self._error(422, "invalid_request", str(exc))

        resolution = (
            resolve_exact_request(payload)
            if route == "general" and self.settings.enable_exact_control
            else None
        )
        streaming = payload.get("stream") is True
        if resolution is None and self.settings.enable_identity:
            messages = payload.get("messages") or []
            if is_direct_identity_request(messages):
                antwort = deterministic_identity_response(messages)
                resolution = ExactResolution(
                    content=antwort["choices"][0]["message"]["content"],
                    capability="identity",
                )
        if resolution is not None:
            control_headers = {
                "X-Hydra-Route": (
                    "identity" if resolution.capability == "identity" else "deterministic"
                ),
                "X-Hydra-Backend": f"control-plane:{resolution.capability}",
                "X-Hydra-Role": role,
            }
            if streaming:
                return StreamingResponse(
                    200,
                    {
                        **control_headers,
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-store",
                    },
                    _deterministic_stream(self.settings.router_model, resolution),
                )
            return UnaryResponse(
                200,
                _deterministic_completion(self.settings.router_model, resolution),
                {**control_headers, "Content-Type": "application/json"},
            )

        endpoint = (
            self.settings.specialist_endpoint
            if route == "specialist"
            else self.settings.general_endpoint
        )
        backend_model = (
            self.settings.specialist_model
            if route == "specialist"
            else self.settings.general_model
        )
        timeout = (
            self.settings.specialist_timeout
            if route == "specialist"
            else self.settings.general_timeout
        )
        route_headers = {
            "X-Hydra-Route": route,
            "X-Hydra-Backend": backend_model,
            "X-Hydra-Role": role,
        }
        try:
            forwarded = build_forward_payload(payload, backend_model)
        except ValueError as exc:
            return self._error(422, "invalid_request", str(exc), headers=route_headers)
        if self.settings.enable_identity:
            # Hinter einem eigenen System-Prompt des Aufrufers, damit dessen
            # Anweisung fuehrt und die Identitaet sie nur ergaenzt.
            forwarded["messages"] = branded_messages(forwarded["messages"])
        if streaming:
            return await self._relay_stream(endpoint, forwarded, timeout, route_headers)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
                trust_env=False,
            ) as client:
                upstream = await client.post(
                    endpoint,
                    json=forwarded,
                    headers={"Content-Type": "application/json"},
                )
        except httpx.TransportError:
            return self._error(
                502,
                "upstream_transport_error",
                "selected backend is unavailable",
                headers=route_headers,
            )
        try:
            decoded = upstream.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(
                502,
                "malformed_upstream_json",
                "selected backend returned invalid JSON",
                headers=route_headers,
            )
        if not isinstance(decoded, dict):
            return self._error(
                502,
                "malformed_upstream_json",
                "selected backend returned a non-object JSON response",
                headers=route_headers,
            )
        headers = dict(route_headers)
        content_type = upstream.headers.get("content-type")
        if content_type:
            headers["Content-Type"] = content_type
        return UnaryResponse(upstream.status_code, upstream.content, headers)

    def _authenticate(self, token: str) -> dict[str, Any] | None:
        try:
            return ApiKeyStore(self.settings.api_key_store).authenticate(token)
        except (OSError, ValueError):
            # Unlesbarer oder unsicherer Speicher heisst: niemand kommt rein.
            return None

    async def _relay_stream(
        self,
        endpoint: str,
        forwarded: dict[str, Any],
        timeout: float,
        route_headers: dict[str, str],
    ) -> UnaryResponse | StreamingResponse:
        """Relay an upstream event stream, failing closed before the first chunk."""
        client = httpx.AsyncClient(
            timeout=timeout,
            transport=self.transport,
            trust_env=False,
        )
        request = client.build_request(
            "POST",
            endpoint,
            json=forwarded,
            headers={"Content-Type": "application/json"},
        )
        try:
            upstream = await client.send(request, stream=True)
        except httpx.TransportError:
            await client.aclose()
            return self._error(
                502,
                "upstream_transport_error",
                "selected backend is unavailable",
                headers=route_headers,
            )

        # Errors arrive before any event, so they can still become a JSON body.
        if upstream.status_code >= 400:
            try:
                body = await upstream.aread()
            finally:
                await upstream.aclose()
                await client.aclose()
            headers = dict(route_headers)
            content_type = upstream.headers.get("content-type")
            if content_type:
                headers["Content-Type"] = content_type
            return UnaryResponse(upstream.status_code, body, headers)

        headers = dict(route_headers)
        headers["Content-Type"] = upstream.headers.get("content-type") or "text/event-stream"
        headers["Cache-Control"] = "no-store"

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.TransportError:
                # The status line is already sent; a truncated stream is all the
                # client can be told. Swallowing keeps prompts out of the log.
                return
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(upstream.status_code, headers, relay())

    @staticmethod
    def _error(
        status: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> UnaryResponse:
        response_headers = {"Content-Type": "application/json"}
        if headers:
            response_headers.update(headers)
        return UnaryResponse(
            status,
            _json_bytes({"error": {"code": code, "message": message}}),
            response_headers,
        )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope.get("type") != "http":
            return
        method = scope.get("method", "")
        path = scope.get("path", "")

        # Ohne Schluesselspeicher gibt es keine Identitaet; dann gilt die
        # eingestellte Rolle fuer alle. validate() laesst das nur auf Loopback zu.
        role = self.settings.anonymous_role

        # /health bleibt offen, damit Monitoring ohne Schluessel prueft, ob der
        # Dienst laeuft; es gibt dort nichts preis, was nicht ohnehin bekannt ist.
        if self.settings.api_key_store is not None and path.rstrip("/") != "/health":
            headers = dict(scope.get("headers") or [])
            token = bearer_token(headers)
            identity = None if token is None else self._authenticate(token)
            if token is None:
                fehler = self._error(401, "missing_api_key", "Bearer API key required")
            elif identity is None:
                fehler = self._error(401, "invalid_api_key", "API key is not valid")
            else:
                role = identity["role"]
                fehler = None
            if fehler is not None:
                raw = [
                    (key.lower().encode("latin-1"), value.encode("latin-1"))
                    for key, value in fehler.headers.items()
                ]
                raw.append((b"www-authenticate", b"Bearer"))
                raw.append((b"content-length", str(len(fehler.body)).encode("ascii")))
                await send(
                    {"type": "http.response.start", "status": 401, "headers": raw}
                )
                await send({"type": "http.response.body", "body": fehler.body})
                return
        result: UnaryResponse | StreamingResponse
        if method == "GET" and path == "/health":
            result = UnaryResponse(
                200,
                _json_bytes(
                    {
                        "status": "ok",
                        "model": self.settings.router_model,
                        "backends": {
                            "specialist": self.settings.specialist_model,
                            "general": self.settings.general_model,
                        },
                    }
                ),
                {"Content-Type": "application/json"},
            )
        elif method == "GET" and path == "/v1/models":
            # Wer ein Modell nicht waehlen darf, bekommt es auch nicht angeboten.
            sichtbar = [
                kennung
                for kennung in (self.settings.router_model, self.settings.specialist_model)
                if role == "admin" or kennung not in self.settings.admin_only_models
            ]
            result = UnaryResponse(
                200,
                _json_bytes(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": kennung,
                                "object": "model",
                                "owned_by": "morningstar-hydra",
                            }
                            for kennung in sichtbar
                        ],
                    }
                ),
                {"Content-Type": "application/json", "X-Hydra-Role": role},
            )
        elif method == "POST" and path == "/v1/chat/completions":
            chunks: list[bytes] = []
            more_body = True
            while more_body:
                event = await receive()
                if event.get("type") == "http.disconnect":
                    return
                chunks.append(event.get("body", b""))
                more_body = bool(event.get("more_body", False))
            result = await self._chat(b"".join(chunks), role)
        else:
            result = self._error(404, "not_found", "route not found")

        raw_headers = [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in result.headers.items()
        ]
        if isinstance(result, StreamingResponse):
            await send(
                {"type": "http.response.start", "status": result.status, "headers": raw_headers}
            )
            try:
                async for chunk in result.chunks:
                    await send(
                        {"type": "http.response.body", "body": chunk, "more_body": True}
                    )
            finally:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        raw_headers.append((b"content-length", str(len(result.body)).encode("ascii")))
        await send(
            {"type": "http.response.start", "status": result.status, "headers": raw_headers}
        )
        await send({"type": "http.response.body", "body": result.body})


def create_app(
    config: RouterConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> HydraRouterApplication:
    return HydraRouterApplication((config or RouterConfig()).validate(), transport)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specialist-endpoint",
        default=os.environ.get("HYDRA_ROUTER_SPECIALIST_ENDPOINT", DEFAULT_SPECIALIST_ENDPOINT),
    )
    parser.add_argument(
        "--general-endpoint",
        default=os.environ.get("HYDRA_ROUTER_GENERAL_ENDPOINT", DEFAULT_GENERAL_ENDPOINT),
    )
    parser.add_argument(
        "--specialist-model",
        default=os.environ.get("HYDRA_ROUTER_SPECIALIST_MODEL", DEFAULT_SPECIALIST_MODEL),
    )
    parser.add_argument(
        "--general-model",
        default=os.environ.get("HYDRA_ROUTER_GENERAL_MODEL", DEFAULT_GENERAL_MODEL),
    )
    parser.add_argument(
        "--router-model",
        default=os.environ.get("HYDRA_ROUTER_MODEL", DEFAULT_ROUTER_MODEL),
    )
    parser.add_argument(
        "--specialist-timeout",
        type=float,
        default=_env_float("HYDRA_ROUTER_SPECIALIST_TIMEOUT", 900.0),
    )
    parser.add_argument(
        "--general-timeout",
        type=float,
        default=_env_float("HYDRA_ROUTER_GENERAL_TIMEOUT", 900.0),
    )
    parser.add_argument("--allow-remote-upstreams", action="store_true")
    parser.add_argument(
        "--api-key-store",
        type=Path,
        default=(Path(os.environ["HYDRA_API_KEY_STORE"])
                 if os.environ.get("HYDRA_API_KEY_STORE") else None),
        help="JSON-Schluesselspeicher; verlangt einen Bearer-Token je Anfrage. "
             "Pflicht, sobald --host ueber Loopback hinaus bindet. "
             "Schluessel anlegen mit scripts/hydra_api_keys.py",
    )
    parser.add_argument(
        "--enable-exact-control",
        action="store_true",
        help="opt in to deterministic exact-answer plugins (disabled by default)",
    )
    parser.add_argument(
        "--no-identity",
        dest="enable_identity",
        action="store_false",
        help="Identitaet nicht einfuegen — fuer Messungen, deren Zahlen sonst "
             "von der zusaetzlichen Systemnachricht abhaengen",
    )
    parser.add_argument(
        "--admin-only-model",
        action="append",
        default=None,
        metavar="MODEL_ID",
        help="Modell, das nur ein Schluessel mit der Rolle 'admin' waehlen darf. "
             "Mehrfach angebbar. Verlangt --api-key-store.",
    )
    parser.add_argument(
        "--anonymous-role",
        choices=sorted(ROLES),
        default=os.environ.get("HYDRA_ROUTER_ANONYMOUS_ROLE", "admin"),
        help="Rolle ohne Schluesselspeicher (Standard: admin). Ohne Speicher "
             "bindet der Router nur an Loopback, wo es kein Privileg zu "
             "schuetzen gibt.",
    )
    parser.add_argument("--host", default=os.environ.get("HYDRA_ROUTER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HYDRA_ROUTER_PORT", "18080")),
    )
    args = parser.parse_args()
    config = RouterConfig(
        specialist_endpoint=args.specialist_endpoint,
        general_endpoint=args.general_endpoint,
        specialist_model=args.specialist_model,
        general_model=args.general_model,
        router_model=args.router_model,
        specialist_timeout=args.specialist_timeout,
        general_timeout=args.general_timeout,
        allow_remote_upstreams=args.allow_remote_upstreams,
        enable_exact_control=args.enable_exact_control,
        enable_identity=args.enable_identity,
        admin_only_models=frozenset(args.admin_only_model or ()),
        anonymous_role=args.anonymous_role,
        host=args.host,
        api_key_store=args.api_key_store,
    ).validate()

    import uvicorn

    uvicorn.run(
        create_app(config),
        host=args.host,
        port=args.port,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
