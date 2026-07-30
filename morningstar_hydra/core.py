from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .identity import (
    CANONICAL_IDENTITY,
    branded_messages,
    contains_identity_request,
    deterministic_identity_response,
    harden_mixed_identity_result,
    harden_unsupported_action_claims,
    is_direct_identity_request,
    strip_unsolicited_identity_prefix,
)

STRUCTURED_OUTPUT_INSTRUCTION = (
    "Structured-output discipline (only for this structured request): For JSON requests, return raw JSON only—no Markdown "
    "fences or prose. In JSON and native tool outputs, encode dates as ISO-8601 YYYY-MM-DD. Echo requested values exactly "
    "without labels, prefixes, or extra words. When tools are provided, use native OpenAI tool_calls; do not describe the "
    "call or serialize it as assistant text."
)
_JSON_STRUCTURE_PATTERN = re.compile(
    r"\b(?:raw\s+json|valid\s+json|json[- ]?(?:object|objekt|objet|oggetto|objeto|schema|output|format))\b",
    re.IGNORECASE,
)
_JSON_OUTPUT_VERB_PATTERN = re.compile(
    r"\b(?:return|output|emit|provide|extract|extrahiere|antworte|gib|liefere|erstelle|renvoie|retourne|"
    r"restituisci|devuelve|retorna)\b",
    re.IGNORECASE,
)
_JSON_STRICT_OUTPUT_PATTERN = re.compile(
    r"(?:\b(?:without|ohne|sans|senza|sin)\s+markdown\b|\b(?:raw\s+)?json\s+only\b|"
    r"\bonly\s+(?:raw\s+)?json\b|\bnothing\s+(?:else|but)\b)",
    re.IGNORECASE,
)
_JSON_DIRECT_REQUEST_PATTERN = re.compile(
    r"\breturn\s+(?:one|a|an)\s+json[- ]?(?:object|objekt|objet|oggetto|objeto)\b",
    re.IGNORECASE,
)


def structured_output_requested(messages: list[dict[str, Any]], extra: dict[str, Any] | None) -> bool:
    """Detect API-level or prompt-level structured requests without affecting ordinary prose chat."""
    options = extra or {}
    if isinstance(options.get("tools"), list) and bool(options["tools"]):
        return True
    response_format = options.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") in {"json_object", "json_schema"}:
        return True
    for item in messages:
        if item.get("role") != "user" or not isinstance(item.get("content"), str):
            continue
        content = item["content"]
        if _JSON_STRUCTURE_PATTERN.search(content) and (
            _JSON_DIRECT_REQUEST_PATTERN.search(content)
            or (
                _JSON_OUTPUT_VERB_PATTERN.search(content)
                and _JSON_STRICT_OUTPUT_PATTERN.search(content)
            )
        ):
            return True
    return False


def apply_structured_output_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    insertion = 0
    while insertion < len(messages) and messages[insertion].get("role") in {"system", "developer"}:
        insertion += 1
    instruction = {"role": "system", "content": STRUCTURED_OUTPUT_INSTRUCTION}
    return [*messages[:insertion], instruction, *messages[insertion:]]


@dataclass(frozen=True)
class RouteDecision:
    selected_expert: str
    executed_expert: str
    score: int
    fallback_reason: str | None


class ChatBackend(Protocol):
    def health(self) -> dict[str, Any]: ...

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class BackendFailure(RuntimeError):
    def __init__(self, message: str, *, transient: bool, status_code: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.status_code = status_code


class HydraManifest:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.shared_parameters = int(data["parameter_accounting"]["shared_parameters"])
        self.active_parameters = int(data["parameter_accounting"]["active_parameters_per_request"])
        self.families = list(data["expert_families"])
        self.by_id = {item["id"]: item for item in self.families}
        self.validate()

    @classmethod
    def load(cls, path: str | Path) -> "HydraManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def validate(self) -> None:
        identity = self.data.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("manifest requires an identity object")
        if set(identity) != set(CANONICAL_IDENTITY):
            raise ValueError("manifest identity must contain only canonical identity fields")
        for key, value in CANONICAL_IDENTITY.items():
            if identity.get(key) != value:
                raise ValueError(f"manifest identity.{key} must be {value!r}")
        if len(self.by_id) != len(self.families):
            raise ValueError("expert family ids must be unique")
        if "general" not in self.by_id:
            raise ValueError("manifest requires a general expert")
        for family in self.families:
            if int(family["parameter_count"]) <= 0:
                raise ValueError(f"invalid parameter count for {family['id']}")
            if family.get("status") not in {"planned", "training", "ready", "disabled"}:
                raise ValueError(f"invalid status for {family['id']}")
            if family.get("status") == "ready":
                if not family.get("artifact_uri") or not family.get("backend_model"):
                    raise ValueError(f"ready expert {family['id']} requires artifact_uri and backend_model")
                if not re.fullmatch(r"[0-9a-f]{64}", str(family.get("sha256") or "")):
                    raise ValueError(f"ready expert {family['id']} requires a lowercase SHA-256")
        expected = self.shared_parameters + sum(int(item["parameter_count"]) for item in self.families)
        declared = int(self.data["parameter_accounting"]["catalog_parameters_planned"])
        if expected != declared:
            raise ValueError(f"parameter accounting mismatch: expected {expected}, declared {declared}")
        general_active = self.shared_parameters + int(self.by_id["general"]["parameter_count"])
        if general_active != self.active_parameters:
            raise ValueError("active parameter count must equal shared trunk plus one expert family")

    @property
    def planned_parameters(self) -> int:
        return self.shared_parameters + sum(int(item["parameter_count"]) for item in self.families)

    @property
    def realized_parameters(self) -> int:
        ready = [item for item in self.families if item.get("status") == "ready"]
        if not ready:
            return 0
        return self.shared_parameters + sum(int(item["parameter_count"]) for item in ready)

    @property
    def claim_safe(self) -> bool:
        ready = [item for item in self.families if item.get("status") == "ready"]
        hashes = [item.get("sha256") for item in ready]
        return (
            len(ready) == len(self.families)
            and len(set(hashes)) == len(hashes)
            and all(item.get("benchmark_uri") for item in ready)
            and self.realized_parameters > 2_800_000_000_000
        )


class KeywordRouter:
    def __init__(self, manifest: HydraManifest):
        self.manifest = manifest

    def route(self, prompt: str) -> RouteDecision:
        text = prompt.casefold()
        best_id = "general"
        best_score = 0
        for family in self.manifest.families:
            if family["id"] == "general" or family.get("status") == "disabled":
                continue
            score = 0
            for keyword in family.get("keywords", []):
                keyword_text = keyword.casefold().strip()
                if keyword_text and re.search(r"(?<!\w)" + re.escape(keyword_text) + r"(?!\w)", text, flags=re.UNICODE):
                    score += 1
            if score > best_score:
                best_id, best_score = family["id"], score
        selected = self.manifest.by_id[best_id]
        if selected.get("status") == "ready" and selected.get("backend_model"):
            return RouteDecision(best_id, best_id, best_score, None)
        reason = None if best_id == "general" else f"expert {best_id} is {selected.get('status', 'unavailable')}"
        return RouteDecision(best_id, "general", best_score, reason)


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 600,
        *,
        headers: dict[str, str] | None = None,
        model_override: str | None = None,
        health_path: str = "/health",
        name: str = "local",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.headers = dict(headers or {})
        self.model_override = model_override
        self.health_path = health_path
        self.name = name

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + self.health_path, headers=self.headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
                return {"ok": response.status == 200, "status": response.status, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": self.model_override or model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if extra:
            request_payload.update(extra)
        started = time.perf_counter()
        data = self._post_with_tool_choice_fallback(request_payload)
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise BackendFailure("backend response did not contain choices", transient=False)
        backend_meta = data.setdefault("hydra_backend", {})
        backend_meta["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        backend_meta["name"] = self.name
        backend_meta["model"] = self.model_override or model
        return data

    @staticmethod
    def _is_unsupported_tool_choice(error: BackendFailure, request_payload: dict[str, Any]) -> bool:
        if error.status_code != 400:
            return False
        tool_choice = request_payload.get("tool_choice")
        if tool_choice in (None, "auto", "none"):
            return False
        return "tool_choice" in str(error).lower()

    def _post_with_tool_choice_fallback(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._post_chat_completion(request_payload)
        except BackendFailure as error:
            if not self._is_unsupported_tool_choice(error, request_payload):
                raise
            # Reasoning ("thinking") backends such as DeepSeek reject a forced
            # tool_choice. Downgrade to "auto" once so tool calling still works
            # instead of failing the whole request.
            coerced = dict(request_payload)
            coerced["tool_choice"] = "auto"
            return self._post_chat_completion(coerced)

    def _post_chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(request_payload).encode("utf-8")
        request_headers = {**self.headers, "Content-Type": "application/json"}
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=payload,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status == 202:
                    raise BackendFailure("backend returned a pending response", transient=True, status_code=202)
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            transient = exc.code == 429 or 500 <= exc.code < 600
            raise BackendFailure(
                f"backend HTTP {exc.code}: {detail}",
                transient=transient,
                status_code=exc.code,
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            raise BackendFailure("backend transport failure", transient=True) from exc


class CascadingBackend:
    """Prefer a fast remote inference backend and fail over to the local model."""

    def __init__(
        self,
        primary: ChatBackend,
        fallback: ChatBackend | None,
        *,
        primary_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        jitter_ratio: float = 0.25,
        sleep_fn: Callable[[float], None] = time.sleep,
        jitter_fn: Callable[[float, float], float] = random.uniform,
    ):
        if primary_retries < 0:
            raise ValueError("primary retries must be non-negative")
        if backoff_base_seconds < 0:
            raise ValueError("backoff base seconds must be non-negative")
        if jitter_ratio < 0:
            raise ValueError("jitter ratio must be non-negative")
        self.primary = primary
        self.fallback = fallback
        self.primary_retries = primary_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.jitter_ratio = jitter_ratio
        self.sleep_fn = sleep_fn
        self.jitter_fn = jitter_fn

    @staticmethod
    def _retryable_primary_failure(error: BackendFailure) -> bool:
        status = error.status_code
        return bool(error.transient and (status is None or status == 429 or 500 <= status < 600))

    @staticmethod
    def _fallback_eligible_primary_failure(error: BackendFailure) -> bool:
        status = error.status_code
        # Persistent rate limiting is infrastructure pressure, not a reason to
        # silently downgrade the request to a weaker local model.
        return bool(error.transient and (status is None or 500 <= status < 600))

    def _sleep_before_retry(self, retry_index: int) -> None:
        base_delay = self.backoff_base_seconds * (2 ** retry_index)
        jitter = self.jitter_fn(0.0, base_delay * self.jitter_ratio) if self.jitter_ratio else 0.0
        self.sleep_fn(base_delay + jitter)

    def health(self) -> dict[str, Any]:
        primary = self.primary.health()
        fallback = self.fallback.health() if self.fallback is not None else {"ok": False, "disabled": True}
        return {
            "ok": bool(primary.get("ok") or fallback.get("ok")),
            "mode": "primary-only" if self.fallback is None else "cascade",
            "primary": primary,
            "fallback": fallback,
        }

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        primary_error: BackendFailure | None = None
        attempts = 0
        for attempt in range(self.primary_retries + 1):
            attempts = attempt + 1
            try:
                result = self.primary.chat(model, messages, max_tokens, temperature, extra)
                backend_meta = result.setdefault("hydra_backend", {})
                backend_meta["fallback_used"] = False
                backend_meta["primary_attempts"] = attempts
                backend_meta["primary_retries"] = attempt
                return result
            except BackendFailure as error:
                primary_error = error
                if not self._retryable_primary_failure(error) or attempt >= self.primary_retries:
                    break
                self._sleep_before_retry(attempt)

        assert primary_error is not None
        if self.fallback is None or not self._fallback_eligible_primary_failure(primary_error):
            raise primary_error
        result = self.fallback.chat(model, messages, max_tokens, temperature, extra)
        backend_meta = result.setdefault("hydra_backend", {})
        backend_meta["fallback_used"] = True
        backend_meta["fallback_reason"] = "primary_persistent_failure"
        backend_meta["primary_attempts"] = attempts
        backend_meta["primary_retries"] = max(0, attempts - 1)
        if primary_error.status_code is not None:
            backend_meta["primary_status"] = primary_error.status_code
        return result


class HydraEngine:
    def __init__(self, manifest: HydraManifest, backend: ChatBackend):
        self.manifest = manifest
        self.backend = backend
        self.router = KeywordRouter(manifest)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.manifest.data["name"],
            "identity": dict(self.manifest.data["identity"]),
            "stage": self.manifest.data["stage"],
            "catalog_parameters_planned": self.manifest.planned_parameters,
            "catalog_parameters_realized": self.manifest.realized_parameters,
            "active_parameters_per_request": self.manifest.active_parameters,
            "expert_families_total": len(self.manifest.families),
            "expert_families_ready": sum(1 for item in self.manifest.families if item.get("status") == "ready"),
            "claim_safe_over_2_8t": self.manifest.claim_safe,
            "backend": self.backend.health(),
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if is_direct_identity_request(messages):
            result = deterministic_identity_response(messages)
            result["hydra"] = {
                "selected_expert": "control-plane",
                "executed_expert": "control-plane",
                "router_score": 0,
                "fallback_reason": None,
                "active_parameters": self.manifest.active_parameters,
                "catalog_parameters_planned": self.manifest.planned_parameters,
                "catalog_parameters_realized": self.manifest.realized_parameters,
                "claim_safe_over_2_8t": self.manifest.claim_safe,
            }
            return result
        prompt = "\n".join(str(item.get("content", "")) for item in messages if item.get("role") == "user")
        decision = self.router.route(prompt)
        family = self.manifest.by_id[decision.executed_expert]
        identity_request = contains_identity_request(messages)
        outbound_messages = branded_messages(messages, protect_identity=identity_request)
        if structured_output_requested(messages, extra):
            outbound_messages = apply_structured_output_instruction(outbound_messages)
        result = self.backend.chat(family["backend_model"], outbound_messages, max_tokens, temperature, extra)
        if identity_request:
            result = harden_mixed_identity_result(result, messages)
        else:
            result = strip_unsolicited_identity_prefix(result)
        result = harden_unsupported_action_claims(result, messages)
        result["hydra"] = {
            "selected_expert": decision.selected_expert,
            "executed_expert": decision.executed_expert,
            "router_score": decision.score,
            "fallback_reason": decision.fallback_reason,
            "active_parameters": self.manifest.active_parameters,
            "catalog_parameters_planned": self.manifest.planned_parameters,
            "catalog_parameters_realized": self.manifest.realized_parameters,
            "claim_safe_over_2_8t": self.manifest.claim_safe,
        }
        return result
