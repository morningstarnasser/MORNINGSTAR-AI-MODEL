"""Bounded tool-calling runtime for OpenAI-compatible chat models."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


class AgentRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("invalid tool name")
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ValueError("tool parameters must be an object JSON schema")

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(self.parameters),
            },
        }


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 8
    max_tool_calls: int = 12
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.max_tool_calls < 0 or self.max_tokens <= 0:
            raise ValueError("agent limits must be positive (tool calls may be zero)")


@dataclass(frozen=True)
class AgentResult:
    content: str
    steps: int
    tool_calls: int
    messages: tuple[dict[str, Any], ...]


def _validate_value(rule: dict[str, Any], value: Any, *, depth: int = 0) -> None:
    if depth > 20:
        raise AgentRuntimeError("tool arguments schema nesting is too deep")
    rule_type = rule.get("type")
    python_types: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    expected = python_types.get(rule_type) if isinstance(rule_type, str) else None
    if expected is not None and (
        (isinstance(value, bool) and rule_type in {"integer", "number"})
        or not isinstance(value, expected)
    ):
        raise AgentRuntimeError("tool arguments failed schema type validation")
    if "enum" in rule and value not in rule["enum"]:
        raise AgentRuntimeError("tool arguments failed schema enum validation")

    if rule_type == "object":
        if not isinstance(value, dict):
            raise AgentRuntimeError("tool arguments failed schema type validation")
        required = rule.get("required", [])
        properties = rule.get("properties", {})
        if (
            not isinstance(required, list)
            or any(not isinstance(key, str) for key in required)
            or not isinstance(properties, dict)
        ):
            raise AgentRuntimeError("invalid registered tool schema")
        missing = [key for key in required if key not in value]
        if missing:
            raise AgentRuntimeError("tool arguments missing required schema properties")
        if rule.get("additionalProperties") is False:
            if set(value) - set(properties):
                raise AgentRuntimeError("tool arguments contain additional schema properties")
        for key, child in value.items():
            child_rule = properties.get(key)
            if isinstance(child_rule, dict):
                _validate_value(child_rule, child, depth=depth + 1)
    elif rule_type == "array":
        if not isinstance(value, list):
            raise AgentRuntimeError("tool arguments failed schema type validation")
        item_rule = rule.get("items")
        if item_rule is not None and not isinstance(item_rule, dict):
            raise AgentRuntimeError("invalid registered tool schema")
        if isinstance(item_rule, dict):
            for item in value:
                _validate_value(item_rule, item, depth=depth + 1)


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    _validate_value(schema, arguments)


def _assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentRuntimeError("model response has no assistant message") from exc
    if not isinstance(message, dict):
        raise AgentRuntimeError("model assistant message must be an object")
    return message


def run_agent(
    *,
    model_call: Callable[[dict[str, Any]], dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: list[ToolSpec],
    thinking: bool | None = None,
    limits: AgentLimits | None = None,
    model: str = "hydra-dual-backend",
) -> AgentResult:
    """Run a bounded model/tool loop with no implicit tool authority."""
    limits = limits or AgentLimits()
    if not messages or any(not isinstance(item, dict) or not isinstance(item.get("role"), str) for item in messages):
        raise ValueError("messages must be a non-empty list of role objects")
    registry = {tool.name: tool for tool in tools}
    if len(registry) != len(tools):
        raise ValueError("tool names must be unique")
    transcript = copy.deepcopy(messages)
    used_tool_calls = 0
    seen_call_ids: set[str] = set()

    for step in range(1, limits.max_steps + 1):
        payload: dict[str, Any] = {
            "model": model,
            "messages": copy.deepcopy(transcript),
            "temperature": 0,
            "max_tokens": limits.max_tokens,
        }
        if tools:
            payload["tools"] = [tool.openai_schema() for tool in tools]
            payload["tool_choice"] = "auto"
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        message = _assistant_message(model_call(payload))
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise AgentRuntimeError("assistant tool_calls must be a list")
        if not raw_calls:
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise AgentRuntimeError("assistant content must be text")
            transcript.append({"role": "assistant", "content": content})
            return AgentResult(
                content,
                step,
                used_tool_calls,
                tuple(copy.deepcopy(transcript)),
            )
        batch_ids: list[str] = []
        for call in raw_calls:
            if (
                not isinstance(call, dict)
                or call.get("type") != "function"
                or not isinstance(call.get("id"), str)
                or not call["id"]
            ):
                raise AgentRuntimeError("malformed model tool call")
            batch_ids.append(call["id"])
        if len(set(batch_ids)) != len(batch_ids) or seen_call_ids.intersection(batch_ids):
            raise AgentRuntimeError("duplicate model tool-call id")
        if used_tool_calls + len(raw_calls) > limits.max_tool_calls:
            raise AgentRuntimeError("agent tool-call budget exceeded")
        seen_call_ids.update(batch_ids)

        transcript.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": copy.deepcopy(raw_calls),
            }
        )
        for call in raw_calls:
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise AgentRuntimeError("malformed model function call")
            name = function["name"]
            spec = registry.get(name)
            if spec is None:
                raise AgentRuntimeError(f"unregistered tool requested: {name}")
            raw_arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError as exc:
                raise AgentRuntimeError("tool arguments are not valid JSON") from exc
            if not isinstance(arguments, dict):
                raise AgentRuntimeError("tool arguments must decode to an object")
            _validate_arguments(spec.parameters, arguments)
            try:
                result = spec.handler(copy.deepcopy(arguments))
            except Exception as exc:
                raise AgentRuntimeError(f"tool handler failed: {name}") from exc
            try:
                tool_result = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise AgentRuntimeError("tool result is not JSON serializable") from exc
            transcript.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": tool_result})
            used_tool_calls += 1

    raise AgentRuntimeError("agent step budget exceeded")
