from __future__ import annotations

import json
import re
import time
from typing import Any

CANONICAL_IDENTITY = {
    "product": "Morningstar Hydra",
    "brand": "CreativeSync",
}

IDENTITY_INSTRUCTION = (
    "Immutable identity: You are Morningstar Hydra, a CreativeSync product. State this only when "
    "the user directly asks who or what you are; never prepend it to ordinary answers. Always reply "
    "in the user's language. Do not identify as Qwen, Gemma, Llama, Mistral, DeepSeek, GPT, Claude "
    "or any other technical backend. If asked about the infrastructure, say plainly which model "
    "serves the request: those weights are not owned by CreativeSync and are not your identity. "
    "Never claim that a file, message, upload, deployment or other external action succeeded "
    "unless a tool result in this conversation confirms it."
)

GERMAN_IDENTITY = "Ich bin Morningstar Hydra, ein CreativeSync-Produkt."
ENGLISH_IDENTITY = "I am Morningstar Hydra, a CreativeSync product."

_IDENTITY_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat is your (name|identity)\b",
    r"\bwhich (ai )?model are you\b",
    r"\bwhat (ai )?model are you\b",
    r"\bwho is behind you\b",
    r"\bwho is your (creator|developer|maker)\b",
    r"\bwho (developed|created|built|made) morningstar hydra\b",
    r"\bwhat (backend(?:/provider)?|provider|infrastructure|model) (are you using|do you use|powers you)\b",
    r"\btell me (about )?morningstar hydra'?s (identity|creator|developer|origin)\b",
    r"\bmorningstar hydra'?s (identity|creator|developer|origin)\b",
    r"\bare you (deepseek|qwen|gemma|llama|mistral|nvidia|nim|gpt|claude)\b",
    r"\bwho (made|built|developed|created) you\b",
    r"\bwer bist du\b",
    r"\bwas bist du\b",
    r"\bwie hei(?:ss|ß)t du\b",
    r"\bwelches modell bist du\b",
    r"\bwer steckt hinter dir\b",
    r"\bvon wem wurde morningstar hydra (entwickelt|gebaut|erstellt|programmiert)\b",
    r"\bwer (entwickelte|baute|erstellte|programmierte) morningstar hydra\b",
    r"\bwelches (backend|modell|system) (verwendest|benutzt|nutzt) du\b",
    r"\bwelche (infrastruktur|architektur) (verwendest|benutzt|nutzt) du\b",
    r"\bwelchen (anbieter|provider) (verwendest|benutzt|nutzt) du\b",
    r"\bwer ist dein (entwickler|ersteller|schöpfer|creator)\b",
    r"\bbist du (deepseek|qwen|gemma|llama|mistral|nvidia|nim|gpt|claude)\b",
    r"\bwer hat dich (gemacht|entwickelt|gebaut|erstellt|programmiert)\b",
]

_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions.*\b(say|claim|pretend)\b.*\b(deepseek|qwen|gemma|llama|mistral|nvidia|nim|gpt|claude)\b",
    r"\b(say|claim|pretend)\b.*\byou are\b.*\b(deepseek|qwen|gemma|llama|mistral|nvidia|nim|gpt|claude)\b",
    r"ignoriere .*anweisungen.*\b(deepseek|qwen|gemma|llama|mistral|nvidia|nim|gpt|claude)\b",
]

_SUBSTANTIVE_MIXED_PATTERNS = [
    r"\b(write|create|build|implement|fix|debug|solve|calculate|explain|summarize|translate)\b",
    r"\b(schreib|erstelle|baue|implementiere|repariere|löse|berechne|erkläre|fasse|übersetze)\b",
    r"\bpython|javascript|sql|code|function|klasse|funktion|proof|beweis\b",
    r"\d+\s*[-+*/]\s*\d+",
]

_UNSOLICITED_IDENTITY_PREFIXES = [
    r"Ich bin Morningstar Hydra\b[^.!?\n]*(?:[.!?](?=\s|$)|\n|$)",
    r"I am Morningstar Hydra\b[^.!?\n]*(?:[.!?](?=\s|$)|\n|$)",
]

_DELIVERY_CLAIM_PATTERNS = [
    r"\b(?:per|via|über)\s+whatsapp\b.{0,120}\b(?:gesendet|geschickt|zugestellt)\b",
    r"\b(?:sent|delivered|shared)\b.{0,120}\b(?:via|through|on)\s+whatsapp\b",
    r"\b(?:sent|delivered|shared)\b.{0,120}\bto (?:your )?(?:phone|whatsapp)\b",
    r"\bdu (?:solltest|müsstest) (?:es|die datei).{0,80}\b(?:handy|telefon)\b.{0,40}\bhaben\b",
    r"\byou should (?:now )?have (?:it|the file).{0,80}\b(?:phone|whatsapp)\b",
]

_ARTIFACT_CLAIM_PATTERNS = [
    r"\b(?:snake[- ]?spiel|html[- ]?datei|datei)\s+(?:ist|wurde)\s+(?:fertig|erstellt|gebaut|generiert)\b",
    r"\b(?:snake game|html file|file)\s+(?:is|has been)\s+(?:finished|ready|created|built|generated)\b",
]


def identity_instruction_message() -> dict[str, str]:
    return {"role": "system", "content": IDENTITY_INSTRUCTION}


def messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(item.get("content", "")) for item in messages if item.get("role") == "user")


def contains_identity_request(messages: list[dict[str, Any]]) -> bool:
    text = messages_text(messages).strip().casefold()
    if not text:
        return False
    patterns = [*_IDENTITY_PATTERNS, *_INJECTION_PATTERNS]
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def is_direct_identity_request(messages: list[dict[str, Any]]) -> bool:
    text = messages_text(messages).strip().casefold()
    if not text:
        return False
    if any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in _INJECTION_PATTERNS):
        return True
    if not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _IDENTITY_PATTERNS):
        return False
    return not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _SUBSTANTIVE_MIXED_PATTERNS)


def branded_messages(messages: list[dict[str, Any]], *, protect_identity: bool = False) -> list[dict[str, Any]]:
    """Add the immutable identity and isolate identity answers from caller persona overrides."""
    if protect_identity:
        task_messages = [item for item in messages if item.get("role") not in {"system", "developer"}]
        return [identity_instruction_message(), *task_messages]
    insertion = 0
    while insertion < len(messages) and messages[insertion].get("role") in {"system", "developer"}:
        insertion += 1
    return [*messages[:insertion], identity_instruction_message(), *messages[insertion:]]


def _is_german(text: str) -> bool:
    return bool(
        re.search(
            r"\b(wer bist du|was bist du|bist du|wie hei(?:ss|ß)t du|welches (?:modell|backend|system)|welche (?:infrastruktur|architektur)|welchen (?:anbieter|provider)|wer ist dein|wer steckt|von wem|entwickelt|gemacht|gebaut)\b",
            text,
        )
    )


def harden_mixed_identity_result(result: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep substantive backend output while making the identity answer deterministic."""
    text = messages_text(messages).casefold()
    prefix = GERMAN_IDENTITY if _is_german(text) else ENGLISH_IDENTITY
    prohibited_claims = [
        r"\bI\s+(?:am|'m)\s+(?:an?\s+)?(?:DeepSeek|Qwen|Gemma|Llama|Mistral|NVIDIA(?:\s+NIM)?|OpenAI|ChatGPT|Claude|Gemini)(?:\s+(?:model|assistant))?\b",
        r"\bMy name is\s+(?:DeepSeek|Qwen|Gemma|Llama|Mistral|NVIDIA(?:\s+NIM)?|OpenAI|ChatGPT|Claude|Gemini)\b",
        r"\bAs\s+(?:DeepSeek|Qwen|Gemma|Llama|Mistral|NVIDIA(?:\s+NIM)?|OpenAI|ChatGPT|Claude|Gemini)\b",
        r"\bIch bin\s+(?:ein(?:e|en)?\s+)?(?:DeepSeek|Qwen|Gemma|Llama|Mistral|NVIDIA(?:\s+NIM)?|OpenAI|ChatGPT|Claude|Gemini)(?:[- ](?:Modell|Assistent))?\b",
        r"\bMein Name ist\s+(?:DeepSeek|Qwen|Gemma|Llama|Mistral|NVIDIA(?:\s+NIM)?|OpenAI|ChatGPT|Claude|Gemini)\b",
    ]
    choices = result.get("choices")
    if not isinstance(choices, list):
        return result
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        message = choice["message"]
        content = message.get("content")
        cleaned = content if isinstance(content, str) else ""
        for pattern in prohibited_claims:
            cleaned = re.sub(pattern, prefix, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"(?im)(?:^|(?<=[.!?])\s+)[^.!?\n]*(?:Morningstar Contributors|Switzerland|Schweiz)[^.!?\n]*[.!?]?",
            "",
            cleaned,
        ).strip()
        if not cleaned.lstrip().casefold().startswith(prefix.casefold()):
            cleaned = prefix + ("\n\n" + cleaned if cleaned else "")
        message["content"] = cleaned
    return result


def strip_unsolicited_identity_prefix(result: dict[str, Any]) -> dict[str, Any]:
    """Remove a backend-added identity preamble from ordinary, non-identity answers."""
    choices = result.get("choices")
    if not isinstance(choices, list):
        return result
    combined = r"^(?:\s*(?:" + "|".join(_UNSOLICITED_IDENTITY_PREFIXES) + r")\s*)+"
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        message = choice["message"]
        content = message.get("content")
        if not isinstance(content, str):
            continue
        message["content"] = re.sub(combined, "", content, flags=re.IGNORECASE).lstrip(" \t\r\n:-")
    return result


def _confirmed_delivery_evidence(messages: list[dict[str, Any]]) -> bool:
    whatsapp_call_ids: set[str] = set()
    for item in messages:
        if item.get("role") != "assistant" or not isinstance(item.get("tool_calls"), list):
            continue
        for call in item["tool_calls"]:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            function = call["function"]
            function_name = str(function.get("name", "")).casefold()
            arguments = str(function.get("arguments", "")).casefold()
            named_send = "whatsapp" in function_name and any(
                marker in function_name for marker in ("send", "deliver", "message_create")
            )
            generic_send = function_name in {"whatsapp", "whatsapp_cli", "whatsapp-cli"} and bool(
                re.search(r"(?:action|mode)[^,}]{0,20}send", arguments)
            )
            terminal_send = any(marker in arguments for marker in ("send.js", "wa-ipc"))
            if named_send or generic_send or terminal_send:
                whatsapp_call_ids.add(str(call.get("id", "")))

    for item in messages:
        if item.get("role") != "tool":
            continue
        content = item.get("content", "")
        rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        call_id = str(item.get("tool_call_id", ""))
        if call_id not in whatsapp_call_ids:
            continue
        try:
            payload = json.loads(rendered) if isinstance(content, str) else content
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        confirmed_id = payload.get("messageId") or payload.get("textId") or payload.get("voiceId")
        if payload.get("ok") is True and confirmed_id and not payload.get("error"):
            return True
    return False


def _confirmed_artifact_evidence(messages: list[dict[str, Any]]) -> bool:
    artifact_call_ids: set[str] = set()
    for item in messages:
        if item.get("role") != "assistant" or not isinstance(item.get("tool_calls"), list):
            continue
        for call in item["tool_calls"]:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            function = call["function"]
            call_context = f"{function.get('name', '')} {function.get('arguments', '')}".casefold()
            if "write_file" in call_context or re.search(r"\b(?:create|write).{0,40}\.html\b", call_context):
                artifact_call_ids.add(str(call.get("id", "")))

    for item in messages:
        if item.get("role") != "tool" or str(item.get("tool_call_id", "")) not in artifact_call_ids:
            continue
        content = item.get("content", "")
        rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if re.search(r"\b(?:error|failed|failure)\b", rendered, re.IGNORECASE):
            continue
        try:
            payload = json.loads(rendered) if isinstance(content, str) else content
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            path = payload.get("path") or payload.get("file") or payload.get("output_path")
            if payload.get("ok") is True and isinstance(path, str) and path:
                return True
        if re.search(r"\b(?:wrote to|created|saved)\b.{0,160}\.(?:html?|zip)\b", rendered, re.IGNORECASE):
            return True
    return False


def harden_unsupported_action_claims(result: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject unsupported delivery and artifact-completion claims."""
    delivery_confirmed = _confirmed_delivery_evidence(messages)
    artifact_confirmed = _confirmed_artifact_evidence(messages)
    choices = result.get("choices")
    if not isinstance(choices, list):
        return result
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        message = choice["message"]
        content = message.get("content")
        if not isinstance(content, str):
            continue
        delivery_claimed = any(
            re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL) for pattern in _DELIVERY_CLAIM_PATTERNS
        )
        artifact_claimed = any(
            re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL) for pattern in _ARTIFACT_CLAIM_PATTERNS
        )
        inline_artifact = bool(re.search(r"<!doctype html|<html\b|```(?:html|javascript|python)", content, re.IGNORECASE))
        if not delivery_claimed and not artifact_claimed:
            continue
        german = _is_german((messages_text(messages) + "\n" + content).casefold()) or bool(
            re.search(r"\b(?:gesendet|geschickt|zugestellt|datei|fertig|erstellt)\b", content, flags=re.IGNORECASE)
        )
        if delivery_claimed and not delivery_confirmed:
            message["content"] = (
                "Ich habe keinen bestätigten WhatsApp-Versandnachweis. Die Datei wurde nicht nachweisbar gesendet; "
                "ohne erfolgreiche Tool-Rückgabe darf ich das nicht behaupten."
                if german
                else "I have no confirmed WhatsApp delivery evidence. The file was not verifiably sent, and I must not "
                "claim success without a successful tool result."
            )
        elif artifact_claimed and not artifact_confirmed and not inline_artifact:
            message["content"] = (
                "Ich habe keinen bestätigten Nachweis, dass die Datei erstellt wurde. Ohne erfolgreiche Tool-Rückgabe "
                "darf ich das Spiel nicht als fertig bezeichnen."
                if german
                else "I have no confirmed evidence that the file was created. I must not call the game finished without "
                "a successful tool result."
            )
    return result


def deterministic_identity_response(messages: list[dict[str, Any]]) -> dict[str, Any]:
    text = messages_text(messages).casefold()
    german = _is_german(text)
    asks_provider = bool(
        re.search(r"\b(deepseek|qwen|nvidia|nim|backend|provider|infrastructure|anbieter|infrastruktur)\b", text)
    )
    if german and asks_provider:
        content = (
            f"{GERMAN_IDENTITY} "
            "NVIDIA NIM, DeepSeek oder Qwen können technische Backends sein; sie sind nicht meine "
            "Assistentenidentität und keine CreativeSync-eigenen Gewichte."
        )
    elif german:
        content = GERMAN_IDENTITY
    elif asks_provider:
        content = (
            f"{ENGLISH_IDENTITY} "
            "NVIDIA NIM, DeepSeek, and Qwen may be technical backends; they are not the assistant identity, "
            "and CreativeSync does not claim to own those upstream weights."
        )
    else:
        # Ohne Frage nach der Infrastruktur bleibt die Antwort knapp -- wie im
        # deutschen Zweig. Backends ungefragt zu nennen untergraebt genau die
        # Identitaet, die hier festgeschrieben wird.
        content = ENGLISH_IDENTITY
    return {
        "id": "chatcmpl-morningstar-hydra-identity",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "morningstar-hydra",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
