from __future__ import annotations

import json
import unittest

from scripts.ollama_openai_proxy import (
    stream_chunk,
    to_ollama,
    to_openai,
)

WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "weather_lookup",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


class RequestTranslationTests(unittest.TestCase):
    def test_openai_fields_map_onto_ollama_options(self) -> None:
        anfrage = to_ollama(
            {
                "messages": [{"role": "user", "content": "hallo"}],
                "temperature": 0,
                "max_tokens": 128,
                "top_p": 0.9,
                "seed": 7,
            },
            model="m",
            think=False,
            stream=False,
        )
        self.assertEqual("m", anfrage["model"])
        self.assertIs(False, anfrage["think"])
        self.assertIs(False, anfrage["stream"])
        self.assertEqual(
            {"temperature": 0.0, "num_predict": 128, "top_p": 0.9, "seed": 7},
            anfrage["options"],
        )

    def test_tool_history_is_translated_back_into_ollamas_shape(self) -> None:
        # Ollama lehnt einen Verlauf ab, dessen Argumente als String ankommen —
        # genau umgekehrt zur Antwortrichtung.
        anfrage = to_ollama(
            {
                "messages": [
                    {"role": "user", "content": "Wetter in Basel?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "weather_lookup",
                                    "arguments": '{"city": "Basel"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "weather_lookup",
                        "content": '{"celsius": 21}',
                    },
                ]
            },
            model="m",
            think=False,
            stream=False,
        )
        assistant, werkzeug = anfrage["messages"][1], anfrage["messages"][2]
        self.assertEqual({"city": "Basel"}, assistant["tool_calls"][0]["function"]["arguments"])
        self.assertNotIn("id", assistant["tool_calls"][0])
        self.assertEqual("weather_lookup", werkzeug["tool_name"])
        self.assertNotIn("tool_call_id", werkzeug)
        self.assertEqual('{"celsius": 21}', werkzeug["content"])

    def test_messages_without_tools_are_passed_through_unchanged(self) -> None:
        original = [{"role": "user", "content": "hallo"}]
        anfrage = to_ollama({"messages": original}, model="m", think=False, stream=False)
        self.assertEqual(original, anfrage["messages"])

    def test_tools_are_forwarded_and_thinking_is_configurable(self) -> None:
        anfrage = to_ollama(
            {"messages": [{"role": "user", "content": "Wetter?"}], "tools": WEATHER},
            model="m",
            think=True,
            stream=True,
        )
        self.assertEqual(WEATHER, anfrage["tools"])
        self.assertIs(True, anfrage["think"])
        self.assertIs(True, anfrage["stream"])


class ResponseTranslationTests(unittest.TestCase):
    def test_plain_answer_becomes_an_openai_completion(self) -> None:
        antwort = to_openai(
            {"message": {"role": "assistant", "content": "ZH"}, "eval_count": 2},
            "m",
        )
        nachricht = antwort["choices"][0]["message"]
        self.assertEqual("ZH", nachricht["content"])
        self.assertEqual("stop", antwort["choices"][0]["finish_reason"])
        self.assertEqual(2, antwort["usage"]["completion_tokens"])

    def test_tool_arguments_are_serialised_as_a_string(self) -> None:
        # Ollama liefert die Argumente als Objekt, OpenAI erwartet einen String.
        antwort = to_openai(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "weather_lookup", "arguments": {"city": "Basel"}}}
                    ],
                }
            },
            "m",
        )
        aufruf = antwort["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual("weather_lookup", aufruf["function"]["name"])
        self.assertEqual({"city": "Basel"}, json.loads(aufruf["function"]["arguments"]))
        self.assertIsNone(antwort["choices"][0]["message"]["content"])
        self.assertEqual("tool_calls", antwort["choices"][0]["finish_reason"])

    def test_thinking_is_reported_separately_from_the_answer(self) -> None:
        antwort = to_openai(
            {"message": {"role": "assistant", "content": "ZH", "thinking": "kurz"}},
            "m",
        )
        self.assertEqual("kurz", antwort["choices"][0]["message"]["reasoning"])
        self.assertEqual("ZH", antwort["choices"][0]["message"]["content"])


class StreamTranslationTests(unittest.TestCase):
    def test_content_becomes_a_delta_event(self) -> None:
        ereignis = stream_chunk({"message": {"content": "Gr"}, "done": False}, "id", "m")
        self.assertTrue(ereignis.startswith(b"data: "))
        geladen = json.loads(ereignis[len("data: "):].decode())
        self.assertEqual("chat.completion.chunk", geladen["object"])
        self.assertEqual("Gr", geladen["choices"][0]["delta"]["content"])
        self.assertIsNone(geladen["choices"][0]["finish_reason"])

    def test_final_event_carries_the_finish_reason(self) -> None:
        ereignis = stream_chunk({"message": {"content": ""}, "done": True}, "id", "m")
        geladen = json.loads(ereignis[len("data: "):].decode())
        self.assertEqual("stop", geladen["choices"][0]["finish_reason"])

    def test_tool_call_in_a_stream_keeps_the_openai_shape(self) -> None:
        ereignis = stream_chunk(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "weather_lookup", "arguments": {"city": "Bern"}}}
                    ],
                },
                "done": True,
            },
            "id",
            "m",
        )
        geladen = json.loads(ereignis[len("data: "):].decode())
        aufruf = geladen["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(0, aufruf["index"])
        self.assertEqual({"city": "Bern"}, json.loads(aufruf["function"]["arguments"]))
        self.assertEqual("tool_calls", geladen["choices"][0]["finish_reason"])

    def test_thinking_never_leaks_into_the_content_delta(self) -> None:
        ereignis = stream_chunk(
            {"message": {"content": "", "thinking": "ueberlege"}, "done": False}, "id", "m"
        )
        geladen = json.loads(ereignis[len("data: "):].decode())
        delta = geladen["choices"][0]["delta"]
        self.assertEqual("ueberlege", delta.get("reasoning"))
        self.assertNotIn("ueberlege", delta.get("content") or "")


if __name__ == "__main__":
    unittest.main()
