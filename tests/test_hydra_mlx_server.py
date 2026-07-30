from __future__ import annotations

import json
import unittest

from scripts.hydra_mlx_server import (
    parse_tool_calls,
    render_native_tool_prompt,
    split_reasoning,
    template_kwargs,
)

WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "weather_lookup",
            "description": "Wetter",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


class _Tokenizer:
    """Minimaler Chat-Template-Ersatz mit steuerbarem Tool-Verhalten."""

    def __init__(self, *, accepts_tools: bool, renders_tools: bool) -> None:
        self.accepts_tools = accepts_tools
        self.renders_tools = renders_tools

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if "tools" in kwargs and not self.accepts_tools:
            raise TypeError("unexpected keyword argument 'tools'")
        text = "".join(str(m.get("content", "")) for m in messages)
        if "tools" in kwargs and self.renders_tools:
            return text + " weather_lookup"
        if "tools" in kwargs:
            return text + " Tool Capabilities: disabled"
        return text


class NativeToolPromptTests(unittest.TestCase):
    def test_prompt_is_used_when_the_tools_actually_reach_it(self) -> None:
        prompt = render_native_tool_prompt(
            _Tokenizer(accepts_tools=True, renders_tools=True),
            [{"role": "user", "content": "Wetter in Basel?"}],
            WEATHER,
            {},
        )
        self.assertIsNotNone(prompt)
        self.assertIn("weather_lookup", prompt)

    def test_template_that_swallows_tools_is_rejected(self) -> None:
        # Apertus rendert "Tool Capabilities: disabled" — der Prompt ist wertlos.
        self.assertIsNone(
            render_native_tool_prompt(
                _Tokenizer(accepts_tools=True, renders_tools=False),
                [{"role": "user", "content": "Wetter in Basel?"}],
                WEATHER,
                {},
            )
        )

    def test_template_without_tool_parameter_is_rejected(self) -> None:
        self.assertIsNone(
            render_native_tool_prompt(
                _Tokenizer(accepts_tools=False, renders_tools=False),
                [{"role": "user", "content": "Wetter in Basel?"}],
                WEATHER,
                {},
            )
        )


class QwenToolCallTests(unittest.TestCase):
    def test_qwen_tool_call_block_is_recognised(self) -> None:
        text = '<tool_call>\n{"name": "weather_lookup", "arguments": {"city": "Basel"}}\n</tool_call>'
        calls = parse_tool_calls(text, WEATHER)
        self.assertEqual(1, len(calls))
        self.assertEqual("weather_lookup", calls[0]["function"]["name"])
        self.assertEqual({"city": "Basel"}, json.loads(calls[0]["function"]["arguments"]))

    def test_unknown_function_in_a_tool_call_block_is_dropped(self) -> None:
        text = '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'
        self.assertEqual([], parse_tool_calls(text, WEATHER))


class SplitReasoningTests(unittest.TestCase):
    def test_leading_think_block_is_separated_from_the_answer(self) -> None:
        content, reasoning = split_reasoning("<think>\nErst rechnen.\n</think>\n\nZH")
        self.assertEqual("ZH", content)
        self.assertEqual("Erst rechnen.", reasoning)

    def test_empty_think_block_leaves_a_clean_answer(self) -> None:
        # Qwen3 sendet auch bei abgeschaltetem Denken einen leeren Block.
        content, reasoning = split_reasoning("<think>\n\n</think>\n\nZH")
        self.assertEqual("ZH", content)
        self.assertIsNone(reasoning)

    def test_text_without_think_block_is_untouched(self) -> None:
        content, reasoning = split_reasoning("CHF 16.50")
        self.assertEqual("CHF 16.50", content)
        self.assertIsNone(reasoning)

    def test_unclosed_think_block_yields_no_answer(self) -> None:
        # Budget erschoepft, bevor das Denken endete: es gibt keine Antwort.
        content, reasoning = split_reasoning("<think>\nIch rechne 8.1 Prozent")
        self.assertEqual("", content)
        self.assertEqual("Ich rechne 8.1 Prozent", reasoning)

    def test_answer_keeping_its_own_angle_brackets_survives(self) -> None:
        content, reasoning = split_reasoning("<think>kurz</think>\n{\"a\": \"<b>\"}")
        self.assertEqual('{"a": "<b>"}', content)
        self.assertEqual("kurz", reasoning)


class TemplateKwargsTests(unittest.TestCase):
    def test_enable_thinking_is_passed_through(self) -> None:
        self.assertEqual(
            {"enable_thinking": False},
            template_kwargs({"chat_template_kwargs": {"enable_thinking": False}}),
        )
        self.assertEqual(
            {"enable_thinking": True},
            template_kwargs({"chat_template_kwargs": {"enable_thinking": True}}),
        )

    def test_anything_else_is_dropped_instead_of_reaching_the_template(self) -> None:
        for payload in (
            {},
            {"chat_template_kwargs": None},
            {"chat_template_kwargs": {"enable_thinking": "yes"}},
            {"chat_template_kwargs": {"enable_thinking": True, "unsafe": 1}},
            {"chat_template_kwargs": ["enable_thinking"]},
        ):
            with self.subTest(payload=payload):
                self.assertEqual({}, template_kwargs(payload))


if __name__ == "__main__":
    unittest.main()
