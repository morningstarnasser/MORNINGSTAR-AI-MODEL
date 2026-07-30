from __future__ import annotations

import unittest

from morningstar_hydra.agent_runtime import (
    AgentLimits,
    AgentRuntimeError,
    ToolSpec,
    run_agent,
)


class AgentRuntimeTests(unittest.TestCase):
    def test_registered_tool_loop_and_thinking_contract(self) -> None:
        calls: list[dict] = []

        def model_call(payload: dict) -> dict:
            calls.append(payload)
            if len(calls) == 1:
                return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}}]}}]}
            return {"choices": [{"message": {"role": "assistant", "content": "5"}}]}

        spec = ToolSpec(
            name="add",
            description="Add two integers",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False},
            handler=lambda args: args["a"] + args["b"],
        )
        result = run_agent(
            model_call=model_call,
            messages=[{"role": "user", "content": "Add two and three."}],
            tools=[spec],
            thinking=True,
            limits=AgentLimits(max_steps=3, max_tool_calls=2),
        )
        self.assertEqual("5", result.content)
        self.assertEqual((2, 1), (result.steps, result.tool_calls))
        self.assertEqual({"enable_thinking": True}, calls[0]["chat_template_kwargs"])
        self.assertEqual("add", calls[0]["tools"][0]["function"]["name"])
        self.assertEqual("tool", calls[1]["messages"][-1]["role"])

    def test_unknown_tool_and_malformed_arguments_fail_closed(self) -> None:
        def unknown(_: dict) -> dict:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]}}]}

        def malformed(_: dict) -> dict:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "add", "arguments": "not-json"}}]}}]}

        spec = ToolSpec("add", "Add", {"type": "object"}, lambda args: 0)
        with self.assertRaisesRegex(AgentRuntimeError, "unregistered tool"):
            run_agent(model_call=unknown, messages=[{"role": "user", "content": "x"}], tools=[spec])
        with self.assertRaisesRegex(AgentRuntimeError, "arguments"):
            run_agent(model_call=malformed, messages=[{"role": "user", "content": "x"}], tools=[spec])

    def test_step_and_tool_budgets_are_enforced(self) -> None:
        def repeated(_: dict) -> dict:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "noop", "arguments": "{}"}}]}}]}

        spec = ToolSpec("noop", "No operation", {"type": "object"}, lambda args: None)
        with self.assertRaisesRegex(AgentRuntimeError, "tool-call budget"):
            run_agent(
                model_call=repeated,
                messages=[{"role": "user", "content": "loop"}],
                tools=[spec],
                limits=AgentLimits(max_steps=3, max_tool_calls=0),
            )
        with self.assertRaisesRegex(AgentRuntimeError, "step budget"):
            run_agent(
                model_call=repeated,
                messages=[{"role": "user", "content": "loop"}],
                tools=[spec],
                limits=AgentLimits(max_steps=1, max_tool_calls=2),
            )

    def test_empty_tools_are_omitted_and_nested_schema_is_enforced(self) -> None:
        seen: list[dict] = []

        def final(payload: dict) -> dict:
            seen.append(payload)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        run_agent(model_call=final, messages=[{"role": "user", "content": "x"}], tools=[])
        self.assertNotIn("tools", seen[0])
        self.assertNotIn("tool_choice", seen[0])

        nested = ToolSpec(
            "nested",
            "Nested input",
            {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["enabled"],
                        "additionalProperties": False,
                    }
                },
                "required": ["config"],
                "additionalProperties": False,
            },
            lambda args: args,
        )

        def invalid_nested(_: dict) -> dict:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "n1", "type": "function", "function": {"name": "nested", "arguments": "{\"config\":{\"enabled\":\"yes\"}}"}}]}}]}

        with self.assertRaisesRegex(AgentRuntimeError, "schema"):
            run_agent(model_call=invalid_nested, messages=[{"role": "user", "content": "x"}], tools=[nested])

    def test_duplicate_call_ids_handler_errors_and_aliasing_fail_safely(self) -> None:
        spec = ToolSpec("noop", "No operation", {"type": "object"}, lambda args: None)

        def duplicate(_: dict) -> dict:
            call = {"id": "same", "type": "function", "function": {"name": "noop", "arguments": "{}"}}
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [call, dict(call)]}}]}

        with self.assertRaisesRegex(AgentRuntimeError, "duplicate"):
            run_agent(model_call=duplicate, messages=[{"role": "user", "content": "x"}], tools=[spec])

        broken = ToolSpec("broken", "Fails", {"type": "object"}, lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

        def call_broken(_: dict) -> dict:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "b1", "type": "function", "function": {"name": "broken", "arguments": "{}"}}]}}]}

        with self.assertRaisesRegex(AgentRuntimeError, "handler failed"):
            run_agent(model_call=call_broken, messages=[{"role": "user", "content": "x"}], tools=[broken])


if __name__ == "__main__":
    unittest.main()
