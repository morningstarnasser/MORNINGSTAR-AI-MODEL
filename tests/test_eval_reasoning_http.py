from __future__ import annotations

import unittest

from scripts.eval_reasoning import (
    answer_from_response,
    chat_payload,
)


class ChatPayloadTests(unittest.TestCase):
    def test_payload_is_deterministic_and_carries_the_budget(self) -> None:
        payload = chat_payload(
            [{"role": "user", "content": "2+2?"}],
            model="m",
            max_tokens=400,
            thinking_mode="default",
        )
        self.assertEqual("m", payload["model"])
        self.assertEqual(0, payload["temperature"])
        self.assertEqual(400, payload["max_tokens"])
        self.assertNotIn("chat_template_kwargs", payload)

    def test_thinking_mode_is_only_sent_when_explicitly_chosen(self) -> None:
        for modus, erwartet in (("on", True), ("off", False)):
            with self.subTest(modus=modus):
                payload = chat_payload(
                    [{"role": "user", "content": "2+2?"}],
                    model="m",
                    max_tokens=64,
                    thinking_mode=modus,
                )
                self.assertEqual({"enable_thinking": erwartet}, payload["chat_template_kwargs"])


class AnswerExtractionTests(unittest.TestCase):
    def test_plain_answer_is_returned(self) -> None:
        text, abgeschnitten = answer_from_response(
            {"choices": [{"message": {"content": "Die Antwort ist 18."}, "finish_reason": "stop"}]}
        )
        self.assertEqual("Die Antwort ist 18.", text)
        self.assertFalse(abgeschnitten)

    def test_thinking_without_an_answer_counts_as_truncated_not_wrong(self) -> None:
        # Das Budget ging vollstaendig in den Denkteil. Als falsche Antwort
        # gewertet wuerde das die Faehigkeit des Modells verschleiern.
        text, abgeschnitten = answer_from_response(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning": "Erst rechne ich 3 mal 6"},
                        "finish_reason": "length",
                    }
                ]
            }
        )
        self.assertEqual("", text)
        self.assertTrue(abgeschnitten)

    def test_empty_answer_without_thinking_is_a_normal_miss(self) -> None:
        text, abgeschnitten = answer_from_response(
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        )
        self.assertEqual("", text)
        self.assertFalse(abgeschnitten)

    def test_answer_survives_when_the_model_also_reports_its_thinking(self) -> None:
        text, abgeschnitten = answer_from_response(
            {
                "choices": [
                    {
                        "message": {"content": "18", "reasoning": "3 mal 6"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        self.assertEqual("18", text)
        self.assertFalse(abgeschnitten)

    def test_malformed_response_is_rejected_instead_of_scored(self) -> None:
        for kaputt in ({}, {"choices": []}, {"choices": [{"message": None}]}):
            with self.subTest(kaputt=kaputt):
                with self.assertRaises(ValueError):
                    answer_from_response(kaputt)


if __name__ == "__main__":
    unittest.main()
