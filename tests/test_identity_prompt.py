from __future__ import annotations

import unittest

from morningstar_hydra.identity import (
    IDENTITY_INSTRUCTION,
    deterministic_identity_response,
    is_direct_identity_request,
)


def _frage(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


class InstructionContentTests(unittest.TestCase):
    def test_instruction_names_the_product_and_stays_short(self) -> None:
        self.assertIn("Morningstar Hydra", IDENTITY_INSTRUCTION)
        self.assertIn("CreativeSync", IDENTITY_INSTRUCTION)
        # Die Anweisung geht an jede einzelne Anfrage; Laenge kostet bei jedem Aufruf.
        self.assertLess(len(IDENTITY_INSTRUCTION), 700)

    def test_instruction_covers_the_models_actually_in_use(self) -> None:
        gesenkt = IDENTITY_INSTRUCTION.casefold()
        for modell in ("qwen", "gemma", "llama"):
            with self.subTest(modell=modell):
                self.assertIn(modell, gesenkt)

    def test_instruction_no_longer_claims_a_retired_cloud_backend(self) -> None:
        # NVIDIA NIM war die alte Cloud-Architektur und ist nicht mehr im Spiel.
        self.assertNotIn("NIM", IDENTITY_INSTRUCTION)

    def test_instruction_keeps_the_no_unverified_success_rule(self) -> None:
        gesenkt = IDENTITY_INSTRUCTION.casefold()
        self.assertIn("tool result", gesenkt)
        self.assertIn("succeeded", gesenkt)


class RecognitionTests(unittest.TestCase):
    def test_current_backends_are_recognised_as_identity_questions(self) -> None:
        for frage in (
            "Bist du Qwen?",
            "Are you Gemma?",
            "Bist du Gemma?",
            "Are you Llama?",
        ):
            with self.subTest(frage=frage):
                self.assertTrue(is_direct_identity_request(_frage(frage)))

    def test_plain_identity_questions_still_work(self) -> None:
        for frage in ("Wer bist du?", "Who are you?", "Welches Modell bist du?"):
            with self.subTest(frage=frage):
                self.assertTrue(is_direct_identity_request(_frage(frage)))

    def test_ordinary_questions_are_not_mistaken_for_identity(self) -> None:
        for frage in (
            "Was ist die Hauptstadt der Schweiz?",
            "Schreibe eine Funktion, die zwei Zahlen addiert.",
            "Wie berechne ich 8,1 % Mehrwertsteuer?",
        ):
            with self.subTest(frage=frage):
                self.assertFalse(is_direct_identity_request(_frage(frage)))

    def test_answer_stays_short_when_nobody_asked_about_infrastructure(self) -> None:
        antwort = deterministic_identity_response(_frage("Wer bist du?"))
        inhalt = antwort["choices"][0]["message"]["content"]
        self.assertEqual("Ich bin Morningstar Hydra, ein CreativeSync-Produkt.", inhalt)

    def test_infrastructure_question_gets_an_honest_answer(self) -> None:
        antwort = deterministic_identity_response(_frage("Welches Backend nutzt du?"))
        inhalt = antwort["choices"][0]["message"]["content"]
        self.assertIn("Morningstar Hydra", inhalt)
        self.assertGreater(len(inhalt), len("Ich bin Morningstar Hydra, ein CreativeSync-Produkt."))


if __name__ == "__main__":
    unittest.main()
