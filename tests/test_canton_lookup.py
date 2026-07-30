"""Prueft die Kantonsauskunft der Control-Plane.

Die Formulierungen hier sind bewusst selbst erdacht und stammen nicht aus dem
Benchmark. Die Regel soll unabhaengig von der Frageform greifen; waeren die
Testsaetze aus den durchgefallenen Faellen abgeleitet, wuerde sie genau die
auswendig lernen.
"""

from __future__ import annotations

import unittest

from morningstar_hydra.deterministic_control import resolve_exact_request


def _loese(text: str):
    return resolve_exact_request({"messages": [{"role": "user", "content": text}]})


class CodeToNameTests(unittest.TestCase):
    def test_various_phrasings_all_resolve_to_the_name(self) -> None:
        for frage in (
            "Welcher Kanton hat das Kürzel VD?",
            "Kanton VD — wie heisst der ausgeschrieben?",
            "Schreib mir zum Kanton GR den Namen.",
            "Für welchen Kanton steht AI?",
            "Kantonskürzel BL: bitte den vollen Namen.",
            "Was bedeutet das Kantonskürzel SH?",
        ):
            with self.subTest(frage=frage):
                treffer = _loese(frage)
                self.assertIsNotNone(treffer, f"nicht erkannt: {frage}")

    def test_the_returned_name_is_the_official_german_one(self) -> None:
        self.assertEqual("Waadt", _loese("Welcher Kanton hat das Kürzel VD?").content)
        self.assertEqual("Wallis", _loese("Welcher Kanton hat das Kürzel VS?").content)
        self.assertEqual("Graubünden", _loese("Für welchen Kanton steht der Code GR?").content)


class NameToCodeTests(unittest.TestCase):
    def test_various_phrasings_all_resolve_to_the_code(self) -> None:
        for frage, erwartet in (
            ("Wie lautet das Kürzel des Kantons Thurgau?", "TG"),
            ("Kanton Neuenburg, bitte nur den Code.", "NE"),
            ("Gib mir für den Kanton Genf die Abkürzung.", "GE"),
            ("Welches Kürzel trägt der Kanton Obwalden?", "OW"),
        ):
            with self.subTest(frage=frage):
                treffer = _loese(frage)
                self.assertIsNotNone(treffer, f"nicht erkannt: {frage}")
                self.assertEqual(erwartet, treffer.content)

    def test_french_and_italian_names_resolve_too(self) -> None:
        self.assertEqual("VD", _loese("Quel est le code du canton de Vaud?").content)
        self.assertEqual("TI", _loese("Qual è la sigla del cantone Ticino?").content)


class AmbiguityTests(unittest.TestCase):
    def test_two_cantons_are_not_resolved(self) -> None:
        self.assertIsNone(_loese("Vergleiche den Kanton Zürich mit dem Kanton Bern."))

    def test_code_and_name_together_are_not_resolved(self) -> None:
        # Beides gegeben heisst: die Richtung ist nicht ableitbar.
        self.assertIsNone(_loese("Stimmt es, dass der Kanton Waadt das Kürzel VD hat?"))

    def test_without_canton_context_nothing_is_resolved(self) -> None:
        for frage in (
            "Ich fahre morgen nach Zürich.",
            "Schreibe eine Funktion namens VD.",
            "Wie ist das Wetter in Bern?",
        ):
            with self.subTest(frage=frage):
                self.assertIsNone(_loese(frage))

    def test_a_question_asking_for_more_than_the_lookup_is_left_to_the_model(self) -> None:
        self.assertIsNone(
            _loese("Welcher Kanton hat das Kürzel VD und wie viele Einwohner hat er?")
        )

    def test_attributes_the_table_does_not_hold_are_left_to_the_model(self) -> None:
        # Die Tabelle kennt genau zwei Groessen: Code und Name. Wird eine dritte
        # verlangt, waere jede Antwort von ihr die Antwort auf eine andere Frage.
        for frage in (
            "Nenne mir den Hauptort des Kantons Wallis.",
            "Wie viele Einwohner hat der Kanton Thurgau?",
            "Welche Amtssprache gilt im Kanton Graubünden?",
            "Wann trat der Kanton Jura der Eidgenossenschaft bei?",
            "Wie gross ist die Fläche des Kantons Uri?",
            "Was ist die Hauptstadt des Kantons Waadt?",
            "Wie sieht das Wappen des Kantons Schwyz aus?",
        ):
            with self.subTest(frage=frage):
                self.assertIsNone(_loese(frage), f"faelschlich aufgeloest: {frage}")


if __name__ == "__main__":
    unittest.main()
