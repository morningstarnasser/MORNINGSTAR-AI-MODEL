from __future__ import annotations

import unittest

from morningstar_hydra.deterministic_control import resolve_exact_request


class DeterministicControlTests(unittest.TestCase):
    def _resolve(self, prompt: str):
        return resolve_exact_request({"messages": [{"role": "user", "content": prompt}]})

    def test_catalog_lookup_supports_locale_data_without_being_the_router(self) -> None:
        code = self._resolve(
            "What is the official two-letter canton code for Zürich? Return only the code."
        )
        name = self._resolve(
            "Expand the canton abbreviation GE. Return only the German canton name."
        )
        german_compound = self._resolve(
            "Für eine Adressierung nach Aargau: nenne das zweistellige Kantonskürzel, sonst nichts."
        )
        german_name = self._resolve(
            "Das Kürzel TI steht für welchen Kanton? Antworte nur mit dem deutschen Namen."
        )
        assert code is not None and name is not None
        assert german_compound is not None and german_name is not None
        self.assertEqual(("ZH", "catalog_lookup"), (code.content, code.capability))
        self.assertEqual(("Genf", "catalog_lookup"), (name.content, name.capability))
        self.assertEqual("AG", german_compound.content)
        self.assertEqual("Tessin", german_name.content)

    def test_catalog_lookup_fails_open_on_ambiguous_non_lookup_requests(self) -> None:
        self.assertIsNone(
            self._resolve("I moved from Bern to Zürich. Compare the quality of life.")
        )
        self.assertIsNone(self._resolve("What does the code BE mean in software?"))

    def test_currency_sum_is_generic_and_exact(self) -> None:
        result = self._resolve(
            "An invoice lists USD 12.50, USD 4.00 and USD 0.05. "
            "Return the total as USD 0.00 and nothing else."
        )
        us_grouping = self._resolve(
            "An invoice lists USD 1,234.56 and USD 5.00. Return the total as USD 0.00."
        )
        eu_grouping = self._resolve(
            "An invoice lists EUR 1.234,56 and EUR 5,00. Return the total as EUR 0.00."
        )
        with_identifier = self._resolve(
            "Request EDE7. From the list price CHF 840.00 subtract a 10% discount. "
            "Return the final price as CHF 0.00."
        )
        assert result is not None and us_grouping is not None and eu_grouping is not None
        assert with_identifier is not None
        self.assertEqual("USD 16.55", result.content)
        self.assertEqual("currency_arithmetic", result.capability)
        self.assertEqual("USD 1239.56", us_grouping.content)
        self.assertEqual("EUR 1239.56", eu_grouping.content)
        self.assertEqual("CHF 756.00", with_identifier.content)

    def test_percentage_and_remainder_use_decimal_half_up(self) -> None:
        tax = self._resolve(
            "Net amount CHF 100.00. Add 8.1% VAT and return the gross amount as CHF 0.00."
        )
        discount = self._resolve(
            "From the list price EUR 99.95 subtract a 20% discount. "
            "Return the final price as EUR 0.00."
        )
        remainder = self._resolve(
            "Of USD 100.00, USD 35.20 has already been paid. "
            "Return the outstanding amount as USD 0.00."
        )
        self.assertEqual("CHF 108.10", tax.content)
        self.assertEqual("EUR 79.96", discount.content)
        self.assertEqual("USD 64.80", remainder.content)

    def test_exact_engine_fails_open_when_contract_or_currency_is_ambiguous(self) -> None:
        self.assertIsNone(self._resolve("CHF 10 or EUR 12 — which is cheaper?"))
        self.assertIsNone(self._resolve("Explain how VAT works on CHF 100."))
        self.assertIsNone(
            self._resolve(
                "An invoice lists USD 12,34,56 and USD 1.00. Return the total as USD 0.00."
            )
        )
        self.assertIsNone(
            resolve_exact_request(
                {
                    "messages": [{"role": "user", "content": "Add CHF 1 and CHF 2."}],
                    "tools": [{"type": "function", "function": {"name": "calculator"}}],
                }
            )
        )

    def test_exact_engine_abstains_on_openai_contracts_it_cannot_honor(self) -> None:
        prompt = "An invoice lists USD 12.50 and USD 4.00. Return the total as USD 0.00."
        base = {"messages": [{"role": "user", "content": prompt}]}
        variants = [
            {"messages": [{"role": "system", "content": "Always explain."}, *base["messages"]]},
            {**base, "n": 2},
            {**base, "response_format": {"type": "json_object"}},
            {**base, "chat_template_kwargs": {"enable_thinking": True}},
        ]
        for payload in variants:
            with self.subTest(payload=payload):
                self.assertIsNone(resolve_exact_request(payload))


if __name__ == "__main__":
    unittest.main()
