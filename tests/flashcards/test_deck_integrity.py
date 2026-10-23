"""Check the evaluated deck against the report, using disposable mutations."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


def checker():
    path = Path(__file__).with_name("check_deck.py")
    if not path.exists():
        raise AssertionError("deck integrity checker has not been implemented")
    spec = importlib.util.spec_from_file_location("check_deck", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_deck


def actual_deck():
    result = subprocess.run(
        ["node", "-e", "global.window={};require(process.argv[1]);"
         "console.log(JSON.stringify(window.PAPER_DECK));",
         str(ROOT / "assets/paper-deck.js")],
        check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def actual_report():
    path = ROOT / "scripts/progress_report.py"
    spec = importlib.util.spec_from_file_location("progress_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_report(ROOT, "2026-10-23")


class DeckIntegrity(unittest.TestCase):
    def test_real_ids_are_unique_and_duplicate_copy_is_rejected(self):
        cards = actual_deck()
        validate = checker()
        validate(cards, len(cards))
        with self.assertRaisesRegex(ValueError, "duplicate.*id"):
            validate(cards + [dict(cards[0])], len(cards) + 1)

    def test_each_required_field_rejects_missing_null_and_empty_values(self):
        cards = actual_deck()
        validate = checker()
        for field in ("id", "paper", "year", "lesson", "front", "back"):
            for value in (None, "", "   ", "MISSING"):
                with self.subTest(field=field, value=value):
                    changed = [dict(card) for card in cards]
                    if value == "MISSING":
                        del changed[0][field]
                    else:
                        changed[0][field] = value
                    with self.assertRaisesRegex(ValueError, "required.*" + field):
                        validate(changed, len(cards))

    def test_actual_report_total_matches_evaluated_deck(self):
        cards = actual_deck()
        report_total = actual_report()["flashcards"]["total"]
        self.assertEqual(len(cards), report_total)
        checker()(cards, report_total)

    def test_removed_card_copy_and_wrong_report_total_are_rejected(self):
        cards = actual_deck()
        validate = checker()
        report_total = actual_report()["flashcards"]["total"]
        with self.assertRaisesRegex(ValueError, "count"):
            validate(cards[:-1], report_total)
        with self.assertRaisesRegex(ValueError, "count"):
            validate(cards, report_total + 1)


if __name__ == "__main__":
    unittest.main()
