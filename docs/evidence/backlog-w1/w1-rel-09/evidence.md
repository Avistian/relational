# REL-09 flashcard integrity source evidence

The JavaScript deck evaluates to 103 cards with 103 unique IDs and no missing or empty required fields. The actual `scripts/progress_report.py` build reports 103 cards. No count discrepancy was found.

The checker is confined to `tests/flashcards/check_deck.py`; deck and report source are unchanged. Tests cover the evaluated deck, duplicate ID mutation, each missing/null/empty/whitespace required field, and removed-card/wrong-report-total mutations. Copies are changed only in memory.

## Exact count audit

Executed from the worker worktree with the shared `/home/avist/Projects/relational/.venv/bin/python`. The test module supplies `actual_deck()` (Node evaluates the real JavaScript) and `actual_report()` (imports and calls the real report). Raw audit output:

```json
{
  "evaluated_cards": 103,
  "unique_ids": 103,
  "duplicate_ids": [],
  "missing_or_empty_fields": [],
  "actual_report_flashcards_total": 103,
  "unquoted_id_lines": 48,
  "quoted_id_lines": 55
}
```

Actual commands and raw outputs:

```sh
rg -c 'id:' assets/paper-deck.js
# stdout:
48
rg -c '"id":' assets/paper-deck.js
# stdout:
55
```

The 48 observation counts only unquoted keys. The other 55 cards have quoted `"id":` keys. The independent evaluated total and report total both equal 48 + 55 = 103. Formatting-sensitive grep is not the deck integrity oracle.

## Test-first evidence

Identical command for [RED](red.log) and [GREEN](green.log):

```sh
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests/flashcards -p 'test_*.py'
```

RED: 4 tests, 4 assertion failures, no errors or skips. Tests-only commit: `f9ec7f75bcb1c75cb4a92a9abb2292a5bdea8451`. GREEN: 4 tests passed. Canonical tests remain byte-identical. Canonical replay and full verification evidence will be recorded against committed source in the task handoff. Independent review, final committed human-proof manifest, acceptance and draft CI are pending the lead bridge.

## Preserved blocked attempt

Previous uncommitted RED and Git read-only errors are preserved under the task's `blocked-snapshots/w1-rel-09` and `recovery-w1-rel-09/prior-evidence`; they are not canonical evidence for this recovery.
