# Human evidence

Source: 8803fdd68f99960148bf0a5f124f4948a6b4df9a
Reviewer: verifier:w1-rel-09

BEFORE failures: 4
AFTER failures: 0

## RED
```text
LEAD_RESULT={"run": 4, "failed": 4, "errors": 0, "skipped": 0}
test_real_ids_are_unique_and_duplicate_copy_is_rejected (lead_replay_0.DeckIntegrity.test_real_ids_are_unique_and_duplicate_copy_is_rejected) ... FAIL
test_each_required_field_rejects_missing_null_and_empty_values (lead_replay_1.DeckIntegrity.test_each_required_field_rejects_missing_null_and_empty_values) ... FAIL
test_actual_report_total_matches_evaluated_deck (lead_replay_2.DeckIntegrity.test_actual_report_total_matches_evaluated_deck) ... FAIL
test_removed_card_copy_and_wrong_report_total_are_rejected (lead_replay_3.DeckIntegrity.test_removed_card_copy_and_wrong_report_total_are_rejected) ... FAIL
======================================================================
FAIL: test_real_ids_are_unique_and_duplicate_copy_is_rejected (lead_replay_0.DeckIntegrity.test_real_ids_are_unique_and_duplicate_copy_is_rejected)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 41, in test_real_ids_are_unique_and_duplicate_copy_is_rejected
    validate = checker()
               ^^^^^^^^^
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 14, in checker
    raise AssertionError("deck integrity checker has not been implemented")
AssertionError: deck integrity checker has not been implemented
======================================================================
FAIL: test_each_required_field_rejects_missing_null_and_empty_values (lead_replay_1.DeckIntegrity.test_each_required_field_rejects_missing_null_and_empty_values)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 48, in test_each_required_field_rejects_missing_null_and_empty_values
    validate = checker()
               ^^^^^^^^^
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 14, in checker
    raise AssertionError("deck integrity checker has not been implemented")
AssertionError: deck integrity checker has not been implemented
======================================================================
FAIL: test_actual_report_total_matches_evaluated_deck (lead_replay_2.DeckIntegrity.test_actual_report_total_matches_evaluated_deck)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 64, in test_actual_report_total_matches_evaluated_deck
    checker()(cards, report_total)
    ^^^^^^^^^
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 14, in checker
    raise AssertionError("deck integrity checker has not been implemented")
AssertionError: deck integrity checker has not been implemented
======================================================================
FAIL: test_removed_card_copy_and_wrong_report_total_are_rejected (lead_replay_3.DeckIntegrity.test_removed_card_copy_and_wrong_report_total_are_rejected)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 68, in test_removed_card_copy_and_wrong_report_total_are_rejected
    validate = checker()
               ^^^^^^^^^
  File "/tmp/lead-tdd-emecl9cf/tree/tests/flashcards/test_deck_integrity.py", line 14, in checker
    raise AssertionError("deck integrity checker has not been implemented")
AssertionError: deck integrity checker has not been implemented
----------------------------------------------------------------------
Ran 4 tests in 0.130s
FAILED (failures=4)
```

## GREEN
```text
LEAD_RESULT={"run": 4, "failed": 0, "errors": 0, "skipped": 0}
test_real_ids_are_unique_and_duplicate_copy_is_rejected (lead_replay_0.DeckIntegrity.test_real_ids_are_unique_and_duplicate_copy_is_rejected) ... ok
test_each_required_field_rejects_missing_null_and_empty_values (lead_replay_1.DeckIntegrity.test_each_required_field_rejects_missing_null_and_empty_values) ... ok
test_actual_report_total_matches_evaluated_deck (lead_replay_2.DeckIntegrity.test_actual_report_total_matches_evaluated_deck) ... ok
test_removed_card_copy_and_wrong_report_total_are_rejected (lead_replay_3.DeckIntegrity.test_removed_card_copy_and_wrong_report_total_are_rejected) ... ok
----------------------------------------------------------------------
Ran 4 tests in 0.151s
OK
```

## VERIFICATION
```text
FAIL hash check:
  x: sha256 mismatch
```

<details>
<summary>Exact commands (all arguments preserved)</summary>

```sh
# RED and GREEN
/home/avist/Projects/relational/.venv/bin/python -B -c '
import importlib.util,json,pathlib,sys,unittest
root=pathlib.Path.cwd(); suite=unittest.TestSuite(); ids=json.loads(sys.argv[1])
for entry in ids:
    file, identity=entry.split('"'"'::'"'"',1)
    spec=importlib.util.spec_from_file_location('"'"'lead_replay_'"'"'+str(len(suite._tests)),root/file)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target=module
    for name in identity.split('"'"'.'"'"'): target=getattr(target,name)
    cls=getattr(module,identity.split('"'"'.'"'"')[0]); suite.addTest(cls(identity.split('"'"'.'"'"')[-1]))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('"'"'LEAD_RESULT='"'"'+json.dumps(dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped))))
sys.exit(0 if result.wasSuccessful() else 1)
' '["tests/flashcards/test_deck_integrity.py::DeckIntegrity.test_real_ids_are_unique_and_duplicate_copy_is_rejected", "tests/flashcards/test_deck_integrity.py::DeckIntegrity.test_each_required_field_rejects_missing_null_and_empty_values", "tests/flashcards/test_deck_integrity.py::DeckIntegrity.test_actual_report_total_matches_evaluated_deck", "tests/flashcards/test_deck_integrity.py::DeckIntegrity.test_removed_card_copy_and_wrong_report_total_are_rejected"]'
# VERIFICATION
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

</details>
