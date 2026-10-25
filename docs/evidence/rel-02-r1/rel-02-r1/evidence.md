# REL-02: manual-human proof

Source: `0a56ce0cab61bb3fc335d6032b3eb734546c86d7`. This packet preserves genuine existing source PASS / riskNO and a fresh configured software PASS. It is assembled under Herder's delegated decision of 2026-10-25; it does not attribute new words to Piotr.

## Result and scope

Removing the unused lesson-58 diagram originally left generator consumers raising `KeyError: 58` and derived publication metadata stale. The repair preserves `MAPS[58] = []`, keeps the unused definition deleted, and updates only route-index/source metadata. The 56 surviving route definitions and 2,837 rendered files remain unchanged according to the retained independent verifier. No new teaching content or rendered asset is included.

| Evidence | Actual result |
|---|---|
| Tests-only RED `8692e141a9ba4d8ced0e3619c12b40d009e6aa3b` | 4 tests: 2 assertion failures, 2 KeyError errors; exit 1 |
| GREEN source `0a56ce0cab61bb3fc335d6032b3eb734546c86d7` | All 4 new tests pass plus 1 existing route test; 5 tests total, exit 0 |
| Frozen assertions | Identical test blob at RED/GREEN; 23 preexisting test files unchanged |
| Retained actual visual contracts | 13/13 PASS |
| Retained refresh / export checks | changed=0 / 56 authored graphs; both exit 0 |
| Fresh configured verification | 72 tests PASS, exit 0, source exact, dirty=false |
| Retained separate source review | verifier:rel-02-r1, pane w1:p4Q, PASS / RISK_FLAG NO |
| Current builder history audit | All 93 retained builder/verifier/overnight artifact hashes match |

RED discovers only the new test module; GREEN discovers the entire visual-contracts test directory. Their four new test IDs match, and GREEN includes one additional old test. The original errors and different command breadth are preserved. These outputs are historical checks, not a canonical replay receipt. No `LEAD_RESULT` row has been added.

## Raw commands and output

### RED

Source: `8692e141a9ba4d8ced0e3619c12b40d009e6aa3b`; exit 1; cwd `/tmp/lead-wt/overnight-rel02-r1`.

```sh
.venv/bin/python -m unittest discover -s tests/visual_contracts -p test_solution_map_consumers.py -v
```

Recorded environment: `{"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/tmp/lead-wt/overnight-rel02-r1/scripts"}`.

[Complete raw stderr](red.log) · SHA256 `7e90c5344e7a0b5fec8d7679af9692e9d5797f0d2075d2aec2fa10f4b50b07a8`

```text
test_atlas_includes_lesson_58_without_an_unused_map_link (test_solution_map_consumers.SolutionMapConsumerContracts.test_atlas_includes_lesson_58_without_an_unused_map_link) ... ERROR
test_lesson_58_revision_reaches_html_output_without_exporting_a_map (test_solution_map_consumers.SolutionMapConsumerContracts.test_lesson_58_revision_reaches_html_output_without_exporting_a_map) ... ERROR
test_read_only_architecture_export_is_current (test_solution_map_consumers.SolutionMapConsumerContracts.test_read_only_architecture_export_is_current) ... FAIL
test_read_only_publication_refresh_is_current (test_solution_map_consumers.SolutionMapConsumerContracts.test_read_only_publication_refresh_is_current) ... FAIL
Ran 4 tests in 0.388s
FAILED (failures=2, errors=2)
```

### GREEN

Source: `0a56ce0cab61bb3fc335d6032b3eb734546c86d7`; exit 0; cwd `/tmp/lead-wt/overnight-rel02-r1`.

```sh
.venv/bin/python -m unittest discover -s tests/visual_contracts -v
```

Recorded environment: `{"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/tmp/lead-wt/overnight-rel02-r1/scripts"}`.

[Complete raw stderr](green.log) · SHA256 `0dcd1e6c19be4cee499a23406a0be520a5ed971bb60111a1b59cc746065aa5eb`

```text
test_atlas_includes_lesson_58_without_an_unused_map_link (test_solution_map_consumers.SolutionMapConsumerContracts.test_atlas_includes_lesson_58_without_an_unused_map_link) ... ok
test_lesson_58_revision_reaches_html_output_without_exporting_a_map (test_solution_map_consumers.SolutionMapConsumerContracts.test_lesson_58_revision_reaches_html_output_without_exporting_a_map) ... ok
test_read_only_architecture_export_is_current (test_solution_map_consumers.SolutionMapConsumerContracts.test_read_only_architecture_export_is_current) ... ok
test_read_only_publication_refresh_is_current (test_solution_map_consumers.SolutionMapConsumerContracts.test_read_only_publication_refresh_is_current) ... ok
test_authored_solution_routes_match_rendered_reading_routes (test_solution_routes.SolutionRouteContracts.test_authored_solution_routes_match_rendered_reading_routes) ... ok
Ran 5 tests in 15.345s
OK
```

### VERIFICATION

Source: `0a56ce0cab61bb3fc335d6032b3eb734546c86d7`; exit 0; cwd `/tmp/lead-wt/ready-rel-finish-20261025-rel-02`.

```sh
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

[Complete raw stderr](verification.log) · SHA256 `6fcc97710c21ee4f4aaf5ee116a7b14029137850bf9c5648fe6db172bfa65863`

```text
Ran 72 tests in 21.982s
OK
```

## Authority and outstanding gates

The [manifest](manifest.json) retains the delegated decision, exact command/status identities, original log hashes, historical audit and source review references. The waiver covers the missing historical invocation ledger only. The current builder recorded its own history audit and configured verification invocations; no lead or verifier invocation history was invented.

Manual-human receipt assembly is distinct from engine execution. Engine acceptance, environment acceptance seal and canonical replay are NOT_RUN. The standard engine cannot consume this history waiver; its renderer also requires canonical outcome rows absent from these genuine logs. This packet is manually rendered and makes no renderer/validator PASS claim. Historical builder full-suite NOT_RUN remains preserved; fresh configured verification and the retained verifier's separate 72-test PASS do not rewrite it.

The retained review covers source only. The parent must arrange a separate read-only final delivery review and obtain actual exact-delivery-head CI before any merge. Browser rendering and full write-mode generation remain NOT_RUN; the separate morning teaching/rendering follow-up remains parked. No model/protocol campaign, install, spend, merge or deployment is part of this packet. PR/CI outcomes are recorded in the external current-finish handoff after publication attempts; this committed packet claims neither.
