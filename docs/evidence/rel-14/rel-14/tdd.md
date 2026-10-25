# REL14 TDD execution

Actor `builder:rel-14`; actual pane `w1:p5X`; role builder.
Tests-only RED: `575b49b9bd335ce1bb64205ff6db8b5c0b22c4c9`.
Production GREEN: `341875ebcad870ffec05b31b1893b01677ba923e`.
Selected standalone identity:
`tests/search/test_external_asset.py::ExternalAssetContract.test_small_home_loads_external_index_and_renders_known_lesson`.
The entire tests tree is unchanged across this pair. No relative imports.
Selected file SHA256: `a966fc719494fe9efd22cf82cdc871bf08c2d1645b7b083454a4bf79067279fd`.

Actual command: `bin/lead-guard tdd validate --report /tmp/rel14-tdd-report.json --receipt /home/avist/agent-stack/lead/.lead/tasks/rel-14/reports/rel-14.tdd-builder.json --worktree /tmp/lead-wt/build-rel14-20261025`.
An initial attempt used unsupported mode `replay` and returned usage exit 2
without executing tests. Supported `validate` passed.

The canonical runner loaded the selected file directly with importlib in
separate detached worktrees using the configured existing interpreter.
RED: run 1, failed 1, errors 0, skipped 0; assertion 169499 is not less than
20000. GREEN: run 1, failed 0, errors 0, skipped 0. Read canonical-red.log,
canonical-green.log, canonical-tdd-report.json and canonical-tdd-receipt.json.
This proves the selected regression, not whole-task engine acceptance.

Development history: initial regression 3eb756b0 found the page-size failure.
The first GREEN attempt exposed the OS command-argument size limit in the test
runner. Before the final pair, the runner switched to stdin, gained a window
adapter, and legacy inline-data tests migrated to the external contract.
Final RED also checks the complete external index before rendering a known
lesson link. No tests or test support changed after final RED.
