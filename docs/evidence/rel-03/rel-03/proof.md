# REL03 builder proof

Normal configured unittest discovery now executes the existing visual script contracts. The wrapper delegates to unittest's module loader; no contract logic is copied, and original module/class/method identities remain intact.

Builder: `builder:rel-03`; actual pane: `w1:p5R` (`HERDR_PANE_ID`). The inherited `LEAD_SELF_PANE=w1:p1` identifies the parent, not this builder. No subagents were launched.

Launch base: `2dfc5817872fa2a0a7768c8951d3a8b5487d09aa`. The parent's launch record explicitly confirms PR31 MERGED with green CI; see `evidence/launch.json`. The local base is that merge commit. This is parent-provided merge confirmation, not a new builder GitHub query.

Tests-only RED: `55ea896b1cbc5920173a0bab468c89284b215126`.
GREEN/source: `32fe8400d83c466ce0e41e4bb3f7740f7227e422`.
The delivery commit adds proof only; its exact identity and clean current verification are recorded in the final task evidence handoff.

## TDD

`lead-guard declare-implementation --task rel-03 --slice rel-03 --worktree /tmp/lead-wt/build-rel03-20261025` succeeded before RED; the declaration pins the wrapper as implementation and the launch base as `before_red`.

Selected immutable regression:
`tests/contracts/test_visual_discovery.py::VisualDiscovery.test_discovery_runs_every_original_visual_contract`.

Local RED command: `/home/avist/Projects/relational/.venv/bin/python -B tests/contracts/test_visual_discovery.py -v`. Exit 1: one assertion failure, zero errors/skips. Public discovery returned zero cases versus the original 13. This command ran against the regression content subsequently committed unchanged in RED. `evidence/red.log` retains the failure.

The same command after the wrapper was added exits 0: one test passes (`evidence/green.log`). The regression and package marker are unchanged between RED and GREEN. It compares complete original IDs with a multiset (detecting omissions and duplicates), checks a minimum of 13, executes the discovered cases, rejects skips, and checks each ID in verbose output. Its subprocess targets only `test_visual_contracts.py`, avoiding recursive discovery. Expected original IDs are loaded independently; a fresh subprocess prevents those imports from supplying wrapper setup.

The actual canonical runner's standalone file-loading code was also executed on the selected regression at GREEN: one run, zero failures/errors/skips (`evidence/standalone-canonical-loader.log`). It uses no relative-package imports. This is a standalone loader check, not a canonical two-commit replay receipt. Parent-supported canonical receipt: NOT_RUN by builder, pending parent.

## Verification

`/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/build-rel03-20261025`: PASS, exit 0, clean source (`dirty:false`), 129 tests, no skips reported. Exact head: `32fe8400d83c466ce0e41e4bb3f7740f7227e422`. Original durable directory: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261025T133346Z`. Raw status/stdout/stderr are copied under `evidence/source-verification-*`.

`/home/avist/Projects/relational/.venv/bin/python -B -m unittest discover -s tests/contracts -p test_visual_contracts.py -v`: exit 0, 13 tests. The verbose log includes both `check_course_visuals` cases and every original module ID.

| Original module | Cases |
|---|---:|
| check_course_visuals | 2 |
| check_lesson_visuals | 3 |
| check_visual_details | 1 |
| check_lesson_openings | 7 |
| Total | 13 |

Full original IDs are in `evidence/original-case-coverage.json`. No cases were moved, skipped or altered. `git diff --exit-code 2dfc5817872fa2a0a7768c8951d3a8b5487d09aa -- scripts` exited 0; `evidence/scripts-diff.txt` is intentionally empty.

## Handoff boundaries

Independent read-only `/blast-radius` review, supported receipts, PR creation, hosted green CI and merge remain parent-owned and pending. The builder did not review its own work, push, merge, install, download, alter profiles, answer permission prompts, or run models. No acceptance or hosted CI success is claimed. Final REPORT.md and evidence/READY.json are written last in `/home/avist/agent-stack/lead/.lead/tasks/rel-03/evidence/current-build` after the delivery commit and current configured verification.
