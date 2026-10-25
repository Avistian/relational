# REL-02 builder handoff: definition removed, verification incomplete

Builder: `builder:rel-02`, pane `w1:p4E`, worktree `/tmp/lead-wt/unblock-rel02`.
Base: `503704f30fa28e9c8896302d1423f0fcca9b539c`.
Tests-only RED commit: `44fb0e22d9f337466ef3dc74d7749a8311655f70`.
Implementation / focused-tested commit: `dfd07f6913fa062381893bc4858e979fb191292d`.
The final handoff SHA and source qualification are in task `evidence/builder/READY.json`.

The sole production change deletes the 12-line lesson-58 `evidence-funnel`
definition from `labs/_solution_map_content.py`. The new test under
`tests/visual_contracts` compares current authored solution routes with actual
lesson responsive maps using the real visual contract's selection and parsing.
No checker assertions were relaxed. No lesson HTML or assets were edited.

Authority is the actual `grill.md` answer: Piotr via the standing order recorded
by Herder, subject to morning confirmation/revision. This is not a new direct
Piotr answer or a new delegation window. Removal only; no lesson-58 rendering,
new content, or P2.

| Check | Before deletion | After deletion |
|---|---|---|
| Selected new regression | RED, 1 assertion failure, 0 errors | GREEN, 1 test, 0 failures/errors |
| Existing 13 visual contracts | 12 pass, 1 fails | 11 pass, 2 fail |
| `refresh_lesson_visuals.py --check` | PASS; 228 lessons, changed=0 | FAIL; source digest mismatch |
| `export_architecture_routes.py --check` | PASS; 57 graphs | FAIL; architecture route snapshot stale |
| Actual rendered bytes | 1700 HTML/asset files captured | All 1700 hashes identical; no added/removed files |
| Configured `/verify-project` | Not executed | NOT_RUN; discovery includes forbidden model training |

The selected regression failed specifically on `solution-evidence-funnel`
before production changes, and passed after deletion. Its assertions and file
hash stayed frozen; see `tdd-summary.json`, `tdd-declaration.json`, and raw
`red.*` / `green.*`. The tests-only RED commit contains only the selected tests.
Raw test statuses truthfully record dirty=true because the proof directory was
untracked while checks ran; production and tests were committed at the shown
SHAs.

The existing route-inventory failure is preexisting. The additional digest
failure and both failing post-change check commands are consequences of the
minimal deletion: the unchanged `assets/architecture-routes.json` still stores
the removed route and the original source hash. Baseline refresh/export PASS
must not be presented as final PASS. Satisfying all requested checks requires
resolving that generated snapshot, outside this task's explicit no-assets-edit
scope. No export was run in write mode, no gates were bypassed, and no synthetic
or temporary route inventory was substituted for the actual files.

`configured-verification.json` pins the installed verification configuration
and explains why `bin/lead-verify relational --worktree ...` was NOT_RUN:
`tests/relmini/test_relmini.py:66-79` invokes `baseline.fit_predict`, which calls
`model.fit` at `labs/relmini/baseline.py:78`. The task forbids model training.
No configuration mapping or test selection was changed to manufacture a pass.

Raw baseline/final command output is in each `*.stdout.txt`, `*.stderr.txt`, and
`*.status.json` triplet. `rendered-before.json`, `rendered-after.json`, and
`rendered-comparison.json` prove actual byte identity independently of git's
change list. `run_check.py` and `prove_rendered.py` preserve reproducible proof
commands; they do not replace project verification.

Independent exact-SHA review remains for a separate verifier launched by the
parent. Canonical execution, acceptance, push, merge, installs, downloads,
spending, and all model training are NOT_RUN. No subagents were launched.
This is a builder handoff with verification gaps, not DONE or acceptance.

Separate morning follow-up: Piotr should confirm or revise whether to render
the diagram in lesson 58 as a later task. Rendering is not part of this change.
The stale generated snapshot is a distinct immediate scope gap for the parent.
