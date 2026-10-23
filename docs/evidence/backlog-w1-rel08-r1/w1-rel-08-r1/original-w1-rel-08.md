# REL-08 independent source review

Verdict: **BLOCKED** — one demonstrated malformed-notebook acceptance defect (R1).
This is a source review, not a DONE acceptance report.

Reviewed source: `ae27156b5b27a583de80c95be1c3bcfa8eec3538`.
Base: `00f3dfd721dc66bbab3f9672705fd0f60fe8a416`.
Reviewer: `verifier:w1-rel-08`; actual counted pane: `w1:p13`.
Worktree: `/tmp/lead-wt/v-w1-rel-08`.
Builder: `builder:w1-rel-08`, a distinct recorded actor.
No subagents, implementation changes, installs, publication, declaration changes,
canonical test edits, intake changes, or acceptance commands were performed.

## R1 — duplicate cell IDs are repaired and accepted instead of blocking

Severity: medium; blocking against spec contract 3, “malformed notebook blocks.”
Location: `tests/notebooks/check_notebooks.py:138` and `:141-146`.
The checker calls `nbformat.validate(document)` with its default repair behavior,
then stores emitted warnings without converting this malformed case into an error.
Two cells with the same ID therefore return `errors: []`. The CLI at line 154
and repository assertion at `tests/notebooks/test_notebooks.py:98` treat that result
as success. Repair occurs in the in-memory dictionary; original bytes remain malformed.
The existing invalid-outputs test does not exercise this path.

Safety fact tested to level 4 (real checker and installed library execution):
malformed notebooks must cause an audit error. This fact is **disproven** for
duplicate cell IDs. A strict-library call on the identical input rejects them.
Installed library: nbformat 5.10.4,
`/home/avist/Projects/relational/.venv/lib/python3.12/site-packages/nbformat/validator.py`.
Lines 362-385 implement ID repair versus ValidationError; lines 464-465 select
repair by default. This is a concrete future drift escape, not a claim that any
of the present 410 notebooks has duplicate IDs. Likelihood is plausible under
notebook cell copy/edit workflows; consequence is malformed cell identity passing CI.

Exact disposable repro:

```bash
cd /tmp/lead-wt/v-w1-rel-08
/home/avist/Projects/relational/.venv/bin/python -B reviews/w1-rel-08-probe.py > reviews/w1-rel-08-probe.txt 2>&1
```

Full retained script: `/tmp/lead-wt/v-w1-rel-08/reviews/w1-rel-08-probe.py`.
Raw output: `/tmp/lead-wt/v-w1-rel-08/reviews/w1-rel-08-probe.txt`.
The probe exited 0 because it asserts and confirms the defect; this is not a
passing malformed-input regression test.

```text
PASS: deleting a currently shipped solution is rejected: ['student_without_solution: labs/0081-mpnn-framework.ipynb']
DUPLICATE-ID AUDIT: {"errors": [], "warning_categories": ["DuplicateCellId"], "count": 2}
STRICT LIBRARY VALIDATION REJECTS: Non-unique cell id 'duplicate' detected.
PASS: all 410 repository notebook hashes unchanged
FINDING: malformed duplicate-ID notebook returns zero audit errors; emitted warning is not a blocking result
```

The strict counterfactual is
`nbformat.validate(copy.deepcopy(malformed), repair_duplicate_cell_ids=False)`.
No strict mode was substituted into production. Both malformed cells are valid
markdown cells individually, with `id: duplicate`; the temporary fixture otherwise
has complete student/HTML/solution pairing. All 410 repository notebook SHA256
values were compared before and after the probe and remained equal.

## Verification evidence

Actual installed skills applied:

- `/home/avist/agent-stack/lead/.agents/skills/blast-radius/SKILL.md`
- `/home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md`

Launch environment command:
`/home/avist/agent-stack/lead/bin/lead-env relational check --worktree /tmp/lead-wt/v-w1-rel-08`.
Evidence: `/home/avist/agent-stack/lead/.lead/evidence/env/relational/1792780874232215849/status.json`.
Actual verdict PASS; source.head equals reviewed SHA; dirty false.
This readiness record expires after five minutes; lead must refresh for acceptance.

Actual verification command:
`/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/v-w1-rel-08`.
Evidence directory:
`/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184123Z`.
Opened status.json, stdout.txt and stderr.txt. Status: `verdict: PASS`,
`exit_code: 0`, `timed_out: false`, `source.head: ae27156b5b27a583de80c95be1c3bcfa8eec3538`,
`dirty: false`. Actual stderr:

```text
.................................................
----------------------------------------------------------------------
Ran 49 tests in 7.729s

OK
```

Verification stdout includes the complete notebook inventory JSON followed by
`FAIL hash check:` / `  x: sha256 mismatch` from another passing test's negative
case. It is not a standalone JSON document. The independently read stderr/status
establish the suite result; stdout alone does not show passing unittest status.
No fixture output establishes browser, notebook execution, or scientific results.

Inventory: 410 validated notebooks, 221 students/templates, 222 top-level lab HTML,
148 solutions; zero current errors. Pair gap counts: student_without_html 1,
html_without_student 2, student_without_solution 73, solution_without_student 0.
All 464 warnings are MissingIDFieldWarning; no observed current DuplicateCellId
warning. Missing-ID warnings were retained; R1 concerns the distinct demonstrated
duplicate-ID case, not a demand to silently reject the documented existing baseline.

## Scope and exceptions checked and cleared

The diff adds the checker and tests, retains diagnostic/recovery evidence, and
adds only `nbformat==5.10.4` to the existing hosted dependency installation line.
No lab notebooks or rendered teaching bodies changed. All five existing CI checks,
contents-read permission, persist-credentials false and step-local TMPDIR remain.
No local installation occurred. Blast radius reaches the unittest discovery CI
step through `test_notebooks.py`; no runtime teaching code imports the checker.

Exceptions are exact paths, not a wildcard freeze:

- 72 named absent teacher copies in `check_notebooks.py:18-91` follow the existing
  local/gitignored teacher-copy policy at `labs/README.md:26-27` and `.gitignore:6`.
  README lines 129-130 additionally state the convention for lessons 058-070.
  The explicit list excludes currently shipped solutions. The independent probe
  deletes the shipped 0081 solution in a temporary paired fixture and gets a
  blocking error, despite the general ignore policy.
- `labs/LAB-TEMPLATE.ipynb` lacks HTML and solution intentionally: the authoring
  convention explicitly identifies it at `labs/README.md:6-8`.
- `labs/html/foundation-sequence.html` is a materials directory, linked as such
  at `notebooks.html:44`, rather than an orphan notebook export.
- `labs/html/0113-scaling-ogb-solution.html` is an explicit suffixed solution
  export in `labs/_execute_l113.py:7,13`.

The seven source tests cover ordinary pair deletion/addition/orphans, malformed
JSON, invalid cell outputs outside the paired labs, counts and actual inventory.
The independent pair-deletion probe confirms a real non-allowlisted existing name
fails. No unexplained current pairing mismatch was found.

## Canonical receipt preserved

Read, did not redeclare or replay against another tree:
`/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/w1-rel-08-source-tdd.json`.
RED commit `9f7f4c71a74a3049d55b72bf92ed3edb7ffb1a1d` contains only the tests and
package marker; the canonical test file is byte-identical at reviewed source.
SHA256: `354bad4635392246382d549f073bcb38b14ea0a4e009a243255f34e1babc0a0d`.
Actual canonical raw lines:

```text
LEAD_RESULT={"run": 7, "failed": 7, "errors": 0, "skipped": 0}
LEAD_RESULT={"run": 7, "failed": 0, "errors": 0, "skipped": 0}
```

Receipt hashes were recomputed and matched:

- red.log: `f172fcb1fee85db5167866307e572a21086e4fbbc913dcf1baa389930aacba8a`
- green.log: `6c5c3b5bc3fbd340d7b0b67207572093a92848ed2c106557649d4d713aaabdbd`

These live under
`/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/w1-rel-08-source-tdd/`.
The original declaration is exclusive to the original builder tree. Intake is
pinned for all four slices. Preserve that declaration, receipt, logs and canonical
seven tests; do not edit intake/pins or weaken immutability to admit this probe.

## Supported next step and delivery limits

Lead coordinates the bounded fix under the existing malformed-notebook scope.
A builder should first turn R1 into a genuine failing duplicate-ID regression,
then implement blocking behavior while preserving explicitly handled existing
warning behavior. If adding that regression requires a fresh canonical cycle,
use a separately admitted, supported follow-on repair task/cycle with its own
immutable tests-only RED, declared implementation seam, GREEN, receipt and source
verification. Do not replace the pinned original declaration or mutate its test
identities/receipt. This verifier has not created such a cycle or changed source.
The cheapest recheck is this exact disposable duplicate-ID input plus the existing
focused notebook suite, followed by required source-bound verification and review.

The source-ready handoff explicitly leaves current human manifest/evidence/PR body
and final-head draft PR CI pending. At this source head, committed evidence.md,
verification.log, source-ready.md and blocked.json contain historical pre-repair
material, identified as such in recovery/tdd.md and the shared handoff. They are
not accepted current human proof. No current manifest/pr-body or final PR/CI URL
was supplied or independently validated; no UI capture applies (FEATURE_KIND non-ui).
Lead must ship and review current raw proof after the source blocker is resolved,
including passing verification stderr/status if the renderer selects stdout only.

Verification invocation was explicitly skill-recorded against the actual nonempty
stderr.txt, reviewer actor, actual pane and reviewed SHA. The review artifact itself
is the actual /blast-radius output to record through the same absolute Lead API.
Only review artifacts are added in this verifier tree after clean-source verification;
no source commit or acceptance report is produced.
