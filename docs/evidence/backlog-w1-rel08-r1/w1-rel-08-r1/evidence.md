# REL-08 R1 builder proof — source ready, acceptance pending

Actor `builder:w1-rel-08-r1`; actual pane `w1:p15`; worktree
`/tmp/lead-wt/w1-rel-08-r1`; branch `lead/w1-rel-08-r1`.
Task `backlog-w1-rel08-r1`, slice `w1-rel-08-r1`, reservation `da011d8df01a`.
This follow-on uses its own declaration, intake and canonical receipt. Parent
REL-08 remains historical BLOCKED/superseded; its evidence was not rewritten.

Piotr's actual answer in [grill.md](grill.md) is:
“Reject duplicates; retain missing-ID warnings (Recommended)”. The original
[blocked review](original-w1-rel-08.md), [probe](original-w1-rel-08-probe.py) and
[raw result](original-w1-rel-08-probe.txt) are preserved verbatim as history.
The old probe deliberately asserts the defect and is not a current passing test.

## Change and bounded behavior

At base `ae27156b5b27a583de80c95be1c3bcfa8eec3538`, nbformat 5.10.4 repairs duplicate
cell IDs in memory and emits DuplicateCellId, while audit returns no error.
The five-line checker addition turns that warning category into a blocking audit
error and retains the warning itself. MissingIDFieldWarning stays nonblocking.
No notebook, teaching content, exception list or workflow changed in R1.
The library remains the existing installed/pinned version; no installs occurred.

## Immutable tests and canonical replay

Before RED, actual absolute command:

```text
/home/avist/agent-stack/lead/bin/lead-guard declare-implementation --task backlog-w1-rel08-r1 --slice w1-rel-08-r1 --worktree /tmp/lead-wt/w1-rel-08-r1
TDD implementation declaration pinned before RED
```

[Exact declaration](implementation-declaration.json) pins only
`tests/notebooks/check_notebooks.py` and the original attached worktree/branch.
Tests-only RED commit: `6729ba07` (full ID in source-report.json).
GREEN implementation commit: `a413484bed60b7ea8052d11b0614993c0d5a9725`.
Original seven tests retain SHA256
`354bad4635392246382d549f073bcb38b14ea0a4e009a243255f34e1babc0a0d`.
New canonical test file retains SHA256
`69a477f87c205777ab161d8fd2e157c7dbcfb11ee156d3b9f42243b6256f8d52`.

Focused command (both phases):

```text
/home/avist/Projects/relational/.venv/bin/python -B -m unittest discover -s tests/notebooks -p 'test_*.py' -v
```

[Raw RED](focused-red.log): 9 tests, one meaningful assertion failure:
`AssertionError: 0 != 1 : Duplicate IDs must block the real audit`.
[Raw GREEN](focused-green.log): 9 tests, OK. Both test files remain unchanged.

Supported canonical command:

```text
/home/avist/agent-stack/lead/bin/lead-guard tdd validate --report /home/avist/agent-stack/lead/.lead/tasks/backlog-w1-rel08-r1/w1-rel-08-r1-source-report.json --receipt /home/avist/agent-stack/lead/.lead/tasks/backlog-w1-rel08-r1/w1-rel-08-r1-source-tdd.json --worktree /tmp/lead-wt/w1-rel-08-r1
TDD replay RED/GREEN verified
```

Actual [RED replay](canonical-red.log):
`LEAD_RESULT={"run": 2, "failed": 1, "errors": 0, "skipped": 0}`.
Actual [GREEN replay](canonical-green.log):
`LEAD_RESULT={"run": 2, "failed": 0, "errors": 0, "skipped": 0}`.
The [receipt](canonical-tdd-receipt.json) and [report](source-report.json) bind this
follow-on task/slice, exact declaration and immutable test identities.

## Inventory, warnings and mutation evidence

[Preservation result](preservation.json) compares all 410 paths and SHA256 values
before/after and compares the complete raw inventory objects for equality.
Counts remain 410 notebooks, 221 student/template notebooks, 222 HTML files,
148 solutions, zero errors, 464 MissingIDFieldWarning warnings.
[Before](inventory-before.json) and [after](inventory-after.json) include every raw
warning, pairing gap and exception; both hash manifests are retained alongside.

[CLI probe source](cli-probe.py) and [raw output](cli-probe.log) run a byte-identical
checker copy against complete disposable student/HTML/solution pairs. The exact
duplicate-ID input exits 1; missing IDs exit 0 with warning. Both preserve input
bytes. Deleting the shipped 0081 solution still exits 1 for the expected pairing
error. Generated repaired IDs in diagnostic warning text are library-generated;
source notebooks are never written.

## Actual configured verification

Read/applied installed skills in order:
`/home/avist/agent-stack/lead/.agents/skills/tdd/SKILL.md`, then
`/home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md`.
The user-approved contract supplies the test plan; no extra interview is needed.
No nested agents or independent review are claimed by this builder.

Actual environment command:
`/home/avist/agent-stack/lead/bin/lead-env relational check --worktree /tmp/lead-wt/w1-rel-08-r1`.
[Raw result](env-source.log): PASS, clean source, implementation head above.
Durable environment evidence:
`/home/avist/agent-stack/lead/.lead/evidence/env/relational/1792781712318166784`.
Readiness expires after five minutes; Lead must refresh for acceptance.

Actual project command:
`/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/w1-rel-08-r1`.
Opened [status](verify-source-status.json), [stdout](verify-source-stdout.txt), and
[stderr](verify-source-stderr.txt). Verdict PASS, exit_code 0, timed_out false,
source.head `a413484bed60b7ea8052d11b0614993c0d5a9725`, dirty false.
Actual stderr: `Ran 51 tests in 5.554s` and `OK`.
Stdout contains inventory JSON followed by `FAIL hash check: x: sha256 mismatch`
from a passing unrelated negative test. Stdout alone is not unittest status.
Durable project evidence:
`/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T185554Z`.

## Remaining Lead-owned delivery

This proof is committed after the clean implementation head; that commit changes
only this R1 evidence directory. The shared source-ready handoff names its full
SHA, confirms code equivalence, and points to fresh final-head environment and
project outputs. Canonical receipt above deliberately binds the implementation
head; no parent receipt or historical record is substituted or overwritten.

Lead runs builder-done and launches the distinct read-only verifier. Verifier must
apply installed /blast-radius and /verify-project to the full feature diff against
`00f3dfd721dc66bbab3f9672705fd0f60fe8a416`, including original intent evidence,
workflow/dependency change and R1. Independent PASS, final human proof/code
equivalence review, acceptance under THIS task/reservation, managed draft PR and
actual final-head CI remain pending. No push, publication, merge, browser checks,
notebook execution or scientific result is claimed. Inspection-only unsuccessful
commands are retained in inspection-attempts.txt; all RED attempts are preserved.
