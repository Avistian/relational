# REL-16 R1 builder handoff

Actor: builder in /tmp/lead-wt/night-rel16-r1. No agents, workers or subagents spawned.
Base reviewed SHA: fa4521280ab256beb622bfa3e0c8ec66e2ef68b7.
Final source SHA: 86fe6e7284421f128686da00240aeddb5595cbab; dirty:false.
Status: READY_FOR_PARENT_VERIFIER; builder repair complete, acceptance NOT_DONE.
RISK_FLAG: YES. This remains a bounded shell file-set interpreter.

## Repair

REL16-V1: quoted wildcard characters are rejected before shlex strips quotes.
This covers single quotes, double quotes, partial-word quoting and concatenated
quoted strings. Literal quoted filenames keep existing behavior. The parser does
not execute shell commands. Unsupported inputs return exit 2 with a diagnostic
on stderr and no JSON on stdout, including either side of --compare.

REL16-V2: rm -rf operands with glob metacharacters are rejected, as is removal
of the normalized public root (public, public/, public//, ./public//). Existing
literal subtree removal remains covered by the unchanged frozen fixture.
Fail-closed behavior was explicitly authorized; these forms are not simulated.

Only scripts/pages_publish_list.py, the new test file under tests/pages, and
the owned evidence README changed. No workflow/course-content edits.

## Genuine local TDD, no new canonical replay

| Behavior | Tests-only RED SHA | GREEN SHA | RED failures / runner errors |
| --- | --- | --- | --- |
| quoted wildcards | b5fdb10040d19b5230c35ce8dd161e70cd6eb724 | 5da4d654763e1fd5364a20c908a5446b07f185c5 | 21 / 0 |
| wildcard removal | 27985b26853357f7812896eef5aaab31a521b2e1 | 26fad3df990bce57b2e8692940d4a1453c54824f | 12 / 0 |
| root removal, corrected fixture | c1a98c2e9e6067aed6dc1197f20563e751f6cf57 | dd7580a7e283c4ceff0aca6a9a41a661e47fd0c2 | 12 / 0 |

Each committed RED was run with dirty:false before its production fix. These
are subtest assertion failure counts, not separate test-method counts. Each
fixture executes the fixed synthetic plan with Bash to establish the actual
file-set disagreement, then checks the public CLI. No implementation imports
or mocked parser logic. Tests are byte-identical between each listed RED/GREEN.
The original seven frozen tests retain their original SHA-256 values.
Intermediate GREEN logs honestly record their RED HEAD plus a dirty production
fix; final-focused.json binds all ten passing tests to the clean final SHA.

Preserved history: 3a628006 initially included public/. and wrongly expected GNU
rm success. GNU rm refuses '.'; its raw log v2-root-red-committed.stderr.txt
retains that failed expectation and nine genuine parser failures. The separate
c1a98c2e correction uses valid root-removal spellings and records twelve genuine
parser failures before the root fix. No history/log was amended or removed.

## Current checks

Focused Pages suite: PASS, exit_code:0, source.head:86fe6e7284421f128686da00240aeddb5595cbab, dirty:false;
ten test methods (original seven plus three new regression methods).
Fresh publication inventory: PASS, Bash/GNU cp and generator --compare exit 0;
18,497 actual = 18,497 generated, zero differences. Inventory also matches the
retained original list exactly. Missing sources, missing references and omitted
references are separately empty. Only the two existing HTML byte mutations are
omitted from the inventory script; every copy/loop/conditional/removal runs.
This is builder proof of file membership, not byte validation or deployment.

/verify-project: NOT_RUN, exit_code:null, source.head:86fe6e7284421f128686da00240aeddb5595cbab, dirty:false.
Installed verify-project/SKILL.md was read. Current configured unittest discovery
includes tests/relmini/test_relmini.py:69-78 calling baseline.fit_predict, which
calls LogisticRegression model.fit at labs/relmini/baseline.py:77-78. The user's
no-training exception applies. No lead-verify run, training or config change.
Historical full-suite PASS belongs to the earlier builder, not this source.

integrity.json confirms the original frozen tests, every historical raw evidence
file, and pages.yml are unchanged; git diff --check passes and source is clean.
The missing checkout-relative bin/lead-link-skills warning is recorded separately
in skill-link-warning.md; the installed absolute linker exists and was not run
or copied. Installed TDD, codebase-design and verify-project guidance read directly.

## Package and remaining authority

Old README.md was committed. Old READY.json was intentionally untracked in the
original worktree after commit; the separate verifier preserved it externally
as evidence/verifier/original-builder-READY.json. No old builder report.md or
READY.json is present in the historical committed tree. The existing README
now says this explicitly. Historical evidence and verifier findings are intact.

R1 raw logs, helper scripts, integrity proof, this report and READY.json are
external task evidence under this directory, not committed repository artifacts.
READY.json is written last, binds the final SHA and hashes every other artifact.
No new canonical TDD replay or lead-guard admission was requested or performed.
No independent verifier/reviewer was launched by this builder; the parent owns
that role and acceptance. Hosted CI, push, merge and deployment: NOT_RUN.
No installs, spending, permission/profile/gate changes or shared-ledger rewrites.

LEAD_RESULT=READY_FOR_PARENT_VERIFIER
OK REL16-V1 and REL16-V2 fail closed
OK Pages fixtures 10
OK publication destinations 18497
NOT_RUN full verify-project: configured model fitting
NOT_RUN new canonical replay
NOT_RUN parent independent review
