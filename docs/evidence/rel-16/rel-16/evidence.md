# REL-16 package for Piotr review

Source: `86fe6e7284421f128686da00240aeddb5595cbab` (verifier observed clean).
Separate verifier: `w1:p3X`, read-only checkout `/tmp/lead-wt/night-vrel16-r1`.
Source verdict: **PASS**. **RISK_FLAG: YES**. Piotr review is pending.
This builder packages retained evidence; no new review or verification is claimed.
Delivery payload SHA is bound in `delivery.json` and the local `PR-body.md` draft;
the exact last package commit is bound by the external piotr-review READY receipt.

## Behavior and retained evidence

The bounded Pages publication parser fails closed with exit 2 and no success JSON
for quoted wildcard syntax, wildcard removals and normalized public-root removals.
REL16-V1 and REL16-V2 are resolved per the independent verifier.
Literal copies, unquoted copy globs, ordinary quoted filenames and literal subtree
removal remain supported. No source, test or workflow changes accompany this package.

| Evidence | Retained result | Raw proof |
| --- | --- | --- |
| Independent focused Pages fixtures | 10 methods PASS, exit 0 at exact source SHA | [status](proof/evidence/r1/verifier/focused.json), [raw unittest output](proof/evidence/r1/verifier/focused.stderr.txt) |
| Independent Bash/GNU cp inventory | 18,497 actual = 18,497 generated = retained inventory; zero membership differences | [parity](proof/evidence/r1/verifier/parity.json), [comparison](proof/evidence/r1/verifier/compare.stdout.json), [inventory](proof/evidence/r1/verifier/actual-files.json) |
| Separate source review | PASS, RISK_FLAG YES | [recorded verdict](proof/verdict-recorded.json), [READY](proof/evidence/r1/verifier/READY.json), [report](proof/evidence/r1/verifier/report.md) |
| Local TDD history | Genuine R1 RED/GREEN retained, inspected statically by verifier | [static history](proof/evidence/r1/verifier/static-history.json), [retained audit](proof/evidence/r1/verifier/retained-audit.json) |

All copied raw files and original paths have SHA-256 hashes in [manifest.json](manifest.json)
and clickable links in [proof-index.md](proof-index.md). Existing builder evidence
remains under `evidence/builder/`, with frozen hashes also inventoried in the manifest.
No proof script was executed for this packaging task.

## Parser limitations and risk

This is a bounded interpreter for the current workflow, not general Bash.
Quoted literal wildcard filenames are deliberately unsupported; arbitrary shell
and destination renames are unsupported. The frozen default plan retains base
503704f3 policy. Equality compares destination file membership, not published
bytes, empty directories, permissions, symlink targets or deployment correctness.
The Bash oracle omitted exactly two existing HTML byte-version mutations.
Manifest diagnostics cover direct slug HTML and labPath references only, not
transitive links; missing/omitted references are reports, never additions.
The workflow does not invoke this generator. No browser/deployment validation
or hosted CI result is claimed. Piotr must review these bounded semantics and
the remaining gates before any acceptance decision.

## Honest provenance and correction history

The historical committed package contained README.md and raw builder evidence,
not builder READY.json or report.md. Original READY was intentionally written
after commit and untracked at
`/tmp/lead-wt/night-rel-16/docs/evidence/rel-16/rel-16/evidence/builder/READY.json`.
Its preserved external copy is now copied under
[proof/evidence/verifier/original-builder-READY.json](proof/evidence/verifier/original-builder-READY.json)
with SHA-256 `0a3b386c62d22f560a107a4958b8bee43ab8dec342c37d33090e31280f1c3e64`.
This new copy does not make the original receipt historically committed.
Earlier reports retain their then-current packaging and review status; this
new package supplies readable evidence without rewriting those claims.

Original RED `9e09a39b909a445b420ca4ff0fbcb5fca8c7aff1` had six assertion
failures across seven methods; GREEN was `1ba8e647d4ef15eb573b8caa13033cdad5e3b9ff`.
All seven tests preceded one implementation commit, which does not meet the
installed one-test-at-a-time cadence. Historical canonical PASS remains bound
to that older implementation and is not R1 acceptance.

| R1 behavior | Clean tests-only RED | Production GREEN | RED assertions / runner errors |
| --- | --- | --- | --- |
| Quoted wildcard | b5fdb10040d19b5230c35ce8dd161e70cd6eb724 | 5da4d654763e1fd5364a20c908a5446b07f185c5 | 21 / 0 |
| Wildcard removal | 27985b26853357f7812896eef5aaab31a521b2e1 | 26fad3df990bce57b2e8692940d4a1453c54824f | 12 / 0 |
| Corrected root removal | c1a98c2e9e6067aed6dc1197f20563e751f6cf57 | dd7580a7e283c4ceff0aca6a9a41a661e47fd0c2 | 12 / 0 |

Intermediate GREEN logs bind preceding RED HEADs with dirty production fixes,
not clean GREEN-SHA runs. Final builder/verifier focused receipts bind the clean
source SHA. Original frozen tests and historical artifacts remain unchanged.

Invalid fixture commit `3a628006af7cf984832ef3f187bf79355d54a257` recorded ten
failures: nine parser assertions and one invalid expectation that GNU rm accepts
`public/.`. Separate correction `c1a98c2e9e6067aed6dc1197f20563e751f6cf57`
used `public//` before the root production fix and established twelve genuine
failures. [Original raw failure](proof/evidence/r1/v2-root-red-committed.stderr.txt)
and [correction diff](proof/evidence/r1/verifier/fixture-correction.diff.txt) remain intact.
The verifier's initial inventory harness failed before Bash launch (argument
size); [attempt proof](proof/evidence/r1/verifier/parity-attempt-1.json) remains.
The same script subsequently passed through stdin once; this was not canonical replay.

## Exact remaining gates

- `/verify-project`: **NOT_RUN** at R1. Configured discovery reaches
  `baseline.fit_predict` and `LogisticRegression.fit`; model fitting is prohibited.
  Exit code is null, not zero. [Retained explanation](proof/evidence/r1/verifier/verify-project-not-run.json).
  Historical 74 PASS applies to older source only and is not applicable to R1.
- `/tdd`: genuine local cycles and frozen history retained; **R1 canonical NOT_RUN**.
  No replay, admission, gate-policy change or new TDD conformance claim.
- Piotr risky-package review: **PENDING**, with separate-verifier source PASS retained.
- Acceptance: **NOT_DONE**, lead/Piotr owned. Source PASS and packaging do not
  establish configured software verification, DONE, acceptance or CI.
- Hosted CI, push, merge and publication: **NOT_RUN**. No remote PR was created.

Operational packaging checks and final commit hashes are recorded outside frozen
inventories in the external piotr-review handoff. No subagents, installs, spend,
training, downloads, linker copies or permission/profile changes were performed.
