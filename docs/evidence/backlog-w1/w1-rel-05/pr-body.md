Manifest drift previously had no read-only integrity check. Eight tests now check the real lesson inventory, referenced labs/notebooks, unique numeric and bridge IDs, reverse coverage, and authored IDs 1–200 with the documented 179 exception. Removing an entry from a disposable manifest is rejected. Teaching assets are unchanged.

Reviewed source: `030df3dc3452b590d52cce14a04476fbef4ca8cc`. Canonical tests are byte-identical in RED `9179723db0de643b65fe943ad181f724d886ed62` and GREEN. RED ran eight tests with ten assertion failures (subtests), zero errors/skips; GREEN ran the same eight with zero failures/errors/skips. Configured verification ran 50 tests: PASS, exit 0, dirty false.

[Rendered human evidence](docs/evidence/backlog-w1/w1-rel-05/evidence.md) · [Exact raw RED lines](docs/evidence/backlog-w1/w1-rel-05/red.log) · [Exact raw GREEN lines](docs/evidence/backlog-w1/w1-rel-05/green.log) · [Verification stdout lines](docs/evidence/backlog-w1/w1-rel-05/verification.log) · [Manifest with retained verification and independent review](docs/evidence/backlog-w1/w1-rel-05/manifest.json)

The deterministic renderer shows verification stdout only. Its `FAIL hash check: x: sha256 mismatch` is an expected negative fixture; the actual passing suite summary is on stderr. Full stdout, stderr, and status follow verbatim and are also retained in manifest.json. Raw phase files are exact selected nonempty canonical lines, not fabricated replacement logs.

<details>
<summary>Passing configured verification: raw stdout, stderr and status</summary>

Command: `/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/w1-rel-05`
Evidence directory: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T183541Z`

stdout:
```text
Deleted-entry mutation rejected:
unlisted lesson: 0001-single-table-assumption.html
authored gap: 1
FAIL hash check:
  x: sha256 mismatch
```

stderr:
```text
..................................................
----------------------------------------------------------------------
Ran 50 tests in 1.606s

OK
```

status.json:
```json
{
  "project": "relational",
  "root": "/tmp/lead-wt/w1-rel-05",
  "argv": [
    "/home/avist/Projects/relational/.venv/bin/python",
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-p",
    "test_*.py"
  ],
  "timeout": 600,
  "exit_code": 0,
  "timed_out": false,
  "verdict": "PASS",
  "duration_s": 4,
  "started_utc": "20261023T183541Z",
  "source": {
    "head": "030df3dc3452b590d52cce14a04476fbef4ca8cc",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}
```

</details>

<details>
<summary>Actual independent source review — verifier:w1-rel-05, pane w1:p11</summary>

Original artifact: `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/reviews/w1-rel-05.md`. SHA256: `ff2cc90ec483c41bd17924b786776a27fb4b592b1e26e4bc852e6a9cadc2f785`.

```text
# Independent review: backlog-w1 / w1-rel-05

VERDICT: PASS
RISK_FLAG: not-risky
Actor: verifier:w1-rel-05
Actual launched pane: w1:p11 (spawn record v-w1-rel-05.json; inherited LEAD_SELF_PANE=w1:p1 is the lead bridge value, not this verifier pane).
SourceSHA: 030df3dc3452b590d52cce14a04476fbef4ca8cc
Builder: builder:w1-rel-05; original worktree /tmp/lead-wt/w1-rel-05.
Verifier worktree: /tmp/lead-wt/v-w1-rel-05.

This is a source-review PASS, not final DONE acceptance. No implementation, TDD execution, publication, nested reviewer, merge or slot cleanup was performed. Only independent evidence files and requested skill attestations were written.

## Change and blast radius

Base 00f3dfd721dc66bbab3f9672705fd0f60fe8a416 to reviewed head adds only tests/manifest/__init__.py, test_manifest.py and check_manifest.py. Eight unittest cases cover manifest entry paths, reverse lesson coverage, referenced notebooks/Python labs, numeric/bridge ID uniqueness, authored numeric gaps and the documented 179 exception. The checker reads JSON and filesystem paths; it does not write teaching content.

The safety fact is proven by real execution (blast-radius level 4): a disposable manifest still resolves references against the real repository and detects removed authored entries without changing content. Independent probe.py imports the exact committed checker. Deleted IDs 1, 200 and B24 all report their unlisted HTML files; IDs 1 and 200 additionally report authored gaps. A missing notebook reports its exact path. All 450 referenced content files hash identically before/after. Baseline inventory: 228 entries, 228 unique IDs, 228 lesson HTML files, 220 nonempty lab references.

Source references: tests/manifest/check_manifest.py:7 defines range 1–200 minus 179; :23 checks duplicates; :30 checks entry files; :33 checks lab references; :35 checks reverse coverage; :38 checks authored gaps. CURRICULUM.md:422 explicitly labels 179 planned. scripts/update-manifest.py:73 generates the same flat lessons/*.html inventory; assets/home.js and assets/notebooks.js consume manifest entries/labPath. None of these consumers or content files changes. Existing .github/workflows/pr-checks.yml recursively discovers tests; the new package marker makes this suite discoverable. No dependencies or workflow edits.

Cleared: silent entry deletion, endpoint range loss, bridge reverse-coverage loss, missing notebook/Python references, numeric/bridge duplicate IDs, and accidental content writes. No confirmed source defects. Malformed schema may raise an exception rather than return an error list, which still blocks unittest; broad schema validation, notebook validity, renderer output and security containment are outside this ticket. No UI change; browser capture is inapplicable.

## Actual independent verification

Launch environment gate: lead-env relational check --worktree /tmp/lead-wt/v-w1-rel-05 exited 0, verdict PASS, source.head matching SourceSHA, dirty false. Durable evidence: /home/avist/agent-stack/lead/.lead/evidence/env/relational/1792780778031943546. This is launch readiness, not an assertion of future receipt freshness.

Focused /verify-project: temporary configuration changes only the root and argv to the issue command; config retained as focused-config.json. lead-verify relational --worktree /tmp/lead-wt/v-w1-rel-05 exited 0, verdict PASS, source.head matching SourceSHA, dirty false; eight tests OK, no skips. Evidence: /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T183953Z/{status.json,stdout.txt,stderr.txt}. Exact output copies accompany this artifact.

Configured project /verify-project: original Lead configuration, same worktree; exited 0, verdict PASS, source.head matching SourceSHA, dirty false; 50 tests OK, no skips. Evidence: /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184014Z/{status.json,stdout.txt,stderr.txt}. stdout contains the expected negative-fixture text `FAIL hash check: x: sha256 mismatch`; actual passing unittest summary is on stderr, and status.json records exit 0. Do not render stdout alone as a failing suite.

Independent probe command: PYTHONDONTWRITEBYTECODE=1 /home/avist/Projects/relational/.venv/bin/python docs/evidence/backlog-w1/w1-rel-05/independent-review/probe.py. Exit 0; raw probe.stdout retained, probe.stderr empty. No installations, network or experiments.

## Original builder proof review

Read source-ready/w1-rel-05/source-ready.md and original canonical recovery-tdd.json with both full raw logs under /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/blocked-snapshots/w1-rel-05/recovery-tdd/. RED is 9179723db0de643b65fe943ad181f724d886ed62: tests/package marker only, no helper. GREEN adds only the declared check_manifest.py. Independent Git blob comparison confirms identical test_manifest.py and __init__.py bytes. Canonical commands and eight IDs are identical. All five receipt artifact SHA256 values independently match.

Original raw RED: run 8, failed 10 (subtests), errors 0, skipped 0. Every failure is AssertionError: Manifest integrity checker is not implemented. Original raw GREEN: run 8, failed 0, errors 0, skipped 0, OK; deletion output reports unlisted lesson 0001 and authored gap 1. The explicit missing-checker assertion is allowed by the task's test-only RED seam; no import-error RED or content mutation. Initial missing package-marker correction and original logs are retained in recovery-red-note.md; this does not alter the reviewed immutable RED/GREEN tests.

Supported receipt validation command (no replay or redeclaration): /home/avist/agent-stack/lead/bin/lead-guard tdd accepted --report /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/blocked-snapshots/w1-rel-05/recovery-source-draft.json --receipt /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/blocked-snapshots/w1-rel-05/recovery-tdd.json --worktree /tmp/lead-wt/w1-rel-05. Exit 0: TDD accepted receipt (no execution). Original builder worktree binding preserved.

## Lead delivery gates remaining

At the reviewed source head, committed docs/evidence/backlog-w1/w1-rel-05 does not yet exist. Builder handoff explicitly defers human-proof manifest/evidence/PR body and evidence-only commit until after this review. Draft PR URL and final-head hosted CI are NOT VERIFIED and must be supplied by Lead before final acceptance. Preserve code/test identity through evidence-only delivery and disclose stdout/stderr behavior. This review makes no DONE acceptance claim.

Skill sources actually followed: /home/avist/agent-stack/vendor/pstack/plugins/pstack/skills/blast-radius/SKILL.md; /home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md. Actual outputs: this independent review, probe stdout, and nonempty passing project stderr. Registered through lead-task skill-record using actual verifier actor/pane and SourceSHA.
```

</details>

The source reviewer returned PASS / not-risky, independently ran all 50 tests and verified the original canonical receipt. That source review preceded this proof packaging. The same verifier must bind the final proof before main acceptance. Hosted draft PR and final-head CI remain pending main publication; neither is claimed by this package. No UI changes, browser capture, installations, experiment or provider execution occurred. Malformed manifest schema still fails the check by exception; notebook validity and broad schema/security validation are outside this slice. Planned lessons 201–240 are excluded by the approved scope.

<details>
<summary>Historical blocked attempt — preserved, superseded by current evidence</summary>

The original attempt could not write Git metadata and encountered an earlier checker gate conflict. Those historical blockers were repaired by lead before this recovery. Original `source-ready.md`, `blocked.json`, and `red.log` remain byte-for-byte at `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/blocked-snapshots/w1-rel-05/recovery-preserved-original/`; original raw recovery RED stdout/stderr and missing-discovery-marker note remain under `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/blocked-snapshots/`. Initial recovery RED `fc7d6ca4` omitted the package marker; before helper implementation its tests-only commit was corrected to `9179723d`, retaining identical canonical test bytes and original failure logs. No declaration was replaced, no assets were altered, and no extra baseline exception was added. Historical NOT_RUN statements describe that stopped attempt, not the completed current local verification.

</details>
