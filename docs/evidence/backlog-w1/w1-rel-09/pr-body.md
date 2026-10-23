Flashcard integrity now checks unique IDs, the six required fields, and count agreement with the actual progress report. Canonical RED executed four tests with four assertion failures; identical GREEN executed four tests with zero failures, errors or skips. The real deck and report both count 103; raw grep counts 48 unquoted plus 55 quoted ID keys.

[Deterministic human proof](docs/evidence/backlog-w1/w1-rel-09/evidence.md), [manifest](docs/evidence/backlog-w1/w1-rel-09/manifest.json), [exact selected RED lines](docs/evidence/backlog-w1/w1-rel-09/red.log), [exact selected GREEN lines](docs/evidence/backlog-w1/w1-rel-09/green.log), [selected verification stdout](docs/evidence/backlog-w1/w1-rel-09/verification.log).

Reviewed source: `8803fdd68f99960148bf0a5f124f4948a6b4df9a`. RED: `f9ec7f75bcb1c75cb4a92a9abb2292a5bdea8451`. Source and canonical tests are unchanged by this evidence-only packaging. The independent source review is PASS. Final proof review and explicit RISK_FLAG binding remain PENDING the same verifier; no risk value or merge eligibility is asserted here. Acceptance, draft publication and actual final-head hosted CI remain PENDING main. No PR/CI success or links are invented.

Local full verification is PASS, exit 0, no timeout, clean reviewed source, 46 tests / OK. **Renderer limitation:** the deterministic manifest validator selects verification stdout only; it contains an expected negative-fixture `FAIL hash check` message. That stdout alone does not convey the passing verdict. Complete fresh status, stdout and stderr follow verbatim.

### Fresh source verification: status.json

```text
{
  "project": "relational",
  "root": "/tmp/lead-wt/w1-rel-09",
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
  "duration_s": 3,
  "started_utc": "20261023T184533Z",
  "source": {
    "head": "8803fdd68f99960148bf0a5f124f4948a6b4df9a",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}
```

### Fresh source verification: stdout.txt

```text
FAIL hash check:
  x: sha256 mismatch
```

### Fresh source verification: stderr.txt

```text
..............................................
----------------------------------------------------------------------
Ran 46 tests in 1.422s

OK
```

Raw directory: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184533Z`. Command: `/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/w1-rel-09`.

### Count audit: exact prior raw output

```text
{
  "evaluated_cards": 103,
  "unique_ids": 103,
  "duplicate_ids": [],
  "missing_or_empty_fields": [],
  "actual_report_flashcards_total": 103,
  "unquoted_id_lines": 48,
  "quoted_id_lines": 55
}
48
55
```

The evaluated JavaScript and actual report builder are called by the canonical tests. The raw counts above came from Python evaluation/counting plus `rg -c 'id:' assets/paper-deck.js` and `rg -c '"id":' assets/paper-deck.js`. A formatting-sensitive search omits 55 quoted keys; there is no genuine count disagreement.

### Independent negative probes: exact output

Actor `verifier:w1-rel-09`, pane `w1:p12`; artifact `/tmp/lead-wt/v-w1-rel-09/reviews/w1-rel-09-evidence/independent-checks.log`. SHA256 `8e5a31b2dd2fb3af64faeba938047b31ebb9a177742790c43a107d04d0869ee8`.

```text
{"evaluated_cards": 103, "unique_ids": 103, "actual_report_total": 103}
rg -c 'id:' assets/paper-deck.js => 48
rg -c '"id":' assets/paper-deck.js => 55
rejected id=None: required field id missing or empty at card 0
rejected id='': required field id missing or empty at card 0
rejected id='   ': required field id missing or empty at card 0
rejected absent id: required field id missing or empty at card 0
rejected paper=None: required field paper missing or empty at card 0
rejected paper='': required field paper missing or empty at card 0
rejected paper='   ': required field paper missing or empty at card 0
rejected absent paper: required field paper missing or empty at card 0
rejected year=None: required field year missing or empty at card 0
rejected year='': required field year missing or empty at card 0
rejected year='   ': required field year missing or empty at card 0
rejected absent year: required field year missing or empty at card 0
rejected lesson=None: required field lesson missing or empty at card 0
rejected lesson='': required field lesson missing or empty at card 0
rejected lesson='   ': required field lesson missing or empty at card 0
rejected absent lesson: required field lesson missing or empty at card 0
rejected front=None: required field front missing or empty at card 0
rejected front='': required field front missing or empty at card 0
rejected front='   ': required field front missing or empty at card 0
rejected absent front: required field front missing or empty at card 0
rejected back=None: required field back missing or empty at card 0
rejected back='': required field back missing or empty at card 0
rejected back='   ': required field back missing or empty at card 0
rejected absent back: required field back missing or empty at card 0
rejected duplicate: duplicate card id: animus-b18-context
rejected removed: deck count 102 differs from report count 103
rejected wrong report: deck count 103 differs from report count 104
rejected nonstring id: required field id must be a string at card 0
PASS: 28 disposable negative cases rejected; input unchanged
RED/GREEN identical: tests/flashcards/test_deck_integrity.py 12dc41c31e55e2e50e6f8653ea8107e0f9f6da4676fbec6ca649606ebf6f6ce6
RED/GREEN identical: tests/flashcards/__init__.py 3b2e0e8f3a6349808b70098fa09980e42c799d5da98a9155d2ae97641ef612b3
base/source unchanged: assets/paper-deck.js
base/source unchanged: scripts/progress_report.py
base/source unchanged: .github/workflows/pr-checks.yml
```

### Independent source review: verbatim snapshot

Artifact `/tmp/lead-wt/v-w1-rel-09/reviews/w1-rel-09.md`. SHA256 `5b29c10d8dc90aa02cbcd86a4ce14113494678829e2ccf34e8650d98254be945`. This source-review snapshot predates the final human-proof packaging; its pending statements describe that stage. The verifier must bind the final proof separately.

<details>
<summary>Exact independent source review</summary>

# REL-09 independent source and proof review

Reviewer actor: verifier:w1-rel-09. Counted verifier pane: w1:p12.
Builder actor: builder:w1-rel-09, pane w1:pZ. Review checkout: /tmp/lead-wt/v-w1-rel-09.
Reviewed source SHA: 8803fdd68f99960148bf0a5f124f4948a6b4df9a.
Base: 00f3dfd721dc66bbab3f9672705fd0f60fe8a416.
Original declared builder worktree: /tmp/lead-wt/w1-rel-09.

## Verdict

Source/risk review: PASS; no blocking source defect found within REL-09's ID, required-field and count-presence contract. Canonical replay now PASS after the single authorized retry. Delivery acceptance remains PENDING final human evidence and actual final-head draft PR CI. This is a separate verifier review, not a DONE acceptance report.

## Actual skills and launch gate

Applied /home/avist/agent-stack/lead/.agents/skills/blast-radius/SKILL.md (installed target /home/avist/agent-stack/vendor/pstack/plugins/pstack/skills/blast-radius/SKILL.md) through diff/consumer review and execution of the real checker with disposable mutations.
Applied /home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md through the actual lead-verify command and reading all three durable files.
Recording follows /home/avist/agent-stack/lead/docs/skills-enforcement.md. No /tdd, implementation, subagent, publication or acceptance was performed by this reviewer.

Launch command: /home/avist/agent-stack/lead/bin/lead-env relational check --worktree /tmp/lead-wt/v-w1-rel-09.
Actual PASS, source.head matches reviewed SHA, dirty=false; retained environment directory: /home/avist/agent-stack/lead/.lead/evidence/env/relational/1792780814628449258. It is a launch snapshot with five-minute expiry, not the builder's eventual acceptance environment.

## What changed and what could break

Only tests/flashcards and source evidence changed from base. Tests evaluate assets/paper-deck.js with Node, then call the actual scripts/progress_report.py build_report. tests/flashcards/__init__.py makes this directory discoverable by the existing recursive unittest command. The checker is test-local; the existing app does not import it. The shared PR workflow at .github/workflows/pr-checks.yml:44 discovers the new tests. This adds a Node runtime dependency to that check; Node was present locally. Hosted availability and actual final-head CI are not proven by local execution.

The key safety fact is that both formats of ID key survive actual JavaScript evaluation, and the independent report parser agrees on the real deck. Proven by execution (blast-radius level 4): 103 evaluated cards, 103 unique IDs, actual report total 103. Raw commands `rg -c 'id:' assets/paper-deck.js` and `rg -c '"id":' assets/paper-deck.js` return 48 and 55. Thus the earlier 48 observation is a formatting-sensitive subset, not a genuine report/deck disagreement.

Raw independent proof: reviews/w1-rel-09-evidence/independent-checks.log. It executes validate_deck from tests/flashcards/check_deck.py:6 and build_report from scripts/progress_report.py:79, rejects 24 missing/null/empty/whitespace required-field cases plus duplicate ID, removed card, wrong report total and nonstring ID (28 total), and confirms original cards remain unchanged. The required-field contract checks presence/nonblank strings, not a broader schema/type or semantic validity guarantee for year/lesson/paper/front/back. Such broader validation is outside this issue.

Cleared: original deck, report and shared workflow are byte-identical to base. Tests are byte-identical between RED f9ec7f75bcb1c75cb4a92a9abb2292a5bdea8451 and reviewed GREEN. RED commit contains only __init__.py and test_deck_integrity.py; the GREEN tests delta adds only pinned check_deck.py. Builder raw red.stderr shows four assertion failures, no import errors; these checks do not temporarily mutate teaching content. Direct source inspection and executed negative cases support the helper's duplicate/field/count rejection behavior.

## Project verification and human proof

Independent command: /home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/v-w1-rel-09.
Retained directory: /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T184036Z.
Read status.json, stdout.txt and stderr.txt: verdict PASS, exit_code 0, timed_out false, source.head 8803fdd68f99960148bf0a5f124f4948a6b4df9a, dirty false; stderr says Ran 46 tests / OK. stdout includes the expected negative-fixture `FAIL hash check: x: sha256 mismatch`; stdout alone is misleading and the PR must disclose that limitation.

Also read original builder /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T183545Z/{status.json,stdout.txt,stderr.txt}: same source SHA, clean, PASS exit 0, 46 tests / OK. These are local/project results, not canonical replay receipts or hosted CI.

Reviewed committed docs/evidence/backlog-w1/w1-rel-09/{evidence.md,red.log,green.log,pr-body.md} and original recovery/source-ready.md. Counts and RED/GREEN descriptions agree with raw logs and independent execution. Final committed human manifest/rendered non-UI proof is still pending. No UI was changed; screenshots/browser capture are not applicable. No PR or CI link exists in this reviewed evidence; lead owns publication and must supply actual final-head links.

## Canonical replay and remaining gates

Original canonical attempt is not PASS: recovery-w1-rel-09/replay.stderr records a 30-second git worktree-add timeout before RED execution. No accepted receipt was present at source review. Preserve that log. The original report binds /tmp/lead-wt/w1-rel-09 and its declaration; do not redeclare against this verifier checkout.

Lead signaled and this verifier performed exactly one supported canonical retry against the original builder worktree. It exited 0: `TDD replay RED/GREEN verified`. Read the original bound report, durable receipt and both raw logs. RED: run=4, failed=4, errors=0, skipped=0; GREEN: run=4, failed=0, errors=0, skipped=0. Receipt binds reviewed head, original RED, original canonical IDs, original declaration, intake and issue hashes. A subsequent `lead-guard tdd accepted` check exited 0 with `TDD accepted receipt (no execution)`; this was receipt validation, not another replay. Original builder head remains the supplied SHA and git status is clean. No source, declaration or pin was changed.

Shared retry outputs and receipt (absolute):
- /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/verifier-retry.stdout
- /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/verifier-retry.stderr (empty; command succeeded)
- /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/tdd-replay.json
- /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/tdd-replay/red.log
- /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/tdd-replay/green.log
- /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/verifier-retry-actor.json

The first timeout remains in replay.stderr; success does not erase that failed attempt.

Before acceptance: canonical receipt is now verified with original declaration binding; review final human proof, fresh original-builder environment, preserved source/test identity in any evidence-only commits, and green actual draft PR CI. Lead validates and releases the counted verifier slot.

</details>

### Preserved failed attempt and successful retry

Original canonical command: `/home/avist/agent-stack/lead/bin/lead-guard tdd validate --report /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/source-report.json --receipt /home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/tdd-replay.json --worktree /tmp/lead-wt/w1-rel-09`. Original exit 1 before RED execution; exact output:

```text
Command '['git', '-C', '/tmp/lead-wt/w1-rel-09', 'worktree', 'add', '--detach', '/tmp/lead-tdd-9q4sd8rt/tree', 'f9ec7f75bcb1c75cb4a92a9abb2292a5bdea8451']' timed out after 30 seconds
```

The separate verifier performed the one lead-authorized retry; exit 0, exact stdout:

```text
TDD replay RED/GREEN verified
```

Receipt: `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/tdd-replay.json`. Retry actor/provenance: `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/verifier-retry-actor.json`. Full canonical logs remain under `/home/avist/agent-stack/lead/.lead/tasks/backlog-w1/recovery-w1-rel-09/tdd-replay`. Original timeout and historical Git read-only blocked evidence remain preserved in the task recovery/blocked-snapshot folders; success does not erase those attempts.

### Scope and limits

Tests and checker are confined to tests/flashcards; original deck, progress report and shared workflow are unchanged. Independent probes rejected 28 disposable cases and confirmed inputs unchanged. Required fields check presence/nonblank strings; broader field types and teaching-content semantics are outside this contract. Node is exercised locally; hosted runner availability requires actual final-head CI. These are local software checks, with no experiment, provider, browser, screenshots or production claims. No local installation, new spend, source changes, publication or merge was performed for this proof continuation. Final verifier review, its explicit RISK_FLAG and main acceptance/publication remain required.

### Bounded admission correction (PROOF_READY2)

The REL-05 rejected acceptance was diagnosed read-only using its preserved receipt/logs. Its actual mismatches were elapsed-time `Ran 8 tests in 0.014s` and `Ran 8 tests in 0.074s` lines. REL-09 selected the same unstable elapsed-time pattern plus temporary traceback paths. Only this slice's selected RED/GREEN proof is corrected: exact numerical LEAD_RESULT and FAILED/OK summaries survive fresh replay; canonical argv and tests remain unchanged. Complete original logs, original proof and candidate are retained under task recovery-w1-rel-09/correction2/original and the original canonical logs remain in tdd-replay/. Original timeout, blocked audit and the REL-05 failed receipt remain preserved. No replay or acceptance was executed by this builder correction.

The builder performed a fresh real /verify-project run against the clean reviewed code SHA in the original reserved worktree, then restored the evidence branch. The latest verify attestation belongs to actual builder:w1-rel-09 / w1:pZ and binds that builder-run stderr. The independent reviewer should record only the review step at final evidence recheck. Fresh complete builder verification follows; the earlier snapshot above remains historical.

#### Refreshed builder verification: status.json

```text
{
  "project": "relational",
  "root": "/tmp/lead-wt/w1-rel-09",
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
  "duration_s": 2,
  "started_utc": "20261023T190139Z",
  "source": {
    "head": "8803fdd68f99960148bf0a5f124f4948a6b4df9a",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}
```

#### Refreshed builder verification: stdout.txt

```text
FAIL hash check:
  x: sha256 mismatch
```

#### Refreshed builder verification: stderr.txt

```text
..............................................
----------------------------------------------------------------------
Ran 46 tests in 1.377s

OK
```

Raw verification directory: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261023T190139Z`. The manifest verification stdout selection remains unchanged because both raw stdout files contain the same exact stable lines. The passing verdict and 46-test summary remain fully embedded above.

Final evidence-diff review for the new delivery is PENDING. The previous explicit verifier risk binding applies to the prior delivery; no new risk flag or merge eligibility is inferred. Main owns prompt acceptance with fresh environment and final reviewer binding, plus managed draft publication and actual hosted CI.
