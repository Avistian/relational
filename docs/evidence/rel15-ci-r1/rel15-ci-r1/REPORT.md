# PR35 bounded CI repair: BLOCKED at canonical replay

Builder `builder:rel15-ci-r1`, Codex `gpt-6.1-sol` high, actual Herdr pane `w1:p5M`, tab `w1:t5M`. No agents or workers launched. Worktree `/tmp/lead-wt/build-rel15-ci-r1-20261025`, branch `lead/build-rel15-ci-r1-20261025`. Started approximately 2026-10-25T12:38Z; handoff 2026-10-25T12:50:08.522729+00:00; within 30-minute timebox.

The source repair passes direct behavior tests and configured software verification. Delivery is BLOCKED because the supported canonical TDD loader cannot import the frozen regression. The builder missed this loader requirement before freezing RED. No canonical receipt or acceptance is claimed. No retry, regression rewrite, guard/profile change, push, merge, browser/model run, install, download, fixture remote, global Git configuration change or spend occurred.

## Exact provenance

- Builder base: `15fa7dc501cab9eaf7ac551e6d4fe5bd1b60792b`.
- Test-only direct RED: `05f90c4c365998ecd7b652bfbabec1523254231e`; 2 assertion failures, 0 errors, 0 skips. No implementation edits in this commit.
- Minimal GREEN / tested software source: `dad549667c767ef4dceee631f60afbd58b6e3b47`.
- Evidence bundle delivery: `6f5160ad9fb5de09534d4b2bc58efe1f9ae96fb0`; only own evidence added after software source.
- Final marker delivery: the evidence-only commit containing this REPORT and READY.json, resolved by `git rev-parse HEAD` and the task-local `reports/rel15-ci-r1.builder-handoff.json`. No self-referential commit hash is embedded in its own files.
- Pre-RED declaration succeeded and is preserved in [implementation-declaration.json](implementation-declaration.json). Existing `tests/browser_checks/test_output_paths.py` alone is declared implementation.

## Diagnosis and counterfactuals

The supplied [raw CI log](pr35-ci-failed.log) shows run 40192853147 failing `OutputPathsTests.test_generated_directories_ignored_and_tracked_pngs_retained` at `git ls-tree -r --name-only 503704f -- reviews`, exit 128, `fatal: Not a valid object name 503704f`. History availability masks the portability defect: the original unchanged unittest passes locally with the old object present, then errors in a one-commit fixture before reaching retention assertions. Removing a fixture PNG produces the same legacy history error; that error does not prove deletion rejection.

[diagnose.py](diagnose.py) loads the actual base-commit unittest bytes with their recorded SHA256. Its [raw stdout](diagnosis.stdout) and [raw stderr](diagnosis.stderr) preserve the missing-history tracebacks independently of the outer regression assertions. It exits 0 only after checking all seven expected outcomes:

| Actual public unittest and context | Failures | Errors | Skips |
|---|---:|---:|---:|
| Legacy, full history, intact | 0 | 0 | 0 |
| Legacy, single commit, intact | 0 | 1 | 0 |
| Repaired, single commit, intact | 0 | 0 | 0 |
| Legacy, single commit, worktree deletion | 0 | 1 | 0 |
| Repaired, single commit, worktree deletion | 1 | 0 | 0 |
| Legacy, single commit, index deletion | 0 | 1 | 0 |
| Repaired, single commit, index deletion | 1 | 0 | 0 |

Fixture files contain synthetic path-existence bytes, not replacement screenshot content. No original PNG is altered. Each fixture has exactly one commit, no remotes and no historical object. The exact public unittest runs with controlled ROOT; Git subprocess behavior is real, not mocked.

The complete frozen inventory derives from real `503704f30fa28e9c8896302d1423f0fcca9b539c` tree output: 1,473 paths, including 590 PNGs. Raw inventory SHA256 is `d3cb9ecfa18a3ecef4bee14ff93e2442f41780f3e7a92cf4fda98d67b13420d3`; [provenance](baseline-provenance.json) and [original bytes](original-review-inventory.txt) are retained. The Python inventory reconstructs these exact bytes. Every original review path remains expected, including non-PNGs. Neither expected list nor subset/file assertions was weakened, and there is no missing-history skip. The source fix imports this frozen list and replaces only the historical ls-tree baseline assignment.

## Test evidence and limits

- [Direct RED](red.stderr): `python -B -m unittest tests.browser_checks.test_history_free_retention -v`, 2 runs, 2 behavior assertion failures, 0 errors, 0 skips, exit 1. The public test's legacy exception is collected as its outcome and causes the outer assertion to fail. This is direct RED, not canonical replay acceptance.
- [Focused GREEN](green.stderr): 9 tests including both new regressions and all 7 existing tests, 0 failures/errors/skips, exit 0.
- [Configured verification](verification/status.json): `/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/build-rel15-ci-r1-20261025`, PASS, exit 0, source `dad549667c767ef4dceee631f60afbd58b6e3b47`, `dirty:true`. The only dirty entry was the builder's then-untracked own evidence directory ([captured status](source-status.stdout)). [Actual stderr](verification/stderr.txt) reports 76 tests, OK, no skips. The supplied hosted log reports 99 tests and 2 skips; these are distinct runs/source contexts and are not conflated.
- Initial environment check PASS was at RED head with a dirty implementation edit; [its status](initial-environment-status.json) is preserved as startup evidence only, not current-source environment acceptance.
- [Frozen hashes](frozen-hashes.json): regression and inventory remain byte-identical to RED. [Scope](scope.stdout) contains only the three tests/browser_checks files. [Preservation check](preservation.exit) exit 0 proves scripts, PNG/review tree, workflows, ignore rules and original evidence paths were unchanged between base and software source.

A pre-freeze fixture setup attempt used command-level Git config and was rejected by the managed wrapper; [raw rejection](fixture-setup-failure.txt) and [setup-only test errors](setup-attempt.stderr) are retained. It was corrected before frozen RED by supplying fixture author/committer identity in environment variables and using `--no-gpg-sign`. No profile/global Git config or fixture remotes were changed.

## Blocking canonical result and remaining gates

Exactly one supported `lead-guard tdd validate` attempt used [this report](tdd-report.json) and [this command](canonical-command.json). It exited 1: `TDD FAIL: runner did not report test outcomes`. The loader assigns the synthetic module name `lead_replay_0`; frozen regression line 11 (`from . import test_output_paths`) raises `ImportError: attempted relative import with no known parent package`. See [raw canonical RED import traceback](canonical-red-import-error.log) and [guard stderr](canonical.stderr). No tests were run in that replay; no GREEN replay or receipt exists. This import failure is not relabeled as genuine canonical RED.

The current authority requires the selected regression to remain immutable and prohibits guard changes. Accordingly, the builder stopped without a second repair or replay. The lead must decide how to clear the runner-compatible regression gate while preserving this history. Independent read-only `/blast-radius` review is NOT_RUN and must be launched by the lead, not this builder. Parent update of existing PR35 without force, exact-head hosted CI, and merge are NOT_RUN/pending after all preceding gates pass. This handoff is not acceptance or permission to publish.

Skill invocation history records actual builder `/tdd` (blocked) and `/verify-project` (PASS), with source SHA, actor and actual pane. No review invocation is fabricated. [Artifact hashes](artifact-hashes.json) cover the evidence bundle before final markers; REPORT.md and READY.json were written last.
