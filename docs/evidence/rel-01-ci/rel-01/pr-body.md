PRs previously had no validation. Add a read-only pull_request workflow running all five REL-01 checks on ubuntu-latest, with explicitly listed check dependencies installed only on disposable GitHub Actions runners. Pages deployment stays in its existing workflow.

[Canonical RED/GREEN evidence (verification stdout includes an expected failing fixture)](docs/evidence/rel-01-ci/rel-01/evidence.md) · [RED](docs/evidence/rel-01-ci/rel-01/red.log) · [GREEN](docs/evidence/rel-01-ci/rel-01/green.log) · [verification stdout](docs/evidence/rel-01-ci/rel-01/verification.log).

Reviewed source: `410644c99e1fb18cdcd1e14671d2485fa7055737`. Canonical regression: before 1 assertion failure, after 0; identical tests, zero errors/skips. Full local verification: exit0, clean source, 41 tests passed. The verification stdout's `FAIL hash check` is the intentionally invalid-hash fixture; the unittest result on stderr is `OK`, retained below. GHA has no local RelBench cache, so the existing real-cache test explicitly skips there; no data download or real experiment is run. Existing synthetic unit tests fit small models as part of the required unittest command. These are CI/software checks, not reproduction results.

YAML parse: `{'pull_request': True, 'jobs': ['checks'], 'permissions': {'contents': 'read'}}`; required-command grep count: `5`.

Exact local CLI commands used the existing main .venv via PATH (no local installs):

```text
$ python3 labs/_test_b05_source_secrets.py
....
----------------------------------------------------------------------
Ran 4 tests in 0.405s

OK

exit_code=0
```

```text
$ python3 scripts/refresh_lesson_visuals.py --check
{"lessons": 228, "stories": 61, "changed": 0}

exit_code=0
```

```text
$ python3 labs/_check_pages_checkout.py
PASS: complete Pages build from Git index (no ignored workspace files)

exit_code=0
```

```text
$ python3 scripts/search_index.py --check
search index up to date

exit_code=0
```

```text
$ python -m unittest discover -s tests -p 'test_*.py'
FAIL hash check:
  x: sha256 mismatch
.........................................
----------------------------------------------------------------------
Ran 41 tests in 1.225s

OK

exit_code=0
```

Full lead-verify stderr:
```text
.........................................
----------------------------------------------------------------------
Ran 41 tests in 1.610s

OK
```

Delivery: draft only, human merge. Incremental USD0; model cost unmeasured. Independent source/local review PASS: verifier:rel-01 (native pane w1:pR), all five checks and 41 tests passed. The deterministic evidence renderer is restricted by the existing validator to verification stdout; use the full stderr above for the passing suite result. Actual Actions URL/head/conclusion are retained in the task delivery report.

<details>
<summary>Retained independent review artifact (source/local scope; hosted CI pending)</summary>

# REL-01 independent verifier: PASS (source/local scope)

Reviewer/actor: `verifier:rel-01`; actual exported pane: `w1:pR`.
Task/slice: `rel-01-ci/rel-01`.
Source: `410644c99e1fb18cdcd1e14671d2485fa7055737`.
Role: separate verifier, read-only source. No implementation, TDD implementation, extra reviewer, subagents, installs, spend, push, PR creation or merge. This is a review artifact, not a DONE acceptance report.

## Verdict and scope

RISK_FLAG: not-risky

Risk judgment applies to source `410644c99e1fb18cdcd1e14671d2485fa7055737`: the bounded CI-only change passed local checks, uses read-only permissions, and leaves Pages unchanged. Evidence limitation: deterministic rendered `evidence.md` is gate-bound; builder plans committed supplementary raw stderr/status links to explain the expected negative-fixture stdout. That supplementation is pending, not verified by this amendment. Hosted-CI limitation: draft PR and actual remote Actions execution remain pending; this flag does not establish hosted dependency setup or CI success and grants no merge authority.

PASS: inspected source meets the five-command PR-workflow contract. Independent environment check, real unittest verification (41 tests), and all four other exact CLI checks passed. No source changes requested. Task delivery remains pending draft PR and actual final-head Actions job URL/head/conclusion; remote CI is NOT VERIFIED here. No empty-check success is inferred. Lead owns validation and counted-slot cleanup.

## Blast radius: safety fact and proof

The safety fact is that this is a separate PR-only validation job with read-only repository permission and no deployment action, while existing Pages deployment bytes stay unchanged. Proof reached skill level 4 (real code run): parsed the actual workflow with installed PyYAML 6.0.3, asserted exact trigger `pull_request`, `contents: read`, one hosted job, no job conditions/environments/permission override, credentials persistence false, only checkout/setup actions and six run steps (one dependency setup plus five exact checks). Compared Pages bytes against base `c200e331`: identical. Ran the commands against the actual checked-out source. See raw proof below.

References: `.github/workflows/pr-checks.yml:3` trigger; `:6` permission; `:10` hosted job and timeout; `:14` checkout; `:23` explicit runner-only dependencies; `:27` through `:40` all five commands. Diff base..source contains only this workflow and tests/ci; no Pages/product changes. `labs/_check_pages_checkout.py:20` exports the index to a disposable directory and executes only the existing Build site block, not deploy steps.

## TDD correctness inspected

Test-only commit `619d164c` precedes the initial implementation. Canonical RED is the later test-only correction `750d863b29d63117cb05443dcf0fff18da73afeb`, after reverting initial workflow; final workflow commit is `410644c`. Independently compared canonical RED test bytes to final source: identical, and RED has no workflow. Inspected serial replay receipt `drafts/rel-01-preview.tdd.json` plus its full raw logs: RED runs 1 / failures 1 / errors 0 / skips 0 for absent workflow; GREEN runs 1 / failures 0 / errors 0 / skips 0. Independent final unittest run also executes the contract. No new TDD run or code slice was authored by this verifier.

## Dependencies and least privilege

Five direct check dependencies are explicitly pinned in workflow line 25; installed versions used locally match those four environment imports plus PyYAML 6.0.3. They cover YAML, numpy/pandas, sklearn baseline imports and parquet IO. Python 3.12 is selected; existing Node/shell/git are assumed on ubuntu-latest. No local installation was attempted. Workflow has no secrets expression, deployment step, environment, token persistence, custom/paid runner, training command or data download command. Existing required unit tests perform tiny synthetic model fits; these are software fixture tests, not a research run. The cache-dependent real-data test is explicitly skipped when its preexisting local cache is absent (`tests/relmini/test_relmini.py:103`), so hosted test totals may include that skip.

## Explicit risks and limits

1. Hosted dependency resolution, action execution and 10-minute job duration remain unproven locally (workflow lines 11–25). Likelihood unknown until first Actions run; impact: CI cannot pass. Cheapest remaining check: draft PR's actual `checks` job on final source head. No remote success claimed.
2. Human-proof readability, confirmed: builder's uncommitted `docs/evidence/rel-01-ci/rel-01/verification.log:1` contains only `FAIL hash check: x: sha256 mismatch`, the expected negative fixture. `evidence.md` repeats that without the passing unittest stderr, whereas `pr-body.md` explains it and includes `Ran 41 tests ... OK`. A reader following the evidence link alone can mistake success for failure (moderate likelihood, proof/review confusion). Lead/builder should retain the actual passing stderr/exit alongside stdout before publishing; do not replace or hide the expected negative output. Manifest hashes currently match raw files. These artifacts remain uncommitted and are not covered by this source SHA; reviewer metadata is a builder-prepared field, not proof this verifier authored them.
3. Regression-test scope: the test itself does not reject job-level `if` or arbitrary deployment shell commands. Current YAML has neither, independently checked. Low immediate risk on this source; future workflow edits still need review. No speculative test additions requested.
4. Audit-script false positive retained transparently: the first ad-hoc proof assertion rejected substring `secrets.` in legitimate filename `_test_b05_source_secrets.py`. Corrected only the ephemeral verifier assertion to match actual `${{ secrets... }}` expressions; corrected proof passed. No source edits.

## Cleared / explicit skips

Pages unchanged; workflow has all exact checks; YAML 1.1 `on` parsing is handled in regression test; test failures propagate (no continue-on-error/conditions); permissions are read-only; no REL-02 check added. No browser/screenshot: issue explicitly non-ui, source changes only CI/tests. No network dependency install probe, hosted CI or publication; no extra reviewer. Earlier builder checkout timeout is historical evidence and is not erased by this independent passing checkout run.

## Actual raw output

### Environment launch gate

Path: `/tmp/lead-wt/process-lessons-v1/.lead/evidence/env/relational/1792772166302094303/status.json`

```text
{
  "schema": 1,
  "project": "relational",
  "root": "/tmp/lead-wt/v-rel-01",
  "profile_sha256": "fbaecf787ed6355dc2b9f85a78c84012f1a2533147667b4ab827a033bf7a18c1",
  "source": {
    "head": "410644c99e1fb18cdcd1e14671d2485fa7055737",
    "dirty": false
  },
  "checked_at": 1792772164,
  "checks": [
    {
      "kind": "lockfile",
      "identity": "/tmp/lead-wt/v-rel-01/requirements-labs.lock.txt",
      "status": "PASS",
      "detail": "235bb5587af6ca0b82a859c30db8b264a482fe71e715031c5e75a2b939aa8958"
    },
    {
      "kind": "dependency",
      "identity": "numpy",
      "status": "PASS",
      "detail": {
        "python": "/home/avist/Projects/relational/.venv/bin/python",
        "result": {
          "version": "2.5.0",
          "file": "/home/avist/Projects/relational/.venv/lib/python3.12/site-packages/numpy/__init__.py"
        }
      }
    },
    {
      "kind": "dependency",
      "identity": "pandas",
      "status": "PASS",
      "detail": {
        "python": "/home/avist/Projects/relational/.venv/bin/python",
        "result": {
          "version": "3.0.3",
          "file": "/home/avist/Projects/relational/.venv/lib/python3.12/site-packages/pandas/__init__.py"
        }
      }
    },
    {
      "kind": "dependency",
      "identity": "sklearn",
      "status": "PASS",
      "detail": {
        "python": "/home/avist/Projects/relational/.venv/bin/python",
        "result": {
          "version": "1.9.0",
          "file": "/home/avist/Projects/relational/.venv/lib/python3.12/site-packages/sklearn/__init__.py"
        }
      }
    },
    {
      "kind": "dependency",
      "identity": "pyarrow",
      "status": "PASS",
      "detail": {
        "python": "/home/avist/Projects/relational/.venv/bin/python",
        "result": {
          "version": "24.0.0",
          "file": "/home/avist/Projects/relational/.venv/lib/python3.12/site-packages/pyarrow/__init__.py"
        }
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/circuits.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/circuits.parquet",
        "sha256": "c36c6537b9c5c9985ec438110717e93289085a5bffbcee07d08bbe1771994d8d",
        "expected": "c36c6537b9c5c9985ec438110717e93289085a5bffbcee07d08bbe1771994d8d"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/constructor_results.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/constructor_results.parquet",
        "sha256": "35582c2f0634d5503b664babf38264311332e6ca3cb612fb0fe737bfaffdc29e",
        "expected": "35582c2f0634d5503b664babf38264311332e6ca3cb612fb0fe737bfaffdc29e"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/constructor_standings.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/constructor_standings.parquet",
        "sha256": "739fc21ba301ef2081dbd8e56d643f148b672224a4d223b95d6107ccd735aa6b",
        "expected": "739fc21ba301ef2081dbd8e56d643f148b672224a4d223b95d6107ccd735aa6b"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/constructors.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/constructors.parquet",
        "sha256": "fc79dd346430186cdca6263e1c532891bf4d4ba7463fd94b398350a9356180c6",
        "expected": "fc79dd346430186cdca6263e1c532891bf4d4ba7463fd94b398350a9356180c6"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/drivers.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/drivers.parquet",
        "sha256": "ba3461b83a1a467a856bde4059519140a5a31952ad24a5e944ea9dd5c5faceae",
        "expected": "ba3461b83a1a467a856bde4059519140a5a31952ad24a5e944ea9dd5c5faceae"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/qualifying.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/qualifying.parquet",
        "sha256": "2d5cb7b4ed6cf4076b74ab1cc79f5f35faf31b7f032b51d2a1ac814e1d0c834b",
        "expected": "2d5cb7b4ed6cf4076b74ab1cc79f5f35faf31b7f032b51d2a1ac814e1d0c834b"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/races.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/races.parquet",
        "sha256": "88bb260f03c9ec5c7b030969d7284531d5f8e8592d73174686ad01cdfc8f87ea",
        "expected": "88bb260f03c9ec5c7b030969d7284531d5f8e8592d73174686ad01cdfc8f87ea"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/results.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/results.parquet",
        "sha256": "b8b0e39b27f318a7bd461f21bb0ba9ef44909ed569d287e0c6b2969c6bd2f598",
        "expected": "b8b0e39b27f318a7bd461f21bb0ba9ef44909ed569d287e0c6b2969c6bd2f598"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/db/standings.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/db/standings.parquet",
        "sha256": "f3f6a34fee190f077f2fe48026a7e5659e4c91a34f72340cdff3cf407c1f3534",
        "expected": "f3f6a34fee190f077f2fe48026a7e5659e4c91a34f72340cdff3cf407c1f3534"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/tasks/driver-dnf/test.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/tasks/driver-dnf/test.parquet",
        "sha256": "6598427cdb1afb5308aa6a1e88593126b539a98a79404c953181ccb25175b20e",
        "expected": "6598427cdb1afb5308aa6a1e88593126b539a98a79404c953181ccb25175b20e"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/tasks/driver-dnf/train.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/tasks/driver-dnf/train.parquet",
        "sha256": "994e9bdd0cf8f1930ac6ac77a5aa417d829f0472f93544a847cd616df99f6604",
        "expected": "994e9bdd0cf8f1930ac6ac77a5aa417d829f0472f93544a847cd616df99f6604"
      }
    },
    {
      "kind": "dataset",
      "identity": "rel-f1/tasks/driver-dnf/val.parquet",
      "status": "PASS",
      "detail": {
        "path": "/home/avist/.cache/relbench/rel-f1/tasks/driver-dnf/val.parquet",
        "sha256": "f8edfda92a493c64fa9bb2da2a1ea54346b5e33c232e6617c1412da613333242",
        "expected": "f8edfda92a493c64fa9bb2da2a1ea54346b5e33c232e6617c1412da613333242"
      }
    },
    {
      "kind": "modal",
      "identity": "not-required",
      "status": "PASS",
      "detail": "profile explicitly requires no remote image"
    }
  ],
  "verdict": "PASS",
  "expires_at": 1792772464
}

```

### Independent blast-radius executable assertions

Path: `/tmp/lead-wt/v-rel-01/.review/rel-01/blast-radius-proof.log`

```text
PASS: exact pull_request trigger; contents read; hosted runner; no job conditions/permission overrides/environments; checkout credentials disabled; only checkout/setup actions and six run steps; Pages bytes unchanged; canonical RED test bytes unchanged; RED lacks workflow; human-proof raw hashes match.
PyYAML=6.0.3
{
  "commands": [
    "python -m pip install 'PyYAML==6.0.3' 'numpy==2.5.0' 'pandas==3.0.3' 'scikit-learn==1.9.0' 'pyarrow==24.0.0'",
    "python3 labs/_test_b05_source_secrets.py",
    "python3 scripts/refresh_lesson_visuals.py --check",
    "python3 labs/_check_pages_checkout.py",
    "python3 scripts/search_index.py --check",
    "python -m unittest discover -s tests -p 'test_*.py'"
  ],
  "pages_sha256": "409cdc482079c45dd42573a5010e6197c4d401c3295152fecc8de89f5ee5d4d5"
}

```

### Four exact CLI checks, reused Python via PATH

Path: `/tmp/lead-wt/v-rel-01/.review/rel-01/four-checks.log`

```text
$ python3 labs/_test_b05_source_secrets.py
....
----------------------------------------------------------------------
Ran 4 tests in 0.372s

OK

exit_code=0
$ python3 scripts/refresh_lesson_visuals.py --check
{"lessons": 228, "stories": 61, "changed": 0}

exit_code=0
$ python3 labs/_check_pages_checkout.py
PASS: complete Pages build from Git index (no ignored workspace files)

exit_code=0
$ python3 scripts/search_index.py --check
search index up to date

exit_code=0

```

### lead-verify status

Path: `/tmp/lead-wt/process-lessons-v1/.lead/evidence/verify/relational/20261023T161628Z/status.json`

```text
{
  "project": "relational",
  "root": "/tmp/lead-wt/v-rel-01",
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
  "started_utc": "20261023T161628Z",
  "source": {
    "head": "410644c99e1fb18cdcd1e14671d2485fa7055737",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}

```

### lead-verify stdout

Path: `/tmp/lead-wt/process-lessons-v1/.lead/evidence/verify/relational/20261023T161628Z/stdout.txt`

```text
FAIL hash check:
  x: sha256 mismatch

```

### lead-verify stderr

Path: `/tmp/lead-wt/process-lessons-v1/.lead/evidence/verify/relational/20261023T161628Z/stderr.txt`

```text
.........................................
----------------------------------------------------------------------
Ran 41 tests in 1.152s

OK

```

### Canonical retained RED

Path: `/tmp/lead-wt/process-lessons-v1/.lead/tasks/rel-01-ci/reports/rel-01-preview.tdd/red.log`

```text
LEAD_RESULT={"run": 1, "failed": 1, "errors": 0, "skipped": 0}
test_pull_requests_run_all_five_checks_without_deploying (lead_replay_0.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... FAIL

======================================================================
FAIL: test_pull_requests_run_all_five_checks_without_deploying (lead_replay_0.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-tdd-1vykld18/tree/tests/ci/test_pr_checks.py", line 14, in test_pull_requests_run_all_five_checks_without_deploying
    self.assertTrue(path.is_file(), 'PRs need a validation workflow')
AssertionError: False is not true : PRs need a validation workflow

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (failures=1)

```

### Canonical retained GREEN

Path: `/tmp/lead-wt/process-lessons-v1/.lead/tasks/rel-01-ci/reports/rel-01-preview.tdd/green.log`

```text
LEAD_RESULT={"run": 1, "failed": 0, "errors": 0, "skipped": 0}
test_pull_requests_run_all_five_checks_without_deploying (lead_replay_0.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.002s

OK

```

Verification CLI: `/tmp/lead-wt/process-lessons-v1/bin/lead-verify relational --worktree /tmp/lead-wt/v-rel-01` returned exit 0, PASS, source.head `410644c99e1fb18cdcd1e14671d2485fa7055737`, dirty false at verification launch. Environment launch gate also PASS, same head and clean tree. Later worktree dirtiness is only verifier review artifacts under `.review/`; no tracked-source edits.

</details>
