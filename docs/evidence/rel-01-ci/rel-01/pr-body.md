PRs previously had no validation. Add a read-only pull_request workflow running all five REL-01 checks on ubuntu-latest, with explicitly pinned check dependencies installed only on disposable GitHub Actions runners. Existing Pages deployment bytes stay unchanged. Publication-step TMPDIR uses disk-backed runner.temp so the unchanged publication checker can export/build its large archive outside /tmp quota.

[Canonical RED/GREEN evidence (verification stdout includes an expected failing fixture)](docs/evidence/rel-01-ci/rel-01/evidence.md) · [RED](docs/evidence/rel-01-ci/rel-01/red.log) · [GREEN](docs/evidence/rel-01-ci/rel-01/green.log) · [verification stdout](docs/evidence/rel-01-ci/rel-01/verification.log). Full passing stderr/status is retained below.

Reviewed source `a1f7b15a288fb432ef4aa24b28b2ae4a36212c32`. Same independent native verifier PASS, RISK_FLAG:not-risky; actual review below. Canonical supported-step-context regression before1failure, after0; identicaltest, zeroerrors/skips. Actual final-source local unittest suite42testsPASS and clean source; status/stderr retained below. Expected negative-hash fixture prints FAIL on stdout while the passing suite reports OK on stderr. The existing deterministic renderer accepts stdout only; its readability limitation is disclosed. Hosted CI is required on the delivery head.

Actual failed counterfactuals: [first run39986031745](https://github.com/Avistian/relational/actions/runs/39986031745) and [root-space cleanup run39987374176](https://github.com/Avistian/relational/actions/runs/39987374176) failed publication-copy with Disk quota exceeded. The second run used Ubuntu26.04.1, logged rootfree88G before cleanup and105G after, with18M free inodes, yet failed at the same file. Root cleanup was therefore removed, not represented as a successful fix. [Primary runner-image issue14777](https://github.com/actions/runner-images/issues/14777) has maintainers' reproduction of Ubuntu26.04 /tmp tmpfs per-user quota and their TMPDIR=runner.temp option. The publication-step TMPDIR setting is a bounded inference from matching actual image/symptom and upstream reproduction; actual new-head CI must confirm it. Failed runs' later search/unittest steps were skipped, never counted as passing.

No local installs/cleanup, new remote/provider/paid runner, merge, data download or real experiment. Existing synthetic tests fit tiny models; the real-cache test explicitly skips on GHA without downloaded data. Incremental USD0, model cost unmeasured. Draft only, humans merge.

Historical local exact-five-command proof below is explicitly source6900f637, before the final workflow-only TMPDIR change. Final source received fresh42-test verification and canonical replay; hosted CI must execute all five checks on its delivery head.

```text
$ python3 labs/_test_b05_source_secrets.py
....
----------------------------------------------------------------------
Ran 4 tests in 0.570s

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
..........................................
----------------------------------------------------------------------
Ran 42 tests in 1.226s

OK

exit_code=0
```

The supervisor caught an additional prepublication schema defect: runner context is unavailable in job.env. [Official GitHub context table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts) permits runner in steps.env. The final workflow puts TMPDIR only on the unchanged publication-check step. The unpublished job-env candidate and its structural/local PASS are retained historically; neither is called hosted-valid. The new test rejects the invalid job placement and requires the supported step placement.

Actual final-source lead-verify status:
```json
{
  "project": "relational",
  "root": "/tmp/lead-wt/rel-01",
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
  "started_utc": "20261023T164145Z",
  "source": {
    "head": "a1f7b15a288fb432ef4aa24b28b2ae4a36212c32",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}
```

Actual final-source lead-verify stderr:
```text
..........................................
----------------------------------------------------------------------
Ran 42 tests in 1.212s

OK
```

<details>
<summary>Actual independent supported-context review (source scope; hosted CI pending)</summary>

# REL-01 corrected-context source review: PASS (source scope)

RISK_FLAG: not-risky

Source: `a1f7b15a288fb432ef4aa24b28b2ae4a36212c32`.
Reviewed delta: `c4f0719eba859ce6d0fbb254e2cb91001614903e..a1f7b15a288fb432ef4aa24b28b2ae4a36212c32`.
Task/slice: `rel-01-ci/rel-01`; reviewer/actor `verifier:rel-01`; actual pane `w1:pR`.
Same independent verifier; no additional reviewer or agent. This is not a DONE acceptance or remote-CI report.

## Verdict and official placement rule

PASS for the narrow source correction. No blocking current-source finding. The job TMPDIR env is removed. Only the exact Pages-check step now sets TMPDIR to runner.temp (`.github/workflows/pr-checks.yml:36`–`:38`).

Independently checked the official [GitHub context availability table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability). It excludes runner at job env and allows runner at step env. The current placement conforms to that documented rule. This is primary-document verification plus source inspection, not a GitHub-hosted schema validator result or workflow admission proof.

`not-risky` applies to this bounded source repair: it limits temp-path redirection to the one command needing it, preserves the exact command and avoids privileged cleanup. Hosted context expansion, scratch capacity and all-five-check execution remain pending actual final-head Actions. No claim of remote PASS or new-head full local verification.

## Preserved missed finding and correction history

The earlier c4f0719 PASS was wrong and is withdrawn. Supervisor caught the unsupported job-env runner expression before the third CI push. I had verified YAML structure and Python tempfile routing, but missed GitHub's context restrictions; PyYAML cannot establish workflow validity. The regression at that head asserted the unsupported placement and therefore could pass despite this defect.

Original artifact is retained unchanged in `rel-01-c4f0719-original-PASS.md`. Explicit corrected historical verdict is `rel-01-c4f0719-correction.md`: BLOCKED / RISK_FLAG risky at c4f0719. Skill records retain the original event and append its correction, then this new-source review. No finding or erroneous result was erased.

Per supervisor, no c4 push occurred and actual PR head remained f76d7572. This is attributed status, not a new forge observation. Historical runs 39986031745 and 39987374176 remain failed; no third-run result is inferred.

## Blast-radius executable proof

Safety fact: this moves only one env binding into a documented supported scope while leaving all run commands and access unchanged. Parsed current and c4 workflows; asserted no job env, exactly one step env on exact Pages-check command with the intended value; moved the binding back only in an in-memory comparison copy and asserted whole-workflow semantic equality with c4. Five commands, actions, dependencies, permissions, trigger and timeout therefore remain identical. Pages/checker bytes unchanged; cleanup remains absent. No source file was modified by this review.

Independent actual two-test module run: 2 tests, 0 failures/errors/skips, exit 0. The regression now rejects job TMPDIR and requires supported env on the exact Pages command (`tests/ci/test_pr_checks.py:45`). Test-only RED `dfc8a15a` has final-identical test bytes and pre-fix workflow. Retained builder RED/GREEN logs are included below as builder evidence, not independently executed RED. No tests were authored by this verifier.

Proof level: real execution for focused tests and structural comparison; primary-source rule check for context availability. Prior tiny tempfile routing probe remains historical API evidence only. No full Pages export, broad checks, local installs, privileged removal, push, spend, new agent or merge.

## Risks and remaining evidence

- Actual GitHub admission and runtime success have not run here. Correct placement eliminates the confirmed documented-rule defect, but local YAML tests are not Actions validation. Require actual final-head job to run all five commands, without skipped-check success.
- Hosted tmpfs diagnosis remains supported by the retained second failure and runner-images maintainer discussion. Redirecting the checker to runner scratch is plausible and uses documented step context; resolving this job's capacity failure still needs remote execution.
- Step env now scopes TMPDIR to the Pages checker and its subprocesses. Other steps retain original temp behavior; they passed until the Pages failure in the historical run. No broader temp override is introduced.
- Docs/evidence changes and builder full gates are not reaccepted by this narrow review. Lead must bind final source and all evidence separately. No old five-command result is promoted to this head.
- Launch environment gate PASS at current exact source; dirty=true reflects only untracked `.review/` artifacts. `git diff --exit-code` passed: no tracked changes. No configured full verify-project run was repeated or claimed.

Lead validates this artifact and latest invocation, then associates final-head serial local gates and actual Actions URL/head/conclusion. Human merge authority remains required.

## Raw output

### Independent focused tests

`/tmp/lead-wt/v-rel-01/.review/rel-01/a1f7b15/focused-tests.log`

```text
$ /home/avist/Projects/relational/.venv/bin/python -B -m unittest tests.ci.test_pr_checks -v
test_publication_scratch_uses_supported_step_runner_context (tests.ci.test_pr_checks.PullRequestChecks.test_publication_scratch_uses_supported_step_runner_context) ... ok
test_pull_requests_run_all_five_checks_without_deploying (tests.ci.test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.004s

OK

exit_code=0

```

### Independent structural/context placement assertions

`/tmp/lead-wt/v-rel-01/.review/rel-01/a1f7b15/placement-proof.log`

```text
PASS: no job env; only exact Pages check step has env TMPDIR=${{ runner.temp }}.
PASS: reversing only that env placement produces previous parsed YAML exactly; five commands, dependencies, actions, trigger, timeout and permissions unchanged.
PASS: Pages/checker unchanged; no cleanup added.
PASS: tests-only RED dfc8a15a has final-identical tests and old invalid job-env placement.
Official-context comparison (independently read GitHub table): runner excluded at jobs.<job_id>.env, included at jobs.<job_id>.steps.env. Old placement violates table; new placement matches it.
LIMIT: YAML assertions are not a GitHub-hosted workflow validator or remote CI execution.

```

### Launch environment gate

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/env/relational/1792773714756560085/status.json`

```text
{
  "schema": 1,
  "project": "relational",
  "root": "/tmp/lead-wt/v-rel-01",
  "profile_sha256": "fbaecf787ed6355dc2b9f85a78c84012f1a2533147667b4ab827a033bf7a18c1",
  "source": {
    "head": "a1f7b15a288fb432ef4aa24b28b2ae4a36212c32",
    "dirty": true
  },
  "checked_at": 1792773713,
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
  "expires_at": 1792774013
}

```

### Builder retained RED

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/step-context-red.log`

```text
test_publication_scratch_uses_supported_step_runner_context (test_pr_checks.PullRequestChecks.test_publication_scratch_uses_supported_step_runner_context) ... FAIL
test_pull_requests_run_all_five_checks_without_deploying (test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

======================================================================
FAIL: test_publication_scratch_uses_supported_step_runner_context (test_pr_checks.PullRequestChecks.test_publication_scratch_uses_supported_step_runner_context)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-wt/rel-01/tests/ci/test_pr_checks.py", line 48, in test_publication_scratch_uses_supported_step_runner_context
    self.assertNotIn('TMPDIR', job.get('env', {}),
AssertionError: 'TMPDIR' unexpectedly found in {'TMPDIR': '${{ runner.temp }}'} : GitHub job.env cannot reference the runner context

----------------------------------------------------------------------
Ran 2 tests in 0.004s

FAILED (failures=1)

```

### Builder retained GREEN

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/step-context-green.log`

```text
test_publication_scratch_uses_supported_step_runner_context (test_pr_checks.PullRequestChecks.test_publication_scratch_uses_supported_step_runner_context) ... ok
test_pull_requests_run_all_five_checks_without_deploying (test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.003s

OK

```

### Source status

```text
a1f7b15a288fb432ef4aa24b28b2ae4a36212c32
?? .review/
```

</details>
