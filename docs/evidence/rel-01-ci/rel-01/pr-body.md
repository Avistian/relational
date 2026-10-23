PRs previously had no validation. Add a read-only pull_request workflow running all five REL-01 checks on ubuntu-latest, with explicitly pinned check dependencies installed only on disposable GitHub Actions runners. Existing Pages deployment bytes stay unchanged. Job TMPDIR uses disk-backed runner.temp so the unchanged publication checker can export/build its large archive outside /tmp quota.

[Canonical RED/GREEN evidence (verification stdout includes an expected failing fixture)](docs/evidence/rel-01-ci/rel-01/evidence.md) · [RED](docs/evidence/rel-01-ci/rel-01/red.log) · [GREEN](docs/evidence/rel-01-ci/rel-01/green.log) · [verification stdout](docs/evidence/rel-01-ci/rel-01/verification.log). Full passing stderr/status is retained below.

Reviewed source `c4f0719eba859ce6d0fbb254e2cb91001614903e`. Same independent native verifier PASS, RISK_FLAG:not-risky; actual review below. Canonical TMPDIR regression before1failure, after0; identicaltest, zeroerrors/skips. Actual final-source local unittest suite42testsPASS and clean source; status/stderr retained below. Expected negative-hash fixture prints FAIL on stdout while the passing suite reports OK on stderr. The existing deterministic renderer accepts stdout only; its readability limitation is disclosed. Hosted CI is required on the delivery head.

Actual failed counterfactuals: [first run39986031745](https://github.com/Avistian/relational/actions/runs/39986031745) and [root-space cleanup run39987374176](https://github.com/Avistian/relational/actions/runs/39987374176) failed publication-copy with Disk quota exceeded. The second run used Ubuntu26.04.1, logged rootfree88G before cleanup and105G after, with18M free inodes, yet failed at the same file. Root cleanup was therefore removed, not represented as a successful fix. [Primary runner-image issue14777](https://github.com/actions/runner-images/issues/14777) has maintainers' reproduction of Ubuntu26.04 /tmp tmpfs per-user quota and their TMPDIR=runner.temp option. The job TMPDIR setting is a bounded inference from matching actual image/symptom and upstream reproduction; actual new-head CI must confirm it. Failed runs' later search/unittest steps were skipped, never counted as passing.

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
  "duration_s": 3,
  "started_utc": "20261023T163727Z",
  "source": {
    "head": "c4f0719eba859ce6d0fbb254e2cb91001614903e",
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
Ran 42 tests in 1.290s

OK
```

<details>
<summary>Actual independent final repair review (source scope; hosted CI pending)</summary>

# REL-01 final bounded repair review: PASS (source scope)

RISK_FLAG: not-risky

Source: `c4f0719eba859ce6d0fbb254e2cb91001614903e`.
Reviewed delta: `6900f637951123104d01377120e0c7140b3fd0b6..c4f0719eba859ce6d0fbb254e2cb91001614903e`.
Task/slice: `rel-01-ci/rel-01`. Reviewer/actor: `verifier:rel-01`; pane: `w1:pR`.
Same independent verifier. Prior source review archived as `rel-01-6900f63.md`; its PASS does not cover current source. This is a source review, not DONE acceptance or remote CI success.

## Verdict and risk judgment

PASS: no blocking source finding in this narrow repair. Two focused workflow tests passed. Exact parsed-workflow comparison proves the only behavioral changes are removing the first tool-directory cleanup step and adding job env `TMPDIR: ${{ runner.temp }}`. Required five commands, setup actions/dependencies, permissions, trigger, timeout and Pages/checker bytes are unchanged. No deployment or removal shell step remains.

`not-risky` applies to this bounded source change: using runner scratch for temporary files avoids altering runner mounts or deleting toolchains. The job-wide env changes temporary-file placement for Python and other tools that honor TMPDIR, including dependency setup; it grants no additional permissions. Hosted filesystem capacity, runtime and final job success remain unverified by these local checks. New actual final-head Actions execution of all five checks remains required. No source edits, installs, push, spend, new reviewer/agent or merge; no multi-gigabyte export.

## Actual second CI failure and updated diagnosis

Inspected retained `final-ci.json` and `ci-capacity-counterfactual.log`: run `39987374176`, head `f76d7572a027ba9a4dd3f2d8c6234071fc5aeddb`, image Ubuntu 26.04.1, failed the checkout publication check with `Disk quota exceeded`. Root free space was 88G before cleanup and 105G after, with 18,028,625 free root inodes afterward. This disproves the previous repair's assumption that reclaiming root capacity was sufficient; root metrics do not measure the separate /tmp quota. Search-index and unittest steps were skipped.

[Failed run](https://github.com/Avistian/relational/actions/runs/39987374176) / [failed job](https://github.com/Avistian/relational/actions/runs/39987374176/job/120529951097). These are historical failures, not the new head's result.

Independently opened the [runner-images maintainer issue](https://github.com/actions/runner-images/issues/14777): contributor v-davit-ioramashvili reports reproducing Ubuntu 26.04 /tmp tmpfs quotas, with ordinary users reaching about 6.3G, and explicitly lists setting TMPDIR to runner.temp as an option. This supports the revised diagnosis and implementation; it does not prove this job now passes. No upstream claim that a blanket root cleanup fixes the quota is made.

## Blast-radius safety fact and executable proof

Safety fact: the unchanged checkout checker uses Python `tempfile.TemporaryDirectory` without an explicit directory, so job TMPDIR can redirect its export/build tree while all required checks retain their semantics.

References: `.github/workflows/pr-checks.yml:15`–`:16` adds TMPDIR; checkout starts at `:18`; exact five commands remain at `:32`, `:35`, `:38`, `:41`, `:44`. `labs/_check_pages_checkout.py:16` uses TemporaryDirectory. No Pages/checker diff.

Proof reached skill level 4 for source structure and tempfile routing: parsed actual old/new YAML, removed the old first cleanup step and new env respectively, and asserted complete equality; asserted no sudo/removal commands remain; ran the real two-test module; launched a fresh process in the existing Python environment with a verifier-owned TMPDIR, invoked the same TemporaryDirectory API, asserted its parent is that directory, wrote two bytes and cleaned its own tiny fixture. Fresh process avoids cached tempfile selection. No full checker/export or hosted-capacity reproduction ran. Hosted filesystem backing and capacity remain the final Actions counterfactual's responsibility.

## TDD correctness inspected

`104936bd` is tests-only (test change replaces the ineffective cleanup contract with the TMPDIR contract). Its test bytes equal current source and its YAML equals pre-repair YAML, which lacks job TMPDIR. Inspected retained builder RED/GREEN logs: identical two tests, RED one assertion failure for missing TMPDIR, GREEN two passing. Independent current-source module run: 2 tests, no failures/errors/skips, exit 0. Builder RED execution is retained evidence, not claimed as independently executed here. No TDD implementation was performed by this verifier.

## Explicit risks, evidence limits and remaining gate

- Host-side outcome is still unproven: context expansion, runner scratch permissions/capacity and elapsed time require actual new-head CI. Impact if wrong is failed validation; no publication occurs. Require all five steps to run successfully, not a green result with skipped checks.
- TMPDIR is job-wide: temp files for multiple steps share runner scratch. Python respects it as independently demonstrated, but this tiny probe does not establish behavior or capacity for every tool. Existing unittest/CLI gates and actual hosted run remain required; old full-local results are historical.
- The focused test proves the configured TMPDIR value, not actual filesystem backing or cleanup removal. Independent semantic comparison and no-removal assertions cover current-source removal; capacity remains remote evidence.
- Documentation/evidence changes in this delta remain historical builder proof and are not accepted as final-head full verification by this review. Rendered proof is gate-bound; builder/lead owns supplementary raw stderr/status links and final evidence binding. No broad evidence re-review performed.
- Launch environment gate PASS at this exact head; `dirty:true` is only retained untracked `.review/` verifier artifacts. No tracked-source changes. Configured full verify-project and four other commands were not repeated in this narrow review; focused output is not misrecorded as a full verify-project invocation.

Lead validates this review, binds new final-head local/remote evidence, and owns counted-slot cleanup. No agent merge authorized.

## Raw evidence

### Independent focused tests

`/tmp/lead-wt/v-rel-01/.review/rel-01/c4f0719/focused-tests.log`

```text
$ /home/avist/Projects/relational/.venv/bin/python -B -m unittest tests.ci.test_pr_checks -v
test_large_publication_exports_use_disk_backed_runner_temp (tests.ci.test_pr_checks.PullRequestChecks.test_large_publication_exports_use_disk_backed_runner_temp) ... ok
test_pull_requests_run_all_five_checks_without_deploying (tests.ci.test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.004s

OK

exit_code=0

```

### Independent executable structural/routing proof

`/tmp/lead-wt/v-rel-01/.review/rel-01/c4f0719/delta-proof.log`

```text
PASS: parsed old YAML minus exactly cleanup step equals new YAML minus exactly TMPDIR job env. Exact five commands, dependencies, actions, trigger, timeout and permissions unchanged.
PASS: no sudo/removal step remains. Pages/checker bytes unchanged.
PASS: tests-only RED104936bd has final-identical tests, pre-fix YAML and no TMPDIR env.
PASS: fresh Python TemporaryDirectory selected TMPDIR; wrote two bytes; cleaned only its own tiny fixture
Hosted runner.temp filesystem capacity is NOT proven by this tiny local routing probe.

```

### Launch environment gate

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/env/relational/1792773457312621794/status.json`

```text
{
  "schema": 1,
  "project": "relational",
  "root": "/tmp/lead-wt/v-rel-01",
  "profile_sha256": "fbaecf787ed6355dc2b9f85a78c84012f1a2533147667b4ab827a033bf7a18c1",
  "source": {
    "head": "c4f0719eba859ce6d0fbb254e2cb91001614903e",
    "dirty": true
  },
  "checked_at": 1792773455,
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
  "expires_at": 1792773755
}

```

### Historical second CI metadata

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/final-ci.json`

```text
{"conclusion":"failure","headSha":"f76d7572a027ba9a4dd3f2d8c6234071fc5aeddb","jobs":[{"completedAt":"2026-10-23T16:34:29Z","conclusion":"failure","databaseId":120529951097,"name":"checks","startedAt":"2026-10-23T16:30:50Z","status":"completed","steps":[{"completedAt":"2026-10-23T16:30:52Z","conclusion":"success","name":"Set up job","number":1,"startedAt":"2026-10-23T16:30:50Z","status":"completed"},{"completedAt":"2026-10-23T16:32:26Z","conclusion":"success","name":"Reclaim space for publication checks","number":2,"startedAt":"2026-10-23T16:30:52Z","status":"completed"},{"completedAt":"2026-10-23T16:33:42Z","conclusion":"success","name":"Run actions/checkout@v5","number":3,"startedAt":"2026-10-23T16:32:26Z","status":"completed"},{"completedAt":"2026-10-23T16:33:43Z","conclusion":"success","name":"Run actions/setup-python@v6","number":4,"startedAt":"2026-10-23T16:33:42Z","status":"completed"},{"completedAt":"2026-10-23T16:34:02Z","conclusion":"success","name":"Install check dependencies","number":5,"startedAt":"2026-10-23T16:33:43Z","status":"completed"},{"completedAt":"2026-10-23T16:34:03Z","conclusion":"success","name":"Check archived B05 sources for Google API keys","number":6,"startedAt":"2026-10-23T16:34:02Z","status":"completed"},{"completedAt":"2026-10-23T16:34:11Z","conclusion":"success","name":"Verify generated lesson visuals","number":7,"startedAt":"2026-10-23T16:34:03Z","status":"completed"},{"completedAt":"2026-10-23T16:34:26Z","conclusion":"failure","name":"Verify clean-checkout publication inputs","number":8,"startedAt":"2026-10-23T16:34:11Z","status":"completed"},{"completedAt":"2026-10-23T16:34:26Z","conclusion":"skipped","name":"Verify search index","number":9,"startedAt":"2026-10-23T16:34:26Z","status":"completed"},{"completedAt":"2026-10-23T16:34:26Z","conclusion":"skipped","name":"Run unit tests","number":10,"startedAt":"2026-10-23T16:34:26Z","status":"completed"},{"completedAt":"2026-10-23T16:34:26Z","conclusion":"skipped","name":"Post Run actions/setup-python@v6","number":19,"startedAt":"2026-10-23T16:34:26Z","status":"completed"},{"completedAt":"2026-10-23T16:34:27Z","conclusion":"success","name":"Post Run actions/checkout@v5","number":20,"startedAt":"2026-10-23T16:34:26Z","status":"completed"},{"completedAt":"2026-10-23T16:34:27Z","conclusion":"success","name":"Complete job","number":21,"startedAt":"2026-10-23T16:34:27Z","status":"completed"}],"url":"https://github.com/Avistian/relational/actions/runs/39987374176/job/120529951097"}],"status":"completed","url":"https://github.com/Avistian/relational/actions/runs/39987374176"}

```

### Builder retained repair RED

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/tmpdir-red.log`

```text
test_large_publication_exports_use_disk_backed_runner_temp (test_pr_checks.PullRequestChecks.test_large_publication_exports_use_disk_backed_runner_temp) ... FAIL
test_pull_requests_run_all_five_checks_without_deploying (test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

======================================================================
FAIL: test_large_publication_exports_use_disk_backed_runner_temp (test_pr_checks.PullRequestChecks.test_large_publication_exports_use_disk_backed_runner_temp)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/lead-wt/rel-01/tests/ci/test_pr_checks.py", line 48, in test_large_publication_exports_use_disk_backed_runner_temp
    self.assertEqual(job.get('env', {}).get('TMPDIR'), '${{ runner.temp }}',
AssertionError: None != '${{ runner.temp }}' : Publication export/build copies exceed Ubuntu 26.04 /tmp user quota

----------------------------------------------------------------------
Ran 2 tests in 0.004s

FAILED (failures=1)

```

### Builder retained repair GREEN

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/tmpdir-green.log`

```text
test_large_publication_exports_use_disk_backed_runner_temp (test_pr_checks.PullRequestChecks.test_large_publication_exports_use_disk_backed_runner_temp) ... ok
test_pull_requests_run_all_five_checks_without_deploying (test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.003s

OK

```

### Historical second CI raw metrics/error

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/ci-capacity-counterfactual.log` SHA256 `39ec5c57a724f3fbe5a68533c4625f076a6a427d788360c15d021d83a3a01352`

```text
checks	Set up job	2026-10-23T16:30:50.7288153Z 26.04.1
checks	Reclaim space for publication checks	2026-10-23T16:30:52.1621275Z /dev/root       145G   57G   88G  40% /
checks	Reclaim space for publication checks	2026-10-23T16:30:52.1622440Z /dev/root      18690560 1031757 17658803    6% /
checks	Reclaim space for publication checks	2026-10-23T16:32:26.5588158Z /dev/root       145G   40G  105G  28% /
checks	Reclaim space for publication checks	2026-10-23T16:32:26.5596831Z /dev/root      18690560 661935 18028625    4% /
checks	Verify clean-checkout publication inputs	2026-10-23T16:34:26.9447965Z cp: error copying 'labs/sources/l187/privacybook.pdf' to 'public/labs/sources/l187/privacybook.pdf': Disk quota exceeded
```

### Source status

```text
c4f0719eba859ce6d0fbb254e2cb91001614903e
?? .review/
```

</details>
