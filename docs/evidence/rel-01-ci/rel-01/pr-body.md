PRs previously had no validation. Add a read-only pull_request workflow running all five REL-01 checks on ubuntu-latest, with explicitly listed dependencies installed only on disposable GitHub Actions runners. Pages deployment bytes are unchanged. The tracked archive is 3.84 GiB; reclaim unused Android/.NET/GHC tools on the disposable runner before checkout so the clean export/publication copies have room. Log available bytes and inodes before and after cleanup.

[Canonical RED/GREEN evidence (verification stdout includes an expected failing fixture)](docs/evidence/rel-01-ci/rel-01/evidence.md) · [RED](docs/evidence/rel-01-ci/rel-01/red.log) · [GREEN](docs/evidence/rel-01-ci/rel-01/green.log) · [verification stdout](docs/evidence/rel-01-ci/rel-01/verification.log). Full passing stderr/status is retained below.

Reviewed source `6900f637951123104d01377120e0c7140b3fd0b6`. Separate same native verifier PASS, RISK_FLAG:not-risky; actual report below. Canonical capacity regression: before1failure, after0; identicaltest, zeroerrors/skips. Local suite42testsPASS, clean reviewed source. The existing verification stdout contains intentional negative-fixture `FAIL hash check`; the full stderr below shows the actual passing result. Existing deterministic renderer accepts stdout only; its limitation is disclosed, not hidden. Hosted CI remains required on this final head.

The first real Actions run [39986031745](https://github.com/Avistian/relational/actions/runs/39986031745) failed publication-copy `Disk quota exceeded` on earlier head1c47ca9e; later search/unittest steps were skipped and are never counted as passing. The precise quota boundary was not logged. The workflow repair changes runner capacity only; actual new-head CI must confirm the counterfactual.

No local installs or cleanup, new remotes, paid runners, merge, data download or real experiment. Existing synthetic tests fit tiny models; the real-cache test explicitly skips on GHA without downloaded data. Incremental USD0, model cost unmeasured. Draft only; humans merge.

Exact required commands, run locally using the existing main .venv via PATH:

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

Full lead-verify status:
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
  "started_utc": "20261023T162606Z",
  "source": {
    "head": "6900f637951123104d01377120e0c7140b3fd0b6",
    "dirty": false
  },
  "config_sha256": "9ab11a32730c2c648094b4f16ab52f3d9e044172ac3c7285f5f63ae709c87d17",
  "note": "software-verified only; not experiment results"
}
```

Full lead-verify stderr:
```text
..........................................
----------------------------------------------------------------------
Ran 42 tests in 1.241s

OK
```

<details>
<summary>Actual independent repair review (source scope; hosted CI pending)</summary>

# REL-01 bounded repair review: PASS (source scope)

RISK_FLAG: not-risky

Source: `6900f637951123104d01377120e0c7140b3fd0b6`.
Compared source: `410644c99e1fb18cdcd1e14671d2485fa7055737..6900f637951123104d01377120e0c7140b3fd0b6`.
Task/slice: `rel-01-ci/rel-01`. Reviewer/actor: `verifier:rel-01`; pane: `w1:pR`.
Same independent verifier; no additional reviewer or agent. This supersedes the old source review for current source. Historical review is retained in `rel-01-410644c.md`; it does not establish verification of this head. No DONE acceptance report.

## Verdict and risk judgment

PASS for this bounded source repair. No blocking code finding. Two focused workflow tests passed independently; executable YAML comparison proves all existing workflow behavior is identical after removing the new first cleanup step. No broad re-review or Pages export was performed. Builder owns fresh required gates in serial; neither old five-command success nor this two-test run is asserted as new-head full verification.

`not-risky` describes source safety: deletion is limited to three literal, unused tool directories on the disposable hosted job, before repository checkout, with no workspace variable or broad path. Repository permission, trigger, dependencies and five required commands are unchanged. It does not assert that deleting those tools solves the observed hosted quota or that Actions passed. Capacity, hosted permissions, image layout, runtime and total job duration remain unproven until the new final-head Actions run. No push, install, spend, merge, source edit or local removal occurred.

## Failure inspected

Retained real Actions run `39986031745` failed on old head `1c47ca9e6104db90d09c5063257041667edeb863`, job `checks`, in `_check_pages_checkout.py` while copying publication files: `Disk quota exceeded`. Search-index and unittest steps were skipped after the failure. This is a genuine failed run, not successful CI evidence for either head.

Run: https://github.com/Avistian/relational/actions/runs/39986031745
Job: https://github.com/Avistian/relational/actions/runs/39986031745/job/120525336228
Draft PR: https://github.com/Avistian/relational/pull/23 (retained snapshot; current new-head state not queried).

## Blast radius and executed proof

Safety fact: the added shell block can only target the three fixed hosted tool paths and does not change the checks that follow. `.github/workflows/pr-checks.yml:16` introduces the first step; `:18`–`:22` report byte/inode availability before and after the removal; `:20` names `/usr/local/lib/android`, `/usr/share/dotnet`, `/opt/ghc`; checkout follows at `:24`. These toolchains are not used by the Python/Node/shell checks. Read-only contents permission stays at `:6` and hosted runner at `:11`. Pages workflow and checkout-checker bytes are unchanged.

Proof reached blast-radius level 4 for the structural assertions: parsed real old/new YAML with installed PyYAML; removed exactly the new first step and asserted complete semantic equality; asserted exact five-line cleanup block, no interpolated paths, and checkout ordering; checked the block with `bash -n` only; ran actual two-test module on this source. No deletion commands from the workflow were executed, mocked as successful, or measured locally. Actual capacity reclamation remains unproven.

Repair RED commit `b1fbc63f` modifies only the test file. Independently compared its test bytes to final source: identical; its YAML equals pre-repair YAML and has no pre-checkout step. This establishes the expected failing seam; this verifier did not replay RED or claim builder's replay logs as independently executed. Final actual module run: 2 tests, 0 failures/errors/skips, exit 0.

## Explicit risks and evidence limits

- Hosted quota outcome: moderate uncertainty, high impact on delivery if it persists. `Disk quota exceeded` does not by itself establish whether bytes, inodes or a quota mechanism is limiting. `df -h`/`df -i` provide observations but impose no capacity threshold. Require the actual new-head job to execute all five checks successfully; no remote PASS is claimed.
- Hosted image layout may change; missing cleanup paths are tolerated by `rm -rf`, so a future image could reclaim little space. The Python tool cache is outside the three target paths. A failed cleanup exits the step and blocks checks; there is no continue-on-error. Cost is failed CI, confined to the disposable job.
- Regression test uses substring assertions, so it does not independently prove exact deletion targets/order or actual reclaimed capacity. Current source was checked with exact-line and whole-workflow equality assertions as additional independent proof. No test changes requested.
- Added docs/evidence are historical proof from the earlier source and do not constitute new-head verification. Supplementary passing stderr/status publication remains a builder/lead evidence responsibility; deterministic rendered evidence is gate-bound. This bounded repair review does not rerender or amend it.
- Launch environment gate: PASS on current exact head, `dirty:true` solely for retained untracked `.review/` artifacts. No tracked-source diff. Existing read-only Python and cache reused; no preparation/install. Configured full `/verify-project` command was intentionally not rerun by this verifier for this delta; focused module output below is not misrecorded as that skill's full output.

Before completion: lead validates this review, builder completes serial gates, and actual final-head Actions CI must pass. Human merge authority remains required.

## Raw output

### Focused actual test command

`/tmp/lead-wt/v-rel-01/.review/rel-01/6900f63/focused-tests.log`

```text
$ /home/avist/Projects/relational/.venv/bin/python -B -m unittest tests.ci.test_pr_checks -v
test_hosted_runner_reclaims_capacity_before_checkout (tests.ci.test_pr_checks.PullRequestChecks.test_hosted_runner_reclaims_capacity_before_checkout) ... ok
test_pull_requests_run_all_five_checks_without_deploying (tests.ci.test_pr_checks.PullRequestChecks.test_pull_requests_run_all_five_checks_without_deploying) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.004s

OK

exit_code=0

```

### Executable blast-radius assertions

`/tmp/lead-wt/v-rel-01/.review/rel-01/6900f63/delta-proof.log`

```text
PASS: new workflow minus exactly its first cleanup step is semantically identical to old workflow (all five checks, actions, dependencies, permissions, trigger and timeout unchanged).
PASS: cleanup shell syntax checked only; no shell commands from the cleanup block were executed.
PASS: cleanup argv is literal and exactly three unused hosted tool directories; no variables/globs/workspace/Python paths. df bytes/inodes precede and follow cleanup; cleanup precedes checkout.
PASS: b1fbc63f is tests-only RED for this repair, tests byte-identical to final source, its workflow equals pre-repair workflow.
PASS: Pages and checker bytes unchanged.
{
  "cleanup_lines": [
    "df -h /",
    "df -i /",
    "sudo rm -rf /usr/local/lib/android /usr/share/dotnet /opt/ghc",
    "df -h /",
    "df -i /"
  ]
}

```

### Environment launch gate

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/env/relational/1792772778161601064/status.json`

```text
{
  "schema": 1,
  "project": "relational",
  "root": "/tmp/lead-wt/v-rel-01",
  "profile_sha256": "fbaecf787ed6355dc2b9f85a78c84012f1a2533147667b4ab827a033bf7a18c1",
  "source": {
    "head": "6900f637951123104d01377120e0c7140b3fd0b6",
    "dirty": true
  },
  "checked_at": 1792772776,
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
  "expires_at": 1792773076
}

```

### First CI failure metadata (historical)

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/remote-ci.json`

```text
{"conclusion":"failure","headSha":"1c47ca9e6104db90d09c5063257041667edeb863","jobs":[{"completedAt":"2026-10-23T16:23:24Z","conclusion":"failure","databaseId":120525336228,"name":"checks","startedAt":"2026-10-23T16:21:10Z","status":"completed","steps":[{"completedAt":"2026-10-23T16:21:11Z","conclusion":"success","name":"Set up job","number":1,"startedAt":"2026-10-23T16:21:10Z","status":"completed"},{"completedAt":"2026-10-23T16:22:29Z","conclusion":"success","name":"Run actions/checkout@v5","number":2,"startedAt":"2026-10-23T16:21:11Z","status":"completed"},{"completedAt":"2026-10-23T16:22:29Z","conclusion":"success","name":"Run actions/setup-python@v6","number":3,"startedAt":"2026-10-23T16:22:29Z","status":"completed"},{"completedAt":"2026-10-23T16:22:50Z","conclusion":"success","name":"Install check dependencies","number":4,"startedAt":"2026-10-23T16:22:29Z","status":"completed"},{"completedAt":"2026-10-23T16:22:50Z","conclusion":"success","name":"Check archived B05 sources for Google API keys","number":5,"startedAt":"2026-10-23T16:22:50Z","status":"completed"},{"completedAt":"2026-10-23T16:23:03Z","conclusion":"success","name":"Verify generated lesson visuals","number":6,"startedAt":"2026-10-23T16:22:50Z","status":"completed"},{"completedAt":"2026-10-23T16:23:23Z","conclusion":"failure","name":"Verify clean-checkout publication inputs","number":7,"startedAt":"2026-10-23T16:23:03Z","status":"completed"},{"completedAt":"2026-10-23T16:23:23Z","conclusion":"skipped","name":"Verify search index","number":8,"startedAt":"2026-10-23T16:23:23Z","status":"completed"},{"completedAt":"2026-10-23T16:23:23Z","conclusion":"skipped","name":"Run unit tests","number":9,"startedAt":"2026-10-23T16:23:23Z","status":"completed"},{"completedAt":"2026-10-23T16:23:23Z","conclusion":"skipped","name":"Post Run actions/setup-python@v6","number":17,"startedAt":"2026-10-23T16:23:23Z","status":"completed"},{"completedAt":"2026-10-23T16:23:23Z","conclusion":"success","name":"Post Run actions/checkout@v5","number":18,"startedAt":"2026-10-23T16:23:23Z","status":"completed"},{"completedAt":"2026-10-23T16:23:23Z","conclusion":"success","name":"Complete job","number":19,"startedAt":"2026-10-23T16:23:23Z","status":"completed"}],"url":"https://github.com/Avistian/relational/actions/runs/39986031745/job/120525336228"}],"status":"completed","url":"https://github.com/Avistian/relational/actions/runs/39986031745"}

```

### First CI failure raw excerpt

`/tmp/lead-wt/process-lessons-v1/.lead/evidence/rel-01-cli/ci-first-failure.log` SHA256 `e584672544207086bd275db97dcf75cb3d1596cdd7b34ba3e93d8bee716739c5`

```text
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2260552Z Traceback (most recent call last):
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2269215Z   File "/home/runner/work/relational/relational/labs/_check_pages_checkout.py", line 32, in <module>
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2270176Z     check()
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2276654Z   File "/home/runner/work/relational/relational/labs/_check_pages_checkout.py", line 27, in check
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2277810Z     raise AssertionError("Clean Git checkout Pages build failed:\n" + result.stderr)
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2278678Z AssertionError: Clean Git checkout Pages build failed:
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2279735Z cp: error copying 'labs/sources/l187/privacybook.pdf' to 'public/labs/sources/l187/privacybook.pdf': Disk quota exceeded
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2281195Z cp: error copying 'labs/sources/l187/source-ledger.json' to 'public/labs/sources/l187/source-ledger.json': Disk quota exceeded
checks	Verify clean-checkout publication inputs	2026-10-23T16:23:23.2282578Z cp: error copying 'labs/sources/l188/2609.02766.html' to 'public/labs/sources/l188/2609.02766.html': Disk quota exceeded
```

### Source status at review

```text
?? .review/
6900f637951123104d01377120e0c7140b3fd0b6
```

</details>
