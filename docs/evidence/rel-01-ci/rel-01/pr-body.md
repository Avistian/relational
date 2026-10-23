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
