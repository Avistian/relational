The PR31 unit discovery imports `bs4` through both visual contract modules, but the hosted runner did not install its distribution. Add `beautifulsoup4==4.15.0` to the existing check dependencies, matching `requirements-labs.lock.txt`.

The unchanged workflow regression fails before the dependency addition and passes afterward. No assertion, skip, discovery filter, or permission was weakened. Production change: one workflow line.

- Source: `76798ba858f873aca043a9031f833d5e6c0cd26c`; RED: `2aaf9e44570e2c4d23b91b31db98d82809b0b674`.
- Configured `/verify-project`: PASS, 73 tests, exit 0, clean source, no skips; all five visual contract tests also passed independently.
- Local discovery inventory includes the three CI tests and all five visual contract tests.
- Canonical replay: one assertion failure in RED, zero failures in identical GREEN; zero errors/skips in both.
- [Builder evidence](docs/evidence/rel-02-ci-r1/rel-02-ci-r1/evidence.md), [manifest](docs/evidence/rel-02-ci-r1/rel-02-ci-r1/manifest.json), [RED](docs/evidence/rel-02-ci-r1/rel-02-ci-r1/red.log), [GREEN](docs/evidence/rel-02-ci-r1/rel-02-ci-r1/green.log), [verification stdout summary](docs/evidence/rel-02-ci-r1/rel-02-ci-r1/verification.log).
- Scientific experiments: NOT_RUN (outside this software repair).

Actual failed [hosted run 40189655234](https://github.com/Avistian/relational/actions/runs/40189655234), extracted unchanged from the existing GitHub CLI cached raw archive (SHA256 `6c8bfd425c3be65925221f015463819ab57c74e51b3157fac6894c125e4933aa`):

```text
2026-10-25T11:40:36.8063887Z ERROR: visual_contracts.test_solution_map_consumers (unittest.loader._FailedTest.visual_contracts.test_solution_map_consumers)
2026-10-25T11:40:36.8082175Z ModuleNotFoundError: No module named 'bs4'
2026-10-25T11:40:36.8083833Z ERROR: visual_contracts.test_solution_routes (unittest.loader._FailedTest.visual_contracts.test_solution_routes)
2026-10-25T11:40:36.8093993Z ModuleNotFoundError: No module named 'bs4'
2026-10-25T11:40:36.8095045Z Ran 88 tests in 11.724s
2026-10-25T11:40:36.8095393Z FAILED (errors=2, skipped=1)
```

The complete archived job, unit-test log, dependency log, archive copy, and extraction provenance are retained in `/home/avist/agent-stack/lead/.lead/tasks/rel-02-ci-r1/evidence/current-build`. Direct `gh-axi` retrieval failed with `error connecting to api.github.com`; no network retry was made. The hosted count of 88 and its existing one skip are historical output, not a claim about this source's local 73-test result.

Pending: parent-launched separate counted `/blast-radius` verifier; replace the explicit pending reviewer in the manifest with that actual review identity and finalize acceptance evidence. Parent then integrates this reviewed descendant into existing PR31 without force and observes actual hosted CI. Hosted CI for this repair is NOT_RUN. This builder did not push, open a PR, or merge.
