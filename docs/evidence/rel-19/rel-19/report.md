# REL-19 builder handoff: BLOCKED

The cached-data cutoff audit passes on unchanged relmini production code.
This task is **not DONE**: genuine TDD RED/GREEN, project verification, and
parent-launched independent outsidecheck/review remain unsatisfied.

## Authority and scope

Read task `spec.md`, answered `grill.md`, and `issues/rel-19.md` in
`/home/avist/agent-stack/lead/.lead/tasks/rel-19`. The grill answer is delegated
Herder authority per its standing-order note, not a fresh Piotr self-answer.
Changes are confined to the two admitted paths. No roles were spawned.
Only the existing `.venv` and cache were used; no training, downloads, fetch,
installs, spend, push, merge, profile changes, or gate changes.

## Builder evidence

Tests-only commit: `e78e3a2a4872e44d92470feb9f3e325aa087271a`.

Command (exit 0):

```
.venv/bin/python -B -m unittest tests.relmini.test_real_leakage -v
```

Two tests passed, zero skipped. `evidence/builder/initial.stdout.txt` and
`initial.stderr.txt` preserve the first actual execution. All labels were checked:

| Cached split | Labels | Labels with history |
| --- | ---: | ---: |
| train | 11411 | 10636 |
| val | 566 | 546 |
| test | 702 | 685 |

The oracle directly filters each driver's source rows with `date < label.date`,
then compares cutoff, count, DNF rate, grid, points, history flag, age, and task
alignment. It does not reuse the cumulative/asof computation or rely solely on
its cutoff metadata. Synthetic equal-time and future-time feature cutoffs are
rejected by the existing public guard. Driver age checks the existing DOB join;
this does not establish historical availability of undated driver metadata.

`evidence/builder/sha256.txt` records frozen test and cache content hashes.
`probe.py` is an evidence-only sensitivity runner, not an implementation helper
used by the tests. Run from the repository root:

```
.venv/bin/python -B docs/evidence/rel-19/rel-19/evidence/builder/probe.py missing
.venv/bin/python -B docs/evidence/rel-19/rel-19/evidence/builder/probe.py mutation
```

Both probe processes exited 0 after checking their expected outcomes. Raw
`missing.*.txt` records one NOT_RUN skip with all five missing files named; its
unittest `OK (skipped=1)` is explicitly **not** audit completion. Raw
`mutation.*.txt` records a real assertion failure when a builder-only in-memory
mutation corrupts the aggregate but retains legal cutoff metadata. This proves
oracle sensitivity, not TDD RED and not independent outsidecheck.

## Unresolved requirements

* **TDD BLOCKED:** the first committed test execution passed existing code.
  There is no genuine failing RED or minimal GREEN change. The issue owns only
  the new test file and evidence path, and declares no `IMPLEMENTATION_FILES`.
  The installed gate requires unchanged canonical tests plus a separate fix.
  No failure was manufactured, no helper was reclassified, and no replay PASS
  is claimed. Parent must resolve this scope/TDD mismatch before acceptance.
* **verify-project NOT_RUN:** read the installed skill and live `lead-verify`
  configuration. `relational` runs full unittest discovery, which includes
  `tests/relmini/test_relmini.py::Determinism` calls to `fit_predict` and model
  training. This conflicts with the task's no-training instruction. Did not
  execute that suite, alter its profile, or call direct tests project VERIFY.
* **Independent outsidecheck / REVIEW NOT_RUN:** reserved for separate counted
  roles launched by the parent. Builder probes are not those roles.

The test remains frozen after its initial commit. Final exact handoff SHA and
artifact hashes are in `evidence/builder/READY.json`, written last after evidence
commit; its name does not imply a passing readiness verdict.
