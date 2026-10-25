# relmini: point-in-time baseline for RelBench rel-f1 `driver-dnf`

## Data: REAL vs SYNTHETIC
| Where | Data | Label |
|---|---|---|
| `baseline.py`, `verify.py` | RelBench **rel-f1** `driver-dnf` (train/val/test) plus `drivers`, `results`, `races` tables from the **local** cache `~/.cache/relbench/rel-f1`, read-only. | **REAL** |
| `tests/relmini` (all but `RealData`) | Generated in-memory by `synthetic()` in the test file. | **SYNTHETIC** (test fixtures only) |

Each file is sha256-pinned in `verify.py` (`PINS`); the pins come from the lead env profile
(`config/verify-projects.json`, project `relational`). Nothing is downloaded and no network is used.

## What it does
- `baseline.build_features`: for each task row (driver, date) it aggregates that driver's `results` rows with
  **date strictly before** the task date (prior races, DNF rate, mean grid, mean points) plus age from `dob`.
  `feature_cutoff` is the latest result date used; `assert_no_leakage` requires `feature_cutoff < date`.
  A prior "DNF" is a proxy (no classified `position`), not RelBench's own label definition.
- Model: `StandardScaler` + sklearn `LogisticRegression`, seed 0, trained on `train`, scored on `val` and `test`.
- `verify.py`: checks hashes, runs the baseline twice and requires identical metrics, writes `receipt.json`.

## Run
```
.venv/bin/python labs/relmini/baseline.py     # prints val/test AUROC
.venv/bin/python labs/relmini/verify.py       # PASS + receipt.json
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Limits
Metrics are **local reproducibility evidence only**. They are **not RelBench-comparable** and make no
leaderboard or benchmark claim (different feature set, no official evaluation harness used).

## Constructor feature pair (REL-18)

`ablation.py` runs the original six-feature baseline and the same model with three
constructor features added: number of prior constructor-result records, mean
points per prior record, and a history indicator. Disabling these features is the
constructor ablation and reproduces the original baseline predictions.

For each label, resolve the driver's constructor at the latest **strictly earlier**
result date. Multiple distinct teams at that date mean unknown (no arbitrary team
choice and no fallback to an older team). Join `constructor_results.parquet`
strictly before the label date, aggregating points and count. Unknown teams and
missing constructor history use zeros. These are historical-team proxy features,
not knowledge of the label-day team. The assignment timestamp and aggregate
cutoff are both checked by `assert_no_leakage`. Original driver features, split
membership/order, seed 0 and model settings are preserved.

Use the existing installed interpreter; all input hashes must match before any
fit. No install, network, download, paid compute, or cache writes are involved.
Use a fresh output directory for each invocation; existing evidence is refused.

```sh
PY=/home/avist/Projects/relational/.venv/bin/python
timeout --kill-after=5 120 "$PY" -B labs/relmini/ablation.py --smoke --output /tmp/relmini-smoke-NEW
# Proceed only after smoke PASS; stop on absent prerequisites (NOT_RUN).
"$PY" -B labs/relmini/ablation.py --output /tmp/relmini-full-NEW
"$PY" -B labs/relmini/verify.py
"$PY" -B -m unittest tests.relmini.test_ablation tests.relmini.test_real_leakage -v
```

Smoke uses the first 128 cached labels per split, with complete cached histories;
it is a runtime check, not the full baseline. Full runs retain every original
label row. Each pair repeats each fit and requires exactly identical predictions,
then saves both raw probability columns, labels, ordered split CSVs, feature/cutoff
CSVs, input/split/artifact hashes, seed, model parameters, software versions,
source file hashes and runtime in `receipt.json`, written last. The original
`verify.py` still verifies the original baseline and writes its own receipt.

Actual builder outputs are under `docs/evidence/rel-18/rel-18/`; see its report for
source identities, successful and failed smoke attempts, and remaining review
gates. **LOCAL reproducibility evidence only; not RelBench-comparable.** A single
seed, proxy historical constructor and unofficial features/evaluation do not
support a scientific improvement or benchmark claim.
