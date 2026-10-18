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
