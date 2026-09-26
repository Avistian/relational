# L109 reproduction contract

Complete selected experiment: Robinson et al., arXiv:2407.20060v1 Table 4, rel-f1/driver-position, all five heuristic columns on validation and test. Extended upstream evidence: reconstruct all released task queries and labels from all nine released database tables. This is not the entire paper.

## Run

From the repository root with Python 3.12:

```bash
.venv/bin/python -m pip install -r labs/requirements-l109-authoring.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_verify_l109.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_graph_l109.py
.venv/bin/python labs/_check_l109.py
.venv/bin/python labs/_audit_l109.py
.venv/bin/python labs/_figures_l109.py
.venv/bin/python labs/_build_l109.py
.venv/bin/python labs/_execute_l109.py
.venv/bin/python labs/_delivery_l109.py
.venv/bin/python labs/_provenance_l109.py
.venv/bin/python labs/_record_l109.py
```

For only the complete numerical reconstruction, the standalone solution notebook authenticates and downloads both archives, visibly implements the label generator and estimators, and runs every selected label and score. No hidden `relkit` import, GPU, paid account or L101 cache is required. Original-source scheduling/SQL parity and the full SQL graph census are separate author checks above. The lesson's three student functions drive historical graph, event eligibility and maturity checks; the real F1 archive lacks the fields needed to exercise those two-clock functions on genuine ingestion history.

## Frozen protocol

| Dimension | Contract |
|---|---|
| Source | RelBench v1.1.0 commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`; original source, MIT license and per-file SHA-256 manifest in `sources/l109/` |
| Database | Public `https://relbench.stanford.edu/download/rel-f1/db.zip`; SHA-256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`; nine tables, 97,606 rows including post-test outcomes |
| Task | Public `https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip`; SHA-256 `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e` |
| Labels | Mean `results.positionOrder` for each driver in `(query, query+60 days]`; reproduce source population unchanged |
| Schedules | Train 332 scheduled / 254 nonempty windows; validation 30 / 23; test 40 / 33. Independent schedule compared via full original `BaseTask._get_table` execution |
| Database scope | Original Dataset loads full archive; original `_get_table` censors train/validation database at 2010-01-01. Test label generation reads post-cutoff outcomes. These outcomes are never predictor inputs |
| Complete query rows | Train 7,453; validation 499; test 760. Full identity comparison on `(date, driverId)`; original archive row order restored before scoring |
| Fitting | Train labels for validation; train+validation labels for test. Every fitting window ends by its fit boundary under release immediate-availability assumption |
| Methods | Global zero, global mean, global median, entity mean, entity median; unseen entity prediction 0 |
| Architecture/optimizer/objective/initialization/epochs | No neural model or optimizer; deterministic location estimators. No training schedule applies |
| Preprocessing | Released stable primary/foreign keys and table clocks; no learned encoder. Predictor sees only query driverId/date and permitted fit labels |
| Selection/seeds | No hyperparameter search or test-based method choice. Deterministic: no seed dispersion is claimed |
| Metric | All-row mean absolute error, not per-driver average. Ten fixed paper cells; absolute tolerance 0.0005 for three-decimal rounding |
| Source execution | Original Table/Database/Dataset/BaseTask/EntityTask modules; unchanged DriverPositionTask class and baseline `evaluate` AST segments. Metric callback captures predictions; unused class metric declarations are placeholders. This is source-function replay, not a full historical environment |
| Independent comparison | Pandas label reconstruction vs unchanged source DuckDB SQL and cached targets; 10 vectors vs original baseline; scalar metric reconstruction; graph arrays vs SQLite joins |

## Results

Every one of 8,712 targets matches both the source SQL and cached labels with maximum absolute error 0. All ten baseline vectors match source to absolute tolerance 1e-12. All ten rounded paper cells MATCH. Details: `_verify_l109_results.json`, `evidence/l109/reproduction.json`, full regenerated CSVs and `predictions.npz`.

| Method | Validation MAE | Test MAE |
|---|---:|---:|
| Zero | 11.083200 | 11.926206 |
| Global mean | 4.334385 | 4.512881 |
| Global median | 4.135571 | 4.399101 |
| Entity mean | 7.181002 | 8.501358 |
| Entity median | 7.114262 | 8.518509 |

All 310 distinct released query cutoffs, 13 FK relations each: 4,030 independent SQL graph comparisons. Each full ordered edge array agrees exactly. Snapshot summaries/hashes are in `evidence/l109/graph-census.json`. The census compares the release-style single-clock policy; it does not demonstrate real arrival-time validity.

## Deviations, source quirks and unrun work

- Driver task population is conditioned on at least one future-window result. The released one-year activity filter has no upper bound. A true prior-year filter would remove 955 train, 33 validation and 42 test query rows; these counts are a diagnostic, not a repaired benchmark score.
- Race timestamps are assigned to results/standings; qualifying uses race time minus one day. Actual outcome publication/ingestion times are unavailable. A scheduled race time is not proof of result availability.
- Undated driver/constructor/circuit tables remain visible at every cutoff under the release convention. Static fields may still require field-level historical provenance. PKs are stable indices in the released database, not original source identifiers.
- A timed child can point to a parent absent from the query snapshot. The course graph materializer keeps the child and omits/counts the edge. It is a deterministic schema handoff, not a replay of a neural sampler or trainer. Versioned synthetic edges use both selected endpoints and the selected child FK.
- Histories of arrivals, corrections and deletions are absent. Synthetic two-clock/version tests are separately labeled course evidence. Effective-step semantics are bounded; overlapping valid intervals and future planned-state features need additional contracts.
- Readiness `max(horizon end, constituent arrivals)` assumes a completeness certificate. It cannot detect unseen late outcomes. Later label corrections need their own version policy.
- Current runtime is Python 3.12.3, pandas 3.0.3, numpy 2.5.0, PyArrow 24.0.0 and DuckDB 1.5.5. An initial DuckDB 1.4.3 attempt rejected pandas 3 string dtype; no benchmark result came from that failed attempt. Original historical runtime/checkout identity is NOT_ESTABLISHED.
- Raw Kaggle-to-database reconstruction, other tasks/datasets and LightGBM/RDL columns are NOT_RUN. Full-paper parity is NOT_ESTABLISHED. No GNN is trained here.
- Author checks and solution execution do not certify learner mastery: PENDING_WRITTEN_DEFENSE. Live Colab and deployment are NOT_CHECKED. Browser/copied-Pages checks have their own delivery report.

## Budget and artifacts

Local CPU only; USD0 paid compute, within the USD10 aggregate ceiling. A 600-second per-experiment cutoff is enforced in documented commands. No paid retries. Small public archives total under 1 MB; the larger course repository is not a notebook prerequisite. Source/data hashes and per-run evidence are explicit; source parity is checked independently from the numerical paper targets.

`_check_l109.py` tests 520 SQL historical selectors and boundary cases. `_audit_l109.py` checks 390 independent synthetic SQL graphs, five deliberate broken variants, and all ten metrics from saved predictions. `_execute_l109.py` executes in a clean directory and records the executed-code hash. `_delivery_l109.py` checks code identity, rendered diagrams, native controls, mobile/print/no-JS, and copied Pages navigation.
