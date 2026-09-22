# L101 reproduction contract

The lesson is a temporal-protocol bridge, not a new GNN architecture. The complete named quantitative target is **Robinson et al., arXiv:2407.20060v1 Table 4, rel-f1/driver-position, all five heuristic columns on validation and test**. The theory source is Fey et al., ICML 2024 §3.2–3.3 and Appendix A–B.

## Execute

From the repository root, with Python 3.12 and `labs/requirements-l101-runtime.txt` installed:

```bash
.venv/bin/python labs/_check_l101.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l101.py
.venv/bin/python labs/_figures_l101.py
.venv/bin/python labs/_build_l101.py
.venv/bin/python labs/_execute_l101.py
.venv/bin/python labs/_delivery_l101.py
```

Notebook/build tooling additionally requires nbformat, nbclient, nbconvert, ipykernel and cairosvg; browser delivery uses Playwright Chromium. Those are course authoring tools, not runtime dependencies of the standalone lesson experiment. The standalone solution visibly includes every implementation function and downloads its checksum-verified data into its own cache. It executes the entire selected paper slice by default; there is no substitute “paper” preset or hidden course import.

## Frozen protocol and audit

| Dimension | Exact contract |
|---|---|
| Source | RelBench v1.1.0, commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`; archived baseline/task source and MIT license in `sources/l101/` |
| Task data | `https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip` |
| SHA-256 | `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e`, checked against release hash registry |
| Complete rows | Train 7453, validation 499, test 760; archive order preserved |
| Label | Average finishing position in `(t,t+60 days]`; released targets read as provided |
| Cutoffs | Validation 2005-01-01; test 2010-01-01; every fitting label window must end by the respective boundary |
| Methods | Global zero, global mean, global median, entity mean, entity median |
| Fitting | Train labels for validation; train+validation labels for test, exactly as released |
| Cold starts | Entity methods predict zero when no fitting label exists for a driver |
| Inputs | Only query driverId/date passed to predictors; target read by evaluator |
| Preprocessing | None beyond numeric targets and driver grouping |
| Optimizer / architecture / initialization / epochs | Not applicable to deterministic heuristic estimators |
| Search / selection | None: replay all five predeclared methods; no choice based on test |
| Metric | Mean absolute error over all query rows; no per-driver averaging |
| Seeds / uncertainty | Deterministic baselines; five repeated runs would give identical values, not uncertainty |
| Paper tolerance | Absolute difference ≤0.0005 for values printed to three decimals |
| Source parity | All ten complete prediction vectors compared with unmodified released `evaluate` function at absolute tolerance 1e-12 |
| Results | Ten of ten cells MATCH; full vectors saved in `_predictions_l101.npz` |

## Deviations and evidence boundaries

- The pinned release is a reproducible source choice, not an identified original experiment checkout. Historical execution identity remains **NOT_ESTABLISHED**.
- Target tables are reproduced from hash-pinned cached task bytes. The source SQL is archived and explained; database extraction, original Kaggle data and task-label regeneration are **NOT_RUN**.
- No real temporal GNN is trained in this lesson. LightGBM/RDL columns, other datasets and other tasks are **NOT_RUN**. Full-paper reproduction remains **NOT_ESTABLISHED**.
- Fey's Algorithm 1 prints a receiver timestamp filter; Appendix A's neighborhood equation filters the source neighbor. The lesson follows the latter. Independent counterexamples reject receiver-only filtering.
- The graph mechanism uses all-neighbor expansion, explicit edge availability and two node clocks. It is a transparent course visibility implementation, not a replay of a particular finite-fanout sampler or full learned RDL model.
- Historical feature versioning is taught but not implemented as a database storage engine. Synthetic records are immutable. No claim of full production PIT correctness follows from these bounded checks.

## Separate course experiment

Three seeds × three visibility rules = nine fresh logistic-regression fits on 600 deliberately constructed queries each. Labels are independent Bernoulli draws; the legal past value is zero. Two records deliberately carry the answer: a late-arriving past event and a future event. C=1, lbfgs, max_iter=200; no tuning. Identical chronological targets across the three arms; 359/119/120 train/val/test, two purged rows. Legal accuracies 50.00%, 45.00%, 38.33%; both illegal rules 100% at every seed. These are a leakage witness, **INCOMPARABLE** to paper scores and not a real-world effect-size estimate.

Random partition audits count training labels that mature after test query times; no cross-partition accuracy comparison is made. All-neighbor graph checks include the named delayed/future two-hop fixture and 32 random graphs × 3 cutoffs × 4 depths = 384 independent SQL-filtered traversal comparisons.

## Artifacts and delivery

`_sources_l101.json` pins source/data identities. `_verify_l101_results.json` records environment, source and prediction hashes. `_paper_l101_results.json` records all target gaps; `_experiment_l101_results.json` preserves fitted parameters and test predictions. `_check_l101_results.json`, `_execution_l101_results.json` and `_delivery_l101_results.json` describe distinct verification layers. Browser and notebook completion do not establish live Colab or deployment: both remain **NOT_CHECKED** unless new evidence changes them. Learner status is **PENDING_WRITTEN_DEFENSE**.
