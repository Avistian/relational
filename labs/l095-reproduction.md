# L095 reproduction contract: bipartite graphs

## Published target and executed coverage

The roadmap assigns no model paper to this representation/evaluation lesson. The named source target is the **GroupLens MovieLens 100K README**, SUMMARY and file descriptions for `u.data` and `u1.base`–`u5.test`. The full release audit recomputes 100,000 unique rating pairs, 943 users, 1,682 items, minimum 20 ratings/user, five 80,000/20,000 partitions, exact row reconstruction and mutually disjoint test sets covering the full data. All targets **MATCH**. No sampled subset stands in for the full release.

Primary data source: https://files.grouplens.org/datasets/movielens/ml-100k.zip

Archive SHA-256: `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`.

Member hashes and source locators: `_sources_l095.json`. Acknowledge GroupLens and Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, DOI 10.1145/2827872. Download data from GroupLens; the archive and extracted data are not distributed by this package.

## Complete course experiment

All five official splits were executed without row or catalog downsampling. Results in `_experiment_l095_results.json` are **MEASURED course evidence**, not a historical recommender-paper score. The default notebook executes this same full experiment. There is no cheaper default substituting for a larger promised run.

| Field | Exact contract | Published/source versus course choice |
|---|---|---|
| Data/splits | Entire ML-100K; original u1–u5 files | Release-defined, byte-pinned |
| IDs | Separate zero-based user/item spaces, fixed 943/1682 catalog | Course representation; known IDs are transductive |
| Graph | Binary likes for rating ≥4; exact reverse view | Course task transformation |
| Observed mask | All fitting-rated items excluded, including ratings <4 | Course ranking rule |
| Inputs | Only fitting like edges; no user demographics, genres, title text or time features | Course choice |
| Scorer | P=row-normalized B; Q=row-normalized Bᵀ; S=PQP | Fully visible deterministic course walk |
| Baseline | Fitting like-degree popularity, normalized to sum one | Course choice |
| Cold policy | Zero-degree users use popularity; if no likes, uniform catalog; unseen items get zero mass | Course choice |
| Selection | 10% of each base reserved by NumPy RNG seed 9500+fold; α∈{0,.5,1}; maximize macro validation NDCG@10, ties lower α | Course choice, not source-reconstructed |
| Refit | Recompute graph/degrees/transitions on all 80,000 base rows after selecting α | Course choice |
| Model training | No neural parameters, initialization, gradients, loss, optimizer or epochs | Not applicable to this count-based scorer |
| Relevance | Held-out ratings ≥4 | Course target; unknown items are not established dislikes |
| Candidates | Every catalog item except the user's fitting-rated items; no sampled negatives | Course choice, fixed full catalog |
| Evaluation | Macro user Recall@10 and binary NDCG@10; only users with ≥1 relevant item; exclusion counts explicit | Course choice |
| Ties | Descending score, ascending item ID | Course choice |
| Summary | Arithmetic mean over five folds, sample fold SD (ddof=1) | Descriptive variability, not dataset-level uncertainty |
| Time | Official offline splits; no global timestamp cutoff | Not a next-event forecast |

Mean Recall@10: popularity **0.1253868170**, walk **0.2259884679**. Mean NDCG@10: popularity **0.2045527829**, walk **0.3278858813**. All folds selected α=1, so the selected mixture coincides with the walk. No broad model-superiority claim follows from one dataset. Source code, environment, validation values, counts and timings are recorded alongside results. Rankings live in `results/l095/fold-*.json`; inner arrays are regenerable local `.npz` artifacts. The independent auditor reconstructs each saved user metric without calling the scorer or evaluator.

## Reproduction and deviation ledger

- Published release counts and all five partitions: **MATCH**.
- Full defined course experiment: **MEASURED**, five of five folds executed.
- Full model-paper reproduction: **NOT_APPLICABLE_NO_MODEL_PAPER**. No model paper is assigned or claimed. Historical model-score parity remains **NOT_ESTABLISHED**; no corresponding historical target exists in this lesson.
- The threshold, walk, selection and ranking protocol are course choices. A published dataset name does not make these published model results.
- Generalization to another dataset, cold-start item features, learned GNN comparison and temporal forecasting: **NOT_RUN** extensions, not missing runs within the defined experiment.
- Live Colab and deployment: **NOT_CHECKED**. Local inline execution and browser checks have separate evidence files.
- Learner mastery: **PENDING_WRITTEN_DEFENSE**.

## Regenerate from repository root

Use the author environment in `requirements-l095-observed.txt`, or the separately tested portable pins in `requirements-l095-runtime.txt`. The main lab requirements already include NumPy, Torch and PyG.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l095.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l095.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_audit_l095.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_figures_l095.py
.venv/bin/python labs/_build_l095.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_execute_l095.py
.venv/bin/python labs/_build_l095.py
.venv/bin/python labs/_delivery_l095.py
```

The executor expects the existing `relational-labs` Jupyter kernel. Alternatively open the standalone solution using your own Python kernel, or use `_portable_l095.py` with the portable environment. All functions are visible in both notebooks; the student has three blank functions and immediate CHECK cells. Rebuilding preserves solution outputs only if ordered executable cells have identical contents. The runner starts fresh; there is no opaque checkpoint-resume cache.

## Delivery and portable environment evidence

The standalone solution executed all 18 code cells and freshly reran all five folds, matching every saved ranking and metric exactly. A second clean Python 3.12 environment with NumPy 2.2.6, Torch 2.8.0 and PyG 2.6.1 reproduced those same results. The transitive package lock is `requirements-l095-portable-lock.txt`; this is a local interpreter check, not a live Colab claim.

```bash
uv venv --python 3.12 /tmp/l095-portable-env
uv pip install --python /tmp/l095-portable-env/bin/python -r labs/requirements-l095-portable-lock.txt
/tmp/l095-portable-env/bin/python labs/_portable_l095.py
```

Behavioral tests cover typed identities, isolated nodes, duplicate rejection, reverse corruption, independent path enumeration, cold-user fallback, candidate masking, metric denominators, ties and empty relevance. The independent full-result audit reconstructs 11,151 user metric records, including the selected-arm records that duplicate the walk. These are not independent users or datasets. Real browser checks cover 1200px/375px, both interventions, reset/keyboard/print, notebook images and copied Pages links; see `_delivery_l095_results.json`.
