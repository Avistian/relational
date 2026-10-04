# B19a · Predictive distributions: reproduction contract

Approved 2026-10-04. Named target **B19a-SCORINGBENCH-TABLES-1-4-5**, ScoringBench arXiv2603.29928v3. Paid execution USD0. Aggregate local numerical limit3600seconds covers preparation, failed attempts, retries, notebook and delivery verification. See `evidence/b19a/local-budget.json`. No fresh cloud/model dispatch authorized; standing USD10 ceiling is not a training allocation.

## Completed selected published-table reconstruction

Historical source commit `ca1660023ba43dde460dbadb389b71e274e66ce1` points to output commit `1cc77f6c79f2c88ef206f4b79f81c67861576ce6`. Selected by submission-day history before comparing outcomes. Authenticated all38model Parquet files against Git LFS SHA256 identities, plus three released leaderboard JSON files. The code/history/paper/output archive manifest hashes every file.

All18,480 keyed `(dataset, model, fold)` score records retained for CRPS, R2 and CRLS. Union100dataset names; retained97 after the original policy (model coverage>=90%, then complete dataset intersection). Abalone, Santander_transaction_value and isolet dropped by that policy; case-sensitive names preserved. Every present dataset/model pair has folds0–4; metric-specific missing values remain missing, never zero or imputed. Do not call 18,480 records independent datasets.

Rebuilt fold means, rank/median, MAD, confidence endpoints, effect size and magnitude using explicit local autorank1.3.0, alpha.05. All114model rows and798displayed fields in Tables1/4/5 match within half the last printed decimal (.000500001 tolerance). Full-precision released JSON fields match atol/rtol1e-10. Portable standard-library replay independently reconstructs all mean ranks and medians; it does not recompute autorank confidence/effect statistics. Repository verifier independently ranks raw Parquet values and authenticates every embedded score against them.

Status: **COMPLETE_SELECTED_PUBLISHED_TABLE_RECONSTRUCTION**. This establishes numerical reconstruction of those three tables from archived author scores. It does not establish historical checkpoint/environment identity, raw predictive-distribution correctness, fresh inference/training, or whole-paper reproduction. Dataset-level medians of raw unitful scores mix target units; within-dataset ranks are a distinct summary, and effect sizes alone are not deployment value.

## Original protocol and unresolved differences

- Paper specifies five stratified folds, seed42, one repeat, cap3000total (2400train/600test). Released `scoringbench/runner.py` uses shuffled ordinary KFold on full data, then independently caps train/test per fold. Preserve the difference; no silent correction in historical replay.
- Loader imputes and label-encodes the complete feature table before splitting. This exposes held-out feature statistics. A new valid comparison should fit preprocessing inside training only; that would be a new protocol. No claim here about the size of the resulting performance bias.
- Original wrappers/distribution conversion and metric implementation are archived. Grid/quantile interpolation, clipping and tail extrapolation matter to CRPS/CRLS and intervals. Aggregate scores cannot verify their per-row operation; raw distributions are absent from this selected packet.
- Paper prose reports leading CRPS mean rank about8.55 while Table1 and recovered scores yield9.7319587629. Rank numbering prose differs (0 vs1); tables and code use1. Preserve tabulated results and disclose stale prose.
- Requirements are unpinned upstream. Local environment versions are recorded in `evidence/b19a/reproduction.json`; matching table output does not recover the historical environment.
- Archived source is a source snapshot, not authentication of original row-level split IDs or model checkpoint bytes. Finetuning, hyperparameter search and all other paper figures/ablation runs remain NOT_RUN.

## Frozen course protocol: B19a-SAME-MEAN

Exact finite forecasts: narrow{-1:.5,1:.5}, calibrated{-2:.25,0:.5,2:.25}, wide{-4:.5,4:.5}; common mean0. Truth{-2:.25,0:.5,2:.25}. Enumerate all9forecast/outcome pairs; report truth-weighted expected squared loss (then sqrt), CRPS, central50% interval score, closed-endpoint coverage and width. Quantiles are the left generalized inverse CDF at.25/.75. Coverage can exceed nominal for discrete outcomes even for the true distribution. All15probability triples at step.25 on{-2,0,2} form the finite strict-propriety diagnostic. Exact CDF-integration oracle is separate from the pairwise-expectation implementation. Eighteen affine tests cover translation and10x unit scaling of CRPS/interval score.

Calibration exercise: predictor fixed at0 before calibration; nine calibration IDs c0–c8 have residuals0–8, alpha.2, k=ceil(10*.8)=8, radius7. Four untouched test IDs t0–t3 have outcomes0,7,8,20. Observed coverage.5; adding100to every test outcome changes coverage to0without changing fitted radius. This deterministic fixture demonstrates data-flow separation; it is not a statistical validation of a conformal theorem. Marginal split-conformal coverage needs exchangeable calibration/new residuals and a predictor fixed independently of their labels. It is not per-subgroup/conditional coverage or protection against arbitrary shift. If k>n, radius is infinity; do not replace it with the largest finite residual silently.

## Run and evidence boundaries

From repository root, use `.venv/bin/python labs/_budget_b19a.py .venv/bin/python labs/_test_b19a.py` (and `_run_b19a.py`, `_verify_b19a.py`, `_build_b19a.py`, `_execute_b19a.py`, `_delivery_b19a.py`, `_pages_b19a.py`). The budget wrapper accounts attempts and stops at the aggregate cap. Serial calls only. Authenticated fetch scripts can reuse existing bytes. Historical table stats need autorank1.3.0 with statsmodels0.14.5, patsy1.0.2 and baycomp1.0.3; local isolated install at `/tmp/b19a-python-deps` is optional to the replay script (normal installed packages also work). Student/solution notebook needs only Python standard library for numerical work, with embedded score packet and figures; no download, repo import or credential.

`_reproduce_b19a.py --fresh` refuses execution because fresh-model protocol identities and compute have not been approved. Notebook replay is saved-score reconstruction; repository additionally authenticates original extraction and full table statistics. Browser/Pages packaging, live Colab, deployment and learner mastery are separate statuses. Learner remains PENDING_WRITTEN_DEFENSE.
