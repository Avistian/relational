# B19 reproduction contract

Approved 2026-10-04: USD0 paid, 3600 aggregate local numerical seconds. Includes preparation, retries, verification and notebook execution. Standing USD10 total is a ceiling for a separately specified/approved fresh experiment, not authorization to dispatch one. No deployment requested.

## Three distinct results

1. **B19-EVIDENCE-BOUNDARIES-v1:** complete finite course diagnostic, 3 seeds × 2 split regimes × 2 fixed predictors = 12 arms / 672 predictions; 21 score cells across two missing-result policies; 27 dataset bootstrap resamples and 729 three-row resamples. Synthetic group memory is not a foundation model. No HPO or learned neural model. Random and grouped test populations differ. The signal predictor wins only two of three grouped cases. The uncertainty fixture is separate from the grouped fixture: three seeded draws of one generator are not three independent real datasets.
2. **Released grouped score-table audit:** 9,540 complete original-suite metric rows, 60 musk folds and 30 SAT11 folds; four classical models have 26 configurations each, two TFMs have one. Authenticate all source bytes, compare extracted rows against original Parquet files and reconstruct twelve default means. No raw prediction rescoring, fresh fits, paper-figure parity or statistical independence of overlapping folds is established.
3. **B19-BEYONDARENA-FIG-F2:** arXiv2606.30410v1 Appendix F.2, two datasets × grouped/IID × six model families. INCOMPLETE_SOURCE_PROTOCOL_GATE; complete figure NOT_RUN. No IID counterparts occur in the six fetched original-suite model result tables. Full original plotting/variant selection, model checkpoints and per-query predictions remain unauthenticated. Current source has both IID curation notebooks, so code is not wholly missing. Searches are bounded and do not prove that artifacts are unavailable elsewhere.

## Paper target to preserve

- Datasets: musk and sat11_hand_algo_runtime, published task construction, grouped and IID outer splits, original group/sample scoring units.
- Models: Linear, ExtraTrees, LightGBM, RealMLP, TabPFN-2.6, TabICLv2; retain every displayed default/tuned/ensemble variant, not just family labels.
- Preserve complete original outer folds/repeats. Do not silently use current `core` or `lite`.
- Preserve inner validation (including grouped and small-data rules), original 25 random configurations plus defaults where applicable, preprocessing, candidate selection, refits and ensemble construction. Do not select models by test scores.
- Reproduce original metric-error definitions, aggregation, rank handling/ties and Kendall correlations, with exact inputs before comparison. Published F.2 captions report tau 0.60 for musk and 0.49 for SAT11; these are source-reported targets, not B19 measured results.
- DataFoundry notebooks use random UUID renaming before sorting: rerunning a notebook alone is not byte-identical historical data. Stored split/data identities are required.
- Pinned source: TabArena 1ce4cae6c12972227dea6247bac83234a2915748; DataFoundry 72c30d48d26c01fba87b49efbc6f5d6ef67d2049. Original-suite cache bytes are hashed at download, not proven immutable historical artifacts.

## Commands (repository root)

```bash
.venv/bin/python labs/_budget_b19.py .venv/bin/python labs/_test_b19.py
.venv/bin/python labs/_budget_b19.py .venv/bin/python labs/_run_b19.py
.venv/bin/python labs/_budget_b19.py .venv/bin/python labs/_verify_b19.py
.venv/bin/python labs/_budget_b19.py .venv/bin/python labs/_reproduce_b19.py --phase audit
.venv/bin/python labs/_reproduce_b19.py --phase paper
```

The last command deliberately refuses; it is not an implemented benchmark trainer. `_sources_b19.py`, `_fetch_results_b19.py` and `_audit_sources_b19.py` document discovery/extraction. Rerunning acquisition fetches potentially changed release bytes: treat that as a new source version and compare manifests before accepting it. The portable notebook embeds the selected score records and authenticates that compact packet; repository audit additionally checks every selected record against the full original tables. These are different verification boundaries.

## Frozen course cases

24 groups × 8 observations, balanced group labels by seeded permutation, noisy binary signal correct with probability .7. Seeds0/1/2. Random regime: hold out two rows per group. Grouped regime: hold out eight whole groups. Group-memory predicts training group prevalence or overall training prevalence for unseen groups; signal predicts .2 or .8 from the feature. Brier loss averages squared probability error per row. All inputs, row IDs and probabilities retained. Group labels enter prediction only through training records.

Fixed missing-score table uses A=[.1,.1,.1,missing], B=[.2,.2,.2,.2], RF=[.25,.25,.25,.9]. The means are commensurate constructed losses. Do not average raw real-world AUROC errors and RMSE across datasets. Common measured support has three datasets; explicit RF substitution has four. Mean-loss ordering changes; neither is an estimate of the missing A measurement.

Uncertainty uses differences A−B: a=[.10,.11,.09], b=[−.04,−.05,−.03], c=[.02,.03,.01]. Average seeds first, then equally weight datasets. Enumerate 27 three-dataset bootstrap draws. Contrast 729 three-row resamples with the same draw count of three only; this is NOT the ordinary nine-row bootstrap. Separately compare sample standard deviation / sqrt(count) using three dataset means versus nine seed rows, and duplicate seed rows with new IDs to expose pseudo-replication. All intervals/SEs are descriptive fixture calculations; real datasets may themselves be dependent.

Learner PENDING_WRITTEN_DEFENSE. Live Colab NOT_CHECKED. Whole paper/pretraining/fresh benchmark training NOT_RUN.
