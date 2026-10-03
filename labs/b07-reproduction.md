# B07 reproduction contract

Approved 2026-10-03. Lesson: semantic transfer, CARTE / ConTextTab / TabSTAR.

## Complete selected course lane: B07-SEMANTIC-ABLATION

Exactly three L074 real wine tables (`wine_pl`, `wine_dot_com_prices`, `wine_vivino_price`), each a pre-existing label-blind 384-row sample after within-source exact name deduplication. No cross-source entity resolution; these are related wine domains. Original row identities and source hashes are in `data/b07/manifest.json`; the entire parent input package is authenticated by `evidence/b07/course-protocol.json`.

Three split seeds 0/1/2 use `default_rng(seed+74).permutation(384)`, 64 train / 64 validation / 256 test, identical across arms. Split seeds measure split sensitivity, not independently pretrained models. The original transformed targets remain unchanged; no claim about error in original currency units.

Three arms: meaningful headers; anonymous headers with all values intact; anonymous headers with all nonnumeric feature columns removed. Original feature slots map to fixed digit names `6,7,...,18,20,0`; use their real cached 300-dimensional FastText vectors. Numeric-only keeps the original slot mapping. Digits retain some semantics; this is one fixed pseudonym scheme, not all possible anonymizations. Removing text also removes categorical predictors and changes graph size. It cannot isolate pure semantic quality. A row with no observed retained numeric features gets a declared zero 300-dimensional representation; no queries are dropped.

Numerical preprocessing is the inherited L074 train-only per-column PowerTransformer; constant observed training columns use StandardScaler, and all-unobserved training columns are omitted. Missing values otherwise preserve graph omissions. Fit once on the same training rows before interventions. No train/validation/test target enters the graph.

Visible CARTE encoder exactly retains L074's graph convention, edge-conditioned center, sender-indexed attention, initial maps and one final readout, with no ordinary hidden graph blocks and no node residual addition. Strictly load the selected `ft_base.*` YAGO tensors. The compact selected checkpoint has bit-identical tensors to the parent. This differs from the release estimator's renamed-key loading and full downstream head/bagging. The encoder is frozen; the only learned task predictor is a Ridge head. Features are standardized using training rows only; choose alpha from `[1,10,100]` by validation MSE, with first-listed ties. Do not refit on validation. Preserve head coefficients/scaler, all candidate validation losses, embeddings and all 6,912 test predictions.

Test metrics: per-run R² and RMSE, per-table R² mean and sample SD over seeds, paired anonymous-minus-meaningful and numeric-only-minus-anonymous differences. SD is split variability, not a confidence interval. Independent auditing solves dual Ridge equations and re-scores using scalar sums, authenticating complete table/seed/arm/query/source identities. Three-domain mean ranks/Friedman/Nemenyi values are exploratory only: three related wine tables do not support an independent-domain significance claim.

Corrections before any complete result report or metric inspection: first attempt found uncached digit19; second found all-missing numeric-only rows. Frozen initial protocols and partial embedding artifacts remain under `evidence/b07/course-protocol-attempt*.json` and `failed-attempt*`. All27 fits restarted with cached labels and the explicit empty-row rule. Failures count toward budget.

```bash
# From repository root; local ledger meters all child execution including failures.
.venv/bin/python labs/_budget_b07.py .venv/bin/python labs/_test_b07.py
.venv/bin/python labs/_budget_b07.py .venv/bin/python labs/_run_b07.py --output /tmp/b07-fresh-runs
.venv/bin/python labs/_budget_b07.py .venv/bin/python labs/_verify_b07.py
```

`_run_b07.py` refuses to overwrite complete evidence. The portable notebook reruns all27 course fits through its visible live functions. This is fresh head training and frozen encoder inference, not fresh CARTE pretraining. A matching rerun is a repeatability check, not additional independent seeds. Hash checks bind local evidence to the frozen contract; they do not independently prove upstream truth.

## Original paper lane: B07-CONTEXTTAB-BAGGING

Target: [ConTextTab arXiv2506.10707v1 Table 2](https://arxiv.org/html/2506.10707v1#S5.T2), binning base versus without bagging on the **complete CARTE subset**, classification accuracy and regression R². Published base values76.0% and71.4%; without-bagging changes−0.4 and−0.4 percentage points. These are separate from the lower half's one-dimensional regression configuration. Published rounded deltas are not exact per-dataset scores.

Archive original June2025 Git commit `f1e4560e28c59dd632a41affb0a6ed4f30af9ebe`, current revision and checkpoint inventory; exact paths/digests in `sources/b07/inventory.json`. Original wrapper defaults to `l2/base.pt` with `regression_type='l2'`; that does not authenticate the Table 2 binning checkpoint. The original tree contains model and inference code but does not establish complete benchmark membership/split identities/evaluator. The current wrapper defaults to a later November2025 checkpoint and samples bags without replacement, whereas original code and paper specify replacement. The SAP-RPT-1-OSS alias is documented by its model card; it does not resolve this historical target.

Status: **INCOMPLETE_SOURCE_PROTOCOL**. Missing: authenticated binning checkpoint/configuration, original complete CARTE membership/splits, evaluator/context/seeds/aggregation. Model-card raw access returned401; no access terms were accepted and no credentials were requested. Missing original protocol is independently sufficient to stop. No paid forecast is meaningful until the exact run is identified; no paid dispatch occurred.

```bash
.venv/bin/python labs/_reproduce_b07.py       # authenticated source preflight
.venv/bin/python labs/_reproduce_b07.py --run # refuses, before any compute dispatch
```

This command is a tested source gate, **not a complete executable original benchmark trainer/evaluator**. Existing source implementation is archived for inspection. It must not be advertised as full paper reproduction. Full model pretraining, original Table 2 inference and whole-paper benchmarks remain **NOT_RUN**. TabSTAR and paper CARTE downstream runs are explained, not newly executed here.

## Budget and delivery

USD 10 aggregate cap; stop new commitments atUSD 8, reserveUSD 2; 3,600 seconds aggregate local numerical cutoff. Count preparation, retries, expected RED tests, verification and failed attempts. Local ledger in `evidence/b07/local-budget.json`; cloud/API USD0. Pin package versions in the protocol; supplied notebook installs missing packages and warns on version drift. Live Colab NOT_CHECKED. No deployment. Author verification never satisfies learner PENDING_WRITTEN_DEFENSE.

License: CARTE source BSD3 (included); FastText derived cache CC BY-SA3.0 with attribution in data README; SAP source Apache2; TabSTAR source MIT. Each archived paper retains its license and attribution.
