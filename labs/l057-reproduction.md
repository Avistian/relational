# L057 reproducibility contract

Scope: a binary OOF probability stack and Caruana-style greedy combiner, implemented
from scratch; the L054 numeric TabM is visible and reused. XGBoost is a provided base
learner. TabICL is a provided pretrained API baseline: its architecture, pretraining
and original paper training results are outside this ensemble lesson.

## Exact measured local procedure

- Datasets: OpenML diabetes 37, blood_transfusion 1464, phoneme 1489. Numeric small
  Tier A substitutes, not the TabArena 51-task roster. Labels map to 0/1 by the course loader.
- Label-blind cap750 with RNG57; 75/25 stratified development/test split seed57;
  3 stratified OOF folds seed57. Original row IDs and exact parquet hashes are in
  `_data_l057.json` and `_verify_l057_results.json`.
- Fitting seeds 0/1/2 (per-fold seed = 100*seed+fold). Same row/fold partition in all runs.
- Each OOF row is excluded from its predictor's preprocessing, context and fitting.
  Fixed training lengths; no early stopping or HPO. No OOF labels inside base fitting.
- XGB: 120trees, depth3, lr.05, row/column subsampling.8, histogram algorithm, one CPU thread.
- TabM-mini: 3 hidden blocks, width48, k8, dropout.1, Adam lr.002, no weight decay,
  32epochs, batch128 including last incomplete batch, no early stopping.
- XGB/TabM preprocessing: fit median imputer and z-score scaler only on fold-fit rows.
  TabICL handles its own transforms on fold-fit rows. No categorical model path measured.
- TabICL package0.1.4, v1.1 checkpoint `tabicl-classifier-v1.1-0506.ckpt`, exact revision,
  URL and SHA256 in `_sources_l057.json`. One transformed view vs default32;
  float32 inference, CPU, one job; no fine-tuning. Context is fold-fit rows only.
- OOF predictions: one per row/family. Test: mean of the same three fold-model
  predictions within family. This also bags TabICL, unlike the paper's refit policy.
- Metric: binary log loss in natural-log units, float64 reduction, clipping1e-7.
  TabArena uses 1−AUC for binary tasks, so scores are not directly comparable.
- Combiner: 40 greedy selections with replacement, best OOF prefix. Numerical ties
  within1e-12 select first column/prefix. No sorted initialization or bagged libraries.
- Comparator: OOF-lowest-loss family, selected before test scoring. Individual test
  winners are descriptive oracle choices. The uniform mixture uses equal family weights.
- 81 fold-family fits/context constructions, 446.3 CPU wall seconds for the main
  three-family run. Recorded installed versions accompany the result. Source hashes
  at training time and a postrun numerical-tie audit are distinguished in the result.
- Committed `data/l057/*.npz` contain OOF/test probabilities and binary labels, not raw
  feature rows or pretrained weights. Corresponding fold audits and all hashes included.

## Evidence and limits

| Bucket | Status | Evidence |
|---|---|---|
| Local three-family training | RUN | 3 real datasets × 3 seeds × 3folds × 3families |
| Archived prediction audit | MATCH | Recompute every score, weight and trace using visible code |
| Pinned selector primitive | MATCH | Five non-tied fixtures; upstream classes executed with local log-loss adapter |
| TabArena Figure6 | CITED / INCOMPARABLE | Different data, roster, metric, budgets, transforms and TFM inference/refitting |
| Larger closer run | NOT_RUN | Runnable operator supplied; do not infer result from more compute |
| Modal run | NOT_RUN | Operator compiled, remote environment and runtime untested |
| Browser / live Colab / deployed site | NOT_CHECKED | Static payload and copied-site checks are not live UI checks |

Stack-minus-OOF-single mean log-loss gaps: diabetes −.003609 (paired seed t95%
[−.007921,+.000703]), blood_transfusion0 (selects TabM alone), phoneme−.023101
[−.028358,−.017845]. Seeds condition on fixed rows/folds. The zero interval is the
same prediction vector, not an equivalence result. A single-view TabICL happens to
produce identical predictions across the declared fitting seeds.

Mean ranks after seed averaging: XGB2.833, TabM5.667, TabICL2.000,
OOF-best4.833, Uniform2.333, Greedy-stack3.333. Friedman p=.0907; Nemenyi CD4.353.
Only three datasets and several dependent procedures: exploratory and low-powered.
The stack does not beat the best individual test score on any dataset. No experiment
here establishes cross-family superiority over matched-budget within-family ensembles.

## Rebuild and verify

From repository root:

```bash
.venv/bin/python labs/_check_l057.py
.venv/bin/python labs/_source_check_l057.py
.venv/bin/pip install -r labs/requirements-l057-tfm.txt
.venv/bin/python labs/_fetch_l057.py
.venv/bin/python labs/_verify_l057.py --include-tfm --checkpoint data/cache/l057-checkpoint/tabicl-classifier-v1.1-0506.ckpt
MPLCONFIGDIR=/tmp/l057-mpl .venv/bin/python labs/_figures_l057.py
.venv/bin/python labs/_build_l057.py
.venv/bin/python labs/_execute_l057.py
.venv/bin/python labs/_delivery_check_l057.py
node labs/_viz_check_l057.js
node labs/_check_pedagogy.js
```

The fetch helper returns an absolute path; pass that printed path to `--checkpoint`
if your working directory differs. `_verify_l057.py` changes into labs before loading
relative paths. Running without `--include-tfm` is the explicitly smaller two-family
track, not the measured three-family main table; choose a separate `--output`.

## Required larger experiment after EXIT

The notebook gate uses the student's live functions. Install optional requirements,
fetch/hash-check the original checkpoint, and run `closer`: cap2000, 5folds, 3seeds,
TabM width128/k32/64epochs, 300trees, four TabICL views. Same data identities, but changed
row pool and budget; still INCOMPARABLE. Local or Colab GPU is supported. Runtime on
free Colab has not been measured; persist outputs before sessions terminate.

```bash
modal run --detach modal/l057_paper_repro.py --preset closer
modal volume get relational-l057 closer-results.json .
```

Presets are `smoke`, `lab`, `closer`; there is deliberately no falsely named faithful
`paper` preset. Full Figure6 requires the pinned TabArena released validation/test
prediction artifacts, complete family/configuration roster, historical evaluator,
8-fold conventional models, prescribed TFM refits, binary AUC and other task metrics,
original outer repetitions, failure/imputation handling and bootstrap. Start from the
L056 pinned official repository and reproduce its full leaderboard before extending
to its cross-model ensemble analysis. A launcher alone would not close those gaps.

## Source attribution

TabArena v1 §3.2 / Figure6: https://arxiv.org/html/2506.16791v1#S3.SS2
Caruana et al. 2004 §2: https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
Pinned AutoGluon selector and Apache2 license: `sources/l057/`.
Selector parity adapts only the metric framework; upstream rounding and tie policies
are not asserted universally equivalent. Original source hashes and package/checkpoint
locations: `_sources_l057.json`. No source code is represented as independently authored
when copied from the upstream validation reference.
