# Lesson 020 published — Q2 Checkpoint: match a published tree baseline

The **Q2 checkpoint** (curriculum lec 020), the gate out of Y1 Q2. Papers: Chen & Guestrin 2016
(XGBoost, [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)) + Ke et al. 2017 (LightGBM,
[NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)).
Not a new concept — a capstone that wires the whole Q2 toolkit (XGB/LGBM, tuning L017, stacking L018)
back onto the Q1 leakage spine (L010) under a **fair reproduction protocol**.

**Single skill:** reproduce a published-style tree baseline under a fair, fixed protocol (same data,
split, metric, disclosed tuning budget), then match or beat it with a tuned XGBoost / LightGBM and a
leak-free OOF stacked ensemble — and report the verdict honestly. The new discipline beyond L010 is the
**fair-comparison contract** (fix data/split/metric/budget/preprocessing-scope up front; report the
verdict + gap size), which is the reviewer stance Q3 (evaluation rigor) and Y1 L038 build on.

**Dataset:** Tier A `adult` (OpenML 1590), 48,842 rows, prevalence 0.239, 6 numeric + 8 categorical.
Cached via `labs/data/fetch_datasets.py` (fetched live this session). Metric: ROC-AUC (comparable to
the commonly reported adult tree baseline ~0.92–0.93), with accuracy + PR-AUC/prevalence alongside.

**Verified live** (`labs/_verify_l020.py`; xgboost 3.3, lightgbm 4.6, sklearn 1.9; 80/20 stratified,
seed 0; 3-fold CV on train only, `RandomizedSearchCV` n_iter=15; solution notebook reproduces
identical numbers):
- **Reference** (fixed-default XGBoost, no tuning): ROC-AUC **0.9282**, acc 0.8733 — inside the
  published range, so the baseline is reproduced.
- **Tuned XGB** 0.9294 (**+0.0012**), **tuned LGBM** 0.9296 (**+0.0014**) — tuning barely moves a
  strong default; accuracy is flat (0.8733→0.8734).
- **Stacked (XGB+LGBM → logistic, OOF)** 0.9297 (+0.0001 over best single). OOF correlation
  **0.997** → the two GBDTs are nearly redundant, so stacking adds essentially nothing (echoes the
  L018 diversity rule).

**Honest framing (the L020 point):** the checkpoint is about the *protocol*, not the headline. A
sensible default GBDT is almost the whole story; heroic tuning is a small polish, and a second GBDT is
not diversity. A tuned model that beats a fair default by a *lot* should trigger suspicion of a leak or
an unfair reference. Beating this bar needs a genuinely different model class — the opening the
relational thesis probes.

**Thesis bridge:** you have now built the incumbent honestly and at full strength. The RDL bet is not
"trees are weak" — it is that the flat `adult` table discards relational structure (employer identity,
shared households, job sequence) a model over the source database could exploit. Y1 closes (L035) by
naming what flattening throws away; Y3–4 try to beat *this exact baseline* by keeping it.

**Lesson:** `lessons/0020-q2-checkpoint.html` — checkpoint format (mirrors L010): Q2-pieces table, the
fair-comparison contract, verified bake-off table, two honesty sections (tuning barely helps; 0.997
correlation kills the ensemble), the **reproduction-protocol rubric** via `assets/checklist.js`, 3
quizzes. No new JS asset — a checkpoint is a checklist (per lesson-visuals decision tree); reuses
`quiz.js` + `checklist.js`. Headless structural check clean (all 4 mount IDs resolve, all local refs
exist). **Browser MCP still unavailable** (empty tools folder) → structural verification only.

**Lab:** `labs/0020-q2-checkpoint.ipynb` — capstone, 4 tasks (crucial fragment each) + stretch. T1 =
the stratified holdout (`train_test_split(..., stratify=y, random_state=RS)`); T2 = wrap the reference
booster in the `pre()` pipeline (per-fold-safe); T3 = the fixed-budget `RandomizedSearchCV` for XGB +
LGBM (CV on train only); T4 = OOF meta-features via `cross_val_predict(..., method="predict_proba")`.
Tier A, uses `relkit.data.load_tier_a`. Student blank (7 `____`, 0 outputs); solution executed clean
(all CHECK + EXIT, ~47s) and gitignored. Committed `_verify_l020.py` + `_build_l020.py`. Manifest → 20
entries (kept `version: 2`, hand-appended to preserve key order; `checkpoint: true`); lab rendered to
`labs/html/`.

Next: Lesson 021 opens **Q3 — Evaluation rigor & benchmark literacy** (temporal vs random splits,
leakage patterns, statistical comparison), building directly on this checkpoint's fair-comparison
contract.
