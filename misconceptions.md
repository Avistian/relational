# Misconceptions Ledger

A living record of misconceptions surfaced (and corrected) during lessons and labs. Each entry is a
durable re-test target: the spaced-retrieval bank (`assets/retrieval-pool.js`) draws from these so a
mistake made once keeps coming back until it is genuinely gone.

**How to use.** When a lab CHECK, warm-up, or check-in exposes a wrong belief, add a row here with the
*wrong* belief, the *correct* one, and the origin lesson. Then add a matching question to
`assets/retrieval-pool.js` tagged `"misconception": true` so it enters the spaced rotation. When the
learner answers a misconception question correctly across ≥2 spaced sessions (Leitner box ≥ 3), mark it
**retired** here (keep the row for history).

Status legend: **active** = still re-testing · **retired** = answered correctly across ≥2 spaced sessions.

---

## Q1 (Lessons 001–010)

| # | Lesson | Wrong belief | Correct belief | Status |
|---|--------|--------------|----------------|--------|
| M01 | L004 | `best_score_` is the training accuracy | `best_score_` is a *validation* score chosen by the same data used to select the model → optimistically biased; report a nested-CV score instead | active |
| M02 | L008 | The PR-AUC baseline is 0.5 (like ROC) | The no-skill PR baseline equals the **prevalence** (positive rate), not 0.5 | active |
| M03 | L008 | `CalibratedClassifierCV.fit()` returns probabilities | `.fit()` returns the estimator (self); get probabilities with `predict_proba(X)[:, 1]` | active |
| M04 | L008 | You can fit the calibrator on the training data | The calibrator must be fit on **held-out** data, or calibration itself leaks | active |
| M05 | L005 | `SelectKBest` (or any fit-on-data step) before CV is fine | Any step that learns from data must be fit **per-fold inside the pipeline**; fitting on all data before CV leaks (0.78 vs 0.44 honest on pure noise) | active |
| M06 | L007 | Apply SMOTE to the whole dataset, then split | Resampling must happen **inside** the CV fold (imblearn Pipeline); SMOTE before the split leaks (F1 0.887 mirage vs 0.479 honest) | active |
| M07 | L009 | Target-encode a category on the full column | Target encoding must be **out-of-fold** (per-fold); full-column encoding leaks the label (0.76 vs 0.50 on a pure-noise category) | active |

## Q2 (Lessons 011–020)

| # | Lesson | Wrong belief | Correct belief | Status |
|---|--------|--------------|----------------|--------|
| M08 | L011 | Gini impurity of a node is `p` | Gini `= 1 − p² − (1−p)²` (probability two random draws differ) | active |
| M09 | L011/L014 | Split gain is `children − parent` | Gain `= parent_impurity − weighted_children_impurity` (parent minus children); in XGBoost the regularized form subtracts a `γ` toll | active |
| M10 | L014 | The optimal leaf weight is the mean residual | `w* = −G / (H + λ)` (sum of gradients over sum of hessians + regularization); matches XGBoost's leaf output exactly | active |
| M11 | L015 | `max_depth` is LightGBM's primary capacity knob | LightGBM grows **leaf-wise (best-first)**, so `num_leaves` is the primary knob; `max_depth` only caps it | active |
| M12 | L015 | LightGBM is "20× faster than XGBoost" | The 20× is vs *conventional pre-histogram* GBDT; modern XGBoost-`hist` is on par. GOSS (`boosting_type='goss'`) is the opt-in extra speedup | active |
| M13 | L015 | GOSS just drops small-gradient rows | GOSS keeps top-`a` gradients + samples `b` of the rest, then **amplifies the sampled rest by `(1−a)/b`** to stay unbiased | active |
| M14 | L016 | Greedy (whole-column) target statistics are fine if you have enough data | Greedy TS leaks a row's own label (worst on rare categories); CatBoost uses **ordered TS** (encode row `i` from its permutation prefix + prior) | active |
| M15 | L016 | `cat_features` survives `sklearn.clone` | It does **not** → `cross_val_score`/`relkit.cv_pr_auc` fail on CatBoost; use a clone-free manual fold loop, and cat columns must be str/int (not NaN) | active |
| M16 | L018 | Averaging/stacking base models always improves the score | Only with **diverse** bases and enough data; on a 1000-row single table the gain was +0.001, and stacking on **in-sample** base preds leaks (crowns the memorizer) — use **out-of-fold** meta-features | active |
| M17 | L019 | "Trees always win on tabular data" | With clean, all-informative, smooth features the MLP won; trees win because *typical* tabular data has irregular targets + junk features + meaningful orientation **at once** | active |

---

*Seeded 2026-07-08 from learning records 0001–0051. Add new rows as misconceptions surface; keep the
matching entry in `assets/retrieval-pool.js` in sync.*
