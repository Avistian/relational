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

## Q3 (Lessons 021–030)

| # | Lesson | Wrong belief | Correct belief | Status |
|---|--------|--------------|----------------|--------|
| M18 | L021 | A random (shuffled) split is the default-safe way to evaluate any tabular model | On time-ordered / drifting data a random split leaks the deployment period into training and is **optimistic** (random-CV 0.846 vs temporal 0.758); use a **temporal split** (train past → test future) / `TimeSeriesSplit` whenever a time order or drift exists — assume temporal until you can argue the rows are truly i.i.d. | active |
| M19 | L022 | A very predictive feature (or a big complex-vs-simple win) is good news | It is a **leak hypothesis first**: an **illegitimate feature** (L2, knowable only at/after the outcome) passes every pipeline/split check yet hands the model the answer, and a flexible model exploits it harder than a linear one — so removing it can collapse a huge apparent win (RF 0.935 vs LR 0.719 → 0.712 vs 0.721). Audit feature **provenance** with a model info sheet | active |
| M20 | L023 | A mean-accuracy gap (or a paired t-test / Wilcoxon over CV folds) tells you model A is really better | k-fold training sets **overlap**, so fold scores are **positively correlated**; a naive paired t-test (and Wilcoxon) treat them as independent and **over-reject** (naive p=1.2e−5 vs corrected 0.19 for a +0.0098 gap). On one dataset use the **corrected resampled t-test** (inflate variance by `1/n + 1/(k−1)`); across many *independent* datasets use **Friedman + Nemenyi CD**. Report an effect size, not just a p-value | active |
| M21 | L024 | Comparing two model families by each one's single best tuned score is a fair benchmark | One tuned number **hides how much tuning each model needed** and its **default quality**, flattering the model that is hardest to tune. Report a **random-search budget curve** (select by validation, report test, avg over orderings) that shows both the default (k=1) and the ceiling; on credit-g the GBT leads at **every** budget (default gap +0.062 → tuned +0.015). Aggregate across datasets via **per-dataset min-max normalization**, and curate datasets by explicit criteria first | active |
| M22 | L025 | Trees beat MLPs on tabular data because trees are simply more powerful / higher-capacity models | It is an **inductive-bias mismatch**, not raw strength. An MLP trained by gradient descent has a **smoothness (spectral) bias** — it fits low-frequency structure easily and high-frequency detail poorly (Rahaman 2019) — while typical tabular targets are **irregular**; a piecewise-constant tree follows the jags. Proof: **Gaussian-smoothing the target** deletes the high-frequency structure and the gap **collapses in lockstep with the variance removed** (repro: +0.33 R² → ~0, MLP even edges ahead once smooth). On a genuinely smooth target the MLP ties or wins | active |
| M23 | L026 | Rotation invariance is an elegant, desirable property — a model that doesn't care about the feature basis is more robust | On **tabular** data it is a **liability**: columns carry individual meaning (age, balance), so the original basis aligns the signal with the axes. An MLP/ResNet is rotationally invariant (linear first layer: `W·(Qx)=(WQ)·x`); a tree and an FT-Transformer are **not**. A random rotation is **lossless** (Q invertible) yet collapses a tree while leaving the MLP unmoved — **reversing the ranking** (repro: tree 0.987→0.747 vs MLP 0.862→0.869, +0.008). Ng 2004: an invariant learner's sample complexity grows **≥ linearly in the number of uninformative features**. Invariance is good on images (rotated cat = cat), bad where a natural basis exists — which is why numeric-feature embeddings that *break* invariance (SAINT, FT-Transformer) help | active |
| M24 | L027 | Pure-noise features carry no signal, so every model just ignores them | Only a **tree** ignores them — via **greedy, gain-gated split selection** (a ~0-gain noise column is never chosen = implicit feature selection). An **MLP is not robust to uninformative features** (Grinsztajn §5.3, Finding 2): it wires every column into its first layer and, being rotationally invariant, has no gate to tell a useful axis from a useless one, so (Ng 2004) it needs **≥ linearly more samples per junk feature** and instead leaks capacity onto the noise. Repro: on a smooth target where the MLP *wins* clean (0.986 vs GBT 0.945), adding 100 junk cols costs the MLP **0.084** vs the GBT **0.032** — the ranking **reverses**; removing junk helps the MLP more. Gate is visible as root-split gain ~**118×** higher on informative vs junk (use root gain, **not** MDI — MDI over-credits noise) | active |

| M26 | L029 | AutoML replaces feature engineering and beats a hand-tuned model — just point it at the data | AutoML (Auto-sklearn) automates the **CASH** search (which algorithm + its hyperparameters, selected by validation) plus meta-learning warm-start and Caruana **ensemble selection** — it does **not** do domain **feature engineering** and does not change the **representation**. On typical tabular data it **ties** a competently tuned GBDT: the big gain is tuning *at all* (credit_g: default XGB 0.775 → tuned 0.806, +0.031), after which a 4-algorithm AutoML with ensembling only matches it (0.803, bands overlap) at far higher compute. It buys automation/robustness, not new accuracy — remaining upside must come from a better representation (the relational thesis), which AutoML never touches | active |
| M25 | L028 | Adding more layers to a neural net always helps (more capacity), and if a deep net does worse it must be overfitting | A deep **plain** MLP gets *worse* with depth — the **degradation problem** (He et al. 2015) — and its **training** accuracy falls too, which rules out overfitting (that would keep train accuracy high). It is an **optimization** failure: the extra `Linear→ReLU` layers cannot easily learn to reproduce their own input. A **residual/skip connection** (output `x + f(x)` instead of `f(x)`) fixes it by making the **identity map free** — `f≈0` is trivial, so a deep ResNet can fall back to a shallow solution and depth can't hurt. It is **not** vanishing gradients (both nets use BatchNorm, which fixes those). Repro: same net, skip OFF plain test 0.917→0.866 & TRAIN 1.000→0.927 (d1→d32); skip ON ResNet holds ~0.90 / train ~1.00. And building the honest baseline does **not** make DL win — a tuned GBDT still leads on small categorical data (credit_g GBDT 0.793 vs MLP 0.752 ≈ ResNet 0.743) | active |
| M27 | L030 | A benchmark must crown a winner; a report that ends in "no significant difference" is a failed or uninformative experiment | A **correctly-established tie is a valid, valuable result.** A reported gap is a random variable, and k-fold scores overlap, so a bigger mean is not a win — you must run a **corrected resampled t-test** (or Friedman + Nemenyi across datasets) and report the effect size. On credit_g the GBDT led the honest MLP baseline by only **+0.0081 ROC-AUC** over 25 paired folds, and the corrected test gave **p=0.64** (naive p=0.22) → **no significant winner**. Writing "GBDT wins" from the +0.008 mean is exactly the L023 mistake. Being willing to report a tie is what makes an eventual "RDL beats the incumbent" claim from the same protocol credible | active |

---

*Seeded 2026-07-08 from learning records 0001–0051. Add new rows as misconceptions surface; keep the
matching entry in `assets/retrieval-pool.js` in sync.*
