"""Build labs/0020-q2-checkpoint.ipynb (student, blanks) and solutions/ (filled).

Q2 checkpoint capstone: reproduce a published-style tree baseline on adult under a
fair fixed protocol, match/beat it with tuned XGBoost + LightGBM, and a leak-free
OOF stacked ensemble. Numbers mirror labs/_verify_l020.py.

Run:  python labs/_build_l020.py
Then execute the solution to verify all CHECK + EXIT pass.
"""
from __future__ import annotations

import json
from pathlib import Path

LABS = Path(__file__).resolve().parent
SOL = LABS / "solutions"
SOL.mkdir(exist_ok=True)

CELLS = []


def md(src):
    CELLS.append(("md", src))


def code(src):
    CELLS.append(("code", src))


def todo(sol, stu):
    CELLS.append(("todo", sol, stu))


# ---------------------------------------------------------------- header
md("""# Lab 020 — Q2 Checkpoint: match a published tree baseline

**Lesson:** [`lessons/0020-q2-checkpoint.html`](../lessons/0020-q2-checkpoint.html) · **Phase / Year:** Year 1 · Q2 (capstone)

**Papers:** Chen & Guestrin 2016, *XGBoost* ([arXiv:1603.02754](https://arxiv.org/abs/1603.02754)) · Ke et al. 2017, *LightGBM* ([NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)).

**Dataset tier:** A — real, open. `adult` (OpenML 1590), income > 50K, ~24% positive. One 80/20 stratified holdout, seed 0.

**Skill you are practising:** reproduce a published-style tree baseline under a *fair, fixed protocol* (same data, split, metric, disclosed tuning budget), then match or beat it with a tuned XGBoost / LightGBM and a leak-free stacked ensemble — and report the verdict honestly.

**Exit criteria:** EXIT TICKET prints the comparison table (reference vs tuned XGB vs tuned LGBM vs stacked: ROC-AUC + accuracy), the base-model OOF correlation, and your one-line honest verdict on whether you matched/beat the reference.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn**, **xgboost**, **lightgbm**. The dataset is cached by `labs/data/fetch_datasets.py` (run once, online). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — the fair-comparison contract

"Match or beat a published number" is meaningless unless the comparison is **fair**. Before touching a model you fix five things and report the sixth:

1. **Same dataset** — the exact rows and target the reference used.
2. **Same split** — one fixed, stratified holdout with a pinned seed; the test set is touched *once*, at the end.
3. **Same metric** — one headline number chosen up front (here **ROC-AUC**), prevalence stated so it is readable.
4. **Disclosed tuning budget** — tuning is cross-validation on **train only**, with a stated number of trials.
5. **Same preprocessing scope** — every fit-bearing transform lives *inside* the pipeline, so no fold sees the others (the [L010](../lessons/0010-baseline-checkpoint.html) leakage spine).
6. **Reported honestly** — the verdict (matched / beat / failed) and the *size* of the gap, not a cherry-picked run.

**The reference.** On `adult`, a competently-configured GBDT is a commonly reported baseline at **ROC-AUC ≈ 0.92–0.93** / **~87% accuracy**. We pre-register the target as a *fixed-default* XGBoost (no tuning, disclosed params) so anyone can reproduce it, then try to beat it.

**Toy micro-example (not this lab's answer).** Suppose a paper reports "AUC 0.90" but tuned on the test set. Your fair reproduction fixes the split first, tunes on train-only CV, and scores the test set once. If you get 0.88 you report *0.88 with a fair protocol* — not 0.90 by peeking. A smaller honest number beats a larger dishonest one.

Full write-up + the reproduction-protocol rubric: [Lesson 020](../lessons/0020-q2-checkpoint.html).""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED — data, fixed split, preprocessing factory, and a scorer.
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
_here = Path(".").resolve()
for _p in (_here, _here.parent, _here / "labs"):   # put labs/ on the path (works from repo root, labs/, or labs/solutions/)
    sys.path.insert(0, str(_p))

import numpy as np
from relkit.data import load_tier_a
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_predict)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RS = 0
X, y = load_tier_a("adult")
cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
num_cols = [c for c in X.columns if c not in cat_cols]
prev = float(y.mean())
print(f"adult: {X.shape[0]} rows, {X.shape[1]} cols, prevalence {prev:.3f} "
      f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")

def pre():
    # per-fold-safe preprocessing: passthrough numerics, one-hot the categoricals.
    return ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

def report(name, model, Xte, yte):
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, (p >= 0.5).astype(int))
    ap = average_precision_score(yte, p)
    print(f"  {name:26s} ROC-AUC {auc:.4f}  acc {acc:.4f}  PR-AUC {ap:.4f}")
    return {"auc": auc, "acc": acc, "ap": ap}

cv = StratifiedKFold(3, shuffle=True, random_state=RS)   # tuning folds — TRAIN only
print("setup ready")""")

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — fix the protocol: one stratified holdout — TODO (crucial fragment)

**Goal:** split off a single 80/20 test set that preserves the class balance and is reproducible.

**Why it matters:** every number in this lab is compared on *this* test set, touched once at the very end. Get the split wrong (unstratified, unseeded, or re-drawn later) and the whole comparison is unfair — the single most common way a "we beat the baseline" claim collapses.

**You implement:** a stratified 80/20 split with the fixed seed `RS`.

**Hint boundary:** `train_test_split` with `test_size=0.2`, `stratify=y`, and `random_state=RS`. Do not change the variable names.""")

todo(
    """# TODO — one fixed, stratified holdout
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
print(f"train {Xtr.shape[0]}  test {Xte.shape[0]}  "
      f"train prev {ytr.mean():.3f}  test prev {yte.mean():.3f}")""",
    """# TODO — one fixed, stratified holdout
Xtr, Xte, ytr, yte = ____     # 80/20, stratified on y, random_state=RS
print(f"train {Xtr.shape[0]}  test {Xte.shape[0]}  "
      f"train prev {ytr.mean():.3f}  test prev {yte.mean():.3f}")""",
)

code("""# CHECK — do not edit
assert abs(Xte.shape[0] / X.shape[0] - 0.2) < 0.01, "Test set should be ~20% of the data."
assert abs(ytr.mean() - yte.mean()) < 0.005, "Stratification should keep train/test prevalence equal."
assert Xtr.shape[0] + Xte.shape[0] == X.shape[0], "Split should partition all rows."
print(f"Task 1 ok — {Xte.shape[0]} held-out rows, prevalence matched to {yte.mean():.3f}. "
      f"Protocol fixed; the test set is now off-limits until the end.")""")

# ---------------------------------------------------------------- Task 2
md(r"""## Task 2 — the reference baseline — TODO (crucial fragment)

**Goal:** build the pre-registered reference — a fixed-default XGBoost *inside a pipeline* — and confirm it reproduces the published range (ROC-AUC ≈ 0.92–0.93).

**Why it matters:** the reference must be reproducible and fair before you try to beat it. Wrapping the booster in the `pre()` pipeline is what keeps preprocessing per-fold-safe once we tune with CV — the leakage spine does not lapse because the model got fancier.

**You implement:** a `Pipeline` whose first step is `pre()` and whose classifier is the provided fixed-default `XGBClassifier`.

**Hint boundary:** `Pipeline([("pre", pre()), ("clf", <the classifier>)])`. The hyper-parameters are given — do not tune here.""")

todo(
    """# TODO — assemble the reference pipeline (preprocessing INSIDE the pipeline)
ref_clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=RS, n_jobs=4)
ref = Pipeline([("pre", pre()), ("clf", ref_clf)])
ref.fit(Xtr, ytr)
ref_m = report("XGB reference (fixed)", ref, Xte, yte)""",
    """# TODO — assemble the reference pipeline (preprocessing INSIDE the pipeline)
ref_clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=RS, n_jobs=4)
ref = ____                    # Pipeline: ("pre", pre()) then ("clf", ref_clf)
ref.fit(Xtr, ytr)
ref_m = report("XGB reference (fixed)", ref, Xte, yte)""",
)

code("""# CHECK — do not edit
assert list(ref.named_steps) == ["pre", "clf"], "Reference must be a Pipeline: pre -> clf."
assert 0.92 <= ref_m["auc"] <= 0.935, "Reference ROC-AUC should land in the published range ~0.92-0.93."
print(f"Task 2 ok — reference reproduced at ROC-AUC {ref_m['auc']:.4f}. "
      f"This is the number to match or beat.")""")

# ---------------------------------------------------------------- Task 3
md(r"""## Task 3 — tune XGBoost and LightGBM (fixed budget, train-only CV) — TODO (crucial fragment)

**Goal:** run a disclosed-budget randomized search for **XGBoost** (Chen 2016) and **LightGBM** (Ke 2017), tuning only on the training folds, and score each once on the test set.

**Why it matters:** this is the fair way to tune. `RandomizedSearchCV` with a fixed `n_iter` and CV on train only means "I searched 15 configurations," not "I searched until I won." Notice how little tuning buys over a strong default.

**You implement:** the `RandomizedSearchCV` object — the estimator pipeline, the space, `n_iter=15`, `scoring="roc_auc"`, and `cv=cv` (the train-only folds). Do this once for each booster.

**Hint boundary:** `RandomizedSearchCV(<pipe>, <space>, n_iter=15, scoring="roc_auc", cv=cv, random_state=RS, n_jobs=4)`. The pipelines and spaces are provided.""")

code("""# PROVIDED — the two pipelines and their search spaces (clf__ prefixes target the classifier step).
xgb_pipe = Pipeline([("pre", pre()),
                     ("clf", XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=4))])
xgb_space = {"clf__n_estimators": [300, 500, 800], "clf__max_depth": [4, 6, 8],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
             "clf__colsample_bytree": [0.7, 0.9, 1.0], "clf__reg_lambda": [1.0, 5.0, 10.0]}

lgbm_pipe = Pipeline([("pre", pre()),
                      ("clf", LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1))])
lgbm_space = {"clf__n_estimators": [300, 500, 800], "clf__num_leaves": [31, 63, 127],
              "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
              "clf__colsample_bytree": [0.7, 0.9, 1.0], "clf__reg_lambda": [0.0, 1.0, 5.0]}
print("pipelines + spaces ready")""")

todo(
    """# TODO — fixed-budget search for each booster (CV on TRAIN only)
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_space, n_iter=15, scoring="roc_auc",
                                cv=cv, random_state=RS, n_jobs=4).fit(Xtr, ytr)
lgbm_search = RandomizedSearchCV(lgbm_pipe, lgbm_space, n_iter=15, scoring="roc_auc",
                                 cv=cv, random_state=RS, n_jobs=4).fit(Xtr, ytr)
print(f"XGB  best CV ROC-AUC {xgb_search.best_score_:.4f}")
print(f"LGBM best CV ROC-AUC {lgbm_search.best_score_:.4f}")
xgb_m = report("XGB tuned", xgb_search.best_estimator_, Xte, yte)
lgbm_m = report("LGBM tuned", lgbm_search.best_estimator_, Xte, yte)""",
    """# TODO — fixed-budget search for each booster (CV on TRAIN only)
xgb_search = ____     # RandomizedSearchCV(xgb_pipe, xgb_space, n_iter=15, scoring="roc_auc", cv=cv, ...).fit(Xtr, ytr)
lgbm_search = ____    # same for lgbm_pipe / lgbm_space
print(f"XGB  best CV ROC-AUC {xgb_search.best_score_:.4f}")
print(f"LGBM best CV ROC-AUC {lgbm_search.best_score_:.4f}")
xgb_m = report("XGB tuned", xgb_search.best_estimator_, Xte, yte)
lgbm_m = report("LGBM tuned", lgbm_search.best_estimator_, Xte, yte)""",
)

code("""# CHECK — do not edit
assert xgb_search.n_iter == 15 and lgbm_search.n_iter == 15, "Disclosed budget: n_iter must be 15."
assert xgb_m["auc"] >= ref_m["auc"] - 0.002, "Tuned XGB should match the reference (within noise)."
assert lgbm_m["auc"] >= ref_m["auc"] - 0.002, "Tuned LGBM should match the reference (within noise)."
print(f"Task 3 ok — tuned XGB {xgb_m['auc']:.4f} (delta {xgb_m['auc']-ref_m['auc']:+.4f}), "
      f"tuned LGBM {lgbm_m['auc']:.4f} (delta {lgbm_m['auc']-ref_m['auc']:+.4f}). "
      f"Note how small the tuning gain is over a strong default.")""")

# ---------------------------------------------------------------- Task 4
md(r"""## Task 4 — leak-free stacked ensemble (OOF) — TODO (crucial fragment)

**Goal:** stack the two tuned boosters using **out-of-fold** meta-features, read their correlation, and see whether combining helps.

**Why it matters:** stacking's fuel is *diversity* ([L018](../lessons/0018-ensembling-stacking.html)). Building meta-features with `cross_val_predict` keeps the meta-learner from ever training on a base model's in-sample predictions (the leak that crowns a memorizer). The correlation tells you whether the two learners are actually different.

**You implement:** the two OOF prediction vectors via `cross_val_predict(..., method="predict_proba")[:, 1]`, using the same train-only `cv`.

**Hint boundary:** `cross_val_predict(best_estimator, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]` for each booster. The meta-learner fit and test stacking are provided.""")

todo(
    """# TODO — build out-of-fold meta-features (no in-sample leak)
best_xgb = xgb_search.best_estimator_
best_lgbm = lgbm_search.best_estimator_
oof_xgb = cross_val_predict(best_xgb, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
oof_lgbm = cross_val_predict(best_lgbm, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]

corr = float(np.corrcoef(oof_xgb, oof_lgbm)[0, 1])
print(f"OOF correlation XGB vs LGBM: {corr:.3f}")

# PROVIDED — fit meta on OOF features, refit base learners on full train, stack the test set
meta = LogisticRegression(max_iter=1000).fit(np.column_stack([oof_xgb, oof_lgbm]), ytr)
best_xgb.fit(Xtr, ytr); best_lgbm.fit(Xtr, ytr)
Z_te = np.column_stack([best_xgb.predict_proba(Xte)[:, 1], best_lgbm.predict_proba(Xte)[:, 1]])
p_stack = meta.predict_proba(Z_te)[:, 1]
stack_m = {"auc": roc_auc_score(yte, p_stack),
           "acc": accuracy_score(yte, (p_stack >= 0.5).astype(int)),
           "ap": average_precision_score(yte, p_stack)}
print(f"  {'Stacked (XGB+LGBM)':26s} ROC-AUC {stack_m['auc']:.4f}  "
      f"acc {stack_m['acc']:.4f}  PR-AUC {stack_m['ap']:.4f}")""",
    """# TODO — build out-of-fold meta-features (no in-sample leak)
best_xgb = xgb_search.best_estimator_
best_lgbm = lgbm_search.best_estimator_
oof_xgb = ____     # cross_val_predict(best_xgb, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
oof_lgbm = ____    # same for best_lgbm

corr = float(np.corrcoef(oof_xgb, oof_lgbm)[0, 1])
print(f"OOF correlation XGB vs LGBM: {corr:.3f}")

# PROVIDED — fit meta on OOF features, refit base learners on full train, stack the test set
meta = LogisticRegression(max_iter=1000).fit(np.column_stack([oof_xgb, oof_lgbm]), ytr)
best_xgb.fit(Xtr, ytr); best_lgbm.fit(Xtr, ytr)
Z_te = np.column_stack([best_xgb.predict_proba(Xte)[:, 1], best_lgbm.predict_proba(Xte)[:, 1]])
p_stack = meta.predict_proba(Z_te)[:, 1]
stack_m = {"auc": roc_auc_score(yte, p_stack),
           "acc": accuracy_score(yte, (p_stack >= 0.5).astype(int)),
           "ap": average_precision_score(yte, p_stack)}
print(f"  {'Stacked (XGB+LGBM)':26s} ROC-AUC {stack_m['auc']:.4f}  "
      f"acc {stack_m['acc']:.4f}  PR-AUC {stack_m['ap']:.4f}")""",
)

code("""# CHECK — do not edit
assert oof_xgb.shape == (Xtr.shape[0],), "OOF vector must have one prediction per TRAIN row."
assert corr > 0.9, "Two tuned GBDTs should be highly correlated (redundant) on this table."
best_single = max(xgb_m["auc"], lgbm_m["auc"])
assert stack_m["auc"] >= best_single - 0.002, "Stack should not be worse than the best single model."
print(f"Task 4 ok — stack {stack_m['auc']:.4f} vs best single {best_single:.4f} "
      f"(delta {stack_m['auc']-best_single:+.4f}); OOF corr {corr:.3f}. "
      f"Two GBDTs are nearly redundant, so stacking adds almost nothing.")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Verdict prompt:** in one sentence, state whether you *reproduced* the published range, whether tuning/stacking *beat* the reference, and by how much — honestly.""")

todo(
    '''# TODO: complete the verdict string
print("=== EXIT TICKET — Lesson 020 (Q2 checkpoint) ===")
print(f"dataset: adult  prevalence {prev:.3f}  |  metric: ROC-AUC  |  split: 80/20 stratified, seed 0")
print(f"{'model':26s} {'ROC-AUC':>8} {'acc':>7}   delta-vs-ref")
print(f"{'XGB reference (fixed)':26s} {ref_m['auc']:>8.4f} {ref_m['acc']:>7.4f}   --")
print(f"{'XGB tuned':26s} {xgb_m['auc']:>8.4f} {xgb_m['acc']:>7.4f}   {xgb_m['auc']-ref_m['auc']:+.4f}")
print(f"{'LGBM tuned':26s} {lgbm_m['auc']:>8.4f} {lgbm_m['acc']:>7.4f}   {lgbm_m['auc']-ref_m['auc']:+.4f}")
print(f"{'Stacked (XGB+LGBM)':26s} {stack_m['auc']:>8.4f} {stack_m['acc']:>7.4f}   {stack_m['auc']-ref_m['auc']:+.4f}")
print(f"OOF correlation XGB vs LGBM: {corr:.3f}")
print()
print("verdict:",
      "Reproduced the published tree baseline (ROC-AUC ~0.928, inside the 0.92-0.93 range) under a fair "
      "fixed protocol; tuning beat the reference by only a few ten-thousandths and stacking two GBDTs "
      "(OOF correlation ~0.997) added essentially nothing -- a strong default GBDT is almost the whole story, "
      "and beating it needs a genuinely different model class, not more tuning.")''',
    '''# TODO: complete the verdict string
print("=== EXIT TICKET — Lesson 020 (Q2 checkpoint) ===")
print(f"dataset: adult  prevalence {prev:.3f}  |  metric: ROC-AUC  |  split: 80/20 stratified, seed 0")
print(f"{'model':26s} {'ROC-AUC':>8} {'acc':>7}   delta-vs-ref")
print(f"{'XGB reference (fixed)':26s} {ref_m['auc']:>8.4f} {ref_m['acc']:>7.4f}   --")
print(f"{'XGB tuned':26s} {xgb_m['auc']:>8.4f} {xgb_m['acc']:>7.4f}   {xgb_m['auc']-ref_m['auc']:+.4f}")
print(f"{'LGBM tuned':26s} {lgbm_m['auc']:>8.4f} {lgbm_m['acc']:>7.4f}   {lgbm_m['auc']-ref_m['auc']:+.4f}")
print(f"{'Stacked (XGB+LGBM)':26s} {stack_m['auc']:>8.4f} {stack_m['acc']:>7.4f}   {stack_m['auc']-ref_m['auc']:+.4f}")
print(f"OOF correlation XGB vs LGBM: {corr:.3f}")
print()
print("verdict:", "____")''',
)

# ---------------------------------------------------------------- stretch
md(r"""## Stretch (optional, ungraded) — find real diversity

1. **Add a genuinely different learner.** Add the Q1 `LogisticRegression` (on scaled/one-hot features) as a third base model. Its OOF correlation with the GBDTs should be much lower than 0.997 — does the stack finally move?
2. **CatBoost as a third booster.** Add a tuned `CatBoostClassifier` ([L016](../lessons/0016-catboost.html)); it is still a GBDT, so predict (and check) that it stays highly correlated with XGB/LGBM.
3. **Change the metric.** Re-run the verdict using **PR-AUC vs prevalence** instead of ROC-AUC. Does the ranking of models change? Which metric would you report to a skeptic on a 24%-positive task, and why?""")

code('''# STRETCH — ungraded.
# from sklearn.preprocessing import StandardScaler
# logit = Pipeline([("pre", pre()), ("sc", StandardScaler(with_mean=False)),
#                   ("clf", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
# oof_lr = cross_val_predict(logit, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
# print("corr(LR, XGB):", round(float(np.corrcoef(oof_lr, oof_xgb)[0, 1]), 3), "  <-- more diverse than 0.997?")''')


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l020-{i:02d}"
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": entry[1].splitlines(keepends=True)})
        elif kind == "todo":
            src = entry[1] if solution else entry[2]
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})
        else:
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": entry[1].splitlines(keepends=True)})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


(LABS / "0020-q2-checkpoint.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0020-q2-checkpoint.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0020-q2-checkpoint.ipynb and solutions/0020-q2-checkpoint.ipynb")
