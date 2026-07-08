"""Verify the Q2-checkpoint reproduction numbers for Lesson 020.

The checkpoint skill: reproduce a *published-style tree baseline* under a fair,
fixed protocol (same split, same metric, disclosed tuning budget), match or beat
it with a tuned XGBoost / LightGBM, and beat it again with a leak-free stacked
ensemble — reporting variance honestly.

Tier A — real: `adult` (OpenML 1590), binary income >50K, prevalence ~24%.
Protocol: one 80/20 stratified holdout (seed 0); tuning is 3-fold CV on TRAIN
ONLY via RandomizedSearchCV with a fixed budget; headline metric ROC-AUC
(comparable to the commonly published adult tree baseline ~0.92-0.93), with
accuracy and PR-AUC/prevalence reported alongside.

Run:  .venv/bin/python labs/_verify_l020.py
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relkit.data import load_tier_a  # noqa: E402

warnings.filterwarnings("ignore")
RS = 0

# ---------------------------------------------------------------- data + protocol
X, y = load_tier_a("adult")
cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
num_cols = [c for c in X.columns if c not in cat_cols]
prev = float(y.mean())
print(f"adult: {X.shape[0]} rows, {X.shape[1]} cols, prevalence {prev:.3f}")
print(f"  {len(num_cols)} numeric, {len(cat_cols)} categorical")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
print(f"  train {Xtr.shape[0]}  test {Xte.shape[0]}")


def pre():
    return ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def report(name, model):
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, (p >= 0.5).astype(int))
    ap = average_precision_score(yte, p)
    print(f"  {name:22s} ROC-AUC {auc:.4f}  acc {acc:.4f}  PR-AUC {ap:.4f}")
    return auc, acc, ap


# ---------------------------------------------------------------- reference baseline
# The "published-style" reference: XGBoost with fixed, disclosed defaults, no tuning.
print("\n=== Reference tree baseline (fixed defaults, no tuning) ===")
ref = Pipeline(
    [
        ("pre", pre()),
        (
            "clf",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RS,
                n_jobs=4,
            ),
        ),
    ]
)
t0 = time.time()
ref.fit(Xtr, ytr)
ref_auc, ref_acc, ref_ap = report("XGB reference", ref)
print(f"  (fit {time.time()-t0:.1f}s)  <-- this is the number to match or beat")

# ---------------------------------------------------------------- tuned XGB
print("\n=== Tuned XGBoost (RandomizedSearchCV, 3-fold CV on TRAIN, n_iter=15) ===")
xgb_pipe = Pipeline([("pre", pre()), ("clf", XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=4))])
xgb_space = {
    "clf__n_estimators": [300, 500, 800],
    "clf__max_depth": [4, 6, 8],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.9, 1.0],
    "clf__reg_lambda": [1.0, 5.0, 10.0],
}
cv = StratifiedKFold(3, shuffle=True, random_state=RS)
t0 = time.time()
xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_space, n_iter=15, scoring="roc_auc", cv=cv, random_state=RS, n_jobs=4
)
xgb_search.fit(Xtr, ytr)
print(f"  best CV ROC-AUC {xgb_search.best_score_:.4f}  (search {time.time()-t0:.1f}s)")
xgb_auc, xgb_acc, xgb_ap = report("XGB tuned", xgb_search.best_estimator_)

# ---------------------------------------------------------------- tuned LGBM
print("\n=== Tuned LightGBM (RandomizedSearchCV, 3-fold CV on TRAIN, n_iter=15) ===")
lgbm_pipe = Pipeline([("pre", pre()), ("clf", LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1))])
lgbm_space = {
    "clf__n_estimators": [300, 500, 800],
    "clf__num_leaves": [31, 63, 127],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.9, 1.0],
    "clf__reg_lambda": [0.0, 1.0, 5.0],
}
t0 = time.time()
lgbm_search = RandomizedSearchCV(
    lgbm_pipe, lgbm_space, n_iter=15, scoring="roc_auc", cv=cv, random_state=RS, n_jobs=4
)
lgbm_search.fit(Xtr, ytr)
print(f"  best CV ROC-AUC {lgbm_search.best_score_:.4f}  (search {time.time()-t0:.1f}s)")
lgbm_auc, lgbm_acc, lgbm_ap = report("LGBM tuned", lgbm_search.best_estimator_)

# ---------------------------------------------------------------- stacked ensemble (OOF)
print("\n=== Stacked ensemble: OOF meta-features (XGB + LGBM) -> logistic meta ===")
best_xgb = xgb_search.best_estimator_
best_lgbm = lgbm_search.best_estimator_
oof_xgb = cross_val_predict(best_xgb, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
oof_lgbm = cross_val_predict(best_lgbm, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
corr = float(np.corrcoef(oof_xgb, oof_lgbm)[0, 1])
print(f"  OOF correlation XGB vs LGBM: {corr:.3f}")
Z_tr = np.column_stack([oof_xgb, oof_lgbm])
meta = LogisticRegression(max_iter=1000).fit(Z_tr, ytr)
# refit base learners on full train, then stack test predictions
best_xgb.fit(Xtr, ytr)
best_lgbm.fit(Xtr, ytr)
Z_te = np.column_stack([best_xgb.predict_proba(Xte)[:, 1], best_lgbm.predict_proba(Xte)[:, 1]])
p_stack = meta.predict_proba(Z_te)[:, 1]
st_auc = roc_auc_score(yte, p_stack)
st_acc = accuracy_score(yte, (p_stack >= 0.5).astype(int))
st_ap = average_precision_score(yte, p_stack)
print(f"  {'Stacked (XGB+LGBM)':22s} ROC-AUC {st_auc:.4f}  acc {st_acc:.4f}  PR-AUC {st_ap:.4f}")

# ---------------------------------------------------------------- summary
print("\n--- summary for lesson/lab tolerances ---")
print(f"prevalence            {prev:.3f}")
print(f"reference   ROC-AUC   {ref_auc:.4f}  acc {ref_acc:.4f}")
print(f"XGB tuned   ROC-AUC   {xgb_auc:.4f}  acc {xgb_acc:.4f}  (delta {xgb_auc-ref_auc:+.4f})")
print(f"LGBM tuned  ROC-AUC   {lgbm_auc:.4f}  acc {lgbm_acc:.4f}  (delta {lgbm_auc-ref_auc:+.4f})")
print(f"stacked     ROC-AUC   {st_auc:.4f}  acc {st_acc:.4f}  (delta {st_auc-ref_auc:+.4f})")
