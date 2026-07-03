"""Verify Lesson 015 (LightGBM) numbers. Run from labs/ so `relkit` imports:

    cd labs && OMP_NUM_THREADS=3 ../.venv/bin/python _verify_l015.py

Produces the real numbers used in the lesson HTML and lab:
  1. GOSS amplified gradient estimate is ~unbiased (Ke et al. 2017, Algorithm 2).
  2. num_leaves sweep on credit_g — leaf-wise overfits small data as trees grow.
  3. credit_g: LightGBM default vs XGBoost default vs tuned LightGBM (CV PR-AUC).
  4. adult: LightGBM vs XGBoost accuracy near-tie.
  5. Honest speed: hist boosters vs a conventional (pre-histogram) GBDT.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0


def section(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)


def prep(name):
    X, y = load_tier_a(name)
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    Xn = X[num].copy()
    for c in [c for c in X.columns if c not in num]:
        Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return Xn, np.asarray(y)


# ---------------------------------------------------------------- 1. GOSS
section("1. GOSS amplified gradient estimate is ~unbiased")
rng = np.random.RandomState(0)
n = 2000
g = rng.standard_normal(n)
region = rng.rand(n) < 0.5
A, B = 0.2, 0.1
true_left = g[region].sum()


def goss_weights(g, a, b, rng):
    n = len(g)
    w = np.zeros(n)
    order = np.argsort(-np.abs(g))
    top_n = int(a * n)
    top_idx = order[:top_n]
    samp = rng.choice(order[top_n:], size=int(b * n), replace=False)
    w[top_idx] = 1.0
    w[samp] = (1 - a) / b
    return w


errs = []
for s in range(200):
    w = goss_weights(g, A, B, np.random.RandomState(s))
    errs.append((w[region] * g[region]).sum() - true_left)
used = (goss_weights(g, A, B, np.random.RandomState(1)) > 0).mean()
print(f"rows used ~ {used:.0%} (= a + b = {A + B:.0%})   amplify (1-a)/b = {(1 - A) / B:.1f}")
print(f"true left grad sum = {true_left:.3f}   mean estimate error over 200 seeds = {np.mean(errs):+.3f}")

# ---------------------------------------------------------------- 2 + 3. credit_g
section("2-3. credit_g — num_leaves sweep + LightGBM vs XGBoost")
Xc, yc = prep("credit_g")
print(f"rows={len(yc)} pos_rate={yc.mean():.3f} features={Xc.shape[1]}")
for nl in [7, 15, 31, 63, 127]:
    m = LGBMClassifier(num_leaves=nl, n_estimators=200, learning_rate=0.05, random_state=RS, n_jobs=1, verbose=-1)
    print(f"  num_leaves={nl:4d}  CV PR-AUC = {cv_pr_auc(m, Xc, yc):.3f}")

lgbm_default = cv_pr_auc(LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1), Xc, yc)
xgb_default = cv_pr_auc(XGBClassifier(random_state=RS, n_jobs=1, eval_metric="logloss", tree_method="hist"), Xc, yc)
param_dist = {
    "n_estimators": [200, 400, 600],
    "num_leaves": [7, 15, 31, 63],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
    "min_child_samples": [5, 20, 50],
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
search = RandomizedSearchCV(
    LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1),
    param_dist, n_iter=40, scoring="average_precision", cv=cv, random_state=RS, n_jobs=4,
)
search.fit(Xc, yc)
print(f"LightGBM default  CV PR-AUC = {lgbm_default:.3f}")
print(f"XGBoost  default  CV PR-AUC = {xgb_default:.3f}")
print(f"LightGBM tuned    CV PR-AUC = {float(search.best_score_):.3f}  best={search.best_params_}")

# ---------------------------------------------------------------- 4. adult
section("4. adult — accuracy near-tie")
Xa, ya = prep("adult")
xa = cv_pr_auc(XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist"), Xa, ya)
la = cv_pr_auc(LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1), Xa, ya)
print(f"rows={len(ya)} pos_rate={ya.mean():.3f}  |  XGBoost {xa:.3f}  LightGBM {la:.3f}")

# ---------------------------------------------------------------- 5. speed
section("5. Honest speed — hist boosters vs conventional (pre-hist) GBDT")
Xb, yb = make_classification(n_samples=50_000, n_features=50, n_informative=15, n_redundant=0, random_state=RS)


def ft(est):
    t0 = time.perf_counter()
    est.fit(Xb, yb)
    return time.perf_counter() - t0


t_conv = ft(GradientBoostingClassifier(n_estimators=100, random_state=RS))
t_xgb = ft(XGBClassifier(n_estimators=100, random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist"))
t_lgbm = ft(LGBMClassifier(n_estimators=100, random_state=RS, n_jobs=4, verbose=-1))
t_goss = ft(LGBMClassifier(n_estimators=100, random_state=RS, n_jobs=4, verbose=-1, boosting_type="goss"))
print(f"sklearn GBDT (conventional): {t_conv:6.2f}s  (1x)")
print(f"XGBoost (hist)             : {t_xgb:6.2f}s  ({t_conv / t_xgb:.0f}x)")
print(f"LightGBM (gbdt default)    : {t_lgbm:6.2f}s  ({t_conv / t_lgbm:.0f}x)")
print(f"LightGBM (goss)            : {t_goss:6.2f}s  ({t_conv / t_goss:.0f}x)")
print("\nDONE")
