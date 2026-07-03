"""Temp measurement script for Lesson 015 (LightGBM). Not committed.
Produces the numbers cited in the lesson table + lab reproduction target."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # labs/ on path for relkit

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0


def prep(name):
    X, y = load_tier_a(name)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    Xn = X[num_cols].copy()
    for c in cat_cols:
        Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return Xn, np.asarray(y)


def timed_cv(est, X, y):
    t0 = time.perf_counter()
    s = cv_pr_auc(est, X, y)
    return s, time.perf_counter() - t0


# ---------------------------------------------------------------- credit_g
Xc, yc = prep("credit_g")
print(f"=== credit_g  rows={len(yc)} pos_rate={yc.mean():.3f} feats={Xc.shape[1]} ===")

gbdt = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RS)
print("  GBDT (L013)          ", round(cv_pr_auc(gbdt, Xc, yc), 3))

xgb = XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist")
print("  XGB default          ", round(cv_pr_auc(xgb, Xc, yc), 3))

lgbm = LGBMClassifier(random_state=RS, n_jobs=-1, verbose=-1)
print("  LGBM default         ", round(cv_pr_auc(lgbm, Xc, yc), 3))

# num_leaves sweep (leaf-wise capacity knob)
print("  -- num_leaves sweep (credit_g) --")
for nl in [7, 15, 31, 63, 127]:
    m = LGBMClassifier(random_state=RS, n_jobs=-1, verbose=-1, num_leaves=nl, n_estimators=200, learning_rate=0.05)
    print(f"     num_leaves={nl:4d}  PR-AUC={cv_pr_auc(m, Xc, yc):.3f}")

# tuned LGBM
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
    param_dist, n_iter=40, scoring="average_precision", cv=cv, random_state=RS, n_jobs=-1,
)
search.fit(Xc, yc)
print("  LGBM tuned           ", round(float(search.best_score_), 3), "| best:", search.best_params_)

# ---------------------------------------------------------------- adult (speed)
Xa, ya = prep("adult")
print(f"\n=== adult  rows={len(ya)} pos_rate={ya.mean():.3f} feats={Xa.shape[1]} ===")
xgb_a = XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist")
lgbm_a = LGBMClassifier(random_state=RS, n_jobs=-1, verbose=-1)
s_x, t_x = timed_cv(xgb_a, Xa, ya)
s_l, t_l = timed_cv(lgbm_a, Xa, ya)
print(f"  XGB default   PR-AUC={s_x:.3f}  5-fold CV wall={t_x:.2f}s")
print(f"  LGBM default  PR-AUC={s_l:.3f}  5-fold CV wall={t_l:.2f}s")
print(f"  speedup (XGB/LGBM time) = {t_x / t_l:.2f}x")

# ---------------------------------------------------------------- GOSS unbiasedness demo
print("\n=== GOSS amplification (unbiased gradient-sum estimate) ===")
rng = np.random.RandomState(0)
n = 2000
g = rng.standard_normal(n)              # per-row gradients
region = rng.rand(n) < 0.5              # a candidate split's left region
a, b = 0.2, 0.1                          # keep top-a by |g|, sample b of the rest
order = np.argsort(-np.abs(g))
topN = int(a * n)
top_idx = order[:topN]
rest_idx = order[topN:]
samp = rng.choice(rest_idx, size=int(b * len(rest_idx)), replace=False)
fact = (1 - a) / b
true_left = g[region].sum()
w = np.zeros(n)
w[top_idx] = 1.0
w[samp] = fact
est_left = (w * g * region)[np.r_[top_idx, samp]].sum() if False else (w[region] * g[region]).sum()
used = (w > 0).sum()
print(f"  rows used = {used}/{n} ({used/n:.0%})  amplify factor (1-a)/b = {fact:.1f}")
print(f"  true left gradient sum      = {true_left:.3f}")
print(f"  GOSS-estimated left grad sum = {est_left:.3f}")
print(f"  abs error = {abs(est_left - true_left):.3f}")
print("done")
