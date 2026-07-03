"""Remaining L015 numbers with BOUNDED parallelism (no nested n_jobs=-1)."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")

import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0

def prep(name):
    X, y = load_tier_a(name)
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    Xn = X[num].copy()
    for c in [c for c in X.columns if c not in num]:
        Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return Xn, np.asarray(y)

# tuned LGBM on credit_g (bounded)
Xc, yc = prep("credit_g")
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
print(f"LGBM tuned credit_g PR-AUC = {float(search.best_score_):.3f}")
print(f"  best: {search.best_params_}")

# adult speed (bounded, same n_jobs for fairness)
Xa, ya = prep("adult")
print(f"adult rows={len(ya)} pos_rate={ya.mean():.3f} feats={Xa.shape[1]}")
for name, est in [
    ("XGB ", XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist")),
    ("LGBM", LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1)),
]:
    t0 = time.perf_counter()
    s = cv_pr_auc(est, Xa, ya)
    print(f"  {name} default  PR-AUC={s:.3f}  5-fold CV wall={time.perf_counter()-t0:.1f}s")

# GOSS unbiasedness demo
print("GOSS amplification demo:")
rng = np.random.RandomState(0)
n = 2000
g = rng.standard_normal(n)
region = rng.rand(n) < 0.5
a, b = 0.2, 0.1
order = np.argsort(-np.abs(g))
top_idx = order[: int(a * n)]
rest_idx = order[int(a * n):]
samp = rng.choice(rest_idx, size=int(b * len(rest_idx)), replace=False)
fact = (1 - a) / b
w = np.zeros(n)
w[top_idx] = 1.0
w[samp] = fact
true_left = g[region].sum()
est_left = (w[region] * g[region]).sum()
used = int((w > 0).sum())
print(f"  rows used = {used}/{n} ({used/n:.0%})  factor (1-a)/b = {fact:.1f}")
print(f"  true left grad sum = {true_left:.3f}  GOSS estimate = {est_left:.3f}  abs err = {abs(est_left-true_left):.3f}")
print("done")
