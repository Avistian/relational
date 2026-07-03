"""Honest speed comparison: LightGBM vs XGBoost as n and #features grow.
Single fit, same n_estimators, same thread budget."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RS = 0
NJ = 4
N_EST = 300

def fit_time(est, X, y):
    t0 = time.perf_counter()
    est.fit(X, y)
    return time.perf_counter() - t0

for n, p, inf in [(50_000, 20, 10), (200_000, 100, 30), (200_000, 400, 40)]:
    X, y = make_classification(n_samples=n, n_features=p, n_informative=inf,
                               n_redundant=0, n_repeated=0, random_state=RS)
    xgb = XGBClassifier(n_estimators=N_EST, random_state=RS, n_jobs=NJ,
                        eval_metric="logloss", tree_method="hist")
    lgbm = LGBMClassifier(n_estimators=N_EST, random_state=RS, n_jobs=NJ, verbose=-1)
    tx = fit_time(xgb, X, y)
    tl = fit_time(lgbm, X, y)
    print(f"n={n:>7} p={p:>3} inf={inf:>2} | XGB {tx:6.2f}s  LGBM {tl:6.2f}s  speedup {tx/tl:4.2f}x")
print("done")
