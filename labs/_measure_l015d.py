"""Honest speedup: LightGBM/XGBoost-hist vs a CONVENTIONAL (pre-histogram) GBDT.
This is the regime the 2017 paper's '20x' referred to."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RS, NJ, N_EST = 0, 4, 100
X, y = make_classification(n_samples=50_000, n_features=50, n_informative=15,
                           n_redundant=0, random_state=RS)

def ft(est):
    t0 = time.perf_counter(); est.fit(X, y); return time.perf_counter() - t0

t_conv = ft(GradientBoostingClassifier(n_estimators=N_EST, random_state=RS))          # pre-hist
t_xgb = ft(XGBClassifier(n_estimators=N_EST, random_state=RS, n_jobs=NJ, eval_metric="logloss", tree_method="hist"))
t_lgbm = ft(LGBMClassifier(n_estimators=N_EST, random_state=RS, n_jobs=NJ, verbose=-1))
t_goss = ft(LGBMClassifier(n_estimators=N_EST, random_state=RS, n_jobs=NJ, verbose=-1, boosting_type="goss"))

print(f"n=50000 p=50 trees={N_EST}")
print(f"  sklearn GBDT (conventional, pre-hist) : {t_conv:6.2f}s")
print(f"  XGBoost (hist)                        : {t_xgb:6.2f}s   ({t_conv/t_xgb:.1f}x vs conventional)")
print(f"  LightGBM (gbdt default)               : {t_lgbm:6.2f}s   ({t_conv/t_lgbm:.1f}x vs conventional)")
print(f"  LightGBM (boosting_type='goss')       : {t_goss:6.2f}s   ({t_conv/t_goss:.1f}x vs conventional)")
print("done")
