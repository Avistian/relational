"""Dump the per-iteration CASH trace (algo, val AUC) for seed 0 so cash-search-viz uses real data."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _verify_l029 import make_prep, sample_config, fit_eval, load_tier_a
from sklearn.model_selection import train_test_split

X, y = load_tier_a("credit_g")
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=ytmp)
rng = np.random.RandomState(0)
trace = []
best = -1.0
for i in range(40):
    algo, clf = sample_config(rng)
    try:
        pipe, va = fit_eval(clf, make_prep(X), Xtr, ytr, Xva, yva)
    except Exception:
        continue
    best = max(best, va)
    trace.append({"i": len(trace) + 1, "algo": str(algo), "val": round(float(va), 3), "best": round(float(best), 3)})
print(json.dumps(trace))
