"""Verify Lesson 017 (Hyperparameter search) numbers. Run from labs/ so `relkit` imports:

    cd labs && OMP_NUM_THREADS=3 ../.venv/bin/python _verify_l017.py

Produces the real numbers used in the lesson HTML and lab:
  1. Bergstra & Bengio 2012 mechanism (synthetic response surface): at an EQUAL
     budget, random search tries far more DISTINCT values on the important axis
     than grid search, and finds a better optimum -- and the gap widens as more
     irrelevant dimensions are added (low effective dimensionality).
  2. Tuning XGBoost on credit_g: GridSearchCV vs RandomizedSearchCV at an equal
     budget, inner CV PR-AUC (callback to L014 tuned XGB ~0.896).
  3. The search score is OPTIMISTIC: nested CV gives an honest, lower estimate
     (callback to L004 selection bias).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from relkit.data import load_tier_a

RS = 0


def section(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ---------------------------------------------------------------- 1. mechanism
section("1. Grid vs random on a synthetic surface (Bergstra & Bengio 2012)")


def response(x, d_important=1):
    """A surface on [0,1]^d. Only the first `d_important` dims matter (sharp
    Gaussian bump at 0.5); the rest are near-flat 'irrelevant' knobs."""
    x = np.atleast_2d(x)
    imp = np.exp(-((x[:, :d_important] - 0.5) ** 2).sum(1) / (2 * 0.08**2))
    unimp = 0.02 * np.cos(6 * x[:, d_important:]).sum(1)
    return imp + unimp


def grid_search(d, budget, d_important=1):
    """Evaluate a near-`budget` full factorial grid; return best value found."""
    k = max(1, round(budget ** (1.0 / d)))  # points per axis
    axis = np.linspace(0.05, 0.95, k)
    mesh = np.array(np.meshgrid(*([axis] * d))).reshape(d, -1).T
    vals = response(mesh, d_important)
    distinct_important = k  # distinct values tried on ONE important axis
    return vals.max(), mesh.shape[0], distinct_important


def random_search(d, budget, d_important=1, seed=0):
    pts = np.random.RandomState(seed).rand(budget, d)
    vals = response(pts, d_important)
    distinct_important = len(np.unique(pts[:, 0]))
    return vals.max(), budget, distinct_important


for d, budget in [(2, 9), (2, 25), (5, 32), (8, 64)]:
    g_best, g_n, g_dist = grid_search(d, budget)
    r = np.array([random_search(d, budget, seed=s) for s in range(200)])
    r_best, r_dist = r[:, 0].mean(), r[:, 2].mean()
    print(
        f"d={d} budget≈{budget:3d} | grid: n={g_n:3d}, distinct important-axis values={g_dist:2d}, "
        f"best={g_best:.3f} | random: n={budget:3d}, distinct≈{r_dist:.0f}, best={r_best:.3f} "
        f"(mean/200 seeds)  Δ={r_best - g_best:+.3f}"
    )

# gap vs number of irrelevant dims at fixed budget
print("\nOptimum found as irrelevant dims grow (budget=27, 1 important dim):")
for d in [1, 2, 4, 8]:
    g_best, _, _ = grid_search(d, 27, d_important=1)
    r_best = np.mean([random_search(d, 27, d_important=1, seed=s)[0] for s in range(200)])
    print(f"  d={d}: grid best={g_best:.3f}  random best={r_best:.3f}  Δ={r_best - g_best:+.3f}")

# ---------------------------------------------------------------- 2. tune XGB
section("2. Tuning XGBoost on credit_g: GridSearchCV vs RandomizedSearchCV")

X, y = load_tier_a("credit_g")
y = np.asarray(y)
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = [c for c in X.columns if c not in num]
Xle = X[num].copy()
for c in cat:
    Xle[c] = LabelEncoder().fit_transform(X[c].astype(str))

base = XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist")
inner = StratifiedKFold(5, shuffle=True, random_state=RS)

# default (untuned) baseline for reference
default_score = cross_val_score(base, Xle, y, cv=inner, scoring="average_precision").mean()
print(f"XGBoost default          CV PR-AUC = {default_score:.3f}")

# grid over 3 params: 3*3*3 = 27 configs
grid = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.1, 0.3],
    "n_estimators": [200, 400, 600],
}
grid_n = int(np.prod([len(v) for v in grid.values()]))
gs = GridSearchCV(base, grid, scoring="average_precision", cv=inner, n_jobs=4)
gs.fit(Xle, y)
print(f"GridSearchCV   ({grid_n} configs) best CV PR-AUC = {gs.best_score_:.3f}  {gs.best_params_}")

# random over 5 params (2 continuous) at the SAME budget
dists = {
    "max_depth": randint(3, 8),
    "learning_rate": loguniform(0.01, 0.3),
    "n_estimators": randint(150, 700),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}
rs = RandomizedSearchCV(
    base, dists, n_iter=grid_n, scoring="average_precision",
    cv=inner, n_jobs=4, random_state=RS,
)
rs.fit(Xle, y)
print(f"RandomizedSearchCV ({grid_n} draws) best CV PR-AUC = {rs.best_score_:.3f}  "
      f"{ {k: (round(v,3) if isinstance(v,float) else v) for k,v in rs.best_params_.items()} }")

# ---------------------------------------------------------------- 3. nested CV
section("3. The search score is optimistic -- nested CV is honest (L004)")
outer = StratifiedKFold(5, shuffle=True, random_state=RS)
nested = cross_val_score(
    RandomizedSearchCV(base, dists, n_iter=grid_n, scoring="average_precision",
                       cv=inner, n_jobs=4, random_state=RS),
    Xle, y, cv=outer, scoring="average_precision",
).mean()
print(f"RandomizedSearchCV best_score_ (optimistic) = {rs.best_score_:.3f}")
print(f"Nested CV estimate (honest)                 = {nested:.3f}")
print(f"Optimism gap                                = {rs.best_score_ - nested:+.3f}")
print("\nDONE")
