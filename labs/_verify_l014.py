"""Verify Lesson 014 (XGBoost) numbers. Run from labs/ so `relkit` imports.

Produces the real numbers used in the lesson HTML and lab:
  1. Regularized leaf weight  w* = -G/(H+lambda)  matches XGBoost's leaf output.
  2. Structure-score split gain (children - parent - gamma) matches the sign of keeping a split.
  3. credit_g: sklearn GBDT vs XGBoost default vs a small tuned XGBoost (CV PR-AUC).
  4. adult (stretch): boosting pulls further ahead with data + signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0


def section(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)


# ---------------------------------------------------------------- 1 + 2
section("1-2. Regularized leaf weight & split gain (by hand vs XGBoost)")

# Toy: squared-error regression so g_i = pred - y, h_i = 1 with base_score 0.5.
rng = np.random.RandomState(0)
Xr = rng.uniform(0, 1, size=(40, 1))
yr = (Xr[:, 0] > 0.5).astype(float) * 4.0 + rng.normal(0, 0.2, 40)

LAMBDA = 3.0
booster = XGBRegressor(
    n_estimators=1, max_depth=1, learning_rate=1.0,
    reg_lambda=LAMBDA, reg_alpha=0.0, gamma=0.0, base_score=0.5,
    objective="reg:squarederror", random_state=RS,
).fit(Xr, yr)

pred0 = np.full(len(yr), 0.5)
g = pred0 - yr          # gradient of 1/2 (pred-y)^2
h = np.ones_like(yr)    # hessian = 1

df = booster.get_booster().trees_to_dataframe()
split_row = df[df["Feature"] != "Leaf"].iloc[0]
thr = float(split_row["Split"])
left_mask = Xr[:, 0] < thr
GL, HL = g[left_mask].sum(), h[left_mask].sum()
GR, HR = g[~left_mask].sum(), h[~left_mask].sum()

wL = -GL / (HL + LAMBDA)
wR = -GR / (HR + LAMBDA)

leaves = df[df["Feature"] == "Leaf"].sort_values("Node")
xgb_leaf_vals = leaves["Gain"].to_numpy()  # for leaves, 'Gain' col holds the leaf weight
print(f"split threshold (XGBoost)      : {thr:.4f}")
print(f"leaf weight by hand  L/R        : {wL:+.4f} / {wR:+.4f}")
print(f"leaf weight XGBoost  (sorted)   : {xgb_leaf_vals[0]:+.4f} / {xgb_leaf_vals[1]:+.4f}")

# structure score of a node = G^2 / (H + lambda); gain = L + R - parent - gamma
def score(G, H, lam=LAMBDA):
    return (G * G) / (H + lam)

G, H = g.sum(), h.sum()
raw_gain = 0.5 * (score(GL, HL) + score(GR, HR) - score(G, H))
print(f"\nparent score G^2/(H+λ)          : {score(G, H):.4f}")
print(f"raw split gain ½[L+R−parent]    : {raw_gain:.4f}")
for gamma in (0.0, raw_gain * 0.5, raw_gain * 1.5):
    net = raw_gain - gamma
    print(f"  gamma={gamma:6.3f} -> net gain {net:+.4f}  ->  {'KEEP' if net > 0 else 'PRUNE'}")

# lambda shrinks leaf weights toward 0
print("\nλ shrinks the leaf weight (left leaf):")
for lam in (0.0, 1.0, 3.0, 10.0):
    print(f"  λ={lam:5.1f}  w*_L = {-GL/(HL+lam):+.4f}")

# ---------------------------------------------------------------- 3
section("3. credit_g — sklearn GBDT vs XGBoost default vs tuned (CV PR-AUC)")

X, y = load_tier_a("credit_g")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
Xn = X[num_cols].copy()
for c in cat_cols:
    Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
y = np.asarray(y)
print(f"rows={len(y)} pos_rate={y.mean():.3f} features={Xn.shape[1]}")

gb = cv_pr_auc(
    GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RS),
    Xn, y,
)
xgb_default = cv_pr_auc(
    XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
    Xn, y,
)

param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
    "min_child_weight": [1, 3, 5],
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
search = RandomizedSearchCV(
    XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
    param_dist, n_iter=40, scoring="average_precision", cv=cv, random_state=RS, n_jobs=-1,
)
search.fit(Xn, y)
xgb_tuned = float(search.best_score_)

print(f"sklearn GBDT (L013 config)      CV PR-AUC = {gb:.3f}")
print(f"XGBoost default                 CV PR-AUC = {xgb_default:.3f}")
print(f"XGBoost tuned (40-iter search)  CV PR-AUC = {xgb_tuned:.3f}")
print(f"best params: {search.best_params_}")

# ---------------------------------------------------------------- 4
section("4. adult (stretch) — XGBoost default vs tuned")
Xa, ya = load_tier_a("adult")
na = Xa.select_dtypes(include=[np.number]).columns.tolist()
Xan = Xa[na].copy()
for c in [c for c in Xa.columns if c not in na]:
    Xan[c] = LabelEncoder().fit_transform(Xa[c].astype(str))
ya = np.asarray(ya)
xgb_a = cv_pr_auc(
    XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
    Xan, ya,
)
gb_a = cv_pr_auc(
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RS),
    Xan, ya,
)
print(f"adult rows={len(ya)} pos_rate={ya.mean():.3f}")
print(f"sklearn GBDT   CV PR-AUC = {gb_a:.3f}")
print(f"XGBoost default CV PR-AUC = {xgb_a:.3f}")

print("\nDONE")
