"""Verify Lesson 016 (CatBoost) numbers. Run from labs/ so `relkit` imports:

    cd labs && OMP_NUM_THREADS=3 ../.venv/bin/python _verify_l016.py

Produces the real numbers used in the lesson HTML and lab:
  1. Greedy (own-row-included) target statistics LEAK on a pure-noise
     high-cardinality category; ordered target statistics do not
     (Prokhorenkova et al. 2018, §3-4 — target leakage / prediction shift).
  2. CatBoost native categoricals vs label-encoded XGBoost / native-cat
     LightGBM on real categorical data (credit_g, adult), CV PR-AUC.
  3. Ordered boosting on / off (small data): does the anti-prediction-shift
     scheme help on tiny, noisy data?
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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from relkit.data import load_tier_a

RS = 0


def section(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


def cv_pr_auc(est, X, y, *, is_catboost=False):
    """Manual 5-fold PR-AUC. CatBoost's cat_features breaks sklearn.clone, so
    we build a fresh estimator per fold instead of cloning."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    Xa = X.reset_index(drop=True)
    scores = []
    for tr, te in cv.split(Xa, y):
        m = CatBoostClassifier(**est) if is_catboost else clone(est)
        m.fit(Xa.iloc[tr], y[tr])
        p = m.predict_proba(Xa.iloc[te])[:, 1]
        scores.append(average_precision_score(y[te], p))
    return float(np.mean(scores))


# ---------------------------------------------------------------- 1. TS leakage
section("1. Greedy target statistics LEAK; ordered TS do not (pure-noise cat)")
rng = np.random.RandomState(0)
n = 2000
K = 700  # high cardinality -> ~3 rows per category
cat = rng.randint(0, K, size=n)
y = (rng.rand(n) < 0.3).astype(int)  # target INDEPENDENT of the category (pure noise)
prior = y.mean()


def greedy_ts(cat, y):
    """Mean target per category over the WHOLE column, including each row's own y."""
    out = np.empty(len(y), float)
    for k in np.unique(cat):
        m = cat == k
        out[m] = y[m].mean()  # includes the row itself -> leakage
    return out


def ordered_ts(cat, y, a=1.0, prior=0.5, seed=0):
    """CatBoost-style ordered TS: encode row i from PRIOR rows in a random permutation."""
    perm = np.random.RandomState(seed).permutation(len(y))
    running_sum: dict[int, float] = {}
    running_cnt: dict[int, int] = {}
    enc = np.empty(len(y), float)
    for i in perm:
        c = cat[i]
        s = running_sum.get(c, 0.0)
        n_seen = running_cnt.get(c, 0)
        enc[i] = (s + a * prior) / (n_seen + a)  # uses only earlier rows
        running_sum[c] = s + y[i]
        running_cnt[c] = n_seen + 1
    return enc


g_enc = greedy_ts(cat, y)
o_enc = ordered_ts(cat, y, a=1.0, prior=prior, seed=0)
# AUC of the encoding treated as a score for y: >0.5 means the feature "predicts" y.
auc_greedy = roc_auc_score(y, g_enc)
auc_ordered = np.mean([roc_auc_score(y, ordered_ts(cat, y, 1.0, prior, s)) for s in range(20)])
print(f"category is PURE NOISE (independent of y); prevalence = {prior:.3f}, cardinality = {K}")
print(f"  greedy TS (own row included)  AUC(enc, y) = {auc_greedy:.3f}   <- spurious signal = LEAK")
print(f"  ordered TS (prior rows only)  AUC(enc, y) = {auc_ordered:.3f}   <- ~0.5, no leak (mean over 20 perms)")

# ---------------------------------------------------------------- 2. real categorical data
section("2. CatBoost native cats vs XGBoost (label-enc) vs LightGBM (native)")


def load_native(name):
    """Return (X_str, y, cat_feature_names) with cats as str for CatBoost/LGBM."""
    X, y = load_tier_a(name)
    y = np.asarray(y)
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    Xs = X.copy()
    for c in cat:
        col = Xs[c].astype("object")
        Xs[c] = col.where(col.notna(), "missing").astype(str)  # CatBoost needs str, not NaN
    return Xs, y, num, cat


def label_encode(X, num, cat):
    Xn = X[num].copy()
    for c in cat:
        Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return Xn


for ds in ["credit_g", "adult"]:
    Xs, y, num, cat = load_native(ds)
    cat_idx = [Xs.columns.get_loc(c) for c in cat]
    Xle = label_encode(Xs, num, cat)
    # LightGBM native categoricals: cast to pandas 'category'
    Xlg = Xs.copy()
    for c in cat:
        Xlg[c] = Xlg[c].astype("category")

    cb_params = dict(
        iterations=400, learning_rate=0.05, depth=6, random_seed=RS,
        cat_features=cat_idx, verbose=0, thread_count=4, allow_writing_files=False,
    )
    xgb = XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist")
    lgbm = LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1)

    cb_s = cv_pr_auc(cb_params, Xs, y, is_catboost=True)
    xgb_s = cv_pr_auc(xgb, Xle, y)
    lgbm_s = cv_pr_auc(lgbm, Xlg, y)
    print(f"\n{ds}: rows={len(y)} pos_rate={y.mean():.3f} #cat={len(cat)} #num={len(num)}")
    print(f"  XGBoost  (label-encoded) CV PR-AUC = {xgb_s:.3f}")
    print(f"  LightGBM (native cats)   CV PR-AUC = {lgbm_s:.3f}")
    print(f"  CatBoost (native cats)   CV PR-AUC = {cb_s:.3f}")

# ---------------------------------------------------------------- 3. ordered boosting toggle
section("3. Ordered boosting vs Plain on small data (credit_g)")
Xs, y, num, cat = load_native("credit_g")
cat_idx = [Xs.columns.get_loc(c) for c in cat]
common = dict(iterations=400, learning_rate=0.05, depth=6, random_seed=RS,
              cat_features=cat_idx, verbose=0, thread_count=4, allow_writing_files=False)
ordered = cv_pr_auc(dict(boosting_type="Ordered", **common), Xs, y, is_catboost=True)
plain = cv_pr_auc(dict(boosting_type="Plain", **common), Xs, y, is_catboost=True)
print(f"  boosting_type='Ordered' (default on small data) CV PR-AUC = {ordered:.3f}")
print(f"  boosting_type='Plain'   (classic GBDT)          CV PR-AUC = {plain:.3f}")
print("\nDONE")
