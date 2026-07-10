"""Verify Lesson 022 claims — label-leakage patterns (Kapoor & Narayanan 2022).

Two mechanisms, honestly isolated on Tier-C synthetic data (no network / OpenML):

  1. ILLEGITIMATE FEATURE (taxonomy L2) → a "reproducibility collapse" in the style
     of the paper's civil-war study: a leaked proxy for the label makes a COMPLEX model
     look like it crushes logistic regression; remove the leak and the gap collapses to
     near-zero (complex model does NOT substantively beat decades-old LR).

  2. NON-INDEPENDENCE / near-duplicates across the split (taxonomy L1.4 + L3.2): copies
     of the same underlying record land in both train and test under a random split, so
     CV over-reports; a GROUPED split (no record straddles the fold) is the honest number.

Verified numbers (sklearn 1.9.0, seed 0):
  Mechanism 1 (illegitimate feature / L2):
    WITH leak     RF 0.935  LR 0.719  gap(RF-LR) +0.217   <- "complex crushes LR"
    WITHOUT leak  RF 0.712  LR 0.721  gap(RF-LR) -0.009   <- complex ~= LR (collapse)
  Mechanism 2 (near-duplicates / L1.4 + L3.2):
    naive random 5-fold CV 0.948  ->  GroupKFold on record id 0.876  (leak +0.071)

Run: .venv/bin/python labs/_verify_l022.py
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 0


# ---------------------------------------------------------------------------
# Mechanism 1 — illegitimate feature (L2): the reproducibility collapse
# ---------------------------------------------------------------------------
def make_legit(n=4000, d=8, seed=SEED):
    """A moderately-hard, mostly-linear classification signal.
    LR should do well here; RF is close but not dramatically better."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([0.9, -0.8, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0])
    logit = X @ w + 1.8 * rng.normal(size=n)
    y = (logit > 0).astype(int)
    return X, y


def leaked_nonmonotone(y, flip=0.06, seed=SEED):
    """One ILLEGITIMATE column that encodes the label NON-MONOTONICALLY:
        y == 1  -> value in the OUTER bands  [0,0.33) U [0.66,1)
        y == 0  -> value in the MIDDLE band  [0.33,0.66)
    A tree recovers y with two splits (at 0.33 and 0.66); a linear model cannot,
    because the class means of the column are both ~0.5 (zero linear signal). This is
    a downstream/proxy column that would not exist at prediction time — it inflates the
    COMPLEX model's apparent edge over LR while doing ~nothing for LR."""
    rng = np.random.default_rng(seed + 1)
    noisy = np.where(rng.random(len(y)) < flip, 1 - y, y)
    val = np.empty(len(y))
    for i, yi in enumerate(noisy):
        if yi == 1:  # outer bands
            val[i] = rng.uniform(0.66, 1.0) if rng.random() < 0.5 else rng.uniform(0.0, 0.33)
        else:        # middle band
            val[i] = rng.uniform(0.33, 0.66)
    return val.reshape(-1, 1)


def collapse():
    X, y = make_legit()
    leak = leaked_nonmonotone(y)
    Xleak = np.hstack([X, leak])

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    def acc(fn, XX):
        return cross_val_score(fn(), XX, y, cv=cv, scoring="accuracy").mean()

    lr_honest, rf_honest = acc(lr, X), acc(rf, X)
    lr_leak, rf_leak = acc(lr, Xleak), acc(rf, Xleak)

    print("== Mechanism 1: illegitimate feature (L2) -> reproducibility collapse ==")
    print(f"  prevalence = {y.mean():.3f}, n={len(y)}, legit d={X.shape[1]}, +1 leaked non-monotone col")
    print(f"  WITH leak:     RF {rf_leak:.3f}   LR {lr_leak:.3f}   gap(RF-LR) {rf_leak-lr_leak:+.3f}")
    print(f"  WITHOUT leak:  RF {rf_honest:.3f}   LR {lr_honest:.3f}   gap(RF-LR) {rf_honest-lr_honest:+.3f}")
    print(f"  --> the leak inflates RF's apparent margin over LR from "
          f"{rf_honest-lr_honest:+.3f} to {rf_leak-lr_leak:+.3f}; remove it and complex ~= LR.")
    return dict(lr_honest=lr_honest, rf_honest=rf_honest, lr_leak=lr_leak, rf_leak=rf_leak)


# ---------------------------------------------------------------------------
# Mechanism 2 — near-duplicates / non-independence (L1.4 + L3.2)
# ---------------------------------------------------------------------------
def make_with_dups(n_unique=1500, dup_frac=0.6, d=6, seed=SEED):
    """Generate unique records, then add near-duplicate copies (same record measured
    again with tiny jitter). group id = the underlying record."""
    rng = np.random.default_rng(seed + 2)
    Xu = rng.normal(size=(n_unique, d))
    w = np.array([1.2, -1.0, 0.8, 0.0, 0.0, 0.0])
    yu = ((Xu @ w + 0.5 * rng.normal(size=n_unique)) > 0).astype(int)
    groups_u = np.arange(n_unique)

    n_dup = int(n_unique * dup_frac)
    pick = rng.integers(0, n_unique, size=n_dup)
    Xd = Xu[pick] + 0.01 * rng.normal(size=(n_dup, d))  # near-copies
    yd = yu[pick]
    groups_d = groups_u[pick]

    X = np.vstack([Xu, Xd])
    y = np.concatenate([yu, yd])
    groups = np.concatenate([groups_u, groups_d])
    perm = rng.permutation(len(y))
    return X[perm], y[perm], groups[perm]


def dedup_leak():
    X, y, groups = make_with_dups()
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

    naive = cross_val_score(
        rf(), X, y, cv=KFold(n_splits=5, shuffle=True, random_state=SEED), scoring="accuracy"
    ).mean()
    grouped = cross_val_score(
        rf(), X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring="accuracy"
    ).mean()

    print("\n== Mechanism 2: near-duplicates across the split (L1.4 / L3.2) ==")
    print(f"  n={len(y)} rows, {len(np.unique(groups))} unique records")
    print(f"  naive random 5-fold CV  {naive:.3f}   (copies straddle train/test -> optimistic)")
    print(f"  GroupKFold on record id {grouped:.3f}   (no record straddles -> honest)")
    print(f"  --> leak size = {naive - grouped:+.3f}")
    return dict(naive=naive, grouped=grouped)


if __name__ == "__main__":
    c = collapse()
    d = dedup_leak()
    print("\n== Summary ==")
    print(f"  collapse: gap {c['rf_leak']-c['lr_leak']:+.3f} (leak) -> {c['rf_honest']-c['lr_honest']:+.3f} (honest)")
    print(f"  dup leak: {d['naive']:.3f} (random) -> {d['grouped']:.3f} (grouped)")
