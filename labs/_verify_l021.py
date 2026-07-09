"""Verify Lesson 021 claims — temporal vs random splits under concept drift.

No network / OpenML (egress blocked): a controlled synthetic Tier-C stream with a
KNOWN, drifting decision rule. The point is the mechanism, honestly isolated:
a random (i.i.d.) split lets the model peek at the deployment period during
training, so it over-reports; a temporal split (train past -> test future) is the
honest estimate of what deployment faces.

Run: .venv/bin/python labs/_verify_l021.py
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

SEED = 0


def make_stream(n=6000, d=6, drift=np.pi / 2, noise=0.6, seed=SEED):
    """Rows sorted by time t in [0,1). The label rule ROTATES with time:
    early rows are labelled by feature x0, late rows by x1 (concept drift).
    Feature distribution is stationary (pure concept drift, no covariate shift)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    X = rng.normal(size=(n, d))
    theta = t * drift  # 0 -> drift
    logit = np.cos(theta) * X[:, 0] + np.sin(theta) * X[:, 1] + 0.4 * X[:, 2]
    logit = 1.6 * logit + rng.normal(scale=noise, size=n)
    y = (logit > 0).astype(int)
    return X, y, t


def temporal_holdout(model_fn, X, y, frac=0.8):
    cut = int(len(X) * frac)
    m = model_fn()
    m.fit(X[:cut], y[:cut])
    return m.score(X[cut:], y[cut:])


def random_holdout(model_fn, X, y, frac=0.8, seed=SEED):
    # same SIZE split, but rows chosen at random (i.i.d. assumption)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * frac)
    tr, te = idx[:cut], idx[cut:]
    m = model_fn()
    m.fit(X[tr], y[tr])
    return m.score(X[te], y[te])


def main():
    X, y, t = make_stream()
    print(f"stream: n={len(X)}, prevalence={y.mean():.3f}, features={X.shape[1]}")

    models = {
        "logistic": lambda: LogisticRegression(max_iter=1000),
        "hist_gbdt": lambda: HistGradientBoostingClassifier(random_state=SEED),
    }

    print("\n== Random 5-fold CV (shuffle=True) vs temporal holdout ==")
    for name, fn in models.items():
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        rand_cv = cross_val_score(fn(), X, y, cv=cv, scoring="accuracy").mean()
        rand_ho = random_holdout(fn, X, y)
        temp_ho = temporal_holdout(fn, X, y)
        tscv = TimeSeriesSplit(n_splits=5)
        ts_cv = cross_val_score(fn(), X, y, cv=tscv, scoring="accuracy").mean()
        print(
            f"{name:10s}  random-CV {rand_cv:.3f}  random-holdout {rand_ho:.3f}  "
            f"|  TimeSeriesSplit {ts_cv:.3f}  temporal-holdout {temp_ho:.3f}  "
            f"|  optimism gap (randCV - tempHO) = {rand_cv - temp_ho:+.3f}"
        )

    # ---- drift made visible: sign-agreement of each feature with the label per time bucket
    print("\n== Concept drift across time buckets (corr of feature with y) ==")
    nb = 5
    edges = np.linspace(0, len(X), nb + 1).astype(int)
    header = "bucket      " + "  ".join(f"x{j}" for j in range(3))
    print(header)
    for b in range(nb):
        s, e = edges[b], edges[b + 1]
        corrs = [np.corrcoef(X[s:e, j], y[s:e])[0, 1] for j in range(3)]
        lbl = f"[{t[s]:.2f},{t[e-1]:.2f}]"
        print(f"{lbl:11s} " + "  ".join(f"{c:+.2f}" for c in corrs))

    # ---- prevalence per bucket (is the base rate stable?)
    print("\n== Prevalence per time bucket ==")
    for b in range(nb):
        s, e = edges[b], edges[b + 1]
        print(f"  bucket {b}: {y[s:e].mean():.3f}")

    # ---- what a fixed global model 'thinks' vs the true drifting rule
    print("\n== Why random CV is optimistic (one sentence) ==")
    print(
        "  Random folds put late-period rows in BOTH train and test, so the model "
        "sees the current rule while training; the temporal split forbids that."
    )


if __name__ == "__main__":
    main()
