"""Verify the Lesson 033 feature-engineering diminishing-returns story on real credit_g.

Domingos 2012 ("A Few Useful Things to Know about Machine Learning"): feature
engineering is where most of the gains live, but returns diminish and each added
feature raises the overfitting/variance risk. We operationalize that as an
FE-budget experiment: start from the raw columns, add hand-crafted features one at
a time in a fixed sensible order, and measure 5-fold CV ROC-AUC (mean +/- std)
after each addition on a fixed HistGradientBoosting model (model held constant so
the ONLY thing changing is the feature set).

Expected shape: a steep early climb (the first few features carry most of the
signal), then a plateau, then a flat/declining tail where extra features stop
paying and CV variance widens -- the "where would you stop" decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relkit.data import load_tier_a  # noqa: E402

SEED = 0
N_SPLITS = 5


def build_features(X: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return X with the first k hand-crafted features appended.

    Feature order is fixed in advance (a sensible analyst's priority list), NOT
    greedily selected on the label -- so the curve is an honest "as you spend
    more effort" trace, not a cherry-picked optimum.
    """
    X = X.copy()
    dur = X["duration"].astype(float)
    amt = X["credit_amount"].astype(float)
    age = X["age"].astype(float)
    inst = X["installment_commitment"].astype(float)
    resid = X["residence_since"].astype(float)
    ncred = X["existing_credits"].astype(float)
    ndep = X["num_dependents"].astype(float)

    feats = [
        ("monthly_payment", amt / dur.clip(lower=1)),          # 1: burden per month
        ("amount_per_age", amt / age.clip(lower=1)),           # 2: exposure vs life stage
        ("installment_burden", inst * amt / 100.0),            # 3: rate x size
        ("amount_x_duration", amt * dur),                      # 4: total exposure
        ("credit_per_existing", amt / ncred.clip(lower=1)),    # 5: load per open line
        ("age_x_residence", age * resid),                      # 6: stability proxy
        ("log_amount", np.log1p(amt)),                         # 7: tame the tail
        ("dur_per_age", dur / age.clip(lower=1)),              # 8: horizon vs age
        ("deps_per_credit", ndep / ncred.clip(lower=1)),       # 9: obligation ratio
        ("amount_sq", amt ** 2),                               # 10: pure nonlinearity (low value)
    ]
    for name, series in feats[:k]:
        X[name] = series
    return X


def _pre(X: pd.DataFrame, *, scale: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        from sklearn.preprocessing import StandardScaler

        num_steps.append(("scale", StandardScaler()))
    from sklearn.pipeline import Pipeline as _P

    return ColumnTransformer(
        [
            ("num", _P(num_steps), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def cv_auc(X: pd.DataFrame, y: pd.Series, model: str) -> tuple[float, float]:
    from sklearn.pipeline import Pipeline

    if model == "gbdt":
        clf = HistGradientBoostingClassifier(random_state=SEED)
        pre = _pre(X, scale=False)
    else:  # linear
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        pre = _pre(X, scale=True)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main() -> None:
    X, y = load_tier_a("credit_g")
    print(f"credit_g: {X.shape[0]} rows, {X.shape[1]} raw columns, positive rate {y.mean():.3f}")
    for model in ("linear", "gbdt"):
        print(f"\n=== {model.upper()} ===")
        print(f"{'k features':>11} | {'CV ROC-AUC':>10} | {'std':>6} | {'delta':>7}")
        print("-" * 46)
        prev = None
        rows = []
        for k in range(0, 11):
            Xk = build_features(X, k)
            mean, std = cv_auc(Xk, y, model)
            delta = "" if prev is None else f"{mean - prev:+.4f}"
            print(f"{k:>11} | {mean:>10.4f} | {std:>6.4f} | {delta:>7}")
            rows.append((k, round(mean, 4), round(std, 4)))
            prev = mean
        print(f"{model}_means =", [r[1] for r in rows])
        print(f"{model}_stds  =", [r[2] for r in rows])


if __name__ == "__main__":
    main()
