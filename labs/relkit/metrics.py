"""Leakage-safe evaluation helpers."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cv_scores(
    pipe,
    X,
    y,
    *,
    scoring: str | list[str] = "average_precision",
    n_splits: int = 5,
    random_state: int = 0,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    if isinstance(scoring, str):
        scoring = [scoring]
    out: dict[str, float] = {}
    for s in scoring:
        out[s] = float(cross_val_score(pipe, X, y, cv=cv, scoring=s).mean())
    return out


def cv_pr_auc(pipe, X, y, *, n_splits: int = 5, random_state: int = 0) -> float:
    return cv_scores(
        pipe, X, y, scoring="average_precision", n_splits=n_splits, random_state=random_state
    )["average_precision"]
