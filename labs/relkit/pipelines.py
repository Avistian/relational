"""Reusable sklearn baseline pipeline from Q1 checkpoint."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_baseline_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    *,
    random_state: int = 0,
) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(random_state=random_state)),
        ]
    )
