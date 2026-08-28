"""Tier-A dataset loaders (OpenML cache or live fetch)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"

SPECS = {
    "credit_g": {"openml_id": 31, "target": "class"},
    "adult": {"openml_id": 1590, "target": "class"},
    "bank_marketing": {"openml_id": 1461, "target": "Class"},
    # Small binary tables for multi-dataset comparisons (NOTES standard #23). Mostly numeric
    # (favour nets) vs credit_g's categorical (favours trees) — so model ranks flip across them.
    "diabetes": {"openml_id": 37, "target": "class"},            # 768 rows, 8 numeric
    "blood_transfusion": {"openml_id": 1464, "target": "Class"}, # 748 rows, 4 numeric
    "kc1": {"openml_id": 1067, "target": "defects"},             # 2109 rows, 21 numeric
    "phoneme": {"openml_id": 1489, "target": "Class"},           # 5404 rows, 5 numeric
    "churn": {"openml_id": 40701, "target": "class"},            # 5000 rows, 20 (4 categorical)
    # Paper-results scale-up (standard #25). Higgs-small is a documented subsample of
    # the UCI Higgs used by NODE (Popov 2020, Table 1/5) and TabNet (Arik 2019, Table 5).
    # Full Higgs is 10.5M rows — name that gap in the ledger; do not quote Table 1/5
    # numbers from a 98k run as a MATCH.
    "higgs_small": {"openml_id": 23512, "target": "class"},       # ~98k rows, 28 numeric
}


def load_tier_a(name: str = "credit_g", *, use_cache: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load a Tier-A dataset; returns (X, y) with y as 0/1 int."""
    if name not in SPECS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(SPECS)}")
    cache_path = CACHE / f"{name}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        spec = SPECS[name]
        bundle = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto")
        df = bundle.data.copy()
        target = bundle.target
        tcol = target.name if getattr(target, "name", None) else spec["target"]
        df[tcol] = target
    tcol = SPECS[name]["target"]
    if tcol not in df.columns:
        tcol = df.columns[-1]
    y_raw = df[tcol]
    X = df.drop(columns=[tcol])
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        classes = sorted(y_raw.astype(str).unique())
        y = (y_raw.astype(str) == classes[-1]).astype(int)
    else:
        y = y_raw.astype(int)
    return X, y
