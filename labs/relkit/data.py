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
