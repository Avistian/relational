#!/usr/bin/env python3
"""Download and cache Tier-A OpenML datasets for labs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

CACHE = Path(__file__).resolve().parent / "cache"

DATASETS = {
    "credit_g": {"openml_id": 31, "target": "class"},
    "adult": {"openml_id": 1590, "target": "class"},
    "bank_marketing": {"openml_id": 1461, "target": "Class"},
}


def fetch(name: str) -> pd.DataFrame:
    spec = DATASETS[name]
    bundle = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto")
    X = bundle.data
    y = bundle.target
    if hasattr(y, "name") and y.name:
        target_col = y.name
    else:
        target_col = spec["target"]
    df = X.copy()
    df[target_col] = y
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Dataset keys to fetch (default: all)",
    )
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        out = CACHE / f"{name}.parquet"
        df = fetch(name)
        df.to_parquet(out, index=False)
        print(f"Wrote {out} ({len(df)} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
