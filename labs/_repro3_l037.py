"""L037 — which "seed" actually matters, and how wide is the irreducible spread?

`src.modeling` sets `RANDOM_STATE = 0` once and passes it to both the LightGBM
estimator and the `StratifiedGroupKFold(shuffle=True)` splitter, so the README
sentence "all randomness is seeded" covers two completely different things:

  model seed    -- consulted only if LightGBM samples (bagging_fraction < 1,
                   feature_fraction < 1, extra_trees). This configuration does
                   none of those, so `_repro2_l037.py` finds seeds 0-4 bitwise
                   identical: the model seed is inert.
  splitter seed -- decides WHICH persons land in which fold. Nothing about the
                   model changes, but the number being reported does.

This harness varies only the splitter seed and reports the spread of the mean
log-loss. That spread is the yardstick a reproduction tolerance has to clear:
a gate tighter than it fires on nothing but the fold draw.

Usage: python _repro3_l037.py
Writes labs/_repro3_l037_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HOMEWORK = Path("/home/avist/Projects/homework")
sys.path.insert(0, str(HOMEWORK))

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.features import build_dataset, make_feature_pipeline, split_xy  # noqa: E402
from src.modeling import CLASSES, RANDOM_STATE  # noqa: E402

OUT = Path(__file__).with_name("_repro3_l037_results.json")


def main() -> None:
    d = HOMEWORK / "data"
    df = build_dataset(
        pd.read_parquet(d / "persons.parquet"),
        pd.read_parquet(d / "situations.parquet"),
        pd.read_parquet(d / "responses.parquet"),
    )
    df_feat, y, mask = split_xy(df)
    X = make_feature_pipeline().fit_transform(df_feat).astype(np.float32)[mask]
    groups = df.loc[mask, "person_id"].to_numpy()
    y_idx = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    res: dict = {}
    means = []
    for split_seed in range(5):
        per_fold = []
        t0 = time.time()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        for i, (tr, te) in enumerate(cv.split(X, y_idx, groups=groups)):
            m = LGBMClassifier(
                objective="multiclass", n_estimators=400, learning_rate=0.05,
                num_leaves=31, min_child_samples=50, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=6, verbose=-1,
            ).fit(X[tr], y_idx[tr])
            per_fold.append(float(log_loss(y_idx[te], m.predict_proba(X[te]),
                                           labels=list(range(len(CLASSES))))))
        res[f"split_seed_{split_seed}"] = {
            "per_fold_log_loss": per_fold,
            "mean_log_loss": float(np.mean(per_fold)),
            "std_log_loss": float(np.std(per_fold, ddof=1)),
            "seconds": round(time.time() - t0, 1),
        }
        means.append(float(np.mean(per_fold)))
        print(f"split_seed={split_seed}: mean ll {means[-1]:.6f} "
              f"(fold sd {res[f'split_seed_{split_seed}']['std_log_loss']:.4f})", flush=True)
        OUT.write_text(json.dumps(res, indent=2))

    res["_summary"] = {
        "mean_log_loss_by_split_seed": means,
        "min": float(min(means)),
        "max": float(max(means)),
        "range": float(max(means) - min(means)),
        "std": float(np.std(means, ddof=1)),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["_summary"], indent=2))


if __name__ == "__main__":
    main()
