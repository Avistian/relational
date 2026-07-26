"""L037 — is the pooled-vs-per-fold ECE gap a fold effect or a binning artifact?

The submission reports top-label ECE twice for the same shipped model on the
same out-of-fold predictions:

  README.md + report.md §4 ship-gate table : 0.018   (pooled: bin all 5,587 OOF rows once)
  report.md §3.1 model-selection table     : 0.0332  (mean of five per-fold ECEs)

Log-loss is a plain mean over rows, so pooling and averaging agree to 5 decimals.
ECE is a *binned* statistic, so they need not — and the question the lesson has to
answer honestly is whether the 1.9x gap says something about the folds or is
simply the small-sample upward bias of a binned estimator.

Test: re-partition the SAME pooled OOF matrix into five RANDOM blocks of 1,117
rows (no fold structure at all) and average their ECEs. If random blocks also
land near 0.033, the gap is an artifact of binning 1,117 rows instead of 5,587.
Also sweeps block size to trace the bias curve.

Usage: python _ece_estimator_l037.py
Writes labs/_ece_estimator_l037_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOMEWORK = Path("/home/avist/Projects/homework")
sys.path.insert(0, str(HOMEWORK))

from src.features import build_dataset, split_xy  # noqa: E402
from src.modeling import CLASSES, _ece, cv_splitter  # noqa: E402

OUT = Path(__file__).with_name("_ece_estimator_l037_results.json")
RNG = np.random.default_rng(0)


def main() -> None:
    z = np.load(HOMEWORK / "artifacts" / "oof_M2a_lgbm_km.npz", allow_pickle=True)
    probs = np.asarray(z["probs"], dtype=np.float64)
    y_idx = np.asarray(z["y_idx"], dtype=int)
    n = len(y_idx)

    d = HOMEWORK / "data"
    df = build_dataset(
        pd.read_parquet(d / "persons.parquet"),
        pd.read_parquet(d / "situations.parquet"),
        pd.read_parquet(d / "responses.parquet"),
    )
    _, y, mask = split_xy(df)
    groups = df.loc[mask, "person_id"].to_numpy()
    assert len(groups) == n

    res: dict = {"n": int(n), "n_classes": len(CLASSES)}
    res["ece_pooled"] = float(_ece(probs, y_idx))

    # The five real CV folds, recomputed from the same splitter the submission used.
    fold_eces = [
        float(_ece(probs[te], y_idx[te]))
        for _, te in cv_splitter().split(probs, y_idx, groups=groups)
    ]
    res["ece_per_real_fold"] = fold_eces
    res["ece_mean_real_folds"] = float(np.mean(fold_eces))

    # Five RANDOM blocks of the same size — no fold structure, no grouping.
    reps = []
    for _ in range(500):
        perm = RNG.permutation(n)
        blocks = np.array_split(perm, 5)
        reps.append(np.mean([_ece(probs[b], y_idx[b]) for b in blocks]))
    res["ece_mean_random_blocks_of_5"] = {
        "mean": float(np.mean(reps)),
        "std": float(np.std(reps, ddof=1)),
        "p05": float(np.percentile(reps, 5)),
        "p95": float(np.percentile(reps, 95)),
        "reps": len(reps),
    }

    # Bias curve: ECE of a random subsample vs its size.
    curve = {}
    for size in (200, 500, 1117, 2000, 3000, 5587):
        vals = []
        for _ in range(300):
            idx = RNG.choice(n, size=size, replace=False)
            vals.append(_ece(probs[idx], y_idx[idx]))
        curve[str(size)] = [float(np.mean(vals)), float(np.std(vals, ddof=1))]
    res["ece_vs_subsample_size"] = curve

    # Fit ECE(n) = a + b/sqrt(n) on the two LARGEST sample sizes only, then check
    # how well it predicts the four smaller ones. `a` is the sampling-noise-free
    # limit the curve is heading for — an extrapolation, not a measurement.
    n_big, n_big2 = 5587, 3000
    e1 = curve[str(n_big)][0]
    e2 = curve[str(n_big2)][0]
    b = (e2 - e1) / (n_big2 ** -0.5 - n_big ** -0.5)
    a = e1 - b * n_big ** -0.5
    res["bias_fit"] = {
        "form": "ECE_hat(n) = a + b/sqrt(n)",
        "fitted_on": [n_big, n_big2],
        "a": float(a),
        "b": float(b),
        "predictions": {
            k: {
                "measured": curve[k][0],
                "predicted": float(a + b * int(k) ** -0.5),
                "rel_error": float(abs(a + b * int(k) ** -0.5 - curve[k][0]) / curve[k][0]),
            }
            for k in curve
        },
    }

    # Control: a PERFECTLY calibrated predictor (labels resampled from its own
    # probabilities) has true ECE = 0 by construction. Whatever the estimator
    # reports for it at each n is pure bias.
    ctrl = {}
    for size in (50, 107, 200, 500, 1117, 2000, 3000, 5587):
        vals = []
        for _ in range(300):
            idx = RNG.choice(n, size=size, replace=False)
            p = probs[idx]
            y_syn = np.array([RNG.choice(len(CLASSES), p=row / row.sum()) for row in p])
            vals.append(_ece(p, y_syn))
        ctrl[str(size)] = [float(np.mean(vals)), float(np.std(vals, ddof=1))]
    res["ece_on_perfectly_calibrated_control"] = ctrl

    # Same question for log-loss, which is a plain per-row mean.
    from sklearn.metrics import log_loss

    ll_folds = [
        float(log_loss(y_idx[te], probs[te], labels=list(range(len(CLASSES)))))
        for _, te in cv_splitter().split(probs, y_idx, groups=groups)
    ]
    res["log_loss_pooled"] = float(log_loss(y_idx, probs, labels=list(range(len(CLASSES)))))
    res["log_loss_mean_real_folds"] = float(np.mean(ll_folds))

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
