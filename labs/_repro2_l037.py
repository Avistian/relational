"""L037 second harness — the knobs that DO move the number, and how far.

`_repro_l037.py` sweeps thread count and LightGBM's `deterministic` flag. This
one covers the rest of the nondeterminism taxonomy on the same pipeline, same
person-grouped 5-fold CV, same OOF-hash comparison against `ref`:

  ref        n_jobs=6, float32 input, seed 0, LightGBM's default histogram mode
  rowwise    force_row_wise=True   -- one of the two modes LightGBM auto-picks
  colwise    force_col_wise=True   -- the other one; the choice is made by TIMING
                                      your machine, so it depends on load, not code
  perm       training rows presented in a different (seeded) order
  f64        X kept in float64 instead of the pipeline's .astype(np.float32)
  seed1..4   random_state varied -- the run-to-run spread you cannot remove, and
             therefore the yardstick any reproduction tolerance has to beat

Usage: python _repro2_l037.py
Writes labs/_repro2_l037_results.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HOMEWORK = Path("/home/avist/Projects/homework")
sys.path.insert(0, str(HOMEWORK))

from sklearn.metrics import log_loss  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402

from src.features import build_dataset, make_feature_pipeline, split_xy  # noqa: E402
from src.modeling import CLASSES, RANDOM_STATE, cv_splitter  # noqa: E402

OUT = Path(__file__).with_name("_repro2_l037_results.json")
N_JOBS = 6


def lgbm(seed=RANDOM_STATE, **extra) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
        **extra,
    )


def load():
    d = HOMEWORK / "data"
    df = build_dataset(
        pd.read_parquet(d / "persons.parquet"),
        pd.read_parquet(d / "situations.parquet"),
        pd.read_parquet(d / "responses.parquet"),
    )
    df_feat, y, mask = split_xy(df)
    ct = make_feature_pipeline()
    X64 = np.asarray(ct.fit_transform(df_feat), dtype=np.float64)
    X = X64[mask].astype(np.float32)
    groups = df.loc[mask, "person_id"].to_numpy()
    y_idx = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return X, X64[mask], y_idx, groups


def run_cv(X, y_idx, groups, *, seed=RANDOM_STATE, permute=False, label="", **extra):
    oof = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float64)
    per_fold = []
    t0 = time.time()
    rng = np.random.default_rng(12345)
    for i, (tr, te) in enumerate(cv_splitter().split(X, y_idx, groups=groups)):
        tr_used = rng.permutation(tr) if permute else tr
        model = lgbm(seed=seed, **extra).fit(X[tr_used], y_idx[tr_used])
        probs = model.predict_proba(X[te])
        oof[te] = probs
        per_fold.append(float(log_loss(y_idx[te], probs, labels=list(range(len(CLASSES)))))) 
        print(f"  [{label}] fold {i}: ll={per_fold[-1]:.6f}", flush=True)
    return {
        "oof": oof,
        "per_fold_log_loss": per_fold,
        "mean_log_loss": float(np.mean(per_fold)),
        "std_log_loss": float(np.std(per_fold, ddof=1)),
        "sha256": hashlib.sha256(oof.tobytes()).hexdigest()[:16],
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("loading...", flush=True)
    X32, X64, y_idx, groups = load()

    configs = [
        ("ref", dict(), X32),
        ("rowwise", dict(force_row_wise=True), X32),
        ("colwise", dict(force_col_wise=True), X32),
        ("perm", dict(permute=True), X32),
        ("f64", dict(), X64),
        ("seed1", dict(seed=1), X32),
        ("seed2", dict(seed=2), X32),
        ("seed3", dict(seed=3), X32),
        ("seed4", dict(seed=4), X32),
    ]

    oofs = {}
    for label, kw, Xin in configs:
        print(f"\n=== {label} {kw} ===", flush=True)
        r = run_cv(Xin, y_idx, groups, label=label, **kw)
        oofs[label] = r.pop("oof")
        r["config"] = {k: str(v) for k, v in kw.items()}
        results[label] = r
        OUT.write_text(json.dumps(results, indent=2, default=str))
        print(f"  -> mean ll {r['mean_log_loss']:.6f} | sha {r['sha256']} "
              f"({r['seconds']}s)", flush=True)

    ref = oofs["ref"]
    for label in oofs:
        d = np.abs(oofs[label] - ref)
        results[label]["vs_ref"] = {
            "bitwise_identical": bool(np.array_equal(oofs[label], ref)),
            "max_abs_dp": float(d.max()),
            "mean_abs_dp": float(d.mean()),
            "argmax_flips": int((oofs[label].argmax(axis=1) != ref.argmax(axis=1)).sum()),
            "d_mean_log_loss": float(
                results[label]["mean_log_loss"] - results["ref"]["mean_log_loss"]
            ),
        }

    seeds = [results[k]["mean_log_loss"] for k in ("ref", "seed1", "seed2", "seed3", "seed4")]
    results["_seed_spread"] = {
        "mean_log_loss_by_seed": seeds,
        "min": float(min(seeds)),
        "max": float(max(seeds)),
        "range": float(max(seeds) - min(seeds)),
        "std": float(np.std(seeds, ddof=1)),
        "fold_std_of_ref": results["ref"]["std_log_loss"],
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print("\n" + json.dumps(results["_seed_spread"], indent=2))
    print("wrote", OUT)
