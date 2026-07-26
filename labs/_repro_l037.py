"""L037 reproducibility harness — what moves the number when nothing "changed".

Runs the homework pipeline's LightGBM arm (plain, uncalibrated — the calibrator
has its own inner CV and would confound the measurement) over the same
person-grouped 5-fold CV under configurations that differ ONLY in knobs a
reader would call irrelevant:

  ref        n_jobs=6, deterministic=False   (the L036 audit's working config)
  repeat     byte-identical rerun of ref      -> rung 1: repeatability
  t1/t2/t12  n_jobs varied, same seed         -> does thread count move the model?
  det_*      deterministic=True + force_row_wise=True at the same thread counts

For each configuration we keep the full out-of-fold probability matrix
(5,587 x 5), hash it, and compare to `ref`: bitwise identical, max |dp|, and
delta mean log-loss. The claim under test is the one every "seeded, therefore
reproducible" README makes.

Also computes the estimator-of-record contrast from the submission's own saved
artifacts: pooled-OOF ECE (README headline) vs mean-over-folds ECE (report.md
§3.1) — the same word, two numbers.

Usage:  python _repro_l037.py [probe|sweep|artifacts|all]
Writes labs/_repro_l037_results.json.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HOMEWORK = Path("/home/avist/Projects/homework")
sys.path.insert(0, str(HOMEWORK))

from sklearn.metrics import log_loss  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402
import lightgbm  # noqa: E402
import sklearn  # noqa: E402

from src.features import build_dataset, make_feature_pipeline, split_xy  # noqa: E402
from src.modeling import CLASSES, RANDOM_STATE, _ece, cv_splitter  # noqa: E402

OUT = Path(__file__).with_name("_repro_l037_results.json")


def lgbm(n_jobs: int, deterministic: bool) -> LGBMClassifier:
    """src.modeling._lgbm() hyperparameters, with only the two knobs varied."""
    kw = dict(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=-1,
    )
    if deterministic:
        kw["deterministic"] = True
        kw["force_row_wise"] = True
    return LGBMClassifier(**kw)


def load():
    d = HOMEWORK / "data"
    df = build_dataset(
        pd.read_parquet(d / "persons.parquet"),
        pd.read_parquet(d / "situations.parquet"),
        pd.read_parquet(d / "responses.parquet"),
    )
    df_feat, y, mask = split_xy(df)
    ct = make_feature_pipeline()
    X_all = ct.fit_transform(df_feat).astype(np.float32)
    X = X_all[mask]
    groups = df.loc[mask, "person_id"].to_numpy()
    y_idx = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return X, y_idx, groups


def run_cv(X, y_idx, groups, *, n_jobs: int, deterministic: bool, label: str) -> dict:
    """One person-grouped 5-fold CV; returns OOF probs + per-fold log-loss."""
    oof = np.zeros((len(y_idx), len(CLASSES)), dtype=np.float64)
    per_fold = []
    t0 = time.time()
    for i, (tr, te) in enumerate(cv_splitter().split(X, y_idx, groups=groups)):
        model = lgbm(n_jobs, deterministic).fit(X[tr], y_idx[tr])
        probs = model.predict_proba(X[te])
        oof[te] = probs
        per_fold.append(float(log_loss(y_idx[te], probs, labels=list(range(len(CLASSES))))))
        print(f"  [{label}] fold {i}: ll={per_fold[-1]:.6f}", flush=True)
    return {
        "oof": oof,
        "per_fold_log_loss": per_fold,
        "mean_log_loss": float(np.mean(per_fold)),
        "std_log_loss": float(np.std(per_fold, ddof=1)),
        "pooled_log_loss": float(log_loss(y_idx, oof, labels=list(range(len(CLASSES))))),
        "ece_pooled": float(_ece(oof, y_idx)),
        "sha256": hashlib.sha256(oof.tobytes()).hexdigest()[:16],
        "seconds": round(time.time() - t0, 1),
    }


def artifacts_contrast() -> dict:
    """Pooled-OOF vs mean-over-folds, on the submission's own saved artifacts."""
    out = {}
    for tag, npz, csv in [
        ("M2a", "oof_M2a_lgbm_km.npz", "cv_folds_M2a_lgbm_km.csv"),
        ("M1", "oof_M1_lgbm_iso.npz", "cv_folds_M1_lgbm_iso.csv"),
    ]:
        p = HOMEWORK / "artifacts" / npz
        c = HOMEWORK / "artifacts" / csv
        if not p.exists() or not c.exists():
            continue
        z = np.load(p, allow_pickle=True)
        keys = list(z.keys())
        probs = z["probs"] if "probs" in keys else z[keys[0]]
        y_true = None
        for k in ("y_true", "y", "y_idx", "labels"):
            if k in keys:
                y_true = z[k]
                break
        rec = {"npz_keys": keys, "shape": list(probs.shape)}
        if y_true is not None:
            if y_true.dtype.kind in "US" or y_true.dtype == object:
                y_idx = pd.Series(y_true).map({c_: i for i, c_ in enumerate(CLASSES)}).to_numpy()
            else:
                y_idx = np.asarray(y_true, dtype=int)
            rec["ece_pooled"] = float(_ece(np.asarray(probs, dtype=np.float64), y_idx))
            rec["log_loss_pooled"] = float(
                log_loss(y_idx, np.asarray(probs, dtype=np.float64),
                         labels=list(range(len(CLASSES))))
            )
        folds = pd.read_csv(c)
        rec["fold_csv_columns"] = list(folds.columns)
        for col in folds.columns:
            if folds[col].dtype.kind in "fi" and col not in ("fold",):
                rec[f"mean_{col}"] = float(folds[col].mean())
                rec[f"std_{col}"] = float(folds[col].std(ddof=1))
        out[tag] = rec
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["_env"] = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.machine(),
    }

    if stage in ("artifacts", "all"):
        results["_artifacts"] = artifacts_contrast()
        OUT.write_text(json.dumps(results, indent=2, default=str))
        print(json.dumps(results["_artifacts"], indent=2, default=str))

    if stage in ("probe", "sweep", "all"):
        print("loading...", flush=True)
        X, y_idx, groups = load()
        print("  X", X.shape, flush=True)
        results["_meta"] = {"n_labelled": int(X.shape[0]), "n_features": int(X.shape[1])}

    if stage == "probe":
        for nj in (1, 6, 12):
            t0 = time.time()
            lgbm(nj, False).fit(X[:4000], y_idx[:4000])
            print(f"n_jobs={nj}: {time.time() - t0:.1f}s per fit", flush=True)
        sys.exit(0)

    if stage in ("sweep", "all"):
        configs = [
            ("ref", 6, False),
            ("repeat", 6, False),
            ("t1", 1, False),
            ("t2", 2, False),
            ("t12", 12, False),
            ("det_t1", 1, True),
            ("det_t2", 2, True),
            ("det_t6", 6, True),
            ("det_t12", 12, True),
        ]
        oofs = {}
        for label, nj, det in configs:
            key = f"cfg_{label}"
            print(f"\n=== {label} (n_jobs={nj}, deterministic={det}) ===", flush=True)
            r = run_cv(X, y_idx, groups, n_jobs=nj, deterministic=det, label=label)
            oofs[label] = r.pop("oof")
            r["n_jobs"] = nj
            r["deterministic"] = det
            results[key] = r
            OUT.write_text(json.dumps(results, indent=2, default=str))
            print(f"  -> mean ll {r['mean_log_loss']:.6f} | sha {r['sha256']} "
                  f"({r['seconds']}s)", flush=True)

        ref = oofs["ref"]
        for label in oofs:
            d = np.abs(oofs[label] - ref)
            results[f"cfg_{label}"]["vs_ref"] = {
                "bitwise_identical": bool(np.array_equal(oofs[label], ref)),
                "max_abs_dp": float(d.max()),
                "mean_abs_dp": float(d.mean()),
                "rows_changed": int((d.max(axis=1) > 0).sum()),
                "argmax_flips": int(
                    (oofs[label].argmax(axis=1) != ref.argmax(axis=1)).sum()
                ),
                "d_mean_log_loss": float(
                    results[f"cfg_{label}"]["mean_log_loss"]
                    - results["cfg_ref"]["mean_log_loss"]
                ),
            }
        np.savez_compressed(Path(__file__).with_name("_repro_l037_oof.npz"), **oofs)
        OUT.write_text(json.dumps(results, indent=2, default=str))

    print("\nwrote", OUT)
