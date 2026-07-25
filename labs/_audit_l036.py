"""L036 audit harness — re-measure the homework pipeline under fold-honest variants.

Runs the user's own Task-2 CV (StratifiedGroupKFold(5) by person_id, LightGBM +
isotonic) four ways so the lesson can quote real before/after numbers:

  A_baseline      as shipped: ColumnTransformer + KMeans fit once on all 119,498
                  rows; CalibratedClassifierCV(cv=5) ungrouped inner split.
  B_grouped_calib fix D1 only: inner calibration split is person-grouped.
  C_infold_prep   fix D2 only: ColumnTransformer refit inside each outer fold.
  D_both          D1 + D2.

Plus the M2a question (D3): does the shipped winner still beat M1 when the
KMeans SSL block is refit inside each outer fold?

Usage: python _audit_l036.py [stage1|stage2|all]
Writes labs/_audit_l036_results.json.
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

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from lightgbm import LGBMClassifier  # noqa: E402

from src.features import build_dataset, make_feature_pipeline, split_xy  # noqa: E402
from src.modeling import (  # noqa: E402
    CLASSES,
    RANDOM_STATE,
    _classwise_ece,
    _ece,
    _multiclass_brier,
    cv_splitter,
)

OUT = Path(__file__).with_name("_audit_l036_results.json")

# Identical to src.modeling._lgbm() except n_jobs: on this 12-core WSL2 box
# n_jobs=-1 costs 210 s per multiclass fit vs 9.2 s at n_jobs=6 (thread
# thrashing on 5.6k rows). Hyperparameters are untouched, so the comparison
# between variants below is unaffected.
N_JOBS = 6


def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=-1,
    )


def load() -> pd.DataFrame:
    d = HOMEWORK / "data"
    return build_dataset(
        pd.read_parquet(d / "persons.parquet"),
        pd.read_parquet(d / "situations.parquet"),
        pd.read_parquet(d / "responses.parquet"),
    )


def fold_metrics(probs: np.ndarray, y_idx: np.ndarray) -> dict:
    k = probs.shape[1]
    return {
        "log_loss": float(log_loss(y_idx, probs, labels=list(range(k)))),
        "brier": _multiclass_brier(probs, y_idx, k),
        "ece_top": _ece(probs, y_idx),
        "ece_classwise": _classwise_ece(probs, y_idx),
        "auc_macro_ovr": float(
            roc_auc_score(y_idx, probs, multi_class="ovr", average="macro",
                          labels=list(range(k)))
        ),
        "accuracy": float((probs.argmax(axis=1) == y_idx).mean()),
    }


def summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {c: [float(df[c].mean()), float(df[c].std(ddof=1))] for c in df.columns}


def run_variant(
    X_all_full: np.ndarray,
    X_all: np.ndarray,
    df_feat: pd.DataFrame,
    mask: np.ndarray,
    y: pd.Series,
    groups: np.ndarray,
    *,
    grouped_inner_calib: bool,
    infold_prep: bool,
    kmeans_mode: str = "none",  # none | global | infold
    label: str = "",
) -> dict:
    """One 5-fold person-grouped CV of LightGBM+isotonic under the given regime."""
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_idx = y.map(class_to_idx).to_numpy()
    rows = []
    t0 = time.time()

    # Global (as-shipped) unsupervised blocks, fit once on all 119,498 rows.
    km_global = None
    if kmeans_mode == "global":
        km_global = KMeans(n_clusters=16, random_state=RANDOM_STATE, n_init=10).fit(X_all)

    for fold_i, (tr, te) in enumerate(cv_splitter().split(X_all, y_idx, groups=groups)):
        assert not (set(groups[tr]) & set(groups[te])), "person_id leak in outer fold"

        if infold_prep:
            # Fit the encoder on training-fold persons only: every row of the
            # 119k whose person is not in the outer test fold.
            test_persons = set(groups[te])
            all_persons = df_feat["_person_id"].to_numpy()
            fit_rows = ~np.isin(all_persons, list(test_persons))
            ct = make_feature_pipeline()
            ct.fit(df_feat.loc[fit_rows])
            Xf_all = ct.transform(df_feat).astype(np.float32)
            Xf = Xf_all[mask]
        else:
            Xf_all, Xf = X_all_full, X_all

        if kmeans_mode == "global":
            km = km_global
        elif kmeans_mode == "infold":
            test_persons = set(groups[te])
            all_persons = df_feat["_person_id"].to_numpy()
            fit_rows = ~np.isin(all_persons, list(test_persons))
            km = KMeans(n_clusters=16, random_state=RANDOM_STATE, n_init=10).fit(
                Xf_all[fit_rows]
            )
        else:
            km = None

        if km is not None:
            oh = np.eye(16, dtype=np.float32)[km.predict(Xf)]
            dist = km.transform(Xf).astype(np.float32)
            Xm = np.concatenate([np.asarray(Xf, dtype=np.float32), oh, dist], axis=1)
        else:
            Xm = Xf

        if grouped_inner_calib:
            inner = list(
                StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                .split(Xm[tr], y_idx[tr], groups=groups[tr])
            )
            model = CalibratedClassifierCV(estimator=_lgbm(), method="isotonic", cv=inner)
        else:
            model = CalibratedClassifierCV(estimator=_lgbm(), method="isotonic", cv=5)

        model.fit(Xm[tr], y.iloc[tr])
        probs = model.predict_proba(Xm[te])
        order = [list(model.classes_).index(c) for c in CLASSES]
        rows.append(fold_metrics(probs[:, order], y_idx[te]))
        print(f"  [{label}] fold {fold_i}: ll={rows[-1]['log_loss']:.4f} "
              f"ece={rows[-1]['ece_top']:.4f}", flush=True)

    out = summarise(rows)
    out["_per_fold_log_loss"] = [r["log_loss"] for r in rows]
    out["_per_fold_ece_top"] = [r["ece_top"] for r in rows]
    out["_seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = json.loads(OUT.read_text()) if OUT.exists() else {}

    print("loading + joining...", flush=True)
    df = load()
    df_feat, y, mask = split_xy(df)
    df_feat = df_feat.copy()
    df_feat["_person_id"] = df["person_id"].to_numpy()
    groups = df.loc[mask, "person_id"].to_numpy()

    print("encoding (global fit on all 119,498 rows — as shipped)...", flush=True)
    ct = make_feature_pipeline()
    X_all_full = ct.fit_transform(df_feat).astype(np.float32)  # notebook cell 3
    X_all = X_all_full[mask]
    print("  X_all", X_all_full.shape, "labelled", X_all.shape, flush=True)
    results["_meta"] = {
        "n_rows_total": int(len(df)),
        "n_labelled": int(mask.sum()),
        "n_features": int(X_all_full.shape[1]),
        "n_persons_labelled": int(pd.Series(groups).nunique()),
    }

    def go(key, **kw):
        if key in results and "--force" not in sys.argv:
            print(f"skip {key} (cached)")
            return
        print(f"\n=== {key} ===", flush=True)
        results[key] = run_variant(X_all_full, X_all, df_feat, mask, y, groups,
                                   label=key, **kw)
        OUT.write_text(json.dumps(results, indent=2))
        s = results[key]
        print(f"  -> log_loss {s['log_loss'][0]:.4f} ± {s['log_loss'][1]:.4f} | "
              f"ece_top {s['ece_top'][0]:.4f} ± {s['ece_top'][1]:.4f} "
              f"({s['_seconds']}s)", flush=True)

    if stage in ("stage1", "all"):
        go("M1_A_baseline", grouped_inner_calib=False, infold_prep=False)
        go("M1_B_grouped_calib", grouped_inner_calib=True, infold_prep=False)

    if stage in ("stage2", "all"):
        go("M1_C_infold_prep", grouped_inner_calib=False, infold_prep=True)
        go("M1_D_both", grouped_inner_calib=True, infold_prep=True)

    if stage in ("stage3", "all"):
        go("M2a_global_km", grouped_inner_calib=False, infold_prep=False,
           kmeans_mode="global")
        go("M2a_infold_km", grouped_inner_calib=False, infold_prep=False,
           kmeans_mode="infold")

    OUT.write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT)
