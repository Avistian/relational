"""L044 paper-results scale-up (NOTES standard #25).

The learning lab races NODE vs CatBoost on four *small* tables with ~64–128 trees.
Popov, Morozov & Babenko 2020 (NODE) claim a *default-HP* win over CatBoost/XGBoost on
six large datasets with **2048 trees of depth 6**. That is a different experiment.

This harness trains the from-scratch NODE against CatBoost on a documented subsample of
the paper's Higgs table (OpenML 23512, ~98k of the paper's 10.5M) — or falls back to
Adult with an honest gap. The paper's Higgs default-HP gap is ~0.002 error (NODE 0.2412
vs CatBoost 0.2434). A 98k subsample will often be a DIRECTION_TIE; that is a lesson
about *statistical power*, not a refutation of Table 1.

Presets: smoke · closer · paper.

Run:
    OMP_NUM_THREADS=1 python labs/_paper_repro_l044.py --preset smoke
    ~/.local/bin/modal run --detach modal/l044_paper_repro.py --preset closer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)

from relkit.paper_repro import (  # noqa: E402
    LabFinding, PaperTarget, ScaleUpRun,
    classify_direction, classify_number, device, format_ledger, hardware_tag,
    print_howto, to_jsonable,
)
from relkit.node import DenseNODE, node_auc, train_node  # noqa: E402

HIGGS_TARGET = PaperTarget(
    paper="NODE (Popov, Morozov & Babenko 2020)", arxiv="1909.06312",
    table="Table 1 — Higgs, default HPs (classification error)",
    dataset="Higgs", metric="error", paper_value=0.2412, paper_std=0.0005, abs_tol=0.005,
    paper_split="paper: 10.5M train / 500k test; we use OpenML 23512 (~98k) or Adult fallback",
    higher_is_better=False,
    notes="Default NODE = 1 layer × 2048 trees × depth 6. CatBoost Table 1 Higgs error = 0.2434.",
)

LAB_FINDINGS = [
    LabFinding("NODE vs CatBoost/MLP/ResNet mean ranks",
               "NODE 3.50 · CatBoost 2.50 · MLP 2.00 · ResNet 2.00 (Friedman p=0.308); "
               "NODE beats CatBoost 1/4; ~70× slower on credit_g",
               "4 small OpenML tables, 64–128 trees, 3 seeds, CPU — NOT the paper's 40+/6-dataset study"),
]


def _dense_from_xy(Xdf, y):
    num = Xdf.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xdf.columns if c not in num]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    return ct.fit_transform(Xdf).astype(np.float32), y.to_numpy().astype(np.float32)


def load_table():
    """Prefer the paper's Higgs (subsampled); fall back to Adult with a gap note."""
    from relkit import load_tier_a
    try:
        Xdf, y = load_tier_a("higgs_small")
        return _dense_from_xy(Xdf, y), "higgs_small", [
            "OpenML 23512 (~98k rows) is a SUBSAMPLE of UCI Higgs; paper used 10.5M / 500k test",
        ]
    except Exception as exc:
        Xdf, y = load_tier_a("adult")
        return _dense_from_xy(Xdf, y), "adult", [
            f"higgs_small failed ({exc}); Adult is NOT a NODE paper dataset — Table 1 number is INCOMPARABLE",
        ]


def _split(X, y, seed):
    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25, random_state=seed, stratify=ytr_f)
    return Xtr, ytr, Xva, yva, Xte, yte


def _error_auc(predict_proba_pos, y):
    pred = (predict_proba_pos >= 0.5).astype(int)
    err = 1.0 - float(accuracy_score(y, pred))
    auc = float(roc_auc_score(y, predict_proba_pos))
    return err, auc


def run_node(Xtr, ytr, Xva, yva, Xte, yte, *, trees, depth, epochs, seed, dev):
    import torch
    model = DenseNODE(Xtr.shape[1], num_trees=trees, depth=depth, n_layers=1)
    t0 = time.time()
    model, _ = train_node(model, Xtr, ytr, Xva, yva, lr=1e-2, max_epochs=epochs,
                          patience=max(6, epochs // 4), batch_size=512, device=dev, seed=seed)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(Xte), dtype=torch.float32, device=dev)).cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    err, auc = _error_auc(p, yte)
    return err, auc, time.time() - t0


def run_catboost(Xtr, ytr, Xva, yva, Xte, yte, *, trees, seed):
    from catboost import CatBoostClassifier
    t0 = time.time()
    m = CatBoostClassifier(iterations=trees, depth=6, learning_rate=0.1, verbose=0,
                           random_seed=seed, od_type="Iter", od_wait=30)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    p = m.predict_proba(Xte)[:, 1]
    err, auc = _error_auc(p, yte)
    return err, auc, time.time() - t0


def preset_cfg(name):
    if name == "smoke":
        return dict(trees=8, depth=3, epochs=2, subsample=800)
    if name == "closer":
        return dict(trees=256, depth=6, epochs=25, subsample=None)
    if name == "paper":
        return dict(trees=2048, depth=6, epochs=40, subsample=None)
    raise ValueError(f"unknown preset {name!r}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("smoke", "closer", "paper"), default="closer")
    args = p.parse_args(argv)
    cfg = preset_cfg(args.preset)
    dev = device()
    hw = hardware_tag()
    print(f"L044 paper-repro  preset={args.preset}  device={hw}")

    (X, y), dname, gaps = load_table()
    if cfg["subsample"] and len(y) > cfg["subsample"]:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=cfg["subsample"], replace=False)
        X, y = X[idx], y[idx]
        gaps = list(gaps) + [f"smoke subsampled to {cfg['subsample']} rows"]
    Xtr, ytr, Xva, yva, Xte, yte = _split(X, y, 0)
    print(f"  table={dname}  n={len(y)}  d={X.shape[1]}")

    node_run = None
    cb_err = None
    try:
        n_err, n_auc, n_wall = run_node(Xtr, ytr, Xva, yva, Xte, yte,
                                        trees=cfg["trees"], depth=cfg["depth"],
                                        epochs=cfg["epochs"], seed=0, dev=dev)
        node_run = ScaleUpRun(
            method="node-scratch", dataset=dname, metric="error",
            value=n_err, n_seeds=1, hardware=hw, wall_s=n_wall,
            protocol_match=False,
            protocol_deviations=gaps + [
                f"trees={cfg['trees']} depth={cfg['depth']} (paper default 2048 × depth 6)",
                f"test AUC={n_auc:.4f}",
            ],
        )
        print(f"  NODE     error={n_err:.4f}  auc={n_auc:.4f}  wall={n_wall:.0f}s")
        cb_err, cb_auc, cb_wall = run_catboost(Xtr, ytr, Xva, yva, Xte, yte,
                                               trees=cfg["trees"], seed=0)
        print(f"  CatBoost error={cb_err:.4f}  auc={cb_auc:.4f}  wall={cb_wall:.0f}s")
    except Exception as exc:
        print(f"  bake-off failed: {exc}")

    extra = []
    if node_run is not None and cb_err is not None:
        d = classify_direction(node_run.value, cb_err, paper_a_beats_b=True,
                               higher_is_better=False, tie_tol=0.005)
        extra.append(
            f"DIRECTION NODE vs CatBoost error on {dname}: {d}. "
            f"Paper Table 1 (default HPs, full Higgs) NODE 0.2412 vs CatBoost 0.2434 — a 0.002 edge. "
            f"A DIRECTION_TIE on this subsample means the gap is smaller than this experiment can see, "
            f"not that Table 1 is false. DIRECTION_FAIL would be a real tension with the paper's claim."
        )

    rows = [(HIGGS_TARGET, node_run, classify_number(HIGGS_TARGET, node_run))]
    text = format_ledger(title="L044 NODE", lab=LAB_FINDINGS, paper=rows, extra_lines=extra)
    print()
    print(text)

    out = {
        "lesson": 44, "preset": args.preset, "hardware": hw, "table": dname,
        "node": to_jsonable(node_run), "catboost_error": cb_err, "ledger": text,
    }
    dest = os.environ.get("PAPER_REPRO_OUT") or os.path.join(
        HERE, f"_paper_repro_l044_{args.preset}_results.json"
    )
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return out


if __name__ == "__main__":
    main()
