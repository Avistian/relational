"""L043 paper-results scale-up (NOTES standard #25).

The learning lab (`0043-tabnet.ipynb`) implements sequential attention and races TabNet on four
*small* OpenML tables. That bake-off is a different experiment from Arik & Pfister 2019. This
harness tries to get closer to the paper so the student can learn the *right* conclusion.

Targets
-------
* Adult Census (paper appendix): TabNet reports **85.7% test accuracy** with
  N_d=N_a=16, N_steps=5, γ=1.5, λ_sparse=1e-4, B=4096. We train the *from-scratch*
  encoder under those HPs. The UCI official test split is **not** what OpenML 1590
  gives us, so the absolute number is INCOMPARABLE even if we land near 0.857 —
  that is the honest verdict, not a cop-out.
* Syn4 mask sharpness (paper Fig. 5): the paper used **10M** samples for the figure.
  The lab used 10k and recovered the instance-wise switch only partially (15.6%).
  Scaling n tests whether sharpness is a data-volume effect (the paper's own caveat).

Presets: smoke (seconds) · closer (T4, ~15–30 min) · paper (hours).

Run:
    OMP_NUM_THREADS=1 python labs/_paper_repro_l043.py --preset smoke
    ~/.local/bin/modal run --detach modal/l043_paper_repro.py --preset closer
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

from relkit import load_tier_a  # noqa: E402
from relkit.paper_repro import (  # noqa: E402
    LabFinding, PaperTarget, ScaleUpRun,
    classify_direction, classify_number, device, format_ledger, hardware_tag,
    print_howto, to_jsonable,
)
from relkit.synth import make_syn4  # noqa: E402
from relkit.tabnet import TabNetEncoder, explain, tabnet_auc, train_tabnet  # noqa: E402

ADULT_TARGET = PaperTarget(
    paper="TabNet (Arik & Pfister 2019)", arxiv="1908.07442",
    table="Appendix — Adult Census Income",
    dataset="adult", metric="accuracy", paper_value=0.857, abs_tol=0.01,
    paper_split="UCI Adult official test file (16,281 rows) — we do NOT have this split on OpenML 1590",
    notes="Paper HPs: N_d=N_a=16, N_steps=5, γ=1.5, λ_sparse=1e-4, B=4096, Adam lr=0.02.",
)
SYN4_TARGET = PaperTarget(
    paper="TabNet (Arik & Pfister 2019)", arxiv="1908.07442",
    table="Fig. 5 (masks; trained at 10M samples)",
    dataset="Syn4", metric="left_group_recovery", paper_value=None, abs_tol=0.05,
    notes="No numeric target in the paper — Fig. 5 is qualitative. We track whether "
          "left_correct climbs vs the lab's 0.156 at n=10k.",
)

LAB_FINDINGS = [
    LabFinding("TabNet vs MLP/ResNet/GBDT mean ranks",
               "TabNet 2.50 · MLP 1.75 · ResNet 2.00 · GBDT 3.75 (Friedman p=0.127)",
               "4 small OpenML tables (credit_g/diabetes/blood_transfusion/kc1), budget 6, 3 seeds, CPU"),
    LabFinding("Syn4 instance-wise mask recovery (X11<0 rows favouring X1–X2)",
               "15.6% (PARTIAL) at n=10k",
               "Chen et al. 2018 generator, n=10k, 5 steps — paper Fig. 5 used 10M"),
]


def _dense(name):
    Xdf, y = load_tier_a(name)
    num = Xdf.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xdf.columns if c not in num]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    return ct.fit_transform(Xdf).astype(np.float32), y.to_numpy().astype(np.float32)


def _split(X, y, seed, test_size=0.30):
    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25, random_state=seed, stratify=ytr_f)
    return Xtr, ytr, Xva, yva, Xte, yte


def _tabnet_acc_auc(model, X, y, *, dev):
    import torch
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor(np.asarray(X), dtype=torch.float32, device=dev))
        logits = logits.detach().cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    return float(accuracy_score(y, (p >= 0.5).astype(int))), float(roc_auc_score(y, p))


def run_adult(*, epochs, seed, dev, n_d=16, n_steps=5, batch_size=1024):
    X, y = _dense("adult")
    Xtr, ytr, Xva, yva, Xte, yte = _split(X, y, seed)
    model = TabNetEncoder(X.shape[1], n_d=n_d, n_a=n_d, n_steps=n_steps,
                          gamma=1.5, lambda_sparse=1e-4, virtual_batch_size=128, momentum=0.02)
    t0 = time.time()
    model, _ = train_tabnet(model, Xtr, ytr, Xva, yva, lr=0.02, max_epochs=epochs,
                            patience=max(8, epochs // 5), batch_size=batch_size,
                            device=dev, seed=seed)
    acc, auc = _tabnet_acc_auc(model, Xte, yte, dev=dev)
    return acc, auc, time.time() - t0, len(y)


def run_xgb_adult(*, seed):
    from xgboost import XGBClassifier
    X, y = _dense("adult")
    Xtr, ytr, Xva, yva, Xte, yte = _split(X, y, seed)
    t0 = time.time()
    m = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, n_jobs=4, eval_metric="logloss", random_state=seed,
    )
    m.fit(Xtr, ytr)
    acc = float(accuracy_score(yte, m.predict(Xte)))
    auc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return acc, auc, time.time() - t0


def run_syn4(*, n, epochs, seed, dev):
    X, y, _ = make_syn4(n=n, seed=seed)
    Xtr, ytr, Xva, yva, Xte, yte = _split(X, y, seed)
    model = TabNetEncoder(X.shape[1], n_d=8, n_a=8, n_steps=5, gamma=1.5, lambda_sparse=5e-3)
    t0 = time.time()
    model, _ = train_tabnet(model, Xtr, ytr, Xva, yva, lr=0.02, max_epochs=epochs,
                            patience=max(8, epochs // 4), batch_size=1024,
                            device=dev, seed=seed)
    auc = float(tabnet_auc(model, Xte, yte, device=dev))
    M, _ = explain(model, Xte, device=dev)
    left = Xte[:, 10] < 0
    left_correct = float((M[left][:, [0, 1]].sum(1) > M[left][:, 2:6].sum(1)).mean()) if left.any() else 0.0
    return auc, left_correct, time.time() - t0


def preset_cfg(name):
    if name == "smoke":
        return dict(adult_epochs=2, syn4_n=800, syn4_epochs=3, n_d=8, n_steps=3,
                    batch_size=256, run_xgb=False)
    if name == "closer":
        return dict(adult_epochs=40, syn4_n=50_000, syn4_epochs=30, n_d=16, n_steps=5,
                    batch_size=2048, run_xgb=True)
    if name == "paper":
        return dict(adult_epochs=80, syn4_n=200_000, syn4_epochs=60, n_d=16, n_steps=5,
                    batch_size=4096, run_xgb=True)
    raise ValueError(f"unknown preset {name!r}; use smoke|closer|paper")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("smoke", "closer", "paper"), default="closer")
    args = p.parse_args(argv)
    cfg = preset_cfg(args.preset)
    dev = device()
    hw = hardware_tag()
    print(f"L043 paper-repro  preset={args.preset}  device={hw}")

    deviations = [
        "OpenML 1590 random stratified 70/15/15 split — NOT the UCI official Adult test file",
        "early-stopping on ROC-AUC (paper reports accuracy; paper used a step LR decay over 7.7k iters)",
        f"batch_size={cfg['batch_size']} (paper B=4096, B_V=128 ghost BN); from-scratch encoder, binary logit",
    ]

    adult_run = None
    xgb_acc = None
    try:
        acc, auc, wall, n = run_adult(epochs=cfg["adult_epochs"], seed=0, dev=dev,
                                      n_d=cfg["n_d"], n_steps=cfg["n_steps"],
                                      batch_size=cfg["batch_size"])
        adult_run = ScaleUpRun(
            method="tabnet-scratch", dataset="adult", metric="accuracy",
            value=acc, n_seeds=1, hardware=hw, wall_s=wall,
            protocol_match=False, protocol_deviations=deviations + [f"n={n} rows, test AUC={auc:.4f}"],
        )
        print(f"  TabNet Adult  acc={acc:.4f}  auc={auc:.4f}  wall={wall:.0f}s")
        if cfg["run_xgb"]:
            xgb_acc, xgb_auc, xgb_wall = run_xgb_adult(seed=0)
            print(f"  XGB    Adult  acc={xgb_acc:.4f}  auc={xgb_auc:.4f}  wall={xgb_wall:.0f}s")
    except Exception as exc:
        print(f"  Adult run failed: {exc}")

    syn_run = None
    syn_left = None
    try:
        auc, left, wall = run_syn4(n=cfg["syn4_n"], epochs=cfg["syn4_epochs"], seed=1, dev=dev)
        syn_left = left
        syn_run = ScaleUpRun(
            method="tabnet-scratch", dataset="Syn4", metric="left_group_recovery",
            value=left, n_seeds=1, hardware=hw, wall_s=wall, protocol_match=False,
            protocol_deviations=[
                f"n={cfg['syn4_n']} (paper Fig. 5 used 10M)",
                f"test AUC={auc:.4f}; lab left_correct at n=10k was 0.156",
            ],
        )
        print(f"  Syn4 n={cfg['syn4_n']}  left_correct={left:.3f}  auc={auc:.4f}  wall={wall:.0f}s")
    except Exception as exc:
        print(f"  Syn4 run failed: {exc}")

    extra = []
    if adult_run is not None and xgb_acc is not None:
        d = classify_direction(adult_run.value, xgb_acc, paper_a_beats_b=True,
                               higher_is_better=True, tie_tol=0.005)
        extra.append(f"DIRECTION TabNet vs XGB accuracy on this Adult split: {d} "
                     f"(paper appendix reports TabNet 85.7% but no XGB number on Adult; "
                     f"this tests 'does TabNet at least match a strong tree on the paper's table').")
    if syn_left is not None:
        extra.append(
            f"Syn4 left_correct {syn_left:.3f} vs lab 0.156. If it climbed, Fig. 5's 'needs 10M' "
            f"caveat is the right reading of the lab's PARTIAL recovery — not 'masks don't work'."
        )

    rows = [
        (ADULT_TARGET, adult_run, classify_number(ADULT_TARGET, adult_run)),
        (SYN4_TARGET, syn_run, classify_number(SYN4_TARGET, syn_run)),
    ]
    text = format_ledger(title="L043 TabNet", lab=LAB_FINDINGS, paper=rows, extra_lines=extra)
    print()
    print(text)

    out = {
        "lesson": 43, "preset": args.preset, "hardware": hw,
        "adult": to_jsonable(adult_run), "syn4": to_jsonable(syn_run),
        "xgb_adult_acc": xgb_acc, "ledger": text,
    }
    dest = os.path.join(HERE, f"_paper_repro_l043_{args.preset}_results.json")
    artifacts = os.environ.get("PAPER_REPRO_OUT")
    if artifacts:
        dest = artifacts
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return out


if __name__ == "__main__":
    main()
