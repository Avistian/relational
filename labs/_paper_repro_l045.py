"""L045 paper-results scale-up (NOTES standard #25).

The learning lab trains TabTransformer on credit_g + *subsampled* Adult (4k) + churn.
Huang et al. 2020 report ~+1.0% AUC over deep baselines across 15 datasets, *matching*
(not beating) GBDTs, and a larger semi-supervised lift (~+2.1%) with more unlabeled data.

This harness re-runs the same from-scratch model on **full Adult** (a paper dataset;
OpenML 1590 ≈ Census Income) against the n_layers=0 ablation and CatBoost, plus an RTD
pre-train at 3% labels with the full unlabeled pool. Absolute Table numbers stay
INCOMPARABLE (we are not the 15-dataset suite). The DIRECTION tests are the ones that
can actually update what you believe:

* contextual vs context-free (paper: contextual helps)
* TabTransformer vs CatBoost (paper: matches GBDTs, does not dominate)
* RTD 3%-label lift (paper: positive, larger than our lab's +0.008)

Presets: smoke · closer · paper.

Run:
    OMP_NUM_THREADS=1 python labs/_paper_repro_l045.py --preset smoke
    ~/.local/bin/modal run --detach modal/l045_paper_repro.py --preset closer
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
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)

from relkit import load_tier_a  # noqa: E402
from relkit.paper_repro import (  # noqa: E402
    LabFinding, PaperTarget, ScaleUpRun,
    classify_direction, classify_number, device, format_ledger, hardware_tag,
    print_howto, to_jsonable,
)
from relkit.tabtransformer import (  # noqa: E402
    TabTransformer, frame_categorical, pretrain_rtd, tabtransformer_auc,
    train_tabtransformer,
)

CONTEXT_TARGET = PaperTarget(
    paper="TabTransformer (Huang et al. 2020)", arxiv="2012.06678",
    table="§4 — +1.0% AUC over deep baselines (15 datasets)",
    dataset="adult (full)", metric="auc_minus_context_free", paper_value=0.010, abs_tol=0.005,
    notes="Paper average over 15 datasets vs *deep* methods, not vs GBDT. We measure "
          "contextual − context-free on full Adult only.",
)
GBDT_TARGET = PaperTarget(
    paper="TabTransformer (Huang et al. 2020)", arxiv="2012.06678",
    table="§4 — matches GBDTs (does not dominate)",
    dataset="adult (full)", metric="auc_minus_catboost", paper_value=0.0, abs_tol=0.01,
    notes="Paper: TabTransformer matches GBDT. A large *loss* to CatBoost here agrees with "
          "that framing; a win would be a new result, not a reproduction.",
)
RTD_TARGET = PaperTarget(
    paper="TabTransformer (Huang et al. 2020)", arxiv="2012.06678",
    table="§4.3 semi-supervised — ~+2.1% at low label fractions",
    dataset="adult (full, 3% labels)", metric="auc_lift", paper_value=0.021, abs_tol=0.01,
    notes="Paper uses a larger unlabeled pool and longer pre-training than the lab.",
)

LAB_FINDINGS = [
    LabFinding("contextual vs context-free vs CatBoost mean ranks",
               "TT 2.33 · context-free 2.67 · CatBoost 1.00 (Friedman p=0.097); "
               "contextual beats context-free 2/3; TT beats CatBoost 0/3",
               "credit_g + Adult subsampled to 4000 + churn, 3 seeds, CPU"),
    LabFinding("RTD label-efficiency lift",
               "+0.008 AUC at 3% labels (adult 16k); +0.001 at 10%",
               "downscaled unlabeled pool; paper ~+2.1% at benchmark scale"),
]


def load_adult_frame(*, cap=None, seed=0):
    Xdf, y = load_tier_a("adult")
    if cap is not None and len(Xdf) > cap:
        idx, _ = train_test_split(np.arange(len(Xdf)), train_size=cap, random_state=seed, stratify=y)
        Xdf, y = Xdf.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(Xdf)
    yv = y.to_numpy().astype(np.float32)
    return dict(Xcat=Xcat, Xnum=Xnum, cards=cards, y=yv,
                cat_names=cat_names, num_names=num_names)


def split_idx(n, y, seed):
    tr, te = train_test_split(np.arange(n), test_size=0.30, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
    return tr, va, te


def run_tt(fr, tr, va, te, cfg, *, epochs, lr, seed, dev):
    m = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **cfg)
    t0 = time.time()
    m, _ = train_tabtransformer(
        m, fr["Xcat"][tr], fr["Xnum"][tr], fr["y"][tr],
        fr["Xcat"][va], fr["Xnum"][va], fr["y"][va],
        lr=lr, max_epochs=epochs, patience=max(6, epochs // 4),
        batch_size=256, device=dev, seed=seed,
    )
    auc = float(tabtransformer_auc(m, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te], device=dev))
    return m, auc, time.time() - t0


def run_catboost_frame(fr, tr, va, te, *, seed):
    from catboost import CatBoostClassifier
    import pandas as pd
    # CatBoost wants a dense table; concatenate scaled numerics + integer cats.
    X = np.concatenate([fr["Xnum"], fr["Xcat"].astype(np.float32)], axis=1)
    t0 = time.time()
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1, verbose=0, random_seed=seed)
    m.fit(X[tr], fr["y"][tr], eval_set=(X[va], fr["y"][va]), use_best_model=True)
    p = m.predict_proba(X[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(fr["y"][te], p)), time.time() - t0


def run_rtd(fr, *, label_frac, pretrain_epochs, ft_epochs, seed, dev):
    n = len(fr["y"])
    labeled, unlabeled = train_test_split(np.arange(n), train_size=label_frac,
                                          random_state=seed, stratify=fr["y"])
    tr, va = train_test_split(labeled, test_size=0.25, random_state=seed, stratify=fr["y"][labeled])
    _, te = train_test_split(np.arange(n), test_size=0.30, random_state=seed + 1, stratify=fr["y"])
    cfg = dict(d=32, n_layers=3, n_heads=4, head_hidden=128, dropout=0.1)
    # from scratch on the small labeled set
    _, scratch, _ = run_tt(fr, tr, va, te, cfg, epochs=ft_epochs, lr=1e-3, seed=seed, dev=dev)
    # pretrain on unlabeled cats, then gentle fine-tune
    m = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **cfg)
    pretrain_rtd(m, fr["Xcat"][unlabeled], fr["cards"],
                 replace_p=0.30, max_epochs=pretrain_epochs, batch_size=256,
                 device=dev, seed=seed)
    m, _ = train_tabtransformer(
        m, fr["Xcat"][tr], fr["Xnum"][tr], fr["y"][tr],
        fr["Xcat"][va], fr["Xnum"][va], fr["y"][va],
        lr=5e-4, max_epochs=ft_epochs, patience=max(6, ft_epochs // 4),
        batch_size=256, device=dev, seed=seed,
    )
    ft = float(tabtransformer_auc(m, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te], device=dev))
    return scratch, ft, ft - scratch


def preset_cfg(name):
    if name == "smoke":
        return dict(cap=700, epochs=2, rtd=False, pretrain_epochs=0, ft_epochs=2)
    if name == "closer":
        return dict(cap=None, epochs=30, rtd=True, pretrain_epochs=20, ft_epochs=20)
    if name == "paper":
        return dict(cap=None, epochs=60, rtd=True, pretrain_epochs=40, ft_epochs=30)
    raise ValueError(f"unknown preset {name!r}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("smoke", "closer", "paper"), default="closer")
    args = p.parse_args(argv)
    cfg = preset_cfg(args.preset)
    dev = device()
    hw = hardware_tag()
    print(f"L045 paper-repro  preset={args.preset}  device={hw}")

    fr = load_adult_frame(cap=cfg["cap"])
    tr, va, te = split_idx(len(fr["y"]), fr["y"], 0)
    print(f"  adult n={len(fr['y'])}  cats={len(fr['cards'])}  nums={fr['Xnum'].shape[1]}")

    ctx_cfg = dict(d=32, n_layers=3, n_heads=4, head_hidden=128, dropout=0.1)
    cf_cfg = dict(d=32, n_layers=0, head_hidden=128, dropout=0.1)
    ctx_run = cf_auc = cb_auc = None
    try:
        _, ctx_auc, ctx_wall = run_tt(fr, tr, va, te, ctx_cfg, epochs=cfg["epochs"],
                                      lr=1e-3, seed=0, dev=dev)
        _, cf_auc, _ = run_tt(fr, tr, va, te, cf_cfg, epochs=cfg["epochs"],
                              lr=1e-3, seed=0, dev=dev)
        cb_auc, cb_wall = run_catboost_frame(fr, tr, va, te, seed=0)
        ctx_run = ScaleUpRun(
            method="tabtransformer-scratch", dataset="adult", metric="auc",
            value=ctx_auc, n_seeds=1, hardware=hw, wall_s=ctx_wall, protocol_match=False,
            protocol_deviations=[
                "single dataset (Adult full), not the paper's 15-dataset mean",
                "OpenML 1590 random split, not a published TabTransformer split",
                f"context-free AUC={cf_auc:.4f}  CatBoost AUC={cb_auc:.4f}",
            ],
        )
        print(f"  contextual {ctx_auc:.4f}  context-free {cf_auc:.4f}  CatBoost {cb_auc:.4f}  "
              f"wall={ctx_wall:.0f}s+{cb_wall:.0f}s")
    except Exception as exc:
        print(f"  bake-off failed: {exc}")

    rtd_run = None
    if cfg["rtd"]:
        try:
            scratch, ft, lift = run_rtd(fr, label_frac=0.03, pretrain_epochs=cfg["pretrain_epochs"],
                                        ft_epochs=cfg["ft_epochs"], seed=0, dev=dev)
            rtd_run = ScaleUpRun(
                method="tabtransformer-rtd", dataset="adult-3pct", metric="auc_lift",
                value=lift, n_seeds=1, hardware=hw, wall_s=0.0, protocol_match=False,
                protocol_deviations=[
                    f"scratch={scratch:.4f}  pretrain+ft={ft:.4f}",
                    "3% labels on OpenML Adult; paper's ~+2.1% used a larger unlabeled regime",
                ],
            )
            print(f"  RTD 3%  scratch={scratch:.4f}  ft={ft:.4f}  lift={lift:+.4f}")
        except Exception as exc:
            print(f"  RTD run failed: {exc}")

    extra = []
    if ctx_run is not None and cf_auc is not None:
        extra.append(
            f"DIRECTION contextual vs context-free: "
            f"{classify_direction(ctx_run.value, cf_auc, paper_a_beats_b=True, tie_tol=0.003)} "
            f"(paper: contextual helps over deep / static embeddings)."
        )
    if ctx_run is not None and cb_auc is not None:
        extra.append(
            f"DIRECTION TabTransformer vs CatBoost: "
            f"{classify_direction(ctx_run.value, cb_auc, paper_a_beats_b=False, tie_tol=0.005)} "
            f"(paper: matches GBDTs, does not dominate — so CatBoost winning is compatible)."
        )

    delta_cf = None if (ctx_run is None or cf_auc is None) else ScaleUpRun(
        method="delta", dataset="adult", metric="auc_minus_context_free",
        value=ctx_run.value - cf_auc, n_seeds=1, hardware=hw,
        protocol_match=False, protocol_deviations=ctx_run.protocol_deviations,
    )
    delta_cb = None if (ctx_run is None or cb_auc is None) else ScaleUpRun(
        method="delta", dataset="adult", metric="auc_minus_catboost",
        value=ctx_run.value - cb_auc, n_seeds=1, hardware=hw,
        protocol_match=False, protocol_deviations=ctx_run.protocol_deviations,
    )

    rows = [
        (CONTEXT_TARGET, delta_cf, classify_number(CONTEXT_TARGET, delta_cf)),
        (GBDT_TARGET, delta_cb, classify_number(GBDT_TARGET, delta_cb)),
        (RTD_TARGET, rtd_run, classify_number(RTD_TARGET, rtd_run)),
    ]
    text = format_ledger(title="L045 TabTransformer", lab=LAB_FINDINGS, paper=rows, extra_lines=extra)
    print()
    print(text)

    out = {
        "lesson": 45, "preset": args.preset, "hardware": hw,
        "contextual": to_jsonable(ctx_run), "rtd": to_jsonable(rtd_run),
        "context_free_auc": cf_auc, "catboost_auc": cb_auc, "ledger": text,
    }
    dest = os.environ.get("PAPER_REPRO_OUT") or os.path.join(
        HERE, f"_paper_repro_l045_{args.preset}_results.json"
    )
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return out


if __name__ == "__main__":
    main()
