"""L046 paper-results scale-up (NOTES standard #25).

The learning lab trains FT-Transformer on small OpenML tables at a tiny budget. Gorishniy et al. 2021
(FT-Transformer, arXiv:2106.11959, Table 2) report — under a *tuned*, unified protocol on 11 datasets —
that FT-Transformer is the **strongest single deep model** and **roughly ties tuned GBDTs**, while
consistently **beating an MLP and TabTransformer**. On some tables (e.g. Adult) GBDTs still edge it.

This harness re-runs the same from-scratch FT-Transformer on a **paper dataset** (Adult, OpenML 1590; or
Higgs-small, OpenML 23512) against TabTransformer (L045), an MLP, and CatBoost, at three budgets. Absolute
Table-2 numbers stay INCOMPARABLE (we use a random OpenML split and do not run the paper's Optuna tuning);
the DIRECTION tests are what can actually update belief:

  * FT-T vs TabTransformer   (paper: FT-T better — tokenising *numerics* is the upgrade)
  * FT-T vs MLP              (paper: FT-T ≥ MLP — the strongest single deep model)
  * FT-T vs CatBoost         (paper: ~tie on average; GBDT can still win a given table)

Presets: smoke · closer · paper.

Run:
    OMP_NUM_THREADS=1 python labs/_paper_repro_l046.py --preset smoke
    ~/.local/bin/modal run --detach modal/l046_paper_repro.py --preset closer
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
from relkit.tabtransformer import (  # noqa: E402
    TabTransformer, frame_categorical, tabtransformer_auc, train_tabtransformer,
)
from relkit.ft_transformer import (  # noqa: E402
    FTTransformer, ft_transformer_auc, train_ft_transformer,
)
from relkit.nets import TabMLP, net_auc, train_net  # noqa: E402

# Gorishniy 2021, Table 2 — Adult, accuracy. FT-T is the top DL model but GBDTs edge it on THIS table.
ADULT_TARGET = PaperTarget(
    paper="FT-Transformer (Gorishniy et al. 2021)", arxiv="2106.11959",
    table="Table 2 — Adult, accuracy (tuned)",
    dataset="adult (full)", metric="accuracy", paper_value=0.859, abs_tol=0.01,
    paper_split="paper: fixed train/val/test + Optuna tuning; we use OpenML 1590 random split, no tuning",
    higher_is_better=True,
    notes="Table 2 default/tuned FT-T ≈ 0.859; CatBoost/XGBoost ≈ 0.873 (GBDT edges FT-T on Adult).",
)
VS_TABT_TARGET = PaperTarget(
    paper="FT-Transformer (Gorishniy et al. 2021)", arxiv="2106.11959",
    table="Table 2 — FT-T > TabTransformer (tokenising numerics)",
    dataset="adult (full)", metric="auc_minus_tabtransformer", paper_value=0.0, abs_tol=0.02,
    notes="Paper: FT-T beats TabTransformer; the mechanism is that numerics are tokenised and attend.",
)
VS_MLP_TARGET = PaperTarget(
    paper="FT-Transformer (Gorishniy et al. 2021)", arxiv="2106.11959",
    table="Table 2 — FT-T is the strongest single deep model (≥ MLP)",
    dataset="adult (full)", metric="auc_minus_mlp", paper_value=0.0, abs_tol=0.02,
    notes="Paper: FT-T ≥ MLP across the benchmark (often close on any single easy table).",
)

LAB_FINDINGS = [
    LabFinding("FT-T vs TabTransformer vs MLP vs CatBoost mean ranks",
               "see labs/_verify_l046_results.json (4 small tables, 3 seeds, CPU)",
               "credit_g + adult(4000) + churn + phoneme — a demonstration, not the 11-dataset benchmark"),
    LabFinding("numeric-bypass fix",
               "FT-T beats TabTransformer most where numerics carry the signal (num_frac high)",
               "mechanism validated exactly; direction reproduced at small scale"),
]


def load_frame(name, *, cap=None, seed=0):
    Xdf, y = load_tier_a(name)
    if cap is not None and len(Xdf) > cap:
        idx, _ = train_test_split(np.arange(len(Xdf)), train_size=cap, random_state=seed, stratify=y)
        Xdf, y = Xdf.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    # one incomplete OpenML row (e.g. higgs_small) breaks StandardScaler → drop NaN-feature rows
    keep = Xdf.notna().all(axis=1)
    if not bool(keep.all()):
        Xdf, y = Xdf.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(Xdf)
    ct = ColumnTransformer([
        ("num", StandardScaler(), num_names),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_names),
    ])
    Xdense = ct.fit_transform(Xdf).astype(np.float32)
    return dict(Xdf=Xdf, Xcat=Xcat, Xnum=Xnum, cards=cards, y=y.to_numpy().astype(np.float32),
                cat_names=cat_names, num_names=num_names, Xdense=Xdense,
                num_frac=len(num_names) / max(len(num_names) + len(cat_names), 1))


def split_idx(n, y, seed):
    tr, te = train_test_split(np.arange(n), test_size=0.30, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
    return tr, va, te


def _acc(p, y):
    return float(accuracy_score(y, (np.asarray(p) >= 0.5).astype(int)))


def run_ft(fr, tr, va, te, *, cfg, epochs, lr, seed, dev):
    m = FTTransformer(fr["Xnum"].shape[1], fr["cards"], **cfg)
    t0 = time.time()
    m, _ = train_ft_transformer(m, fr["Xnum"][tr], fr["Xcat"][tr], fr["y"][tr],
                                fr["Xnum"][va], fr["Xcat"][va], fr["y"][va],
                                lr=lr, max_epochs=epochs, patience=max(6, epochs // 4),
                                batch_size=256, device=dev, seed=seed)
    import torch
    with torch.no_grad():
        xn = torch.as_tensor(np.asarray(fr["Xnum"][te]), dtype=torch.float32, device=dev)
        xc = torch.as_tensor(np.asarray(fr["Xcat"][te]), dtype=torch.long, device=dev)
        p = torch.sigmoid(m(xn, xc)).cpu().numpy()
    return {"auc": float(roc_auc_score(fr["y"][te], p)), "acc": _acc(p, fr["y"][te])}, time.time() - t0


def run_tabt(fr, tr, va, te, *, epochs, lr, seed, dev):
    cfg = dict(d=32, n_layers=3, n_heads=4, head_hidden=128, dropout=0.1)
    m = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **cfg)
    m, _ = train_tabtransformer(m, fr["Xcat"][tr], fr["Xnum"][tr], fr["y"][tr],
                                fr["Xcat"][va], fr["Xnum"][va], fr["y"][va],
                                lr=lr, max_epochs=epochs, patience=max(6, epochs // 4),
                                batch_size=256, device=dev, seed=seed)
    return float(tabtransformer_auc(m, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te], device=dev))


def run_mlp(fr, tr, va, te, *, epochs, seed):
    m = TabMLP(fr["Xdense"].shape[1], d_block=128, n_blocks=2, dropout=0.1)
    m, _ = train_net(m, fr["Xdense"][tr], fr["y"][tr], fr["Xdense"][va], fr["y"][va],
                     lr=2e-3, max_epochs=epochs, patience=16, seed=seed)
    return float(net_auc(m, fr["Xdense"][te], fr["y"][te]))


def run_catboost(fr, tr, va, te, *, seed):
    from catboost import CatBoostClassifier
    Xdf = fr["Xdf"]
    cat_idx = [Xdf.columns.get_loc(c) for c in fr["cat_names"]]
    Xstr = Xdf.copy()
    for c in fr["cat_names"]:
        Xstr[c] = Xstr[c].map(str).astype(object)
    clf = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, verbose=0,
                             random_seed=seed, allow_writing_files=False)
    clf.fit(Xstr.iloc[tr], fr["y"][tr].astype(int), cat_features=cat_idx,
            eval_set=(Xstr.iloc[va], fr["y"][va].astype(int)), use_best_model=True)
    p = clf.predict_proba(Xstr.iloc[te])[:, 1]
    return {"auc": float(roc_auc_score(fr["y"][te], p)), "acc": _acc(p, fr["y"][te])}


def preset_cfg(name):
    if name == "smoke":
        return dict(cap=700, epochs=2, ft=dict(d=32, n_layers=1, n_heads=4, dropout=0.1))
    if name == "closer":
        return dict(cap=None, epochs=30, ft=dict(d=64, n_layers=3, n_heads=8, dropout=0.1))
    if name == "paper":
        # Gorishniy default FT-T is d≈192, 3 layers, 8 heads. Heavier; T4/paper budget.
        return dict(cap=None, epochs=50, ft=dict(d=192, n_layers=3, n_heads=8, dropout=0.1))
    raise ValueError(f"unknown preset {name!r}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("smoke", "closer", "paper"), default="closer")
    p.add_argument("--dataset", choices=("adult", "higgs_small"), default="adult")
    args = p.parse_args(argv)
    cfg = preset_cfg(args.preset)
    dev = device()
    hw = hardware_tag()
    print(f"L046 paper-repro  preset={args.preset}  dataset={args.dataset}  device={hw}")

    fr = load_frame(args.dataset, cap=cfg["cap"])
    tr, va, te = split_idx(len(fr["y"]), fr["y"], 0)
    print(f"  {args.dataset} n={len(fr['y'])}  cats={len(fr['cards'])}  nums={fr['Xnum'].shape[1]}  "
          f"num_frac={fr['num_frac']:.2f}")

    ft_res = ft_run = tabt_auc = mlp_auc = cb_res = None
    try:
        ft_res, ft_wall = run_ft(fr, tr, va, te, cfg=cfg["ft"], epochs=cfg["epochs"], lr=1e-3, seed=0, dev=dev)
        tabt_auc = run_tabt(fr, tr, va, te, epochs=cfg["epochs"], lr=1e-3, seed=0, dev=dev)
        mlp_auc = run_mlp(fr, tr, va, te, epochs=max(cfg["epochs"] * 3, 60), seed=0)
        cb_res = run_catboost(fr, tr, va, te, seed=0)
        ft_run = ScaleUpRun(
            method="ft-transformer-scratch", dataset=args.dataset, metric="accuracy",
            value=ft_res["acc"], n_seeds=1, hardware=hw, wall_s=ft_wall, protocol_match=False,
            protocol_deviations=[
                f"single dataset ({args.dataset}), not the paper's 11-dataset tuned mean",
                "OpenML random split + no Optuna tuning (paper tunes every model)",
                f"FT-T auc={ft_res['auc']:.4f}  TabT auc={tabt_auc:.4f}  "
                f"MLP auc={mlp_auc:.4f}  CatBoost auc={cb_res['auc']:.4f} acc={cb_res['acc']:.4f}",
            ],
        )
        print(f"  FT-T acc={ft_res['acc']:.4f} auc={ft_res['auc']:.4f} | TabT auc={tabt_auc:.4f} | "
              f"MLP auc={mlp_auc:.4f} | CatBoost acc={cb_res['acc']:.4f} auc={cb_res['auc']:.4f} "
              f"| wall={ft_wall:.0f}s")
    except Exception as exc:
        import traceback
        print(f"  bake-off failed: {exc}")
        traceback.print_exc()

    extra = []
    if ft_res is not None:
        extra.append(
            f"DIRECTION FT-T vs TabTransformer (auc): "
            f"{classify_direction(ft_res['auc'], tabt_auc, paper_a_beats_b=True, tie_tol=0.005)} "
            f"(paper: FT-T beats TabTransformer by tokenising numerics)."
        )
        extra.append(
            f"DIRECTION FT-T vs MLP (auc): "
            f"{classify_direction(ft_res['auc'], mlp_auc, paper_a_beats_b=True, tie_tol=0.005)} "
            f"(paper: FT-T is the strongest single deep model, ≥ MLP)."
        )
        extra.append(
            f"DIRECTION FT-T vs CatBoost (auc): "
            f"{classify_direction(ft_res['auc'], cb_res['auc'], paper_a_beats_b=False, tie_tol=0.005)} "
            f"(paper: FT-T ~ties tuned GBDTs on average; on Adult, GBDT edges it — a loss is compatible)."
        )

    delta_tabt = None if ft_res is None else ScaleUpRun(
        method="delta", dataset=args.dataset, metric="auc_minus_tabtransformer",
        value=ft_res["auc"] - tabt_auc, n_seeds=1, hardware=hw,
        protocol_match=False, protocol_deviations=ft_run.protocol_deviations)
    delta_mlp = None if ft_res is None else ScaleUpRun(
        method="delta", dataset=args.dataset, metric="auc_minus_mlp",
        value=ft_res["auc"] - mlp_auc, n_seeds=1, hardware=hw,
        protocol_match=False, protocol_deviations=ft_run.protocol_deviations)

    rows = [
        (ADULT_TARGET, ft_run, classify_number(ADULT_TARGET, ft_run)),
        (VS_TABT_TARGET, delta_tabt, classify_number(VS_TABT_TARGET, delta_tabt)),
        (VS_MLP_TARGET, delta_mlp, classify_number(VS_MLP_TARGET, delta_mlp)),
    ]
    text = format_ledger(title="L046 FT-Transformer", lab=LAB_FINDINGS, paper=rows, extra_lines=extra)
    print()
    print(text)

    out = {
        "lesson": 46, "preset": args.preset, "dataset": args.dataset, "hardware": hw,
        "ft": to_jsonable(ft_run), "ft_auc": None if ft_res is None else ft_res["auc"],
        "tabtransformer_auc": tabt_auc, "mlp_auc": mlp_auc,
        "catboost": cb_res, "ledger": text,
    }
    dest = os.environ.get("PAPER_REPRO_OUT") or os.path.join(
        HERE, f"_paper_repro_l046_{args.preset}_results.json"
    )
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")
    return out


if __name__ == "__main__":
    main()
