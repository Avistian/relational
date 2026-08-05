"""Verify Lesson 042 — the shared protocol across MULTIPLE datasets (NOTES standards #22, #23).

What this establishes, rigorously:
  1. FROM-SCRATCH models learn the skill; rtdl only VALIDATES them (#22). We train the from-scratch
     `relkit.nets.TabResNet` / `TabMLP` and, on a couple of datasets, confirm the from-scratch ResNet
     reproduces rtdl's reference ResNet within noise under the identical protocol.
  2. NO conclusion from one dataset (#23). We run the shared protocol on several small real OpenML
     tables, report per-dataset mean ± std over seeds, then summarise ACROSS datasets with mean ranks
     and a Friedman test — the honest way to say "no universal winner". The large-N claim itself is
     grounded in the published benchmarks (Grinsztajn 2022 ~45 datasets; Gorishniy 2021), cited in the
     lesson; this harness demonstrates the method + the ranking flip on k datasets.

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l042.py   (writes _verify_l042_results.json)
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from relkit import load_tier_a  # noqa: E402
from relkit.nets import TabResNet, TabMLP, train_net, net_auc  # from-scratch (#22)
from rtdl_revisiting_models import ResNet as RtdlResNet  # reference, for validation only (#22)

DEVICE = "cpu"
DATASETS = ["credit_g", "diabetes", "blood_transfusion", "kc1"]
BUDGET, SEEDS = 6, [0, 1, 2]
EPOCHS, PATIENCE = 120, 12


def load_dense(name):
    Xdf, y = load_tier_a(name)
    num = Xdf.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xdf.columns if c not in num]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    X = ct.fit_transform(Xdf).astype(np.float32)
    return X, y.to_numpy().astype(np.float32)


def frame(X, y, seed):
    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25, random_state=seed, stratify=ytr_f)
    return Xtr, ytr, Xva, yva, Xte, yte


def sample(kind, rng, d_in):
    opt = dict(lr=float(10 ** rng.uniform(-3.3, -2.5)), wd=float(10 ** rng.uniform(-6, -3)))
    if kind == "resnet":
        d_main = int(rng.choice([64, 128, 192]))
        cfg = dict(d_in=d_in, d_main=d_main, d_hidden=int(d_main * rng.choice([1.0, 2.0])),
                   n_blocks=int(rng.choice([1, 2, 3])),
                   dropout1=float(rng.uniform(0.0, 0.3)), dropout2=float(rng.uniform(0.0, 0.3)))
    else:
        cfg = dict(d_in=d_in, d_block=int(rng.choice([64, 128, 256])),
                   n_blocks=int(rng.choice([1, 2, 3])), dropout=float(rng.uniform(0.0, 0.4)))
    return cfg, opt


def make(kind, cfg):
    return TabResNet(**cfg) if kind == "resnet" else TabMLP(**cfg)


def search_net(kind, Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        cfg, opt = sample(kind, rng, Xtr.shape[1])
        m, val = train_net(make(kind, cfg), Xtr, ytr, Xva, yva, lr=opt["lr"], wd=opt["wd"],
                           max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": net_auc(m, Xte, yte)}
    return best["test"]


def search_rtdl_resnet(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    """Reference ResNet under the SAME protocol — used only to validate the from-scratch one (#22)."""
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        d_main = int(rng.choice([64, 128, 192]))
        cfg = dict(d_in=Xtr.shape[1], d_out=1, n_blocks=int(rng.choice([1, 2, 3])), d_block=d_main,
                   d_hidden_multiplier=float(rng.choice([1.0, 2.0])),
                   dropout1=float(rng.uniform(0.0, 0.3)), dropout2=float(rng.uniform(0.0, 0.3)))
        opt = dict(lr=float(10 ** rng.uniform(-3.3, -2.5)), wd=float(10 ** rng.uniform(-6, -3)))
        m, val = train_net(RtdlResNet(**cfg), Xtr, ytr, Xva, yva, lr=opt["lr"], wd=opt["wd"],
                           max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": net_auc(m, Xte, yte)}
    return best["test"]


def search_gbt(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed + 999)
    best = {"val": -1.0, "test": None}
    for _ in range(budget):
        cfg = dict(learning_rate=float(10 ** rng.uniform(-1.5, -0.7)),
                   max_leaf_nodes=int(rng.choice([15, 31, 63])),
                   l2_regularization=float(rng.uniform(0.0, 3.0)), max_iter=400)
        clf = HistGradientBoostingClassifier(random_state=seed, **cfg).fit(Xtr, ytr)
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]


def main():
    t0 = time.time()
    per = {}          # dataset -> model -> {mean, std, seeds}
    validation = {}   # dataset -> {scratch, rtdl, delta}

    for name in DATASETS:
        X, y = load_dense(name)
        print(f"\n=== {name}: X {X.shape}, pos {y.mean():.3f} ===", flush=True)
        rows = {"resnet": [], "mlp": [], "gbt": []}
        for s in SEEDS:
            Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, s)
            rows["resnet"].append(search_net("resnet", Xtr, ytr, Xva, yva, Xte, yte, budget=BUDGET, seed=s))
            rows["mlp"].append(search_net("mlp", Xtr, ytr, Xva, yva, Xte, yte, budget=BUDGET, seed=s))
            rows["gbt"].append(search_gbt(Xtr, ytr, Xva, yva, Xte, yte, budget=BUDGET, seed=s))
            print(f"  seed {s}: resnet {rows['resnet'][-1]:.3f} | mlp {rows['mlp'][-1]:.3f} | gbt {rows['gbt'][-1]:.3f}", flush=True)
        per[name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "seeds": [round(x, 3) for x in v]}
                     for k, v in rows.items()}

        # rtdl validation (#22) on the first two datasets — is our from-scratch ResNet ~ the reference?
        if name in DATASETS[:2]:
            rt = [search_rtdl_resnet(*frame(X, y, s), budget=BUDGET, seed=s) for s in SEEDS]
            sc = per[name]["resnet"]["mean"]; rtm = float(np.mean(rt))
            validation[name] = {"scratch": round(sc, 3), "rtdl": round(rtm, 3), "delta": round(sc - rtm, 3)}
            print(f"  rtdl-validate: from-scratch ResNet {sc:.3f} vs rtdl {rtm:.3f} "
                  f"(|Δ| {abs(sc-rtm):.3f})", flush=True)

    # --- cross-dataset rank summary (#23) ---
    models = ["resnet", "mlp", "gbt"]
    score_matrix = np.array([[per[d][m]["mean"] for m in models] for d in DATASETS])  # datasets x models
    ranks = np.array([rankdata(-row, method="average") for row in score_matrix])       # 1 = best
    mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(models)}
    fried = friedmanchisquare(*[score_matrix[:, i] for i in range(len(models))])

    print("\n================ SUMMARY ================")
    print(f"{'dataset':>18} | " + " | ".join(f"{m:>16}" for m in models))
    for i, d in enumerate(DATASETS):
        print(f"{d:>18} | " + " | ".join(
            f"{per[d][m]['mean']:.3f}±{per[d][m]['std']:.3f}(r{int(ranks[i,j])})" for j, m in enumerate(models)))
    print(f"{'MEAN RANK':>18} | " + " | ".join(f"{mean_rank[m]:>16.2f}" for m in models))
    print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f}  "
          f"(k={len(models)} models, N={len(DATASETS)} datasets)")
    print(f"rtdl validation: " + "; ".join(
        f"{d}: scratch {v['scratch']} vs rtdl {v['rtdl']} (|Δ|={abs(v['delta'])})" for d, v in validation.items()))
    print(f"\nbudget={BUDGET}/model, seeds={SEEDS}, elapsed {time.time()-t0:.1f}s")

    out = {"datasets": DATASETS, "budget": BUDGET, "seeds": SEEDS, "metric": "roc_auc",
           "per_dataset": per, "mean_rank": {m: round(mean_rank[m], 2) for m in models},
           "friedman": {"chi2": round(float(fried.statistic), 3), "p": round(float(fried.pvalue), 3),
                        "k": len(models), "n_datasets": len(DATASETS)},
           "rtdl_validation": validation,
           "note": "single-dataset = demonstration; large-N 'no universal winner' grounded in "
                   "Grinsztajn 2022 (~45 datasets) & Gorishniy 2021 (cited in lesson)."}
    with open(os.path.join(HERE, "_verify_l042_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote labs/_verify_l042_results.json")


if __name__ == "__main__":
    main()
