"""Verify Lesson 044 — NODE held to the baseline-first rule, head-to-head with CatBoost (#22/#23/#24).

Three things, all reproducible:

  1. **Paper-mechanism validation (#22).** The from-scratch alpha-entmax feature choice and the two-class
     "entmoid" soft split — the operations that make an oblivious tree differentiable — match the
     reference `entmax` package (Peters et al. 2019) to ~1e-6. The library is the checker, our code the
     teacher.

  2. **The head-to-head that the paper is about (#24).** NODE vs **CatBoost** — BOTH are ensembles of
     *oblivious* (symmetric) decision trees (L016); the only difference is NODE trains its splits by
     gradient descent end-to-end while CatBoost grows them greedily. This isolates "does making the
     oblivious tree differentiable buy accuracy?" — the paper's central claim.

  3. **The baseline-first bake-off (#23).** NODE vs CatBoost vs the L042 strong-simple baselines (MLP,
     ResNet), one shared frame (split / metric / budget / selection rule), per-dataset mean +- std over
     seeds, then mean ranks + a Friedman test across datasets.

HONEST SCOPE (#20 + #23): 4 small OpenML tables is a *demonstration of the method*, not proof at
benchmark scale. NODE's paper claim ("beats GBDT on most of 40+ tasks, by a small margin") is grounded in
the paper's own large study and in Grinsztajn 2022 / Gorishniy 2021 — not in these 4 tables. We reproduce
the *mechanism* and the *shape* of the result (NODE competitive-but-not-dominant, at much higher cost).

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l044.py   (writes _verify_l044_results.json)
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.stats import friedmanchisquare, rankdata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit import load_tier_a                                        # noqa: E402
from relkit.nets import TabResNet, TabMLP, train_net, net_auc         # from-scratch (L042)
from relkit.node import (entmax15, entmoid15, DenseNODE,              # from-scratch (L044)
                         train_node, node_auc)

DATASETS = ["credit_g", "diabetes", "blood_transfusion", "kc1"]
BUDGET, SEEDS = 4, [0, 1, 2]
EPOCHS, PATIENCE = 100, 10


# ---------------------------------------------------------------- shared frame (identical across arms)
def load_dense(name):
    Xdf, y = load_tier_a(name)
    num = Xdf.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xdf.columns if c not in num]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    return ct.fit_transform(Xdf).astype(np.float32), y.to_numpy().astype(np.float32)


def frame(X, y, seed):
    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25, random_state=seed, stratify=ytr_f)
    return Xtr, ytr, Xva, yva, Xte, yte


# ---------------------------------------------------------------- per-model search spaces
def sample_node(rng, d_in):
    return dict(in_features=d_in,
                num_trees=int(rng.choice([64, 128])),
                depth=int(rng.choice([3, 4, 5])),
                n_layers=int(rng.choice([1, 2]))), dict(lr=float(rng.choice([0.005, 0.01, 0.02])))


def search_node(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None, "cfg": None}
    for t in range(budget):
        cfg, opt = sample_node(rng, Xtr.shape[1])
        torch.manual_seed(seed + t)
        m, val = train_node(DenseNODE(**cfg), Xtr, ytr, Xva, yva, lr=opt["lr"],
                            max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": node_auc(m, Xte, yte), "cfg": cfg}
    return best["test"]


def search_catboost(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    """CatBoost = greedy oblivious/symmetric trees (Prokhorenkova 2018, L016). NODE's direct rival."""
    from catboost import CatBoostClassifier
    rng = np.random.default_rng(seed + 7)
    best = {"val": -1.0, "test": None}
    for _ in range(budget):
        clf = CatBoostClassifier(
            depth=int(rng.choice([4, 6, 8])),                         # symmetric tree depth
            learning_rate=float(10 ** rng.uniform(-1.5, -0.7)),
            l2_leaf_reg=float(rng.choice([1.0, 3.0, 5.0, 9.0])),
            iterations=400, grow_policy="SymmetricTree",              # oblivious trees, explicitly
            random_seed=seed, thread_count=2, verbose=0, allow_writing_files=False)
        clf.fit(Xtr, ytr.astype(int))
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]


def sample_net(kind, rng, d_in):
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


def search_net(kind, Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        cfg, opt = sample_net(kind, rng, Xtr.shape[1])
        model = TabResNet(**cfg) if kind == "resnet" else TabMLP(**cfg)
        m, val = train_net(model, Xtr, ytr, Xva, yva, lr=opt["lr"], wd=opt["wd"],
                           max_epochs=200, patience=16, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": net_auc(m, Xte, yte)}
    return best["test"]


# ---------------------------------------------------------------- (1) mechanism validation (#22)
def verify_mechanism():
    print("\n########## 1. MECHANISM — from-scratch entmax/entmoid vs the reference (#22) ##########")
    from entmax import entmax15 as ref_entmax15
    rng = np.random.default_rng(0)
    Z = torch.tensor(rng.normal(size=(300, 16)) * 3, dtype=torch.float64)
    d_ent = (entmax15(Z, dim=-1, n_iter=50) - ref_entmax15(Z, dim=-1)).abs().max().item()
    t = torch.tensor(rng.normal(size=800) * 3, dtype=torch.float64)
    two_class = ref_entmax15(torch.stack([t, torch.zeros_like(t)], -1), dim=-1)[..., 0]
    d_moid = (entmoid15(t) - two_class).abs().max().item()
    zeros = float((entmax15(Z, dim=-1) == 0).double().mean())
    print(f"entmax15 vs entmax.entmax15   : max |Δ| {d_ent:.2e}  ({'VALIDATED' if d_ent < 1e-5 else 'FAIL'})")
    print(f"entmoid15 vs two-class entmax : max |Δ| {d_moid:.2e}  ({'VALIDATED' if d_moid < 1e-5 else 'FAIL'})")
    print(f"entmax15 sparsity on random logits: {zeros*100:.0f}% exact zeros")
    return {"entmax15_max_abs_delta": d_ent, "entmoid15_max_abs_delta": d_moid,
            "entmax15_frac_zeros": round(zeros, 3),
            "validated": d_ent < 1e-5 and d_moid < 1e-5}


# ---------------------------------------------------------------- (2) the bake-off + NODE vs CatBoost
def verify_bakeoff():
    print("\n########## 2. BAKE-OFF — NODE vs CatBoost vs the strong simple baselines ##########")
    models = ["node", "catboost", "mlp", "resnet"]
    per = {}
    for name in DATASETS:
        X, y = load_dense(name)
        print(f"\n=== {name}: X {X.shape}, pos {y.mean():.3f} ===", flush=True)
        rows = {m: [] for m in models}
        for s in SEEDS:
            f = frame(X, y, s)
            rows["node"].append(search_node(*f, budget=BUDGET, seed=s))
            rows["catboost"].append(search_catboost(*f, budget=BUDGET, seed=s))
            rows["mlp"].append(search_net("mlp", *f, budget=BUDGET, seed=s))
            rows["resnet"].append(search_net("resnet", *f, budget=BUDGET, seed=s))
            print("  seed {}: ".format(s) + " | ".join(f"{m} {rows[m][-1]:.3f}" for m in models), flush=True)
        per[name] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                         "seeds": [round(float(x), 3) for x in v]} for m, v in rows.items()}

    score = np.array([[per[d][m]["mean"] for m in models] for d in DATASETS])
    ranks = np.array([rankdata(-row, method="average") for row in score])
    mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(models)}
    fried = friedmanchisquare(*[score[:, i] for i in range(len(models))])

    # the paper's head-to-head, isolated: NODE vs CatBoost (both oblivious tree ensembles)
    node_beats_cat = sum(per[d]["node"]["mean"] > per[d]["catboost"]["mean"] for d in DATASETS)

    print("\n---------------- bake-off summary ----------------")
    print(f"{'dataset':>18} | " + " | ".join(f"{m:>16}" for m in models))
    for i, d in enumerate(DATASETS):
        print(f"{d:>18} | " + " | ".join(
            f"{per[d][m]['mean']:.3f}±{per[d][m]['std']:.3f}(r{int(ranks[i,j])})"
            for j, m in enumerate(models)))
    print(f"{'MEAN RANK':>18} | " + " | ".join(f"{mean_rank[m]:>16.2f}" for m in models))
    print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (k={len(models)}, N={len(DATASETS)})")
    print(f"NODE beats CatBoost on {node_beats_cat}/{len(DATASETS)} tables (both oblivious tree ensembles)")
    return {"models": models, "per_dataset": per,
            "mean_rank": {m: round(mean_rank[m], 2) for m in models},
            "friedman": {"chi2": round(float(fried.statistic), 3), "p": round(float(fried.pvalue), 3),
                         "k": len(models), "n_datasets": len(DATASETS)},
            "node_beats_catboost": f"{node_beats_cat}/{len(DATASETS)}"}


# ---------------------------------------------------------------- (3) cost — the honest caveat
def verify_cost():
    print("\n########## 3. COST — the price of differentiability (NODE vs CatBoost wall-clock) ##########")
    X, y = load_dense("credit_g")
    Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, 0)

    t0 = time.time(); torch.manual_seed(0)
    m = DenseNODE(X.shape[1], num_trees=128, depth=6, n_layers=1)
    m, _ = train_node(m, Xtr, ytr, Xva, yva, lr=1e-2, max_epochs=EPOCHS, patience=PATIENCE, seed=0)
    node_s = time.time() - t0; node_test = node_auc(m, Xte, yte)

    from catboost import CatBoostClassifier
    t0 = time.time()
    clf = CatBoostClassifier(depth=6, iterations=400, learning_rate=0.05,
                             grow_policy="SymmetricTree", random_seed=0, thread_count=2,
                             verbose=0, allow_writing_files=False).fit(Xtr, ytr.astype(int))
    cat_s = time.time() - t0; cat_test = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    print(f"NODE     (128 trees, depth 6): {node_s:6.1f}s  test AUC {node_test:.3f}")
    print(f"CatBoost (400 trees, depth 6): {cat_s:6.1f}s  test AUC {cat_test:.3f}")
    print(f"NODE is {node_s / max(cat_s, 1e-6):.0f}x slower to train on this table")
    return {"node_seconds": round(node_s, 1), "catboost_seconds": round(cat_s, 1),
            "node_test_auc": round(node_test, 3), "catboost_test_auc": round(cat_test, 3),
            "slowdown": round(node_s / max(cat_s, 1e-6), 1)}


def main():
    t0 = time.time()
    mech = verify_mechanism()
    bakeoff = verify_bakeoff()
    cost = verify_cost()

    out = {"paper": "Popov, Morozov & Babenko 2019, arXiv:1909.06312 (NODE)", "metric": "roc_auc",
           "budget": BUDGET, "seeds": SEEDS, "datasets": DATASETS,
           "epochs": EPOCHS, "patience": PATIENCE,
           "mechanism": mech, "bakeoff": bakeoff, "cost": cost,
           "elapsed_s": round(time.time() - t0, 1),
           "note": "NODE and CatBoost are BOTH oblivious (symmetric) tree ensembles; the experiment "
                   "isolates the effect of making the tree differentiable. 4 small tables is a "
                   "demonstration, not benchmark-scale proof — the paper's 'beats GBDT on most tasks by "
                   "a small margin' claim is grounded in its own 40+ dataset study and Grinsztajn 2022."}
    with open(os.path.join(HERE, "_verify_l044_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote labs/_verify_l044_results.json  (total {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
