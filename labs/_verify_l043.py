"""Verify Lesson 043 — TabNet held to the baseline-first rule, and its masks read (standards #22/#23/#24).

Three things, all reproducible:

  1. **Paper-mirror interpretability (the paper's own data).** On Syn2 / Syn4 (TabNet Table 1 / Fig. 5),
     does the aggregate mask M_agg land on the truly-relevant features, and does the *instance-wise*
     selection on Syn4 switch with X11? This reproduces the paper's mask-reading claim.

  2. **The honest bake-off (#23).** From-scratch TabNet vs the L042 strong simple baselines (MLP,
     ResNet) and a tuned GBDT, on several small real tables, identical frame (split / metric / budget /
     selection rule), per-dataset mean +- std over seeds, then mean ranks + Friedman across datasets.

  3. **Reference validation (#22).** The from-scratch TabNet is compared with `pytorch_tabnet` under the
     same protocol; the library is a validation point, never the teacher.

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l043.py   (writes _verify_l043_results.json)
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
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit import load_tier_a                                              # noqa: E402
from relkit.nets import TabResNet, TabMLP, train_net, net_auc               # from-scratch (L042)
from relkit.tabnet import TabNetEncoder, train_tabnet, tabnet_auc, explain  # from-scratch (L043)
from relkit.synth import make_syn2, make_syn4                              # noqa: E402

DATASETS = ["credit_g", "diabetes", "blood_transfusion", "kc1"]
BUDGET, SEEDS = 6, [0, 1, 2]
EPOCHS, PATIENCE = 120, 12


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


# ---------------------------------------------------------------- per-model search spaces (paper App. F)
def sample_tabnet(rng, d_in):
    n_units = int(rng.choice([8, 16, 24]))
    return dict(d_in=d_in, n_d=n_units, n_a=n_units,
                n_steps=int(rng.choice([3, 4, 5])),
                gamma=float(rng.choice([1.0, 1.2, 1.5, 2.0])),
                lambda_sparse=float(rng.choice([0.0, 1e-6, 1e-4, 1e-3, 1e-2])),
                virtual_batch_size=int(rng.choice([64, 128])),
                momentum=0.02), dict(lr=float(rng.choice([0.005, 0.01, 0.02, 0.025])))


def search_tabnet(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None, "cfg": None}
    for t in range(budget):
        cfg, opt = sample_tabnet(rng, Xtr.shape[1])
        torch.manual_seed(seed + t)
        m, val = train_tabnet(TabNetEncoder(**cfg), Xtr, ytr, Xva, yva, lr=opt["lr"],
                              max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": tabnet_auc(m, Xte, yte), "cfg": cfg}
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


def search_ref_tabnet(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    """Reference pytorch_tabnet under the SAME protocol — validation point only (#22)."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        n_units = int(rng.choice([8, 16, 24]))
        clf = TabNetClassifier(
            n_d=n_units, n_a=n_units, n_steps=int(rng.choice([3, 4, 5])),
            gamma=float(rng.choice([1.0, 1.2, 1.5, 2.0])),
            lambda_sparse=float(rng.choice([0.0, 1e-6, 1e-4, 1e-3, 1e-2])),
            optimizer_params=dict(lr=float(rng.choice([0.005, 0.01, 0.02, 0.025]))),
            seed=seed + t, verbose=0, device_name="cpu")
        clf.fit(Xtr, ytr.astype(int), eval_set=[(Xva, yva.astype(int))], eval_metric=["auc"],
                max_epochs=EPOCHS, patience=PATIENCE, batch_size=1024,
                virtual_batch_size=128, drop_last=False)
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]


# ---------------------------------------------------------------- (1) paper-mirror: read the masks
def verify_masks():
    print("\n########## 1. MASK READING — the paper's own synthetic data (Table 1 / Fig. 5) ##########")
    out = {}

    # --- Syn2: relevant features are GLOBAL (X3-X6). ---
    X, y, rel = make_syn2(n=4000, seed=0)
    Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, 0)
    torch.manual_seed(0)
    m = TabNetEncoder(X.shape[1], n_d=16, n_a=16, n_steps=4, gamma=2.0, lambda_sparse=1e-2,
                      virtual_batch_size=128)
    m, _ = train_tabnet(m, Xtr, ytr, Xva, yva, lr=0.02, max_epochs=200, patience=25,
                        batch_size=1024, seed=0)
    auc = tabnet_auc(m, Xte, yte)
    M_agg, masks = explain(m, Xte)
    imp = M_agg.mean(0)
    mass = float(imp[rel].sum())
    top4 = sorted(np.argsort(-imp)[:4].tolist())
    print(f"Syn2: test AUC {auc:.3f} | mask mass on true X3-X6 = {mass*100:.1f}% | top-4 = {top4} "
          f"(truth {rel})")
    print("      per-feature M_agg: " + " ".join(f"X{j+1}:{v:.3f}" for j, v in enumerate(imp)))
    out["syn2"] = {"auc": round(float(auc), 3), "mass_on_relevant": round(mass, 3),
                   "top4": top4, "relevant": rel, "M_agg_mean": [round(float(v), 4) for v in imp],
                   "recovered": top4 == rel}

    # --- Syn4: relevant features are INSTANCE-WISE, switched by X11. ---
    X, y, _ = make_syn4(n=6000, seed=0)
    Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, 0)
    # Relevance is recomputed from the test rows themselves (X11's sign), so the split shuffle is safe.
    torch.manual_seed(0)
    m4 = TabNetEncoder(X.shape[1], n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=5e-3,
                       virtual_batch_size=128)
    m4, _ = train_tabnet(m4, Xtr, ytr, Xva, yva, lr=0.02, max_epochs=200, patience=25,
                         batch_size=1024, seed=0)
    auc4 = tabnet_auc(m4, Xte, yte)
    M_agg4, _ = explain(m4, Xte)

    left = Xte[:, 10] < 0                       # rows where X1-X2 (XOR) is the relevant group
    xor_mass = M_agg4[:, [0, 1]].sum(1)
    orange_mass = M_agg4[:, 2:6].sum(1)
    # The instance-wise claim: on X11<0 rows the mask should favour X1-X2; on X11>0 rows, X3-X6.
    left_prefers_xor = float((xor_mass[left] > orange_mass[left]).mean())
    right_prefers_orange = float((orange_mass[~left] > xor_mass[~left]).mean())
    switch_importance = float(M_agg4[:, 10].mean())
    print(f"Syn4: test AUC {auc4:.3f} | X11 (the switch) mask weight {switch_importance:.3f}")
    print(f"      rows with X11<0 whose mask favours X1-X2 : {left_prefers_xor*100:.1f}%")
    print(f"      rows with X11>0 whose mask favours X3-X6 : {right_prefers_orange*100:.1f}%")
    print(f"      mean mask mass  X1-X2: left {xor_mass[left].mean():.3f} / right {xor_mass[~left].mean():.3f}"
          f"   X3-X6: left {orange_mass[left].mean():.3f} / right {orange_mass[~left].mean():.3f}")
    out["syn4"] = {"auc": round(float(auc4), 3),
                   "switch_feature_weight": round(switch_importance, 3),
                   "left_prefers_xor": round(left_prefers_xor, 3),
                   "right_prefers_orange": round(right_prefers_orange, 3),
                   "xor_mass_left": round(float(xor_mass[left].mean()), 3),
                   "xor_mass_right": round(float(xor_mass[~left].mean()), 3),
                   "orange_mass_left": round(float(orange_mass[left].mean()), 3),
                   "orange_mass_right": round(float(orange_mass[~left].mean()), 3),
                   "instance_wise_evidence": left_prefers_xor > 0.5 and right_prefers_orange > 0.5}

    # A global (dataset-level) selector cannot express this: report the single best fixed subset.
    global_imp = M_agg4.mean(0)
    out["syn4"]["global_M_agg_mean"] = [round(float(v), 4) for v in global_imp]
    print("      dataset-level (global) M_agg: " +
          " ".join(f"X{j+1}:{v:.3f}" for j, v in enumerate(global_imp)))
    return out


# ---------------------------------------------------------------- (2) the honest bake-off
def verify_bakeoff():
    print("\n########## 2. BAKE-OFF — TabNet vs the strong simple baselines, one shared frame ##########")
    models = ["tabnet", "mlp", "resnet", "gbt"]
    per = {}
    for name in DATASETS:
        X, y = load_dense(name)
        print(f"\n=== {name}: X {X.shape}, pos {y.mean():.3f} ===", flush=True)
        rows = {m: [] for m in models}
        for s in SEEDS:
            f = frame(X, y, s)
            rows["tabnet"].append(search_tabnet(*f, budget=BUDGET, seed=s))
            rows["mlp"].append(search_net("mlp", *f, budget=BUDGET, seed=s))
            rows["resnet"].append(search_net("resnet", *f, budget=BUDGET, seed=s))
            rows["gbt"].append(search_gbt(*f, budget=BUDGET, seed=s))
            print("  seed {}: ".format(s) + " | ".join(f"{m} {rows[m][-1]:.3f}" for m in models), flush=True)
        per[name] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                         "seeds": [round(float(x), 3) for x in v]} for m, v in rows.items()}

    score = np.array([[per[d][m]["mean"] for m in models] for d in DATASETS])
    ranks = np.array([rankdata(-row, method="average") for row in score])
    mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(models)}
    fried = friedmanchisquare(*[score[:, i] for i in range(len(models))])

    print("\n---------------- bake-off summary ----------------")
    print(f"{'dataset':>18} | " + " | ".join(f"{m:>18}" for m in models))
    for i, d in enumerate(DATASETS):
        print(f"{d:>18} | " + " | ".join(
            f"{per[d][m]['mean']:.3f}±{per[d][m]['std']:.3f}(r{int(ranks[i,j])})"
            for j, m in enumerate(models)))
    print(f"{'MEAN RANK':>18} | " + " | ".join(f"{mean_rank[m]:>18.2f}" for m in models))
    print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} "
          f"(k={len(models)}, N={len(DATASETS)})")
    return {"models": models, "per_dataset": per,
            "mean_rank": {m: round(mean_rank[m], 2) for m in models},
            "friedman": {"chi2": round(float(fried.statistic), 3), "p": round(float(fried.pvalue), 3),
                         "k": len(models), "n_datasets": len(DATASETS)}}


# ---------------------------------------------------------------- (3) reference validation
def verify_reference(bakeoff):
    print("\n########## 3. VALIDATE the from-scratch TabNet against pytorch_tabnet (#22) ##########")
    val = {}
    for name in DATASETS[:2]:
        X, y = load_dense(name)
        try:
            ref = [search_ref_tabnet(*frame(X, y, s), budget=BUDGET, seed=s) for s in SEEDS]
        except Exception as e:                                 # noqa: BLE001
            print(f"  {name}: reference unavailable ({e})")
            continue
        ours = bakeoff["per_dataset"][name]["tabnet"]["mean"]
        refm = float(np.mean(ref))
        val[name] = {"scratch": round(ours, 3), "reference": round(refm, 3),
                     "delta": round(ours - refm, 3), "within_tol": abs(ours - refm) < 0.04}
        print(f"  {name}: from-scratch {ours:.3f} vs pytorch_tabnet {refm:.3f} "
              f"(|Δ| {abs(ours-refm):.3f}, tol 0.04) -> "
              f"{'VALIDATED' if val[name]['within_tol'] else 'OUT OF TOLERANCE'}")
    return val


def main():
    t0 = time.time()
    masks = verify_masks()
    bakeoff = verify_bakeoff()
    reference = verify_reference(bakeoff)

    out = {"paper": "Arik & Pfister 2019, arXiv:1908.07442", "metric": "roc_auc",
           "budget": BUDGET, "seeds": SEEDS, "datasets": DATASETS,
           "epochs": EPOCHS, "patience": PATIENCE,
           "mask_reading": masks, "bakeoff": bakeoff, "reference_validation": reference,
           "elapsed_s": round(time.time() - t0, 1),
           "note": "Mask reading mirrors TabNet Table 1/Fig. 5 on the paper's synthetic generators "
                   "(claim reproduced = which features are selected, NOT the paper's exact AUCs, which "
                   "used far larger budgets). Bake-off is 4 small tables: a demonstration of the method, "
                   "not proof at benchmark scale — the representative verdict is grounded in Grinsztajn "
                   "2022 (~45 datasets) and Gorishniy 2021."}
    with open(os.path.join(HERE, "_verify_l043_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote labs/_verify_l043_results.json  (total {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
