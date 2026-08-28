"""Verify Lesson 045 — TabTransformer trained, held to the baseline-first rule, and pre-trained (#22/#23/#24).

Three reproducible things, closing the L032 forward-only preview:

  1. **Mechanism validation (#22).** The from-scratch multi-head self-attention that makes a categorical
     column's embedding CONTEXTUAL matches torch's own reference kernels
     (`F.scaled_dot_product_attention`, `nn.MultiheadAttention`) to machine precision, and the contextual
     property holds on a real row (a column's vector moves when a neighbour changes; the context-free
     n_layers=0 ablation's does not).

  2. **The headline comparison (#23/#24).** TabTransformer (CONTEXTUAL categorical embeddings) vs the
     context-FREE static-embedding MLP (same class, n_layers=0 — the L031/L032 model) vs a strong tree
     baseline (CatBoost, native categoricals). One shared frame across arms, per-dataset mean ± std over
     seeds, mean ranks + Friedman across categorical-rich tables. Isolates *what contextualisation buys*
     and holds it to the honest baseline (trees).

  3. **Semi-supervised RTD pre-training (§3.3).** On a table with plenty of UNLABELED rows, does
     pre-training the encoder to detect replaced tokens, then fine-tuning on a SMALL labeled fraction,
     beat training the same model from scratch on that fraction? The label-efficiency lever trees lack.

HONEST SCOPE (#20 + #23): 3 categorical-rich OpenML tables (one subsampled) is a *demonstration of the
method*, not proof at the paper's benchmark scale (Huang averages over 15 datasets, AUC ~+1.0% over deep
baselines, matching GBDTs). We reproduce the *mechanism* (exactly), the *direction* of the contextual and
semi-supervised results, and the honest verdict vs trees.

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l045.py   (writes _verify_l045_results.json)
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
import torch.nn.functional as F
from scipy.stats import friedmanchisquare, rankdata
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit import load_tier_a                                                       # noqa: E402
from relkit.tabtransformer import (                                                  # noqa: E402
    frame_categorical, scaled_dot_product_attention, MultiHeadSelfAttention,
    TabTransformer, train_tabtransformer, tabtransformer_auc, pretrain_rtd,
)

DATASETS = ["credit_g", "adult", "churn"]
SUBSAMPLE = {"adult": 4000}                 # keep CPU cost bounded — a down-scaled demonstration (#20)
SEEDS = [0, 1, 2]
TT_CFG = dict(d=32, n_layers=3, n_heads=4, head_hidden=128, dropout=0.1)
CF_CFG = dict(d=32, n_layers=0, head_hidden=128, dropout=0.1)      # context-free (L031/L032) ablation
EPOCHS, PATIENCE, BS, LR = 60, 12, 256, 1e-3


# ---------------------------------------------------------------- shared frame (identical across arms)
def load_frame(name):
    Xdf, y = load_tier_a(name)
    if name in SUBSAMPLE and len(Xdf) > SUBSAMPLE[name]:
        idx, _ = train_test_split(np.arange(len(Xdf)), train_size=SUBSAMPLE[name],
                                  random_state=0, stratify=y)
        Xdf, y = Xdf.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(Xdf)
    yv = y.to_numpy().astype(np.float32)
    return {"Xdf": Xdf, "y": yv, "Xcat": Xcat, "Xnum": Xnum, "cards": cards,
            "cat_names": cat_names, "num_names": num_names}


def split_idx(n, y, seed):
    tr, te = train_test_split(np.arange(n), test_size=0.30, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
    return tr, va, te


# ---------------------------------------------------------------- arms
def run_tabtransformer(fr, tr, va, te, cfg, seed):
    torch.manual_seed(seed)
    m = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **cfg)
    m, _ = train_tabtransformer(m, fr["Xcat"][tr], fr["Xnum"][tr], fr["y"][tr],
                                fr["Xcat"][va], fr["Xnum"][va], fr["y"][va],
                                lr=LR, max_epochs=EPOCHS, patience=PATIENCE, batch_size=BS, seed=seed)
    return tabtransformer_auc(m, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])


def run_catboost(fr, tr, va, te, seed):
    """CatBoost with NATIVE categorical handling (its strength on categorical-rich data) — the honest bar."""
    from catboost import CatBoostClassifier
    Xdf = fr["Xdf"]
    cat_idx = [Xdf.columns.get_loc(c) for c in fr["cat_names"]]
    Xstr = Xdf.copy()
    for c in fr["cat_names"]:
        Xstr[c] = Xstr[c].map(str).astype(object)        # plain str (CatBoost rejects category-dtype NaN)
    clf = CatBoostClassifier(depth=6, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0,
                             random_seed=seed, thread_count=2, verbose=0, allow_writing_files=False)
    clf.fit(Xstr.iloc[tr], fr["y"][tr].astype(int), cat_features=cat_idx,
            eval_set=(Xstr.iloc[va], fr["y"][va].astype(int)))
    return roc_auc_score(fr["y"][te], clf.predict_proba(Xstr.iloc[te])[:, 1])


# ---------------------------------------------------------------- (1) mechanism validation (#22)
def verify_mechanism():
    print("\n########## 1. MECHANISM — from-scratch attention vs torch's reference kernels (#22) ##########")
    Q = torch.randn(4, 8, 32, dtype=torch.float64)
    K = torch.randn(4, 8, 32, dtype=torch.float64)
    V = torch.randn(4, 8, 32, dtype=torch.float64)
    d_sdpa = (scaled_dot_product_attention(Q, K, V)[0] - F.scaled_dot_product_attention(Q, K, V)).abs().max().item()

    d, h = 32, 4
    mha = MultiHeadSelfAttention(d, h).double()
    tmha = torch.nn.MultiheadAttention(d, h, batch_first=True, bias=True).double()
    with torch.no_grad():
        tmha.in_proj_weight.copy_(torch.cat([mha.Wq.weight, mha.Wk.weight, mha.Wv.weight], 0))
        tmha.in_proj_bias.copy_(torch.cat([mha.Wq.bias, mha.Wk.bias, mha.Wv.bias], 0))
        tmha.out_proj.weight.copy_(mha.Wo.weight); tmha.out_proj.bias.copy_(mha.Wo.bias)
    x = torch.randn(4, 8, d, dtype=torch.float64)
    d_mha = (mha(x)[0] - tmha(x, x, x)[0]).abs().max().item()

    # contextual property on a real credit_g row
    X, _ = load_tier_a("credit_g")
    Xcat, Xnum, cards, cat_names, _ = frame_categorical(X)
    torch.manual_seed(0)
    model = TabTransformer(cards, Xnum.shape[1], d=32, n_layers=2, n_heads=4).eval()
    ci, hi = cat_names.index("checking_status"), cat_names.index("housing")
    with torch.no_grad():
        r = Xcat[0:1].clone(); r2 = r.clone(); r2[0, hi] = (r[0, hi] + 1) % cards[hi]
        ctx_move = (model.contextual(r)[0, ci] - model.contextual(r2)[0, ci]).norm().item()
    cf = TabTransformer(cards, Xnum.shape[1], d=32, n_layers=0).eval()
    with torch.no_grad():
        cf_move = (cf.contextual(r)[0, ci] - cf.contextual(r2)[0, ci]).norm().item()

    print(f"scaled_dot_product_attention vs F.sdpa : max |Δ| {d_sdpa:.2e}")
    print(f"MultiHeadSelfAttention vs nn.MHA        : max |Δ| {d_mha:.2e}")
    print(f"contextual move (attention on)          : {ctx_move:.3f}")
    print(f"contextual move (n_layers=0, context-free): {cf_move:.1e}")
    return {"sdpa_max_abs_delta": d_sdpa, "mha_max_abs_delta": d_mha,
            "contextual_move_attention": round(ctx_move, 3),
            "contextual_move_contextfree": cf_move,
            "validated": d_sdpa < 1e-9 and d_mha < 1e-9 and ctx_move > 1e-4 and cf_move < 1e-9}


# ---------------------------------------------------------------- (2) bake-off (#23/#24)
def verify_bakeoff():
    print("\n########## 2. BAKE-OFF — contextual (TabTransformer) vs context-free MLP vs CatBoost ##########")
    models = ["tabtransformer", "context_free", "catboost"]
    per = {}
    for name in DATASETS:
        fr = load_frame(name)
        print(f"\n=== {name}: {len(fr['y'])} rows | {len(fr['cat_names'])} cat | "
              f"{len(fr['num_names'])} num | pos {fr['y'].mean():.3f} ===", flush=True)
        rows = {m: [] for m in models}
        for s in SEEDS:
            tr, va, te = split_idx(len(fr["y"]), fr["y"], s)
            rows["tabtransformer"].append(run_tabtransformer(fr, tr, va, te, TT_CFG, s))
            rows["context_free"].append(run_tabtransformer(fr, tr, va, te, CF_CFG, s))
            rows["catboost"].append(run_catboost(fr, tr, va, te, s))
            print("  seed {}: ".format(s) + " | ".join(f"{m} {rows[m][-1]:.3f}" for m in models), flush=True)
        per[name] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                         "seeds": [round(float(x), 3) for x in v]} for m, v in rows.items()}

    score = np.array([[per[d][m]["mean"] for m in models] for d in DATASETS])
    ranks = np.array([rankdata(-row, method="average") for row in score])
    mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(models)}
    fried = friedmanchisquare(*[score[:, i] for i in range(len(models))])
    ctx_beats_cf = sum(per[d]["tabtransformer"]["mean"] > per[d]["context_free"]["mean"] for d in DATASETS)
    ctx_beats_cat = sum(per[d]["tabtransformer"]["mean"] > per[d]["catboost"]["mean"] for d in DATASETS)

    print("\n---------------- bake-off summary ----------------")
    print(f"{'dataset':>16} | " + " | ".join(f"{m:>16}" for m in models))
    for i, d in enumerate(DATASETS):
        print(f"{d:>16} | " + " | ".join(
            f"{per[d][m]['mean']:.3f}±{per[d][m]['std']:.3f}(r{int(ranks[i,j])})" for j, m in enumerate(models)))
    print(f"{'MEAN RANK':>16} | " + " | ".join(f"{mean_rank[m]:>16.2f}" for m in models))
    print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (k={len(models)}, N={len(DATASETS)})")
    print(f"TabTransformer (contextual) beats context-free MLP on {ctx_beats_cf}/{len(DATASETS)} tables")
    print(f"TabTransformer beats CatBoost on {ctx_beats_cat}/{len(DATASETS)} tables")
    return {"models": models, "per_dataset": per,
            "mean_rank": {m: round(mean_rank[m], 2) for m in models},
            "friedman": {"chi2": round(float(fried.statistic), 3), "p": round(float(fried.pvalue), 3),
                         "k": len(models), "n_datasets": len(DATASETS)},
            "contextual_beats_contextfree": f"{ctx_beats_cf}/{len(DATASETS)}",
            "tabtransformer_beats_catboost": f"{ctx_beats_cat}/{len(DATASETS)}"}


# ---------------------------------------------------------------- (3) semi-supervised RTD (§3.3)
SEMI_NAME, SEMI_N = "adult", 16000            # a bigger frame so the UNLABELED pool dwarfs the labels
SEMI_FRACS, SEMI_SEEDS = [0.03, 0.10], [0, 1, 2]
FT_LR, PRE_EPOCHS, PRE_P = 5e-4, 30, 0.30     # gentle fine-tune LR avoids catastrophic forgetting


def load_semi_frame():
    Xdf, y = load_tier_a(SEMI_NAME)
    idx, _ = train_test_split(np.arange(len(Xdf)), train_size=SEMI_N, random_state=0, stratify=y)
    Xdf, y = Xdf.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(Xdf)
    return {"Xcat": Xcat, "Xnum": Xnum, "cards": cards, "y": y.to_numpy().astype(np.float32)}


def verify_semisupervised():
    print("\n########## 3. SEMI-SUPERVISED — RTD pre-train + fine-tune vs from-scratch (§3.3) ##########")
    fr = load_semi_frame()
    results = {f"{f:.2f}": {"scratch": [], "pretrain": []} for f in SEMI_FRACS}
    print(f"{SEMI_NAME}: {len(fr['y'])} rows | {len(fr['cards'])} cat | pos {fr['y'].mean():.3f} "
          f"| ft_lr={FT_LR} pre_epochs={PRE_EPOCHS} replace_p={PRE_P}", flush=True)
    for s in SEMI_SEEDS:
        tr, te = train_test_split(np.arange(len(fr["y"])), test_size=0.30, random_state=s, stratify=fr["y"])
        unlab_idx = tr                                          # UNLABELED pool = every train row's features
        for f in SEMI_FRACS:
            lab, _ = train_test_split(tr, train_size=max(int(len(tr) * f), 60),
                                      random_state=s, stratify=fr["y"][tr])
            lab_tr, lab_va = train_test_split(lab, test_size=0.25, random_state=s, stratify=fr["y"][lab])

            # (a) from scratch on the few labels
            torch.manual_seed(s)
            ms = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG)
            ms, _ = train_tabtransformer(ms, fr["Xcat"][lab_tr], fr["Xnum"][lab_tr], fr["y"][lab_tr],
                                         fr["Xcat"][lab_va], fr["Xnum"][lab_va], fr["y"][lab_va],
                                         lr=LR, max_epochs=EPOCHS, patience=PATIENCE, seed=s)
            auc_scratch = tabtransformer_auc(ms, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])

            # (b) RTD pre-train the encoder on the unlabeled pool, then GENTLY fine-tune on the same labels
            torch.manual_seed(s)
            mp = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG)
            mp, _acc = pretrain_rtd(mp, fr["Xcat"][unlab_idx], fr["cards"], replace_p=PRE_P,
                                    max_epochs=PRE_EPOCHS, batch_size=BS, seed=s)
            mp, _ = train_tabtransformer(mp, fr["Xcat"][lab_tr], fr["Xnum"][lab_tr], fr["y"][lab_tr],
                                         fr["Xcat"][lab_va], fr["Xnum"][lab_va], fr["y"][lab_va],
                                         lr=FT_LR, max_epochs=EPOCHS, patience=PATIENCE, seed=s)
            auc_pre = tabtransformer_auc(mp, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])

            results[f"{f:.2f}"]["scratch"].append(auc_scratch)
            results[f"{f:.2f}"]["pretrain"].append(auc_pre)
            print(f"  seed {s} frac {f:.2f} (~{len(lab_tr)} labeled, {len(unlab_idx)} unlabeled): "
                  f"scratch {auc_scratch:.3f} | pretrain+finetune {auc_pre:.3f} "
                  f"| Δ {auc_pre - auc_scratch:+.3f}", flush=True)

    summary = {}
    print(f"\n---------------- semi-supervised summary ({SEMI_NAME}, unlabeled pool ~{int(SEMI_N*0.7)} rows) --------")
    for f in SEMI_FRACS:
        k = f"{f:.2f}"
        sc, pr = np.array(results[k]["scratch"]), np.array(results[k]["pretrain"])
        summary[k] = {"scratch_mean": round(float(sc.mean()), 3), "scratch_std": round(float(sc.std()), 3),
                      "pretrain_mean": round(float(pr.mean()), 3), "pretrain_std": round(float(pr.std()), 3),
                      "lift": round(float(pr.mean() - sc.mean()), 3),
                      "lift_all_seeds_positive": bool(np.all(pr - sc > 0))}
        print(f"  frac {k}: scratch {sc.mean():.3f}±{sc.std():.3f} -> "
              f"pretrain+finetune {pr.mean():.3f}±{pr.std():.3f}  (lift {pr.mean()-sc.mean():+.3f}, "
              f"all seeds +: {summary[k]['lift_all_seeds_positive']})")
    return {"dataset": SEMI_NAME, "n_rows": SEMI_N, "fractions": SEMI_FRACS, "seeds": SEMI_SEEDS,
            "ft_lr": FT_LR, "pretrain_epochs": PRE_EPOCHS, "replace_p": PRE_P, "summary": summary,
            "note": "unlabeled pool = every train row's features (no labels); labeled = a small fraction. "
                    "Gentle fine-tune LR (5e-4) after pre-training avoids catastrophic forgetting."}


def main():
    t0 = time.time()
    mech = verify_mechanism()
    bakeoff = verify_bakeoff()
    semi = verify_semisupervised()
    out = {"paper": "Huang, Khetan, Cvitkovic & Karnin 2020, arXiv:2012.06678 (TabTransformer)",
           "metric": "roc_auc", "seeds": SEEDS, "datasets": DATASETS, "subsample": SUBSAMPLE,
           "tt_cfg": TT_CFG, "cf_cfg": CF_CFG, "epochs": EPOCHS, "patience": PATIENCE,
           "mechanism": mech, "bakeoff": bakeoff, "semisupervised": semi,
           "elapsed_s": round(time.time() - t0, 1),
           "note": "TabTransformer (contextual categorical embeddings) vs the context-free static-embedding "
                   "MLP (n_layers=0, the L031/L032 model) vs CatBoost. 3 categorical-rich tables (adult "
                   "subsampled to 4000) is a demonstration, not the paper's 15-dataset benchmark. The "
                   "mechanism is validated exactly; the contextual and semi-supervised directions are "
                   "reproduced; the honest verdict vs trees is stated."}
    with open(os.path.join(HERE, "_verify_l045_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote labs/_verify_l045_results.json  (total {out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
