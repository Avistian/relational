"""Verify Lesson 031 claims — encoding categoricals: taxonomy, target-encoding leakage, entity embeddings.

Lesson 031 (Y1 Q4 opener) teaches four ways to turn a categorical column into numbers a model can eat:
one-hot, ordinal, target (mean) encoding, and learned entity embeddings (Guo & Berkhahn 2016). The two
verifiable mechanisms (torch-free, real credit_g Tier-A):

  1. THE ENCODING BAKE-OFF (honest, out-of-fold everywhere).
     Compare one-hot / ordinal / out-of-fold(smoothed) target encoding on credit_g, same CV folds, two
     model families (a linear model that CANNOT split a false order, and a GBDT that can). Shows:
       - ordinal encoding hurts a LINEAR model (invents a false order it must honour) but a GBDT shrugs
         it off (axis-aligned splits recover the grouping);
       - one-hot and honest target encoding are both fine; target encoding is far more compact.

  2. TARGET-ENCODING LEAKAGE (the pitfall the lesson is named for).
     NAIVE target encoding computes each level's mean target INCLUDING the row being encoded. On a
     high-cardinality / near-unique column this memorises the label: train AUC -> ~1.0, test AUC -> chance.
     OUT-OF-FOLD (K-fold) + smoothing removes the row's own label -> honest, no train/test gap.
     We show it on (a) a synthetic near-unique id column, and (b) real credit_g `purpose`.

  Also dumps per-level target means (for the taxonomy viz) and the single-row leak example
  (for the target-leak viz).

Run: .venv/bin/python labs/_verify_l031.py
"""
from __future__ import annotations

import os
# Cap OpenMP threads BEFORE importing numpy/sklearn/torch. On this aarch64/WSL2 box the torch install
# pulls in a libgomp that makes sklearn's HistGradientBoosting oversubscribe threads (~21s/fit vs 0.3s).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relkit import load_tier_a

SEED = 0
K = 5


# ---------------------------------------------------------------------------
# Out-of-fold, smoothed target encoding (the honest recipe; Micci-Barreca 2001).
# For each fold, encode its rows using level means computed ONLY on the other folds.
# Smoothing shrinks a level's mean toward the global mean by its count:
#     enc(level) = (n_L * mean_L + m * global) / (n_L + m)
# so rare levels (small n_L) fall back to the prior instead of memorising few rows.
# ---------------------------------------------------------------------------
def oof_target_encode(col: pd.Series, y: np.ndarray, m: float = 20.0, seed: int = SEED) -> np.ndarray:
    col = col.astype(str).to_numpy()
    y = np.asarray(y)
    out = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, va in skf.split(col, y):
        glob = y[tr].mean()
        df = pd.DataFrame({"c": col[tr], "y": y[tr]})
        agg = df.groupby("c")["y"].agg(["mean", "count"])
        smooth = (agg["count"] * agg["mean"] + m * glob) / (agg["count"] + m)
        mapping = smooth.to_dict()
        out[va] = [mapping.get(c, glob) for c in col[va]]
    return out


def naive_target_encode(col: pd.Series, y: np.ndarray) -> np.ndarray:
    """LEAKY: each level's mean includes the row itself (fit on ALL rows, applied to all rows)."""
    col = col.astype(str).to_numpy()
    y = np.asarray(y)
    df = pd.DataFrame({"c": col, "y": y})
    mapping = df.groupby("c")["y"].mean().to_dict()
    return np.array([mapping[c] for c in col], dtype=float)


def cv_auc_matrix(Xnum: np.ndarray, y: np.ndarray, model: str, seed: int = SEED) -> float:
    """Mean test AUC over K folds for a fully-numeric design matrix."""
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Xnum, y):
        if model == "linear":
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        else:
            clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(Xnum[tr], y[tr])
        p = clf.predict_proba(Xnum[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def build_matrix(X: pd.DataFrame, y: np.ndarray, num, cat, scheme: str) -> np.ndarray:
    """Encode categoricals by `scheme`, keep numerics as-is; return a dense float matrix.

    Target encodings are computed OUT-OF-FOLD on the WHOLE dataset here purely to build a single
    design matrix for the bake-off; the per-fold CV below re-selects folds, so the OOF target
    encoding is (mildly) optimistic but leak-safe within each level. The point of the bake-off is the
    RELATIVE ordering across schemes, verified stable across seeds.
    """
    Xn = X[num].to_numpy(dtype=float) if num else np.empty((len(X), 0))
    if scheme == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xc = enc.fit_transform(X[cat])
    elif scheme == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xc = enc.fit_transform(X[cat].astype(str))
    elif scheme == "target_oof":
        Xc = np.column_stack([oof_target_encode(X[c], y) for c in cat])
    else:
        raise ValueError(scheme)
    return np.column_stack([Xn, Xc])


def main():
    X, y = load_tier_a("credit_g")
    y = np.asarray(y)
    cat = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    print(f"credit_g: {X.shape[0]} rows, {len(num)} numeric, {len(cat)} categorical; "
          f"P(good)={y.mean():.3f}")
    print(f"categoricals: {cat}")
    tot_levels = sum(X[c].astype(str).nunique() for c in cat)
    print(f"total categorical levels: {tot_levels}  (one-hot width = {tot_levels} extra columns)")

    out = {"dataset": "credit_g", "n": int(X.shape[0]), "p_good": round(float(y.mean()), 3)}

    # -----------------------------------------------------------------
    # 1. Encoding bake-off: {onehot, ordinal, target_oof} x {linear, gbdt}
    # -----------------------------------------------------------------
    print("\n=== 1. ENCODING BAKE-OFF (mean test AUC over 5 folds, 3 seeds) ===")
    print(f"{'scheme':<12} {'#cols':>6} {'linear':>9} {'gbdt':>9}")
    bake = {}
    widths = {"onehot": None, "ordinal": len(num) + len(cat), "target_oof": len(num) + len(cat)}
    for scheme in ["onehot", "ordinal", "target_oof"]:
        lin_s, gb_s, width = [], [], None
        for seed in [0, 1, 2]:
            M = build_matrix(X, y, num, cat, scheme)
            width = M.shape[1]
            lin_s.append(cv_auc_matrix(M, y, "linear", seed))
            gb_s.append(cv_auc_matrix(M, y, "gbdt", seed))
        bake[scheme] = {
            "cols": int(width),
            "linear": round(float(np.mean(lin_s)), 3),
            "linear_sd": round(float(np.std(lin_s)), 3),
            "gbdt": round(float(np.mean(gb_s)), 3),
            "gbdt_sd": round(float(np.std(gb_s)), 3),
        }
        print(f"{scheme:<12} {width:>6} {bake[scheme]['linear']:>9.3f} {bake[scheme]['gbdt']:>9.3f}")
    out["bakeoff"] = bake

    # -----------------------------------------------------------------
    # 2. Target-encoding LEAKAGE — synthetic near-unique id column
    # -----------------------------------------------------------------
    print("\n=== 2a. TARGET-ENCODING LEAK — synthetic near-unique id ===")
    rng = np.random.default_rng(SEED)
    # id with ~2 rows per level -> naive mean is basically the row's own label
    n = len(y)
    ids = pd.Series((np.arange(n) // 2).astype(str))  # 500 levels, 2 rows each
    naive = naive_target_encode(ids, y).reshape(-1, 1)
    oof = oof_target_encode(ids, y).reshape(-1, 1)
    leak = {}
    for name, feat in [("naive", naive), ("oof", oof)]:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
        tr_auc, te_auc = [], []
        for tr, te in skf.split(feat, y):
            clf = HistGradientBoostingClassifier(random_state=SEED)
            clf.fit(feat[tr], y[tr])
            tr_auc.append(roc_auc_score(y[tr], clf.predict_proba(feat[tr])[:, 1]))
            te_auc.append(roc_auc_score(y[te], clf.predict_proba(feat[te])[:, 1]))
        leak[name] = {"train_auc": round(float(np.mean(tr_auc)), 3),
                      "test_auc": round(float(np.mean(te_auc)), 3)}
        print(f"{name:<6} id-only feature: train AUC {leak[name]['train_auc']:.3f}  "
              f"test AUC {leak[name]['test_auc']:.3f}")
    out["leak_synthetic"] = leak

    # -----------------------------------------------------------------
    # 2b. Target-encoding leak on a REAL credit_g column (purpose)
    # -----------------------------------------------------------------
    print("\n=== 2b. TARGET-ENCODING LEAK — real credit_g 'purpose' as the ONLY feature ===")
    colname = "purpose" if "purpose" in cat else cat[0]
    col = X[colname]
    real = {}
    for name, feat in [("naive", naive_target_encode(col, y).reshape(-1, 1)),
                       ("oof", oof_target_encode(col, y).reshape(-1, 1))]:
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
        tr_auc, te_auc = [], []
        for tr, te in skf.split(feat, y):
            clf = HistGradientBoostingClassifier(random_state=SEED)
            clf.fit(feat[tr], y[tr])
            tr_auc.append(roc_auc_score(y[tr], clf.predict_proba(feat[tr])[:, 1]))
            te_auc.append(roc_auc_score(y[te], clf.predict_proba(feat[te])[:, 1]))
        real[name] = {"train_auc": round(float(np.mean(tr_auc)), 3),
                      "test_auc": round(float(np.mean(te_auc)), 3)}
        print(f"{name:<6} {colname}-only: train AUC {real[name]['train_auc']:.3f}  "
              f"test AUC {real[name]['test_auc']:.3f}")
    out["leak_real"] = {"column": colname, **real}

    # -----------------------------------------------------------------
    # 3. Per-level target means for the taxonomy viz (checking_status)
    # -----------------------------------------------------------------
    print("\n=== 3. PER-LEVEL TARGET MEANS (taxonomy viz) ===")
    tax_col = "checking_status" if "checking_status" in cat else cat[0]
    glob = y.mean()
    dfc = pd.DataFrame({"c": X[tax_col].astype(str).to_numpy(), "y": y})
    agg = dfc.groupby("c")["y"].agg(["mean", "count"]).sort_values("mean")
    m = 20.0
    levels = []
    for lvl, row in agg.iterrows():
        smooth = (row["count"] * row["mean"] + m * glob) / (row["count"] + m)
        levels.append({"level": lvl, "count": int(row["count"]),
                       "raw_mean": round(float(row["mean"]), 3),
                       "smoothed": round(float(smooth), 3)})
        print(f"  {lvl:<28} n={int(row['count']):>4}  raw P(good)={row['mean']:.3f}  "
              f"smoothed={smooth:.3f}")
    out["taxonomy"] = {"column": tax_col, "global_mean": round(float(glob), 3), "levels": levels}

    # -----------------------------------------------------------------
    # 4. Single-row leak example for the target-leak viz
    # -----------------------------------------------------------------
    # Pick a rare level and one of its rows; show naive mean (incl. self) vs leave-one-out mean.
    print("\n=== 4. SINGLE-ROW LEAK EXAMPLE ===")
    rare = agg.index[0]  # lowest-mean level
    mask = X[tax_col].astype(str).to_numpy() == rare
    idxs = np.where(mask)[0]
    r = idxs[0]
    lvl_y = y[mask]
    naive_mean = lvl_y.mean()
    loo_mean = (lvl_y.sum() - y[r]) / (len(lvl_y) - 1)
    out["single_row"] = {
        "column": tax_col, "level": rare, "n_level": int(mask.sum()),
        "row_label": int(y[r]), "naive_incl_self": round(float(naive_mean), 3),
        "loo_excl_self": round(float(loo_mean), 3), "global_mean": round(float(glob), 3),
    }
    print(f"  level={rare!r} (n={int(mask.sum())}), row label={int(y[r])}: "
          f"naive mean(incl self)={naive_mean:.3f}  LOO mean(excl self)={loo_mean:.3f}")

    # -----------------------------------------------------------------
    # 5. Entity embeddings (Guo & Berkhahn 2016) — real, learned end-to-end (torch)
    # -----------------------------------------------------------------
    print("\n=== 5. ENTITY EMBEDDINGS (learned NN, credit_g) ===")
    embed_col = "purpose" if "purpose" in cat else max(cat, key=lambda c: X[c].astype(str).nunique())
    emb = entity_embedding_experiment(X, y, num, cat, tax_col=embed_col)
    out["embeddings"] = emb
    print(f"  entity-embedding MLP test AUC: {emb['test_auc']:.3f}±{emb['test_sd']:.3f} "
          f"(fair one-hot MLP {emb['onehot_auc']:.3f}±{emb['onehot_sd']:.3f}) — 3 splits")
    print(f"  learned {emb['column']} embedding: dim={emb['dim']}, "
          f"{len(emb['levels'])} levels -> PCA 2-D dumped for viz")
    for lv in emb["levels"]:
        print(f"    {lv['level']:<24} risk(P good)={lv['target_mean']:.3f}  "
              f"xy=({lv['x']:+.2f},{lv['y']:+.2f})")

    path = Path(__file__).resolve().parent / "html" / "_l031_numbers.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


def entity_embedding_experiment(X, y, num, cat, tax_col, seed: int = SEED):
    """Train an entity-embedding MLP on credit_g; return test AUC + a learned 2-D embedding.

    Architecture (Guo & Berkhahn 2016): one trainable embedding table per categorical column
    (dim = min(50, (cardinality+1)//2)), concatenated with standardised numerics, fed to a small MLP.
    The embeddings are learned by backprop jointly with the classifier — similar categories (by their
    effect on the target) end up near each other in embedding space. We extract the `tax_col` table,
    PCA it to 2-D, and report each level's coordinates + its target mean so the viz can show clustering.
    """
    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA

    y = np.asarray(y)

    # integer-code each categorical (0..card-1); numerics standardised inside the split
    codes, cards = {}, {}
    for c in cat:
        cats = sorted(X[c].astype(str).unique())
        cards[c] = len(cats)
        lut = {v: i for i, v in enumerate(cats)}
        codes[c] = X[c].astype(str).map(lut).to_numpy()
    Xcode = np.column_stack([codes[c] for c in cat]).astype(np.int64)
    Xnum = X[num].to_numpy(dtype=float) if num else np.zeros((len(X), 0))
    dims = {c: min(50, (cards[c] + 1) // 2) for c in cat}

    class EmbMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(cards[c], dims[c]) for c in cat])
            in_dim = sum(dims.values()) + len(num)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, xc, xn):
            e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
            h = torch.cat(e + ([xn] if xn.shape[1] else []), dim=1)
            return self.mlp(h).squeeze(1)

    def train_one(seed_i):
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.25, random_state=seed_i, stratify=y)
        sc = StandardScaler().fit(Xnum[tr])
        Xnum_s = sc.transform(Xnum) if num else Xnum
        model = EmbMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        xc_tr = torch.tensor(Xcode[tr]); xn_tr = torch.tensor(Xnum_s[tr], dtype=torch.float32)
        yt_tr = torch.tensor(y[tr], dtype=torch.float32)
        xc_te = torch.tensor(Xcode[te]); xn_te = torch.tensor(Xnum_s[te], dtype=torch.float32)
        model.train()
        for _ in range(120):
            opt.zero_grad()
            loss = lossf(model(xc_tr, xn_tr), yt_tr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(xc_te, xn_te)).numpy()
        emb_auc = float(roc_auc_score(y[te], p))
        # FAIR one-hot MLP baseline: same split, early stopping so it isn't undertrained (L028 lesson)
        pre = ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
        oh = make_pipeline(pre, MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                                              early_stopping=True, n_iter_no_change=20,
                                              random_state=seed_i))
        oh.fit(X.iloc[tr], y[tr])
        oh_auc = float(roc_auc_score(y[te], oh.predict_proba(X.iloc[te])[:, 1]))
        return emb_auc, oh_auc, model

    emb_aucs, oh_aucs = [], []
    model = None
    for s in [0, 1, 2]:
        ea, oa, m = train_one(s)
        emb_aucs.append(ea); oh_aucs.append(oa)
        if s == 0:
            model = m  # keep seed-0 model for the embedding extraction
    test_auc = float(np.mean(emb_aucs))
    test_sd = float(np.std(emb_aucs))
    onehot_auc = float(np.mean(oh_aucs))
    onehot_sd = float(np.std(oh_aucs))

    # extract the learned embedding table for tax_col (seed-0 model), PCA -> 2-D
    ci = cat.index(tax_col)
    W = model.embs[ci].weight.detach().numpy()
    cats = sorted(X[tax_col].astype(str).unique())
    if W.shape[1] >= 2:
        xy = PCA(n_components=2, random_state=seed).fit_transform(W)
    else:
        xy = np.column_stack([W[:, 0], np.zeros(len(W))])
    glob = y.mean()
    dfc = pd.DataFrame({"c": X[tax_col].astype(str).to_numpy(), "y": y})
    means = dfc.groupby("c")["y"].mean().to_dict()
    counts = dfc.groupby("c")["y"].count().to_dict()
    levels = []
    for i, lvl in enumerate(cats):
        levels.append({"level": lvl, "x": round(float(xy[i, 0]), 3),
                       "y": round(float(xy[i, 1]), 3),
                       "target_mean": round(float(means.get(lvl, glob)), 3),
                       "count": int(counts.get(lvl, 0))})
    return {"column": tax_col, "dim": int(dims[tax_col]),
            "test_auc": round(test_auc, 3), "test_sd": round(test_sd, 3),
            "onehot_auc": round(onehot_auc, 3), "onehot_sd": round(onehot_sd, 3),
            "levels": levels}


if __name__ == "__main__":
    main()
