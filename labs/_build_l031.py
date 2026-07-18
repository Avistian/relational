"""Build Lab 031 (Embeddings for categoricals — encode four ways, break the leak, fix it) — student + solution.

Tier A (real credit_g via relkit). Three tasks + a torch stretch:
  * Task 1 (crucial fragment) — the smoothed OUT-OF-FOLD target encoder: encode each fold from the
    OTHER folds, shrinking each level's mean toward the global prior by its count.
  * Task 2 — the encoding bake-off: one-hot vs ordinal vs OOF-target, scored by a linear model and a
    GBDT. Reproduces: ordinal wrecks the linear model (false order) but not the tree; OOF target matches
    one-hot at a third of the width.
  * Task 3 — the leak: naive target encoding of a near-unique id scores ~0.89 AUC on pure noise; the
    out-of-fold encoder from Task 1 restores it to ~0.50 (chance).
  * Stretch (ungraded) — a real PyTorch entity-embedding net (torch is installed in the lab venv).

Run: .venv/bin/python labs/_build_l031.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0031-embeddings-for-categoricals.ipynb labs/solutions/0031-embeddings-for-categoricals.ipynb
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from _colab import bootstrap_cells  # noqa: E402


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


SETUP = r'''# PROVIDED — imports, real credit_g (Tier A), column types, and scoring helpers. Just run.
import os
# Cap OpenMP threads before sklearn imports (the torch install can make HistGB oversubscribe on some boxes).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 0
K = 5
np.random.seed(SEED)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
X, y = load_tier_a("credit_g")
y = np.asarray(y)
NUM = X.select_dtypes(include="number").columns.tolist()
CAT = [c for c in X.columns if c not in NUM]
GLOBAL = y.mean()
print(f"credit_g (Tier A): X {X.shape}, P(good) {GLOBAL:.3f}, {len(NUM)} numeric + {len(CAT)} categorical cols")
print(f"total categorical levels: {sum(X[c].astype(str).nunique() for c in CAT)} "
      f"(that many one-hot columns)")

def cv_auc(feat, model, seed=SEED):
    """Mean test ROC-AUC over K stratified folds for a numeric design matrix `feat`."""
    feat = np.asarray(feat, dtype=float)
    if feat.ndim == 1:
        feat = feat.reshape(-1, 1)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(feat, y):
        clf = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
               if model == "linear" else HistGradientBoostingClassifier(random_state=seed))
        clf.fit(feat[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(feat[te])[:, 1]))
    return float(np.mean(aucs))

print("ready — helper: cv_auc(feature_matrix, 'linear'|'gbdt')")'''


# ---- Task 1: smoothed out-of-fold target encoder ----
T1_MD = r'''## Task 1 — the smoothed out-of-fold target encoder (crucial fragment) — TODO

**Goal:** write the leak-free target encoder. For each fold, encode its rows using level means computed
**only from the other folds**, and **shrink** each level's mean toward the global prior by its count.

**Why it matters:** target encoding is built *from the label*, so it leaks unless a row is never encoded
with its own label (Task 3 shows the damage). Shrinkage stops a rare level — seen two or three times —
from memorising those few rows. This is the same "fit on data you don't score" discipline as
cross-validation (L003) and leakage-safe pipelines (L005), now applied to a *feature*.

**Hint boundary:** you fill the **smoothing formula** only. Given a level's `mean` and `count` (from the
*other* folds) and the fold's `glob` prior, the shrunk value is
`(count·mean + m·glob) / (count + m)` — big levels keep their mean, tiny levels fall back to `glob`.'''

T1_CODE = r'''# TODO — fill the smoothing formula (the crucial fragment)
def oof_target_encode(col, y, n_splits=K, m=20.0, seed=SEED):
    col = col.astype(str).to_numpy()
    y = np.asarray(y)
    out = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(col, y):
        glob = y[tr].mean()                                    # prior from the OTHER folds only
        agg = pd.DataFrame({"c": col[tr], "y": y[tr]}).groupby("c")["y"].agg(["mean", "count"])
        shrunk = ____                                          # (count*mean + m*glob) / (count + m)
        mapping = shrunk.to_dict()
        out[va] = [mapping.get(c, glob) for c in col[va]]      # unseen level -> prior
    return out

te_check = oof_target_encode(X["checking_status"], y)
print("checking_status OOF-encoded -> values in "
      f"[{te_check.min():.3f}, {te_check.max():.3f}], single-feature AUC {cv_auc(te_check, 'gbdt'):.3f}")'''

T1_SOL = T1_CODE.replace(
    "        shrunk = ____                                          # (count*mean + m*glob) / (count + m)",
    "        shrunk = (agg['count'] * agg['mean'] + m * glob) / (agg['count'] + m)")

T1_CHECK = r'''# CHECK — do not edit
def _ref_oof(col, y, n_splits=K, m=20.0, seed=SEED):
    col = col.astype(str).to_numpy(); y = np.asarray(y); out = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(col, y):
        glob = y[tr].mean()
        agg = pd.DataFrame({"c": col[tr], "y": y[tr]}).groupby("c")["y"].agg(["mean", "count"])
        s = (agg["count"] * agg["mean"] + m * glob) / (agg["count"] + m)
        mp = s.to_dict(); out[va] = [mp.get(c, glob) for c in col[va]]
    return out
_ref = _ref_oof(X["checking_status"], y)
assert np.allclose(te_check, _ref), "Smoothing formula is off — expected (count*mean + m*glob)/(count + m)."
assert te_check.min() >= 0.0 and te_check.max() <= 1.0, "Encoded values are probabilities in [0,1]."
assert cv_auc(te_check, "gbdt") > 0.60, "checking_status is predictive; its OOF encoding should beat chance."
print("Task 1 ok — out-of-fold, smoothed target encoder works. Each row is encoded from OTHER folds only.")'''


# ---- Task 2: the encoding bake-off ----
T2_MD = r'''## Task 2 — the encoding bake-off — TODO

**Goal:** encode *all* categoricals three ways — one-hot, ordinal, and your out-of-fold target encoder —
and score each with a **linear** model and a **GBDT**. You fill the target-encoding branch (reuse Task 1).

**Why it matters:** the right encoding depends on the model. You will reproduce the two headline effects:
ordinal encoding **wrecks the linear model** (it must honour a fabricated order) but barely dents the tree
(axis-aligned splits carve the integer axis back into groups); and OOF target encoding **matches one-hot
at a third of the width** — the compactness that becomes decisive at high cardinality.

**Hint boundary:** in the `"target"` branch, stack one out-of-fold-encoded column per categorical:
`np.column_stack([oof_target_encode(X[c], y) for c in CAT])`.'''

T2_CODE = r'''# TODO — build the target-encoded matrix (one OOF column per categorical)
Xn = X[NUM].to_numpy(dtype=float)

def design_matrix(scheme):
    if scheme == "onehot":
        Xc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(X[CAT])
    elif scheme == "ordinal":
        Xc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[CAT].astype(str))
    elif scheme == "target":
        Xc = ____                                             # np.column_stack of oof_target_encode per CAT col
    return np.column_stack([Xn, Xc])

results = {}
for scheme in ["onehot", "ordinal", "target"]:
    M = design_matrix(scheme)
    results[scheme] = {"cols": M.shape[1], "linear": cv_auc(M, "linear"), "gbdt": cv_auc(M, "gbdt")}
    r = results[scheme]
    print(f"{scheme:<8} cols={r['cols']:>3}  linear={r['linear']:.3f}  gbdt={r['gbdt']:.3f}")'''

T2_SOL = T2_CODE.replace(
    '        Xc = ____                                             # np.column_stack of oof_target_encode per CAT col',
    '        Xc = np.column_stack([oof_target_encode(X[c], y) for c in CAT])')

T2_CHECK = r'''# CHECK — do not edit
assert results["target"]["cols"] < results["onehot"]["cols"], "Target encoding should be far more compact than one-hot."
assert results["ordinal"]["linear"] < results["onehot"]["linear"] - 0.015, \
    "Ordinal's false order should measurably hurt the LINEAR model vs one-hot."
assert results["ordinal"]["gbdt"] > results["ordinal"]["linear"], \
    "The GBDT should shrug off the false order that hurts the linear model."
assert abs(results["target"]["linear"] - results["onehot"]["linear"]) < 0.05, \
    "OOF target encoding should roughly TIE one-hot for the linear model, at a fraction of the width."
print("Task 2 ok — ordinal hurts the linear model (false order), the tree shrugs, and OOF target "
      f"ties one-hot ({results['target']['cols']} cols vs {results['onehot']['cols']}).")'''


# ---- Task 3: the leak ----
T3_MD = r'''## Task 3 — break it: target-encoding leakage on a near-unique id — TODO

**Goal:** manufacture a `customer_id`-style column with ~2 rows per level and **no real signal**, then
target-encode it two ways: **naively** (each level's mean over *all* rows, computed before any split —
so it includes each row's own label) and **out-of-fold** (Task 1). Feed each, alone, to a GBDT.

**Why it matters:** this is the pitfall the lesson is named for. The naive encoding of a near-unique id
is a copy of the label, so a *content-free* column scores a huge AUC that vanishes in production. The
out-of-fold encoder makes each unique id fall back to the prior — no information, the honest truth.

**Hint boundary:** fill the **naive** encoding — map each id to the mean of `y` over *all* its rows
(pandas `groupby(...).transform("mean")` computes exactly that, including the current row).'''

T3_CODE = r'''# TODO — implement the NAIVE (leaky) encoding, then compare to out-of-fold
ids = pd.Series((np.arange(len(y)) // 2).astype(str))         # ~2 rows per level; carries NO real signal

naive_id = ____                                               # each id's mean(y) over ALL its rows (incl. self)
oof_id = oof_target_encode(ids, y)                            # honest: encoded from other folds only

naive_auc = cv_auc(naive_id, "gbdt")
oof_auc = cv_auc(oof_id, "gbdt")
print(f"naive (leaky)  id-only AUC: {naive_auc:.3f}   <- looks brilliant, it's a mirage")
print(f"out-of-fold    id-only AUC: {oof_auc:.3f}   <- chance, the honest truth")'''

T3_SOL = T3_CODE.replace(
    '''naive_id = ____                                               # each id's mean(y) over ALL its rows (incl. self)''',
    '''naive_id = pd.DataFrame({"c": ids, "y": y}).groupby("c")["y"].transform("mean").to_numpy()''')

T3_CHECK = r'''# CHECK — do not edit
assert naive_auc > 0.80, "Naive target encoding of a near-unique id should leak the label (huge fake AUC)."
assert oof_auc < 0.60, "Out-of-fold encoding of a signal-free id should sit near chance (~0.5)."
assert naive_auc - oof_auc > 0.25, "The leak gap (naive - oof) should be large and obvious."
print(f"Task 3 ok — the leak is real: naive {naive_auc:.3f} vs out-of-fold {oof_auc:.3f} on a column with "
      f"NO signal. Always encode targets out-of-fold, fit inside the CV fold.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** print your deliverable — the bake-off, the leak gap — and a one-line takeaway.

**Takeaway prompt:** in one sentence, why does ordinal encoding hurt a linear model but not a tree, why
must target encoding be done out-of-fold, and what do entity embeddings add that these encodings cannot?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 031 (embeddings for categoricals) ===")
print(f"bake-off (linear): onehot {results['onehot']['linear']:.3f} | ordinal {results['ordinal']['linear']:.3f} "
      f"| target {results['target']['linear']:.3f}  (cols: {results['onehot']['cols']} vs {results['target']['cols']})")
print(f"bake-off (gbdt)  : onehot {results['onehot']['gbdt']:.3f} | ordinal {results['ordinal']['gbdt']:.3f} "
      f"| target {results['target']['gbdt']:.3f}")
print(f"target-encoding leak: naive {naive_auc:.3f} -> out-of-fold {oof_auc:.3f} (signal-free id)")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"Ordinal encoding invents a false order a linear model is forced to honour (so it drops) while a '
    'tree splits the integer axis back into the true groups; target encoding must be computed out-of-fold '
    '(and smoothed) because it is built from the label and otherwise copies the answer -- a signal-free id '
    'leaked 0.89 AUC that out-of-fold restored to 0.50; and entity embeddings generalise target encoding '
    'to a learned d-dimensional vector per level that captures similarity (the 1-D case is target '
    'encoding), the first learned representation and the bridge to embedding the foreign-key entities of a '
    'relational database."')


TORCH_STRETCH = r'''# STRETCH (optional, ungraded) — a REAL entity-embedding net (torch is installed in the lab venv).
# Uncomment to run: trains an embedding table per categorical, prints test AUC, and extracts a learned
# 2-D projection of the `purpose` embedding. On credit_g it ~TIES a fair one-hot MLP (small flat table).
#
# import torch, torch.nn as nn
# torch.set_num_threads(1)
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# codes, cards = {}, {}
# for c in CAT:
#     cats = sorted(X[c].astype(str).unique()); cards[c] = len(cats)
#     lut = {v: i for i, v in enumerate(cats)}; codes[c] = X[c].astype(str).map(lut).to_numpy()
# Xcode = np.column_stack([codes[c] for c in CAT]).astype(np.int64)
# Xnum = X[NUM].to_numpy(float)
# dims = {c: min(50, (cards[c] + 1) // 2) for c in CAT}   # Guo & Berkhahn / fast.ai heuristic
# tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=SEED, stratify=y)
# mu, sd = Xnum[tr].mean(0), Xnum[tr].std(0) + 1e-9; Xs = (Xnum - mu) / sd
# class EmbMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embs = nn.ModuleList([nn.Embedding(cards[c], dims[c]) for c in CAT])
#         d = sum(dims.values()) + len(NUM)
#         self.mlp = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Dropout(0.4),
#                                  nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))
#     def forward(self, xc, xn):
#         e = [emb(xc[:,i]) for i,emb in enumerate(self.embs)]
#         return self.mlp(torch.cat(e + [xn], 1)).squeeze(1)
# torch.manual_seed(SEED)
# net = EmbMLP(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
# lossf = nn.BCEWithLogitsLoss()
# xc_tr, xn_tr = torch.tensor(Xcode[tr]), torch.tensor(Xs[tr], dtype=torch.float32)
# yt = torch.tensor(y[tr], dtype=torch.float32)
# net.train()
# for _ in range(120):
#     opt.zero_grad(); lossf(net(xc_tr, xn_tr), yt).backward(); opt.step()
# net.eval()
# with torch.no_grad():
#     p = torch.sigmoid(net(torch.tensor(Xcode[te]), torch.tensor(Xs[te], dtype=torch.float32))).numpy()
# print("entity-embedding MLP test AUC:", round(roc_auc_score(y[te], p), 3), "(expect ~0.77, a tie with one-hot)")
# W = net.embs[CAT.index("purpose")].weight.detach().numpy()
# xy = PCA(n_components=2).fit_transform(W)
# for lvl, (a,b) in zip(sorted(X["purpose"].astype(str).unique()), xy):
#     print(f"  {lvl:<20} ({a:+.2f}, {b:+.2f})")
print("Stretch cell present — uncomment to train a real entity-embedding net (needs torch).")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 031 — Embeddings for categoricals: encode four ways, break the leak, fix it

**Lesson:** [`lessons/0031-embeddings-for-categoricals.html`](../lessons/0031-embeddings-for-categoricals.html) · **Phase / Year:** Year 1 · Q4 (opener)

**Primary reading:** Guo & Berkhahn 2016, *Entity Embeddings of Categorical Variables* ([arXiv:1604.06737](https://arxiv.org/abs/1604.06737)) — §2–4.

**Dataset tier:** **A** (real `credit_g` via `relkit`).

**Skill you are practising:** the four ways to encode a categorical (one-hot, ordinal, target, entity embedding), why ordinal fakes an order, why target encoding **leaks** unless done out-of-fold, and how a learned embedding generalises target encoding into a similarity-capturing representation.

**Exit criteria:** EXIT TICKET prints the encoding bake-off (one-hot vs ordinal vs OOF-target, linear vs GBDT), the target-encoding leak gap (naive vs out-of-fold on a signal-free id), and your one-line takeaway.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, column types, scoring helper); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs scikit-learn (in `requirements-labs.txt`); the optional stretch needs **torch** (now in the lab venv). `credit_g` is fetched from OpenML on first run then cached. Whole lab runs in ~1 minute on CPU.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — turning a categorical into numbers

A model does arithmetic on numeric vectors, so every categorical column must be **encoded** first. Four ways, each a different **representation**:

- **One-hot** — a level of a *k*-level column becomes *k* binary columns. Asserts nothing false (all levels equidistant), but wide, and captures no similarity.
- **Ordinal (label)** — each level becomes an integer `0,1,2,…`. Compact, but integers carry an **order and spacing**, so it fabricates a **false order**. A linear/neural model must obey it; a tree can split around it.
- **Target (mean) encoding** — each level becomes the **mean of the target** for that level, `P(y | level)`. One dense column, ordered by risk, scales to any cardinality (Micci-Barreca 2001). But it is built **from the label**, so it leaks unless computed carefully.
- **Entity embedding** — each level becomes a short **learned dense vector**, trained by backprop with the model (Guo & Berkhahn 2016). Similar levels move close together — a similarity structure the others lack. Target encoding is the 1-D special case.

**The target-encoding fix.** Encode **out-of-fold**: split the training rows into folds and encode each fold from the *other* folds only, never a row's own label. **Smooth** each level toward the global mean by its count so rare levels don't memorise a few rows:

`enc(L) = (n_L·mean_L + m·global) / (n_L + m)`

and fit the encoder **inside** each CV fold, like any learned preprocessing (L005).

**Toy micro-example (not this lab's answer).** A `color` column `{red, red, blue}` with labels `{1, 0, 1}`. Naive target-encoding the first `red` row gives `mean(1,0)=0.5` — but that mean *includes the row's own label 1*. Encode it from the *other* rows only and `red`'s value is just `0` (the other red row), no self-leak. For a color seen **once**, the naive mean *is* its label — a perfect leak.

Full write-up + visuals: [Lesson 031](../lessons/0031-embeddings-for-categoricals.html).'''),
        md("## Setup — PROVIDED (data + column types + scoring helper)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — a real entity embedding

The three tasks were torch-free. This stretch trains an actual **entity-embedding network** (one learned
embedding table per categorical, `dim = min(50, (card+1)//2)`), reports its test AUC, and extracts a 2-D
projection of the learned `purpose` embedding — the real version of the lesson's embedding-space viz.

Expect a **tie** with a fair one-hot MLP (~0.77): on a 1000-row flat table there is little similarity or
cardinality for a learned representation to exploit. The payoff is representational — and shows up at
scale and on the high-cardinality **foreign keys** that point into other tables (the bridge to RDL).'''),
        code(TORCH_STRETCH),
    ]

    nb_obj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for c in nb_obj["cells"]:
        if isinstance(c["source"], str):
            c["source"] = c["source"].splitlines(keepends=True)
    return nb_obj


def main():
    with open(os.path.join(HERE, "0031-embeddings-for-categoricals.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0031-embeddings-for-categoricals.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0031-embeddings-for-categoricals.ipynb + solution")


if __name__ == "__main__":
    main()
