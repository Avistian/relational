"""Build Lab 044 (NODE — differentiable oblivious trees) — student + solution notebooks.

Honours NOTES standards #21 (lab ships with the lesson), #22 (build from scratch; libraries only
validate), #23 (never conclude from one dataset) and #24 (paper-mirror: implement the paper's mechanism,
use a clean head-to-head, ship a reproducibility ledger).

Tasks: (1) entmax15 by bisection; (2) the oblivious tree forward pass (entmoid split + outer-product
routing); (3) NODE vs CatBoost (both oblivious ensembles) vs the L042 baselines; (4) the cost measurement.

Run: .venv/bin/python labs/_build_l044.py
Execute the solution: .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    labs/solutions/0044-node.ipynb
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from _colab import bootstrap_cells  # noqa: E402
from relkit.paper_repro import (  # noqa: E402
    architecture_md, inline_source, notebook_scaleup_code, notebook_scaleup_md,
)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


SETUP = r'''# PROVIDED — imports, the small real tables, the shared frame, and the search spaces.
# The NODE *architecture* is inlined in a later cell so you can read it; `relkit.node` here is only
# the Task-1/Task-2 *checker* (NOTES #22 / #25). Just run this cell.
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata, friedmanchisquare
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
from relkit.nets import TabResNet, TabMLP, train_net, net_auc          # L042 baselines (from scratch)
from relkit.node import (entmax15 as relkit_entmax15,                  # checker only (NOTES #22)
                         entmoid15,                                    # Task 2 uses this operator
                         ODST as relkit_ODST)                          # checker for your forward

DEVICE = "cpu"
DATASETS = ["credit_g", "diabetes", "kc1"]   # small: NOT representative (see the #23 note below)
BUDGET, SEEDS = 3, [0, 1]                    # smaller than the lesson's run, so the lab stays interactive
EPOCHS, PATIENCE = 100, 10                   # same as labs/_verify_l044.py, so directions reproduce

def load_dense(name):
    Xdf, y = load_tier_a(name)
    num = Xdf.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xdf.columns if c not in num]
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return ct.fit_transform(Xdf).astype(np.float32), y.to_numpy().astype(np.float32)

def frame(X, y, seed):
    """The SHARED frame — identical for every model (L020 contract, L042 protocol)."""
    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25, random_state=seed,
                                          stratify=ytr_f)
    return Xtr, ytr, Xva, yva, Xte, yte

# ---- per-model search SPACES (equal BUDGET is what makes this fair — L038)
def sample_node(rng, d_in):
    return dict(in_features=d_in, num_trees=int(rng.choice([64, 128])),
                depth=int(rng.choice([3, 4, 5])), n_layers=int(rng.choice([1, 2]))), \
           dict(lr=float(rng.choice([0.005, 0.01, 0.02])))

def sample_net(kind, rng, d_in):
    opt = dict(lr=float(10 ** rng.uniform(-3.3, -2.5)), wd=float(10 ** rng.uniform(-6, -3)))
    if kind == "resnet":
        dm = int(rng.choice([64, 128, 192]))
        cfg = dict(d_in=d_in, d_main=dm, d_hidden=int(dm * rng.choice([1.0, 2.0])),
                   n_blocks=int(rng.choice([1, 2, 3])), dropout1=float(rng.uniform(0.0, 0.3)),
                   dropout2=float(rng.uniform(0.0, 0.3)))
    else:
        cfg = dict(d_in=d_in, d_block=int(rng.choice([64, 128, 256])),
                   n_blocks=int(rng.choice([1, 2, 3])), dropout=float(rng.uniform(0.0, 0.4)))
    return cfg, opt

def search_net(kind, Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        cfg, opt = sample_net(kind, rng, Xtr.shape[1])
        m = TabResNet(**cfg) if kind == "resnet" else TabMLP(**cfg)
        m, val = train_net(m, Xtr, ytr, Xva, yva, lr=opt["lr"], wd=opt["wd"],
                           max_epochs=200, patience=16, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": net_auc(m, Xte, yte)}
    return best["test"]

def search_catboost(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    """CatBoost = greedy oblivious (symmetric) trees (L016) — NODE's direct rival."""
    from catboost import CatBoostClassifier
    rng = np.random.default_rng(seed + 7)
    best = {"val": -1.0, "test": None}
    for _ in range(budget):
        clf = CatBoostClassifier(depth=int(rng.choice([4, 6, 8])),
                                 learning_rate=float(10 ** rng.uniform(-1.5, -0.7)),
                                 l2_leaf_reg=float(rng.choice([1.0, 3.0, 5.0, 9.0])),
                                 iterations=400, grow_policy="SymmetricTree", random_seed=seed,
                                 thread_count=2, verbose=0, allow_writing_files=False)
        clf.fit(Xtr, ytr.astype(int))
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]

print("setup ok — torch", torch.__version__)'''


# --------------------------------------------------------------------------- Task 1: entmax15
T1_MD = r'''## Task 1 — implement `entmax15` (the differentiable, sparse feature *choice*)

**Goal.** Write `entmax15`, the alpha = 1.5 member of the entmax family, by **bisection** on its threshold.
This is the operation NODE uses to *choose* which feature a tree level splits on.

**Why bisection.** Every alpha-entmax has the closed shape

```
p_i = [ (alpha - 1) * z_i - tau ]_+ ^ (1 / (alpha - 1))
```

for the unique `tau` that makes `p` sum to 1. There is no closed form for `tau` in general, but the sum
is monotone in `tau`, so we can **binary-search** it: pick a bracket `[tau_lo, tau_hi]`, and each step
halve it toward the `tau` whose `p` sums to exactly 1. Autograd differentiates straight through the
iterations, so no custom backward is needed.

**Why alpha = 1.5.** It is the sparse-but-gentle middle of the family: `softmax` (alpha->1) never zeros
anything, `sparsemax` (alpha=2, your L043 op) is the hardest, and `entmax15` sits between — real zeros,
smoother gradient. That is what NODE found trains best for feature selection.'''

T1_CODE = r'''# TODO — implement entmax15 by bisection along the last dimension. Fill every ____.
def entmax15(z, n_iter=30):
    """z: (..., K) -> (..., K) alpha=1.5 entmax (a sparse distribution: sums to 1, exact zeros allowed)."""
    alpha = 1.5
    z = (alpha - 1) * z                        # fold (alpha-1) into z, so p_i = [z_i - tau]_+ ^ 2
    zmax = z.max(dim=-1, keepdim=True).values
    tau_lo = zmax - 1.0
    tau_hi = zmax - (1.0 / z.shape[-1]) ** (alpha - 1)
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        # p from the current tau (clip negatives, then raise to 1/(alpha-1) = power 2)
        p = ____
        # if the mass is too small, tau is too high -> move the ceiling down; else move the floor up
        below = p.sum(dim=-1, keepdim=True) < 1.0
        tau_hi = torch.where(below, tau, tau_hi)
        tau_lo = torch.where(below, tau_lo, tau)
    tau = (tau_lo + tau_hi) / 2
    p = torch.clamp(z - tau, min=0) ** (1.0 / (alpha - 1))
    # normalise to kill tiny residual (bisection is approximate)
    return ____

zz = torch.tensor([[2.0, 1.5, 0.5, 0.0, -1.0]])
print("entmax15 :", np.round(entmax15(zz).numpy(), 3))
print("softmax  :", np.round(torch.softmax(zz, -1).numpy(), 3))'''

T1_SOL = (T1_CODE
          .replace("""        # p from the current tau (clip negatives, then raise to 1/(alpha-1) = power 2)
        p = ____""",
                   """        # p from the current tau (clip negatives, then raise to 1/(alpha-1) = power 2)
        p = torch.clamp(z - tau, min=0) ** (1.0 / (alpha - 1))""")
          .replace("""    # normalise to kill tiny residual (bisection is approximate)
    return ____""",
                   """    # normalise to kill tiny residual (bisection is approximate)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)""")
          .replace("# TODO — implement entmax15 by bisection along the last dimension. Fill every ____.",
                   "# SOLUTION — entmax15 by bisection (Peters, Niculae & Martins 2019)"))

T1_CHECK = r'''# CHECK — the properties that make entmax15 a differentiable feature SELECTOR (do not edit)
rng = np.random.default_rng(0)
Z = torch.tensor(rng.normal(size=(200, 12)) * 3, dtype=torch.float64)
P = entmax15(Z, n_iter=50)

ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("sums to 1 along the feature axis", torch.allclose(P.sum(-1), torch.ones(200, dtype=torch.float64), atol=1e-6),
    f"max dev {(P.sum(-1) - 1).abs().max():.2e}")
chk("non-negative", bool((P >= 0).all()))
chk("contains EXACT zeros (softmax never does)",
    bool((P == 0).any()) and bool((torch.softmax(Z, -1) > 0).all()),
    f"{(P == 0).double().mean() * 100:.0f}% of entries are 0")
chk("sits BETWEEN softmax and sparsemax in sparsity",
    int((P == 0).sum()) > 0 and (P == 0).double().mean() < (Z.shape[1] - 1) / Z.shape[1])

# VALIDATION against the reference implementation (NOTES #22): the library checks you.
from entmax import entmax15 as ref_entmax15, sparsemax as ref_sparsemax
d = (P - ref_entmax15(Z, dim=-1)).abs().max().item()
chk("VALIDATED against entmax.entmax15", d < 1e-5, f"max |Δ| {d:.2e}")
chk("agrees with relkit.node.entmax15", torch.allclose(P, relkit_entmax15(Z, n_iter=50), atol=1e-9))

# entmax15 is denser than sparsemax on the same logits (fewer zeros)
zeros_ent = int((ref_entmax15(Z, dim=-1) == 0).sum())
zeros_spa = int((ref_sparsemax(Z, dim=-1) == 0).sum())
chk("entmax15 keeps MORE features alive than sparsemax", zeros_ent < zeros_spa,
    f"entmax zeros {zeros_ent} < sparsemax zeros {zeros_spa}")
print("\nTask 1", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------- Task 2: the oblivious tree forward pass
T2_MD = r'''## Task 2 — the oblivious tree forward pass (entmoid split + outer-product routing)

**Goal.** Implement the forward pass of one *ensemble of differentiable oblivious trees* (an `ODST` layer):
soft-split every level with the **entmoid**, then route each row to all `2^depth` leaves by an **outer
product**, and read off the weighted leaf responses.

The pieces, per (batch, tree, level):

```
f_hat = <x, entmax15(feature_logits)>            # the feature CHOICE (Task 1), one per level
c     = entmoid( (f_hat - threshold) / exp(log_temp) )   # the soft SPLIT, in [0, 1]
bins  = stack([c, 1 - c])                         # go-right / go-left probabilities
w     = product over levels of the matching bin   # leaf ROUTING weights, 2^depth of them, sum to 1
out   = sum over leaves of  w * response           # weighted average of the leaf responses
```

**Why this matters.** This forward pass is the whole trick: because every step is a smooth function
(entmax choice, entmoid split, product routing), the 2^depth-leaf tree is differentiable and trains by
backprop. The `entmoid` is just the two-class entmax15 — it saturates to an exact 0/1 for a decisive gap
(a real decision) but stays smooth, so a gradient flows. You will confirm your forward matches the
from-scratch reference `relkit_ODST` exactly when they share parameters.'''

T2_CODE = r'''# TODO — fill every ____.  (entmax15 from Task 1; entmoid15 provided by relkit.)
def odst_forward(x, feature_logits, thresholds, log_temperatures, response, bin_codes_1hot):
    """One ensemble of oblivious trees. Shapes:
       x:(B,in)  feature_logits:(in,T,d)  thresholds:(T,d)  log_temperatures:(T,d)
       response:(T,tree_dim,2^d)  bin_codes_1hot:(d,2^d,2)   ->  (B, T*tree_dim)
    """
    B = x.shape[0]
    feature_selectors = entmax15(feature_logits.movedim(0, -1)).movedim(-1, 0)   # entmax over features
    f_hat = torch.einsum("bi,itl->btl", x, feature_selectors)                    # (B,T,d) chosen values

    # the soft split: scale the gap by 1/temperature, then entmoid -> P(go right) in [0,1]
    c = ____
    bins = torch.stack([c, 1 - c], dim=-1)                                       # (B,T,d,2)

    # route to leaves: match each leaf's per-level bit, then take the PRODUCT over levels
    bin_matches = torch.einsum("btds,dls->btdl", bins, bin_codes_1hot)           # (B,T,d,2^d)
    weights = ____                                                               # (B,T,2^d), sums to 1

    # weighted average of leaf responses
    out = torch.einsum("btl,tcl->btc", weights, response)                        # (B,T,tree_dim)
    return out.reshape(B, -1)

# Build a reference ODST, then run YOUR forward on ITS parameters and compare (NOTES #22).
torch.manual_seed(0)
ref = relkit_ODST(in_features=10, num_trees=8, depth=4, tree_dim=1).eval()
x = torch.randn(32, 10)
mine = odst_forward(x, ref.feature_logits, ref.thresholds, ref.log_temperatures,
                    ref.response, ref.bin_codes_1hot)
with torch.no_grad():
    theirs = ref(x)
print("max |Δ| vs reference ODST:", (mine - theirs).abs().max().item())'''

T2_SOL = (T2_CODE
          .replace("""    # the soft split: scale the gap by 1/temperature, then entmoid -> P(go right) in [0,1]
    c = ____""",
                   """    # the soft split: scale the gap by 1/temperature, then entmoid -> P(go right) in [0,1]
    c = entmoid15((f_hat - thresholds) * torch.exp(-log_temperatures))""")
          .replace("""    weights = ____                                                               # (B,T,2^d), sums to 1""",
                   """    weights = bin_matches.prod(dim=-2)                                           # (B,T,2^d), sums to 1""")
          .replace("# TODO — fill every ____.  (entmax15 from Task 1; entmoid15 provided by relkit.)",
                   "# SOLUTION  (entmax15 from Task 1; entmoid15 provided by relkit.)"))

T2_CHECK = r'''# CHECK — the invariants of the differentiable oblivious tree (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

# recompute the routing weights to inspect them directly
fs = entmax15(ref.feature_logits.movedim(0, -1)).movedim(-1, 0)
f_hat = torch.einsum("bi,itl->btl", x, fs)
c = entmoid15((f_hat - ref.thresholds) * torch.exp(-ref.log_temperatures))
bins = torch.stack([c, 1 - c], dim=-1)
weights = torch.einsum("btds,dls->btdl", bins, ref.bin_codes_1hot).prod(dim=-2)

chk("forward MATCHES the reference ODST on shared parameters",
    torch.allclose(mine, ref(x), atol=1e-5), f"max |Δ| {(mine - ref(x)).abs().max():.2e}")
chk("leaf-routing weights sum to 1 per (row, tree)",
    torch.allclose(weights.sum(-1), torch.ones(32, 8), atol=1e-4),
    f"max dev {(weights.sum(-1) - 1).abs().max():.2e}")
chk("there are 2^depth = 16 leaves", weights.shape[-1] == 16)
chk("entmoid gives a genuine hard decision for a decisive gap (exact 0/1)",
    float(entmoid15(torch.tensor([9.0]))) == 1.0 and float(entmoid15(torch.tensor([-9.0]))) == 0.0)

# temperature -> 0 collapses the soft routing onto ONE leaf (a hard tree)
c_hard = entmoid15((f_hat - ref.thresholds) / 0.01)
w_hard = torch.stack([c_hard, 1 - c_hard], -1)
w_hard = torch.einsum("btds,dls->btdl", w_hard, ref.bin_codes_1hot).prod(dim=-2)
chk("tau -> 0 makes routing ~one-hot (max leaf weight > 0.98)", float(w_hard.max(-1).values.mean()) > 0.98,
    f"mean max-leaf weight {float(w_hard.max(-1).values.mean()):.3f}")
print("\nTask 2", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------- Task 3: NODE vs CatBoost vs baselines
T3_MD = r'''## Task 3 — race CatBoost (the clean head-to-head) and the L042 baselines

**Goal.** Run NODE against **CatBoost** and the tuned MLP/ResNet under one shared frame on several tables,
then summarise with **mean ranks** and a Friedman test.

**Why CatBoost specifically.** NODE and CatBoost are the *same tree shape* — ensembles of oblivious
(symmetric) trees — differing only in how splits are chosen (NODE by gradient descent, CatBoost greedily).
So this isolates the paper's central question: does making the oblivious tree differentiable buy accuracy?
And it is [L042](../lessons/0042-mlp-resnet-baselines.html)'s baseline-first rule: the cheap strong models
are run *first*, to the same protocol, and the expensive new model must beat them to earn its place.

**And why several datasets (NOTES #23).** A single table is a demonstration, never evidence. This set is
small and deliberately *not* representative — the strong claim stays cited to the paper's own 40+ dataset
study (which used far more trees and tuning than this budget).'''

T3_CODE = r'''# TODO — fill every ____.
def search_node(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    """Validation-selected random search over NODE's space — SAME budget as every other model."""
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        cfg, opt = sample_node(rng, Xtr.shape[1])
        torch.manual_seed(seed + t)
        model, val = train_node(DenseNODE(**cfg), Xtr, ytr, Xva, yva, lr=opt["lr"],
                                max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        # keep the trial with the best VALIDATION score, and score IT on test (never select on test)
        if ____:
            best = {"val": val, "test": ____}
    return best["test"]

MODELS = ["node", "catboost", "mlp", "resnet"]
table = {}
for name in DATASETS:
    X, y = load_dense(name)
    rows = {m: [] for m in MODELS}
    for s in SEEDS:
        f = frame(X, y, s)
        rows["node"].append(search_node(*f, budget=BUDGET, seed=s))
        rows["catboost"].append(search_catboost(*f, budget=BUDGET, seed=s))
        rows["mlp"].append(search_net("mlp", *f, budget=BUDGET, seed=s))
        rows["resnet"].append(search_net("resnet", *f, budget=BUDGET, seed=s))
    table[name] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}
    print(f"{name:>10}: " + " | ".join(f"{m} {table[name][m][0]:.3f}±{table[name][m][1]:.3f}"
                                       for m in MODELS), flush=True)

# --- cross-dataset summary: mean ranks (1 = best per dataset) + Friedman
score = np.array([[table[d][m][0] for m in MODELS] for d in DATASETS])   # datasets x models
ranks = np.array([rankdata(-row, method="average") for row in score])    # HIGHEST score -> rank 1
mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(MODELS)}
fried = friedmanchisquare(*[score[:, i] for i in range(len(MODELS))])
node_beats_cat = sum(table[d]["node"][0] > table[d]["catboost"][0] for d in DATASETS)

print("\nmean ranks:", {m: round(r, 2) for m, r in mean_rank.items()})
print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} "
      f"(k={len(MODELS)}, N={len(DATASETS)})")
print(f"NODE beats CatBoost on {node_beats_cat}/{len(DATASETS)} tables")'''

T3_SOL = (T3_CODE
          .replace("""        if ____:
            best = {"val": val, "test": ____}""",
                   """        if val > best["val"]:
            best = {"val": val, "test": node_auc(model, Xte, yte)}""")
          .replace("# TODO — fill every ____.", "# SOLUTION"))

T3_CHECK = r'''# CHECK — the verdict, read the disciplined way (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("ranks are a valid per-dataset ranking of the 4 models",
    ranks.shape == (len(DATASETS), 4) and bool(np.allclose(ranks.sum(1), 10.0)))
chk("best score on each dataset got rank 1",
    all(ranks[i][np.argmax(score[i])] == 1 for i in range(len(DATASETS))))
chk("every model was given the SAME budget", True, f"budget={BUDGET} per model, seeds={SEEDS}")

cat_rank = mean_rank["catboost"]
print(f"""
VERDICT — NODE mean rank {mean_rank['node']:.2f} vs CatBoost {cat_rank:.2f} (its own tree shape, grown greedily)
  baseline-first outcome : {'CLEARED the bar' if mean_rank['node'] < min(mean_rank['catboost'], mean_rank['mlp'], mean_rank['resnet']) else 'did NOT clear the bar'}
  Friedman p             : {fried.pvalue:.3f} -> {'a difference is detectable' if fried.pvalue < 0.05 else 'CANNOT distinguish these models on this sample'}

Keep the two statements separate. p > 0.05 means you may NOT claim NODE is *significantly worse* — this
sample is far too small. But the burden of proof sits with the NEW, more expensive model, so ranking
behind the CatBoost it generalises means it did not clear the bar HERE. With a small budget and {len(SEEDS)}
seeds your exact numbers will differ from the lesson's fuller run (labs/_verify_l044.py: NODE 3.50,
CatBoost 2.50, MLP 2.00, ResNet 2.00, p = 0.308) — what should reproduce is the DIRECTION. And this is a
down-scaled experiment: NODE's paper win is at benchmark scale with thousands of trees, not this budget.""")
print("Task 3", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------------------- Task 4: the cost of differentiability
T4_MD = r'''## Task 4 — the other axis: what does the differentiability cost?

**Goal.** Time NODE against the CatBoost it imitates on one table, so your verdict weighs accuracy *and*
compute — the honest full picture.

**Why.** Accuracy alone hides half the story. NODE and CatBoost are the same tree shape, but NODE trains
every split by backprop over many epochs while CatBoost grows them greedily in one pass. Measuring the
wall-clock gap is what turns "NODE is a bit behind" into "NODE is a bit behind *and* far more expensive" —
which is the point of the lesson.'''

T4_CODE = r'''# TODO — fill every ____.
from catboost import CatBoostClassifier
X, y = load_dense("credit_g")
Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, 0)

# time NODE (128 trees, depth 6)
t0 = time.time(); torch.manual_seed(0)
node = DenseNODE(X.shape[1], num_trees=128, depth=6, n_layers=1)
node, _ = train_node(node, Xtr, ytr, Xva, yva, lr=1e-2, max_epochs=EPOCHS, patience=PATIENCE, seed=0)
node_s = time.time() - t0
node_test = node_auc(node, Xte, yte)

# time CatBoost (400 symmetric trees) on the SAME split
t0 = time.time()
cat = CatBoostClassifier(depth=6, iterations=400, learning_rate=0.05, grow_policy="SymmetricTree",
                         random_seed=0, thread_count=2, verbose=0, allow_writing_files=False)
cat.fit(Xtr, ytr.astype(int))
cat_s = time.time() - t0
cat_test = roc_auc_score(yte, cat.predict_proba(Xte)[:, 1])

# the slowdown factor
slowdown = ____

print(f"NODE     : {node_s:6.1f}s   test AUC {node_test:.3f}")
print(f"CatBoost : {cat_s:6.1f}s   test AUC {cat_test:.3f}")
print(f"NODE is {slowdown:.0f}x slower to train on this table")'''

T4_SOL = T4_CODE.replace("""# the slowdown factor
slowdown = ____""", """# the slowdown factor
slowdown = node_s / max(cat_s, 1e-6)""").replace("# TODO — fill every ____.", "# SOLUTION")

T4_CHECK = r'''# CHECK — cost is real and belongs in the verdict (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("NODE is materially slower than CatBoost (>= 5x)", slowdown >= 5, f"{slowdown:.0f}x")
chk("both produce a valid AUC", 0.5 < node_test < 1.0 and 0.5 < cat_test < 1.0)
print(f"""
The lesson's measured run: NODE 60.2 s vs CatBoost 0.9 s (~70x), NODE 0.793 vs CatBoost 0.813 AUC. Your
absolute times depend on your CPU, but the SHAPE holds: the differentiable tree costs far more and, on one
flat table, buys nothing on the metric. That is the whole trade — differentiability is worth paying for
only when the tree must COMPOSE with other learned modules (joint embeddings, DenseNet stacking,
end-to-end multi-modal), which is exactly the relational setting the thesis cares about.""")
print("Task 4", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------------------------------------ exit ticket
EXIT_MD = r'''## EXIT TICKET

Paste this output to your teacher, or just say *"lab done."*'''

EXIT_CODE = r'''# EXIT TICKET
print("=== LAB 044 — NODE (differentiable oblivious trees) ===")
print(f"entmax15         : validated (exact zeros, between softmax & sparsemax, matches reference)")
print(f"oblivious tree   : forward matches relkit.node.ODST; leaf routing sums to 1; tau->0 => one leaf")
print(f"bake-off ranks   : " + ", ".join(f"{m} {mean_rank[m]:.2f}" for m in MODELS))
print(f"Friedman         : chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (N={len(DATASETS)} datasets)")
print(f"NODE vs CatBoost : NODE wins {node_beats_cat}/{len(DATASETS)} tables")
print(f"cost             : NODE {node_s:.1f}s vs CatBoost {cat_s:.1f}s ({slowdown:.0f}x slower)")
print(f"cleared the bar? : {'YES' if mean_rank['node'] < min(mean_rank['catboost'], mean_rank['mlp'], mean_rank['resnet']) else 'NO'}")
print()
print("when would you reach for a differentiable tree over CatBoost?:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"When the tree must COMPOSE with other differentiable modules — not for accuracy on one flat table, '
    'where CatBoost\'s greedy oblivious trees rank ABOVE NODE and train ~70x faster. NODE makes the three '
    'discrete steps differentiable (entmax15 feature choice with real zeros, the entmoid soft split, and '
    'the outer-product routing that sends a row to all 2^d leaves as a distribution), so hundreds of trees '
    'train by backprop and stack DenseNet-style. That end-to-end trainability is worthless on a single '
    'table but essential when the tree co-learns with embeddings/encoders, stacks into a hierarchy, or '
    'sits in a multi-modal pipeline — i.e. exactly the relational setting where the signal spans joined '
    'tables and a greedy GBDT cannot see the structure. My run reproduces the DIRECTION (NODE behind '
    'CatBoost, far slower), not the paper\'s benchmark-scale win, which needs thousands of trees and heavy '
    'tuning."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 044 — NODE: build a differentiable oblivious tree, then race CatBoost

**Lesson:** [`lessons/0044-node.html`](../lessons/0044-node.html) · **Phase / Year:** Year 2 · Q1

**Paper:** Popov, Morozov & Babenko 2019, *Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data* ([arXiv:1909.06312](https://arxiv.org/abs/1909.06312)) — §2 (architecture, Eq. 1–4) + Fig. 1 (DenseNet stacking). entmax: Peters, Niculae & Martins 2019 ([arXiv:1905.05702](https://arxiv.org/abs/1905.05702)). Oblivious/symmetric trees: CatBoost ([L016](../lessons/0016-catboost.html)).

**Dataset tier:** **A** — small real OpenML tables via `relkit` (CPU-cheap; deliberately NOT representative — see the #23 note in Task 3).

**Skill you are practising:** implement a differentiable oblivious tree **from scratch** — (1) `entmax15` by bisection, (2) the **oblivious tree forward pass** (entmoid split + outer-product routing) — then (3) race **CatBoost** (the same tree shape, grown greedily) under [L042](../lessons/0042-mlp-resnet-baselines.html)'s baseline-first rule, and (4) measure the cost.

**Exit criteria:** EXIT TICKET prints your entmax15 validation, the ODST forward match + leaf-routing check, the NODE-vs-CatBoost rank table, the cost ratio, and one sentence — *when is a differentiable tree actually worth it?*

---

### How this notebook works
- **PROVIDED** cells — boilerplate (data, frame, search spaces, CatBoost/net searches) **and** NODE (ODST, DenseNODE, train loop) copied into the notebook (not hidden behind `import relkit.node`); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. After EXIT, a **NEXT STEP** cell trains closer to the paper (Colab GPU or Modal). When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` → kernel **Relational Labs (.venv)**. Needs **torch** + scikit-learn + **catboost** (CPU is fine); the **`entmax`** package is used only to validate your entmax15. Real datasets fetch from OpenML on first run then cache. Budget: **~4–8 minutes on CPU** — set `OMP_NUM_THREADS=1` if a search feels slow (that has been the real cause of every slow lab so far). This lab uses a deliberately small budget/seed/dataset count to stay interactive; the lesson's headline numbers come from the fuller `labs/_verify_l044.py` run plus the paper's benchmark.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — what NODE actually does

**The idea.** Take an **ensemble of oblivious decision trees** — the *symmetric* trees from [L016](../lessons/0016-catboost.html), where every node on a level shares one (feature, threshold), so a depth-`d` tree is a `2^d`-leaf lookup — and make the whole thing **differentiable**, so it trains end-to-end by gradient descent and stacks into deep layers like any neural net. A GBDT cannot do this: its greedy, discrete splits have no gradient, so it can never be a *layer* inside a larger learned model.

**The three softenings.**
1. **Feature choice → `entmax15`** (Task 1). Replace "pick feature j" with `f_hat = <x, entmax15(logits)>`, a learned sparse distribution over columns. `entmax15` (alpha = 1.5) is the middle of the family softmax (alpha→1, dense) — entmax15 (real zeros) — sparsemax (alpha = 2, sparsest, your L043 op), so the choice is genuinely selective yet differentiable.
2. **Split → `entmoid`** (Task 2). Replace the hard `1[f_hat > b]` with `c = entmoid((f_hat - b)/tau)`, the two-class entmax15. It saturates to an exact 0/1 for a decisive gap (a real decision) but smoothly, so a gradient flows; `tau` is the softness knob.
3. **Routing → outer product** (Task 2). A soft tree sends a *fraction* of the row to *every* leaf: leaf weight = product over levels of `c` or `1-c`. The outer product of the per-level `[c, 1-c]` gives all `2^d` leaf weights at once, summing to 1; the output is their weighted average of the leaf responses.

**NODE the network.** A NODE *layer* is an ensemble of hundreds of these trees; layers stack DenseNet-style (each sees the input plus earlier layers' outputs), and the prediction averages all trees. That stacking is the one thing a GBDT structurally cannot do.

**The honest verdict (Tasks 3–4).** On a single flat table this buys nothing: CatBoost — the same tree shape, grown greedily — ranks *above* NODE and trains ~70× faster. Differentiability pays off only when the tree must **compose** with other learned modules (embeddings, stacking, multi-modal) — which is the relational setting the thesis is about.

Full write-up + the interactive routing widget: [Lesson 044](../lessons/0044-node.html).'''),
        md("## Setup — PROVIDED (real tables + shared frame + search spaces)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        code(r'''# PROVIDED — adapter: your Task-1 entmax15 is last-axis only; the inlined ODST calls dim=0.
# Bind `_impl` at def-time (and stash it) so re-running this cell cannot wrap the wrapper —
# that RecursionError's on the next train, not here.
_impl = getattr(entmax15, "_impl", entmax15)
def entmax15(z, dim=-1, n_iter=30, *, _impl=_impl):
    """Keep YOUR Task-1 implementation; accept the encoder's dim= keyword."""
    if dim in (-1, z.ndim - 1):
        return _impl(z, n_iter=n_iter)
    z_t = z.transpose(dim, -1)
    return _impl(z_t, n_iter=n_iter).transpose(dim, -1)
entmax15._impl = _impl
'''),
        md(architecture_md(
            "ODST, DenseNODE, and the training loop",
            "labs/relkit/node.py",
            "`entmax15`",
        )),
        code(inline_source(
            os.path.join(HERE, "relkit/node.py"),
            skip_defs={"entmax15"},
        )),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(T4_MD), code(T4_SOL if solution else T4_CODE), code(T4_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(notebook_scaleup_md(
            lesson=44,
            paper="Popov, Morozov & Babenko 2020, Neural Oblivious Decision Ensembles",
            arxiv="1909.06312",
            lab_rows=[
                ("mean ranks", "NODE 3.50 vs CatBoost 2.50 / MLP 2.00 / ResNet 2.00, Friedman p=0.308"),
                ("cost", "~70× slower than CatBoost on credit_g at this budget"),
            ],
            paper_rows=[
                ("Higgs default-HP error", "NODE 0.2412 vs CatBoost 0.2434 (Table 1). We use OpenML 23512 (~98k of 10.5M) → expect INCOMPARABLE on the absolute number; read DIRECTION."),
            ],
            modal="modal/l044_paper_repro.py",
        )),
        code(notebook_scaleup_code(
            lesson=44,
            harness_path=os.path.join(HERE, "_paper_repro_l044.py"),
            modal="modal/l044_paper_repro.py",
            skip_imports={"relkit.node"},
        )),
        md(r'''## Stretch (optional, ungraded) — after the scale-up

1. **The DenseNet depth claim.** Compare `n_layers = 1` vs `2` vs `3` on one table at fixed total trees.
   NODE's pitch is that stacking lets later trees split on earlier trees' decisions — does depth help here,
   or is one layer enough at this scale?
2. **alpha as a knob.** Swap `entmax15` for `sparsemax` (alpha = 2) and softmax (alpha → 1) in the feature
   choice and re-race. The paper picks 1.5 deliberately — do you see why the extremes are worse?
3. **Give NODE its real budget.** The paper uses up to ~2000 trees per layer. Push `num_trees` up (expect
   minutes per fit on CPU) on one table and watch whether NODE closes the gap to CatBoost — and how the
   cost ratio explodes. This is the honest way to see the paper's regime.
4. **Where differentiability should win.** Concatenate two tables on a key (a tiny relational join), feed
   the joined rows to NODE, and train an embedding jointly with the trees. This is the toy version of the
   Year 4 relational setting — the case where end-to-end training is the point.'''),
        code(r'''# STRETCH — ungraded.
# X, y = load_dense("kc1")
# for L in (1, 2, 3):
#     f = frame(X, y, 0); torch.manual_seed(0)
#     m = DenseNODE(X.shape[1], num_trees=128, depth=4, n_layers=L)
#     m, _ = train_node(m, *f[:4], lr=1e-2, max_epochs=EPOCHS, patience=PATIENCE, seed=0)
#     print(f"n_layers={L}: test AUC {node_auc(m, f[4], f[5]):.3f}")'''),
    ]

    nb_obj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python",
                           "name": "python3"},
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
    with open(os.path.join(HERE, "0044-node.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0044-node.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0044-node.ipynb + solution")


if __name__ == "__main__":
    main()
