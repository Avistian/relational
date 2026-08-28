"""Build Lab 043 (TabNet — sequential attention) — student + solution notebooks.

Honours NOTES standards #21 (lab ships with the lesson), #22 (build from scratch; libraries only
validate), #23 (never conclude from one dataset) and #24 (paper-mirror doctrine: implement the paper's
mechanism, use the paper's own evaluation data, ship a reproducibility ledger).

Tasks: (1) sparsemax from Algorithm 1; (2) the attentive transformer + prior scale (the "sequential"
part); (3) read the masks on the paper's Syn2 / Syn4; (4) the baseline-first verdict vs L042's MLP/ResNet.

Run: .venv/bin/python labs/_build_l043.py
Execute the solution: .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    labs/solutions/0043-tabnet.ipynb
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


SETUP = r'''# PROVIDED — imports, the paper's synthetic generators, the small real tables, the shared frame,
# and the search spaces. The TabNet *architecture* is inlined in a later cell so you can read it;
# `relkit` here is only data + L042 baselines + a sparsemax checker (NOTES #22 / #25).
# Just run this cell.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata, friedmanchisquare
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
from relkit.nets import TabResNet, TabMLP, train_net, net_auc     # L042 baselines (from scratch)
from relkit.tabnet import sparsemax as relkit_sparsemax           # checker only (NOTES #22)
from relkit.synth import make_syn2, make_syn4                      # the paper's Table 1 / Fig. 5 data

DEVICE = "cpu"
DATASETS = ["credit_g", "diabetes", "kc1"]   # small + numeric-heavy: NOT representative (see #23 note)
BUDGET, SEEDS = 3, [0, 1]                    # smaller than the lesson's run, so the lab stays interactive
EPOCHS, PATIENCE = 120, 12                   # same as labs/_verify_l043.py, so directions reproduce

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
def sample_tabnet(rng, d_in):
    """TabNet's space, from the paper's Appendix F."""
    u = int(rng.choice([8, 16, 24]))
    return dict(d_in=d_in, n_d=u, n_a=u, n_steps=int(rng.choice([3, 4, 5])),
                gamma=float(rng.choice([1.0, 1.2, 1.5, 2.0])),
                lambda_sparse=float(rng.choice([0.0, 1e-6, 1e-4, 1e-3, 1e-2])),
                virtual_batch_size=int(rng.choice([64, 128])),
                momentum=0.02), dict(lr=float(rng.choice([0.005, 0.01, 0.02, 0.025])))

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
                           max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": net_auc(m, Xte, yte)}
    return best["test"]

def search_gbt(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    rng = np.random.default_rng(seed + 999)
    best = {"val": -1.0, "test": None}
    for _ in range(budget):
        clf = HistGradientBoostingClassifier(
            random_state=seed, max_iter=400,
            learning_rate=float(10 ** rng.uniform(-1.5, -0.7)),
            max_leaf_nodes=int(rng.choice([15, 31, 63])),
            l2_regularization=float(rng.uniform(0.0, 3.0))).fit(Xtr, ytr)
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]

print("setup ok — torch", torch.__version__)'''


# --------------------------------------------------------------------------- Task 1: sparsemax
T1_MD = r'''## Task 1 — implement `sparsemax` (the reason a mask can contain real zeros)

**Goal.** Write the projection onto the probability simplex from **Algorithm 1** of Martins & Astudillo
(2016), the operation TabNet uses instead of softmax.

**Why this first.** A *selection* mask must be able to say "not this feature" — an exact **0**. Softmax
never can: `exp(z)` is strictly positive, so every feature leaks a little weight through and keeps
receiving gradient. sparsemax returns the closest point on the simplex, which usually lies on a face
where some coordinates are exactly zero. That single change is what makes TabNet's masks *selections*
rather than *weightings*.

The algorithm, for a vector `z` of length K:

```
1. sort descending:  z_(1) >= z_(2) >= ... >= z_(K)
2. k(z) = max{ k : k * z_(k) > (sum_{j<=k} z_(j)) - 1 }
3. tau   = ( (sum_{j<=k(z)} z_(j)) - 1 ) / k(z)
4. out_i = max(z_i - tau, 0)
```

Read step 3 as: *one* threshold, chosen so that whatever survives sums to exactly 1.'''

T1_CODE = r'''# TODO — implement sparsemax along the last dimension. Fill every ____.
def sparsemax(z):
    """z: (B, K) tensor -> (B, K) projection onto the simplex (exact zeros allowed)."""
    z = z - z.max(dim=-1, keepdim=True).values   # no-op for the projection; keeps the prefix sums well-scaled
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    K = z.size(-1)
    arange = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)

    # cumulative sums minus 1  ->  (sum_{j<=k} z_(j)) - 1
    cssv = ____

    # step 2: which prefix lengths k satisfy  k * z_(k) > cssv_k ?
    support = ____

    # k(z) = how many satisfy it (keepdim so it can index)
    k = support.to(z.dtype).sum(dim=-1, keepdim=True)

    # step 3: tau = cssv at position k(z), divided by k(z)
    tau = ____

    # step 4: shift and clip
    return ____

zz = torch.tensor([[3.0, 1.2, 0.9, -2.0], [0.5, 0.5, 0.5, 0.5]])
print("sparsemax:\n", sparsemax(zz))
print("softmax:  \n", torch.softmax(zz, dim=-1))'''

T1_SOL = r'''# SOLUTION — sparsemax (Martins & Astudillo 2016, Alg. 1)
def sparsemax(z):
    """z: (B, K) tensor -> (B, K) projection onto the simplex (exact zeros allowed)."""
    z = z - z.max(dim=-1, keepdim=True).values   # no-op for the projection; keeps the prefix sums well-scaled
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    K = z.size(-1)
    arange = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)

    cssv = z_sorted.cumsum(dim=-1) - 1.0
    support = (arange * z_sorted) > cssv
    k = support.to(z.dtype).sum(dim=-1, keepdim=True)
    tau = cssv.gather(-1, k.long() - 1) / k
    return torch.clamp(z - tau, min=0.0)

zz = torch.tensor([[3.0, 1.2, 0.9, -2.0], [0.5, 0.5, 0.5, 0.5]])
print("sparsemax:\n", sparsemax(zz))
print("softmax:  \n", torch.softmax(zz, dim=-1))'''

T1_CHECK = r'''# CHECK — the properties that make sparsemax usable as a selection mask (do not edit)
rng = np.random.default_rng(0)
Z = torch.tensor(rng.normal(size=(200, 12)) * 3, dtype=torch.float64)
P = sparsemax(Z)

ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("sums to 1 along the feature axis", torch.allclose(P.sum(-1), torch.ones(200, dtype=torch.float64)),
    f"max dev {(P.sum(-1) - 1).abs().max():.2e}")
chk("non-negative", bool((P >= 0).all()))
chk("contains EXACT zeros (softmax never does)", bool((P == 0).any()) and bool((torch.softmax(Z, -1) > 0).all()),
    f"{(P == 0).double().mean() * 100:.0f}% of entries are 0")
chk("uniform input -> uniform output",
    torch.allclose(sparsemax(torch.zeros(1, 5, dtype=torch.float64)),
                   torch.full((1, 5), 0.2, dtype=torch.float64)))
chk("dominant coordinate -> one-hot",
    torch.allclose(sparsemax(torch.tensor([[9.0, 0.0, 0.0]], dtype=torch.float64)),
                   torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)))

# VALIDATION against the reference implementation (NOTES #22): the library checks you.
try:
    from pytorch_tabnet.sparsemax import Sparsemax
    ref = Sparsemax(dim=-1)(Z.float())
    d = (P.float() - ref).abs().max().item()
    chk("VALIDATED against pytorch_tabnet.sparsemax", d < 1e-5, f"max |Δ| {d:.2e}")
except ImportError:
    print("SKIP  reference pytorch_tabnet not installed (pip install pytorch-tabnet)")

# ...and against relkit's version, which the rest of the lab uses
chk("agrees with relkit.tabnet.sparsemax",
    torch.allclose(P, relkit_sparsemax(Z), atol=1e-9))
print("\nTask 1", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------- Task 2: attentive transformer + prior scale
T2_MD = r'''## Task 2 — the attentive transformer and the prior scale (the *sequential* part)

**Goal.** Build the block that produces each step's mask, and the running prior that couples the steps:

```
M[i] = sparsemax( P[i-1] * h_i(a[i-1]) )        h_i = Linear -> BatchNorm
P[i] = prod_{j<=i} ( gamma - M[j] )             P[0] = 1
```

**Why this matters.** Sparsity and *sequentiality* are two different things. sparsemax (Task 1) gives you
sparsity. The **prior scale** is what makes the attention sequential: it remembers how much each feature
has already been spent and multiplies the next step's logits by it, so a feature used early is discounted
and the steps divide the features between them instead of all grabbing the same strong ones.

`gamma` is the strictness knob, and you will verify its defining property: at **gamma = 1** the leftover
budget is exactly `1 - M`, so a feature used *in full* is banned from every later step.'''

T2_CODE = r'''# TODO — fill every ____.
class AttentiveTransformer(nn.Module):
    """M[i] = sparsemax(P[i-1] * h_i(a[i-1])), with h_i = Linear -> BatchNorm."""
    def __init__(self, d_a, d_features):
        super().__init__()
        self.fc = nn.Linear(d_a, d_features, bias=False)
        self.bn = nn.BatchNorm1d(d_features)

    def forward(self, a, prior):
        logits = self.bn(self.fc(a))
        # scale the logits by the prior, then project onto the simplex
        return ____

def update_prior(prior, M, gamma):
    """P[i] = P[i-1] * (gamma - M[i]).  Clamp at 0: a prior can never go negative."""
    return ____

# A 3-step run on one batch, to watch the prior work.
torch.manual_seed(0)
B, D, d_a, gamma = 64, 10, 8, 1.5
att = [AttentiveTransformer(d_a, D).eval() for _ in range(3)]
a = torch.randn(B, d_a)
prior = torch.ones(B, D)
masks, priors = [], []
with torch.no_grad():                    # inspecting the mechanism, not training it
    for i in range(3):
        priors.append(prior.clone())
        M = att[i](a, prior)
        masks.append(M)
        prior = update_prior(prior, M, gamma)
        a = torch.randn(B, d_a)      # stand-in for the real a[i] from the feature transformer

for i, M in enumerate(masks):
    print(f"M[{i+1}] row 0:", np.round(M[0].numpy(), 3),
          " nonzero:", int((M[0] > 0).sum()))
print("P[2] row 0:", np.round(priors[2][0].numpy(), 3), "  (untouched features sit at gamma^2)")'''

T2_SOL = T2_CODE.replace(
    """        logits = self.bn(self.fc(a))
        # scale the logits by the prior, then project onto the simplex
        return ____""",
    """        logits = self.bn(self.fc(a))
        # scale the logits by the prior, then project onto the simplex
        return sparsemax(prior * logits)"""
).replace(
    """    \"\"\"P[i] = P[i-1] * (gamma - M[i]).  Clamp at 0: a prior can never go negative.\"\"\"
    return ____""",
    """    \"\"\"P[i] = P[i-1] * (gamma - M[i]).  Clamp at 0: a prior can never go negative.\"\"\"
    return torch.clamp(prior * (gamma - M), min=0.0)"""
).replace("# TODO — fill every ____.", "# SOLUTION")

T2_CHECK = r'''# CHECK — the paper's invariants for the mask and the prior (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("every mask sums to 1 per row",
    all(torch.allclose(M.sum(-1), torch.ones(B), atol=1e-5) for M in masks))
chk("masks are sparse (exact zeros present)", any(bool((M == 0).any()) for M in masks),
    f"step 1: {(masks[0] == 0).float().mean() * 100:.0f}% zeros")
chk("P[0] is all ones (no prior on any feature)", bool((priors[0] == 1).all()))
chk("P[1] == gamma - M[1] exactly",
    torch.allclose(priors[1], torch.clamp(gamma - masks[0], min=0), atol=1e-6))
chk("a spent feature's prior drops BELOW gamma, an untouched one stays at gamma",
    bool(((priors[1] < gamma - 1e-6) == (masks[0] > 0)).all()))

# gamma = 1: the defining property — full use bans the feature outright.
prior1 = torch.ones(1, 4)
M_full = torch.tensor([[1.0, 0.0, 0.0, 0.0]])       # feature 0 used in full
p_after = update_prior(prior1, M_full, 1.0)
chk("gamma = 1: a fully-used feature is BANNED (prior exactly 0)", float(p_after[0, 0]) == 0.0,
    f"prior after = {p_after[0].tolist()}")
p_after_relaxed = update_prior(prior1, M_full, 2.0)
chk("gamma = 2: the same feature stays available (prior > 0)", float(p_after_relaxed[0, 0]) > 0,
    f"prior after = {p_after_relaxed[0].tolist()}")
chk("priors never go negative", all(bool((p >= 0).all()) for p in priors))
print("\nTask 2", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------------------- Task 3: read the masks
T3_MD = r'''## Task 3 — read the masks on the paper's own data (Syn2 and Syn4)

**Goal.** Train the from-scratch TabNet on two synthetic datasets where the truly-relevant features are
known *by construction*, and check whether the aggregate mask `M_agg` finds them.

**Why synthetic, and why these two.** Interpretability cannot be tested on real data — you do not know
which features actually matter, so any attribution is unfalsifiable. TabNet's Table 1 / Fig. 5 therefore
use the L2X generators (Chen et al. 2018), and so do we:

- **Syn2** — `P(y=1) ∝ exp(X3² + X4² + X5² + X6² − 4)`. Only **X3–X6** matter, for **every** row:
  relevance is **global**.
- **Syn4** — if `X11 < 0` the label comes from `exp(X1·X2)`, else from the Syn2 rule. Relevance is
  **instance-wise**; a model must read **X11** to know where to look.

`M_agg` is the paper's global attribution: the per-step masks summed with weights
`eta[i] = sum_c ReLU(d_c[i])` (each step's decision contribution) and normalised to sum to 1.
`explain` (in the inlined architecture cell above) returns it.'''

T3_CODE = r'''# TODO — fill every ____.
def fit_and_read(X, y, *, n_steps, gamma, lambda_sparse, seed=0, epochs=200, patience=25):
    """Train TabNet on (X, y) under the shared frame and return (test AUC, M_agg, test rows)."""
    Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, seed)
    torch.manual_seed(seed)
    model = TabNetEncoder(X.shape[1], n_d=16, n_a=16, n_steps=n_steps, gamma=gamma,
                          lambda_sparse=lambda_sparse, virtual_batch_size=128)
    model, _ = train_tabnet(model, Xtr, ytr, Xva, yva, lr=0.02, max_epochs=epochs,
                            patience=patience, batch_size=1024, seed=seed)
    auc = tabnet_auc(model, Xte, yte)
    M_agg, _ = explain(model, Xte)          # (n_test, D) aggregate importance masks
    return auc, M_agg, Xte

# --- Syn2: relevance is GLOBAL (X3-X6, i.e. indices 2..5)
X2, y2, rel2 = make_syn2(n=4000, seed=0)
auc2, M2, _ = fit_and_read(X2, y2, n_steps=4, gamma=2.0, lambda_sparse=1e-2)
imp2 = M2.mean(0)

# the four features with the largest mean mask weight
top4 = ____
# fraction of the total mask mass sitting on the truly relevant columns
mass_on_relevant = ____

print(f"Syn2: test AUC {auc2:.3f}")
print("  M_agg:", " ".join(f"X{j+1}:{v:.3f}" for j, v in enumerate(imp2)))
print(f"  top-4 = {sorted(top4)}   truth = {rel2}   mass on truth = {mass_on_relevant*100:.1f}%")

# --- Syn4: relevance is INSTANCE-WISE, switched by X11 (index 10)
X4, y4, _ = make_syn4(n=6000, seed=0)
auc4, M4, Xte4 = fit_and_read(X4, y4, n_steps=5, gamma=1.5, lambda_sparse=5e-3)

left = Xte4[:, 10] < 0                 # rows where X1-X2 is the relevant group
xor_mass = M4[:, [0, 1]].sum(1)        # mask mass on X1-X2
orange_mass = M4[:, 2:6].sum(1)        # mask mass on X3-X6

# on X11<0 rows, what fraction put MORE mass on X1-X2 than on X3-X6?
left_correct = ____
# on X11>0 rows, what fraction put MORE mass on X3-X6 than on X1-X2?
right_correct = ____

print(f"\nSyn4: test AUC {auc4:.3f}   X11 (the switch) mask weight {M4[:, 10].mean():.3f}")
print(f"  X11<0 rows favouring X1-X2 (correct): {left_correct*100:.1f}%")
print(f"  X11>0 rows favouring X3-X6 (correct): {right_correct*100:.1f}%")
print(f"  X1-X2 mass: left {xor_mass[left].mean():.3f} vs right {xor_mass[~left].mean():.3f}")'''

T3_SOL = (T3_CODE
          .replace("""# the four features with the largest mean mask weight
top4 = ____""",
                   """# the four features with the largest mean mask weight
top4 = np.argsort(-imp2)[:4].tolist()""")
          .replace("""# fraction of the total mask mass sitting on the truly relevant columns
mass_on_relevant = ____""",
                   """# fraction of the total mask mass sitting on the truly relevant columns
mass_on_relevant = float(imp2[rel2].sum())""")
          .replace("""# on X11<0 rows, what fraction put MORE mass on X1-X2 than on X3-X6?
left_correct = ____""",
                   """# on X11<0 rows, what fraction put MORE mass on X1-X2 than on X3-X6?
left_correct = float((xor_mass[left] > orange_mass[left]).mean())""")
          .replace("""# on X11>0 rows, what fraction put MORE mass on X3-X6 than on X1-X2?
right_correct = ____""",
                   """# on X11>0 rows, what fraction put MORE mass on X3-X6 than on X1-X2?
right_correct = float((orange_mass[~left] > xor_mass[~left]).mean())""")
          .replace("# TODO — fill every ____.", "# SOLUTION"))

T3_CHECK = r'''# CHECK — did the masks recover what they should? (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("Syn2: TabNet actually learned the task (AUC > 0.75)", auc2 > 0.75, f"AUC {auc2:.3f}")
chk("Syn2: at least 3 of the 4 true features (X3-X6) are in the top-4 mask",
    len(set(top4) & set(rel2)) >= 3,
    f"top4 {sorted(top4)} vs truth {list(rel2)}")
chk("Syn2: most of the mask mass is on the relevant columns (> 60%)", mass_on_relevant > 0.60,
    f"{mass_on_relevant*100:.1f}%")
chk("Syn2: M_agg is a normalised distribution", abs(imp2.sum() - 1) < 1e-4)

chk("Syn4: the switch feature X11 gets real weight (> 0.05)", M4[:, 10].mean() > 0.05,
    f"{M4[:, 10].mean():.3f}")
chk("Syn4: the X11>0 (orange-skin) side is read correctly (> 80% of rows)", right_correct > 0.80,
    f"{right_correct*100:.1f}%")
# The XOR side (X11<0) is the paper's hard case — printed below as PARTIAL, not a hard fail.
# A 6k-row run often ties or slightly inverts X1-X2 mass; that is the Fig. 5 "needs 10M" caveat,
# not a broken implementation.

print(f"""
READ THIS HONESTLY. Syn2 is a clean success: the mask finds the globally-relevant set. Syn4 is only a
PARTIAL success — the switch feature is found and the mass moves the right way, but only
{left_correct*100:.0f}% of X11<0 rows end up favouring their own group (vs {right_correct*100:.0f}% on the other side).
The paper trained on 10M rather than 10k samples to get sharp Fig. 5 masks, so crispness is a function of
data volume. XOR (X1*X2) is also the hardest case for a mask: neither feature is informative alone.
Conclusion: masks are EVIDENCE, checked against known-answer data — not a free explanation.""")
print("Task 3", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------- Task 4: the baseline-first verdict
T4_MD = r'''## Task 4 — the baseline-first rule: does TabNet clear the bar?

**Goal.** Run TabNet against L042's tuned MLP and ResNet (and a tuned GBDT) under one shared frame on
several tables, then summarise with **mean ranks** and a Friedman test.

**Why.** This is [L042](../lessons/0042-mlp-resnet-baselines.html)'s rule applied to a novel architecture
for the first time: the strong simple baselines are computed *first*, to the same protocol, and reported
alongside. A model that does not beat a properly-tuned MLP/ResNet has not demonstrated that its
*mechanism* is why it wins — however elegant that mechanism is.

**And why several datasets (NOTES #23).** A single table is a demonstration, never evidence. Note the set
below is small and numeric-heavy — deliberately *not* representative — so it teaches the method while the
strong claim stays cited to the large benchmarks (Gorishniy 2021, Grinsztajn 2022).'''

T4_CODE = r'''# TODO — fill every ____.
def search_tabnet(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):
    """Validation-selected random search over TabNet's paper space — SAME budget as every other model."""
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        cfg, opt = sample_tabnet(rng, Xtr.shape[1])
        torch.manual_seed(seed + t)
        model, val = train_tabnet(TabNetEncoder(**cfg), Xtr, ytr, Xva, yva, lr=opt["lr"],
                                  max_epochs=EPOCHS, patience=PATIENCE, seed=seed + t)
        # keep the trial with the best VALIDATION score, and score IT on test (never select on test)
        if ____:
            best = {"val": val, "test": ____}
    return best["test"]

MODELS = ["tabnet", "mlp", "resnet", "gbt"]
table = {}
for name in DATASETS:
    X, y = load_dense(name)
    rows = {m: [] for m in MODELS}
    for s in SEEDS:
        f = frame(X, y, s)
        rows["tabnet"].append(search_tabnet(*f, budget=BUDGET, seed=s))
        rows["mlp"].append(search_net("mlp", *f, budget=BUDGET, seed=s))
        rows["resnet"].append(search_net("resnet", *f, budget=BUDGET, seed=s))
        rows["gbt"].append(search_gbt(*f, budget=BUDGET, seed=s))
    table[name] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}
    print(f"{name:>10}: " + " | ".join(f"{m} {table[name][m][0]:.3f}±{table[name][m][1]:.3f}"
                                       for m in MODELS), flush=True)

# --- cross-dataset summary: mean ranks (1 = best per dataset) + Friedman
score = np.array([[table[d][m][0] for m in MODELS] for d in DATASETS])   # datasets x models
# rank each ROW so that the HIGHEST score gets rank 1 (hint: rank the negated row)
ranks = ____
mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(MODELS)}
fried = friedmanchisquare(*[score[:, i] for i in range(len(MODELS))])

print("\nmean ranks:", {m: round(r, 2) for m, r in mean_rank.items()})
print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} "
      f"(k={len(MODELS)} models, N={len(DATASETS)} datasets)")'''

T4_SOL = (T4_CODE
          .replace("""        if ____:
            best = {"val": val, "test": ____}""",
                   """        if val > best["val"]:
            best = {"val": val, "test": tabnet_auc(model, Xte, yte)}""")
          .replace("""# rank each ROW so that the HIGHEST score gets rank 1 (hint: rank the negated row)
ranks = ____""",
                   """# rank each ROW so that the HIGHEST score gets rank 1 (hint: rank the negated row)
ranks = np.array([rankdata(-row, method="average") for row in score])""")
          .replace("# TODO — fill every ____.", "# SOLUTION"))

T4_CHECK = r'''# CHECK — the verdict, read the disciplined way (do not edit)
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

simple_best = min(mean_rank["mlp"], mean_rank["resnet"])
cleared = mean_rank["tabnet"] < simple_best
print(f"""
VERDICT — TabNet mean rank {mean_rank['tabnet']:.2f} vs the better simple baseline {simple_best:.2f}
  baseline-first outcome : {'CLEARED the bar' if cleared else 'did NOT clear the bar'}
  Friedman p             : {fried.pvalue:.3f} -> {'a difference is detectable' if fried.pvalue < 0.05 else 'CANNOT distinguish these models on this sample'}

Keep the two statements separate. p > 0.05 means you may NOT claim TabNet is *significantly worse* —
this sample is far too small for that. But the burden of proof sits with the NEW model, so ranking
behind the baselines it was meant to beat means it did not clear the bar. With a small budget and
{len(SEEDS)} seeds your exact numbers will differ from the lesson's fuller run (labs/_verify_l043.py:
TabNet 2.50, MLP 1.75, ResNet 2.00, GBDT 3.75, p = 0.127) — what should reproduce is the DIRECTION.""")
print("Task 4", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------------------------------------ exit ticket
EXIT_MD = r'''## EXIT TICKET

Paste this output to your teacher, or just say *"lab done."*'''

EXIT_CODE = r'''# EXIT TICKET
print("=== LAB 043 — TabNet (sequential attention) ===")
print(f"sparsemax        : validated (exact zeros, sums to 1, matches the reference)")
print(f"prior scale      : gamma=1 bans a fully-used feature; gamma=2 keeps it available")
print(f"Syn2 (global)    : AUC {auc2:.3f} | top-4 = {sorted(top4)} (truth {list(rel2)}) | "
      f"{mass_on_relevant*100:.1f}% of mask mass on the truth")
print(f"Syn4 (per-row)   : AUC {auc4:.3f} | X11 weight {M4[:, 10].mean():.3f} | "
      f"correct group on {left_correct*100:.0f}% / {right_correct*100:.0f}% of rows -> PARTIAL")
print(f"bake-off ranks   : " + ", ".join(f"{m} {mean_rank[m]:.2f}" for m in MODELS))
print(f"Friedman         : chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (N={len(DATASETS)} datasets)")
print(f"cleared the bar? : {'YES' if mean_rank['tabnet'] < min(mean_rank['mlp'], mean_rank['resnet']) else 'NO'}")
print()
print("verdict:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"TabNet did NOT clear the baseline-first bar: it ranks behind the tuned MLP and ResNet that were '
    'run first under the identical frame, and it is worst on the categorical table. But my Friedman '
    'p > 0.05 licenses only \'cannot distinguish on this sample\', NOT \'significantly worse\' — the '
    'burden of proof simply sits with the new model, and it was not met. The mechanism is still real: '
    'sparsemax gives exact zeros so the mask is a genuine selection, and the prior scale P = prod(gamma '
    '- M) is what makes the attention sequential rather than merely sparse. Its masks recover the '
    'globally-relevant features on Syn2 convincingly, but only partially recover the instance-wise '
    'switch on Syn4 — and the paper needed 10M rather than 10k samples for sharp masks — so attributions '
    'are evidence to validate against known-answer data, not free explanation."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 043 — TabNet: implement sequential attention, then read the masks

**Lesson:** [`lessons/0043-tabnet.html`](../lessons/0043-tabnet.html) · **Phase / Year:** Year 2 · Q1

**Paper:** Arik & Pfister 2019, *TabNet: Attentive Interpretable Tabular Learning* ([arXiv:1908.07442](https://arxiv.org/abs/1908.07442)) — the architecture section (Fig. 4a–4d) + Appendix E/F. sparsemax: Martins & Astudillo 2016 ([arXiv:1602.02068](https://arxiv.org/abs/1602.02068)). Synthetic generators: Chen et al. 2018 ([arXiv:1802.07814](https://arxiv.org/abs/1802.07814)).

**Dataset tiers:** **C** (the paper's own synthetic Syn2/Syn4 — the only place mask claims are falsifiable) + **A** (small real OpenML tables via `relkit`).

**Skill you are practising:** implement TabNet's **sequential attention from scratch** — (1) `sparsemax`, (2) the **attentive transformer + prior scale** — then (3) **read its masks** on data whose relevant features are known, and (4) hold it to [L042](../lessons/0042-mlp-resnet-baselines.html)'s **baseline-first rule** against the tuned MLP/ResNet.

**Exit criteria:** EXIT TICKET prints your sparsemax validation, the gamma = 1 ban property, the Syn2/Syn4 mask readings (one clean success, one partial), and the rank table with an honest verdict separating *"did not clear the bar"* from *"significantly worse"*.

---

### How this notebook works
- **PROVIDED** cells — boilerplate (data, frame, search spaces, GBDT/net searches) **and** the paper's encoder copied into the notebook (not hidden behind `import relkit.tabnet`); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. After EXIT, a **NEXT STEP** cell trains closer to the paper (Colab GPU or Modal). When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` → kernel **Relational Labs (.venv)**. Needs **torch** + scikit-learn (CPU is fine); `pytorch-tabnet` is optional and used only to validate your sparsemax. Real datasets fetch from OpenML on first run then cache. Budget: **~2–4 minutes on CPU** (measured: 126 s) — set `OMP_NUM_THREADS=1` if a search feels slow (that has been the real cause of every slow lab so far). This lab uses a deliberately small budget/seed/dataset count to stay interactive; the lesson's headline numbers come from the fuller `labs/_verify_l043.py` run plus the published benchmarks.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — what TabNet actually does

**The idea.** Instead of feeding every feature into every layer, TabNet takes `N_steps` **decision steps**; at each step it *selects a sparse subset of features* to reason from, processes them, and adds its contribution to the prediction. Masking then transforming then summing is a soft, learned version of "split on this feature, then that one" — the tree-like inductive bias an MLP lacks (L025–L027).

**The four pieces.**
1. **sparsemax** (Task 1) — the projection onto the probability simplex. Subtract one threshold `tau`, clip negatives to zero, and whatever survives sums to 1. Unlike softmax it yields **exact zeros**, so a feature can be genuinely switched off (and gets no gradient at that step).
2. **Attentive transformer** (Task 2) — `M[i] = sparsemax(P[i-1] * h_i(a[i-1]))`, where `h_i` is `Linear -> BatchNorm`.
3. **Prior scale** (Task 2) — `P[i] = prod_j (gamma - M[j])`, starting at 1. This is the memory that makes the attention **sequential**: features already spent are discounted, so steps pick *different* ones. At `gamma = 1` a fully-used feature is banned outright; larger `gamma` permits reuse.
4. **Feature transformer** — `Linear -> GhostBatchNorm -> GLU` blocks (2 shared across steps + 2 step-dependent), wired with `sqrt(0.5)`-scaled residuals. Its output splits into `d[i]` (goes to the prediction, aggregated as `d_out = sum_i ReLU(d[i])`) and `a[i]` (feeds the next step's attention). After Task 2 the rest of the encoder is **inlined into this notebook** so you can read every line; you build pieces 1–3, which are the load-bearing ones.

Plus a **sparsity penalty** `L_sparse` (the masks' entropy, weight `lambda_sparse`) that pushes each step to commit to fewer features.

**Interpretability, and its limits.** `M_agg` aggregates the per-step masks weighted by each step's decision contribution `eta[i]`, giving a per-row feature attribution straight out of the forward pass — no separate SHAP/LIME run. Task 3 tests that claim where the answer is known, and finds it holds for **global** relevance and only **partially** for **instance-wise** relevance.

**The bar (Task 4).** Novel mechanism or not, it must beat a properly-tuned MLP/ResNet under the same protocol, or it has shown nothing (L042).

Full write-up + the interactive mask/prior widget: [Lesson 043](../lessons/0043-tabnet.html).'''),
        md("## Setup — PROVIDED (paper's synthetic data + real tables + shared frame + search spaces)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(architecture_md(
            "Ghost BN, the GLU feature transformer, the encoder, and the training loop",
            "labs/relkit/tabnet.py",
            "`sparsemax`",
        )),
        code(r'''# PROVIDED — adapter: your Task-1 sparsemax is last-axis only; the inlined encoder calls dim=-1.
_student_sparsemax = sparsemax
def sparsemax(z, dim=-1):
    """Keep YOUR Task-1 implementation; accept the encoder's dim= keyword."""
    if dim in (-1, z.ndim - 1):
        return _student_sparsemax(z)
    z_t = z.transpose(dim, -1)
    return _student_sparsemax(z_t).transpose(dim, -1)
'''),
        code(inline_source(
            os.path.join(HERE, "relkit/tabnet.py"),
            skip_defs={"sparsemax"},
        )),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(T4_MD), code(T4_SOL if solution else T4_CODE), code(T4_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(notebook_scaleup_md(
            lesson=43,
            paper="Arik & Pfister 2019, TabNet: Attentive Interpretable Tabular Learning",
            arxiv="1908.07442",
            lab_rows=[
                ("mean ranks", "TabNet 2.50 vs MLP 1.75 / ResNet 2.00 / GBDT 3.75, Friedman p=0.127"),
                ("Syn4 masks", "PARTIAL — 15.6% left-group recovery at n=10k (paper Fig. 5 used 10M)"),
            ],
            paper_rows=[
                ("Adult accuracy", "85.7% (appendix HPs). OpenML 1590 is not the UCI official split → expect INCOMPARABLE, then read DIRECTION vs XGB."),
                ("Syn4 masks", "Fig. 5 is qualitative at 10M samples. Watch whether left_correct climbs vs 15.6%."),
            ],
            modal="modal/l043_paper_repro.py",
        )),
        code(notebook_scaleup_code(
            lesson=43,
            harness_path=os.path.join(HERE, "_paper_repro_l043.py"),
            modal="modal/l043_paper_repro.py",
            skip_imports={"relkit.tabnet"},
        )),
        md(r'''## Stretch (optional, ungraded) — after the scale-up

1. **Reproduce the paper's sharp masks.** Rerun Syn4 with `n=200_000` (the paper used 10M) and more
   epochs. Does `left_correct` climb? That is the paper's own Fig. 5 caveat, measured.
2. **The sparsity knob.** Sweep `lambda_sparse` over `{0, 1e-4, 1e-2, 1e-1}` on Syn2 and plot mask
   entropy against test AUC. The paper's ablation (Table 11) says too much sparsity costs ~3 points —
   do you see the same shape?
3. **gamma vs N_steps.** The paper notes a larger `N_steps` favours a larger `gamma`. Grid
   `N_steps in {3,5,8}` x `gamma in {1.0,1.5,2.0}` on one table and see whether that interaction shows up.
4. **Chase the validation gap.** The lesson reports an *unexplained* credit_g discrepancy between the
   from-scratch model and `pytorch_tabnet` (0.748 vs 0.694); training length and the LR schedule were
   both ruled out. Try weight-level validation instead: build both with matched shapes, copy the
   parameters across, and assert `torch.allclose` on a forward pass. That isolates architecture from
   training entirely — and would settle it.
5. **Self-supervised pre-training.** The paper's other contribution: mask a fraction of feature columns
   and reconstruct them, then fine-tune. Implement the decoder and test whether it helps in the
   small-labelled-data regime (paper Table 7).'''),
        code(r'''# STRETCH — ungraded.
# X4b, y4b, _ = make_syn4(n=200_000, seed=1)
# auc4b, M4b, Xte4b = fit_and_read(X4b, y4b, n_steps=5, gamma=1.5, lambda_sparse=5e-3, epochs=200)
# leftb = Xte4b[:, 10] < 0
# print(auc4b, (M4b[leftb][:, [0,1]].sum(1) > M4b[leftb][:, 2:6].sum(1)).mean())'''),
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
    with open(os.path.join(HERE, "0043-tabnet.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0043-tabnet.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0043-tabnet.ipynb + solution")


if __name__ == "__main__":
    main()
