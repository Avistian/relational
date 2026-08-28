"""Build Lab 045 (TabTransformer — contextual categorical embeddings + RTD pre-training) — student + solution.

Honours NOTES standards #21 (lab ships with the lesson), #22 (build from scratch; libraries only validate),
#23 (never conclude from one dataset) and #24 (paper-mirror: implement the paper's mechanism, use a clean
head-to-head, ship a reproducibility ledger).

L032 built the TabTransformer FORWARD pass from scratch. L045 promotes it to a *trained* model and adds the
two things this lesson is about:
  Task 1 — RTD corruption (`corrupt_categorical`): the ELECTRA-style pretext that needs no labels.
  Task 2 — the RTD pre-training STEP: corrupt -> contextual encoder -> per-column detector -> BCE.
  Task 3 — the bake-off: CONTEXTUAL (n_layers=3) vs CONTEXT-FREE (n_layers=0, the L031/L032 model) vs CatBoost.
  Task 4 — label efficiency: pre-train on unlabeled rows + fine-tune on a few labels vs from-scratch.

`relkit.tabtransformer` holds the from-scratch model (built from the paper); torch's own attention kernels
are imported ONLY to validate in labs/_check_l045.py — never used to build the model.

Run: .venv/bin/python labs/_build_l045.py
Execute the solution: cd labs && OMP_NUM_THREADS=1 ../.venv/bin/jupyter nbconvert --to notebook \
    --execute --inplace solutions/0045-tabtransformer.ipynb
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


SETUP = r'''# PROVIDED — imports, the categorical-rich tables, the shared frame, and the budgets. Just run.
# The TabTransformer *architecture* is inlined in a later cell so you can read it (Huang 2020, Fig. 1 + §3).
# `relkit.tabtransformer` here is only a data helper + a Task-1 checker (NOTES #22 / #25).
# The attention itself you already built forward-only in Lab 032; it is re-validated in labs/_check_l045.py.
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
from scipy.stats import rankdata, friedmanchisquare
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
from relkit.tabtransformer import (
    frame_categorical,                                         # data helper (not the model)
    corrupt_categorical as relkit_corrupt,                     # checker only (NOTES #22)
)

DEVICE = "cpu"
DATASETS  = ["credit_g", "adult"]     # categorical-rich; adult subsampled below (churn is numeric-heavy)
SUBSAMPLE = {"adult": 3000}           # keep CPU cost bounded — a down-scaled demonstration (#20/#23)
SEEDS = [0, 1]                        # smaller than the lesson's 3-seed/3-dataset run, to stay interactive
EPOCHS, PATIENCE, BS, LR = 50, 10, 256, 1e-3
TT_CFG = dict(d=32, n_layers=3, n_heads=4, head_hidden=128, dropout=0.1)   # CONTEXTUAL
CF_CFG = dict(d=32, n_layers=0,          head_hidden=128, dropout=0.1)     # CONTEXT-FREE (L031/L032)
# Task 4 needs a LARGE unlabeled pool so pre-training has something to learn and fine-tuning does not
# catastrophically forget (a small pool was what produced negative lifts — see the reproducibility ledger).
SEMI_N, SEMI_FRAC, PRE_EPOCHS, FT_LR = 14000, 0.03, 25, 5e-4

def load_frame(name, cap=None):
    """Encode a table into TabTransformer's inputs: integer-coded categoricals + standardised numerics.
    `cap` (or the SUBSAMPLE default) bounds the row count so the lab stays CPU-cheap."""
    Xdf, y = load_tier_a(name)
    cap = cap if cap is not None else SUBSAMPLE.get(name)
    if cap and len(Xdf) > cap:
        idx, _ = train_test_split(np.arange(len(Xdf)), train_size=cap, random_state=0, stratify=y)
        Xdf, y = Xdf.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(Xdf)
    return {"Xdf": Xdf, "y": y.to_numpy().astype(np.float32), "Xcat": Xcat, "Xnum": Xnum,
            "cards": cards, "cat_names": cat_names, "num_names": num_names}

def split_idx(n, y, seed):
    """The SHARED frame — identical train/val/test for every arm (L020 contract, L042 protocol)."""
    tr, te = train_test_split(np.arange(n), test_size=0.30, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
    return tr, va, te

def run_tt(fr, tr, va, te, cfg, seed):
    """Train one TabTransformer arm (cfg picks CONTEXTUAL vs CONTEXT-FREE) and score it on test."""
    torch.manual_seed(seed)
    m = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **cfg)
    m, val = train_tabtransformer(m, fr["Xcat"][tr], fr["Xnum"][tr], fr["y"][tr],
                                  fr["Xcat"][va], fr["Xnum"][va], fr["y"][va],
                                  lr=LR, max_epochs=EPOCHS, patience=PATIENCE, batch_size=BS, seed=seed)
    return val, tabtransformer_auc(m, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])

def run_catboost(fr, tr, va, te, seed):
    """CatBoost with NATIVE categorical handling (its strength on categorical-rich data) — the honest bar."""
    from catboost import CatBoostClassifier
    Xdf = fr["Xdf"]; cat_idx = [Xdf.columns.get_loc(c) for c in fr["cat_names"]]
    Xstr = Xdf.copy()
    for c in fr["cat_names"]:
        Xstr[c] = Xstr[c].map(str).astype(object)     # plain str (CatBoost rejects category-dtype NaN)
    clf = CatBoostClassifier(depth=6, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0,
                             random_seed=seed, thread_count=2, verbose=0, allow_writing_files=False)
    clf.fit(Xstr.iloc[tr], fr["y"][tr].astype(int), cat_features=cat_idx,
            eval_set=(Xstr.iloc[va], fr["y"][va].astype(int)))
    return roc_auc_score(fr["y"][te], clf.predict_proba(Xstr.iloc[te])[:, 1])

print("setup ok — torch", torch.__version__)'''


# --------------------------------------------------------------------------- Task 1: RTD corruption
T1_MD = r'''## Task 1 — `corrupt_categorical`: the label-free pretext (Replaced Token Detection)

**Goal.** Write the corruption that powers self-supervised pre-training: independently, with probability
`p`, replace each categorical token with **another category drawn uniformly** from that column's range,
and return the 0/1 **label** the detector must recover — *which cells were tampered with?*

**Why it needs no labels.** The target is manufactured from the row itself, so *any* unlabeled row is
training signal. Detecting a swap forces the encoder to learn what a **coherent row** looks like — which
category values plausibly co-occur — the exact structure that then transfers to the real task
([Huang §3.3](https://arxiv.org/abs/2012.06678), ELECTRA-style).

**The one subtlety.** A uniform redraw can land on the *original* value. That cell is **not** replaced, so
its label must be 0. Define the label by comparing the corrupted tokens to the originals, not by the coin
flip. That is why the *effective* replaced fraction is `p·(1 − 1/card)`, strictly below `p`.'''

T1_CODE = r'''# TODO — implement the RTD corruption. Fill every ____.
def corrupt_categorical(Xcat, cards, p, generator):
    """Xcat: LongTensor [N, m] integer-coded categoricals. cards[j] = # categories in column j.
    Returns (Xcorrupt [N, m] long, replaced [N, m] float 0/1)."""
    N, m = Xcat.shape
    # 1) coin flip per cell: replace with probability p
    replace_mask = torch.rand(N, m, generator=generator) < p
    # 2) a uniform random category per column (the candidate replacement)
    rand_vals = torch.zeros(N, m, dtype=torch.long)
    for j, c in enumerate(cards):
        rand_vals[:, j] = torch.randint(0, max(c, 1), (N,), generator=generator)
    # 3) where the coin said "replace", swap in the random category; else keep the original
    Xcorrupt = ____
    # 4) the LABEL is "did the value actually change?" — a redraw equal to the original counts as UNCHANGED
    replaced = ____
    return Xcorrupt, replaced

# quick look
g = torch.Generator().manual_seed(0)
demo = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
xc, rep = corrupt_categorical(demo, [3, 3, 3], 0.5, g)
print("original :\n", demo.numpy())
print("corrupted:\n", xc.numpy())
print("replaced :\n", rep.numpy(), "  (1 = detector must flag this cell)")'''

T1_SOL = (T1_CODE
          .replace("""    # 3) where the coin said "replace", swap in the random category; else keep the original
    Xcorrupt = ____""",
                   """    # 3) where the coin said "replace", swap in the random category; else keep the original
    Xcorrupt = torch.where(replace_mask, rand_vals, Xcat)""")
          .replace("""    # 4) the LABEL is "did the value actually change?" — a redraw equal to the original counts as UNCHANGED
    replaced = ____""",
                   """    # 4) the LABEL is "did the value actually change?" — a redraw equal to the original counts as UNCHANGED
    replaced = (Xcorrupt != Xcat).float()""")
          .replace("# TODO — implement the RTD corruption. Fill every ____.",
                   "# SOLUTION — RTD corruption (Huang §3.3, ELECTRA-style)"))

T1_CHECK = r'''# CHECK — the invariants that make RTD a valid label-free pretext (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

rng = np.random.default_rng(0)
cards = [2, 5, 10, 3, 7]
X = torch.tensor(np.stack([rng.integers(0, c, size=4000) for c in cards], axis=1), dtype=torch.long)

# p = 0 changes nothing -> no signal without corruption
g0 = torch.Generator().manual_seed(1)
xc0, rep0 = corrupt_categorical(X, cards, 0.0, g0)
chk("p=0 corrupts nothing", bool((xc0 == X).all()) and float(rep0.sum()) == 0.0)

# a labelled cell truly changed; an unlabelled cell truly did not
g1 = torch.Generator().manual_seed(2)
xc, rep = corrupt_categorical(X, cards, 0.30, g1)
chk("every replaced=1 cell actually changed value", bool((xc[rep == 1] != X[rep == 1]).all()))
chk("every replaced=0 cell is unchanged", bool((xc[rep == 0] == X[rep == 0]).all()))

# effective replaced fraction = mean_j p*(1 - 1/card_j), STRICTLY below p (collisions relabelled)
eff = rep.mean().item()
expected = float(np.mean([0.30 * (1 - 1 / c) for c in cards]))
chk("effective replaced fraction matches p*(1-1/card), below p", abs(eff - expected) < 0.02 and eff < 0.30,
    f"eff {eff:.3f} vs expected {expected:.3f} (< p=0.30)")

# deterministic in (p, seed); higher p replaces at least as much
ga, gb = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
chk("deterministic given the generator seed",
    bool((corrupt_categorical(X, cards, 0.3, ga)[0] == corrupt_categorical(X, cards, 0.3, gb)[0]).all()))
g2, g3 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
chk("higher p replaces more (in expectation)",
    corrupt_categorical(X, cards, 0.6, g2)[1].mean() > corrupt_categorical(X, cards, 0.2, g3)[1].mean())

# VALIDATION against the from-scratch reference (NOTES #22)
ga, gb = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
mine = corrupt_categorical(X, cards, 0.3, ga)[0]
ref = relkit_corrupt(X, cards, 0.3, gb)[0]
chk("VALIDATED against relkit.tabtransformer.corrupt_categorical", bool((mine == ref).all()))
print("\nTask 1", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------- Task 2: the RTD pre-training step
T2_MD = r'''## Task 2 — the pre-training STEP: corrupt → contextual encoder → per-column detector

**Goal.** Write one self-supervised training step. Given a batch of clean categorical rows: corrupt them
(Task 1), run the **corrupted** tokens through the encoder to get **contextual** embeddings, let a small
per-column detector predict the 0/1 replaced-label from each token's contextual vector, and return the
**binary cross-entropy** loss.

**Why contextual is the whole point.** A *context-free* embedding of a swapped category looks perfectly
normal on its own — the swap is only detectable by how it clashes with the rest of the row. So the detector
can only succeed if the encoder mixes columns together, which is exactly what the Transformer's
self-attention does ([L032](../lessons/0032-tabtransformer.html)). The pretext *rewards* contextualisation.

The detector (`RTDHead`, provided) is one tiny `Linear(d, 1)` per column. Only the **encoder**
(`model.embs` + `model.blocks`) and this throwaway head are trained here; the supervised head is fit later.'''

T2_CODE = r'''# TODO — fill every ____.  (corrupt_categorical from Task 1; RTDHead + model provided.)
import torch.nn as nn
bce = nn.BCEWithLogitsLoss()

def rtd_step(model, head, xc_batch, cards, p, generator):
    """One pretext step. Returns (loss, logits, replaced) — loss is what you backprop."""
    # 1) manufacture corrupted tokens + their replaced labels (no real labels used)
    xcorr, replaced = ____
    # 2) CONTEXTUAL embeddings of the CORRUPTED tokens (embeddings -> Transformer blocks)
    ctx = model.contextual(xcorr)              # [B, m, d]
    # 3) per-column detector logits, then BCE against the replaced labels
    logits = head(ctx)                         # [B, m]
    loss = ____
    return loss, logits, replaced

# --- PROVIDED harness: pre-train the encoder on credit_g's UNLABELED train features, report detector skill.
fr = load_frame("credit_g")
tr, va, te = split_idx(len(fr["y"]), fr["y"], 0)
Xunlab = fr["Xcat"][tr]                          # features only — labels are never touched here
torch.manual_seed(0)
enc = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG)
head = RTDHead(len(fr["cards"]), enc.d)
opt = torch.optim.AdamW(list(enc.embs.parameters()) + list(enc.blocks.parameters())
                        + list(head.parameters()), lr=1e-3)
gen = torch.Generator().manual_seed(0)
before = enc.embs[0].weight.detach().clone()
for ep in range(15):
    enc.train(); head.train()
    perm = torch.randperm(len(Xunlab), generator=gen)
    for st in range(0, len(Xunlab), BS):
        xb = Xunlab[perm[st:st + BS]]
        if xb.shape[0] < 2: continue
        opt.zero_grad()
        loss, logits, replaced = rtd_step(enc, head, xb, fr["cards"], 0.30, gen)
        loss.backward(); opt.step()
# detector AUC on a fresh corruption of the held-out val rows: can it spot the swaps?
enc.eval(); head.eval()
with torch.no_grad():
    xcv, repv = corrupt_categorical(fr["Xcat"][va], fr["cards"], 0.30, torch.Generator().manual_seed(99))
    scores = torch.sigmoid(head(enc.contextual(xcv))).numpy().ravel()
det_auc = roc_auc_score(repv.numpy().ravel(), scores)
moved = (enc.embs[0].weight.detach() - before).norm().item()
print(f"final pretext loss : {float(loss):.4f}")
print(f"detector AUC (spotting replaced tokens): {det_auc:.3f}   (0.5 = blind guessing)")
print(f"column-0 embedding moved during pre-training by L2 = {moved:.3f}")'''

T2_SOL = (T2_CODE
          .replace("""    # 1) manufacture corrupted tokens + their replaced labels (no real labels used)
    xcorr, replaced = ____""",
                   """    # 1) manufacture corrupted tokens + their replaced labels (no real labels used)
    xcorr, replaced = corrupt_categorical(xc_batch, cards, p, generator)""")
          .replace("""    logits = head(ctx)                         # [B, m]
    loss = ____""",
                   """    logits = head(ctx)                         # [B, m]
    loss = bce(logits, replaced)""")
          .replace("# TODO — fill every ____.  (corrupt_categorical from Task 1; RTDHead + model provided.)",
                   "# SOLUTION  (corrupt_categorical from Task 1; RTDHead + model provided.)"))

T2_CHECK = r'''# CHECK — the pretext actually taught the encoder something (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("pretext loss is a finite scalar", np.isfinite(float(loss)) and float(loss) > 0)
chk("detector beats blind guessing (AUC > 0.6) — context reveals the swaps", det_auc > 0.6,
    f"AUC {det_auc:.3f}")
chk("pre-training MOVED the encoder's embeddings (it learned from unlabeled rows)", moved > 1e-3,
    f"L2 move {moved:.3f}")

# the concept underneath: only the CONTEXTUAL encoder mixes columns; n_layers=0 leaves each token alone
torch.manual_seed(0)
ctxm = TabTransformer(fr["cards"], fr["Xnum"].shape[1], d=32, n_layers=2).eval()
cfm  = TabTransformer(fr["cards"], fr["Xnum"].shape[1], d=32, n_layers=0).eval()
r = fr["Xcat"][0:1].clone(); r2 = r.clone(); r2[0, 1] = (r[0, 1] + 1) % fr["cards"][1]   # flip a NEIGHBOUR
with torch.no_grad():
    move_ctx = (ctxm.contextual(r)[0, 0] - ctxm.contextual(r2)[0, 0]).norm().item()   # column 0's vector
    move_cf  = (cfm.contextual(r)[0, 0]  - cfm.contextual(r2)[0, 0]).norm().item()
chk("contextual: column 0's vector MOVES when a neighbour changes", move_ctx > 1e-4, f"{move_ctx:.3f}")
chk("context-free (n_layers=0): the SAME vector does NOT move", move_cf < 1e-9, f"{move_cf:.1e}")
print("\nTask 2", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------- Task 3: contextual vs context-free vs CatBoost
T3_MD = r'''## Task 3 — the bake-off: what did *context* buy, and where didn't it?

**Goal.** Race three models under one shared frame on categorical-rich tables, then summarise with **mean
ranks** and a Friedman test:

- **TabTransformer** — *contextual* categorical embeddings (`n_layers=3`).
- **Context-free MLP** — the **same class at `n_layers=0`**: entity embeddings with **no attention**. This
  *is* the static-embedding model from [L031/L032](../lessons/0031-entity-embeddings.html). Flipping one
  hyper-parameter isolates *exactly what contextualisation buys*.
- **CatBoost** — native categorical handling; the honest tree bar ([L042](../lessons/0042-mlp-resnet-baselines.html)'s baseline-first rule).

**Why this trio (NOTES #23/#24).** Contextual-vs-context-free is the paper's own ablation; adding the tree
keeps us honest about whether *any* neural approach is worth it here. Two small tables is a demonstration,
not proof — the strong claim stays cited to the paper's 15-dataset study.'''

T3_CODE = r'''# TODO — fill every ____.
def search_tt(fr, tr, va, te, cfg, seed):
    """Train one TabTransformer arm and keep its test score (val is only for early stopping inside run_tt)."""
    val, test = run_tt(fr, tr, va, te, cfg, seed)
    return test

MODELS = ["tabtransformer", "context_free", "catboost"]
table = {}
for name in DATASETS:
    fr = load_frame(name)
    print(f"\n=== {name}: {len(fr['y'])} rows | {len(fr['cat_names'])} cat | "
          f"{len(fr['num_names'])} num ===", flush=True)
    rows = {m: [] for m in MODELS}
    for s in SEEDS:
        tr, va, te = split_idx(len(fr["y"]), fr["y"], s)
        rows["tabtransformer"].append(search_tt(fr, tr, va, te, TT_CFG, s))
        # the CONTEXT-FREE arm is the SAME model with the attention removed -> n_layers = 0
        rows["context_free"].append(search_tt(fr, tr, va, te, ____, s))
        rows["catboost"].append(run_catboost(fr, tr, va, te, s))
        print("  seed {}: ".format(s) + " | ".join(f"{m} {rows[m][-1]:.3f}" for m in MODELS), flush=True)
    table[name] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}

# --- cross-dataset summary: mean ranks (1 = best per dataset) + Friedman
score = np.array([[table[d][m][0] for m in MODELS] for d in DATASETS])   # datasets x models
ranks = np.array([rankdata(-row, method="average") for row in score])    # HIGHEST score -> rank 1
mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(MODELS)}
fried = friedmanchisquare(*[score[:, i] for i in range(len(MODELS))])
ctx_beats_cf = sum(table[d]["tabtransformer"][0] > table[d]["context_free"][0] for d in DATASETS)

print("\nmean ranks:", {m: round(r, 2) for m, r in mean_rank.items()})
print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (k={len(MODELS)}, N={len(DATASETS)})")
print(f"contextual beats context-free on {ctx_beats_cf}/{len(DATASETS)} tables")'''

T3_SOL = (T3_CODE
          .replace('rows["context_free"].append(search_tt(fr, tr, va, te, ____, s))',
                   'rows["context_free"].append(search_tt(fr, tr, va, te, CF_CFG, s))')
          .replace("# TODO — fill every ____.", "# SOLUTION"))

T3_CHECK = r'''# CHECK — read the verdict the disciplined way (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("ranks are a valid per-dataset ranking of the 3 models",
    ranks.shape == (len(DATASETS), 3) and bool(np.allclose(ranks.sum(1), 6.0)))
chk("best score on each dataset got rank 1",
    all(ranks[i][np.argmax(score[i])] == 1 for i in range(len(DATASETS))))
chk("every arm ran on the same shared frame", True, f"datasets={DATASETS}, seeds={SEEDS}")

print(f"""
VERDICT — contextual (TabTransformer) mean rank {mean_rank['tabtransformer']:.2f}
          context-free MLP (n_layers=0)  mean rank {mean_rank['context_free']:.2f}
          CatBoost (native categoricals) mean rank {mean_rank['catboost']:.2f}
  contextual vs free  : {ctx_beats_cf}/{len(DATASETS)} tables here — the gap is TINY and WITHIN NOISE at
                        two tables, so it can (and here may) INVERT. That is the honest reading, not a bug.
  the ROBUST shape    : {'CatBoost still ranks best' if mean_rank['catboost'] <= min(mean_rank['tabtransformer'], mean_rank['context_free']) else 'a neural model led (unusual at this scale)'} — numeric features BYPASS the attention (the paper's known limit)
  Friedman p          : {fried.pvalue:.3f} -> {'a difference is detectable' if fried.pvalue < 0.05 else 'CANNOT distinguish these models on so few tables'}

Read this the disciplined way (NOTES #23). The contextual-vs-context-free difference is real but small; on
the lesson's FULLER run (labs/_verify_l045.py: 3 datasets x 3 seeds, 60 epochs) it resolves to mean ranks
TabTransformer 2.33 / context-free 2.67 / CatBoost 1.00 (contextual ahead on 2/3, Friedman p = 0.097). At
TWO tables you cannot expect that thin edge to survive the noise — you may see context-free win, as an
honest small experiment sometimes does. What IS robust at every scale here is the bottom line: TabTransformer
beats CatBoost on 0 tables, because the NUMERIC features never touch the attention. That gap — not the
contextual micro-edge — is what motivates FT-Transformer (L046), which tokenises the numeric features too.""")
print("Task 3", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------------------- Task 4: label efficiency
T4_MD = r'''## Task 4 — label efficiency: does pre-training beat training from scratch on few labels?

**Goal.** On a table with plenty of **unlabeled** rows but few **labeled** ones, compare two encoders:
(a) trained **from scratch** on the small labeled set, vs (b) **RTD pre-trained** on all the unlabeled
features (Task 2's pretext), then **gently fine-tuned** on the same small labeled set.

**Why "gently".** Fine-tuning re-uses the pre-trained encoder, so a large learning rate would overwrite
(catastrophically forget) what pre-training learned. A small fine-tune LR (`5e-4`, half the from-scratch
`1e-3`) protects the transferred representation — this was the fix that turned an unstable, sometimes
*negative* lift into a consistent one (see the reproducibility ledger).

**Why it matters.** This label-efficiency lever is the thing trees structurally lack: a GBDT cannot pre-train
on unlabeled rows. It is the paper's [§3.3](https://arxiv.org/abs/2012.06678) headline and the bridge to why
self-supervision matters for the relational setting, where labels are scarce but rows are plentiful.'''

T4_CODE = r'''# TODO — fill every ____.
fr = load_frame("adult", cap=SEMI_N)                  # a BIG frame so the unlabeled pool dwarfs the labels
FRAC = SEMI_FRAC                                       # a small labeled fraction; the rest stays unlabeled
scratch, pretrained = [], []
for s in SEEDS:
    tr, te = train_test_split(np.arange(len(fr["y"])), test_size=0.30, random_state=s, stratify=fr["y"])
    unlab = tr                                         # UNLABELED pool = every train row's features
    lab, _ = train_test_split(tr, train_size=max(int(len(tr) * FRAC), 60), random_state=s,
                              stratify=fr["y"][tr])
    lab_tr, lab_va = train_test_split(lab, test_size=0.25, random_state=s, stratify=fr["y"][lab])

    # (a) FROM SCRATCH on the few labels
    torch.manual_seed(s)
    ms = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG)
    ms, _ = train_tabtransformer(ms, fr["Xcat"][lab_tr], fr["Xnum"][lab_tr], fr["y"][lab_tr],
                                 fr["Xcat"][lab_va], fr["Xnum"][lab_va], fr["y"][lab_va],
                                 lr=LR, max_epochs=EPOCHS, patience=PATIENCE, seed=s)
    auc_scratch = tabtransformer_auc(ms, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])

    # (b) RTD PRE-TRAIN the encoder on the unlabeled pool, then GENTLY fine-tune on the same few labels
    torch.manual_seed(s)
    mp = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG)
    mp, _acc = pretrain_rtd(mp, fr["Xcat"][unlab], fr["cards"], replace_p=0.30,
                            max_epochs=PRE_EPOCHS, batch_size=BS, seed=s)
    # fine-tune: SAME training call, but with the gentle fine-tune LR to avoid catastrophic forgetting
    mp, _ = train_tabtransformer(mp, fr["Xcat"][lab_tr], fr["Xnum"][lab_tr], fr["y"][lab_tr],
                                 fr["Xcat"][lab_va], fr["Xnum"][lab_va], fr["y"][lab_va],
                                 lr=____, max_epochs=EPOCHS, patience=PATIENCE, seed=s)
    auc_pre = tabtransformer_auc(mp, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])

    scratch.append(auc_scratch); pretrained.append(auc_pre)
    print(f"  seed {s} (~{len(lab_tr)} labeled, {len(unlab)} unlabeled): "
          f"scratch {auc_scratch:.3f} | pretrain+finetune {auc_pre:.3f} | Δ {auc_pre - auc_scratch:+.3f}",
          flush=True)

lift = float(np.mean(pretrained) - np.mean(scratch))
print(f"\nfrac {FRAC:.0%}: scratch {np.mean(scratch):.3f} -> pretrain+finetune {np.mean(pretrained):.3f} "
      f"(lift {lift:+.3f})")'''

T4_SOL = (T4_CODE
          .replace("""                                 lr=____, max_epochs=EPOCHS, patience=PATIENCE, seed=s)""",
                   """                                 lr=FT_LR, max_epochs=EPOCHS, patience=PATIENCE, seed=s)""")
          .replace("# TODO — fill every ____.", "# SOLUTION"))

T4_CHECK = r'''# CHECK — the label-efficiency lever, read honestly (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("both arms produced valid AUCs", all(0.5 < a < 1.0 for a in scratch + pretrained))
chk("used a GENTLE fine-tune LR (< the from-scratch LR)", FT_LR < LR, f"ft_lr={FT_LR} < lr={LR}")
chk("pre-training used ONLY features (no labels touched)", True, "unlabeled pool = train-row features")
print(f"""
This run: {SEMI_N} rows, {SEMI_FRAC:.0%} labels, {PRE_EPOCHS} pretrain epochs, {len(SEEDS)} seeds -> lift {lift:+.3f}.
The lesson's fuller run (labs/_verify_l045.py: adult, 16000 rows, 3 seeds, 30 pretrain epochs):
  3% labels : scratch 0.825 -> pretrain+finetune 0.833  (lift +0.008, all 3 seeds positive)
  10% labels: scratch 0.861 -> pretrain+finetune 0.862  (lift +0.001)
The lift is REAL but SMALL, and it shrinks as labels grow — pre-training helps most when labels are
scarcest. At this down-scaled budget ({len(SEEDS)} seeds) it lands near zero and can wobble either side of
it; two things make it collapse to NEGATIVE, and you saw both diagnosed in the ledger: too SMALL an
unlabeled pool, and too LARGE a fine-tune LR (catastrophic forgetting). The honest takeaway: self-supervision
is a genuine lever trees lack, but on small flat tables it is a modest edge, not a revolution.""")
print("Task 4", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------------------------------------ exit ticket
EXIT_MD = r'''## EXIT TICKET

Paste this output to your teacher, or just say *"lab done."*'''

EXIT_CODE = r'''# EXIT TICKET
print("=== LAB 045 — TabTransformer (contextual embeddings + RTD pre-training) ===")
print(f"RTD corruption   : validated vs reference; effective fraction = p*(1-1/card) < p")
print(f"pretext step     : detector AUC {det_auc:.3f} (>0.5 = context reveals swaps); embeddings moved")
print(f"contextual test  : column moves under a neighbour flip WITH attention, not at n_layers=0")
print(f"bake-off ranks   : " + ", ".join(f"{m} {mean_rank[m]:.2f}" for m in MODELS))
print(f"Friedman         : chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (N={len(DATASETS)} tables)")
print(f"context vs free  : contextual beats context-free on {ctx_beats_cf}/{len(DATASETS)} tables")
print(f"label efficiency : scratch {np.mean(scratch):.3f} -> pretrain+finetune {np.mean(pretrained):.3f} (lift {lift:+.3f})")
print()
print("in one sentence, what does the Transformer buy over the static (context-free) embedding, and what is its limitation?:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"The Transformer makes each categorical embedding CONTEXTUAL — a column\'s vector is re-mixed by '
    'self-attention over the other columns in the SAME row, so the same category can mean different things '
    'in different rows (and the RTD pretext can only spot a swapped token because context exposes the '
    'clash). That buys a small, consistent edge over the L031/L032 STATIC embedding, plus a label-efficiency '
    'lever trees lack (pre-train on unlabeled rows, gently fine-tune). Its limitation: only CATEGORICALS go '
    'through attention — NUMERIC features bypass the Transformer entirely (just LayerNorm + concat), so on '
    'numeric-heavy or flat tables it does not beat CatBoost. FT-Transformer (L046) removes exactly this '
    'limit by tokenising numeric features too."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 045 — TabTransformer: contextual categorical embeddings + self-supervised pre-training

**Lesson:** [`lessons/0045-tabtransformer.html`](../lessons/0045-tabtransformer.html) · **Phase / Year:** Year 2 · Q1

**Paper:** Huang, Khetan, Cvitkovic & Karnin 2020, *TabTransformer: Tabular Data Modeling Using Contextual Embeddings* ([arXiv:2012.06678](https://arxiv.org/abs/2012.06678)) — Fig. 1 (architecture), §3.1–3.2 (column embedding + Transformer), §3.3 (RTD pre-training). Self-attention: Vaswani et al. 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)). Static entity embeddings: [L031](../lessons/0031-entity-embeddings.html)/[L032](../lessons/0032-tabtransformer.html).

**Dataset tier:** **A** — small real OpenML tables via `relkit` (CPU-cheap; deliberately NOT representative — see the #23 notes in Tasks 3–4).

**Skill you are practising:** promote the forward-only model you built in [L032](../lessons/0032-tabtransformer.html) to a *trained*, *pre-trainable* one — (1) the **RTD corruption**, (2) the **self-supervised pre-training step**, then (3) race **contextual vs context-free (n_layers=0) vs CatBoost**, and (4) measure the **label-efficiency** payoff of pre-training. Name the limitation: numeric features bypass the attention.

**Exit criteria:** EXIT TICKET prints your RTD validation, the pretext-step detector AUC, the contextual-vs-context-free probe, the bake-off ranks, the label-efficiency lift, and one sentence — *what does context buy, and where does it stop?*

---

### How this notebook works
- **PROVIDED** cells — boilerplate (data, frame, budgets, CatBoost/pretrain harnesses) **and** the paper's encoder / train / RTD loops copied into the notebook (not hidden behind `import relkit.tabtransformer`); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. After EXIT, a **NEXT STEP** cell trains closer to the paper (Colab GPU or Modal). When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` → kernel **Relational Labs (.venv)**. Needs **torch** + scikit-learn + **catboost** (CPU is fine). Real datasets fetch from OpenML on first run then cache. Budget: **~8–12 minutes on CPU** — set `OMP_NUM_THREADS=1` if a search feels slow (that has been the real cause of every slow lab so far). This lab uses a deliberately small budget/seed/dataset count to stay interactive; the lesson's headline numbers come from the fuller `labs/_verify_l045.py` run plus the paper's 15-dataset benchmark.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — what TabTransformer adds to the static embedding

**The one idea.** In [L031/L032](../lessons/0031-entity-embeddings.html) each category owned a **static**
embedding vector — the same vector no matter what row it appeared in. TabTransformer makes those embeddings
**contextual**: it runs the row's categorical tokens through **N Transformer blocks**, so self-attention
lets each column's vector absorb the other columns *in that row*. The same `education = Masters` now reads
differently next to `age = 22` than next to `age = 55`.

**The architecture (Huang 2020, Fig. 1).**
1. **Column embedding** — each categorical column has its own embedding table (the L031 entity embedding).
2. **Transformer stack** — `n_layers` blocks of multi-head self-attention + FFN turn the static tokens into
   **contextual** ones. `n_layers = 0` is the *context-free ablation* — literally the L031/L032 model.
3. **Numerics bypass** — continuous features skip the Transformer, get a `LayerNorm`, and are **concatenated**
   to the flattened contextual tokens. **This is the known limitation:** numbers never attend to anything.
4. **MLP head** — the concatenated vector → MLP → one logit.

**Self-supervised pre-training (§3.3).** Because the pretext task — **Replaced Token Detection** — is built
from the row itself, TabTransformer can learn from **unlabeled** rows: corrupt some categorical tokens, and
train a per-column detector to flag them. Detecting a swap *requires* context (a swapped value only looks
wrong next to its neighbours), so the pretext directly sharpens the contextual encoder. Then fine-tune on a
few labels. Trees cannot do this.

**The honest verdict (Tasks 3–4).** Context gives a small, consistent edge over the static embedding, and
pre-training adds a modest label-efficiency lift — but CatBoost still wins the flat-table metric because
numerics bypass the attention. That gap is exactly what **FT-Transformer (L046)** closes.

Full write-up + the static-vs-contextual and RTD widgets: [Lesson 045](../lessons/0045-tabtransformer.html).'''),
        md("## Setup — PROVIDED (categorical-rich tables + shared frame + budgets)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(architecture_md(
            "column embeddings, the Transformer stack, RTDHead, and the train / pre-train loops",
            "labs/relkit/tabtransformer.py",
            "`corrupt_categorical`",
        )),
        code(inline_source(
            os.path.join(HERE, "relkit/tabtransformer.py"),
            skip_defs={"corrupt_categorical"},
        )),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(T4_MD), code(T4_SOL if solution else T4_CODE), code(T4_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(notebook_scaleup_md(
            lesson=45,
            paper="Huang, Khetan, Cvitkovic & Karnin 2020, TabTransformer",
            arxiv="2012.06678",
            lab_rows=[
                ("mean ranks", "TT 2.33 vs context-free 2.67 vs CatBoost 1.00, Friedman p=0.097"),
                ("RTD lift", "+0.008 AUC at 3% labels on adult-16k (paper ~+2.1% at benchmark scale)"),
            ],
            paper_rows=[
                ("+1.0% AUC over deep baselines", "15-dataset mean. We measure contextual − context-free on full Adult → INCOMPARABLE to the 15-dataset figure; read DIRECTION."),
                ("matches GBDTs", "Paper does not claim a win over trees. CatBoost winning here is compatible."),
                ("semi-supervised ~+2.1%", "Needs a large unlabeled pool. Scale-up uses full Adult at 3% labels."),
            ],
            modal="modal/l045_paper_repro.py",
        )),
        code(notebook_scaleup_code(
            lesson=45,
            harness_path=os.path.join(HERE, "_paper_repro_l045.py"),
            modal="modal/l045_paper_repro.py",
            skip_imports={"relkit.tabtransformer"},
        )),
        md(r'''## Stretch (optional, ungraded) — after the scale-up

1. **Depth of context.** Sweep `n_layers` ∈ {0, 1, 2, 3} on `credit_g` at fixed `d`. Does more attention
   keep helping, or does one block capture most of the contextual signal on a 13-column table?
2. **How much corruption?** Vary `replace_p` ∈ {0.1, 0.3, 0.5} in pre-training and watch the detector AUC
   and the downstream lift. Too little signal vs too much noise — where is the sweet spot the paper picks?
3. **The numeric-bypass limit, made visible.** Take a numeric-heavy table (e.g. `churn`) and re-run the
   bake-off. TabTransformer's edge over CatBoost should *shrink* — because most of the signal never touches
   the attention. This is the concrete motivation for FT-Transformer (L046).
4. **Freeze vs fine-tune.** After pre-training, try fine-tuning ONLY the head (freeze `embs` + `blocks`) vs
   fine-tuning everything gently. Which transfers the pre-trained representation better at 3% labels?'''),
        code(r'''# STRETCH — ungraded.
# fr = load_frame("credit_g")
# for L in (0, 1, 2, 3):
#     cfg = dict(d=32, n_layers=L, n_heads=4, head_hidden=128, dropout=0.1)
#     accs = []
#     for s in SEEDS:
#         tr, va, te = split_idx(len(fr["y"]), fr["y"], s)
#         accs.append(run_tt(fr, tr, va, te, cfg, s)[1])
#     print(f"n_layers={L}: test AUC {np.mean(accs):.3f}")'''),
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
    with open(os.path.join(HERE, "0045-tabtransformer.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0045-tabtransformer.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0045-tabtransformer.ipynb + solution")


if __name__ == "__main__":
    main()
