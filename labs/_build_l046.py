"""Build Lab 046 (FT-Transformer — feature tokenizer + [CLS] readout) — student + solution.

Honours NOTES standards #21 (lab ships with the lesson), #22 (build from scratch; libraries only validate),
#23 (never conclude from one dataset), #24 (paper-mirror), #25 (visible implementation + paper-results track).

FT-Transformer (Gorishniy 2021, §3.3) is a targeted edit to the TabTransformer you trained in L045: make
EVERY feature a token — numerics included — and read the row out through a learned [CLS] token. The two
load-bearing pieces the student writes:
  Task 1 — `affine_numeric_tokens`: the numeric half of the tokenizer, T_j = b_j + x_j·W_j.
  Task 2 — `prepend_cls`: prepend the learned [CLS] token to the feature-token sequence.
Both are inlined-and-KEPT: the notebook copies relkit/ft_transformer.py but SKIPS these two names, so the
model the student trains calls THEIR functions (not a hidden package copy).
  Task 3 — the bake-off: FT-T vs TabTransformer (numeric-bypass) vs MLP vs CatBoost, mean ranks + Friedman.
  Task 4 — the numerics-attend probe: a numeric change moves FT-T's [CLS] readout but not TabTransformer's.

Run: .venv/bin/python labs/_build_l046.py
Execute the solution: cd labs && OMP_NUM_THREADS=1 ../.venv/bin/jupyter nbconvert --to notebook \
    --execute --inplace solutions/0046-ft-transformer.ipynb
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


SETUP = r'''# PROVIDED — imports, tables, the shared frame, and budgets. Just run.
# The FT-Transformer *model* is inlined in a later cell so you can read it (Gorishniy 2021, §3.3, Fig. 2).
# `relkit.ft_transformer` here provides only the two Task checkers; the baselines (TabTransformer from L045,
# MLP from L042) are imported for the bake-off — the model being TAUGHT (FT-T) is the one you build (NOTES #25).
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
from scipy.stats import rankdata, friedmanchisquare
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
from relkit.tabtransformer import (
    frame_categorical,                                         # data helper (not the model)
    TabTransformer, train_tabtransformer, tabtransformer_auc,  # L045 baseline (numeric-bypass)
)
from relkit.nets import TabMLP, train_net, net_auc             # L042 baseline (strong deep model)
from relkit.ft_transformer import (
    affine_numeric_tokens as relkit_affine,                    # checker only (NOTES #22)
    prepend_cls as relkit_prepend,                             # checker only (NOTES #22)
)

DEVICE = "cpu"
DATASETS  = ["adult", "churn"]        # a mixed table + a numeric-heavy one, to show the numeric-bypass fix
SUBSAMPLE = {"adult": 3000}           # keep CPU cost bounded — a down-scaled demonstration (#20/#23)
SEEDS = [0, 1]                        # smaller than the lesson's 4-table/3-seed run, to stay interactive
EPOCHS, PATIENCE, BS, LR = 50, 10, 256, 1e-3
FT_CFG = dict(d=64, n_layers=3, n_heads=8, dropout=0.1)                    # FT-Transformer
TT_CFG = dict(d=32, n_layers=3, n_heads=4, head_hidden=128, dropout=0.1)   # TabTransformer (L045)

def load_frame(name, cap=None):
    """Encode a table into three shared frames: integer cats + scaled nums (transformers), a dense one-hot
    matrix (MLP), and the raw DataFrame (CatBoost native categoricals). `cap` bounds rows for CPU speed."""
    Xdf, y = load_tier_a(name)
    cap = cap if cap is not None else SUBSAMPLE.get(name)
    if cap and len(Xdf) > cap:
        idx, _ = train_test_split(np.arange(len(Xdf)), train_size=cap, random_state=0, stratify=y)
        Xdf, y = Xdf.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(Xdf)
    ct = ColumnTransformer([("num", StandardScaler(), num_names),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_names)])
    Xdense = ct.fit_transform(Xdf).astype(np.float32)
    return {"Xdf": Xdf, "y": y.to_numpy().astype(np.float32), "Xcat": Xcat, "Xnum": Xnum, "cards": cards,
            "cat_names": cat_names, "num_names": num_names, "Xdense": Xdense,
            "num_frac": len(num_names) / max(len(num_names) + len(cat_names), 1)}

def split_idx(n, y, seed):
    """The SHARED frame — identical train/val/test for every arm (L020 contract, L042 protocol)."""
    tr, te = train_test_split(np.arange(n), test_size=0.30, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=y[tr])
    return tr, va, te

print("setup ok — torch", torch.__version__)'''


# --------------------------------------------------------------------------- Task 1: numeric tokenizer
T1_MD = r'''## Task 1 — `affine_numeric_tokens`: turn a NUMBER into a token

**Goal.** Write the numeric half of FT-Transformer's Feature Tokenizer. Each numeric feature *j* gets its
own learned weight vector `W_j` and bias `b_j` (both of width *d*), and its token is the **affine** map

$$T_j = b_j + x_j \cdot W_j$$

**Why affine, not a lookup.** TabTransformer only knew how to tokenise *categories* (an embedding lookup),
so its numerics bypassed the attention entirely. Embedding a number as `b_j + x_j·W_j` places the scalar on
a learned direction `W_j`, preserving order (39 and 40 land near each other) — so a *number* becomes a
first-class token that can attend. This one function is the whole idea of the lesson.

**Worked micro-example.** With `W_j = [2, 0]`, `b_j = [1, 5]`, and `x_j = 3`: the token is
`[1, 5] + 3·[2, 0] = [7, 5]`. Bump `x_j` to 4 and it becomes `[9, 5]` — moved by `Δx·W_j = [2, 0]`.'''

T1_CODE = r'''# TODO — implement the numeric tokenizer. Fill every ____.
def affine_numeric_tokens(x_num, weight, bias):
    """x_num: [B, n_num] float · weight, bias: [n_num, d] · returns tokens [B, n_num, d].
    Each numeric feature j -> token_j = bias_j + x_num[:, j] * weight_j."""
    # broadcast the scalar over the d embedding dims: [B, n_num, 1] * [n_num, d] -> [B, n_num, d]
    return ____

# quick look on the worked micro-example
W = torch.tensor([[2., 0.]]); b = torch.tensor([[1., 5.]])   # one numeric feature, d=2
x = torch.tensor([[3.], [4.]])                               # two rows
print("tokens:\n", affine_numeric_tokens(x, W, b).squeeze(1).numpy())  # -> [[7,5],[9,5]]'''

T1_SOL = (T1_CODE
          .replace("""    # broadcast the scalar over the d embedding dims: [B, n_num, 1] * [n_num, d] -> [B, n_num, d]
    return ____""",
                   """    # broadcast the scalar over the d embedding dims: [B, n_num, 1] * [n_num, d] -> [B, n_num, d]
    return x_num.unsqueeze(-1) * weight + bias""")
          .replace("# TODO — implement the numeric tokenizer. Fill every ____.",
                   "# SOLUTION — the numeric Feature Tokenizer (Gorishniy 2021, §3.3)"))

T1_CHECK = r'''# CHECK — the numeric tokenizer is affine and per-feature (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

torch.manual_seed(0)
B, n_num, d = 16, 5, 8
Xn = torch.randn(B, n_num)
Wt = torch.randn(n_num, d); bs = torch.randn(n_num, d)
T = affine_numeric_tokens(Xn, Wt, bs)
chk("shape is [B, n_num, d]", tuple(T.shape) == (B, n_num, d), f"{tuple(T.shape)}")

# affine: bumping feature j by Δ moves token j by exactly Δ·W_j, and moves NO other token
j, delta = 2, 2.5
Xn2 = Xn.clone(); Xn2[:, j] += delta
dT = affine_numeric_tokens(Xn2, Wt, bs) - T
chk("token j moves by exactly Δx·W_j (affine in the scalar)",
    torch.allclose(dT[:, j], (delta * Wt[j]).expand(B, d), atol=1e-5))
others = dT.clone(); others[:, j] = 0.0
chk("changing feature j leaves every OTHER token unchanged", others.abs().max().item() < 1e-6)

# VALIDATION against the from-scratch reference (NOTES #22)
chk("VALIDATED against relkit.ft_transformer.affine_numeric_tokens",
    torch.allclose(affine_numeric_tokens(Xn, Wt, bs), relkit_affine(Xn, Wt, bs), atol=1e-6))
print("\nTask 1", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------------------------------- Task 2: [CLS] token
T2_MD = r'''## Task 2 — `prepend_cls`: the learned [CLS] readout token

**Goal.** Prepend a single learned `[CLS]` token at position 0 of the token sequence. It represents **no
feature**; its job is to *collect* information from all the feature tokens through attention, and after the
Transformer its final vector is the row summary the head reads out (the BERT trick).

**Shapes.** `tokens` is `[B, k, d]` (the *k* feature tokens); `cls` is a single learned `[1, 1, d]` vector.
You expand `cls` across the batch and concatenate it in front, giving `[B, k+1, d]` with `[CLS]` at index 0.

**Why a dedicated token.** You *could* average the feature tokens, but a learned [CLS] lets the model choose,
through attention, how much to read from each feature — a strictly more flexible pool, and the one the paper
uses.'''

T2_CODE = r'''# TODO — prepend the [CLS] token. Fill every ____.
def prepend_cls(tokens, cls):
    """tokens: [B, k, d] · cls: [1, 1, d] (a single learned vector) · returns [B, k+1, d]."""
    B = tokens.shape[0]
    # expand the one learned cls vector across the batch, then concatenate it at position 0
    cls_batch = cls.expand(B, 1, cls.shape[-1])
    return ____

# quick look
toks = torch.zeros(4, 3, 8)                       # 4 rows, 3 feature tokens, d=8
cls = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
seq = prepend_cls(toks, cls)
print("sequence shape:", tuple(seq.shape), " (should be [4, 4, 8])")
print("position 0 == the cls vector for every row:", bool(torch.allclose(seq[:, 0], cls.reshape(1, 8).expand(4, 8))))'''

T2_SOL = (T2_CODE
          .replace("""    # expand the one learned cls vector across the batch, then concatenate it at position 0
    cls_batch = cls.expand(B, 1, cls.shape[-1])
    return ____""",
                   """    # expand the one learned cls vector across the batch, then concatenate it at position 0
    cls_batch = cls.expand(B, 1, cls.shape[-1])
    return torch.cat([cls_batch, tokens], dim=1)""")
          .replace("# TODO — prepend the [CLS] token. Fill every ____.",
                   "# SOLUTION — the [CLS] readout token (Gorishniy 2021, §3.3)"))

T2_CHECK = r'''# CHECK — [CLS] is prepended correctly (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

torch.manual_seed(0)
toks = torch.randn(6, 5, 8); cls = torch.randn(1, 1, 8)
seq = prepend_cls(toks, cls)
chk("sequence grew by one token: [B, k+1, d]", tuple(seq.shape) == (6, 6, 8), f"{tuple(seq.shape)}")
chk("position 0 is the SAME learned [CLS] for every row", torch.allclose(seq[:, 0], cls.reshape(1, 8).expand(6, 8)))
chk("positions 1..k are the original feature tokens, unchanged", torch.allclose(seq[:, 1:], toks))
chk("VALIDATED against relkit.ft_transformer.prepend_cls", torch.allclose(seq, relkit_prepend(toks, cls)))
print("\nTask 2", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------- Task 3: the bake-off
T3_MD = r'''## Task 3 — the bake-off: FT-Transformer vs TabTransformer vs MLP vs CatBoost

**Goal.** Race four models under one shared frame, then summarise with **mean ranks** and a Friedman test:

- **FT-Transformer** — tokenises *every* feature (your Tasks 1–2 are its load-bearing pieces).
- **TabTransformer** — the L045 model; numerics bypass the attention.
- **MLP** — the strong deep baseline ([L042](../lessons/0042-mlp-resnet-baselines.html)).
- **CatBoost** — the honest tree bar ([L042](../lessons/0042-mlp-resnet-baselines.html)'s baseline-first rule).

**What to look for (NOTES #23/#24).** Two readings, kept apart: (1) does FT-T beat TabTransformer, and is
the gap biggest where numerics dominate (`churn` has num-frac 0.80)? (2) is FT-T the best *neural* model,
while still losing to the tree? Two tables is a demonstration; the strong claim stays cited to the paper.'''

T3_CODE = r'''# TODO — fill every ____.  (FTTransformer / train_ft_transformer / ft_transformer_auc are inlined below.)
def run_ft(fr, tr, va, te, seed):
    torch.manual_seed(seed)
    # build the FT-Transformer: it takes (n_num, cards, **cfg) — numerics AND categoricals become tokens
    m = FTTransformer(fr["Xnum"].shape[1], fr["cards"], **FT_CFG)
    m, _ = train_ft_transformer(m, fr["Xnum"][tr], fr["Xcat"][tr], fr["y"][tr],
                                fr["Xnum"][va], fr["Xcat"][va], fr["y"][va],
                                lr=LR, max_epochs=EPOCHS, patience=PATIENCE, batch_size=BS, seed=seed)
    return ft_transformer_auc(m, fr["Xnum"][te], fr["Xcat"][te], fr["y"][te])

def run_tabt(fr, tr, va, te, seed):
    torch.manual_seed(seed)
    m = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG)
    m, _ = train_tabtransformer(m, fr["Xcat"][tr], fr["Xnum"][tr], fr["y"][tr],
                                fr["Xcat"][va], fr["Xnum"][va], fr["y"][va],
                                lr=LR, max_epochs=EPOCHS, patience=PATIENCE, batch_size=BS, seed=seed)
    return tabtransformer_auc(m, fr["Xcat"][te], fr["Xnum"][te], fr["y"][te])

def run_mlp(fr, tr, va, te, seed):
    torch.manual_seed(seed)
    m = TabMLP(fr["Xdense"].shape[1], d_block=128, n_blocks=2, dropout=0.1)
    m, _ = train_net(m, fr["Xdense"][tr], fr["y"][tr], fr["Xdense"][va], fr["y"][va],
                     lr=2e-3, max_epochs=200, patience=16, seed=seed)
    return net_auc(m, fr["Xdense"][te], fr["y"][te])

def run_catboost(fr, tr, va, te, seed):
    from catboost import CatBoostClassifier
    Xdf = fr["Xdf"]; cat_idx = [Xdf.columns.get_loc(c) for c in fr["cat_names"]]
    Xstr = Xdf.copy()
    for c in fr["cat_names"]:
        Xstr[c] = Xstr[c].map(str).astype(object)
    clf = CatBoostClassifier(depth=6, iterations=300, learning_rate=0.05, l2_leaf_reg=3.0,
                             random_seed=seed, thread_count=2, verbose=0, allow_writing_files=False)
    clf.fit(Xstr.iloc[tr], fr["y"][tr].astype(int), cat_features=cat_idx,
            eval_set=(Xstr.iloc[va], fr["y"][va].astype(int)))
    return roc_auc_score(fr["y"][te], clf.predict_proba(Xstr.iloc[te])[:, 1])

MODELS = ["ft_transformer", "tabtransformer", "mlp", "catboost"]
RUN = {"ft_transformer": run_ft, "tabtransformer": run_tabt, "mlp": run_mlp, "catboost": run_catboost}
table = {}
for name in DATASETS:
    fr = load_frame(name)
    print(f"\n=== {name}: {len(fr['y'])} rows | num_frac {fr['num_frac']:.2f} ===", flush=True)
    rows = {m: [] for m in MODELS}
    for s in SEEDS:
        tr, va, te = split_idx(len(fr["y"]), fr["y"], s)
        for m in MODELS:
            # call each model's runner on the SAME split
            rows[m].append(RUN[m](fr, tr, va, te, ____))
        print("  seed {}: ".format(s) + " | ".join(f"{m} {rows[m][-1]:.3f}" for m in MODELS), flush=True)
    table[name] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in rows.items()}
    table[name]["num_frac"] = fr["num_frac"]

# --- cross-dataset summary: mean ranks (1 = best per dataset) + Friedman
score = np.array([[table[d][m][0] for m in MODELS] for d in DATASETS])
ranks = np.array([rankdata(-row, method="average") for row in score])          # HIGHEST score -> rank 1
mean_rank = {m: float(ranks[:, i].mean()) for i, m in enumerate(MODELS)}
fried = friedmanchisquare(*[score[:, i] for i in range(len(MODELS))])
ft_beats_tabt = sum(table[d]["ft_transformer"][0] > table[d]["tabtransformer"][0] for d in DATASETS)
best_neural = min(["ft_transformer", "tabtransformer", "mlp"], key=lambda m: mean_rank[m])

print("\nmean ranks:", {m: round(r, 2) for m, r in mean_rank.items()})
print(f"Friedman chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (k={len(MODELS)}, N={len(DATASETS)})")
print(f"FT-T beats TabTransformer on {ft_beats_tabt}/{len(DATASETS)}; best neural model = {best_neural}")'''

T3_SOL = (T3_CODE
          .replace("rows[m].append(RUN[m](fr, tr, va, te, ____))",
                   "rows[m].append(RUN[m](fr, tr, va, te, s))")
          .replace("# TODO — fill every ____.  (FTTransformer / train_ft_transformer / ft_transformer_auc are inlined below.)",
                   "# SOLUTION  (FTTransformer / train_ft_transformer / ft_transformer_auc are inlined below.)"))

T3_CHECK = r'''# CHECK — read the verdict the disciplined way (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("ranks are a valid per-dataset ranking of the 4 models",
    ranks.shape == (len(DATASETS), 4) and bool(np.allclose(ranks.sum(1), 10.0)))
chk("best score on each dataset got rank 1",
    all(ranks[i][np.argmax(score[i])] == 1 for i in range(len(DATASETS))))

print(f"""
VERDICT — FT-Transformer mean rank {mean_rank['ft_transformer']:.2f}
          TabTransformer  mean rank {mean_rank['tabtransformer']:.2f}
          MLP             mean rank {mean_rank['mlp']:.2f}
          CatBoost        mean rank {mean_rank['catboost']:.2f}
  FT-T vs TabTransformer : {ft_beats_tabt}/{len(DATASETS)} tables here — the numeric tokenizer should help MOST
                           on the numeric-heavy table (churn, num-frac 0.80).
  best NEURAL model      : {best_neural}
  vs trees               : {'CatBoost still ranks best' if mean_rank['catboost'] <= min(mean_rank[m] for m in ['ft_transformer','tabtransformer','mlp']) else 'a neural model led (unusual at this scale)'}

Read this the disciplined way (NOTES #23). At two tables and two seeds you have little power, so treat the
DIRECTION, not the exact ranks, as the signal. On the lesson's FULLER run (labs/_verify_l046.py: 4 datasets
x 3 seeds) it resolves to mean ranks FT-T 2.50 / MLP 2.75 / TabTransformer 3.75 / CatBoost 1.00 — FT-T beats
TabTransformer on 3/4 (all but the most-categorical credit_g), is the best neural model, and CatBoost wins
all 4 (Friedman p=0.026). FT-Transformer is the strongest single DEEP model AND still loses to the tree —
Gorishniy 2021's exact finding. Do not average the two readings into "FT-T won" or "FT-T lost".""")
print("Task 3", "OK" if ok else "-- fix the FAILs above")'''


# --------------------------------------------------- Task 4: numerics-attend probe
T4_MD = r'''## Task 4 — the probe: do numerics actually attend?

**Goal.** Show the one structural difference between FT-Transformer and TabTransformer *directly*: bump a
single **numeric** feature and measure how much each model's row representation moves.

- **FT-Transformer** reads out the final `[CLS]` vector (`model.cls_readout(x_num, x_cat)`). Because numerics
  are tokens, a numeric change flows through attention into `[CLS]` — the readout should **move**.
- **TabTransformer** builds its representation from the categorical contextual tokens only
  (`model.contextual(x_cat)`); numerics never enter it — so the same numeric change moves it **exactly 0**.

This is the lesson's headline made measurable: FT-Transformer fixes the numeric bypass.'''

T4_CODE = r'''# TODO — fill every ____.
fr = load_frame("churn")                                    # numeric-heavy (num-frac 0.80)
torch.manual_seed(0)
ft = FTTransformer(fr["Xnum"].shape[1], fr["cards"], **FT_CFG).eval()
tt = TabTransformer(fr["cards"], fr["Xnum"].shape[1], **TT_CFG).eval()

xn, xc = fr["Xnum"][:64], fr["Xcat"][:64]
xn_bumped = xn.clone(); xn_bumped[:, 0] += 3.0              # bump the FIRST numeric feature

with torch.no_grad():
    # FT-Transformer: read out the final [CLS] vector before vs after the numeric bump
    z1 = ft.cls_readout(xn, xc)
    z2 = ft.cls_readout(xn_bumped, xc)
    ft_move = (z1 - z2).norm(dim=1).mean().item()
    # TabTransformer: its representation is the categorical contextual tokens — a numeric bump cannot reach it
    c1 = tt.contextual(xc).flatten(1)
    c2 = ____                                               # recompute the SAME thing (numerics don't enter it)
    tt_move = (c1 - c2).norm(dim=1).mean().item()

print(f"FT-Transformer [CLS] readout move when a numeric feature changes: {ft_move:.3f}")
print(f"TabTransformer contextual move for the same numeric change      : {tt_move:.1e}")'''

T4_SOL = (T4_CODE
          .replace("""    c2 = ____                                               # recompute the SAME thing (numerics don't enter it)""",
                   """    c2 = tt.contextual(xc).flatten(1)                       # recompute the SAME thing (numerics don't enter it)""")
          .replace("# TODO — fill every ____.", "# SOLUTION"))

T4_CHECK = r'''# CHECK — the numeric-bypass fix, made measurable (do not edit)
ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not cond: ok = False

chk("FT-Transformer's [CLS] readout MOVES when a numeric feature changes (numbers attend)",
    ft_move > 1e-4, f"L2 move {ft_move:.3f}")
chk("TabTransformer's representation does NOT move for a numeric change (numerics bypass attention)",
    tt_move < 1e-9, f"L2 move {tt_move:.1e}")
print(f"""
This is FT-Transformer's whole contribution in one number. Because numerics are TOKENS, changing a number
reshapes the row summary ([CLS] move {ft_move:.3f}); in TabTransformer the same change moves the
representation {tt_move:.0e} — the numeric bypass L045 named. The lesson's verify records FT-T 0.438 vs
TabTransformer 0.0 on adult. That structural fix is why FT-T beats TabTransformer most where numerics carry
the signal — and why it, not TabTransformer, is the strongest single neural baseline.""")
print("Task 4", "OK" if ok else "-- fix the FAILs above")'''


# ------------------------------------------------------------------------------ exit ticket
EXIT_MD = r'''## EXIT TICKET

Paste this output to your teacher, or just say *"lab done."*'''

EXIT_CODE = r'''# EXIT TICKET
print("=== LAB 046 — FT-Transformer (feature tokenizer + [CLS] readout) ===")
print(f"numeric tokenizer : affine T_j = b_j + x_j·W_j, validated vs reference (per-feature, order-preserving)")
print(f"[CLS] readout     : prepended at position 0, validated vs reference")
print(f"bake-off ranks    : " + ", ".join(f"{m} {mean_rank[m]:.2f}" for m in MODELS))
print(f"Friedman          : chi2={fried.statistic:.3f}, p={fried.pvalue:.3f} (N={len(DATASETS)} tables)")
print(f"FT-T vs TabT      : FT-Transformer beats TabTransformer on {ft_beats_tabt}/{len(DATASETS)} tables")
print(f"numerics attend   : FT-T [CLS] move {ft_move:.3f} vs TabTransformer {tt_move:.0e} (the bypass, fixed)")
print()
print("in one sentence, what does tokenising the numeric features buy over TabTransformer, and does it beat a tree?:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"FT-Transformer tokenises EVERY feature — including numerics, as T_j = b_j + x_j·W_j — and reads the row '
    'out through a learned [CLS] token, so unlike TabTransformer a numeric feature enters the attention (its '
    '[CLS] readout moves when a number changes; TabTransformer\'s does not). That fixes the numeric bypass and '
    'makes FT-T beat TabTransformer most where numerics carry the signal (3/4 tables in the lesson run), and '
    'makes it the strongest single NEURAL baseline (best neural mean rank) — but CatBoost still wins every '
    'table, so on flat data FT-Transformer is the best deep model and still a notch below a tree, exactly '
    'Gorishniy 2021\'s finding."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 046 — FT-Transformer: feature tokenizer + [CLS] readout

**Lesson:** [`lessons/0046-ft-transformer.html`](../lessons/0046-ft-transformer.html) · **Phase / Year:** Year 2 · Q1

**Paper:** Gorishniy, Rubachev, Khrulkov & Babenko 2021, *Revisiting Deep Learning Models for Tabular Data* ([arXiv:2106.11959](https://arxiv.org/abs/2106.11959)) — §3.3 (the Feature Tokenizer, the [CLS] token, the Transformer), §5 (the fair-comparison result: FT-Transformer is the top deep model and ~ties tuned GBDTs). Self-attention: Vaswani et al. 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)). Predecessor it fixes: [L045 TabTransformer](../lessons/0045-tabtransformer.html) (numeric-bypass).

**Dataset tier:** **A** — small real OpenML tables via `relkit` (CPU-cheap; deliberately NOT representative — see the #23 notes in Task 3).

**Skill you are practising:** build FT-Transformer's two load-bearing pieces — (1) the **numeric tokenizer** `T_j = b_j + x_j·W_j` and (2) the **[CLS]** readout — then race FT-Transformer vs TabTransformer vs MLP vs CatBoost, and prove the numeric-bypass fix with a direct probe. The model you train calls *your* Task-1/Task-2 functions (they are inlined-and-kept, not hidden behind a package).

**Exit criteria:** EXIT TICKET prints your tokenizer + [CLS] validation, the bake-off ranks, the FT-T-vs-TabTransformer count, the numerics-attend probe, and one sentence — *what does tokenising numerics buy, and does it beat a tree?*

---

### How this notebook works
- **PROVIDED** cells — boilerplate (data, frames, budgets, baselines) **and** the paper's FT-Transformer copied into the notebook (not hidden behind `import relkit.ft_transformer`); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. After EXIT, a **NEXT STEP** cell trains closer to the paper (Colab GPU or Modal). When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` → kernel **Relational Labs (.venv)**. Needs **torch** + scikit-learn + **catboost** (CPU is fine). Real datasets fetch from OpenML on first run then cache. Budget: **~8–12 minutes on CPU** — set `OMP_NUM_THREADS=1` if a run feels slow (thread oversubscription has been the real cause of every slow lab). This lab uses a deliberately small budget/seed/dataset count to stay interactive; the lesson's headline numbers come from the fuller `labs/_verify_l046.py` run plus the paper's 11-dataset tuned benchmark.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — FT-Transformer = Feature Tokenizer + Transformer

**The one idea.** A Transformer eats a sequence of **tokens** (vectors of a common width *d*). [TabTransformer](../lessons/0045-tabtransformer.html)
only knew how to make tokens from *categories* (an embedding lookup), so its **numeric** features bypassed
the attention — LayerNorm'd and concatenated after the Transformer, never contextualised. FT-Transformer's
**Feature Tokenizer** adds the missing half: a way to turn a *number* into a token.

**The tokenizer (Gorishniy 2021, §3.3).**
- **Numeric feature *j*:** `T_j = b_j + x_j · W_j` — an **affine** embedding of the scalar onto a learned
  direction `W_j` (with `W_j, b_j ∈ ℝ^d`). Order is preserved: 39 and 40 land near each other. *(Task 1.)*
- **Categorical feature *j*:** `T_j = b_j + e_j[x_j]` — the [L031](../lessons/0031-embeddings-for-categoricals.html)
  entity embedding plus a per-feature bias.

**The [CLS] readout.** Prepend one learned `[CLS]` token (representing no feature) at position 0. Through the
Transformer it *collects* information from every feature token, and its final vector is the row summary the
head reads. *(Task 2.)*

**Why it matters.** Because numerics are now tokens, a numeric feature both attends and is attended to — so a
change in `age` can reshape the whole row's [CLS] summary (Task 4 measures this: FT-T moves, TabTransformer
moves 0). That fix makes FT-Transformer the **strongest single neural baseline** — beating TabTransformer
most where numerics carry the signal — while still, honestly, losing the flat-table metric to CatBoost.

Full write-up + the tokenizer and architecture widgets: [Lesson 046](../lessons/0046-ft-transformer.html).'''),
        md("## Setup — PROVIDED (tables + shared frames + baselines + budgets)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(architecture_md(
            "the Feature Tokenizer, the [CLS] token, the PreNorm Transformer block, and the train / eval loops",
            "labs/relkit/ft_transformer.py",
            "`affine_numeric_tokens` and `prepend_cls`",
        )),
        code(inline_source(
            os.path.join(HERE, "relkit/ft_transformer.py"),
            skip_defs={"affine_numeric_tokens", "prepend_cls"},
        )),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(T4_MD), code(T4_SOL if solution else T4_CODE), code(T4_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(notebook_scaleup_md(
            lesson=46,
            paper="Gorishniy, Rubachev, Khrulkov & Babenko 2021, FT-Transformer",
            arxiv="2106.11959",
            lab_rows=[
                ("mean ranks", "FT-T 2.50 vs MLP 2.75 vs TabTransformer 3.75 vs CatBoost 1.00, Friedman p=0.026"),
                ("numeric-bypass fix", "FT-T beats TabTransformer 3/4 (all but the most-categorical table)"),
            ],
            paper_rows=[
                ("FT-T is the strongest single deep model", "11-dataset tuned mean. We use a random OpenML split + no Optuna → INCOMPARABLE on Table 2; read DIRECTION vs MLP / TabTransformer."),
                ("ties tuned GBDTs", "Paper: FT-T ~ties GBDT on average. On Adult a tree edges it — CatBoost winning here is compatible."),
                ("Adult accuracy ≈ 0.859 (tuned)", "Table 2 default/tuned FT-T; our untuned random-split number is INCOMPARABLE."),
            ],
            modal="modal/l046_paper_repro.py",
        )),
        code(notebook_scaleup_code(
            lesson=46,
            harness_path=os.path.join(HERE, "_paper_repro_l046.py"),
            modal="modal/l046_paper_repro.py",
            skip_imports={"relkit.ft_transformer"},
        )),
        md(r'''## Stretch (optional, ungraded) — after the scale-up

1. **The numeric-fraction sweep.** Add `phoneme` (all numeric) and `credit_g` (mostly categorical) to
   `DATASETS` and re-run Task 3. FT-Transformer's edge over TabTransformer should *grow* with the numeric
   fraction — and vanish (or invert) on the most categorical table. This is the mechanism, plotted.
2. **How wide a token?** Sweep `d ∈ {16, 32, 64, 128}` in `FT_CFG` on `churn`. Where do returns flatten? The
   paper's default is ~192, far above what a small table needs.
3. **Does [CLS] beat mean-pooling?** Replace the [CLS] readout with the mean of the feature tokens (edit a
   copy of `cls_readout`) and compare. How much does the learned readout actually buy on these tables?
4. **Depth.** Sweep `n_layers ∈ {1, 2, 3}`. On small tables, does one attention layer already capture most of
   the cross-feature signal?'''),
        code(r'''# STRETCH — ungraded.
# for extra in (["phoneme"], ["credit_g"]):
#     for name in extra:
#         fr = load_frame(name)
#         accs_ft, accs_tt = [], []
#         for s in SEEDS:
#             tr, va, te = split_idx(len(fr["y"]), fr["y"], s)
#             accs_ft.append(run_ft(fr, tr, va, te, s)); accs_tt.append(run_tabt(fr, tr, va, te, s))
#         print(f"{name} (num_frac {fr['num_frac']:.2f}): FT-T {np.mean(accs_ft):.3f}  TabT {np.mean(accs_tt):.3f}")'''),
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
    with open(os.path.join(HERE, "0046-ft-transformer.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0046-ft-transformer.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0046-ft-transformer.ipynb + solution")


if __name__ == "__main__":
    main()
