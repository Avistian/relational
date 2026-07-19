"""Build Lab 032 (TabTransformer preview — implement the architecture, forward-only) — student + solution.

Paper-implementation lab (standard #18). TabTransformer is a TWO-lesson paper:
  * L032 (this lab, preview) — implement the ARCHITECTURE and run a FORWARD pass on real credit_g:
    scaled dot-product self-attention -> the Transformer BLOCK (attention + residual + FFN + LayerNorm,
    the paper's Fig. 1 layer) -> the full TabTransformer forward (embed -> contextual -> concat -> head).
    NO training here.
  * L045 (Year 2) — training, semi-supervised pre-training, and benchmarking vs trees.

Tier A (real credit_g via relkit), torch (forward-only, CPU, deterministic). Three tasks:
  * Task 1 (crucial fragment) — scaled dot-product self-attention, softmax(Q·Kᵀ/√d)·V (Vaswani §3.2 eq. 1).
  * Task 2 — the Transformer BLOCK (Fig. 1 layer): wire the two residual connections around attention & FFN.
  * Task 3 — the full TabTransformer forward: concat flattened contextual ⊕ normalised continuous -> MLP head;
    run on real credit_g, verify shapes, and prove the CONTEXTUAL property on a real row.

Run: .venv/bin/python labs/_build_l032.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output 0032-tabtransformer-preview.ipynb labs/solutions/0032-tabtransformer-preview.ipynb
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


SETUP = r'''# PROVIDED — imports, real credit_g (Tier A), integer-coded categoricals + standardised numerics. Just run.
import os
# Cap OpenMP threads BEFORE numpy/torch import (torch+sklearn can oversubscribe on some boxes — see L031).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
X, y = load_tier_a("credit_g")
NUM = X.select_dtypes(include="number").columns.tolist()
CAT = [c for c in X.columns if c not in NUM]

# Integer-code each categorical (the stage-1 input) and standardise the continuous features.
CARDS, codes = [], []
for c in CAT:
    levels = sorted(X[c].astype(str).unique())
    lut = {v: i for i, v in enumerate(levels)}
    codes.append(X[c].astype(str).map(lut).to_numpy())
    CARDS.append(len(levels))
Xcat = torch.tensor(np.stack(codes, axis=1), dtype=torch.long)      # (N, m) category indices
_Xn = X[NUM].to_numpy(float)
_mu, _sd = _Xn.mean(0), _Xn.std(0) + 1e-9
Xnum = torch.tensor((_Xn - _mu) / _sd, dtype=torch.float32)         # (N, n_num) standardised
yt = torch.tensor(np.asarray(y), dtype=torch.float32)

M, N_NUM, D = len(CAT), len(NUM), 16                               # m tokens, n_num continuous, token width d
print(f"credit_g (Tier A): {Xcat.shape[0]} rows | m={M} categorical tokens | n_num={N_NUM} continuous | token width d={D}")
print("ready — we will BUILD the architecture and forward-pass it (no training; that is Lesson 045).")'''


# ---- Task 1: scaled dot-product self-attention ----
T1_MD = r'''## Task 1 — scaled dot-product self-attention (crucial fragment) — TODO

**Paper element:** Vaswani et al. 2017, §3.2 eq. 1 — the operation every Transformer (and TabTransformer)
is built on.

**Goal:** implement `scaled_dot_product_attention(Q, K, V)`. Given queries/keys/values of shape
`(batch, m, d)` (one row = `m` tokens of width `d`), return a new `(batch, m, d)` where each token is a
weighted blend of all tokens, plus the `(batch, m, m)` weight matrix.

**Why it matters:** this is the one new mechanism over L031's entity embeddings — it lets a column's vector
depend on the other columns. The weight token *i* puts on token *j* is how well *i*'s **query** matches
*j*'s **key**; those weights blend the **values**.

**The formula:** `Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`

**Hint boundary:** fill (a) the **scaled scores** `Q·Kᵀ / √d` (use `K.transpose(-2, -1)`) and (b) the
**weighted sum** `weights · V`. `torch.softmax` over the last dim is provided.'''

T1_CODE = r'''# TODO — fill the two blanks of scaled dot-product self-attention (Vaswani §3.2 eq. 1)
def scaled_dot_product_attention(Q, K, V):
    d = Q.shape[-1]
    scores  = ____                               # (a) Q · Kᵀ / √d   -> (batch, m, m)
    weights = torch.softmax(scores, dim=-1)      # each query's weights over keys sum to 1
    out     = ____                               # (b) weights · V   -> (batch, m, d)
    return out, weights

# sanity: 2 rows of M tokens, width D
_q = torch.randn(2, M, D)
_o, _w = scaled_dot_product_attention(_q, _q, _q)
print("attention output:", tuple(_o.shape), "| weight rows sum to 1:", bool(torch.allclose(_w.sum(-1), torch.ones(2, M))))'''

T1_SOL = (T1_CODE
    .replace("    scores  = ____                               # (a) Q · Kᵀ / √d   -> (batch, m, m)",
             "    scores  = Q @ K.transpose(-2, -1) / math.sqrt(d)   # (a) Q · Kᵀ / √d   -> (batch, m, m)")
    .replace("    out     = ____                               # (b) weights · V   -> (batch, m, d)",
             "    out     = weights @ V                              # (b) weights · V   -> (batch, m, d)"))

T1_CHECK = r'''# CHECK — do not edit
_o, _w = scaled_dot_product_attention(_q, _q, _q)
assert _o.shape == (2, M, D), "Attention returns one (d,)-vector per token: (batch, m, d)."
assert _w.shape == (2, M, M), "Attention weights are token-by-token: (batch, m, m)."
assert torch.allclose(_w.sum(dim=-1), torch.ones(2, M)), "Each token's weights must sum to 1 (softmax over keys)."
_ref = torch.softmax(_q @ _q.transpose(-2, -1) / math.sqrt(D), dim=-1) @ _q
assert torch.allclose(_o, _ref, atol=1e-6), "Output must equal softmax(Q·Kᵀ/√d)·V."
print("Task 1 ok — scaled dot-product self-attention: softmax(Q·Kᵀ/√d)·V, weights sum to 1.")'''


# ---- Task 2: the Transformer block ----
T2_MD = r'''## Task 2 — the Transformer block (Fig. 1 layer) — TODO

**Paper element:** Huang et al. 2020, Fig. 1 — the "Transformer layer" the categorical embeddings are pushed
through *N* times. A layer is **two residual sub-layers**: self-attention, then a position-wise feed-forward
network (FFN), each wrapped in a **residual connection + LayerNorm**.

**Why it matters:** the residual (skip) connection is the same idea that made deep nets trainable in L028 —
`output = LayerNorm(x + sublayer(x))`, so a sub-layer only has to learn a *correction* to its input. You
already have `SelfAttention` (Q/K/V projections around your Task-1 attention) and the FFN + LayerNorms
provided; you wire the two skips.

**Hint boundary:** fill (a) the residual around attention — `LayerNorm₁(x + a)` — and (b) the residual around
the FFN — `LayerNorm₂(x + f)`. Do not add new layers; just combine `x`, the sub-layer output, and the norm.'''

T2_CODE = r'''# PROVIDED — Q/K/V projections wrap your Task-1 attention (this is one attention "head").
class SelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
    def forward(self, x):
        return scaled_dot_product_attention(self.Wq(x), self.Wk(x), self.Wv(x))   # (out, weights)

# TODO — wire the two residual sub-layers of the Transformer block (Fig. 1 layer)
class TransformerBlock(nn.Module):
    def __init__(self, d, ff_hidden=32):
        super().__init__()
        self.attn  = SelfAttention(d)
        self.ffn   = nn.Sequential(nn.Linear(d, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
    def forward(self, x):
        a, w = self.attn(x)
        x = ____                     # (a) residual around attention:  LayerNorm1(x + a)
        f = self.ffn(x)
        x = ____                     # (b) residual around the FFN:    LayerNorm2(x + f)
        return x, w

torch.manual_seed(SEED)
_blk = TransformerBlock(D)
_xo, _w = _blk(torch.randn(2, M, D))
print("block output:", tuple(_xo.shape), "| finite:", bool(torch.isfinite(_xo).all()))'''

T2_SOL = (T2_CODE
    .replace("        x = ____                     # (a) residual around attention:  LayerNorm1(x + a)",
             "        x = self.norm1(x + a)         # (a) residual around attention:  LayerNorm1(x + a)")
    .replace("        x = ____                     # (b) residual around the FFN:    LayerNorm2(x + f)",
             "        x = self.norm2(x + f)         # (b) residual around the FFN:    LayerNorm2(x + f)"))

T2_CHECK = r'''# CHECK — do not edit
torch.manual_seed(SEED)
_blk = TransformerBlock(D)
_x = torch.randn(2, M, D)
_xo, _w = _blk(_x)
assert _xo.shape == (2, M, D), "A Transformer block preserves the (batch, m, d) shape."
assert torch.isfinite(_xo).all(), "Block output must be finite."
# LayerNorm is applied LAST, so each output token is ~zero-mean across its d features:
assert _xo.mean(dim=-1).abs().max() < 1e-4, "Did you wrap the FFN sub-layer in norm2(x + f)?"
# The block must transform its input (skip present, not identity, not zeros):
assert not torch.allclose(_xo, _x) and _xo.abs().sum() > 0, "The block should transform x via its sub-layers."
print("Task 2 ok — Transformer block: LayerNorm(x + attention) then LayerNorm(x + FFN). Residuals wired.")'''


# ---- Task 3: the full TabTransformer forward ----
T3_MD = r'''## Task 3 — the full TabTransformer forward pass — TODO

**Paper element:** Huang et al. 2020, Fig. 1 (whole model). Assemble the data-flow and run it on **real
credit_g**: `categorical → embed → N Transformer blocks (contextual) → flatten` ⊕ `continuous → LayerNorm`
→ **MLP head**.

**Why it matters:** reading the figure means knowing exactly what is concatenated before the head — the
**flattened contextual embeddings** (`m·d` numbers) and the **normalised continuous features**. Only the
categoricals were contextualised; the numerics bypassed the Transformer. We forward-pass an untrained model
(training is L045) and then *prove the contextual property on a real row*.

**Hint boundary:** fill the **merge** — concatenate the flattened contextual vector with the normalised
continuous features along the feature dim: `torch.cat([ctx_flat, num], dim=1)`.'''

T3_CODE = r'''# TODO — assemble the full architecture (fill the stage-4 merge)
class TabTransformer(nn.Module):
    def __init__(self, cards, n_num, d=D, n_layers=2, ff_hidden=32, head_hidden=64):
        super().__init__()
        self.embs     = nn.ModuleList([nn.Embedding(c, d) for c in cards])              # stage 1: entity embeddings
        self.blocks   = nn.ModuleList([TransformerBlock(d, ff_hidden) for _ in range(n_layers)])  # stage 2
        self.num_norm = nn.LayerNorm(n_num)                                             # stage 3: continuous bypass
        self.head     = nn.Sequential(nn.Linear(len(cards) * d + n_num, head_hidden),   # stage 4: MLP head
                                      nn.ReLU(), nn.Linear(head_hidden, 1))
    def embed(self, x_cat):                                                             # (B, m, d)
        return torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
    def contextual(self, x_cat):                                                        # (B, m, d) after Transformer
        h = self.embed(x_cat)
        for blk in self.blocks:
            h, _ = blk(h)
        return h
    def forward(self, x_cat, x_num):
        ctx_flat = self.contextual(x_cat).flatten(1)     # (B, m*d) flattened contextual embeddings
        num      = self.num_norm(x_num)                  # (B, n_num) bypassed continuous
        merged   = ____                                  # stage 4: concat [ctx_flat, num] along dim=1
        return self.head(merged).squeeze(-1)

torch.manual_seed(SEED)
model = TabTransformer(CARDS, N_NUM).eval()
with torch.no_grad():
    logits = model(Xcat, Xnum)
n_params = sum(p.numel() for p in model.parameters())
print(f"forward pass on {Xcat.shape[0]} real rows -> logits {tuple(logits.shape)} | {n_params:,} params (untrained)")

# Prove the CONTEXTUAL property on a REAL row: change ONLY a neighbour column, watch 'checking_status' move.
chk, hidx = CAT.index("checking_status"), CAT.index("housing")
with torch.no_grad():
    r  = Xcat[0:1].clone()
    r2 = r.clone(); r2[0, hidx] = (r[0, hidx] + 1) % CARDS[hidx]        # flip housing only
    delta = (model.contextual(r)[0, chk] - model.contextual(r2)[0, chk]).norm().item()
print(f"'checking_status' context-free embedding fixed, but its CONTEXTUAL vector moved {delta:.3f} when 'housing' changed")'''

T3_SOL = T3_CODE.replace(
    "        merged   = ____                                  # stage 4: concat [ctx_flat, num] along dim=1",
    "        merged   = torch.cat([ctx_flat, num], dim=1)     # stage 4: concat [ctx_flat, num] along dim=1")

T3_CHECK = r'''# CHECK — do not edit
assert logits.shape == (Xcat.shape[0],), "One logit per row: shape (N,)."
assert torch.isfinite(logits).all(), "Forward pass must be finite."
# 'checking' was NOT changed, so its context-free embedding is identical in both rows...
assert torch.allclose(model.embs[chk].weight[r[0, chk]], model.embs[chk].weight[r2[0, chk]]), \
    "'checking' unchanged -> its entity embedding must be identical."
# ...but its CONTEXTUAL vector must move once a NEIGHBOUR changed:
assert delta > 1e-5, "The contextual vector of 'checking' should change when a neighbour column changes."
assert n_params > 1000, "This should be a real (if small) TabTransformer, not a toy."
print(f"Task 3 ok — full TabTransformer forward on real credit_g: {M}×{D} embeddings -> contextual -> "
      f"flatten ({M*D}) ⊕ {N_NUM} numeric -> MLP -> 1 logit. Contextual delta {delta:.3f}. You built the figure.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** print your deliverable — the architecture shape trace, the parameter count, and the proof that a
column's representation is contextual — plus a one-line takeaway.

**Takeaway prompt:** in one sentence, what does self-attention add over an entity-embedding MLP, what does
"contextual" mean, and (honestly) how does TabTransformer compare to trees — and why does that support the
relational thesis?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
with torch.no_grad():
    logits = model(Xcat, Xnum)
print("=== EXIT TICKET — Lesson 032 (TabTransformer preview) ===")
print(f"architecture : {M} categorical tokens x d={D} -> {len(model.blocks)} Transformer blocks -> "
      f"contextual -> flatten ({M*D}) ⊕ {N_NUM} numeric -> MLP head -> {tuple(logits.shape)} logit")
print(f"model        : {sum(p.numel() for p in model.parameters()):,} params, forward-only (training is Lesson 045)")
print(f"contextual   : 'checking_status' vector moved {delta:.3f} when a neighbour changed (a context-free embedding would move 0)")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"Self-attention adds a Transformer block (softmax(Q·Kᵀ/√d)·V, then a residual + FFN + LayerNorm) that '
    'rewrites each categorical column\'s embedding as a weighted blend of all the columns in the row, making '
    'it CONTEXTUAL -- its vector now depends on the rest of the row (river-bank vs savings-bank) instead of '
    'the fixed context-free entity embedding of L031; continuous features bypass the Transformer and are '
    'concatenated before an MLP head. Honestly, on a flat table TabTransformer only MATCHES gradient-boosted '
    'trees (its +1.0% is over other deep methods; the real wins are robustness and semi-supervised '
    'pre-training) -- which is the thesis, because attention over a row\'s columns is still single-table '
    'cleverness with no new structural information, while the same attend-to-related-information operation '
    'over the relational graph across tables (a GNN aggregating an entity\'s foreign-key neighbours) is where '
    'the untapped value is bet to live."')


STRETCH = r'''# STRETCH (optional, FORWARD-ONLY) — which columns does each column attend to?
# No training here: full training + semi-supervised pre-training + benchmarking vs trees is Lesson 045.
with torch.no_grad():
    _, W = model.blocks[0](model.embed(Xcat[0:1]))     # first block's attention weights: (1, m, m)
W = W[0].numpy()
row = 0
print(f"First-layer attention weights for real row {row} (weight[i, j] = how much column i attends to column j):")
print("            " + "  ".join(f"{c[:6]:>6}" for c in CAT))
for i, c in enumerate(CAT):
    print(f"{c[:11]:<11} " + "  ".join(f"{W[i, j]:6.2f}" for j in range(M)))
print("\nEach ROW sums to 1. Untrained, so the pattern is not yet meaningful — but the machinery is real, "
      "and after training (L045) these weights say which columns inform each column's contextual embedding.")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 032 — TabTransformer preview: implement the architecture (forward-only)

**Lesson:** [`lessons/0032-tabtransformer-preview.html`](../lessons/0032-tabtransformer-preview.html) · **Phase / Year:** Year 1 · Q4 (bridge to neural tabular)

**Primary reading:** Huang et al. 2020, *TabTransformer* ([arXiv:2012.06678](https://arxiv.org/abs/2012.06678)) — Fig. 1 + §2; attention from Vaswani et al. 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) §3.2.

**Dataset tier:** **A** (real `credit_g` via `relkit`).

**Skill you are practising:** *reading the architecture figure by building it.* You implement the paper's real machinery in **torch** — scaled dot-product self-attention, the Transformer block (attention + residual + FFN + LayerNorm), and the full TabTransformer forward pass — and run it on real data.

**Implementation scope (standard #18).** TabTransformer is a **two-lesson paper**, so its implementation is **split**:
- **This lab (L032, preview):** build the **architecture** and **forward-pass** it on real `credit_g`; inspect the contextual embeddings. **No training.**
- **Lesson 045 (Year 2):** **train** it, add **semi-supervised pre-training**, and **benchmark** it against trees.

**Exit criteria:** EXIT TICKET prints the architecture shape trace + parameter count and the proof that a column's contextual vector changes with its row, plus your one-line takeaway.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, module scaffolding); just run.
- **TODO** cells — blanks (`____`); you implement the load-bearing code.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **torch** (in `requirements-labs.txt`, installed in the lab venv). `credit_g` is fetched from OpenML on first run then cached. Forward-only, CPU, deterministic — runs in a few seconds.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — TabTransformer in one figure

TabTransformer (Huang et al. 2020) takes the **entity embeddings** of L031 and adds one new thing: a stack of **Transformer self-attention layers** that make them **contextual**. You will build each piece.

**The data-flow (Fig. 1):**

```
categorical x_cat -> column embedding -> [ N x Transformer block ] -> contextual embeddings -> flatten -\
continuous  x_cont -> LayerNorm --------------------------------------------------------------> concat -> MLP -> y_hat
```

**Self-attention (Task 1 — Vaswani §3.2 eq. 1).** Each token (a categorical column's embedding) emits a query `Q`, key `K`, value `V` (linear projections). The output is a weighted blend of the values:

`Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`

the weights (a softmax, ≥ 0 and summing to 1) say how much each token contributes to each token's output.

**The Transformer block (Task 2 — Fig. 1 layer).** One layer is two residual sub-layers:

`x ← LayerNorm(x + SelfAttention(x))`, then `x ← LayerNorm(x + FFN(x))`

The **residual** (`x + sublayer(x)`) is the L028 skip-connection idea: a sub-layer learns a *correction* to its input, so depth stays trainable. **LayerNorm** rescales each token to zero mean / unit variance across its features.

**The full model (Task 3 — Fig. 1 whole).** Embed the categoricals → push through *N* blocks → **flatten** the contextual embeddings (`m·d` numbers) → **concatenate** the LayerNorm'd continuous features → **MLP head** → prediction. Only categoricals are contextualised; numerics bypass.

**"Contextual" — the whole point.** A *context-free* embedding (L031) gives `checking = <0` the same vector everywhere; a *contextual* one gives it a different vector next to `housing = rent` than `housing = own` — like "bank" in *river bank* vs *savings bank*. Task 3 proves this on a real row.

**Honest verdict (why we don't train here).** On supervised tabular data TabTransformer **matches** tree ensembles (a tie); the +1.0% is over other *deep* methods, and its real wins are robustness to noise/missingness and a +2.1% semi-supervised lift. Training, pre-training, and the benchmark vs trees are **Lesson 045**; here we own the architecture.

Full write-up + visuals: [Lesson 032](../lessons/0032-tabtransformer-preview.html).'''),
        md("## Setup — PROVIDED (real credit_g, integer-coded categoricals + standardised numerics)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, forward-only) — inspect the attention weights

Peek at which columns each column attends to in the first Transformer block, for a real row. It is
**untrained**, so the pattern is not yet meaningful — the point is that the machinery is real and produces
a valid `(m, m)` attention matrix (each row summing to 1). After training (Lesson 045) these weights tell
you which columns inform each column's contextual embedding.'''),
        code(STRETCH),
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
    with open(os.path.join(HERE, "0032-tabtransformer-preview.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0032-tabtransformer-preview.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0032-tabtransformer-preview.ipynb + solution")


if __name__ == "__main__":
    main()
