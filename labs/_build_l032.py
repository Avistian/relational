"""Build Lab 032 (TabTransformer preview — trace the shapes through the architecture) — student + solution.

Tier C (a tiny toy row, NO training — a "read the architecture figure" exercise made concrete). numpy only.

Three tasks:
  * Task 1 (crucial fragment) — scaled dot-product SELF-ATTENTION: softmax(Q·Kᵀ/√d)·V. This is the one
    new piece of machinery TabTransformer adds over an entity-embedding MLP.
  * Task 2 — the CONTEXTUAL property: embed a row's categoricals (stage 1), run attention, and show that a
    column's output vector CHANGES when a neighbouring column changes — that is what "contextual" means.
  * Task 3 — assemble the full data-flow: contextual (categorical) ⊕ normalised continuous → MLP head,
    checking the tensor shape at every stage against the figure.

Run: .venv/bin/python labs/_build_l032.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0032-tabtransformer-preview.ipynb labs/solutions/0032-tabtransformer-preview.ipynb
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


SETUP = r'''# PROVIDED — a tiny toy row + fixed random weights (no training, no downloads). Just run.
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

# One toy row of a credit-style table.
CAT   = ["purpose", "checking", "housing"]     # m = 3 categorical columns (tokens)
CARD  = {"purpose": 4, "checking": 4, "housing": 3}   # cardinalities (levels per column)
NUMC  = ["age", "duration"]                    # continuous columns (bypass the Transformer)
D     = 4                                       # embedding / token width d
M     = len(CAT)

# One learned embedding TABLE per categorical (context-free — this is the L031 entity embedding).
TABLES = {c: rng.normal(size=(CARD[c], D)) for c in CAT}

# Self-attention's three learned projections (query, key, value), shared across tokens here.
Wq = rng.normal(size=(D, D)) * 0.5
Wk = rng.normal(size=(D, D)) * 0.5
Wv = rng.normal(size=(D, D)) * 0.5

# Our toy row: an index into each categorical + the two continuous values.
row_cat = {"purpose": 0, "checking": 3, "housing": 1}   # e.g. new car / <0 / rent
row_num = np.array([0.4, -1.1])                          # standardised age, duration

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

print(f"toy row: {M} categorical tokens (width d={D}) + {len(NUMC)} continuous features")
print("ready — helper: softmax(z, axis)")'''


# ---- Task 1: scaled dot-product self-attention ----
T1_MD = r'''## Task 1 — scaled dot-product self-attention (crucial fragment) — TODO

**Goal:** implement the one operation TabTransformer adds over L031 — **self-attention**. Given a token
matrix `X` of shape `(m, d)` (one row = one token), it returns a new `(m, d)` matrix where each token has
been rewritten as a weighted blend of all tokens.

**Why it matters:** this is the machinery that makes embeddings *contextual*. Each token emits a **query**,
**key**, and **value** (three linear projections). The weight token *i* puts on token *j* is how well *i*'s
query matches *j*'s key; those weights (via softmax) blend the **values** into each token's output.

**The formula (Vaswani et al. 2017, §3.2):**
`Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`

**Hint boundary:** you fill (a) the **scaled score matrix** `Q·Kᵀ / √d` and (b) the **weighted sum**
`weights · V`. The `softmax` helper (row-wise) is already provided.'''

T1_CODE = r'''# TODO — fill the two blanks of scaled dot-product self-attention
def self_attention(X, Wq, Wk, Wv):
    d = X.shape[1]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores  = ____                       # (a) Q @ K.T / sqrt(d)  -> shape (m, m)
    weights = softmax(scores, axis=1)    # each row sums to 1
    out     = ____                       # (b) weights @ V        -> shape (m, d)
    return out, weights

# sanity: run it on 3 random tokens
_X = rng.normal(size=(M, D))
_out, _w = self_attention(_X, Wq, Wk, Wv)
print("attention output shape:", _out.shape, "| weight rows sum to 1:", np.allclose(_w.sum(1), 1))'''

T1_SOL = (T1_CODE
    .replace("    scores  = ____                       # (a) Q @ K.T / sqrt(d)  -> shape (m, m)",
             "    scores  = Q @ K.T / np.sqrt(d)        # (a) Q @ K.T / sqrt(d)  -> shape (m, m)")
    .replace("    out     = ____                       # (b) weights @ V        -> shape (m, d)",
             "    out     = weights @ V                 # (b) weights @ V        -> shape (m, d)"))

T1_CHECK = r'''# CHECK — do not edit
_out, _w = self_attention(_X, Wq, Wk, Wv)
assert _out.shape == (M, D), "Attention must return one (d,)-vector per token: shape (m, d)."
assert _w.shape == (M, M), "Attention weights are token-by-token: shape (m, m)."
assert np.allclose(_w.sum(axis=1), 1.0), "Each token's attention weights must sum to 1 (softmax over keys)."
_ref = softmax((_X @ Wq) @ (_X @ Wk).T / np.sqrt(D), axis=1) @ (_X @ Wv)
assert np.allclose(_out, _ref), "Output must equal softmax(Q·Kᵀ/√d)·V."
print("Task 1 ok — scaled dot-product self-attention: softmax(Q·Kᵀ/√d)·V, weights sum to 1.")'''


# ---- Task 2: the contextual property ----
T2_MD = r'''## Task 2 — the contextual property — TODO

**Goal:** show what "contextual" *means*. First embed a row's categoricals into a token matrix (stage 1 —
the L031 entity-embedding lookup). Then run your attention from Task 1 and check that a column's output
vector **changes when a neighbouring column changes** — even though that column's own value did not.

**Why it matters:** a context-free embedding (L031) gives `checking = <0` the same vector everywhere. A
contextual embedding lets it depend on the rest of the row — the "bank" (river vs savings) idea. That row
dependence is the entire point of putting a Transformer in the middle.

**Hint boundary:** fill the **embedding lookup** — stack each categorical's embedding-table row for the
given level: `np.stack([TABLES[c][row[c]] for c in CAT])`.'''

T2_CODE = r'''# TODO — build the (m, d) token matrix by looking up each column's embedding
def embed_row(row):
    tokens = ____                        # np.stack of TABLES[c][row[c]] for c in CAT  -> shape (m, d)
    return tokens

base = row_cat.copy()
ctx_base, _ = self_attention(embed_row(base), Wq, Wk, Wv)

# change ONLY a neighbour (housing), keep 'checking' fixed, and re-contextualise
changed = row_cat.copy(); changed["housing"] = 0
ctx_changed, _ = self_attention(embed_row(changed), Wq, Wk, Wv)

chk = CAT.index("checking")
delta = np.linalg.norm(ctx_base[chk] - ctx_changed[chk])
print(f"'checking' context-free vector fixed, but its CONTEXTUAL vector moved by {delta:.3f} when 'housing' changed")'''

T2_SOL = T2_CODE.replace(
    "    tokens = ____                        # np.stack of TABLES[c][row[c]] for c in CAT  -> shape (m, d)",
    "    tokens = np.stack([TABLES[c][row[c]] for c in CAT])   # shape (m, d)")

T2_CHECK = r'''# CHECK — do not edit
assert embed_row(row_cat).shape == (M, D), "embed_row must return the (m, d) token matrix."
# the context-free embedding of 'checking' is identical in both rows...
assert np.allclose(TABLES["checking"][base["checking"]], TABLES["checking"][changed["checking"]]), \
    "'checking' was not changed, so its context-free embedding must be identical."
# ...but its CONTEXTUAL vector must differ once a neighbour changed:
assert delta > 1e-6, "The contextual vector of 'checking' should change when a neighbour column changes."
# and re-running the SAME row is deterministic (contextual != random):
_again, _ = self_attention(embed_row(base), Wq, Wk, Wv)
assert np.allclose(_again, ctx_base), "Same row must give the same contextual embeddings (deterministic)."
print(f"Task 2 ok — contextual: 'checking' kept its value but its vector moved {delta:.3f} with the row. "
      "That is what a Transformer buys over a context-free entity embedding.")'''


# ---- Task 3: the full data-flow ----
T3_MD = r'''## Task 3 — assemble the full TabTransformer data-flow — TODO

**Goal:** put the figure together end to end and check the shape at every stage:
`categorical → embed → Transformer(contextual) → flatten` ⊕ `continuous → LayerNorm(bypass)` → **MLP head**.

**Why it matters:** reading the architecture figure means knowing exactly what is concatenated before the
head. The contextual embeddings are **flattened** to `m·d` numbers and concatenated with the normalised
continuous features; an ordinary MLP maps that to the prediction. Only categoricals were contextualised;
the numerics bypassed the Transformer.

**Hint boundary:** fill the **merge** — concatenate the flattened contextual vector with the normalised
continuous features: `np.concatenate([ctx_flat, num_norm])`.'''

T3_CODE = r'''# PROVIDED head pieces
def layernorm(v):
    return (v - v.mean()) / (v.std() + 1e-9)

W1 = rng.normal(size=(M * D + len(NUMC), 8)) * 0.3   # MLP head: hidden layer
W2 = rng.normal(size=(8, 1)) * 0.3                    # MLP head: output
def mlp_head(x):
    return float((np.maximum(0, x @ W1) @ W2).ravel()[0])   # Linear -> ReLU -> Linear -> scalar logit

# TODO — assemble the forward pass
def tabtransformer_forward(row_cat, row_num):
    tokens   = embed_row(row_cat)                     # stage 1: (m, d)  entity embeddings
    ctx, _   = self_attention(tokens, Wq, Wk, Wv)     # stage 2: (m, d)  CONTEXTUAL embeddings
    ctx_flat = ctx.reshape(-1)                        #          (m*d,)  flatten
    num_norm = layernorm(row_num)                     # stage 3: (n_num,) continuous bypass
    merged   = ____                                   # stage 4: concat [ctx_flat, num_norm]
    return mlp_head(merged), {"tokens": tokens.shape, "contextual": ctx.shape,
                              "flattened": ctx_flat.shape, "merged": merged.shape}

yhat, shapes = tabtransformer_forward(row_cat, row_num)
for stage, s in shapes.items():
    print(f"  {stage:<11} {s}")
print("prediction (pre-sigmoid logit):", float(yhat))'''

T3_SOL = T3_CODE.replace(
    "    merged   = ____                                   # stage 4: concat [ctx_flat, num_norm]",
    "    merged   = np.concatenate([ctx_flat, num_norm])   # stage 4: concat [ctx_flat, num_norm]")

T3_CHECK = r'''# CHECK — do not edit
yhat, shapes = tabtransformer_forward(row_cat, row_num)
assert shapes["tokens"] == (M, D) and shapes["contextual"] == (M, D), "Embed & attention keep shape (m, d)."
assert shapes["flattened"] == (M * D,), "Contextual embeddings flatten to m*d numbers."
assert shapes["merged"] == (M * D + len(NUMC),), \
    "The head input is the flattened contextual vector CONCATENATED with the normalised continuous features."
assert np.isscalar(float(yhat)), "The MLP head outputs a single prediction logit."
print(f"Task 3 ok — full data-flow assembled: {M}x{D} embeddings -> contextual -> flatten "
      f"({M*D}) ⊕ {len(NUMC)} numeric -> MLP -> 1 logit. You can read the figure.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** print your deliverable — the shape trace and the contextual delta — plus a one-line takeaway.

**Takeaway prompt:** in one sentence, what does self-attention add over an entity-embedding MLP, what does
"contextual" mean, and why does TabTransformer only *matching* trees support the relational thesis?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
yhat, shapes = tabtransformer_forward(row_cat, row_num)
print("=== EXIT TICKET — Lesson 032 (TabTransformer preview) ===")
print(f"data-flow shapes : embed {shapes['tokens']} -> contextual {shapes['contextual']} -> "
      f"flatten {shapes['flattened']} -> merge {shapes['merged']} -> logit ()")
print(f"contextual proof : 'checking' vector moved {delta:.3f} when a neighbour column changed (context-free would move 0)")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"Self-attention adds a Transformer that rewrites each categorical column\'s embedding as a '
    'softmax(Q·Kᵀ/√d)-weighted blend of all the columns in the row, making it CONTEXTUAL (its vector now '
    'depends on the rest of the row, like river-bank vs savings-bank) instead of the fixed context-free '
    'entity embedding of L031; continuous features bypass the Transformer and are concatenated before an '
    'MLP head. On a flat table this only MATCHES gradient-boosted trees (its +1.0% is over other deep '
    'methods; the real wins are robustness and semi-supervised pre-training) -- which is the thesis, because '
    'attention over a row\'s columns is still single-table cleverness with no new structural information, '
    'while the same attend-to-related-information operation over the relational graph across tables (a GNN '
    'aggregating an entity\'s foreign-key neighbours) is where the untapped value is bet to live."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 032 — TabTransformer preview: trace the shapes through the architecture

**Lesson:** [`lessons/0032-tabtransformer-preview.html`](../lessons/0032-tabtransformer-preview.html) · **Phase / Year:** Year 1 · Q4 (bridge to neural tabular)

**Primary reading:** Huang et al. 2020, *TabTransformer: Tabular Data Modeling Using Contextual Embeddings* ([arXiv:2012.06678](https://arxiv.org/abs/2012.06678)) — read Fig. 1 + §2.

**Dataset tier:** **C** (a tiny toy row — this is a *read-the-architecture-figure* exercise, **no training**).

**Skill you are practising:** reading the TabTransformer figure by building its forward pass — scaled dot-product self-attention (`softmax(Q·Kᵀ/√d)·V`), the *contextual* property it produces, and the full data-flow (embed → attention → bypass continuous → concat → MLP head) with the tensor shape checked at every stage.

**Exit criteria:** EXIT TICKET prints the shape trace through the architecture, the proof that a column's contextual vector changes with its row, and your one-line takeaway.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (toy row, fixed weights, `softmax`); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
`numpy` only — no downloads, no GPU, no training. One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Runs in a few seconds.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — TabTransformer in one figure

TabTransformer (Huang et al. 2020) takes the **entity embeddings** of L031 and adds one new thing: a stack of **Transformer self-attention layers** that make them **contextual**.

**The data-flow (Fig. 1):**

```
categorical x_cat -> column embedding -> [ N x Transformer (self-attention + FFN) ] -> contextual embeddings -> flatten -\
continuous  x_cont -> LayerNorm ------------------------------------------------------------------------------> concat -> MLP -> y_hat
```

- **Stage 1 — embed categoricals.** Each categorical column becomes a learned dense vector (a *token*). This is L031's entity embedding, unchanged. *m* columns → *m* vectors of width *d*.
- **Stage 2 — the Transformer (the ONLY new machinery).** Self-attention lets every column attend to every other column in the row, producing **contextual** embeddings: a column's vector now depends on the rest of the row.
- **Stage 3 — continuous bypass.** Numeric features skip the Transformer; they are just layer-normalised.
- **Stage 4 — concat + MLP head.** Flatten the contextual vectors (m·d numbers), concatenate the numeric features, and an ordinary MLP predicts.

**Self-attention (the crucial fragment).** Each token emits a query `Q`, key `K`, value `V` (linear projections). The output is a weighted blend of the values:

`Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`

the weights (a softmax, so ≥ 0 and summing to 1) say how much each token contributes to each token's output.

**"Contextual" — the whole point.** A *context-free* embedding gives `checking = <0` the same vector everywhere; a *contextual* one gives it a different vector next to `housing = rent` than next to `housing = own` — exactly like "bank" in *river bank* vs *savings bank*.

**Honest verdict (skim the experiments).** On supervised tabular data TabTransformer **matches** tree ensembles (a tie); its +1.0% is over other *deep* methods, and its real wins are **robustness** to noisy/missing features and a **+2.1% semi-supervised** lift from unlabeled data.

Full write-up + visuals: [Lesson 032](../lessons/0032-tabtransformer-preview.html).'''),
        md("## Setup — PROVIDED (toy row + fixed weights + softmax helper)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
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
