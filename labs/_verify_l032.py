"""Standalone verification of the L032 lab claims (numpy only, no training, no downloads).

Reproduces:
  * scaled dot-product self-attention softmax(Q·Kᵀ/√d)·V returns (m, d) with weight rows summing to 1;
  * the CONTEXTUAL property: a column's output vector changes when a NEIGHBOUR column changes, though its
    own context-free embedding is unchanged; same row is deterministic;
  * the full data-flow shapes: embed (m,d) -> contextual (m,d) -> flatten (m·d,) -> merge (m·d+n_num,) -> logit.

Run: .venv/bin/python labs/_verify_l032.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

CAT = ["purpose", "checking", "housing"]
CARD = {"purpose": 4, "checking": 4, "housing": 3}
NUMC = ["age", "duration"]
D, M = 4, 3

TABLES = {c: rng.normal(size=(CARD[c], D)) for c in CAT}
Wq = rng.normal(size=(D, D)) * 0.5
Wk = rng.normal(size=(D, D)) * 0.5
Wv = rng.normal(size=(D, D)) * 0.5
row_cat = {"purpose": 0, "checking": 3, "housing": 1}
row_num = np.array([0.4, -1.1])


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    d = X.shape[1]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    weights = softmax(Q @ K.T / np.sqrt(d), axis=1)
    return weights @ V, weights


def embed_row(row):
    return np.stack([TABLES[c][row[c]] for c in CAT])


def main():
    out, w = self_attention(embed_row(row_cat), Wq, Wk, Wv)
    assert out.shape == (M, D) and w.shape == (M, M)
    assert np.allclose(w.sum(axis=1), 1.0)
    print(f"self-attention: output {out.shape}, weight rows sum to 1: {np.allclose(w.sum(1), 1)}")

    base = row_cat.copy()
    changed = row_cat.copy(); changed["housing"] = 0
    ctx_b, _ = self_attention(embed_row(base), Wq, Wk, Wv)
    ctx_c, _ = self_attention(embed_row(changed), Wq, Wk, Wv)
    chk = CAT.index("checking")
    delta = float(np.linalg.norm(ctx_b[chk] - ctx_c[chk]))
    assert np.allclose(TABLES["checking"][base["checking"]], TABLES["checking"][changed["checking"]])
    assert delta > 1e-6
    ctx_again, _ = self_attention(embed_row(base), Wq, Wk, Wv)
    assert np.allclose(ctx_again, ctx_b)
    print(f"contextual: 'checking' context-free vector fixed, contextual vector moved {delta:.3f} "
          f"when 'housing' changed; same row deterministic: {np.allclose(ctx_again, ctx_b)}")

    ctx, _ = self_attention(embed_row(row_cat), Wq, Wk, Wv)
    ctx_flat = ctx.reshape(-1)
    num_norm = (row_num - row_num.mean()) / (row_num.std() + 1e-9)
    merged = np.concatenate([ctx_flat, num_norm])
    assert ctx_flat.shape == (M * D,) and merged.shape == (M * D + len(NUMC),)
    print(f"data-flow: embed {ctx.shape} -> flatten {ctx_flat.shape} -> merge {merged.shape} "
          f"(= m·d + n_num = {M*D} + {len(NUMC)})")
    print("\nAll L032 lab claims verified.")


if __name__ == "__main__":
    main()
