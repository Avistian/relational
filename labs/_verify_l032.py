"""Standalone verification of the L032 lab claims (torch, real credit_g, forward-only — no training).

Reproduces the architecture the lab builds and checks:
  * scaled dot-product self-attention softmax(Q·Kᵀ/√d)·V returns (B,m,d) with weight rows summing to 1;
  * the Transformer block preserves shape and applies LayerNorm last (tokens ~zero-mean across features);
  * the full TabTransformer forward on real credit_g yields one logit per row and is finite;
  * the CONTEXTUAL property: a column's contextual vector moves when a NEIGHBOUR column changes, though its
    own context-free entity embedding is unchanged.

Run: .venv/bin/python labs/_verify_l032.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

sys.path.insert(0, str(Path("labs").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
from relkit import load_tier_a

X, y = load_tier_a("credit_g")
NUM = X.select_dtypes(include="number").columns.tolist()
CAT = [c for c in X.columns if c not in NUM]
CARDS, codes = [], []
for c in CAT:
    levels = sorted(X[c].astype(str).unique())
    lut = {v: i for i, v in enumerate(levels)}
    codes.append(X[c].astype(str).map(lut).to_numpy())
    CARDS.append(len(levels))
Xcat = torch.tensor(np.stack(codes, axis=1), dtype=torch.long)
_Xn = X[NUM].to_numpy(float)
Xnum = torch.tensor((_Xn - _Xn.mean(0)) / (_Xn.std(0) + 1e-9), dtype=torch.float32)
M, N_NUM, D = len(CAT), len(NUM), 16


def attention(Q, K, V):
    d = Q.shape[-1]
    w = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(d), dim=-1)
    return w @ V, w


class SelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return attention(self.Wq(x), self.Wk(x), self.Wv(x))


class TransformerBlock(nn.Module):
    def __init__(self, d, ff=32):
        super().__init__()
        self.attn = SelfAttention(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        a, w = self.attn(x)
        x = self.n1(x + a)
        x = self.n2(x + self.ffn(x))
        return x, w


class TabTransformer(nn.Module):
    def __init__(self, cards, n_num, d=D, n_layers=2):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c in cards])
        self.blocks = nn.ModuleList([TransformerBlock(d) for _ in range(n_layers)])
        self.num_norm = nn.LayerNorm(n_num)
        self.head = nn.Sequential(nn.Linear(len(cards) * d + n_num, 64), nn.ReLU(), nn.Linear(64, 1))

    def contextual(self, xc):
        h = torch.stack([e(xc[:, i]) for i, e in enumerate(self.embs)], dim=1)
        for b in self.blocks:
            h, _ = b(h)
        return h

    def forward(self, xc, xn):
        h = torch.cat([self.contextual(xc).flatten(1), self.num_norm(xn)], dim=1)
        return self.head(h).squeeze(-1)


def main():
    q = torch.randn(2, M, D)
    o, w = attention(q, q, q)
    assert o.shape == (2, M, D) and w.shape == (2, M, M)
    assert torch.allclose(w.sum(-1), torch.ones(2, M))
    print(f"self-attention: output {tuple(o.shape)}, weight rows sum to 1: {bool(torch.allclose(w.sum(-1), torch.ones(2, M)))}")

    torch.manual_seed(SEED)
    blk = TransformerBlock(D)
    xo, _ = blk(torch.randn(2, M, D))
    assert xo.shape == (2, M, D) and xo.mean(-1).abs().max() < 1e-4
    print(f"transformer block: output {tuple(xo.shape)}, LayerNorm-last (token mean |·|<1e-4): "
          f"{float(xo.mean(-1).abs().max()):.2e}")

    torch.manual_seed(SEED)
    model = TabTransformer(CARDS, N_NUM).eval()
    with torch.no_grad():
        logits = model(Xcat, Xnum)
    n_params = sum(p.numel() for p in model.parameters())
    assert logits.shape == (Xcat.shape[0],) and torch.isfinite(logits).all()

    chk, hidx = CAT.index("checking_status"), CAT.index("housing")
    with torch.no_grad():
        r = Xcat[0:1].clone()
        r2 = r.clone(); r2[0, hidx] = (r[0, hidx] + 1) % CARDS[hidx]
        delta = (model.contextual(r)[0, chk] - model.contextual(r2)[0, chk]).norm().item()
    assert torch.allclose(model.embs[chk].weight[r[0, chk]], model.embs[chk].weight[r2[0, chk]])
    assert delta > 1e-5
    print(f"full TabTransformer: {Xcat.shape[0]} rows -> logits {tuple(logits.shape)}, {n_params:,} params (untrained)")
    print(f"contextual: 'checking_status' entity embedding fixed, contextual vector moved {delta:.3f} "
          f"when 'housing' changed")
    print("\nAll L032 lab claims verified (architecture forward-only; training is L045).")


if __name__ == "__main__":
    main()
