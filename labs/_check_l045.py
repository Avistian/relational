"""Correctness checks for the from-scratch TabTransformer (Lesson 045) — the paper's mechanism, validated.

The load-bearing operations (Huang 2020, §3): multi-head self-attention over the categorical tokens,
the Transformer block's residuals, the contextual property, and the RTD pre-training pretext. We validate
the attention against torch's OWN reference kernels — `nn.functional.scaled_dot_product_attention` and
`nn.MultiheadAttention` — the library is the checker, our code is the teacher (NOTES #22).

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l045.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit import load_tier_a                                                       # noqa: E402
from relkit.tabtransformer import (                                                  # noqa: E402
    frame_categorical, scaled_dot_product_attention, MultiHeadSelfAttention,
    TransformerBlock, TabTransformer, corrupt_categorical, RTDHead,
    train_tabtransformer, tabtransformer_auc, pretrain_rtd,
)

N_PASS = 0
N_FAIL = 0


def chk(name, cond, detail=""):
    global N_PASS, N_FAIL
    ok = bool(cond)
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    N_PASS += ok
    N_FAIL += (not ok)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---- scaled dot-product attention: matches torch's reference kernel + weights are a distribution ---
    Q = torch.randn(3, 6, 16, dtype=torch.float64)
    K = torch.randn(3, 6, 16, dtype=torch.float64)
    V = torch.randn(3, 6, 16, dtype=torch.float64)
    out, w = scaled_dot_product_attention(Q, K, V)
    ref = F.scaled_dot_product_attention(Q, K, V)
    d_sdpa = (out - ref).abs().max().item()
    chk("scaled_dot_product_attention VALIDATED against torch F.sdpa", d_sdpa < 1e-9, f"max |Δ| {d_sdpa:.2e}")
    chk("attention weights sum to 1 per query (softmax over keys)",
        torch.allclose(w.sum(-1), torch.ones(3, 6, dtype=torch.float64), atol=1e-9))
    chk("attention weights non-negative", bool((w >= 0).all()))

    # ---- multi-head self-attention: matches nn.MultiheadAttention with copied weights ------------------
    d, h = 16, 4
    mha = MultiHeadSelfAttention(d, h).double()
    tmha = torch.nn.MultiheadAttention(d, h, batch_first=True, bias=True).double()
    with torch.no_grad():
        tmha.in_proj_weight.copy_(torch.cat([mha.Wq.weight, mha.Wk.weight, mha.Wv.weight], 0))
        tmha.in_proj_bias.copy_(torch.cat([mha.Wq.bias, mha.Wk.bias, mha.Wv.bias], 0))
        tmha.out_proj.weight.copy_(mha.Wo.weight)
        tmha.out_proj.bias.copy_(mha.Wo.bias)
    x = torch.randn(4, 6, d, dtype=torch.float64)
    o_mine, _ = mha(x)
    o_ref, _ = tmha(x, x, x)
    d_mha = (o_mine - o_ref).abs().max().item()
    chk("MultiHeadSelfAttention VALIDATED against nn.MultiheadAttention", d_mha < 1e-9, f"max |Δ| {d_mha:.2e}")

    # ---- Transformer block: shape preserved + LayerNorm applied last -----------------------------------
    blk = TransformerBlock(d, n_heads=h, dropout=0.0)
    xo, w = blk(torch.randn(4, 6, d))
    chk("TransformerBlock preserves (batch, m, d) shape", tuple(xo.shape) == (4, 6, d))
    chk("TransformerBlock applies LayerNorm last (token ~zero-mean over d)",
        xo.mean(dim=-1).abs().max().item() < 1e-4, f"max |mean| {xo.mean(-1).abs().max():.1e}")

    # ---- contextual property: a column's vector moves when a NEIGHBOUR changes; context-free does not ---
    X, y = load_tier_a("credit_g")
    Xcat, Xnum, cards, cat_names, num_names = frame_categorical(X)
    torch.manual_seed(0)
    model = TabTransformer(cards, Xnum.shape[1], d=32, n_layers=2, n_heads=4).eval()
    chk_i, nb_i = cat_names.index("checking_status"), cat_names.index("housing")
    with torch.no_grad():
        r = Xcat[0:1].clone()
        r2 = r.clone(); r2[0, nb_i] = (r[0, nb_i] + 1) % cards[nb_i]                  # flip housing only
        ctx_delta = (model.contextual(r)[0, chk_i] - model.contextual(r2)[0, chk_i]).norm().item()
    emb_same = torch.allclose(model.embs[chk_i].weight[r[0, chk_i]], model.embs[chk_i].weight[r2[0, chk_i]])
    chk("checking_status context-free embedding is byte-identical (neighbour changed, not it)", emb_same)
    chk("checking_status CONTEXTUAL vector MOVES when housing changes (attention did something)",
        ctx_delta > 1e-4, f"L2 move {ctx_delta:.3f}")

    # context-free model (n_layers=0) must NOT move — no attention to mix neighbours
    cf = TabTransformer(cards, Xnum.shape[1], d=32, n_layers=0).eval()
    with torch.no_grad():
        cf_delta = (cf.contextual(r)[0, chk_i] - cf.contextual(r2)[0, chk_i]).norm().item()
    chk("n_layers=0 ablation is context-FREE (a column's vector does NOT move)", cf_delta < 1e-9,
        f"L2 move {cf_delta:.1e}")

    # ---- RTD corruption + detector: a valid self-supervised pretext -----------------------------------
    gen = torch.Generator().manual_seed(0)
    xcorr, replaced = corrupt_categorical(Xcat, cards, 0.3, gen)
    frac = replaced.mean().item()
    chk("RTD corruption replaces ~30% of tokens (minus accidental self-draws)", 0.18 < frac < 0.30,
        f"replaced {frac:.3f}")
    chk("replaced label is exactly where the token changed", bool(((xcorr != Xcat).float() == replaced).all()))
    chk("unchanged tokens keep their original value", bool((xcorr[replaced == 0] == Xcat[replaced == 0]).all()))
    head = RTDHead(len(cards), 32)
    chk("RTDHead emits one detection logit per column", tuple(head(model.contextual(Xcat[:4])).shape) == (4, len(cards)))

    # ---- RTD pre-training beats chance: the encoder learns which rows are coherent ---------------------
    torch.manual_seed(0)
    pm = TabTransformer(cards, Xnum.shape[1], d=32, n_layers=2, n_heads=4)
    pm, acc = pretrain_rtd(pm, Xcat, cards, replace_p=0.3, max_epochs=20, seed=0)
    chk("RTD detector learns above chance on unlabeled rows (acc > 0.80)", acc > 0.80, f"detector acc {acc:.3f}")

    # ---- learning sanity: TabTransformer fits a categorical interaction ------------------------------
    n = 2000
    a = rng.integers(0, 4, size=n); b = rng.integers(0, 4, size=n)
    ys = ((a >= 2) ^ (b >= 2)).astype(np.float32)                                     # a 2-column interaction
    Xc = torch.tensor(np.stack([a, b], 1), dtype=torch.long)
    Xn = torch.zeros(n, 0)
    tr, va, te = slice(0, 1400), slice(1400, 1700), slice(1700, n)
    torch.manual_seed(0)
    mm = TabTransformer([4, 4], 0, d=16, n_layers=2, n_heads=2)
    mm, _ = train_tabtransformer(mm, Xc[tr], Xn[tr], ys[tr], Xc[va], Xn[va], ys[va],
                                 max_epochs=60, patience=10, seed=0)
    auc = tabtransformer_auc(mm, Xc[te], Xn[te], ys[te])
    chk("TabTransformer learns a 2-column interaction (test AUC > 0.90)", auc > 0.90, f"AUC {auc:.3f}")

    print(f"\n_check_l045: {N_PASS} passed, {N_FAIL} failed")
    sys.exit(1 if N_FAIL else 0)


if __name__ == "__main__":
    main()
