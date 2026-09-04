"""Correctness checks for the from-scratch FT-Transformer (Lesson 046) — the paper's mechanism, validated.

The load-bearing ideas (Gorishniy 2021, §3.3): the **Feature Tokenizer** that turns *every* feature —
numeric included — into a token, the learned **[CLS]** token, and the property that separates FT-T from
TabTransformer (L045): a **numeric** feature now attends, so changing it moves the [CLS] readout. We
validate attention against torch's OWN kernels (the library is the checker, our code is the teacher, #22).

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l046.py
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
    scaled_dot_product_attention, MultiHeadSelfAttention, frame_categorical,
)
from relkit.ft_transformer import (                                                  # noqa: E402
    FeatureTokenizer, FTTransformerBlock, FTTransformer,
    train_ft_transformer, ft_transformer_auc,
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

    # ---- attention kernel: matches torch's reference (reused, validated to machine precision in L045) ----
    Q = torch.randn(3, 6, 16, dtype=torch.float64)
    K = torch.randn(3, 6, 16, dtype=torch.float64)
    V = torch.randn(3, 6, 16, dtype=torch.float64)
    d_sdpa = (scaled_dot_product_attention(Q, K, V)[0] - F.scaled_dot_product_attention(Q, K, V)).abs().max().item()
    chk("scaled_dot_product_attention VALIDATED against torch F.sdpa", d_sdpa < 1e-9, f"max |Δ| {d_sdpa:.2e}")
    d, h = 16, 4
    mha = MultiHeadSelfAttention(d, h).double()
    tmha = torch.nn.MultiheadAttention(d, h, batch_first=True, bias=True).double()
    with torch.no_grad():
        tmha.in_proj_weight.copy_(torch.cat([mha.Wq.weight, mha.Wk.weight, mha.Wv.weight], 0))
        tmha.in_proj_bias.copy_(torch.cat([mha.Wq.bias, mha.Wk.bias, mha.Wv.bias], 0))
        tmha.out_proj.weight.copy_(mha.Wo.weight)
        tmha.out_proj.bias.copy_(mha.Wo.bias)
    x = torch.randn(4, 6, d, dtype=torch.float64)
    d_mha = (mha(x)[0] - tmha(x, x, x)[0]).abs().max().item()
    chk("MultiHeadSelfAttention VALIDATED against nn.MultiheadAttention", d_mha < 1e-9, f"max |Δ| {d_mha:.2e}")

    # ---- Feature Tokenizer: every feature -> one d-token; numeric token is AFFINE in x (the key upgrade) --
    n_num, cards, dd = 4, [3, 5, 2], 8
    tok = FeatureTokenizer(n_num, cards, dd)
    Xn = torch.randn(10, n_num)
    Xc = torch.stack([torch.randint(0, c, (10,)) for c in cards], dim=1)
    T = tok(Xn, Xc)
    chk("tokenizer emits one d-token per feature: [B, n_num+n_cat, d]",
        tuple(T.shape) == (10, n_num + len(cards), dd), f"shape {tuple(T.shape)}")
    with torch.no_grad():
        # numeric token j is b_j + x_j * W_j  =>  moving ONLY feature j by Δ moves token j by Δ*W_j (linear),
        # and leaves every other token byte-identical.
        j = 1
        Xn2 = Xn.clone(); Xn2[:, j] += 2.5
        dT = (tok(Xn2, Xc) - T)
        moved = dT[:, j]                                       # [10, d]
        others_zero = dT.clone(); others_zero[:, j] = 0.0
    expect = 2.5 * tok.num_weight[j].detach()                 # Δ * W_j, same for every row
    chk("numeric token is AFFINE in its feature (Δx moves token j by Δx·W_j, equal across rows)",
        torch.allclose(moved, expect.expand_as(moved), atol=1e-5),
        f"max |Δ| {(moved - expect).abs().max().item():.1e}")
    chk("changing numeric feature j leaves all OTHER tokens unchanged (tokenizer is per-feature)",
        others_zero.abs().max().item() < 1e-6)
    with torch.no_grad():
        # categorical token j = bias_j + embedding_j[value]; two rows with the same category share it
        cat0 = tok(torch.zeros(2, n_num), torch.zeros(2, len(cards), dtype=torch.long))
    chk("categorical token = per-column embedding + per-column bias (identical for identical rows)",
        torch.allclose(cat0[0], cat0[1]))

    # ---- [CLS] token: prepended at position 0, learned, independent of the input row --------------------
    model = FTTransformer(n_num, cards, d=dd, n_layers=2, n_heads=2)
    with torch.no_grad():
        H = model.tokens_with_cls(Xn, Xc)
        cls_rows = H[:, 0]
    chk("tokens_with_cls prepends the [CLS] token: length k+1",
        tuple(H.shape) == (10, n_num + len(cards) + 1, dd))
    chk("position 0 is the SAME learned [CLS] token for every row (before attention)",
        torch.allclose(cls_rows, model.cls.reshape(1, dd).expand(10, dd)))

    # ---- THE upgrade over TabTransformer: a NUMERIC feature moves the [CLS] readout (numbers attend) -----
    model.eval()
    with torch.no_grad():
        z1 = model.cls_readout(Xn, Xc)
        Xn_bumped = Xn.clone(); Xn_bumped[:, 0] += 3.0
        z2 = model.cls_readout(Xn_bumped, Xc)
        num_move = (z1 - z2).norm(dim=1).mean().item()
    chk("changing a NUMERIC feature MOVES the [CLS] readout (numbers attend — TabTransformer's fix)",
        num_move > 1e-4, f"mean L2 move {num_move:.3f}")
    # n_layers=0 ablation: no attention -> [CLS] readout is the raw constant token -> input cannot reach it
    flat = FTTransformer(n_num, cards, d=dd, n_layers=0).eval()
    with torch.no_grad():
        z0a = flat.cls_readout(Xn, Xc)
        z0b = flat.cls_readout(Xn_bumped, Xc)
        flat_move = (z0a - z0b).norm(dim=1).mean().item()
    chk("n_layers=0 ablation: NO attention, so the [CLS] readout ignores the input entirely", flat_move < 1e-9,
        f"mean L2 move {flat_move:.1e}")

    # ---- learning sanity: FT-T learns a NUMERIC-only interaction (proves numerics carry signal via attn) --
    n = 2400
    a = rng.normal(size=n); b = rng.normal(size=n)
    ys = ((a > 0) ^ (b > 0)).astype(np.float32)                # XOR of two numeric thresholds
    Xnum = torch.tensor(np.stack([a, b], 1), dtype=torch.float32)
    Xcat = torch.zeros(n, 0, dtype=torch.long)
    tr, va, te = slice(0, 1600), slice(1600, 2000), slice(2000, n)
    torch.manual_seed(0)
    mm = FTTransformer(2, [], d=32, n_layers=2, n_heads=4)
    mm, _ = train_ft_transformer(mm, Xnum[tr], Xcat[tr], ys[tr], Xnum[va], Xcat[va], ys[va],
                                 max_epochs=60, patience=12, seed=0)
    auc = ft_transformer_auc(mm, Xnum[te], Xcat[te], ys[te])
    chk("FT-Transformer learns a NUMERIC 2-feature interaction (test AUC > 0.90)", auc > 0.90, f"AUC {auc:.3f}")

    # ---- end-to-end on a real mixed table: forward pass is finite, block preserves shape ----------------
    X, y = load_tier_a("adult")
    Xcatr, Xnumr, cardsr, cat_names, num_names = frame_categorical(X.iloc[:512])
    fm = FTTransformer(Xnumr.shape[1], cardsr, d=32, n_layers=2, n_heads=4).eval()
    with torch.no_grad():
        out = fm(Xnumr[:64], Xcatr[:64])
    chk("forward on real adult rows returns one finite logit per row",
        tuple(out.shape) == (64,) and bool(torch.isfinite(out).all()))
    blk = FTTransformerBlock(32, n_heads=4, dropout=0.0)
    xo, _ = blk(torch.randn(4, 7, 32))
    chk("FTTransformerBlock preserves (batch, tokens, d) shape", tuple(xo.shape) == (4, 7, 32))

    print(f"\n_check_l046: {N_PASS} passed, {N_FAIL} failed")
    sys.exit(1 if N_FAIL else 0)


if __name__ == "__main__":
    main()
