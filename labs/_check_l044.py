"""Correctness checks for the from-scratch NODE (Lesson 044) — the paper's mechanism, validated.

The novel operations NODE is built from are the alpha-entmax feature choice and the two-class "entmoid"
soft split (Popov et al. 2019, §2). We validate BOTH against the reference `entmax` package (Peters et
al. 2019) — the library is the checker, our code is the teacher (NOTES #22) — and assert the structural
invariants of the oblivious tree: leaf routing is a probability distribution, and the tree is genuinely
*oblivious* (one feature per level).

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l044.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit.node import entmax15, entmoid15, ODST, DenseNODE, train_node, node_auc  # noqa: E402

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

    # ---- entmax15: simplex + exact zeros + reference agreement -----------------------------------
    Z = torch.tensor(rng.normal(size=(200, 12)) * 3, dtype=torch.float64)
    P = entmax15(Z, dim=-1, n_iter=50)
    chk("entmax15 sums to 1", torch.allclose(P.sum(-1), torch.ones(200, dtype=torch.float64), atol=1e-6),
        f"max dev {(P.sum(-1) - 1).abs().max():.2e}")
    chk("entmax15 non-negative", bool((P >= 0).all()))
    chk("entmax15 has EXACT zeros (softmax never does)",
        bool((P == 0).any()) and bool((torch.softmax(Z, -1) > 0).all()),
        f"{(P == 0).double().mean() * 100:.0f}% zeros")
    chk("entmax15 uniform input -> uniform output",
        torch.allclose(entmax15(torch.zeros(1, 5, dtype=torch.float64)),
                       torch.full((1, 5), 0.2, dtype=torch.float64), atol=1e-6))

    from entmax import entmax15 as ref_entmax15
    d_ref = (P - ref_entmax15(Z, dim=-1)).abs().max().item()
    chk("entmax15 VALIDATED against entmax.entmax15", d_ref < 1e-5, f"max |Δ| {d_ref:.2e}")

    # gradient flows only through the support (a=1.5 property, like sparsemax)
    Zg = torch.tensor([[2.0, 1.0, -3.0, -5.0]], requires_grad=True)
    entmax15(Zg).sum().backward()
    chk("entmax15 is differentiable (finite grads on support)", torch.isfinite(Zg.grad).all())

    # ---- entmoid15: equals the two-class entmax, and is in [0,1] ----------------------------------
    t = torch.tensor(rng.normal(size=500) * 3, dtype=torch.float64)
    two_class = entmax15(torch.stack([t, torch.zeros_like(t)], -1), dim=-1)[..., 0]
    d_moid = (entmoid15(t) - two_class).abs().max().item()
    chk("entmoid15 == two-class entmax15([x,0])[0]", d_moid < 1e-6, f"max |Δ| {d_moid:.2e}")
    chk("entmoid15 in [0,1]", bool((entmoid15(t) >= 0).all() and (entmoid15(t) <= 1).all()))
    chk("entmoid15 saturates to EXACT 0/1 for large |x| (a genuine hard decision)",
        float(entmoid15(torch.tensor([9.0]))) == 1.0 and float(entmoid15(torch.tensor([-9.0]))) == 0.0)
    chk("entmoid15(0) = 0.5", abs(float(entmoid15(torch.tensor([0.0]))) - 0.5) < 1e-6)

    # ---- ODST forward: routing is a distribution over 2^depth leaves -----------------------------
    depth, trees = 4, 8
    layer = ODST(in_features=10, num_trees=trees, depth=depth, tree_dim=1)
    x = torch.randn(64, 10)
    # reach inside to grab the routing weights and assert they sum to 1 per (row, tree)
    fs = entmax15(layer.feature_logits, dim=0)
    f_hat = torch.einsum("bi,itl->btl", x, fs)
    c = entmoid15((f_hat - layer.thresholds) * torch.exp(-layer.log_temperatures))
    bins = torch.stack([c, 1 - c], dim=-1)
    weights = torch.einsum("btds,dls->btdl", bins, layer.bin_codes_1hot).prod(dim=-2)
    chk("ODST leaf-routing weights sum to 1 per (row, tree)",
        torch.allclose(weights.sum(-1), torch.ones(64, trees), atol=1e-4),
        f"max dev {(weights.sum(-1) - 1).abs().max():.2e}")
    chk(f"ODST has 2^depth = {2**depth} leaves", weights.shape[-1] == 2 ** depth)
    chk("ODST forward returns [batch, trees*tree_dim]", tuple(layer(x).shape) == (64, trees))
    ref = torch.einsum("btl,tcl->btc", weights, layer.response).reshape(x.shape[0], trees)
    chk("ODST forward matches paper outer-product einsum",
        torch.allclose(layer(x), ref, atol=1e-5),
        f"max |Δ| {(layer(x) - ref).abs().max().item():.2e}")

    # ---- oblivious property: the feature choice is shared across a level (one feature per level) ---
    # entmax over features is per (tree, level): a level's choice vector is a single distribution.
    fs_sel = entmax15(layer.feature_logits, dim=0)   # [in, trees, depth]
    per_level_sum = fs_sel.sum(dim=0)                # should be ~1 for every (tree, level)
    chk("oblivious: exactly ONE feature-choice distribution per (tree, level)",
        torch.allclose(per_level_sum, torch.ones(trees, depth), atol=1e-4))

    # ---- learning sanity: NODE fits a simple axis-aligned signal --------------------------------
    n = 1500
    Xs = rng.normal(size=(n, 6)).astype(np.float32)
    ys = ((Xs[:, 0] > 0.3) ^ (Xs[:, 1] > -0.2)).astype(np.float32)   # a 2-feature interaction
    Xtr, ytr = Xs[:1000], ys[:1000]
    Xva, yva = Xs[1000:1250], ys[1000:1250]
    Xte, yte = Xs[1250:], ys[1250:]
    torch.manual_seed(0)
    model = DenseNODE(6, num_trees=64, depth=4, n_layers=1)
    model, val = train_node(model, Xtr, ytr, Xva, yva, lr=1e-2, max_epochs=60, patience=8, seed=0)
    auc = node_auc(model, Xte, yte)
    chk("NODE learns a 2-feature interaction (test AUC > 0.90)", auc > 0.90, f"AUC {auc:.3f}")

    # ---- paper-repro: OpenML Higgs has 1 incomplete row; scaler leaves NaNs that
    # explode NODE logits and make sklearn raise "Input contains NaN."
    import pandas as pd
    from _paper_repro_l044 import _dense_from_xy
    Xnan = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": [1.0, 2.0, 3.0]})
    ynan = pd.Series([0, 1, 0])
    X_clean, y_clean = _dense_from_xy(Xnan, ynan)
    chk("paper-repro drops incomplete rows before scaling",
        np.isfinite(X_clean).all() and len(y_clean) == 2,
        f"shape={X_clean.shape} nans={int(np.isnan(X_clean).sum())}")

    # ---- paper-repro: val/test used to run ODST on the full Higgs split at once.
    # Routing tensor is [N, trees, depth, 2^depth] — 29k × 256 × 6 × 64 × 4B ≈ 11 GB,
    # which OOMs a Colab T4. Eval must chunk.
    torch.manual_seed(0)
    big = DenseNODE(5, num_trees=4, depth=3, n_layers=1)
    Xbig = rng.normal(size=(2000, 5)).astype(np.float32)
    ybig = (Xbig[:, 0] > 0).astype(np.float32)
    seen = []
    _orig = ODST.forward
    def _counted(self, x):
        seen.append(int(x.shape[0]))
        return _orig(self, x)
    ODST.forward = _counted
    try:
        node_auc(big, Xbig, ybig)
    finally:
        ODST.forward = _orig
    chk("node_auc evaluates in batches (max <= 512)",
        bool(seen) and max(seen) <= 512,
        f"forward sizes {seen[:8]}{'...' if len(seen) > 8 else ''}")

    print(f"\n_check_l044: {N_PASS} passed, {N_FAIL} failed")
    sys.exit(1 if N_FAIL else 0)


if __name__ == "__main__":
    main()
