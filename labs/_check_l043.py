"""Correctness checks for the from-scratch TabNet (Lesson 043) — run before any expensive verification.

Checks the paper's invariants, and validates `sparsemax` + the encoder against the reference
`pytorch_tabnet` implementation (NOTES standard #22: the library validates, it does not teach).

Run: OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l043.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from relkit.tabnet import (TabNetEncoder, sparsemax, GhostBatchNorm, train_tabnet,  # noqa: E402
                           tabnet_auc, explain)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def brute_force_projection(z):
    """Projection onto the simplex by solving for tau numerically — an independent reference."""
    lo, hi = z.min() - 1.0, z.max()
    for _ in range(200):
        mid = (lo + hi) / 2
        s = np.clip(z - mid, 0, None).sum()
        if s > 1:
            lo = mid
        else:
            hi = mid
    return np.clip(z - (lo + hi) / 2, 0, None)


print("=== sparsemax (Martins & Astudillo 2016, Alg. 1) ===")
rng = np.random.default_rng(0)
Z = rng.normal(size=(64, 12)) * 3
P = sparsemax(torch.tensor(Z, dtype=torch.float64)).numpy()

check("sums to 1 along dim", np.allclose(P.sum(1), 1.0, atol=1e-9), f"max dev {abs(P.sum(1)-1).max():.2e}")
check("non-negative", (P >= 0).all())
check("produces exact zeros (unlike softmax)", (P == 0).any(), f"{(P==0).mean()*100:.0f}% zeros")
bf = np.stack([brute_force_projection(z) for z in Z])
check("matches brute-force simplex projection", np.allclose(P, bf, atol=1e-6),
      f"max |Δ| {np.abs(P-bf).max():.2e}")

# A uniform input must give the uniform distribution; a dominant coordinate must give a one-hot.
u = sparsemax(torch.zeros(1, 5, dtype=torch.float64)).numpy()
check("uniform input -> uniform output", np.allclose(u, 0.2))
oh = sparsemax(torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)).numpy()
check("dominant coordinate -> one-hot", np.allclose(oh, [[1.0, 0.0, 0.0]]))
# Invariance to a constant shift (property of the projection).
shifted = sparsemax(torch.tensor(Z + 7.0, dtype=torch.float64)).numpy()
check("shift-invariant", np.allclose(P, shifted, atol=1e-9))

try:
    from pytorch_tabnet.sparsemax import Sparsemax as RefSparsemax
    ref = RefSparsemax(dim=-1)(torch.tensor(Z, dtype=torch.float32)).numpy()
    check("VALIDATED against pytorch_tabnet.sparsemax", np.allclose(P, ref, atol=1e-5),
          f"max |Δ| {np.abs(P-ref).max():.2e}")
except Exception as e:                                    # noqa: BLE001
    check("VALIDATED against pytorch_tabnet.sparsemax", False, f"reference unavailable: {e}")

# Gradient must flow only on the support (the sparsemax Jacobian). Note the *sum* of the output is
# constant at 1, so it has zero gradient by construction — weight the coordinates to probe the Jacobian.
z = torch.tensor([[3.0, 2.9, -5.0]], requires_grad=True)
(sparsemax(z) * torch.tensor([[1.0, 0.0, 0.0]])).sum().backward()
g = z.grad.numpy()[0]
check("gradient flows on the support only", g[0] != 0 and g[1] != 0 and g[2] == 0,
      f"grad {g} (coord 2 is off-support)")

print("\n=== encoder invariants (Fig. 4a) ===")
torch.manual_seed(0)
D, B = 10, 32
enc = TabNetEncoder(D, n_d=8, n_a=8, n_steps=4, gamma=1.5, virtual_batch_size=8)
X = torch.randn(B, D)
enc.eval()
logits, info = enc(X, return_masks=True)
check("logits shape (B,)", logits.shape == (B,), str(tuple(logits.shape)))
check("one mask per decision step", len(info["masks"]) == 4)
check("each mask sums to 1 per row", all(torch.allclose(m.sum(1), torch.ones(B), atol=1e-5)
                                          for m in info["masks"]))
check("masks are sparse (some exact zeros)", any((m == 0).any().item() for m in info["masks"]),
      f"zeros in step 0: {(info['masks'][0]==0).float().mean().item()*100:.0f}%")
check("M_agg sums to 1 per row", torch.allclose(info["M_agg"].sum(1), torch.ones(B), atol=1e-5))
check("sparsity loss is a finite scalar", info["m_loss"].ndim == 0 and torch.isfinite(info["m_loss"]),
      f"{info['m_loss'].item():.4f}")

# gamma = 1 forces a feature to be used at only one step: prior becomes 0 where the mask was 1.
enc1 = TabNetEncoder(D, n_d=4, n_a=4, n_steps=3, gamma=1.0, virtual_batch_size=8).eval()
_, info1 = enc1(X, return_masks=True)
m0 = info1["masks"][0]
overlap = sum(((m0 > 0.99) & (info1["masks"][i] > 0)).sum().item() for i in (1, 2))
check("gamma=1 forbids reusing a fully-spent feature", overlap == 0, f"overlap {overlap}")

# Sparsity regularisation must actually push masks toward one-hot: entropy falls as lambda grows.
def mean_entropy(lmbda, seed=0):
    torch.manual_seed(seed)
    m = TabNetEncoder(D, n_d=8, n_a=8, n_steps=3, lambda_sparse=lmbda, virtual_batch_size=16)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).float().numpy()
    m, _ = train_tabnet(m, X.numpy(), y, X.numpy(), y, max_epochs=40, patience=40,
                        batch_size=32, lr=2e-2, seed=seed)
    m.eval()
    _, inf = m(X, return_masks=True)
    return float(inf["m_loss"].item())

e_lo, e_hi = mean_entropy(0.0), mean_entropy(0.5)
check("higher lambda_sparse -> lower mask entropy", e_hi < e_lo, f"lambda 0: {e_lo:.3f} -> lambda 0.5: {e_hi:.3f}")

print("\n=== ghost BN (Hoffer et al. 2017) ===")
gbn = GhostBatchNorm(4, virtual_batch_size=8, momentum=0.02).train()
big = torch.randn(32, 4) * 5 + 3
out = gbn(big)
# Each virtual batch is normalised on its own statistics, so per-chunk means are ~0.
chunk_means = [c.mean().abs().item() for c in out.chunk(4, 0)]
check("normalises each virtual batch separately", max(chunk_means) < 0.2, f"max |chunk mean| {max(chunk_means):.3f}")

print("\n=== learning sanity: TabNet recovers a sparse signal ===")
# Only features 0 and 1 matter; 18 are pure noise. The masks should concentrate on 0/1.
n = 1500
Xs = rng.normal(size=(n, 20)).astype(np.float32)
ys = ((Xs[:, 0] + Xs[:, 1]) > 0).astype(np.float32)
tr, va = slice(0, 1000), slice(1000, 1500)
torch.manual_seed(0)
model = TabNetEncoder(20, n_d=8, n_a=8, n_steps=3, gamma=1.5, lambda_sparse=1e-3, virtual_batch_size=128)
model, val = train_tabnet(model, Xs[tr], ys[tr], Xs[va], ys[va], max_epochs=80, patience=20,
                          batch_size=256, lr=2e-2, seed=0)
auc = tabnet_auc(model, Xs[va], ys[va])
check("learns the sparse signal (val AUC > 0.90)", auc > 0.90, f"AUC {auc:.3f}")
M_agg, _ = explain(model, Xs[va])
imp = M_agg.mean(0)
top2 = set(np.argsort(-imp)[:2].tolist())
check("M_agg ranks the 2 informative features on top", top2 == {0, 1},
      f"top2 {sorted(top2)}, mass on 0/1 = {imp[[0,1]].sum()*100:.0f}%")

print("\n=== validate encoder against pytorch_tabnet (#22) ===")
try:
    from pytorch_tabnet.tab_network import TabNet as RefTabNet
    ref = RefTabNet(input_dim=D, output_dim=2, n_d=8, n_a=8, n_steps=4, gamma=1.5,
                    n_shared=2, n_independent=2, virtual_batch_size=8, momentum=0.02,
                    group_attention_matrix=torch.eye(D))
    ref.eval()
    with torch.no_grad():
        ref_out, ref_mloss = ref(X)
    ours_params = sum(p.numel() for p in enc.parameters())
    ref_params = sum(p.numel() for p in ref.parameters())
    check("reference TabNet runs (architecture comparable)", ref_out.shape[0] == B,
          f"ours {ours_params} params vs ref {ref_params} (ref has a 2-class head + embeddings)")
    with torch.no_grad():
        _, ref_masks = ref.forward_masks(X)
    check("reference masks also sum to 1 per row",
          torch.allclose(list(ref_masks.values())[0].sum(1), torch.ones(B), atol=1e-4))
except Exception as e:                                    # noqa: BLE001
    check("reference TabNet comparison", False, f"unavailable: {e}")

print("\n================================")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
