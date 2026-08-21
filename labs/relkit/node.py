"""From-scratch NODE (Popov, Morozov & Babenko 2019, arXiv:1909.06312) — Lesson 044.

Neural Oblivious Decision Ensembles. Built from the paper, not from a library (NOTES standards
#18/#22/#24). The `entmax` package (Peters et al. 2019) is used ONLY as a VALIDATION point in
`labs/_verify_l044.py` and `labs/_check_l044.py` — never imported here.

Paper map — every piece below cites the element it realises (paper §2 "Neural Oblivious Decision
Ensembles", Eq. 1–4, Fig. 1):

| Paper                                                        | Here                        |
|-------------------------------------------------------------|-----------------------------|
| alpha-entmax feature choice  F_hat_i = <x, entmax_a(F_i)>    | `entmax15` + `ODST.forward` |
| two-class entmax split (the "entmoid")  c_i in [0,1]         | `entmoid15`                 |
| choice tensor  C(x) = outer_i [c_i, 1-c_i]  (2^d leaves)     | `ODST.forward` (`weights`)  |
| tree output  h(x) = <C(x), R>                                | `ODST.forward` (`response`) |
| oblivious tree: ONE (feature, threshold) shared per level    | `ODST` parameter shapes     |
| DenseNet-style multi-layer stacking (Fig. 1)                | `DenseNODE`                 |

Binary classification only (tree_dim = 1, averaged over trees to one logit) — all Lesson 044 needs.
CPU is fine for the small Tier-A tables.

Why a differentiable oblivious tree at all? An oblivious decision tree (CatBoost's symmetric tree,
L016) uses the SAME split feature + threshold at every node of a level, so a depth-`d` tree is fully
described by `d` features, `d` thresholds, and 2^d leaf responses. That regularity is what lets us make
it differentiable: replace "pick feature j" with a sparse `entmax` choice over features, and replace the
hard comparison `f > b` with a soft, temperature-controlled `entmoid`. Every leaf then receives a
*fraction* of the row (they sum to 1), so the whole tree is a smooth function of its parameters and
trains by gradient descent end-to-end — unlike a GBDT, whose splits are chosen greedily and are not
differentiable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------- sparse transforms
def entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 30) -> torch.Tensor:
    """alpha=1.5 entmax by bisection on the threshold tau (Peters, Niculae & Martins 2019).

    entmax_a(z) = argmax_p <p, z> + H_a(p) over the probability simplex, with Tsallis entropy H_a.
    Its solution has the closed shape

        p_i = [ (a-1) z_i - tau ]_+ ^ (1/(a-1))

    for the unique tau that makes the coordinates sum to 1. Like sparsemax (a=2, L043) it produces
    EXACT zeros — so it is a differentiable *selection* — but at a=1.5 it sits between softmax (a->1,
    dense) and sparsemax (a=2, hardest), which the paper found trains best for feature choice.

    We find tau by bisection. Autograd differentiates through the (fixed `n_iter`) iterations, so no
    custom backward is needed; the forward values match `entmax.entmax15` to ~1e-6 (see _check_l044).
    """
    alpha = 1.5
    z = (alpha - 1) * z                                   # fold (a-1) into z so p_i = [z_i - tau]_+^2
    zmax = z.max(dim=dim, keepdim=True).values
    # p_i is maximal when z_i = zmax; the simplex solution's tau lies in this bracket:
    tau_lo = zmax - 1.0
    tau_hi = zmax - (1.0 / z.shape[dim]) ** (alpha - 1)
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / (alpha - 1))
        below = p.sum(dim=dim, keepdim=True) < 1.0        # sum too small -> tau too high, lower it
        tau_hi = torch.where(below, tau, tau_hi)
        tau_lo = torch.where(below, tau_lo, tau)
    tau = (tau_lo + tau_hi) / 2
    p = torch.clamp(z - tau, min=0) ** (1.0 / (alpha - 1))
    return p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def entmoid15(x: torch.Tensor) -> torch.Tensor:
    """Two-class alpha=1.5 entmax — the "entmoid", NODE's differentiable, sparse replacement for sigmoid.

    entmoid15(x) is exactly `entmax15([x, 0])[..., 0]`: the probability the soft split sends a row to
    the "greater-than-threshold" side. Because it is entmax, once `|x|` is large enough the row is routed
    fully (probability an exact 0 or 1) — a genuinely discrete decision, reached smoothly. We use the
    closed form (NODE repo `lib/nn_utils.py`), which _check_l044 asserts equals the two-class entmax.

        tau = (|x| + sqrt(relu(8 - x^2))) / 2 ;  y = 0.25 * relu(tau - |x|)^2  (the smaller-class mass)
    """
    is_pos = x >= 0
    ax = x.abs()
    tau = (ax + torch.sqrt(F.relu(8 - ax ** 2))) / 2
    tau = torch.where(tau <= ax, torch.full_like(tau, 2.0), tau)
    y_neg = 0.25 * F.relu(tau - ax) ** 2
    return torch.where(is_pos, 1 - y_neg, y_neg)


# ---------------------------------------------------------------------------- one NODE layer (ODST)
class ODST(nn.Module):
    """An ensemble of `num_trees` differentiable Oblivious Decision Trees of depth `depth` (paper §2).

    "Oblivious" = the tree uses ONE (feature, threshold) pair per LEVEL, shared by every node in that
    level (CatBoost's symmetric tree, L016). So the parameters are just:

      feature_logits  F : [in_features, num_trees, depth]   -> a sparse feature choice per (tree, level)
      thresholds      b : [num_trees, depth]                 -> one split point per (tree, level)
      log_temp    log_t : [num_trees, depth]                 -> soft-split temperature per (tree, level)
      response        R : [num_trees, tree_dim, 2^depth]     -> a learnable answer per leaf

    Forward (Eq. 2–4):
      1. feature choice   f_hat[b,t,l] = <x[b], entmax15(F[:,t,l])>            (differentiable "pick")
      2. soft split       c[b,t,l]     = entmoid15( (f_hat - b) * exp(-log_t) )
      3. routing tensor   w[b,t,leaf]  = prod_l  ( c or 1-c, per the leaf's bit at level l )
      4. tree output      out[b,t,:]   = sum_leaf w * R                        (weighted leaf average)

    Returns [batch, num_trees * tree_dim].
    """

    def __init__(self, in_features, num_trees=128, depth=6, tree_dim=1):
        super().__init__()
        self.in_features, self.num_trees, self.depth, self.tree_dim = in_features, num_trees, depth, tree_dim

        self.feature_logits = nn.Parameter(torch.randn(in_features, num_trees, depth))
        self.thresholds = nn.Parameter(torch.randn(num_trees, depth))
        self.log_temperatures = nn.Parameter(torch.zeros(num_trees, depth))
        self.response = nn.Parameter(torch.randn(num_trees, tree_dim, 2 ** depth) * 0.1)

        # bin_codes_1hot[l, leaf, s]: at level l, which side (s=0 "right"/c, s=1 "left"/1-c) leaf uses.
        leaves = torch.arange(2 ** depth)
        offsets = 2 ** torch.arange(depth)
        bit = (leaves.view(1, -1) // offsets.view(-1, 1)) % 2          # [depth, 2^depth], 1 => "right"
        bin_codes = torch.stack([bit, 1 - bit], dim=-1).float()        # [depth, 2^depth, 2]
        self.register_buffer("bin_codes_1hot", bin_codes)

    def initialize(self, x: torch.Tensor):
        """Data-aware init (paper App.): set thresholds to sampled feature quantiles so early splits are
        informative, and scale temperatures to the spread of the chosen feature values."""
        with torch.no_grad():
            fs = entmax15(self.feature_logits, dim=0)                  # [in, trees, depth]
            f_hat = torch.einsum("bi,itl->btl", x, fs)                 # [batch, trees, depth]
            # thresholds <- a random row's feature value per (tree, level); temperature <- its std.
            idx = torch.randint(0, x.shape[0], (self.num_trees, self.depth))
            self.thresholds.data = f_hat[idx, torch.arange(self.num_trees).view(-1, 1),
                                         torch.arange(self.depth).view(1, -1)]
            self.log_temperatures.data = torch.log(f_hat.std(dim=0).clamp_min(1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_selectors = entmax15(self.feature_logits, dim=0)       # sparse choice over features
        f_hat = torch.einsum("bi,itl->btl", x, feature_selectors)      # [batch, trees, depth]
        logits = (f_hat - self.thresholds) * torch.exp(-self.log_temperatures)
        c = entmoid15(logits)                                          # [batch, trees, depth] soft split
        bins = torch.stack([c, 1 - c], dim=-1)                         # [batch, trees, depth, 2]
        # match each leaf's per-level bit -> [batch, trees, depth, 2^depth], then product over levels
        bin_matches = torch.einsum("btds,dls->btdl", bins, self.bin_codes_1hot)
        weights = bin_matches.prod(dim=-2)                             # [batch, trees, 2^depth]
        out = torch.einsum("btl,tcl->btc", weights, self.response)     # [batch, trees, tree_dim]
        return out.reshape(x.shape[0], self.num_trees * self.tree_dim)


class DenseNODE(nn.Module):
    """NODE = one or more ODST layers, DenseNet-style: each layer sees the input PLUS every earlier
    layer's tree outputs (Fig. 1). The prediction averages the first output unit of every tree across
    all layers (paper: average of all trees' responses), giving one logit for binary classification.
    """

    def __init__(self, in_features, num_trees=128, depth=6, n_layers=1, tree_dim=1):
        super().__init__()
        self.n_layers, self.num_trees, self.tree_dim = n_layers, num_trees, tree_dim
        self.layers = nn.ModuleList()
        d = in_features
        for _ in range(n_layers):
            self.layers.append(ODST(d, num_trees=num_trees, depth=depth, tree_dim=tree_dim))
            d += num_trees * tree_dim                                  # dense concat of this layer's out

    @torch.no_grad()
    def initialize(self, x):
        h = x
        for layer in self.layers:
            layer.initialize(h)
            h = torch.cat([h, layer(h)], dim=-1)

    def forward(self, x):
        h = x
        outputs = []
        for layer in self.layers:
            out = layer(h)                                             # [batch, trees*tree_dim]
            outputs.append(out.view(x.shape[0], self.num_trees, self.tree_dim)[..., 0])
            h = torch.cat([h, out], dim=-1)
        # average the first response unit over all trees in all layers -> one logit
        return torch.cat(outputs, dim=1).mean(dim=1)


def train_node(model, Xtr, ytr, Xva, yva, *, lr=1e-3, wd=0.0, max_epochs=100, patience=10,
               batch_size=512, device="cpu", seed=0):
    """Mini-batch Adam with early stopping on validation ROC-AUC — the same fair, shared-protocol
    contract as `relkit.nets.train_net` (L042) and `relkit.tabnet.train_tabnet` (L043): every model
    picks its own training length by validation, so no arm is accidentally under- or over-trained.

    NODE-specific: a data-aware `initialize` pass on the first training batch (paper appendix) so the
    thresholds start at real feature quantiles rather than random N(0,1) draws.

    Returns (model_with_best_val_weights, best_val_auc).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    Xt = torch.tensor(np.asarray(Xtr), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.float32, device=device)
    Xv = torch.tensor(np.asarray(Xva), dtype=torch.float32, device=device)
    n = Xt.size(0)
    bs = min(batch_size, n)

    model.initialize(Xt[:min(n, 2048)])                                # data-aware threshold init
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    best_auc, best_state, since = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).cpu().numpy()
        val_auc = roc_auc_score(yva, pv)
        if val_auc > best_auc:
            best_auc, since = val_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def node_auc(model, X, y, *, device="cpu"):
    model.eval()
    logits = model(torch.tensor(np.asarray(X), dtype=torch.float32, device=device))
    return roc_auc_score(y, torch.sigmoid(logits).cpu().numpy())
