"""From-scratch TabNet (Arik & Pfister 2019, arXiv:1908.07442) — Lesson 043.

Built from the paper, not from a library (NOTES standards #18/#22/#24). `pytorch_tabnet` is used only
as a VALIDATION point in `labs/_verify_l043.py`.

Paper map — every piece below cites the element it realises:

| Paper                        | Here                                        |
|------------------------------|---------------------------------------------|
| sparsemax (Martins 2016 Alg. 1) | `sparsemax`                              |
| Ghost BN (Fig. 4c, Hoffer 2017) | `GhostBatchNorm`                         |
| FC -> BN -> GLU (Fig. 4c)       | `GLUBlock`                               |
| Feature transformer (Fig. 4c)   | `FeatureTransformer`                     |
| M[i] = sparsemax(P[i-1] * h_i(a[i-1])) (Fig. 4d) | `AttentiveTransformer`  |
| P[i] = prod_j (gamma - M[j])    | `TabNetEncoder.forward` (`prior`)        |
| L_sparse = sum -M log(M+eps) / (N_steps*B) | `TabNetEncoder.forward` (`m_loss`) |
| d_out = sum_i ReLU(d[i])        | `TabNetEncoder.forward` (`d_out`)        |
| M_agg (eta-weighted masks)      | `TabNetEncoder.forward` (`M_agg`)        |

Binary classification only (one logit) — that is all the Lesson 043 lab needs. CPU is fine for the
small Tier-A tables.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection of `z` onto the probability simplex (Martins & Astudillo 2016, Alg. 1).

    Unlike softmax, the output has EXACT zeros: coordinates below a threshold tau are clipped away, so
    the mask can genuinely switch a feature off. Still sums to 1 along `dim`.

        1. sort z descending -> z_(1) >= ... >= z_(K)
        2. k(z) = max{ k : k * z_(k) > (sum_{j<=k} z_(j)) - 1 }
        3. tau(z) = ((sum_{j<=k(z)} z_(j)) - 1) / k(z)
        4. p_i = max(z_i - tau(z), 0)

    Autograd differentiates the projection directly, which reproduces the paper's sparsemax Jacobian
    (gradient flows only on the support), so no custom backward is needed.
    """
    # Shifting by the max is a no-op for the projection but keeps the sums well-scaled.
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    shape = [1] * z.dim()
    shape[dim] = -1
    arange = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(shape)
    cssv = z_sorted.cumsum(dim) - 1.0            # (sum_{j<=k} z_(j)) - 1
    support = (arange * z_sorted) > cssv          # step 2 condition
    k = support.to(z.dtype).sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, k.long() - 1) / k      # step 3
    return torch.clamp(z - tau, min=0.0)          # step 4


class GhostBatchNorm(nn.Module):
    """BatchNorm over virtual sub-batches of size `virtual_batch_size` (Hoffer et al. 2017).

    TabNet trains with very large batches (Appendix F: 1-10% of the training set). Normalising the
    whole batch at once would average away too much noise, so the batch is split into virtual batches
    and each is normalised on its own statistics. The paper keeps ordinary BN on the input features and
    uses ghost BN everywhere inside the blocks.

    NOTE on momentum: the paper's m_B (0.6-0.98) is TensorFlow-style *decay*. PyTorch's `momentum` is
    the complement, so pass `1 - m_B`.
    """

    def __init__(self, dim: int, virtual_batch_size: int = 128, momentum: float = 0.02):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(dim, momentum=momentum)

    def forward(self, x):
        if self.training and x.size(0) > self.virtual_batch_size:
            n_chunks = math.ceil(x.size(0) / self.virtual_batch_size)
            return torch.cat([self.bn(chunk) for chunk in x.chunk(n_chunks, dim=0)], dim=0)
        return self.bn(x)


class GLUBlock(nn.Module):
    """FC -> (ghost) BN -> GLU, the unit the feature transformer is stacked from (Fig. 4c).

    The FC layer emits `2 * d_out` values; GLU (Dauphin et al. 2016) splits them and computes
    `a * sigmoid(b)`, so half the activations act as a learned gate on the other half. The paper's
    ablation (Table 11) shows GLU beating ReLU here.

    `shared_linear` lets a layer be shared across all decision steps (parameter efficiency, Fig. 4c).
    """

    def __init__(self, d_in, d_out, *, shared_linear=None, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.fc = shared_linear if shared_linear is not None else nn.Linear(d_in, 2 * d_out, bias=False)
        self.bn = GhostBatchNorm(2 * d_out, virtual_batch_size, momentum)

    def forward(self, x):
        x = self.bn(self.fc(x))
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class FeatureTransformer(nn.Module):
    """Shared + step-dependent GLU blocks with sqrt(0.5)-normalised residuals (Fig. 4c).

    The paper's default is 2 shared layers (same weights at every decision step, since every step sees
    the same features) followed by 2 step-dependent layers. Residual connections are scaled by
    sqrt(0.5) so the variance does not grow as blocks are stacked (Gehring et al. 2017).

    The first block changes width (D -> n_d + n_a) so it has no residual.
    """

    SCALE = math.sqrt(0.5)

    def __init__(self, d_in, d_out, shared_linears, *, n_independent=2,
                 virtual_batch_size=128, momentum=0.02):
        super().__init__()
        blocks, d = [], d_in
        for lin in shared_linears:
            blocks.append(GLUBlock(d, d_out, shared_linear=lin,
                                   virtual_batch_size=virtual_batch_size, momentum=momentum))
            d = d_out
        for _ in range(n_independent):
            blocks.append(GLUBlock(d, d_out, virtual_batch_size=virtual_batch_size, momentum=momentum))
            d = d_out
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            out = block(x)
            x = out if i == 0 else (x + out) * self.SCALE
        return x


class AttentiveTransformer(nn.Module):
    """M[i] = sparsemax(P[i-1] * h_i(a[i-1]))  (Fig. 4d).

    `h_i` is one FC layer + ghost BN. The prior scale `P[i-1]` multiplies the logits, so a feature that
    has already been spent at earlier steps is pushed down and sparsemax is likely to zero it out.
    """

    def __init__(self, d_a, d_features, *, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.fc = nn.Linear(d_a, d_features, bias=False)
        self.bn = GhostBatchNorm(d_features, virtual_batch_size, momentum)

    def forward(self, a, prior):
        return sparsemax(prior * self.bn(self.fc(a)), dim=-1)


class TabNetEncoder(nn.Module):
    """TabNet encoder for binary classification (Fig. 4a).

    Args mirror the paper's names: `n_d` (N_d, decision width), `n_a` (N_a, attention width),
    `n_steps` (N_steps), `gamma` (relaxation), `lambda_sparse`.

    forward returns `(logits, info)` where `info` carries the per-step masks, the aggregate importance
    mask `M_agg`, and the sparsity loss `m_loss` — the interpretability outputs Lesson 043 reads.
    """

    def __init__(self, d_in, *, n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                 n_shared=2, n_independent=2, virtual_batch_size=128, momentum=0.02, epsilon=1e-15):
        super().__init__()
        self.d_in, self.n_d, self.n_a, self.n_steps = d_in, n_d, n_a, n_steps
        self.gamma, self.lambda_sparse, self.epsilon = gamma, lambda_sparse, epsilon

        # Ordinary BN on the raw input features (the paper deliberately does NOT ghost-BN these).
        self.initial_bn = nn.BatchNorm1d(d_in, momentum=0.01)

        d_split = n_d + n_a
        # Shared FC layers: reused by the initial transformer AND every step's transformer (Fig. 4c).
        shared = []
        for i in range(n_shared):
            shared.append(nn.Linear(d_in if i == 0 else d_split, 2 * d_split, bias=False))
        self.shared_linears = nn.ModuleList(shared)

        ft_kwargs = dict(n_independent=n_independent, virtual_batch_size=virtual_batch_size,
                         momentum=momentum)
        # Step 0 has no mask yet: an initial transformer produces a[0] to seed the first attention.
        self.initial_transformer = FeatureTransformer(d_in, d_split, shared, **ft_kwargs)
        self.step_transformers = nn.ModuleList(
            [FeatureTransformer(d_in, d_split, shared, **ft_kwargs) for _ in range(n_steps)])
        self.attentive = nn.ModuleList(
            [AttentiveTransformer(n_a, d_in, virtual_batch_size=virtual_batch_size, momentum=momentum)
             for _ in range(n_steps)])
        self.final = nn.Linear(n_d, 1, bias=False)

    def forward(self, x, *, return_masks: bool = False):
        x = self.initial_bn(x)
        prior = torch.ones_like(x)                       # P[0] = 1 (no prior on any feature)
        a = self.initial_transformer(x)[:, self.n_d:]    # a[0]
        d_out = torch.zeros(x.size(0), self.n_d, device=x.device, dtype=x.dtype)
        m_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        masks, etas = [], []

        for step in range(self.n_steps):
            M = self.attentive[step](a, prior)           # M[i]
            # L_sparse: entropy of the mask, summed over features, averaged over batch and steps.
            m_loss = m_loss + (-M * torch.log(M + self.epsilon)).sum(dim=1).mean() / self.n_steps
            prior = prior * (self.gamma - M)             # P[i] = prod_j (gamma - M[j])

            out = self.step_transformers[step](M * x)    # f_i(M[i] . f)
            d = F.relu(out[:, :self.n_d])                # d[i], ReLU per the decision aggregation
            a = out[:, self.n_d:]                        # a[i] feeds the next step's attention
            d_out = d_out + d
            if return_masks:
                masks.append(M)
                etas.append(d.sum(dim=1, keepdim=True))  # eta_b[i] = sum_c ReLU(d_b,c[i])

        logits = self.final(d_out).squeeze(-1)
        info = {"m_loss": m_loss}
        if return_masks:
            stacked_eta = torch.stack(etas, dim=0)               # steps x B x 1
            stacked_M = torch.stack(masks, dim=0)                # steps x B x D
            agg = (stacked_eta * stacked_M).sum(dim=0)           # B x D
            info.update(masks=[m.detach() for m in masks],
                        etas=[e.detach() for e in etas],
                        M_agg=(agg / agg.sum(dim=1, keepdim=True).clamp_min(1e-12)).detach())
        return logits, info


def train_tabnet(model, Xtr, ytr, Xva, yva, *, lr=2e-2, wd=0.0, max_epochs=120, patience=15,
                 batch_size=1024, decay_rate=0.9, decay_every=20, device="cpu", seed=0):
    """Mini-batch Adam training with early stopping on validation ROC-AUC.

    Same fair-protocol contract as `relkit.nets.train_net` (L042): the model picks its own training
    length by validation, so no arm is accidentally under- or over-trained. Two TabNet-specific pieces
    from the paper: the sparsity penalty `lambda_sparse * m_loss` is added to the loss, and the learning
    rate decays geometrically (Appendix F: "initially large learning rate ... gradually decayed").

    Returns (model_with_best_val_weights, best_val_auc).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_every, gamma=decay_rate)
    lossf = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(np.asarray(Xtr), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.float32, device=device)
    Xv = torch.tensor(np.asarray(Xva), dtype=torch.float32, device=device)
    n = Xt.size(0)
    bs = min(batch_size, n)
    generator = torch.Generator().manual_seed(seed)

    best_auc, best_state, since = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:      # BatchNorm needs >1 row
                continue
            opt.zero_grad()
            logits, info = model(Xt[idx])
            loss = lossf(logits, yt[idx]) + model.lambda_sparse * info["m_loss"]
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)[0]).cpu().numpy()
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
def tabnet_auc(model, X, y, *, device="cpu"):
    model.eval()
    logits, _ = model(torch.tensor(np.asarray(X), dtype=torch.float32, device=device))
    return roc_auc_score(y, torch.sigmoid(logits).cpu().numpy())


@torch.no_grad()
def explain(model, X, *, device="cpu"):
    """Return (M_agg, per-step masks) for rows `X` — the paper's local/global interpretability outputs."""
    model.eval()
    _, info = model(torch.tensor(np.asarray(X), dtype=torch.float32, device=device), return_masks=True)
    return info["M_agg"].cpu().numpy(), [m.cpu().numpy() for m in info["masks"]]
