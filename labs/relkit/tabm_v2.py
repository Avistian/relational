"""Corrected numeric TabM v2 (2026-09-10), paper §3 (Gorishniy, Kotelnikov & Babenko, 2410.24210, ICLR 2025).

Independent teaching implementation of the *parameter-efficient ensemble* of MLPs.
One TabM object holds k implicit submodels that share the backbone weights W and
differ only through small per-member adapters. The prediction is the mean of the
k submodel predictions.

Scope and honesty boundary
--------------------------
* Numeric inputs only. Categorical one-hot / piecewise-linear feature embeddings
  (the paper's ``dagger`` variants) are deliberately excluded.
* Two variants are implemented: ``mini`` (MiniEnsemble: a single first adapter R)
  and ``tabm`` (BatchEnsemble backbone with the paper's ``R=S=1`` init for every
  adapter except the very first R). ``TabM_packed`` is not implemented: it shares the independent ensemble
  architecture but uses collective stopping and tuning, unlike our deep ensemble.
* This is not the authors' code and does not reproduce the 46-dataset benchmark.

Core BatchEnsemble linear layer (paper §3.2), batched over k members::

    l_BE(X) = ((X ⊙ R) W) ⊙ S + B ,  X ∈ R[B,k,d_in], W ∈ R[d_in,d_out],
                                       R ∈ R[k,d_in], S,B ∈ R[k,d_out]

which is equivalent to giving member i the weight matrix W_i = W ⊙ (r_i s_i^T) in this row-vector storage convention.
"""
import math
import numpy as np
import torch
from torch import nn


def kaiming_(t):
    """Linear-style U(-1/sqrt(d_in), +1/sqrt(d_in)) for [...,d_in,d_out].

    PyTorch's kaiming helper assumes [d_out,d_in] and is wrong for this layout.
    Explicit fan-in also handles the independent [k,d_in,d_out] heads.
    """
    bound = 1 / math.sqrt(t.shape[-2])
    return nn.init.uniform_(t, -bound, bound)


def sign_pm1(k, d, generator=None):
    """Random ±1 initialisation for the first adapter (paper §3.2: diversity)."""
    return torch.where(torch.rand(k, d, generator=generator) < .5, -1., 1.)


# ---- core operations (implemented as TODOs in the companion notebook) ----

def batchensemble_linear(x, weight, R, S, bias):
    """BatchEnsemble layer over k members (paper §3.2).

    x: [B, k, d_in]; weight (shared W): [d_in, d_out]; R: [k, d_in] or None;
    S: [k, d_out] or None; bias (B): [k, d_out]. Returns [B, k, d_out] computed as
    ((x ⊙ R) W) ⊙ S + B, with R/S skipped when None (the mini variant).
    """
    if R is not None:
        x = x * R
    h = torch.einsum('bki,io->bko', x, weight)
    if S is not None:
        h = h * S
    return h + bias


def packed_head(h, weight, bias):
    """k independent output heads. h: [B,k,d]; weight: [k,d,d_y]; bias: [k,d_y]."""
    return torch.einsum('bkd,kdo->bko', h, weight) + bias


def member_mean_loss(out, y, regression):
    """Mean over the k members of each member's loss (paper §5.1). out: [B,k,d_y]."""
    k = out.shape[1]
    if regression:
        return ((out[:, :, 0] - y.unsqueeze(1)) ** 2).mean()
    logp = out.log_softmax(-1)
    nll = -logp.gather(-1, y.view(-1, 1, 1).expand(-1, k, 1)).squeeze(-1)
    return nll.mean()


def ensemble_predict(out, regression):
    """Mean prediction over members: raw for regression, probabilities otherwise."""
    if regression:
        return out.mean(1)[:, 0]
    return out.softmax(-1).mean(1)


class MLP(nn.Module):
    """Plain MLP baseline: N blocks of Linear→ReLU→Dropout, then a linear head.

    This is both the k=1 reference and the member architecture of the deep
    ensemble MLP^{×k}. Definition follows paper §3.3.
    """
    def __init__(self, din, width=64, depth=3, dropout=.1, regression=False, k=1):
        super().__init__()  # k is accepted and ignored so MLP and TabM share a signature
        dims = [din] + [width] * depth
        self.blocks = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1 if regression else 2)
        self.regression = regression

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = self.dropout(torch.relu(block(h)))
        return self.head(h)


class BatchEnsembleLinear(nn.Module):
    """One shared W with per-member multiplicative (R,S) and additive (B) adapters.

    ``first`` selects the diversity-creating adapter that is initialised with
    random ±1; all other adapters are initialised so they have no effect at the
    start (later R=S=1, identical random bias rows), matching the ``TabM`` initialisation in paper §3.3.
    ``shared_rs`` (mini) keeps only the first R, with ordinary shared biases.
    There is no S even in the first mini layer; all later R/S are absent.
    """
    def __init__(self, din, dout, k, first=False, shared_rs=False, generator=None):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(kaiming_(torch.empty(din, dout)))
        # Full TabM starts each member bias with the same nn.Linear-style draw.
        base_bias = torch.empty(dout).uniform_(-1 / math.sqrt(din), 1 / math.sqrt(din))
        self.bias = nn.Parameter(base_bias if shared_rs else base_bias.repeat(k, 1))
        self.R = nn.Parameter(sign_pm1(k, din, generator) if first else torch.ones(k, din)) \
            if first or not shared_rs else None
        self.S = None if shared_rs else nn.Parameter(torch.ones(k, dout))

    def forward(self, x):                                          # x: [B, k, din]
        return batchensemble_linear(x, self.weight, self.R, self.S, self.bias)


class PackedHead(nn.Module):
    """k independent output heads (one per submodel); d_y is tiny so this is cheap."""
    def __init__(self, width, dy, k):
        super().__init__()
        self.weight = nn.Parameter(kaiming_(torch.empty(k, width, dy)))
        self.bias = nn.Parameter(torch.empty(k, dy).uniform_(-1 / math.sqrt(width), 1 / math.sqrt(width)))

    def forward(self, h):                                          # h: [B, k, width]
        return packed_head(h, self.weight, self.bias)


class TabM(nn.Module):
    """One model imitating an ensemble of k MLPs (paper §3.3).

    forward returns per-member outputs of shape [B, k, d_y]. Use
    :meth:`predict` for the mean (ensemble) prediction and :meth:`member_losses`
    for the mean-over-members training objective.
    """
    def __init__(self, din, k=32, width=64, depth=3, dropout=.1,
                 regression=False, arch='mini', seed=0):
        super().__init__()
        assert arch in ('mini', 'tabm')
        g = torch.Generator().manual_seed(seed)
        self.k, self.regression, self.arch = k, regression, arch
        dims = [din] + [width] * depth
        shared_rs = arch == 'mini'
        self.blocks = nn.ModuleList(
            BatchEnsembleLinear(a, b, k, first=(i == 0), shared_rs=shared_rs, generator=g)
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])))
        self.dropout = nn.Dropout(dropout)
        self.head = PackedHead(width, 1 if regression else 2, k)

    def forward(self, x):
        # 3-D inputs permit independent per-member batches in the closer track.
        h = x.unsqueeze(1).expand(-1, self.k, -1) if x.ndim == 2 else x
        if h.ndim != 3 or h.shape[1] != self.k:
            raise ValueError('Expected [B,din] or [B,k,din]')
        for block in self.blocks:
            h = self.dropout(torch.relu(block(h)))
        return self.head(h)                                       # [B,k,d_y]

    def predict(self, x):
        """Ensemble prediction: mean over members (probabilities if classifying)."""
        return ensemble_predict(self(x), self.regression)

    def member_losses(self, x, y):
        """Mean over the k members of each member's supervised loss (paper §5.1)."""
        return member_mean_loss(self(x), y, self.regression)

    def member_predictions(self, x):
        """Per-member predictions [B,k,(d_y)] for individual-vs-collective analysis."""
        out = self(x)
        if self.regression:
            return out[:, :, 0]
        return out.softmax(-1)
