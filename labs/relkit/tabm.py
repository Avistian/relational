"""Numeric TabM, paper §3 (Gorishniy, Kotelnikov & Babenko, 2410.24210, ICLR 2025).

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
  adapter except the very first R). ``TabM_packed`` (k fully independent MLPs) is
  obtained here by the separate deep-ensemble baseline, not a distinct class.
* This is not the authors' code and does not reproduce the 46-dataset benchmark.

Core BatchEnsemble linear layer (paper §3.2), batched over k members::

    l_BE(X) = ((X ⊙ R) W) ⊙ S + B ,  X ∈ R[B,k,d_in], W ∈ R[d_in,d_out],
                                       R ∈ R[k,d_in], S,B ∈ R[k,d_out]

which is equivalent to giving member i the weight matrix W_i = W ⊙ (s_i r_i^T).
"""
import math
import numpy as np
import torch
from torch import nn


def kaiming_(t):
    """nn.Linear-style init so the shared W starts like a plain MLP layer."""
    nn.init.kaiming_uniform_(t, a=math.sqrt(5))
    return t


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
    start (R=S=1, B=0), matching the ``TabM`` initialisation in paper §3.3.
    ``shared_rs`` (used by the ``mini`` variant) drops the R and S adapters
    entirely for every layer except the first.
    """
    def __init__(self, din, dout, k, first=False, shared_rs=False, generator=None):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(kaiming_(torch.empty(din, dout)))
        self.bias = nn.Parameter(torch.zeros(k, dout))            # additive adapter B
        if shared_rs and not first:
            self.register_parameter('R', None)
            self.register_parameter('S', None)
        else:
            r0 = sign_pm1(k, din, generator) if first else torch.ones(k, din)
            self.R = nn.Parameter(r0)
            self.S = nn.Parameter(torch.ones(k, dout))

    def forward(self, x):                                          # x: [B, k, din]
        return batchensemble_linear(x, self.weight, self.R, self.S, self.bias)


class PackedHead(nn.Module):
    """k independent output heads (one per submodel); d_y is tiny so this is cheap."""
    def __init__(self, width, dy, k):
        super().__init__()
        self.weight = nn.Parameter(kaiming_(torch.empty(k, width, dy)))
        self.bias = nn.Parameter(torch.zeros(k, dy))

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
        h = x.unsqueeze(1).expand(-1, self.k, -1)                 # [B,k,din], k shared copies
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
