"""Readable DCNv2: Wang et al., arXiv:2008.13535v2, Eqs. 1–4 / Fig. 1.

Rows are stored as [batch, d]; torch Linear stores [out, in], hence W.T.
Dense and linear low-rank crosses have degree <= L+1 in x0. The nonlinear
expert and input-dependent gate do NOT inherit that polynomial guarantee.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def cross_step(x0, x, weight, bias):
    """Eq. 1. Bias is INSIDE multiplication by the original input."""
    return x + x0 * F.linear(x, weight, bias)


def lowrank_step(x0, x, u, v, bias):
    """Eq. 2: W = U V.T. Both U and V have shape [d, r]."""
    hidden = x @ v
    return x + x0 * (hidden @ u.T + bias)


def mix_step(x0, x, u, v, c, bias, gate_logits):
    """Eqs. 3–4, chosen tanh activation and normalized softmax gates.

    u,v: [K,d,r]; c: [K,r,r]; shared bias: [d]; gate_logits: [B,K].
    The paper also allows other gates/activations; this is one stated variant.
    """
    hidden = torch.tanh(torch.einsum('bd,kdr->bkr', x, v))
    hidden = torch.tanh(torch.einsum('bkr,ksr->bks', hidden, c))
    experts = x0[:, None, :] * (torch.einsum('bkr,kdr->bkd', hidden, u) + bias)
    gates = gate_logits.softmax(dim=-1)
    return x + (gates[:, :, None] * experts).sum(dim=1)


class CrossLayer(nn.Module):
    """Choose a dense matrix, linear factorization, or nonlinear experts."""
    def __init__(self, d, kind='dense', rank=4, experts=2):
        super().__init__()
        if kind not in ('dense', 'lowrank', 'mix'):
            raise ValueError('Unknown cross kind')
        if not 1 <= rank <= d:
            raise ValueError('rank must be between 1 and input width')
        self.kind = kind
        self.bias = nn.Parameter(torch.zeros(d))
        if kind == 'dense':
            self.weight = nn.Parameter(torch.empty(d, d))
            nn.init.kaiming_normal_(self.weight)
        elif kind == 'lowrank':
            self.u = nn.Parameter(torch.empty(d, rank))
            self.v = nn.Parameter(torch.empty(d, rank))
            nn.init.xavier_normal_(self.u)
            nn.init.xavier_normal_(self.v)
        else:
            self.u = nn.Parameter(torch.empty(experts, d, rank))
            self.v = nn.Parameter(torch.empty(experts, d, rank))
            self.c = nn.Parameter(torch.empty(experts, rank, rank))
            self.gate = nn.Linear(d, experts, bias=False)
            for stack in (self.u, self.v, self.c):
                for matrix in stack:
                    nn.init.xavier_normal_(matrix)

    def forward(self, x0, x):
        if self.kind == 'dense':
            return cross_step(x0, x, self.weight, self.bias)
        if self.kind == 'lowrank':
            return lowrank_step(x0, x, self.u, self.v, self.bias)
        return mix_step(x0, x, self.u, self.v, self.c, self.bias, self.gate(x))


class RowEmbedding(nn.Module):
    """§3.1: concatenate categorical embeddings and normalized numerics.

    Vocabulary code 0 represents missing/unseen categories. No labels enter.
    No learned numeric token or CLS token: x0 is one flat vector per row.
    """
    def __init__(self, n_num, cards, embedding_dim=4):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, embedding_dim) for n in cards])
        self.d = n_num + len(cards) * embedding_dim

    def forward(self, xn, xc):
        pieces = [emb(xc[:, j]) for j, emb in enumerate(self.embeddings)]
        pieces.append(torch.nan_to_num(xn))  # standardized training mean = 0
        return torch.cat(pieces, dim=-1)


def combine_paths(x0, crossed, deep, layout):
    """Fig. 1. Return the representation presented to the output head."""
    if layout == 'stacked':
        return deep(crossed)
    if layout == 'parallel':
        return torch.cat([crossed, deep(x0)], dim=-1)
    if layout == 'mlp':
        return deep(x0)
    raise ValueError('layout must be stacked, parallel, or mlp')


class DCNv2(nn.Module):
    """Full supervised model; same embedding and MLP defaults across local arms."""
    def __init__(self, n_num, cards, embedding_dim=4, depth=2,
                 hidden=(32, 32), kind='dense', rank=4, experts=2, layout='parallel'):
        super().__init__()
        self.layout = layout
        self.embedding = RowEmbedding(n_num, cards, embedding_dim)
        d = self.embedding.d
        self.crosses = nn.ModuleList([] if layout == 'mlp' else
                                    [CrossLayer(d, kind, min(rank, d), experts) for _ in range(depth)])
        sizes = [d, *hidden]
        self.deep = nn.Sequential(*[layer for a, b in zip(sizes[:-1], sizes[1:])
                                    for layer in (nn.Linear(a, b), nn.ReLU())])
        self.head = nn.Linear(sizes[-1] + (d if layout == 'parallel' else 0), 1)
        for layer in [*self.deep, self.head]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, xn, xc):
        x0 = self.embedding(xn, xc)
        x = x0
        for layer in self.crosses:
            x = layer(x0, x)  # original x0 is anchored across ALL layers
        return self.head(combine_paths(x0, x, self.deep, self.layout)).squeeze(-1)


@torch.no_grad()
def predict_dcn(model, xn, xc, batch_size=1024):
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(xn), batch_size):
        a = torch.as_tensor(xn[start:start+batch_size], dtype=torch.float32, device=device)
        b = torch.as_tensor(xc[start:start+batch_size], dtype=torch.long, device=device)
        chunks.append(model(a, b).sigmoid().cpu().numpy())
    return np.concatenate(chunks)


def train_dcn(model, xn, xc, y, train, valid, seed=0, epochs=20,
              batch_size=64, lr=.001, weight_decay=0., device='cpu', ema_decay=None):
    """Visible train loop. Select on VALIDATION log loss; no test labels here.

    Local default: Adam, fixed epochs, no EMA. The MovieLens track uses optional
    parameter EMA (0.9999): optimization updates raw weights, validation selects
    EMA snapshots. A completed run returns the selected inference weights.
    """
    from sklearn.metrics import log_loss
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    numeric = torch.as_tensor(xn, dtype=torch.float32, device=device)
    category = torch.as_tensor(xc, dtype=torch.long, device=device)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = copy.deepcopy(model) if ema_decay is not None else None
    best_loss, best, history = float('inf'), None, []
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(train)
        total = 0.
        for start in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[start:start+batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(numeric[ids], category[ids]), target[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; inspect scaling and cross activations.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            if ema is not None:
                with torch.no_grad():
                    for shadow, parameter in zip(ema.parameters(), model.parameters()):
                        shadow.mul_(ema_decay).add_(parameter, alpha=1-ema_decay)
        inference = model if ema is None else ema
        p = predict_dcn(inference, xn[valid], xc[valid])
        score = float(log_loss(y[valid], p, labels=[0, 1]))
        history.append({'epoch': epoch, 'train_loss': total/len(train), 'valid_loss': score})
        if score < best_loss:
            best_loss = score
            best = {k: v.detach().cpu().clone() for k, v in inference.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    return model, history
