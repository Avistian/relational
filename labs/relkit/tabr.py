"""Numeric TabR-S: Eq. 5, linear encoder, one residual predictor block.

Paper: https://arxiv.org/abs/2307.14338v2, Figs. 2–4, §3.2.
Exact search uses torch instead of Faiss. Selection is discrete; selected
distances and values are recomputed with gradients. No numerical embeddings.
"""
import math
import torch
from torch import nn


def eligible_mask(query_ids, candidate_ids):
    """Identity, not feature equality, determines self-exclusion."""
    return query_ids[:, None] != candidate_ids[None, :]


def select_neighbors(keys, candidate_keys, m, allowed):
    """Return [batch, m] legal nearest indices, without differentiating top-k."""
    if m < 1 or m > candidate_keys.shape[0] or (allowed.sum(1) < m).any():
        raise ValueError('Every query needs at least m eligible candidates')
    with torch.no_grad():
        distances = (keys.square().sum(1, keepdim=True)
                     - 2 * keys @ candidate_keys.T
                     + candidate_keys.square().sum(1)[None, :])
        distances = distances.masked_fill(~allowed, torch.inf)
        return distances.topk(m, largest=False).indices


def context_value(keys, neighbor_keys, label_embeddings, correction):
    """Eq. 5: label contribution plus query-minus-neighbor correction."""
    return label_embeddings + correction(keys[:, None, :] - neighbor_keys)


def aggregate_context(keys, neighbor_keys, values, dropout):
    """Softmax over neighbors; no 1/sqrt(d). Dropout is NOT renormalized."""
    scores = -(keys[:, None, :] - neighbor_keys).square().sum(-1)
    weights = dropout(scores.softmax(dim=1))
    return (weights[:, :, None] * values).sum(1)


class TabRS(nn.Module):
    def __init__(self, n_features, d=32, m=16, multiplier=2,
                 dropout=0.1, context_dropout=0.1, regression=False,
                 retrieval=True):
        super().__init__()
        self.m, self.regression, self.retrieval = m, regression, retrieval
        h = int(d * multiplier)
        self.linear = nn.Linear(n_features, d)
        self.K = nn.Linear(d, d)
        self.label_encoder = nn.Linear(1, d) if regression else nn.Embedding(2, d)
        self.T = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(h, d, bias=False))
        self.context_dropout = nn.Dropout(context_dropout)
        self.block = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(h, d), nn.Dropout(0.0))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
        if regression:
            nn.init.uniform_(self.label_encoder.weight, -1/math.sqrt(2), 1/math.sqrt(2))
            nn.init.uniform_(self.label_encoder.bias, -1/math.sqrt(2), 1/math.sqrt(2))
        else:
            nn.init.uniform_(self.label_encoder.weight, -1, 1)

    def forward(self, x, candidate_x, candidate_y, query_ids=None, candidate_ids=None):
        representation = self.linear(x)
        if self.retrieval:
            keys = self.K(representation)
            candidate_keys = self.K(self.linear(candidate_x))
            allowed = torch.ones((len(x), len(candidate_x)), dtype=torch.bool, device=x.device)
            if query_ids is not None:
                if candidate_ids is None:
                    raise ValueError('Candidate identities are required with query identities')
                allowed = eligible_mask(query_ids, candidate_ids)
            indices = select_neighbors(keys, candidate_keys, self.m, allowed)
            neighbor_keys = candidate_keys[indices]
            labels = candidate_y[indices]
            label_embeddings = self.label_encoder(labels[..., None] if self.regression else labels.long())
            values = context_value(keys, neighbor_keys, label_embeddings, self.T)
            representation = representation + aggregate_context(keys, neighbor_keys, values, self.context_dropout)
        representation = representation + self.block(representation)
        return self.head(representation).squeeze(-1)
