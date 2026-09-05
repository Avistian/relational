"""L047: supervised SAINT, written from scratch against the released code.

Paper: Somepalli et al., arXiv:2106.01342v1, §3 / Algorithm 1.
Reference: somepago/saint e288e84c77a54cfd2ffb55a53678fb7cbbb16630.
We deliberately match released PreNorm(Residual(F)): LN(x) + F(LN(x)),
NOT the paper's LN(F(x)) + x. Numeric embeddings follow released 1→100→d
MLPs. Attention dropout is absent in released forward; FF dropout is active.
The supervised path is complete; InfoNCE is an isolated §4 exercise, not a
claim that we reproduced the semi-supervised pretraining experiments.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


def pack_rows(tokens):
    """Algorithm 1: [B,T,d] → [1,B,T*d], including CLS; no column transpose."""
    b, t, d = tokens.shape
    return tokens.reshape(1, b, t * d)


def unpack_rows(rows, n_tokens):
    """Inverse reshape; preserve feature/embedding order."""
    return rows.reshape(rows.shape[1], n_tokens, rows.shape[-1] // n_tokens)


def attention_weights(q, k):
    """§3: normalize over KEYS, separately for every query and head."""
    return (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(dim=-1)


def info_nce(z, z_view, temperature=0.7):
    """§4 eq.5 contrastive term, mean instead of sum; no implicit normalization.
    The paired original/view is at the same row index. This is one-way eq.5;
    the released pretraining code also offers a symmetric normalized variant.
    """
    logits = z @ z_view.T / temperature
    return F.cross_entropy(logits, torch.arange(len(z), device=z.device))


class Attention(nn.Module):
    def __init__(self, width, heads=4, head_dim=16):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_qkv = nn.Linear(width, 3 * heads * head_dim, bias=False)
        self.to_out = nn.Linear(heads * head_dim, width)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [v.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
                   for v in self.to_qkv(x).chunk(3, dim=-1)]
        a = attention_weights(q, k)
        return self.to_out((a @ v).transpose(1, 2).reshape(b, t, -1))


class GEGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)


class ReleasedResidual(nn.Module):
    """Mirror the executable nesting, not a conventional PreNorm assumption."""
    def __init__(self, width, fn):
        super().__init__()
        self.norm, self.fn = nn.LayerNorm(width), fn

    def forward(self, x):
        u = self.norm(x)
        return u + self.fn(u)


def feedforward(width, dropout):
    return nn.Sequential(nn.Linear(width, 8 * width), GEGLU(),
                         nn.Dropout(dropout), nn.Linear(4 * width, width))


class SaintStage(nn.Module):
    """Fig.1a: feature attention/FF, then entire-row attention/FF."""
    def __init__(self, tokens, d, heads=4, ff_dropout=0.1, variant='colrow'):
        super().__init__()
        if variant not in ('col', 'row', 'colrow'):
            raise ValueError(variant)
        self.variant = variant
        self.col = nn.ModuleList([
            ReleasedResidual(d, Attention(d, heads, 16)),
            ReleasedResidual(d, feedforward(d, ff_dropout)),
        ]) if variant != 'row' else nn.ModuleList()
        width = tokens * d
        self.row = nn.ModuleList([
            ReleasedResidual(width, Attention(width, heads, 64)),
            ReleasedResidual(width, feedforward(width, ff_dropout)),
        ]) if variant != 'col' else nn.ModuleList()

    def forward(self, x):
        for layer in self.col:
            x = layer(x)
        if self.row:
            t = x.shape[1]
            x = pack_rows(x)
            for layer in self.row:
                x = layer(x)
            x = unpack_rows(x, t)
        return x


class SAINT(nn.Module):
    """Released supervised path: CLS, cats, numeric MLPs, blocks, CLS→1000→2.
    Missing numeric inputs are NaN and get feature-specific learned mask tokens;
    categorical code 0 is the missing/unseen token reserved by our train-only encoder.
    """
    def __init__(self, n_num, cards, d=8, depth=1, heads=4,
                 ff_dropout=0.1, variant='colrow'):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        self.cats = nn.ModuleList([nn.Embedding(c, d) for c in cards])
        self.nums = nn.ModuleList([nn.Sequential(nn.Linear(1, 100), nn.ReLU(),
                                                nn.Linear(100, d)) for _ in range(n_num)])
        self.missing_num = nn.Parameter(torch.randn(n_num, d))
        t = 1 + n_num + len(cards)
        self.stages = nn.ModuleList([SaintStage(t, d, heads, ff_dropout, variant)
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(d, 1000), nn.ReLU(), nn.Linear(1000, 2))

    def tokenize(self, x_num, x_cat):
        b = len(x_num)
        parts = [self.cls.expand(b, -1, -1)]
        parts += [emb(x_cat[:, j]).unsqueeze(1) for j, emb in enumerate(self.cats)]
        for j, mlp in enumerate(self.nums):
            value = x_num[:, j:j+1]
            embedded = mlp(torch.nan_to_num(value))
            parts.append(torch.where(torch.isnan(value), self.missing_num[j], embedded).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def encode(self, x_num, x_cat):
        x = self.tokenize(x_num, x_cat)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x_num, x_cat):
        return self.head(self.encode(x_num, x_cat)[:, 0])


@torch.no_grad()
def predict_saint(model, xn, xc, batch_size=64, device='cpu'):
    """Within-split sequential batches. Their order/membership are model inputs.
    Never mix validation and test rows, and never pass labels into forward.
    """
    model.eval()
    out = []
    for start in range(0, len(xn), batch_size):
        num = torch.as_tensor(xn[start:start+batch_size], dtype=torch.float32, device=device)
        cat = torch.as_tensor(xc[start:start+batch_size], dtype=torch.long, device=device)
        out.append(model(num, cat).softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_saint(model, xn, xc, y, train, valid, *, seed=0, epochs=20,
                batch_size=64, lr=1e-3, device='cpu', checkpoint=None,
                select_metric='auc', validate_every=1):
    """Supervised AdamW, validation-only selection, checkpoint best weights.
    Call torch.manual_seed BEFORE constructing the model as well as here.
    No early stopping: a fixed budget makes the local ablation interpretable.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num = torch.as_tensor(xn, dtype=torch.float32, device=device)
    cat = torch.as_tensor(xc, dtype=torch.long, device=device)
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    best, best_score, history = None, -float('inf'), []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(num[idx], cat[idx]), labels[idx])
            loss.backward()
            optimizer.step()
        if epoch % validate_every == 0:
            p = predict_saint(model, xn[valid], xc[valid], batch_size, device)
            score = (float(np.mean((p >= .5) == y[valid])) if select_metric == 'accuracy'
                     else float(roc_auc_score(y[valid], p)))
            history.append({'epoch': epoch + 1, 'valid_score': score})
            if score > best_score:
                best_score, best = score, copy.deepcopy(model.state_dict())
                if checkpoint:
                    torch.save({'weights': best, 'epoch': epoch + 1, 'seed': seed,
                                'valid_score': score}, checkpoint)
    model.load_state_dict(best)
    return model, history
