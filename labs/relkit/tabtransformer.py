"""From-scratch TabTransformer (Huang, Khetan, Cvitkovic & Karnin 2020, arXiv:2012.06678) — Lesson 045.

Built from the paper (Fig. 1 + §3), not from a library (NOTES standards #18/#22/#24). This module
*promotes* the architecture the student wrote forward-only in Lab 032 and adds the two things L045 is
about: **supervised training** and **semi-supervised RTD pre-training**. torch's own
`nn.functional.scaled_dot_product_attention` and `nn.MultiheadAttention` are used ONLY as VALIDATION
points in `labs/_verify_l045.py` / `labs/_check_l045.py` — never imported here.

Paper map — every piece cites the element it realises (§3.1 "Column Embedding", §3.2 the Transformer
stack, Fig. 1 the whole model, §3.3 "Pre-training"):

| Paper                                                                | Here                          |
|----------------------------------------------------------------------|-------------------------------|
| column (entity) embedding for each categorical feature               | `TabTransformer.embs`         |
| N Transformer layers over the categorical tokens -> contextual embs  | `TabTransformer.contextual`   |
| continuous features bypass the Transformer, are LayerNorm'd          | `TabTransformer.num_norm`     |
| concat[flatten(contextual), norm(continuous)] -> MLP head -> logit   | `TabTransformer.forward`      |
| multi-head self-attention  softmax(QKᵀ/√d)V per head, concat, project | `MultiHeadSelfAttention`      |
| Transformer layer = residual(attn) + residual(FFN), each + LayerNorm  | `TransformerBlock`            |
| RTD pre-training: replace tokens, a per-column detector predicts them | `pretrain_rtd` + `RTDHead`    |

The n_layers=0 case is the **context-free ablation**: entity embeddings with NO attention — exactly the
static-embedding MLP of L031/L032. Running the same class at n_layers=0 vs n_layers>0 isolates *what the
attention (contextualisation) buys*, which is the lesson's headline comparison.

Binary classification only (one logit). CPU is fine for the small / subsampled Tier-A tables L045 uses.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- data prep (shared by verify/check/lab)
def frame_categorical(Xdf: pd.DataFrame):
    """Encode a mixed table into TabTransformer's inputs.

    Returns (Xcat: LongTensor [N, m], Xnum: FloatTensor [N, n_num], cards: list[int],
             cat_names: list[str], num_names: list[str]).

    Categoricals are integer-coded per column (the stage-1 token indices); continuous features are
    standardised. Unlike the one-hot frame the tree/MLP baselines use, TabTransformer wants the raw
    integer code so each category owns a learnable embedding vector.
    """
    num_names = Xdf.select_dtypes(include="number").columns.tolist()
    cat_names = [c for c in Xdf.columns if c not in num_names]
    codes, cards = [], []
    for c in cat_names:
        col = Xdf[c].map(str).astype(object)             # plain str objects (dodges category-dtype quirks)
        levels = sorted(col.unique())
        lut = {v: i for i, v in enumerate(levels)}
        codes.append(col.map(lut).to_numpy())
        cards.append(len(levels))
    Xcat = (torch.tensor(np.stack(codes, axis=1), dtype=torch.long)
            if cat_names else torch.zeros((len(Xdf), 0), dtype=torch.long))
    if num_names:
        Xn = Xdf[num_names].to_numpy(float)
        mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
        Xnum = torch.tensor((Xn - mu) / sd, dtype=torch.float32)
    else:
        Xnum = torch.zeros((len(Xdf), 0), dtype=torch.float32)
    return Xcat, Xnum, cards, cat_names, num_names


# ---------------------------------------------------------------- attention (Vaswani §3.2, Huang §3.2)
def scaled_dot_product_attention(Q, K, V):
    """Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V. Works on the last two dims (so multi-head just adds
    a head axis). Matches `torch.nn.functional.scaled_dot_product_attention` to ~1e-6 (see _check_l045)."""
    d = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (Vaswani §3.2): project x into `n_heads` independent Q/K/V subspaces,
    attend within each, concat, and project back. TabTransformer uses this over the categorical tokens
    so a column's new vector can blend in the columns it attends to."""

    def __init__(self, d, n_heads=4):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d, self.n_heads, self.dh = d, n_heads, d // n_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)

    def _split(self, t):
        B, m, _ = t.shape
        return t.view(B, m, self.n_heads, self.dh).transpose(1, 2)   # [B, h, m, dh]

    def forward(self, x):
        B, m, _ = x.shape
        q, k, v = self._split(self.Wq(x)), self._split(self.Wk(x)), self._split(self.Wv(x))
        out, w = scaled_dot_product_attention(q, k, v)               # [B, h, m, dh], [B, h, m, m]
        out = out.transpose(1, 2).reshape(B, m, self.d)              # concat heads
        return self.Wo(out), w.mean(dim=1)                          # avg heads for a legible (m,m) map


class TransformerBlock(nn.Module):
    """One Transformer layer (Huang Fig. 1): two residual sub-layers — multi-head self-attention, then a
    position-wise FFN — each wrapped in `LayerNorm(x + sublayer(x))`. The residual is the L028 skip idea:
    a sub-layer only learns a correction, so depth stays trainable."""

    def __init__(self, d, n_heads=4, ff_hidden=None, dropout=0.0):
        super().__init__()
        ff_hidden = ff_hidden or 4 * d
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d, ff_hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff_hidden, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, w = self.attn(x)
        x = self.norm1(x + self.drop(a))          # residual around attention
        f = self.ffn(x)
        x = self.norm2(x + self.drop(f))          # residual around the FFN
        return x, w


class TabTransformer(nn.Module):
    """TabTransformer (Huang 2020, Fig. 1). categorical -> embed -> N Transformer blocks (contextual) ->
    flatten ⊕ LayerNorm(continuous) -> MLP head -> 1 logit.

    n_layers = 0 is the **context-free ablation** (entity embeddings, no attention — the L031/L032 static
    embedding MLP). Only categoricals are contextualised; numerics bypass the Transformer entirely — the
    paper's known limitation that FT-Transformer (L046) removes by tokenising numerics too.
    """

    def __init__(self, cards, n_num, d=32, n_layers=3, n_heads=4, ff_hidden=None,
                 head_hidden=128, dropout=0.1):
        super().__init__()
        self.cards, self.n_num, self.d, self.n_layers = list(cards), n_num, d, n_layers
        self.m = len(cards)
        self.embs = nn.ModuleList([nn.Embedding(max(c, 1), d) for c in cards])
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads, ff_hidden, dropout)
                                     for _ in range(n_layers)])
        self.num_norm = nn.LayerNorm(n_num) if n_num > 0 else None
        feat_dim = self.m * d + n_num
        self.head = nn.Sequential(nn.Linear(feat_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, 1))

    def embed(self, x_cat):                                          # [B, m, d] context-free embeddings
        if self.m == 0:
            return torch.zeros(x_cat.shape[0], 0, self.d, device=x_cat.device)
        return torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)

    def contextual(self, x_cat):                                    # [B, m, d] after N Transformer blocks
        h = self.embed(x_cat)
        for blk in self.blocks:
            h, _ = blk(h)
        return h

    def features(self, x_cat, x_num):                               # the vector fed to the head/probe
        ctx_flat = self.contextual(x_cat).flatten(1)                # [B, m*d]
        if self.num_norm is not None and x_num.shape[1] > 0:
            return torch.cat([ctx_flat, self.num_norm(x_num)], dim=1)
        return ctx_flat

    def forward(self, x_cat, x_num):
        return self.head(self.features(x_cat, x_num)).squeeze(-1)


# ---------------------------------------------------------------- supervised training (fair protocol)
def _to_tensors(Xcat, Xnum, y=None, device="cpu"):
    xc = Xcat.to(device) if torch.is_tensor(Xcat) else torch.as_tensor(np.asarray(Xcat), dtype=torch.long, device=device)
    xn = Xnum.to(device) if torch.is_tensor(Xnum) else torch.as_tensor(np.asarray(Xnum), dtype=torch.float32, device=device)
    if y is None:
        return xc, xn
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    return xc, xn, yt


def train_tabtransformer(model, Xcat_tr, Xnum_tr, ytr, Xcat_va, Xnum_va, yva, *,
                         lr=1e-3, wd=1e-5, max_epochs=80, patience=12, batch_size=256,
                         device="cpu", seed=0):
    """Mini-batch AdamW with early stopping on validation ROC-AUC — the same fair, shared-protocol
    contract as `relkit.nets.train_net` (L042) and `relkit.node.train_node` (L044): every arm picks its
    own training length by validation, so none is accidentally under- or over-trained.

    Returns (model_with_best_val_weights, best_val_auc).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    xc, xn, yt = _to_tensors(Xcat_tr, Xnum_tr, ytr, device)
    xcv, xnv = _to_tensors(Xcat_va, Xnum_va, device=device)
    n = xc.shape[0]
    bs = min(batch_size, n)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, since = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(xc[idx], xn[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xcv, xnv)).cpu().numpy()
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
def tabtransformer_auc(model, Xcat, Xnum, y, *, device="cpu"):
    model.eval()
    xc, xn = _to_tensors(Xcat, Xnum, device=device)
    return roc_auc_score(y, torch.sigmoid(model(xc, xn)).cpu().numpy())


# ---------------------------------------------------------------- RTD self-supervised pre-training (§3.3)
def corrupt_categorical(Xcat, cards, p, generator):
    """Replaced-Token-Detection corruption (Huang §3.3, ELECTRA-style). Independently, with probability
    `p`, replace each categorical token with another category drawn UNIFORMLY from that column's range.
    Returns (Xcorrupt [N, m] long, replaced [N, m] float 0/1) — the label the detector must recover.

    No labels are used, so any unlabeled row provides training signal: the pretext task is "which of my
    columns were tampered with?", and solving it forces the encoder to learn what a *coherent* row looks
    like (which category values co-occur) — the representation that then transfers to the real task.
    """
    N, m = Xcat.shape
    replace_mask = torch.rand(N, m, generator=generator) < p
    rand_vals = torch.zeros(N, m, dtype=torch.long)
    for j, c in enumerate(cards):
        rand_vals[:, j] = torch.randint(0, max(c, 1), (N,), generator=generator)
    # If the random draw equals the original, it is NOT a replacement — relabel those as unchanged.
    Xcorrupt = torch.where(replace_mask, rand_vals, Xcat)
    replaced = (Xcorrupt != Xcat).float()
    return Xcorrupt, replaced


class RTDHead(nn.Module):
    """A per-column binary detector (Huang §3.3): from each token's CONTEXTUAL embedding, predict whether
    that token was replaced. One small linear per column (a column's own detector), applied to the
    contextual vector the Transformer produced for it."""

    def __init__(self, m, d):
        super().__init__()
        self.det = nn.ModuleList([nn.Linear(d, 1) for _ in range(m)])

    def forward(self, ctx):                                          # ctx: [B, m, d]
        logits = [self.det[j](ctx[:, j]) for j in range(len(self.det))]
        return torch.cat(logits, dim=1)                             # [B, m]


def pretrain_rtd(model, Xcat_unlab, cards, *, replace_p=0.30, lr=1e-3, wd=1e-5,
                 max_epochs=60, batch_size=256, device="cpu", seed=0, verbose=False):
    """Self-supervised RTD pre-training of the TabTransformer ENCODER on UNLABELED rows (Huang §3.3).

    Each step: corrupt a batch's categorical tokens, run them through embeddings + Transformer blocks,
    and train a per-column detector (`RTDHead`) to flag the replaced tokens (BCE). Only the encoder
    (`embs` + `blocks`) and the throwaway RTD head are updated; the supervised head is untouched and is
    (re)trained later in `train_tabtransformer`.

    Returns (model, final_detector_accuracy). The model's embeddings + blocks now carry a representation
    learned from unlabeled data — the label-efficiency lever trees lack.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    xc = Xcat_unlab.to(device) if torch.is_tensor(Xcat_unlab) else torch.as_tensor(
        np.asarray(Xcat_unlab), dtype=torch.long, device=device)
    m = xc.shape[1]
    head = RTDHead(m, model.d).to(device)
    params = list(model.embs.parameters()) + list(model.blocks.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    n = xc.shape[0]
    bs = min(batch_size, n)
    acc = 0.0
    for ep in range(max_epochs):
        model.train(); head.train()
        perm = torch.randperm(n, generator=gen)
        correct = total = 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            xcorr, replaced = corrupt_categorical(xc[idx].cpu(), cards, replace_p, gen)
            xcorr, replaced = xcorr.to(device), replaced.to(device)
            opt.zero_grad()
            ctx = model.contextual(xcorr)
            logits = head(ctx)
            loss = lossf(logits, replaced)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = (logits > 0).float()
                correct += (pred == replaced).sum().item()
                total += replaced.numel()
        acc = correct / max(total, 1)
        if verbose and (ep % 10 == 0 or ep == max_epochs - 1):
            print(f"  [pretrain] epoch {ep:2d}  detector acc {acc:.3f}")
    return model, acc
