"""From-scratch FT-Transformer (Gorishniy, Rubachev, Khrulkov & Babenko 2021, arXiv:2106.11959) — Lesson 046.

Built from the paper (Fig. 2 + §3.3), not from a library (NOTES standards #18/#22/#24). rtdl's own
`FTTransformer` and torch's `nn.MultiheadAttention` / `F.scaled_dot_product_attention` are VALIDATION
points only (`labs/_check_l046.py`, `labs/_verify_l046.py`) — never imported here to *build* the model.

FT-Transformer = **Feature Tokenizer + Transformer**. The one idea that separates it from TabTransformer
(L045): *every* feature becomes a token — **numeric features too** — so numbers finally attend. A learned
`[CLS]` token is appended; after the Transformer stack its final embedding is the row representation the
head reads. This removes TabTransformer's numeric-bypass limitation (L045's headline gap).

Paper map — every piece cites the element it realises (Gorishniy 2021, §3.3 "FT-Transformer",
Fig. 2 left = Feature Tokenizer, Fig. 2 right = Transformer + [CLS] readout):

| Paper                                                                     | Here                    |
|---------------------------------------------------------------------------|-------------------------|
| numeric feature j -> token  T_j = b_j + x_j · W_j   (W_j, b_j in R^d)      | `FeatureTokenizer.num_*`|
| categorical feature j -> token  T_j = b_j + e_j[x_j]  (per-column embed)   | `FeatureTokenizer.cat_*`|
| stack the k feature tokens -> [B, k, d]                                    | `FeatureTokenizer.forward`|
| prepend a learned [CLS] token -> [B, k+1, d]                              | `FTTransformer.cls`     |
| L Transformer layers, **PreNorm** (LN before each sub-layer)              | `FTTransformerBlock`    |
| readout: take the final [CLS] token -> LN -> ReLU -> Linear -> 1 logit    | `FTTransformer.forward` |
| multi-head self-attention  softmax(QKᵀ/√d)V per head, concat, project      | reused `MultiHeadSelfAttention` |

Fidelity notes (stated so the student is not misled, NOTES #20): the paper uses a PreNorm block, skips the
very first LayerNorm (a detail that matters little at our scale), and uses a ReGLU FFN; we use a plain
GELU FFN. The load-bearing ideas — per-feature tokenization *including numerics*, the [CLS] readout, and
PreNorm depth — are reproduced exactly. Binary classification only (one logit). CPU is fine for Tier-A.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Reuse the from-scratch attention validated in L045 (machine-precision vs torch in _check_l045).
# This is not "importing a library model" — it is our own paper-built kernel promoted to relkit (#22).
from relkit.tabtransformer import MultiHeadSelfAttention, frame_categorical  # noqa: F401 (re-exported)


# ---------------------------------------------------------------- the two load-bearing tokenizer ops
# These are module-level so the lab notebook can inline the rest of this file while KEEPING the student's
# own versions (the lab writes exactly these two functions; standard #25 — the paper's code is visible AND
# the student's code is the one that runs).
def affine_numeric_tokens(x_num, weight, bias):
    """The numeric half of the Feature Tokenizer (Gorishniy §3.3): token_j = b_j + x_j · W_j.

    x_num: [B, n_num] float · weight, bias: [n_num, d] · returns [B, n_num, d].
    Broadcasting [B, n_num, 1] * [n_num, d] gives each numeric feature its own learned direction W_j; the
    token is AFFINE in the scalar x_j (this is what lets a number attend, unlike TabTransformer).
    """
    return x_num.unsqueeze(-1) * weight + bias


def prepend_cls(tokens, cls):
    """Prepend the learned [CLS] token at position 0 (Gorishniy §3.3, the BERT readout trick).

    tokens: [B, k, d] · cls: [1, 1, d] · returns [B, k+1, d]. [CLS] carries no feature; the Transformer
    pools the row into it, and the head reads it out.
    """
    B = tokens.shape[0]
    return torch.cat([cls.expand(B, 1, cls.shape[-1]), tokens], dim=1)


# ---------------------------------------------------------------- Feature Tokenizer (Gorishniy §3.3, Fig. 2 left)
class FeatureTokenizer(nn.Module):
    """Turn every feature — numeric AND categorical — into a d-dim token (Gorishniy 2021, §3.3).

    numeric  T_j = b_j + x_j · W_j         (W_j, b_j learned d-vectors; the token is AFFINE in x_j)
    category T_j = b_j + e_j[x_j]          (per-column embedding table + a per-column bias)

    Output: [B, k, d] with k = n_num + n_cat tokens. This is the whole difference from TabTransformer,
    which never tokenises numerics — here a numeric feature is a first-class token that attends and is
    attended to.
    """

    def __init__(self, n_num: int, cards, d: int):
        super().__init__()
        self.n_num, self.cards, self.d = n_num, list(cards), d
        self.n_cat = len(self.cards)
        # numeric: one learned weight + bias vector per numeric feature (element-wise linear -> d dims)
        self.num_weight = nn.Parameter(torch.empty(n_num, d)) if n_num > 0 else None
        self.num_bias = nn.Parameter(torch.empty(n_num, d)) if n_num > 0 else None
        # categorical: an embedding table + a learned bias per categorical feature
        self.cat_embs = nn.ModuleList([nn.Embedding(max(c, 1), d) for c in self.cards])
        self.cat_bias = nn.Parameter(torch.empty(self.n_cat, d)) if self.n_cat > 0 else None
        self.reset_parameters()

    def reset_parameters(self):
        # Small uniform init (the paper's Kaiming-uniform-style start; the exact scheme is not load-bearing).
        bound = 1.0 / (self.d ** 0.5)
        if self.num_weight is not None:
            nn.init.uniform_(self.num_weight, -bound, bound)
            nn.init.uniform_(self.num_bias, -bound, bound)
        for emb in self.cat_embs:
            nn.init.uniform_(emb.weight, -bound, bound)
        if self.cat_bias is not None:
            nn.init.uniform_(self.cat_bias, -bound, bound)

    def forward(self, x_num, x_cat):
        """x_num: [B, n_num] float, x_cat: [B, n_cat] long. Returns tokens [B, k, d]."""
        B = x_num.shape[0] if self.n_num > 0 else x_cat.shape[0]
        tokens = []
        if self.n_num > 0:
            # broadcast: [B, n_num, 1] * [n_num, d] -> [B, n_num, d], then add the per-feature bias
            num_tok = affine_numeric_tokens(x_num, self.num_weight, self.num_bias)
            tokens.append(num_tok)
        if self.n_cat > 0:
            cat_tok = torch.stack(
                [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)], dim=1
            ) + self.cat_bias
            tokens.append(cat_tok)
        if not tokens:
            return torch.zeros(B, 0, self.d, device=x_cat.device)
        return torch.cat(tokens, dim=1)                        # [B, k, d]


# ---------------------------------------------------------------- Transformer block (PreNorm, Gorishniy §3.3)
class FTTransformerBlock(nn.Module):
    """One PreNorm Transformer layer (Gorishniy 2021 uses PreNorm, unlike TabTransformer's PostNorm):
    LayerNorm BEFORE each sub-layer, residual AROUND it. PreNorm keeps deeper stacks trainable — the
    residual carries the identity path, each sub-layer only adds a correction (the L028 skip idea)."""

    def __init__(self, d, n_heads=8, ff_hidden=None, dropout=0.1):
        super().__init__()
        ff_hidden = ff_hidden or int(d * 4 / 3)              # paper's ~4/3 ratio (ReGLU); we use GELU
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, n_heads)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff_hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff_hidden, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, w = self.attn(self.norm1(x))                      # PreNorm: normalise, then attend
        x = x + self.drop(a)                                 # residual around attention
        f = self.ffn(self.norm2(x))
        x = x + self.drop(f)                                 # residual around the FFN
        return x, w


# ---------------------------------------------------------------- FT-Transformer (Gorishniy §3.3, Fig. 2)
class FTTransformer(nn.Module):
    """FT-Transformer = Feature Tokenizer + [CLS] + Transformer + [CLS]-readout head (Gorishniy 2021).

    Flow: tokenize every feature (numeric + categorical) -> prepend a learned [CLS] token -> L PreNorm
    Transformer blocks -> read the FINAL [CLS] token -> LayerNorm -> ReLU -> Linear -> 1 logit.

    Because numerics are tokens, a numeric feature both attends and is attended to — the exact capability
    TabTransformer (L045) lacked. n_layers=0 degenerates to "just the [CLS] readout of the raw tokens"
    (no attention), a useful ablation but NOT a standard baseline.
    """

    def __init__(self, n_num, cards, d=64, n_layers=3, n_heads=8, ff_hidden=None,
                 head_hidden=None, dropout=0.1):
        super().__init__()
        self.n_num, self.cards, self.d, self.n_layers = n_num, list(cards), d, n_layers
        self.k = n_num + len(self.cards)
        self.tokenizer = FeatureTokenizer(n_num, cards, d)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.uniform_(self.cls, -1.0 / (d ** 0.5), 1.0 / (d ** 0.5))
        self.blocks = nn.ModuleList([FTTransformerBlock(d, n_heads, ff_hidden, dropout)
                                     for _ in range(n_layers)])
        self.head_norm = nn.LayerNorm(d)
        self.head_act = nn.ReLU()
        self.head = nn.Linear(d, 1)

    def tokens_with_cls(self, x_num, x_cat):
        """Feature tokens with the [CLS] token prepended: [B, k+1, d]. Position 0 is always [CLS]."""
        tok = self.tokenizer(x_num, x_cat)                   # [B, k, d]
        return prepend_cls(tok, self.cls)                    # [B, k+1, d], [CLS] at position 0

    def cls_readout(self, x_num, x_cat):
        """Run the Transformer and return the FINAL [CLS] embedding — the row representation (Fig. 2)."""
        h = self.tokens_with_cls(x_num, x_cat)
        for blk in self.blocks:
            h, _ = blk(h)
        return h[:, 0]                                        # the [CLS] token, position 0

    def forward(self, x_num, x_cat):
        z = self.cls_readout(x_num, x_cat)
        return self.head(self.head_act(self.head_norm(z))).squeeze(-1)


# ---------------------------------------------------------------- supervised training (fair protocol)
def _to_tensors(Xnum, Xcat, y=None, device="cpu"):
    xn = Xnum.to(device) if torch.is_tensor(Xnum) else torch.as_tensor(np.asarray(Xnum), dtype=torch.float32, device=device)
    xc = Xcat.to(device) if torch.is_tensor(Xcat) else torch.as_tensor(np.asarray(Xcat), dtype=torch.long, device=device)
    if y is None:
        return xn, xc
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    return xn, xc, yt


def train_ft_transformer(model, Xnum_tr, Xcat_tr, ytr, Xnum_va, Xcat_va, yva, *,
                         lr=1e-3, wd=1e-5, max_epochs=80, patience=12, batch_size=256,
                         device="cpu", seed=0):
    """Mini-batch AdamW with early stopping on validation ROC-AUC — the same fair, shared-protocol
    contract as `relkit.nets.train_net` (L042) and `relkit.tabtransformer.train_tabtransformer` (L045):
    every arm picks its own training length by validation, so none is under- or over-trained.

    Returns (model_with_best_val_weights, best_val_auc).
    """
    torch.manual_seed(seed)
    model = model.to(device)
    xn, xc, yt = _to_tensors(Xnum_tr, Xcat_tr, ytr, device)
    xnv, xcv = _to_tensors(Xnum_va, Xcat_va, device=device)
    n = xn.shape[0]
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
            loss = lossf(model(xn[idx], xc[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xnv, xcv)).cpu().numpy()
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
def ft_transformer_auc(model, Xnum, Xcat, y, *, device="cpu"):
    model.eval()
    xn, xc = _to_tensors(Xnum, Xcat, device=device)
    return roc_auc_score(y, torch.sigmoid(model(xn, xc)).cpu().numpy())
