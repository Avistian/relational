"""L049. Released-order ExcelFormer; Trompt Eq.4/5 key parts.

ExcelFormer inputs MUST be ordered by decreasing TRAIN-only mutual information.
Implements the released numeric, pre-norm, uncompressed path; not all variants.
Trompt primitives mirror equations, not a claim of full-model reproduction.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spa_attention(q, k, v, dropout=0., training=False):
    """Queries are receivers; columns are senders, strongest feature first."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    blocked = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
    weights = torch.softmax(scores.masked_fill(blocked, -1e4), dim=-1)
    return F.dropout(weights, dropout, training) @ v, weights


def prompt_weights(fused_prompts, columns):
    """Trompt Eq.4: [B,P,d] @ [d,C] -> [B,P,C]; no sqrt(d) in paper."""
    return torch.softmax(fused_prompts @ columns.T, dim=-1)


def prompt_reduce(weights, expanded_features):
    """Trompt Eq.5: preserve batch/prompt/embedding, sum over columns."""
    return (weights[..., None] * expanded_features).sum(dim=2)


def feat_mix(tokens, y, importance, mask, permutation):
    """ExcelFormer Eq.7/8. mask=True retains that feature from original row."""
    total = importance.sum()
    # No-information case is undefined in Eq.8; explicit count-based fallback.
    ratio = ((mask * importance).sum(-1) / total if total > 0 else mask.float().mean(-1))
    mixed = torch.where(mask[..., None], tokens, tokens[permutation])
    return mixed, ratio*y + (1-ratio)*y[permutation]


class GatedTokenizer(nn.Module):
    def __init__(self, features, d):
        super().__init__()
        for name in ('weight', 'weight2', 'bias', 'bias2'):
            p = nn.Parameter(torch.empty(features, d))
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            setattr(self, name, p)

    def forward(self, x):
        a = x[..., None]*self.weight + self.bias
        b = x[..., None]*self.weight2 + self.bias2
        return a * torch.tanh(b)


class SPABlock(nn.Module):
    def __init__(self, d, heads, dropout, scale, first):
        super().__init__()
        assert d % heads == 0
        self.heads, self.dropout = heads, dropout
        self.q, self.k, self.v = [nn.Linear(d, d) for _ in range(3)]
        self.out = nn.Linear(d, d) if heads > 1 else nn.Identity()
        self.gate = nn.Linear(d, 2*d)
        self.norm0 = nn.Identity() if first else nn.LayerNorm(d)
        self.norm1 = nn.LayerNorm(d)
        for layer in (self.q, self.k, self.v, self.gate):
            with torch.no_grad(): layer.weight.mul_(scale)
            nn.init.zeros_(layer.bias)
        if heads > 1: nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, d = x.shape
        z = self.norm0(x)
        q, k, v = [layer(z).reshape(b,c,self.heads,d//self.heads).transpose(1,2)
                   for layer in (self.q,self.k,self.v)]
        update, _ = spa_attention(q,k,v,self.dropout,self.training)
        x = x + self.out(update.transpose(1,2).reshape(b,c,d))
        a, gate = self.gate(self.norm1(x)).chunk(2, dim=-1)
        return x + a * torch.tanh(gate)


class ExcelFormer(nn.Module):
    """Numeric, pre-norm released variant; logits for binary classification.

    Lab defaults shrink width/depth. Released implementation scales Q/K/V and
    GLU weights; the attention output map uses ordinary Kaiming initialization.
    """
    def __init__(self, features, d=32, heads=4, layers=2, dropout=.1, scale=.01):
        super().__init__()
        self.tokenizer = GatedTokenizer(features,d)
        self.blocks = nn.ModuleList([SPABlock(d,heads,dropout,scale,i==0) for i in range(layers)])
        self.pool = nn.Linear(features,1)
        self.norm = nn.LayerNorm(d)
        self.act = nn.PReLU()
        self.head = nn.Linear(d,1)

    def forward(self,x,mix=None):
        z = self.tokenizer(x)
        if mix is not None: z = mix(z)
        for block in self.blocks: z = block(z)
        z = self.pool(z.transpose(1,2)).squeeze(-1)
        return self.head(self.act(self.norm(z))).squeeze(-1)


def train_excel(model, x, y, train, valid, *, seed=0, epochs=30, lr=.001,
                batch=64, patience=12, augmentation=False, importance=None, device='cpu'):
    """Visible learning loop: select checkpoint using validation log loss only."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    xt = torch.as_tensor(x,dtype=torch.float32,device=device)
    yt = torch.as_tensor(y,dtype=torch.float32,device=device)
    imp = torch.as_tensor(importance,dtype=torch.float32,device=device) if importance is not None else None
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0)
    best, best_state, stale, history = float('inf'), None, 0, []
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(train), max(1,math.ceil(len(train)/batch))):
            target = yt[ids]
            mix = None
            if augmentation:
                keep = rng.beta(.5,.5)
                mask = torch.rand((len(ids),x.shape[1]),device=device) < keep
                perm = torch.randperm(len(ids),device=device)
                # Compute targets via the SAME primitive that mixes embeddings.
                _, target = feat_mix(xt[ids,...,None],target,imp,mask,perm)
                mix = lambda z: feat_mix(z,yt[ids],imp,mask,perm)[0]
            logits = model(xt[ids],mix=mix)
            loss = F.binary_cross_entropy_with_logits(logits,target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(F.binary_cross_entropy_with_logits(model(xt[valid]),yt[valid]))
        history.append({'epoch':epoch+1,'valid_log_loss':score})
        if score < best:
            best,best_state,stale = score,copy.deepcopy(model.state_dict()),0
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    return model,history


@torch.no_grad()
def predict_excel(model,x,batch=1024):
    device = next(model.parameters()).device
    return np.concatenate([torch.sigmoid(model(torch.as_tensor(a,dtype=torch.float32,device=device))).cpu().numpy()
                           for a in np.array_split(x,max(1,math.ceil(len(x)/batch)))])
