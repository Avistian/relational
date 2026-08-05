"""From-scratch tabular neural baselines (promoted from Lab 028) + a fair training harness.

These are the student's own implementations — an MLP and a pre-activation ResNet (Gorishniy 2021 §3.2 /
He 2015) — kept here so later labs BUILD ON them instead of importing a black-box library. A reference
library (rtdl) is used only to *validate* these (see `labs/_verify_l042.py`), per NOTES standard #22.

Requires torch. CPU is fine for the small Tier-A tables used in the labs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class ResNetBlock(nn.Module):
    """Pre-activation residual block: out = x + f(x) with the skip, else f(x).

    f = BatchNorm -> Linear -> ReLU -> Dropout -> Linear -> Dropout  (Gorishniy 2021 §3.2).
    The skip makes the identity map free, so depth stops degrading (He 2015).
    """

    def __init__(self, d_main, d_hidden, dropout1=0.0, dropout2=0.0, use_skip=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_main)
        self.lin1 = nn.Linear(d_main, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_main)
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        self.use_skip = use_skip

    def forward(self, x):
        z = self.drop2(self.lin2(self.drop1(F.relu(self.lin1(self.bn(x))))))
        return x + z if self.use_skip else z


class TabResNet(nn.Module):
    """From-scratch ResNet: embed -> N residual blocks -> BatchNorm/ReLU head -> 1 logit.

    Set use_skip=False to get the plain-MLP-family ablation (same params, no residual).
    """

    def __init__(self, d_in, d_main=128, d_hidden=256, n_blocks=2, dropout1=0.0, dropout2=0.0, use_skip=True):
        super().__init__()
        self.embed = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList([
            ResNetBlock(d_main, d_hidden, dropout1, dropout2, use_skip) for _ in range(n_blocks)
        ])
        self.head_bn = nn.BatchNorm1d(d_main)
        self.head = nn.Linear(d_main, 1)

    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(F.relu(self.head_bn(x))).squeeze(-1)


class TabMLP(nn.Module):
    """From-scratch MLP: a stack of Dropout(ReLU(Linear)) blocks -> 1 logit (Gorishniy 2021 §3.2)."""

    def __init__(self, d_in, d_block=128, n_blocks=2, dropout=0.0):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_blocks):
            layers += [nn.Linear(d, d_block), nn.ReLU(), nn.Dropout(dropout)]
            d = d_block
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def train_net(model, Xtr, ytr, Xva, yva, *, lr=2e-3, wd=1e-5, max_epochs=200, patience=16,
              device="cpu", seed=0):
    """Full-batch AdamW training with early stopping on validation ROC-AUC (small tables).

    Returns (model_with_best_val_weights, best_val_auc). This is the fair, shared-protocol recipe:
    every model picks its own training length by validation, none is under- or over-trained by accident.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(np.asarray(Xtr), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.float32, device=device)
    Xv = torch.tensor(np.asarray(Xva), dtype=torch.float32, device=device)
    best_auc, best_state, since = -1.0, None, 0
    def _logits(m, x):
        out = m(x)
        return out.squeeze(-1) if out.ndim > 1 else out   # accept rtdl's (batch, 1) too

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(_logits(model, Xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(_logits(model, Xv)).cpu().numpy()
        val_auc = roc_auc_score(yva, pv)
        if val_auc > best_auc:
            best_auc, best_state, since = val_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def net_auc(model, X, y, *, device="cpu"):
    model.eval()
    out = model(torch.tensor(np.asarray(X), dtype=torch.float32, device=device))
    if out.ndim > 1:
        out = out.squeeze(-1)
    return roc_auc_score(y, torch.sigmoid(out).cpu().numpy())
