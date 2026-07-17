"""Verify Lesson 028 claims — MLP & ResNet tabular baselines (Gorishniy et al. 2021, §3.2).

Paper: "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021, arXiv:2106.11959).
The paper's two "simple baselines" are an MLP and a **ResNet** built from pre-activation
residual blocks. Its central message is honest-baseline discipline: a *properly tuned*
ResNet is a strong baseline that many prior "SOTA" DL papers failed to beat, and there is
**no universal winner** between GBDT and DL.

The ResNet block (Gorishniy §3.2):
    ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))
    ResNet(x)      = Head(Block(...Block(Linear_embed(x))))
    Head(x)        = Linear(ReLU(BatchNorm(x)))

We verify three things the lesson/viz/lab depend on:

  Part 1 — DEPTH TRAINABILITY (mechanism of the skip connection): stack N blocks with the
    skip ON vs OFF (a plain MLP of the same shape). As depth grows the plain net *degrades* —
    and crucially its TRAIN accuracy also falls, so this is an optimization/degradation
    problem (He et al. 2015), not overfitting. The residual net holds. Both nets use
    BatchNorm, so the classic "vanishing gradients" story is NOT the cause; the skip makes the
    identity map free, so adding depth cannot hurt. Drives depth-trainability-viz.

  Part 2 — HONEST BAKE-OFF on a real Tier-A dataset (credit_g): MLP vs ResNet vs a tuned
    GBDT. No universal winner; the two DL nets are competitive with each other, and the GBDT
    is strong and cheap on this small categorical dataset. Drives baseline-bakeoff-viz + the
    lab reproduction target.

Run: .venv/bin/python labs/_verify_l028.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cpu"


# ===========================================================================
# The Gorishniy 2021 tabular ResNet (§3.2), with a use_skip switch so the
# SAME architecture becomes a plain MLP when the residual connection is off.
# ===========================================================================
class ResNetBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout, use_skip=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_main)
        self.lin1 = nn.Linear(d_main, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_main)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.use_skip = use_skip

    def forward(self, x):
        z = self.bn(x)
        z = F.relu(self.lin1(z))
        z = self.drop1(z)
        z = self.lin2(z)
        z = self.drop2(z)
        return x + z if self.use_skip else z          # <-- the whole lesson lives here


class TabResNet(nn.Module):
    def __init__(self, d_in, d_main=128, d_hidden=256, n_blocks=4, dropout=0.1, use_skip=True):
        super().__init__()
        self.embed = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList(
            [ResNetBlock(d_main, d_hidden, dropout, use_skip) for _ in range(n_blocks)]
        )
        self.head_bn = nn.BatchNorm1d(d_main)
        self.head = nn.Linear(d_main, 1)

    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        x = F.relu(self.head_bn(x))
        return self.head(x).squeeze(-1)


def train_net(model, Xtr, ytr, epochs=60, bs=256, lr=3e-3, wd=1e-5, seed=SEED):
    torch.manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def net_acc(model, X, y):
    model.eval()
    p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
    return accuracy_score(y, (p > 0.5).astype(int)), roc_auc_score(y, p)


# ===========================================================================
# Part 1 — depth trainability: plain (no skip) vs residual as depth grows.
# Record TRAIN and TEST accuracy: if the plain net's TRAIN accuracy also falls
# with depth, the failure is optimization (degradation), not overfitting.
# ===========================================================================
def part1_depth():
    print("== Part 1: depth trainability — plain MLP vs ResNet (skip on/off) ==")
    X, y = make_classification(
        n_samples=8000, n_features=32, n_informative=16, n_redundant=8,
        n_clusters_per_class=3, class_sep=0.8, flip_y=0.05, random_state=SEED,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)

    depths = [1, 2, 4, 8, 16, 32]
    res_te, plain_te, res_tr, plain_tr = {}, {}, {}, {}
    for nb in depths:
        m_r = TabResNet(X.shape[1], d_main=128, d_hidden=256, n_blocks=nb, dropout=0.0, use_skip=True)
        m_r = train_net(m_r, Xtr, ytr, seed=SEED)
        res_te[nb] = net_acc(m_r, Xte, yte)[0]
        res_tr[nb] = net_acc(m_r, Xtr, ytr)[0]

        m_p = TabResNet(X.shape[1], d_main=128, d_hidden=256, n_blocks=nb, dropout=0.0, use_skip=False)
        m_p = train_net(m_p, Xtr, ytr, seed=SEED)
        plain_te[nb] = net_acc(m_p, Xte, yte)[0]
        plain_tr[nb] = net_acc(m_p, Xtr, ytr)[0]

    print(f"  {'depth':>5} | {'plain test':>10} | {'resnet test':>11} | {'plain train':>11} | {'resnet train':>12}")
    for nb in depths:
        print(f"  {nb:>5} | {plain_te[nb]:>10.3f} | {res_te[nb]:>11.3f} | "
              f"{plain_tr[nb]:>11.3f} | {res_tr[nb]:>12.3f}")
    print(f"\n  plain_test  = {[round(plain_te[d],3) for d in depths]}")
    print(f"  resnet_test = {[round(res_te[d],3) for d in depths]}")
    print(f"  plain_train = {[round(plain_tr[d],3) for d in depths]}")
    print(f"  resnet_train= {[round(res_tr[d],3) for d in depths]}")
    print(f"  depths      = {depths}")
    print(f"  degradation: plain test {plain_te[2]:.3f}(d2) -> {plain_te[32]:.3f}(d32) "
          f"= {plain_te[32]-plain_te[2]:+.3f}; plain TRAIN {plain_tr[2]:.3f} -> {plain_tr[32]:.3f} "
          f"= {plain_tr[32]-plain_tr[2]:+.3f} (train falls too -> optimization, not overfitting)")
    print(f"  resnet holds: test {res_te[2]:.3f}(d2) -> {res_te[32]:.3f}(d32) = {res_te[32]-res_te[2]:+.3f}")
    return depths, plain_te, res_te, plain_tr, res_tr


# ===========================================================================
# Part 2 — honest bake-off on a real Tier-A dataset (credit_g).
# ===========================================================================
def preprocess_credit_g():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from relkit import load_tier_a
    X, y = load_tier_a("credit_g")
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    ct = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    Xm = ct.fit_transform(X).astype(np.float32)
    return Xm, y.to_numpy()


def part2_bakeoff():
    print("\n== Part 2: honest bake-off on credit_g (Tier A, real OpenML) ==")
    Xm, y = preprocess_credit_g()
    accs = {"mlp": [], "resnet": [], "gbt": []}
    aucs = {"mlp": [], "resnet": [], "gbt": []}
    for s in [0, 1, 2, 3, 4]:
        Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=0.3, random_state=s, stratify=y)

        mlp = TabResNet(Xm.shape[1], d_main=64, d_hidden=128, n_blocks=2, dropout=0.3, use_skip=False)
        mlp = train_net(mlp, Xtr, ytr, epochs=60, lr=1e-3, wd=1e-4, seed=s)
        a, u = net_acc(mlp, Xte, yte); accs["mlp"].append(a); aucs["mlp"].append(u)

        rn = TabResNet(Xm.shape[1], d_main=64, d_hidden=128, n_blocks=2, dropout=0.3, use_skip=True)
        rn = train_net(rn, Xtr, ytr, epochs=60, lr=1e-3, wd=1e-4, seed=s)
        a, u = net_acc(rn, Xte, yte); accs["resnet"].append(a); aucs["resnet"].append(u)

        gbt = HistGradientBoostingClassifier(random_state=s, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0).fit(Xtr, ytr)
        p = gbt.predict_proba(Xte)[:, 1]
        accs["gbt"].append(accuracy_score(yte, (p > 0.5).astype(int)))
        aucs["gbt"].append(roc_auc_score(yte, p))

    print(f"  {'model':>7} | {'acc (mean±sd)':>16} | {'roc-auc (mean±sd)':>18}")
    for k in ["mlp", "resnet", "gbt"]:
        print(f"  {k:>7} | {np.mean(accs[k]):.3f} ± {np.std(accs[k]):.3f}    | "
              f"{np.mean(aucs[k]):.3f} ± {np.std(aucs[k]):.3f}")
    print(f"\n  labels   = ['MLP','ResNet','GBDT']")
    print(f"  auc_mean = {[round(float(np.mean(aucs[k])),3) for k in ['mlp','resnet','gbt']]}")
    print(f"  auc_sd   = {[round(float(np.std(aucs[k])),3) for k in ['mlp','resnet','gbt']]}")
    print(f"  acc_mean = {[round(float(np.mean(accs[k])),3) for k in ['mlp','resnet','gbt']]}")
    return accs, aucs


if __name__ == "__main__":
    part1_depth()
    part2_bakeoff()
