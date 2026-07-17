"""Build Lab 028 (MLP & ResNet tabular baselines) — emits the blank student notebook and a filled
solution notebook. First **PyTorch** lab; Tier A (real credit_g via relkit) for the bake-off + a
synthetic Tier-C block for the depth-degradation mechanism.

Crucial fragment: implement `ResNetBlock.forward` — the residual `return x + f(x)` (skip ON) vs
`return f(x)` (skip OFF), Gorishniy 2021 §3.2. Then show a deep PLAIN net degrades (train accuracy
falls -> optimization, not overfitting) while the ResNet holds, and run the honest
MLP-vs-ResNet-vs-GBDT bake-off (no universal winner; GBDT wins on small categorical data).

Run: .venv/bin/python labs/_build_l028.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0028-mlp-resnet-baselines.ipynb \
    labs/solutions/0028-mlp-resnet-baselines.ipynb
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from _colab import bootstrap_cells  # noqa: E402


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# ---- shared PROVIDED setup ----
SETUP = r'''# PROVIDED — imports, real credit_g (Tier A) + synthetic data, and the train/score harness. Just run.
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
torch.manual_seed(SEED); np.random.seed(SEED)

# --- Tier A: real credit_g via relkit (OneHot + scale -> dense float matrix for the nets) ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
Xdf, ycg = load_tier_a("credit_g")
num = Xdf.select_dtypes(include="number").columns.tolist()
cat = [c for c in Xdf.columns if c not in num]
ct = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
])
Xcg = ct.fit_transform(Xdf).astype(np.float32)
ycg = ycg.to_numpy()
print(f"credit_g (Tier A): X {Xcg.shape}, positives {ycg.mean():.2f}")

# --- Tier C: synthetic data with enough signal to make the DEPTH mechanism visible ---
def make_synth(n=8000, seed=SEED):
    X, y = make_classification(n_samples=n, n_features=32, n_informative=16, n_redundant=8,
                               n_clusters_per_class=3, class_sep=0.8, flip_y=0.05, random_state=seed)
    return StandardScaler().fit_transform(X).astype(np.float32), y

def train_net(model, Xtr, ytr, epochs=40, bs=256, lr=2e-3, wd=1e-5, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
    return model

@torch.no_grad()
def net_acc(model, X, y):
    model.eval()
    p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
    return accuracy_score(y, (p > 0.5).astype(int)), roc_auc_score(y, p)

print("ready — helpers: make_synth(), train_net(model, Xtr, ytr), net_acc(model, X, y) -> (acc, auc)")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — implement the residual block's forward — TODO (crucial fragment)

**Goal:** complete `ResNetBlock.forward`. Compute the residual branch `f(x)` (the six operations),
then return **`x + f(x)`** when the skip is on, or just **`f(x)`** when it is off. That single
`x +` is the entire difference between a ResNet and a plain MLP.

**Why it matters:** the skip connection makes the *identity map free* — the branch can learn
`f(x) ≈ 0` and the skip carries `x` through — so stacking many blocks cannot hurt. Turn it off and
the same deep network **degrades** (Task 2). This is Gorishniy 2021 §3.2 / He et al. 2015.

**The branch f(x)** (Gorishniy's pre-activation order): `BatchNorm → Linear → ReLU → Dropout →
Linear → Dropout`. The layers are already created in `__init__`; you wire them together.

**Hint boundary:** apply `self.bn`, `self.lin1`, `F.relu`, `self.drop1`, `self.lin2`, `self.drop2`
in order to get `z = f(x)`; then `return x + z if self.use_skip else z`.'''

T1_CODE = r'''# TODO — implement ResNetBlock.forward (the crucial fragment)
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
        z = ____                                  # the branch f(x): bn -> lin1 -> relu -> drop1 -> lin2 -> drop2
        return ____                               # x + z if the skip is on, else just z

# PROVIDED — the full model: embed -> N blocks -> head. Uses your ResNetBlock.
class TabResNet(nn.Module):
    def __init__(self, d_in, d_main=128, d_hidden=256, n_blocks=4, dropout=0.0, use_skip=True):
        super().__init__()
        self.embed = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList([ResNetBlock(d_main, d_hidden, dropout, use_skip) for _ in range(n_blocks)])
        self.head_bn = nn.BatchNorm1d(d_main)
        self.head = nn.Linear(d_main, 1)
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        return self.head(F.relu(self.head_bn(x))).squeeze(-1)

print("ResNetBlock + TabResNet defined")'''

T1_SOL = (T1_CODE
    .replace('        z = ____                                  # the branch f(x): bn -> lin1 -> relu -> drop1 -> lin2 -> drop2',
             '        z = self.drop2(self.lin2(self.drop1(F.relu(self.lin1(self.bn(x))))))')
    .replace('        return ____                               # x + z if the skip is on, else just z',
             '        return x + z if self.use_skip else z'))

T1_CHECK = r'''# CHECK — do not edit
b = ResNetBlock(d_main=8, d_hidden=16, dropout=0.0, use_skip=True).eval()
x = torch.randn(5, 8)
with torch.no_grad():
    zf = b.drop2(b.lin2(b.drop1(F.relu(b.lin1(b.bn(x))))))   # the branch f(x)
    out_skip = b(x)
    b.use_skip = False
    out_plain = b(x)
assert out_skip.shape == x.shape, "A residual block must preserve shape so x + f(x) is defined."
assert torch.allclose(out_skip, x + zf, atol=1e-5), "skip ON must return x + f(x)."
assert torch.allclose(out_plain, zf, atol=1e-5), "skip OFF must return f(x) only."
assert not torch.allclose(out_skip, out_plain), "The skip must actually change the output."
print("Task 1 ok — the block returns x + f(x) with the skip, f(x) without it. That one line is the whole lesson.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — depth: the plain net degrades, the ResNet holds — TODO

**Goal:** train the SAME architecture at growing depth, once with the skip **on** (ResNet) and once
**off** (plain MLP), and record **train and test** accuracy. Watch the plain net degrade — and its
*training* accuracy fall — while the ResNet stays put.

**Why it matters:** if the plain net's *training* accuracy drops with depth, the failure is
**optimization (the degradation problem)**, not overfitting (which would keep train accuracy high).
That is the evidence the skip connection is doing real work. (Tier C synthetic here — we need enough
data/signal to make depth matter; on credit_g's 1000 rows it would not.)

**You implement:** inside the loop, build a `TabResNet` with the right `use_skip` flag, train it, and
read its train and test accuracy.

**Hint boundary:** `TabResNet(d_in, n_blocks=nb, use_skip=<True/False>)`, then `train_net(...)`, then
`net_acc(model, Xte, yte)` and `net_acc(model, Xtr, ytr)`. `net_acc` returns `(acc, auc)` — take `[0]`.'''

T2_CODE = r'''# TODO — depth sweep, skip on vs off (~1-2 min)
Xs, ys = make_synth()
Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.3, random_state=SEED)

depths = [2, 8, 16]
res, plain = {}, {}
for nb in depths:
    m_r = ____                                    # a TabResNet with n_blocks=nb and the skip ON
    m_r = train_net(m_r, Xtr, ytr)
    res[nb] = {"test": net_acc(m_r, Xte, yte)[0], "train": net_acc(m_r, Xtr, ytr)[0]}

    m_p = ____                                    # the SAME TabResNet but with the skip OFF (plain MLP)
    m_p = train_net(m_p, Xtr, ytr)
    plain[nb] = {"test": net_acc(m_p, Xte, yte)[0], "train": net_acc(m_p, Xtr, ytr)[0]}

print(f"{'depth':>5} | {'plain test':>10} | {'resnet test':>11} | {'plain train':>11} | {'resnet train':>12}")
for nb in depths:
    print(f"{nb:>5} | {plain[nb]['test']:>10.3f} | {res[nb]['test']:>11.3f} | "
          f"{plain[nb]['train']:>11.3f} | {res[nb]['train']:>12.3f}")

plain_train_drop = plain[depths[-1]]['train'] - plain[depths[0]]['train']
print(f"\nplain TRAIN accuracy change {depths[0]} -> {depths[-1]} blocks: {plain_train_drop:+.3f}")'''

T2_SOL = (T2_CODE
    .replace("    m_r = ____                                    # a TabResNet with n_blocks=nb and the skip ON",
             "    m_r = TabResNet(Xs.shape[1], n_blocks=nb, use_skip=True)")
    .replace("    m_p = ____                                    # the SAME TabResNet but with the skip OFF (plain MLP)",
             "    m_p = TabResNet(Xs.shape[1], n_blocks=nb, use_skip=False)"))

T2_CHECK = r'''# CHECK — do not edit
assert plain[16]['test'] < plain[2]['test'], "The plain MLP should DEGRADE (lose test accuracy) as depth grows."
assert plain_train_drop < 0, "The plain net's TRAINING accuracy should fall too -> optimization (degradation), not overfitting."
assert res[16]['test'] >= plain[16]['test'] - 0.005, "At depth 16 the ResNet should hold at least as well as the plain net."
assert res[16]['train'] > 0.95, "The ResNet's training accuracy should stay high at every depth (the skip keeps it trainable)."
print(f"Task 2 ok — plain train fell {plain_train_drop:+.3f} with depth (degradation, not overfitting); "
      f"the ResNet held. The skip makes the identity free.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — the honest bake-off on credit_g — TODO

**Goal:** on the real Tier-A `credit_g` table, compare a plain **MLP** (skip off, shallow), a
**ResNet** (skip on), and a tuned **GBDT** (HistGradientBoosting), by ROC-AUC over a few seeds.

**Why it matters:** this is Gorishniy's real message. A *properly built* neural baseline makes the
comparison **fair** — but on a small, categorical, irregular table the tuned tree still wins (L024–
L027), and the two neural baselines tie. "No universal winner"; the ResNet is an honest baseline, not
a headline winner. Reporting a weak MLP would have flattered a fake DL "win".

**You implement:** the GBDT's predicted positive-class probability, so you can score its ROC-AUC the
same way as the nets.

**Hint boundary:** `gbt.predict_proba(Xte)` returns a 2-column array; the positive-class probability
is column `[:, 1]`. Score with `roc_auc_score(yte, p)`.'''

T3_CODE = r'''# TODO — bake-off: MLP vs ResNet vs GBDT on credit_g (~1 min)
auc = {"mlp": [], "resnet": [], "gbt": []}
for s in [0, 1, 2]:
    Xtr, Xte, ytr, yte = train_test_split(Xcg, ycg, test_size=0.3, random_state=s, stratify=ycg)

    mlp = TabResNet(Xcg.shape[1], d_main=64, d_hidden=128, n_blocks=2, dropout=0.3, use_skip=False)
    mlp = train_net(mlp, Xtr, ytr, epochs=60, lr=1e-3, wd=1e-4, seed=s)
    auc["mlp"].append(net_acc(mlp, Xte, yte)[1])

    rn = TabResNet(Xcg.shape[1], d_main=64, d_hidden=128, n_blocks=2, dropout=0.3, use_skip=True)
    rn = train_net(rn, Xtr, ytr, epochs=60, lr=1e-3, wd=1e-4, seed=s)
    auc["resnet"].append(net_acc(rn, Xte, yte)[1])

    gbt = HistGradientBoostingClassifier(random_state=s, learning_rate=0.05,
                                         max_leaf_nodes=31, l2_regularization=1.0).fit(Xtr, ytr)
    p = ____                                      # GBDT positive-class probability on Xte
    auc["gbt"].append(roc_auc_score(yte, p))

for k in ["mlp", "resnet", "gbt"]:
    print(f"{k:>7} ROC-AUC: {np.mean(auc[k]):.3f} ± {np.std(auc[k]):.3f}")'''

T3_SOL = T3_CODE.replace(
    "    p = ____                                      # GBDT positive-class probability on Xte",
    "    p = gbt.predict_proba(Xte)[:, 1]              # GBDT positive-class probability on Xte")

T3_CHECK = r'''# CHECK — do not edit
mlp_m, rn_m, gbt_m = np.mean(auc["mlp"]), np.mean(auc["resnet"]), np.mean(auc["gbt"])
assert gbt_m > max(mlp_m, rn_m), "On small categorical credit_g the tuned GBDT should lead (L024-L027 territory)."
assert abs(mlp_m - rn_m) < 0.05, "The two neural baselines should be competitive with each other (a tie)."
print(f"Task 3 ok — GBDT {gbt_m:.3f} > MLP {mlp_m:.3f} ~ ResNet {rn_m:.3f}. No universal winner; the "
      f"neural baseline is honest, not a headline winner.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, what does the skip connection fix, how do you know it is an
optimization problem (not overfitting), and what does building the neural baseline correctly buy you?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 028 ===")
print(f"depth (synthetic): plain test {plain[2]['test']:.3f}(d2) -> {plain[16]['test']:.3f}(d16); "
      f"ResNet {res[2]['test']:.3f} -> {res[16]['test']:.3f}")
print(f"degradation      : plain TRAIN {plain[2]['train']:.3f} -> {plain[16]['train']:.3f} "
      f"(train falls -> optimization, not overfitting)")
print(f"bake-off credit_g: MLP {np.mean(auc['mlp']):.3f} ~ ResNet {np.mean(auc['resnet']):.3f} "
      f"< GBDT {np.mean(auc['gbt']):.3f} (no universal winner)")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A residual block returns x + f(x) instead of f(x), which makes the identity map free and fixes the '
    'degradation problem — a deep PLAIN MLP whose TRAINING accuracy falls with depth (so it is an '
    'optimization failure, not overfitting) — while the ResNet holds; building this honest, properly-tuned '
    'neural baseline does not win on small categorical data (a tuned GBDT still leads and the two nets tie), '
    'but it makes the comparison fair, which is exactly the bar an RDL result must clear."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 028 — MLP & ResNet tabular baselines: the skip connection

**Lesson:** [`lessons/0028-mlp-resnet-baselines.html`](../lessons/0028-mlp-resnet-baselines.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Gorishniy, Rubachev, Khrulkov & Babenko 2021, *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 ([arXiv:2106.11959](https://arxiv.org/abs/2106.11959)) — §3.2 (the MLP and ResNet baselines). Residual idea: He et al. 2015, *Deep Residual Learning* ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)).

**Dataset tiers:** **A** (real `credit_g` via `relkit`, for the honest bake-off) + **C** (synthetic, to isolate the depth mechanism — 1000 rows can't show a degradation effect that is about training *deep* nets).

**Skill you are practising:** implement the **residual block's `forward`** — the single line `return x + f(x)` (skip ON) vs `return f(x)` (skip OFF). Then show a deep **plain** net degrades (its *training* accuracy falls — optimization, not overfitting) while the **ResNet** holds, and run the honest MLP-vs-ResNet-vs-GBDT bake-off.

**Exit criteria:** EXIT TICKET prints the plain net degrading with depth (train accuracy falling), the ResNet holding, and — on credit_g — the two neural baselines tying while the tuned GBDT leads.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, the train/score harness); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. This is the **first PyTorch lab**: it needs **torch** (CPU is fine) in addition to scikit-learn. `credit_g` is fetched from OpenML on first run (needs network) then cached. Training a few small nets takes ~2–3 minutes on CPU.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — the residual block, and the degradation problem it fixes

**MLP.** A stack of linear layers with a non-linearity (ReLU) between them: `Linear → ReLU → Linear → …`. Gorishniy's MLP block is `Dropout(ReLU(Linear(x)))`.

**The problem.** Make a plain MLP *deep* and it gets **worse** — and, tellingly, its **training** accuracy drops too. That rules out overfitting (which keeps train accuracy high). The deep plain stack is simply **hard to optimize**: the extra `Linear→ReLU` layers cannot easily learn to reproduce their own input. He et al. (2015) named this the **degradation problem** and showed it is *not* vanishing gradients (BatchNorm already fixes those).

**The fix — a residual (skip) connection.** Make each block output `x + f(x)` instead of `f(x)`. The branch `f` (Gorishniy's order: `BatchNorm → Linear → ReLU → Dropout → Linear → Dropout`) only learns a small *correction*; the skip carries `x` through. This makes the **identity map free**: `f(x) ≈ 0` is trivial to learn, so a deep ResNet can always fall back to a shallow solution — adding depth can't hurt.

**BatchNorm** stabilizes activation scale (faster, less fiddly training). **Dropout** randomly zeroes activations to regularize. Neither causes the depth problem; the **skip** is what fixes it.

**The honest-baseline point.** A *properly tuned* MLP/ResNet is the neural baseline the field reports against — many "SOTA" tabular DL papers failed to beat it. On small categorical data a tuned GBDT still wins (L024–L027), so the ResNet is an honest baseline, **not** a headline winner.

Full write-up + the block diagram, the depth curves, and the bake-off: [Lesson 028](../lessons/0028-mlp-resnet-baselines.html).'''),
        md("## Setup — PROVIDED (data + train/score harness)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — probe the baseline

1. **How deep does the skip start to matter?** In Task 2 the gap opens up past depth ~4. Re-run with
   `depths = [1, 2, 4, 8, 16, 32]` and find the depth where the plain net's *training* accuracy first
   drops below 0.99. That is where the degradation problem bites.
2. **Remove BatchNorm.** Delete `self.bn` (replace with identity) from the block and re-run the depth
   sweep. Both nets get harder to train — evidence the degradation problem and vanishing gradients are
   *two* distinct failures (BatchNorm fixes the latter, the skip the former).
3. **Tune the neural baseline.** Give the credit_g nets a small random search over width, depth,
   dropout, and learning rate (L017 discipline, honest budget). Can a tuned ResNet close the gap to the
   GBDT — or does the tree hold? Report honestly.
4. **A learned gate preview.** The Year-2 fix for the tree advantage is per-feature embeddings /
   attention (FT-Transformer). As a cheap proxy, does standardizing + one-hot (already done) plus more
   epochs move the neural AUC? Where does it plateau vs the GBDT?'''),
        code(r'''# STRETCH — ungraded.
# depths = [1, 2, 4, 8, 16, 32]
# for nb in depths:
#     m = TabResNet(Xs.shape[1], n_blocks=nb, use_skip=False)
#     m = train_net(m, Xtr, ytr)
#     print(nb, "plain train", round(net_acc(m, Xtr, ytr)[0], 3))'''),
    ]

    nb_obj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for c in nb_obj["cells"]:
        if isinstance(c["source"], str):
            c["source"] = c["source"].splitlines(keepends=True)
    return nb_obj


def main():
    student = build(solution=False)
    sol = build(solution=True)
    with open(os.path.join(HERE, "0028-mlp-resnet-baselines.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0028-mlp-resnet-baselines.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0028-mlp-resnet-baselines.ipynb + solution")


if __name__ == "__main__":
    main()
