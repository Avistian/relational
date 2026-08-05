"""Build Lab 042 (MLP & ResNet baselines — do these first) — student + solution notebooks.

Honours NOTES standards #22 (build from scratch; rtdl only validates) and #23 (never conclude from one
dataset). The models are the student's FROM-SCRATCH `relkit.nets.TabResNet` / `TabMLP` (promoted from
Lab 028); rtdl is used once to VALIDATE them. The bake-off runs across SEVERAL small real datasets and is
read with mean ranks — "no universal winner" is shown, not asserted from a single table.

Tasks: (1) the fair early-stopping training loop, from scratch; (2) validate the from-scratch ResNet
against rtdl's reference; (3) the multi-dataset bake-off + rank verdict.

Run: .venv/bin/python labs/_build_l042.py
Execute the solution: .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    labs/solutions/0042-mlp-resnet-baselines.ipynb
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


SETUP = r'''# PROVIDED — imports, small real datasets (Tier A), the shared frame, per-model search SPACES,
# scoring, and the GBDT search. The neural models are your FROM-SCRATCH ones from relkit.nets
# (the code you wrote in Lab 028, promoted so labs build on it — NOTES #22). rtdl is imported only to
# VALIDATE them in Task 2. Just run this cell.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
from relkit.nets import TabResNet, TabMLP              # FROM-SCRATCH models (yours, from Lab 028)
from rtdl_revisiting_models import ResNet as RtdlResNet  # reference — for validation only (Task 2)

import torch
import torch.nn as nn
DEVICE = "cpu"

# Small real binary tables (registered in relkit.data). NOTE this set is numeric-heavy (diabetes, kc1),
# a regime that tends to favour nets — so it is NOT representative. That is deliberate: it lets you SEE
# why dataset choice decides the answer, and why the real verdict needs a large, varied benchmark (#23).
DATASETS = ["credit_g", "diabetes", "kc1"]

def load_dense(name):
    Xdf, y = load_tier_a(name)
    num = Xdf.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xdf.columns if c not in num]
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return ct.fit_transform(Xdf).astype(np.float32), y.to_numpy().astype(np.float32)

def frame(X, y, seed):
    Xtr_f, Xte, ytr_f, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25, random_state=seed, stratify=ytr_f)
    return Xtr, ytr, Xva, yva, Xte, yte

def sample_config(kind, rng, d_in):
    optim = dict(lr=float(10 ** rng.uniform(-3.3, -2.5)), wd=float(10 ** rng.uniform(-6, -3)))
    if kind == "resnet":
        d_main = int(rng.choice([64, 128, 192]))
        cfg = dict(d_in=d_in, d_main=d_main, d_hidden=int(d_main * rng.choice([1.0, 2.0])),
                   n_blocks=int(rng.choice([1, 2, 3])),
                   dropout1=float(rng.uniform(0.0, 0.3)), dropout2=float(rng.uniform(0.0, 0.3)))
    else:
        cfg = dict(d_in=d_in, d_block=int(rng.choice([64, 128, 256])),
                   n_blocks=int(rng.choice([1, 2, 3])), dropout=float(rng.uniform(0.0, 0.4)))
    return cfg, optim

def build_model(kind, cfg):
    return TabResNet(**cfg) if kind == "resnet" else TabMLP(**cfg)

@torch.no_grad()
def net_test_auc(model, X, y):
    model.eval()
    out = model(torch.tensor(X, device=DEVICE))
    if out.ndim > 1: out = out.squeeze(-1)
    return roc_auc_score(y, torch.sigmoid(out).cpu().numpy())

def search_gbt(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):   # shares the budget; provided
    rng = np.random.default_rng(seed + 999)
    best = {"val": -1.0, "test": None}
    for _ in range(budget):
        cfg = dict(learning_rate=float(10 ** rng.uniform(-1.5, -0.7)),
                   max_leaf_nodes=int(rng.choice([15, 31, 63])),
                   l2_regularization=float(rng.uniform(0.0, 3.0)), max_iter=400)
        clf = HistGradientBoostingClassifier(random_state=seed, **cfg).fit(Xtr, ytr)
        val = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if val > best["val"]:
            best = {"val": val, "test": roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])}
    return best["test"]

print("ready — from-scratch models: TabResNet/TabMLP; datasets:", DATASETS)'''


T1_MD = r'''## Task 1 — a fair, early-stopped training loop (from scratch) — TODO (crucial fragment)

**Goal:** complete `train_net` so it trains a net and **selects on the validation metric**: after each
epoch measure validation ROC-AUC, keep the weights from the *best* validation epoch, and **stop early**
once validation has not improved for `patience` epochs.

**Why it matters:** early stopping is how a neural baseline picks its own training length *fairly* — the
analogue of a GBDT's early-stopped `n_estimators`. Selecting on validation (never test) is the
load-bearing move of the shared protocol: every model is judged by the same rule. You are writing this
loop yourself (NOTES #22) — the library is not doing it for you.

**Three blanks:** the validation ROC-AUC, the "is this the best so far?" test, and the early-stop check.

**Hint boundary:** `roc_auc_score(yva, pv)`; keep it if greater than the best seen; stop once `since`
reaches `patience`.'''

T1_CODE = r'''# TODO — complete the early-stopping / validation-selection loop (the crucial fragment)
def train_net(model, Xtr, ytr, Xva, yva, *, lr, wd, max_epochs=120, patience=12, seed=0):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)   # AdamW + weight decay
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=DEVICE); yt = torch.tensor(ytr, device=DEVICE)
    Xv = torch.tensor(Xva, device=DEVICE)
    def logits(x):
        o = model(x); return o.squeeze(-1) if o.ndim > 1 else o
    best_auc, best_state, since = -1.0, None, 0
    for _ in range(max_epochs):
        model.train(); opt.zero_grad()
        loss = lossf(logits(Xt), yt); loss.backward(); opt.step()      # full-batch is fine on small tables
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(logits(Xv)).cpu().numpy()
        val_auc = ____                                     # validation ROC-AUC of pv against yva
        if ____:                                           # is this the best validation score so far?
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if ____:                                       # stop when no improvement for `patience` epochs
                break
    if best_state is not None:
        model.load_state_dict(best_state)                  # restore the best-validation weights
    return model, best_auc

def search(kind, Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):   # uses your train_net
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        cfg, optim = sample_config(kind, rng, Xtr.shape[1])
        m, val = train_net(build_model(kind, cfg), Xtr, ytr, Xva, yva,
                           lr=optim["lr"], wd=optim["wd"], seed=seed + t)
        if val > best["val"]:
            best = {"val": val, "test": net_test_auc(m, Xte, yte)}
    return best["test"]

print("train_net + search defined")'''

T1_SOL = (T1_CODE
    .replace('        val_auc = ____                                     # validation ROC-AUC of pv against yva',
             '        val_auc = roc_auc_score(yva, pv)                   # validation ROC-AUC of pv against yva')
    .replace('        if ____:                                           # is this the best validation score so far?',
             '        if val_auc > best_auc:                             # is this the best validation score so far?')
    .replace('            if ____:                                       # stop when no improvement for `patience` epochs',
             '            if since >= patience:                          # stop when no improvement for `patience` epochs'))

T1_CHECK = r'''# CHECK — do not edit
_Xtr, _ytr, _Xva, _yva, _Xte, _yte = frame(*load_dense("credit_g"), 0)
_m, _val = train_net(build_model("resnet", sample_config("resnet", np.random.default_rng(0), _Xtr.shape[1])[0]),
                     _Xtr, _ytr, _Xva, _yva, lr=2e-3, wd=1e-5, seed=0)
assert 0.5 <= _val <= 1.0, "best validation AUC should be a real ROC-AUC in [0.5, 1]."
assert net_test_auc(_m, _Xva, _yva) >= _val - 1e-6, "the restored weights should reproduce the best-validation score."
print(f"Task 1 ok — from-scratch ResNet trained with AdamW + early stopping; best val AUC {_val:.3f}. "
      f"You wrote the fair training loop yourself.")'''


T2_MD = r'''## Task 2 — validate your from-scratch ResNet against rtdl — TODO

**Goal:** confirm your hand-built `TabResNet` is *correct* by checking it reproduces the **rtdl reference
ResNet** under the identical protocol. Train both (a small search) on one dataset over a couple of seeds
and compare their mean test ROC-AUC.

**Why it matters (NOTES #22):** you learn by building from scratch — but a from-scratch model can hide a
silent bug or be quietly under-powered (the L038 weak-baseline trap). A reference implementation is the
**validation point**: if your ResNet lands within noise of rtdl's, your implementation is trustworthy;
if it is far below, you have a bug to find. rtdl is the *checker*, not the teacher.

**You implement:** the from-scratch-vs-rtdl comparison — the absolute gap in mean test AUC, and the
pass/fail decision at a tolerance.

**Hint boundary:** `abs(scratch_mean - rtdl_mean)`; it passes when that is `< tol`.'''

T2_CODE = r'''# TODO — validate the from-scratch ResNet against rtdl's reference (~1-2 min)
def search_rtdl(Xtr, ytr, Xva, yva, Xte, yte, *, budget, seed):   # reference ResNet, SAME protocol
    rng = np.random.default_rng(seed)
    best = {"val": -1.0, "test": None}
    for t in range(budget):
        d_main = int(rng.choice([64, 128, 192]))
        cfg = dict(d_in=Xtr.shape[1], d_out=1, n_blocks=int(rng.choice([1, 2, 3])), d_block=d_main,
                   d_hidden_multiplier=float(rng.choice([1.0, 2.0])),
                   dropout1=float(rng.uniform(0.0, 0.3)), dropout2=float(rng.uniform(0.0, 0.3)))
        m, val = train_net(RtdlResNet(**cfg), Xtr, ytr, Xva, yva,
                           lr=float(10**rng.uniform(-3.3,-2.5)), wd=float(10**rng.uniform(-6,-3)), seed=seed+t)
        if val > best["val"]:
            best = {"val": val, "test": net_test_auc(m, Xte, yte)}
    return best["test"]

X, y = load_dense("credit_g")
scratch = [search("resnet", *frame(X, y, s), budget=4, seed=s) for s in [0, 1]]
rtdl    = [search_rtdl(*frame(X, y, s), budget=4, seed=s) for s in [0, 1]]
scratch_mean, rtdl_mean = float(np.mean(scratch)), float(np.mean(rtdl))

tol = 0.04
delta = ____                              # absolute gap between the two mean test AUCs
validated = ____                          # True if the from-scratch ResNet is within `tol` of rtdl

print(f"from-scratch ResNet {scratch_mean:.3f} vs rtdl reference {rtdl_mean:.3f}  |Δ|={delta:.3f}  "
      f"-> {'VALIDATED' if validated else 'CHECK YOUR IMPLEMENTATION'}")'''

T2_SOL = (T2_CODE
    .replace('delta = ____                              # absolute gap between the two mean test AUCs',
             'delta = abs(scratch_mean - rtdl_mean)     # absolute gap between the two mean test AUCs')
    .replace('validated = ____                          # True if the from-scratch ResNet is within `tol` of rtdl',
             'validated = delta < tol                   # True if the from-scratch ResNet is within `tol` of rtdl'))

T2_CHECK = r'''# CHECK — do not edit
assert delta == abs(scratch_mean - rtdl_mean), "delta must be the absolute gap in mean test AUC."
assert validated, (f"from-scratch ResNet ({scratch_mean:.3f}) is not within {tol} of rtdl ({rtdl_mean:.3f}) "
                   f"— likely an implementation bug or too-short training.")
print(f"Task 2 ok — your from-scratch ResNet reproduces rtdl within {tol} (|Δ|={delta:.3f}). "
      f"The library validated your code; it did not write it.")'''


T3_MD = r'''## Task 3 — the multi-dataset bake-off + rank verdict — TODO

**Goal:** run the shared protocol on **several** datasets and read the result with **ranks**. For each
dataset, rank the three baselines (1 = best); the honest cross-dataset summary is the **mean rank** per
model. "No universal winner" means *the ranking changes across datasets* — not a single-table result.

**Why it matters (NOTES #23):** a comparison on one dataset is a demonstration, not evidence. On this
small numeric-heavy set the nets tend to rank ahead and the nominal winner flips *between the two nets*
(GBDT trails) — but that sample is biased toward nets, so it is NOT a verdict. The rigorous field summary
is mean ranks + a Friedman test + a critical-difference diagram (L023/L030); the large-N "no universal
winner" (trees usually ahead on categorical/irregular data) is established over ~45 datasets by Grinsztajn
2022 and Gorishniy 2021. We reproduce the *method* here and see how dataset choice drives the result.

**You implement:** the per-dataset ranks (higher AUC → rank 1) and the mean rank per model.

**Hint boundary:** `rankdata(-means_for_this_dataset, method="average")` gives ranks with 1 = best;
average each model's rank across datasets.'''

T3_CODE = r'''# TODO — multi-dataset bake-off, then mean ranks (~4-7 min on CPU)
BUDGET, SEEDS = 4, [0, 1]
MODELS = ["resnet", "mlp", "gbt"]
table = {}   # dataset -> model -> mean test AUC
for name in DATASETS:
    X, y = load_dense(name)
    acc = {m: [] for m in MODELS}
    for s in SEEDS:
        Xtr, ytr, Xva, yva, Xte, yte = frame(X, y, s)
        acc["resnet"].append(search("resnet", Xtr, ytr, Xva, yva, Xte, yte, budget=BUDGET, seed=s))
        acc["mlp"].append(search("mlp", Xtr, ytr, Xva, yva, Xte, yte, budget=BUDGET, seed=s))
        acc["gbt"].append(search_gbt(Xtr, ytr, Xva, yva, Xte, yte, budget=BUDGET, seed=s))
    table[name] = {m: float(np.mean(acc[m])) for m in MODELS}
    print(f"  {name:>10}: " + " ".join(f"{m} {table[name][m]:.3f}" for m in MODELS))

# per-dataset ranks (1 = best) and the mean rank per model
per_dataset_ranks = {}
for name in DATASETS:
    means = [table[name][m] for m in MODELS]
    ranks = ____                                   # ranks of `means` with 1 = best (highest AUC)
    per_dataset_ranks[name] = dict(zip(MODELS, ranks))

mean_rank = {m: ____ for m in MODELS}              # average each model's rank across DATASETS

print("\nmean rank (1 = best):", {m: round(mean_rank[m], 2) for m in MODELS})
best_overall = min(mean_rank, key=mean_rank.get)
flips = len({min(per_dataset_ranks[d], key=per_dataset_ranks[d].get) for d in DATASETS}) > 1
print(f"per-dataset winner flips across datasets? {flips}  (best mean rank: {best_overall})")'''

T3_SOL = (T3_CODE
    .replace('    ranks = ____                                   # ranks of `means` with 1 = best (highest AUC)',
             '    ranks = rankdata([-v for v in means], method="average")   # 1 = best (highest AUC)')
    .replace('mean_rank = {m: ____ for m in MODELS}              # average each model\'s rank across DATASETS',
             'mean_rank = {m: float(np.mean([per_dataset_ranks[d][m] for d in DATASETS])) for m in MODELS}'))

T3_CHECK = r'''# CHECK — do not edit
for name in DATASETS:
    assert abs(sum(per_dataset_ranks[name].values()) - 6.0) < 1e-6, "3 models must have ranks summing to 1+2+3=6."
    for m in MODELS:
        assert 0.60 <= table[name][m] <= 1.0, f"{m} on {name} = {table[name][m]:.3f} is outside a plausible range."
assert abs(sum(mean_rank.values()) - 6.0) < 1e-6, "mean ranks across 3 models must sum to 6."
print(f"Task 3 ok — evaluated on {len(DATASETS)} datasets. Winner flips across datasets: {flips}. "
      f"That — not one table — is what 'no universal winner' means.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence — why do you build the model from scratch and use rtdl only to
validate it, why is a within-noise, rank-flipping result across datasets the *honest* form of "no
universal winner", and why must you never conclude it from a single table?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 042 ===")
print(f"validation (credit_g): from-scratch ResNet {scratch_mean:.3f} ~ rtdl {rtdl_mean:.3f} (|Δ|={delta:.3f})")
print("per-dataset mean test ROC-AUC:")
for name in DATASETS:
    print(f"  {name:>10}: " + " ".join(f"{m} {table[name][m]:.3f}(r{int(per_dataset_ranks[name][m])})" for m in MODELS))
print("mean rank (1=best):", {m: round(mean_rank[m], 2) for m in MODELS}, "| winner flips:", flips)
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"You BUILD the model from scratch so the understanding is yours and use rtdl only to validate it '
    '(a from-scratch net can hide a bug or be too weak — the reference catches that); and because a '
    'comparison on one table is a demonstration, not evidence, you evaluate the shared protocol across '
    'several datasets with per-dataset CIs + mean ranks. Here the tuned nets rank ahead and the winner '
    'flips between MLP and ResNet, but this set is numeric-heavy and tiny — a biased sample — so it shows '
    'the METHOD and that the neural baseline is strong, NOT that nets beat trees; the representative '
    '\'no universal winner\' is grounded in the ~45-dataset benchmarks (Grinsztajn 2022, Gorishniy 2021), '
    'never a single dataset or a handful we picked."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 042 — MLP & ResNet baselines: the shared protocol, from scratch, across datasets

**Lesson:** [`lessons/0042-mlp-resnet-baselines.html`](../lessons/0042-mlp-resnet-baselines.html) · **Phase / Year:** Year 2 · Q1

**Paper:** Gorishniy et al. 2021, *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 ([arXiv:2106.11959](https://arxiv.org/abs/2106.11959)) — §3.2 + the tuning protocol. Cross-dataset "no universal winner": Grinsztajn et al. 2022 ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815), ~45 datasets).

**Dataset tier:** **A** — several small real OpenML tables (`credit_g`, `diabetes`, `kc1`) via `relkit`.

**Skill you are practising:** the **shared tuning protocol**, done the rigorous way — (1) you write the fair early-stopping training loop **from scratch** for your own `relkit.nets` models (the code you built in [Lab 028](0028-mlp-resnet-baselines.ipynb)); (2) you **validate** that from-scratch ResNet against the **rtdl reference** (the library is a checker, not the teacher — NOTES #22); (3) you run the bake-off **across several datasets** and read it with **mean ranks**, because one table is a demonstration, not evidence (NOTES #23).

**Exit criteria:** EXIT TICKET prints your from-scratch ResNet validated against rtdl, per-dataset scores with ranks, and the mean-rank summary showing the winner **flips** across datasets — the honest form of "no universal winner".

---

### How this notebook works
- **PROVIDED** cells — boilerplate (data, frame, search spaces, GBDT, scoring); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` → kernel **Relational Labs (.venv)**. Needs **torch** + **`rtdl_revisiting_models`** (CPU fine) + scikit-learn. The datasets fetch from OpenML on first run (network) then cache. The multi-dataset bake-off trains many tiny nets — **~5–8 minutes on CPU**; set `OMP_NUM_THREADS=1` if a search feels slow (NOTES #20). This lab uses a small budget/seed/dataset count to stay interactive; the lesson's headline numbers come from the fuller `labs/_verify_l042.py` run plus the published benchmarks.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — protocol, build-from-scratch, and multi-dataset rigor

**The two baselines.** A tuned **MLP** (`Dropout(ReLU(Linear))` blocks) and a tuned **ResNet** (blocks add a residual skip, `x + f(x)`, so the identity map is free and depth stops degrading — He 2015). You built these **from scratch** in [Lab 028](0028-mlp-resnet-baselines.ipynb); they now live in `relkit.nets` so this lab builds on them. **rtdl** ships the same models as a *reference* — we use it once, in Task 2, only to **validate** your code (NOTES #22): the library checks you, it does not teach you.

**The shared tuning protocol = a shared FRAME + a per-model SPACE.**
- **Frame (identical for every model):** same train/valid/test split (L020), same metric (ROC-AUC), same search **budget**, selection by **validation** — test touched once.
- **Space (per model):** each searches its own knobs. **Fairness = equal budget, not equal knobs** (the L038 HP-parity trap).
- **AdamW + early stopping:** keep the best-validation weights; a net picks its own training length, the analogue of a GBDT's early-stopped `n_estimators`.

**Why "do these first".** A tuned ResNet *alone* matches many published "novel" architectures, so a new model that does not beat it has shown nothing — run the strong simple baselines **before** any fancy one.

**Multi-dataset rigor (NOTES #23).** "No universal winner" is a statement about *datasets*, so you cannot read it off one table. You evaluate across several, report per-dataset CIs, and summarise with **mean ranks** (+ Friedman / a critical-difference diagram — L023/L030). On this small, numeric-heavy set the tuned nets rank ahead of the GBDT and the nominal winner flips *between the two nets* — but that is a **biased sample** (numeric tables are exactly where nets do relatively well, per Grinsztajn 2022), so it is a *demonstration of the method*, **not** proof nets beat trees. The authoritative large-N verdict — no single model dominates, trees usually ahead on categorical/irregular data — is Grinsztajn 2022 (~45 datasets) and Gorishniy 2021, which we cite. Which datasets you pick decides the answer: that is the rigor lesson.

Full write-up + the interactive bake-off + rank view: [Lesson 042](../lessons/0042-mlp-resnet-baselines.html).'''),
        md("## Setup — PROVIDED (datasets + shared frame + search spaces + scoring)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — push the rigor further

1. **Friedman + critical-difference diagram.** Add more datasets (`blood_transfusion`, `phoneme`,
   `churn` are registered in `relkit.data`), run `scipy.stats.friedmanchisquare` on the per-dataset
   scores, and sketch a Nemenyi CD diagram (L023/L030). With few datasets the test is under-powered —
   note that, and compare to Grinsztajn's ~45-dataset conclusion.
2. **More seeds, tighter CIs.** Raise `SEEDS` to `[0,1,2,3,4]` and report 95% CIs per cell. Do any
   per-dataset "wins" survive as significant, or are they all ties within noise?
3. **Weight-level validation.** Instead of comparing scores, build your `TabResNet` and rtdl's `ResNet`
   with matched shapes, copy weights across, and assert `torch.allclose` on the forward pass — the
   strongest form of the Task-2 validation.
4. **Break fairness on purpose.** Give the ResNet `budget=40` and the GBDT `budget=2`; watch a fake
   "net win" appear — the HP-budget artifact (L038) the shared protocol exists to prevent.'''),
        code(r'''# STRETCH — ungraded.
# from scipy.stats import friedmanchisquare
# M = [[table[d][m] for d in DATASETS] for m in MODELS]
# print(friedmanchisquare(*M))'''),
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
    with open(os.path.join(HERE, "0042-mlp-resnet-baselines.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0042-mlp-resnet-baselines.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0042-mlp-resnet-baselines.ipynb + solution")


if __name__ == "__main__":
    main()
