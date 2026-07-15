"""Build Lab 024 (the Grinsztajn benchmark protocol) — emits the blank student
notebook and a filled solution notebook. Tier-A reproduction lab (real OpenML data
via relkit); mirrors the L021–L023 concept-lab structure but reproduces a protocol.

Run: .venv/bin/python labs/_build_l024.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0024-grinsztajn-benchmark.ipynb labs/solutions/0024-grinsztajn-benchmark.ipynb
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# ---- shared PROVIDED setup (mirrors labs/_verify_l024.py) ----
SETUP = r'''# PROVIDED — data, models, config samplers, and the (slow) evaluation. Just run.
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
# make relkit importable whether the notebook runs from labs/ or labs/solutions/
sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path(".").resolve().parent))

import numpy as np
from relkit import load_tier_a
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 0
N_CONFIGS = 30      # random-search iterations per model
N_SHUFFLES = 40     # random orderings to average the budget curve over

X, y = load_tier_a("credit_g")            # real OpenML data (Tier A)
y = np.asarray(y)
cat = X.select_dtypes(include=["category", "object"]).columns.tolist()
num = [c for c in X.columns if c not in cat]
print(f"credit-g: n={len(X)}, d={X.shape[1]} ({len(num)} num + {len(cat)} cat), prevalence={y.mean():.3f}")

# train / valid / test = 60 / 20 / 20  (valid selects hyperparameters, test reports)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
print(f"split: train {len(Xtr)}  valid {len(Xva)}  test {len(Xte)}")

def make_gbt(cfg):
    return HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"], max_leaf_nodes=cfg["max_leaf_nodes"],
        max_iter=cfg["max_iter"], l2_regularization=cfg["l2"], random_state=SEED)

def make_mlp(cfg):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    clf = MLPClassifier(hidden_layer_sizes=cfg["hidden"], alpha=cfg["alpha"],
                        learning_rate_init=cfg["lr"], max_iter=300,
                        early_stopping=True, random_state=SEED)
    return make_pipeline(pre, clf)

def sample_gbt_configs(rng, n):
    return [dict(learning_rate=float(10**rng.uniform(-2.5,-0.3)),
                 max_leaf_nodes=int(rng.integers(8,64)),
                 max_iter=int(rng.integers(100,600)),
                 l2=float(10**rng.uniform(-4,1))) for _ in range(n)]

def sample_mlp_configs(rng, n):
    W, D = [64,128,256], [1,2,3]
    out = []
    for _ in range(n):
        w=int(rng.choice(W)); dep=int(rng.choice(D))
        out.append(dict(hidden=tuple([w]*dep), alpha=float(10**rng.uniform(-5,-1)),
                        lr=float(10**rng.uniform(-4,-2))))
    return out

def evaluate(build, cfgs):
    """Fit each config on train; return (valid_acc, test_acc) arrays."""
    va, te = [], []
    for cfg in cfgs:
        m = build(cfg); m.fit(Xtr, ytr)
        va.append(accuracy_score(yva, m.predict(Xva)))
        te.append(accuracy_score(yte, m.predict(Xte)))
    return np.array(va), np.array(te)

rng = np.random.default_rng(SEED)
gbt_va, gbt_te = evaluate(make_gbt, sample_gbt_configs(rng, N_CONFIGS))
mlp_va, mlp_te = evaluate(make_mlp, sample_mlp_configs(rng, N_CONFIGS))
print("evaluated", N_CONFIGS, "configs per model (valid + test accuracy each)")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — the best-so-far budget curve — TODO (crucial fragment)

**Goal:** turn each model's `(valid, test)` scores into a **budget curve**: the expected test
accuracy of the best-**validation** config found among the first `k` random draws, averaged over
many random orderings of the draw sequence.

**Why it matters:** this is the methodological heart of Grinsztajn 2022 (§4). Reporting one tuned
peak flatters whichever model is hardest to tune; the curve shows both the **default** (k=1) and the
**tuned ceiling** (large k). The non-negotiable rule: **select the config by the validation split,
report its test score** — never pick by test.

**You implement:** inside the loop, (1) update `best_i` to the index of the best *validation* score
seen so far, and (2) accumulate the *test* score of that `best_i`.

**Hint boundary:** you compare `v[k]` to `v[best_i]` to update `best_i`; you add `t[best_i]` (the test
score of the current best-by-validation config) to `curve[k]`. Do not peek at test to choose.'''

T1_CODE = r'''# TODO — implement the select-by-validation, report-test budget curve
def budget_curve(valid, test, n_shuffles=N_SHUFFLES, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(valid)
    curve = np.zeros(n)
    for _ in range(n_shuffles):
        order = rng.permutation(n)
        v, t = valid[order], test[order]
        best_i = 0
        for k in range(n):
            if ____:                      # update best_i when a better VALIDATION score appears
                best_i = k
            curve[k] += ____              # accumulate the TEST score of the best-by-validation config
    return curve / n_shuffles

gbt_curve = budget_curve(gbt_va, gbt_te)
mlp_curve = budget_curve(mlp_va, mlp_te)

print(f"{'budget k':>9} | {'GBT':>7} | {'MLP':>7} | GBT-MLP")
for k in [1, 2, 5, 10, 20, 30]:
    i = k - 1
    print(f"{k:>9} | {gbt_curve[i]:.4f} | {mlp_curve[i]:.4f} | {gbt_curve[i]-mlp_curve[i]:+.4f}")'''

T1_SOL = (T1_CODE
    .replace('if ____:                      # update best_i when a better VALIDATION score appears',
             'if v[k] > v[best_i]:          # update best_i when a better VALIDATION score appears')
    .replace('curve[k] += ____              # accumulate the TEST score of the best-by-validation config',
             'curve[k] += t[best_i]         # accumulate the TEST score of the best-by-validation config'))

T1_CHECK = r'''# CHECK — do not edit
assert len(gbt_curve) == N_CONFIGS and len(mlp_curve) == N_CONFIGS
assert gbt_curve[0] > mlp_curve[0], "GBT should beat the MLP by DEFAULT (budget k=1)."
assert (gbt_curve - mlp_curve).min() > 0, "GBT should be above the MLP at EVERY budget."
assert 0.04 < (gbt_curve[0] - mlp_curve[0]) < 0.09, "Default gap should be ~+0.06."
print(f"Task 1 ok — GBT leads at every budget; default gap "
      f"{gbt_curve[0]-mlp_curve[0]:+.4f}, fully-tuned gap {gbt_curve[-1]-mlp_curve[-1]:+.4f}. "
      f"Tuning helps the MLP more, but never enough to catch up.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — per-dataset normalization — TODO

**Goal:** implement Grinsztajn's affine (min-max) normalization that makes datasets commensurable so
their curves can be averaged across the suite.

**Why it matters:** an 80% on an easy dataset and 80% on a hard one are not comparable (echo of Demšar,
L023 — don't average raw accuracies). Rescaling each dataset's scores to `[0, 1]` (worst model → 0,
best → 1) stops one high-variance dataset from hijacking the benchmark mean.

**The formula:** `normalized(s) = (s − worst) / (best − worst)`, where `worst`/`best` are the min/max
test scores over **all models on this dataset** (here, both curves together).

**You implement:** `lo` and `hi` over the two curves, then the affine rescale.

**Hint boundary:** `lo` is the min over *both* curves, `hi` the max over both; then apply the formula to
each curve.'''

T2_CODE = r'''# TODO — affine min-max normalize both curves onto [0, 1]
both = np.concatenate([gbt_curve, mlp_curve])
lo = ____                                  # worst (min) test score over ALL models on this dataset
hi = ____                                  # best (max) test score over ALL models on this dataset

def normalize(curve):
    return ____                            # (curve - lo) / (hi - lo)

gbt_norm = normalize(gbt_curve)
mlp_norm = normalize(mlp_curve)
print(f"raw range on credit-g: [{lo:.4f}, {hi:.4f}]")
print(f"normalized GBT: k=1 {gbt_norm[0]:.3f}  k=30 {gbt_norm[-1]:.3f}")
print(f"normalized MLP: k=1 {mlp_norm[0]:.3f}  k=30 {mlp_norm[-1]:.3f}")'''

T2_SOL = (T2_CODE
    .replace('lo = ____                                  # worst (min) test score over ALL models on this dataset',
             'lo = both.min()                            # worst (min) test score over ALL models on this dataset')
    .replace('hi = ____                                  # best (max) test score over ALL models on this dataset',
             'hi = both.max()                            # best (max) test score over ALL models on this dataset')
    .replace('    return ____                            # (curve - lo) / (hi - lo)',
             '    return (curve - lo) / (hi - lo)        # affine rescale onto [0, 1]'))

T2_CHECK = r'''# CHECK — do not edit
assert abs(min(gbt_norm.min(), mlp_norm.min()) - 0.0) < 1e-9, "The worst model should map to 0."
assert abs(max(gbt_norm.max(), mlp_norm.max()) - 1.0) < 1e-9, "The best model should map to 1."
assert gbt_norm[0] > mlp_norm[0] and (gbt_norm - mlp_norm).min() > -1e-9, "Ordering must be preserved."
print(f"Task 2 ok — worst model -> 0.00, best -> 1.00, ordering preserved. "
      f"These bounded per-dataset curves are what Grinsztajn averages across the 45-dataset suite.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — read the verdict — TODO

**Goal:** compute the gap curve and state, in numbers, the two honest readings of the benchmark.

**Why it matters:** the benchmark's headline is not "trees win" but *trees win at every budget, and are
the stronger default*. You are extracting exactly that from your own reproduction.

**You implement:** the elementwise `gap = gbt_curve - mlp_curve`, and the boolean
`gbt_wins_everywhere` = "is the minimum gap still positive?".

**Hint boundary:** `gap` is an array subtraction; `gbt_wins_everywhere` uses `gap.min() > 0`.'''

T3_CODE = r'''# TODO — the gap curve and the verdict
gap = ____                                 # gbt_curve - mlp_curve (elementwise)
gbt_wins_everywhere = ____                 # True iff the GBT leads at EVERY budget (min gap > 0)

print(f"default gap (k=1):      {gap[0]:+.4f}")
print(f"fully-tuned gap (k=30): {gap[-1]:+.4f}")
print(f"gap narrows with budget: {gap[0] > gap[-1]}")
print(f"GBT leads at every budget: {gbt_wins_everywhere}")'''

T3_SOL = (T3_CODE
    .replace('gap = ____                                 # gbt_curve - mlp_curve (elementwise)',
             'gap = gbt_curve - mlp_curve                # elementwise')
    .replace('gbt_wins_everywhere = ____                 # True iff the GBT leads at EVERY budget (min gap > 0)',
             'gbt_wins_everywhere = bool(gap.min() > 0)  # True iff GBT leads at EVERY budget'))

T3_CHECK = r'''# CHECK — do not edit
assert gbt_wins_everywhere is True, "On credit-g the GBT should lead at every budget."
assert gap[0] > gap[-1], "The gap should NARROW as tuning budget grows (tuning helps the MLP more)."
print(f"Task 3 ok — the gap narrows from {gap[0]:+.4f} (default) to {gap[-1]:+.4f} (tuned) "
      f"but never closes. That is Grinsztajn's Figure 1 in miniature.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why is a random-search budget curve (with per-dataset
normalization) a fairer way to compare model families than each model's single best tuned score?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 024 ===")
print(f"protocol : random-search budget curve, select by valid / report test, avg of {N_SHUFFLES} orderings")
print(f"default  : GBT {gbt_curve[0]:.4f} vs MLP {mlp_curve[0]:.4f}  (gap {gap[0]:+.4f})")
print(f"tuned    : GBT {gbt_curve[-1]:.4f} vs MLP {mlp_curve[-1]:.4f}  (gap {gap[-1]:+.4f})")
print(f"verdict  : GBT leads at every budget = {gbt_wins_everywhere}; gap narrows but never closes")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A single tuned number hides how much tuning each model needed and its default quality, so Grinsztajn '
    'reports the expected test score of the best-validation config vs the random-search budget — revealing both '
    'the default (k=1) and the ceiling (large k) — and affine-normalizes each dataset (worst->0, best->1) so the '
    'incommensurable per-dataset scores can be averaged across the suite without one dataset dominating."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 024 — The Grinsztajn benchmark protocol: budget curves, not single numbers

**Lesson:** [`lessons/0024-grinsztajn-benchmark.html`](../lessons/0024-grinsztajn-benchmark.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 D&B ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) — §3 (dataset criteria), §4 (random-search budget curves + normalization). Repo: [tabular-benchmark](https://github.com/LeoGrin/tabular-benchmark).

**Dataset tier:** A — real OpenML `credit-g` (via `relkit.load_tier_a`). This is a *single-dataset* reproduction of the **protocol**, not the full 45-dataset suite.

**Skill you are practising:** compare two model families the fair way — a **random-search budget curve** (select by validation, report test, averaged over orderings) instead of one tuned peak, plus **per-dataset min-max normalization** so datasets can be averaged.

**Exit criteria:** EXIT TICKET prints the GBT-vs-MLP budget curve (GBT ahead at every budget; default gap ≈ +0.06, fully-tuned gap ≈ +0.015) and the normalized curves (worst→0, best→1).

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, models, the slow evaluation); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** + **relkit** and a network connection for the first OpenML fetch (then cached). The evaluation cell trains 60 models and takes ~1–2 minutes.'''),
        md(r'''## Concept recap — a benchmark is an instrument, and it can be rigged

**The tuning trap.** "Model A: 0.82, Model B: 0.81" is meaningless without knowing how much each was tuned. A neural net with many sensitive knobs might need 200 random-search iterations to reach 0.82, while a GBT reaches 0.81 on defaults. Reporting only the peak flatters the model that is hardest to tune — and hides its deployment cost.

**The budget curve (§4).** For each model, draw many random hyperparameter configs; for a budget `k`, take the config with the best **validation** score among the first `k` draws and record its **test** score; average over many random orderings of the draws. The result is a curve of test score vs number of search iterations — it shows the **default** (k=1) and the **tuned ceiling** (large k) at once. The rule that keeps it honest: *select by validation, report test* (callback L003/L004 — never pick by the test set).

**Normalization (§4).** To average across datasets you must first make them commensurable (raw accuracies are not — echo of Demšar, L023). Affine-rescale each dataset's scores so the worst model → 0 and the best → 1, then average those.

**Curation (§3).** Before any of this, the datasets are filtered by explicit rules: real, tabular, medium-sized, heterogeneous columns, not too easy. Curation is half the experiment — it is where a benchmark can quietly cheat.

**Toy micro-example (not this lab's answer).** Say draw order gives validation `[.70, .74, .72]` and test `[.69, .73, .71]`. Best-so-far *by validation*: k=1 pick idx0 → report test .69; k=2 idx1 is better → report .73; k=3 idx1 still best → report .73. You never report .71's test just because it came last.

Full write-up + the budget-curve and dataset-funnel widgets: [Lesson 024](../lessons/0024-grinsztajn-benchmark.html).'''),
        md("## Setup — PROVIDED (loads data, sizes the split, evaluates 60 configs — ~1–2 min)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — push the protocol

1. **Selection honesty.** Build a *cheating* curve that selects the best config by the **test** score directly (`best_i` by `t`, not `v`). It will sit above the honest curve — that gap is the optimism you avoid by selecting on validation. This is L003/L004 in one plot.
2. **More budget.** Raise `N_CONFIGS` to 60 and re-run. Does the MLP ever catch the GBT? Does the GBT curve's small-data dip (validation overfitting) get worse?
3. **A second dataset.** Swap `load_tier_a("adult")` (larger, less noisy). Recompute both curves and normalize. Is the gap the same shape? One dataset is an anecdote; the suite is the benchmark.
4. **Default vs ceiling.** Report, for each model, `curve[0]` (default) and `curve.max()` (ceiling) separately — the two numbers a single tuned score collapses into one.'''),
        code(r'''# STRETCH — ungraded.
# def cheating_curve(valid, test, n_shuffles=N_SHUFFLES, seed=SEED):
#     rng = np.random.default_rng(seed); n=len(valid); curve=np.zeros(n)
#     for _ in range(n_shuffles):
#         order=rng.permutation(n); t=test[order]; best=0
#         for k in range(n):
#             if t[k] > t[best]: best = k          # peeks at TEST -> optimistic
#             curve[k] += t[best]
#     return curve/n_shuffles
# gbt_cheat = cheating_curve(gbt_va, gbt_te)
# print("optimism from selecting on test (GBT):", np.round(gbt_cheat - gbt_curve, 4)[:5])'''),
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
    with open(os.path.join(HERE, "0024-grinsztajn-benchmark.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0024-grinsztajn-benchmark.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0024-grinsztajn-benchmark.ipynb + labs/solutions/0024-grinsztajn-benchmark.ipynb")


if __name__ == "__main__":
    main()
