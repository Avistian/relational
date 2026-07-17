"""Build Lab 027 (the uninformative-features inductive bias) — emits the blank student notebook and a
filled solution notebook. Tier-C mechanism-isolation lab (synthetic); mirrors the L021/L023/L025/L026
concept-lab structure.

Crucial fragment: implement `add_junk` (append k pure-noise columns to a fixed informative block) —
Grinsztajn §5.3 / Finding 2 — then show a tree/GBT is robust while an MLP's clean-data lead erodes as
junk accumulates, and measure the gate (root-split gain: ~0 on junk).

Run: .venv/bin/python labs/_build_l027.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0027-inductive-bias-uninformative-features.ipynb \
    labs/solutions/0027-inductive-bias-uninformative-features.ipynb
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
SETUP = r'''# PROVIDED — synthetic data with a SMOOTH target, models, and the fit/score harness. Just run.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 0
D_INFORM = 5

def make_informative(n=4000, d_inform=D_INFORM, seed=SEED):
    """Return the INFORMATIVE block only: 5 Gaussian features driving a SMOOTH, mildly-nonlinear
    target. The target is smooth on purpose so an MLP WINS on clean data — that isolates the
    uninformative-feature effect from the smoothness (L025) and rotation (L026) biases."""
    rng = np.random.default_rng(seed)
    Xi = rng.normal(size=(n, d_inform))
    s = 1.3 * Xi[:, 0] + 1.0 * Xi[:, 1] + 0.8 * Xi[:, 2] + 0.6 * Xi[:, 3] * Xi[:, 4]
    y = (s > np.median(s)).astype(int)
    return Xi, y

Xi, y = make_informative()
print(f"informative block: n={len(Xi)}, d_inform={Xi.shape[1]}, positives={y.mean():.2f}")

def fit_acc(X, y, seed=SEED):
    """Split, fit a tree / GBT / MLP, return test accuracy for each. The MLP is standardized;
    tree-based models are scale-free."""
    Xs = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.4, random_state=seed)
    tree = DecisionTreeClassifier(max_leaf_nodes=64, random_state=seed).fit(Xtr, ytr)
    gbt = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4,
                      learning_rate_init=3e-3, max_iter=800, random_state=seed),
    ).fit(Xtr, ytr)
    return {"tree": accuracy_score(yte, tree.predict(Xte)),
            "gbt":  accuracy_score(yte, gbt.predict(Xte)),
            "mlp":  accuracy_score(yte, mlp.predict(Xte))}

print("ready — harness fit_acc(X, y) -> {'tree':..., 'gbt':..., 'mlp':...}")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — add uninformative features — TODO (crucial fragment)

**Goal:** implement `add_junk(X, k, seed)` — return `X` with `k` extra **pure-noise** columns appended.
Each junk column is an independent Gaussian, statistically independent of the label.

**Why it matters:** this is the scalpel of Grinsztajn's Finding 2. The junk columns carry *zero*
information about `y`, so any accuracy change when you add them is purely a model failing to ignore
useless inputs — that is what isolates the *bias* rather than a change in the task's difficulty.

**You implement:** (1) draw a `(n, k)` matrix of standard-normal noise, and (2) horizontally stack it
onto `X`.

**Hint boundary:** `rng.normal(size=(n, k))` makes the noise; `np.hstack([...])` appends columns. Return
`X` unchanged when `k == 0`.

**Toy check (not this lab's answer):** appending 3 junk columns to a `(4000, 5)` matrix should give
`(4000, 8)`, and the first 5 columns must be untouched.'''

T1_CODE = r'''# TODO — implement add_junk (the crucial fragment)
def add_junk(X, k, seed):
    if k == 0:
        return X
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    noise = ____                              # a (n, k) block of pure-noise Gaussian columns
    return ____                               # append the noise columns to X (columns, not rows)

# sanity: shapes and that the informative block is untouched
Xj = add_junk(Xi, 3, seed=99)
print(f"shape before {Xi.shape} -> after add_junk(k=3) {Xj.shape}")
print(f"informative block untouched? {np.allclose(Xj[:, :D_INFORM], Xi)}")'''

T1_SOL = (T1_CODE
    .replace('    noise = ____                              # a (n, k) block of pure-noise Gaussian columns',
             '    noise = rng.normal(size=(n, k))           # a (n, k) block of pure-noise Gaussian columns')
    .replace('    return ____                               # append the noise columns to X (columns, not rows)',
             '    return np.hstack([X, noise])              # append the noise columns to X (columns, not rows)'))

T1_CHECK = r'''# CHECK — do not edit
Xj = add_junk(Xi, 3, seed=99)
assert Xj.shape == (Xi.shape[0], Xi.shape[1] + 3), "add_junk should append k columns."
assert np.allclose(Xj[:, :D_INFORM], Xi), "The informative block must be left untouched."
assert np.allclose(add_junk(Xi, 0, seed=1), Xi), "k=0 should return X unchanged."
# the junk columns must be ~uncorrelated with y (they carry no signal)
corr = abs(np.corrcoef(Xj[:, D_INFORM], y)[0, 1])
assert corr < 0.1, "A junk column should be ~uncorrelated with the label."
print("Task 1 ok — add_junk appends pure-noise columns that carry no signal. That is Grinsztajn's scalpel.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — sweep the junk and watch the MLP's lead erode — TODO

**Goal:** for a grid of junk counts `k`, fit the models on the augmented data and record each one's
accuracy. Then quantify the two robustness signatures: how much the **MLP** loses vs how much the
**GBT** loses across the sweep.

**Why it matters:** this is Finding 2 in one comparison. On clean data (`k=0`) the MLP wins (smooth
target). If MLPs are less robust to junk, the MLP should fall much faster than the GBT — enough to flip
the ranking.

**You implement:** call the harness on the junk-augmented features inside the loop, then compute the
MLP's and GBT's change from `k=0` to the largest `k`.

**Hint boundary:** `add_junk(Xi, k, seed=...)` then `fit_acc(...)`. A "change" is
`acc_at_max_k[model] - acc_at_0[model]`.'''

T2_CODE = r'''# TODO — sweep k, record accuracies (~30s)
ks = [0, 15, 50, 100]
rows = {}
for k in ks:
    Xk = ____                                 # append k junk columns to the informative block
    rows[k] = fit_acc(Xk, y)

print(f"{'k_junk':>6} | {'tree':>6} | {'gbt':>6} | {'mlp':>6} | {'gbt-mlp':>8}")
for k in ks:
    r = rows[k]
    print(f"{k:>6} | {r['tree']:>6.3f} | {r['gbt']:>6.3f} | {r['mlp']:>6.3f} | {r['gbt']-r['mlp']:>+8.3f}")

mlp_loss = ____                               # rows[100]['mlp'] - rows[0]['mlp'] (a negative number)
gbt_loss = ____                               # rows[100]['gbt'] - rows[0]['gbt']
print(f"\nMLP change over the sweep: {mlp_loss:+.3f}   GBT change: {gbt_loss:+.3f}")
print(f"gap (gbt-mlp): {rows[0]['gbt']-rows[0]['mlp']:+.3f} at k=0  ->  {rows[100]['gbt']-rows[100]['mlp']:+.3f} at k=100")'''

T2_SOL = (T2_CODE
    .replace("    Xk = ____                                 # append k junk columns to the informative block",
             "    Xk = add_junk(Xi, k, seed=100 + k)        # append k junk columns to the informative block")
    .replace("mlp_loss = ____                               # rows[100]['mlp'] - rows[0]['mlp'] (a negative number)",
             "mlp_loss = rows[100]['mlp'] - rows[0]['mlp']  # a negative number")
    .replace("gbt_loss = ____                               # rows[100]['gbt'] - rows[0]['gbt']",
             "gbt_loss = rows[100]['gbt'] - rows[0]['gbt']"))

T2_CHECK = r'''# CHECK — do not edit
assert rows[0]['mlp'] > rows[0]['gbt'], "On the clean smooth target the MLP should WIN at k=0."
assert mlp_loss < gbt_loss, "The MLP should lose MORE accuracy to junk than the GBT (less robust)."
assert mlp_loss < -0.03, "The MLP should visibly degrade as junk is added."
assert (rows[100]['gbt'] - rows[100]['mlp']) > (rows[0]['gbt'] - rows[0]['mlp']), \
    "Adding junk should shift the gap in the GBT's favour (widen it)."
print(f"Task 2 ok — MLP lost {mlp_loss:+.3f} vs GBT {gbt_loss:+.3f}; the junk-free MLP lead eroded. "
      f"MLPs are not robust to uninformative features.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — see the gate: root-split gain on junk vs signal — TODO

**Goal:** measure *why* the tree is robust. For each feature, compute the **root-split gain** — the
Gini impurity a single best threshold removes on the full data — and compare the mean gain on the
informative columns vs the junk columns.

**Why it matters:** a tree grows greedily, always taking the highest-gain split. If junk columns have
~0 root gain, the tree simply never picks them near the top — that is the *implicit feature selection*
that makes it robust. (Honest caveat: deep in a tree, spurious high-gain noise splits can appear on
small subsets — which is why impurity-based *MDI* importance over-credits noise. The **root** gain on
the full data is where the gate is real.)

**You implement:** the Gini impurity of a label array, and the gain of a threshold split
(`parent − weighted children`).

**Hint boundary:** Gini of a binary array is `1 − p² − (1−p)²` with `p = mean(y)`. Gain =
`parent_gini − (n_left/n)·gini(left) − (n_right/n)·gini(right)`.'''

T3_CODE = r'''# TODO — implement Gini + best-split gain, then compare informative vs junk
def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = labels.mean()
    return ____                               # Gini impurity: 1 - p^2 - (1-p)^2

def root_split_gain(col, labels):
    """Best single-threshold Gini decrease for one feature on the full node."""
    n = len(labels)
    parent = gini(labels)
    order = np.argsort(col)
    ys = labels[order]
    best = 0.0
    for q in np.linspace(0.05, 0.95, 19):     # candidate thresholds at percentiles
        i = int(q * n)
        if i <= 0 or i >= n:
            continue
        left, right = ys[:i], ys[i:]
        g = ____                              # parent - weighted average of children's gini
        best = max(best, g)
    return best

# 5 informative + 20 junk; measure root gain per feature
Xg = add_junk(Xi, 20, seed=7)
gains = np.array([root_split_gain(Xg[:, j], y) for j in range(Xg.shape[1])])
g_inform = gains[:D_INFORM].mean()
g_junk = gains[D_INFORM:].mean()
print(f"root-split gain  informative mean {g_inform:.4f}  |  junk mean {g_junk:.4f}")
print(f"gate ratio informative/junk: {g_inform / max(g_junk, 1e-9):.0f}x")'''

T3_SOL = (T3_CODE
    .replace('    return ____                               # Gini impurity: 1 - p^2 - (1-p)^2',
             '    return 1.0 - p**2 - (1.0 - p)**2          # Gini impurity: 1 - p^2 - (1-p)^2')
    .replace('        g = ____                              # parent - weighted average of children\'s gini',
             '        g = parent - (len(left)/n)*gini(left) - (len(right)/n)*gini(right)'))

T3_CHECK = r'''# CHECK — do not edit
assert abs(gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-9, "Gini of a 50/50 node is 0.5."
assert gini(np.array([1, 1, 1, 1])) == 0.0, "Gini of a pure node is 0."
assert g_inform > 10 * g_junk, "Informative features should have MUCH higher root gain than junk (the gate)."
assert g_junk < 0.01, "A pure-noise column's best root split should remove almost no impurity."
print(f"Task 3 ok — the tree's best split on junk removes ~0 impurity ({g_inform/max(g_junk,1e-9):.0f}x gate). "
      f"That greedy gain gate is the implicit feature selection an MLP lacks.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why does adding pure-noise features hurt an MLP more than a tree,
and what mechanism makes the tree robust?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 027 ===")
print(f"experiment : add k pure-noise columns to 5 informative features; refit tree/GBT/MLP")
print(f"clean (k=0): tree {rows[0]['tree']:.3f} · GBT {rows[0]['gbt']:.3f} · MLP {rows[0]['mlp']:.3f} (MLP wins)")
print(f"k=100 junk : tree {rows[100]['tree']:.3f} · GBT {rows[100]['gbt']:.3f} · MLP {rows[100]['mlp']:.3f}")
print(f"robustness : MLP lost {rows[100]['mlp']-rows[0]['mlp']:+.3f} vs GBT {rows[100]['gbt']-rows[0]['gbt']:+.3f}")
print(f"the gate   : root-split gain informative {g_inform:.4f} vs junk {g_junk:.4f} "
      f"({g_inform/max(g_junk,1e-9):.0f}x)")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A decision tree grows greedily, always taking the highest-gain (impurity-reducing) split, so a '
    'pure-noise column — whose best split has ~0 gain — is never chosen and is gated out for free '
    '(implicit feature selection); an MLP feeds every feature into its first layer and, being '
    'rotationally invariant, has no gate to tell a useful axis from a useless one, so (per Ng 2004) it '
    'needs at least linearly more samples per junk feature and instead leaks capacity onto the noise, '
    'losing more than twice the accuracy the GBT loses and handing over its clean-data lead."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 027 — The uninformative-features bias: junk sinks the MLP, not the tree

**Lesson:** [`lessons/0027-inductive-bias-uninformative-features.html`](../lessons/0027-inductive-bias-uninformative-features.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 D&B ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) — §5.3 (Finding 2: MLP-like architectures are not robust to uninformative features). Theory backing: Andrew Ng 2004, *Feature selection, L1 vs L2 regularization, and rotational invariance* ([ICML 2004](https://icml.cc/Conferences/2004/proceedings/papers/354.pdf)).

**Dataset tier:** C — synthetic, to *isolate the mechanism* (per `lab-authoring`: Tier C is for mechanism isolation). No real dataset lets you dial the amount of pure noise up and down while holding the real signal fixed; a synthetic one does.

**Skill you are practising:** implement the **add-junk operator** (append `k` pure-noise columns to a fixed informative block), then show a tree/GBT is **robust** while an MLP's clean-data lead **erodes** as junk accumulates — and measure the **gate** (root-split gain ≈ 0 on junk) that makes the tree robust. That is the experiment that proves Finding 2.

**Exit criteria:** EXIT TICKET prints the MLP winning on clean data, the MLP losing more accuracy to junk than the GBT (flipping the ranking), and the tree's root-split gain being far higher on informative features than on junk.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, models, the fit/score harness); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** only (no network — the data is synthetic). The sweep fits a few models per junk level and takes ~30 seconds.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — implicit feature selection, and why an MLP lacks it

**Uninformative (junk) feature.** A column independent of the label — here a pure-noise Gaussian. Real tables are full of near-junk columns (surrogate IDs, unrelated measurements). The question: what does a model do with a pile of them?

**Split gain (recall L011).** A tree grows greedily: at each node it scores every candidate split by its **gain** — the Gini impurity it removes, `gain = parent − weighted children` — and keeps only the **highest-gain** split. A junk column's best split leaves the children as mixed as the parent, so its gain is ≈ 0.

**Implicit feature selection.** Because only the highest-gain feature is ever chosen, a junk column is simply never split on near the top of the tree. The tree *selects* the useful features as a side-effect of how it grows. That is why adding junk barely dents it.

**The MLP has no gate.** Every feature is wired into the first linear layer, and — because the MLP is **rotationally invariant** (L026: `W·(Qx)=(WQ)·x`) — it cannot cheaply tell a useful axis from a useless one. Ng (2004) proved the price: a rotation-invariant learner needs **at least linearly more training samples per uninformative feature**. With finite data, the leftover weight on junk directions is extra variance, so the MLP partly fits the noise and its accuracy falls.

**The experiment (§5.3).** *Add* junk → the tree–MLP gap widens (MLP hurt more). *Remove* junk → the gap shrinks (MLP helped more). We deliberately use a **smooth** target so the MLP WINS on clean data — that isolates *this* bias from smoothness (L025) and rotation (L026).

**Toy micro-example (not this lab's answer).** Gini of `[0,0,1,1]` is `1 − 0.5² − 0.5² = 0.5`; Gini of a pure node `[1,1,1,1]` is `0`. A split that removes impurity has positive gain; a split on noise has ≈ 0.

Full write-up + the add-junk curve and the gate viz: [Lesson 027](../lessons/0027-inductive-bias-uninformative-features.html).'''),
        md("## Setup — PROVIDED (synthetic informative block, models, fit/score harness)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — probe the bias

1. **Remove, don't add.** Start from `add_junk(Xi, 50, seed=0)` and use a Random Forest's
   `feature_importances_` to rank features; refit the MLP keeping only the top-`k`. Does removing junk
   help the MLP *more* than the GBT (the other direction of Finding 2)?
2. **How much data buys off the bias?** Ng's bound is about *sample complexity*. Re-run the `k=100`
   sweep with `n=20000` instead of `4000`. Does the MLP recover (does more data let it learn to ignore
   the junk)?
3. **A learned gate.** The flat-table fix is to let a neural model *learn to attend* to informative
   features. As a cheap proxy, put an L1-penalized `LogisticRegression` (`penalty="l1",
   solver="liblinear"`) next to the MLP: L1 is **not** rotationally invariant, so it should shrug off
   junk far better than the L2 MLP. Confirm it.
4. **MDI vs the gate.** Fit a Random Forest on the junk-augmented data and sum
   `feature_importances_` over the junk columns. Is it ≈ 0 (as the *gate* suggests) or surprisingly
   large (the known MDI-over-credits-noise artifact)? Why is root-split gain the honest measure?'''),
        code(r'''# STRETCH — ungraded.
# from sklearn.linear_model import LogisticRegression
# Xk = add_junk(Xi, 100, seed=100)
# Xs = StandardScaler().fit_transform(Xk)
# from sklearn.model_selection import train_test_split
# Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.4, random_state=0)
# l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.2).fit(Xtr, ytr)
# print("L1 logistic (NOT rotation-invariant) acc:", (l1.predict(Xte) == yte).mean())'''),
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
    with open(os.path.join(HERE, "0027-inductive-bias-uninformative-features.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0027-inductive-bias-uninformative-features.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0027-inductive-bias-uninformative-features.ipynb + solution")


if __name__ == "__main__":
    main()
