"""Build Lab 026 (the rotation inductive bias) — emits the blank student notebook and a filled
solution notebook. Tier-C mechanism-isolation lab (synthetic axis-aligned classification);
mirrors the L021/L023/L025 concept-lab structure.

Crucial fragment: implement the RANDOM-ROTATION operator (Grinsztajn §5.4 / Ng 2004) — draw an
orthogonal Q from a QR decomposition and apply the SAME Q to train + test — then show a tree
collapses under rotation while the MLP is invariant, reversing the ranking.

Run: .venv/bin/python labs/_build_l026.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0026-inductive-bias-rotation.ipynb labs/solutions/0026-inductive-bias-rotation.ipynb
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
SETUP = r'''# PROVIDED — synthetic AXIS-ALIGNED data, models, and the fit/score harness. Just run.
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

def make_axis_aligned(n=4000, d_inform=6, d_noise=10, seed=SEED):
    """An AXIS-ALIGNED target: the label is the majority vote of the SIGNS of the informative
    features, so each informative feature matters on its own axis (trees' home turf). The noise
    columns are uninformative. A rotation will smear the per-axis structure across all coordinates."""
    rng = np.random.default_rng(seed)
    Xi = rng.normal(size=(n, d_inform))
    Xn = rng.normal(size=(n, d_noise))          # uninformative features
    X = np.hstack([Xi, Xn]).astype(np.float64)
    y = (np.sum(np.sign(Xi), axis=1) > 0).astype(int)
    return X, y

# Gaussianize once (standardize) so a rotation keeps each column ~unit-variance
# (Grinsztajn Gaussianizes before rotating); distances are preserved either way.
X, y = make_axis_aligned()
X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=SEED)
print(f"synthetic axis-aligned classification: n={len(X)}, d={X.shape[1]} "
      f"(6 informative + 10 uninformative), train {len(Xtr)} / test {len(Xte)}")

def fit_acc(Xtr_, Xte_):
    """Fit a tree, a GBT, and an MLP; return their test accuracy. The tree-based models are
    axis-aligned; the MLP (linear first layer) is rotationally invariant."""
    tree = DecisionTreeClassifier(max_leaf_nodes=64, random_state=SEED).fit(Xtr_, ytr)
    gbt = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr_, ytr)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4,
                      learning_rate_init=3e-3, max_iter=800, random_state=SEED),
    ).fit(Xtr_, ytr)
    return {"tree": accuracy_score(yte, tree.predict(Xte_)),
            "gbt":  accuracy_score(yte, gbt.predict(Xte_)),
            "mlp":  accuracy_score(yte, mlp.predict(Xte_))}

print("ready — harness fit_acc(Xtr_, Xte_) -> {'tree':..., 'gbt':..., 'mlp':...}")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — the random-rotation operator — TODO (crucial fragment)

**Goal:** implement `random_rotation(d, seed)` — draw a random **orthogonal** matrix `Q` (a rotation)
and `rotate(Xtr, Xte, Q)` — apply the **same** `Q` to train and test. A rotation preserves every
distance and angle; it only spins the axes.

The clean way to get a random orthogonal matrix is the **QR decomposition** of a random Gaussian
matrix `A = QR`: `Q` is orthogonal by construction ($Q^\top Q = I$).

**Why it matters:** this is the scalpel of the whole experiment. Rotating is *lossless* (Q is
invertible), so any accuracy change is purely a model failing to use the **still-present** signal in a
scrambled basis. That is what isolates a *bias* rather than a change in difficulty.

**You implement:** (1) the QR decomposition of `A` (returns `Q, R`), and (2) applying `Q` to `Xtr`
and `Xte` by matrix multiplication.

**Hint boundary:** `np.linalg.qr(A)` returns `(Q, R)`; a rotated matrix is `X @ Q`. Apply the *same*
`Q` to both splits.

**Toy check (not this lab's answer):** for any orthogonal `Q`, `Q.T @ Q` is the identity, and
`‖xQ − x'Q‖ = ‖x − x'‖` — distances survive the spin.'''

T1_CODE = r'''# TODO — implement the random-rotation operator (the crucial fragment)
def random_rotation(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))               # a random square matrix
    Q, R = ____                               # QR decomposition -> Q is orthogonal
    return Q

def rotate(Xtr_, Xte_, Q):
    Xtr_rot = ____                            # apply the SAME Q to train
    Xte_rot = ____                            # ...and to test
    return Xtr_rot, Xte_rot

# sanity: Q is orthogonal, and rotation preserves pairwise distances
Q = random_rotation(Xtr.shape[1], seed=100)
print(f"Q orthogonal (QᵀQ = I)? {np.allclose(Q.T @ Q, np.eye(Q.shape[0]))}")
Xtr_r, Xte_r = rotate(Xtr, Xte, Q)
d_before = np.linalg.norm(Xtr[0] - Xtr[1])
d_after = np.linalg.norm(Xtr_r[0] - Xtr_r[1])
print(f"distance between rows 0,1 before {d_before:.4f} vs after {d_after:.4f} "
      f"(preserved? {np.isclose(d_before, d_after)})")'''

T1_SOL = (T1_CODE
    .replace('    Q, R = ____                               # QR decomposition -> Q is orthogonal',
             '    Q, R = np.linalg.qr(A)                    # QR decomposition -> Q is orthogonal')
    .replace('    Xtr_rot = ____                            # apply the SAME Q to train',
             '    Xtr_rot = Xtr_ @ Q                        # apply the SAME Q to train')
    .replace('    Xte_rot = ____                            # ...and to test',
             '    Xte_rot = Xte_ @ Q                        # ...and to test'))

T1_CHECK = r'''# CHECK — do not edit
Q = random_rotation(Xtr.shape[1], seed=100)
assert np.allclose(Q.T @ Q, np.eye(Q.shape[0]), atol=1e-8), "Q must be orthogonal (QᵀQ = I)."
Xtr_r, Xte_r = rotate(Xtr, Xte, Q)
assert Xtr_r.shape == Xtr.shape and Xte_r.shape == Xte.shape, "Rotation must not change shape."
# rotation preserves all pairwise distances (lossless)
import numpy.linalg as la
assert np.isclose(la.norm(Xtr[0] - Xtr[5]), la.norm(Xtr_r[0] - Xtr_r[5])), "Rotation must preserve distances."
assert not np.allclose(Xtr_r, Xtr), "A real rotation should actually change the coordinates."
print("Task 1 ok — Q is orthogonal, rotation preserves distances (lossless) but spins the axes. "
      "That is Grinsztajn's rotation scalpel.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — fit on the original vs rotated basis — TODO

**Goal:** fit the models on the **original** features and on a **randomly-rotated** copy (same `Q` on
train + test), and record the tree-vs-MLP gap in each basis.

**Why it matters:** this is Grinsztajn's Finding 3 in one comparison. Tree-based models are
axis-aligned (not invariant); the MLP's linear first layer can absorb the rotation (invariant). If the
tree's edge is its use of the meaningful axes, then a rotation should collapse it while the MLP holds.

**You implement:** rotate the data with your `rotate`, then compute the gap `tree − mlp` on each basis.

**Hint boundary:** `acc_orig = fit_acc(Xtr, Xte)` is done; call `fit_acc` on the rotated splits, and a
gap is `acc[...]['tree'] - acc[...]['mlp']`.'''

T2_CODE = r'''# TODO — fit on original vs rotated features, record the gap (~15s)
acc_orig = fit_acc(Xtr, Xte)

Q = random_rotation(Xtr.shape[1], seed=100)
Xtr_r, Xte_r = ____                           # rotate BOTH splits with the same Q
acc_rot = fit_acc(Xtr_r, Xte_r)

gap_orig = ____                               # tree - mlp on the original basis
gap_rot = ____                                # tree - mlp on the rotated basis

print(f"{'model':>6} | {'original':>9} | {'rotated':>9} | {'change':>8}")
for k in ["tree", "gbt", "mlp"]:
    print(f"{k:>6} | {acc_orig[k]:>9.3f} | {acc_rot[k]:>9.3f} | {acc_rot[k]-acc_orig[k]:>+8.3f}")
print(f"\ntree-vs-MLP gap: original {gap_orig:+.3f}  ->  rotated {gap_rot:+.3f}")'''

T2_SOL = (T2_CODE
    .replace("Xtr_r, Xte_r = ____                           # rotate BOTH splits with the same Q",
             "Xtr_r, Xte_r = rotate(Xtr, Xte, Q)            # rotate BOTH splits with the same Q")
    .replace("gap_orig = ____                               # tree - mlp on the original basis",
             "gap_orig = acc_orig['tree'] - acc_orig['mlp'] # tree - mlp on the original basis")
    .replace("gap_rot = ____                                # tree - mlp on the rotated basis",
             "gap_rot = acc_rot['tree'] - acc_rot['mlp']    # tree - mlp on the rotated basis"))

T2_CHECK = r'''# CHECK — do not edit
assert gap_orig > 0.05, "On the ORIGINAL axis-aligned basis the tree should clearly beat the MLP."
assert gap_rot < gap_orig - 0.15, "A rotation should erase most of the tree's advantage."
assert (acc_orig['tree'] - acc_rot['tree']) > 0.1, "The tree should collapse under rotation (not invariant)."
print(f"Task 2 ok — tree-vs-MLP gap {gap_orig:+.3f} (original) -> {gap_rot:+.3f} (rotated). "
      f"The tree's edge lived in the alignment of the signal with the axes.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — read the mechanism: invariance + reversal — TODO

**Goal:** show the two signatures of the rotation bias: the **MLP is (near-)invariant** (its accuracy
barely moves) while the **tree collapses**, and the ranking **reverses** sign.

**Why it matters:** a rotation is lossless, so the MLP being unmoved (invariance) *and* the tree
dropping is the fingerprint that the tree relied on the axis alignment — exactly Ng's (2004) point that
a rotation-invariant learner ignores orientation. That is what licenses "rotation is *the* mechanism".

**You implement:** the MLP's change and the tree's change under rotation.

**Hint boundary:** a change is `acc_rot[k] - acc_orig[k]`. The MLP change should be ≈ 0 (invariant);
the tree change should be a large negative number.'''

T3_CODE = r'''# TODO — quantify invariance (MLP) vs collapse (tree)
mlp_change = ____                             # acc_rot['mlp'] - acc_orig['mlp']
tree_change = ____                            # acc_rot['tree'] - acc_orig['tree']

print(f"MLP change under rotation : {mlp_change:+.3f}   (≈ 0 -> rotationally INVARIANT)")
print(f"tree change under rotation: {tree_change:+.3f}   (large drop -> NOT invariant)")
print(f"ranking reversed? original tree{'>' if gap_orig>0 else '<'}mlp, "
      f"rotated tree{'>' if gap_rot>0 else '<'}mlp -> {'YES' if (gap_orig>0)!=(gap_rot>0) else 'no'}")'''

T3_SOL = (T3_CODE
    .replace("mlp_change = ____                             # acc_rot['mlp'] - acc_orig['mlp']",
             "mlp_change = acc_rot['mlp'] - acc_orig['mlp'] # MLP is invariant")
    .replace("tree_change = ____                            # acc_rot['tree'] - acc_orig['tree']",
             "tree_change = acc_rot['tree'] - acc_orig['tree'] # tree is not"))

T3_CHECK = r'''# CHECK — do not edit
assert abs(mlp_change) < 0.05, "The MLP should be ~invariant to rotation (|change| small)."
assert tree_change < -0.1, "The tree should lose a lot under rotation."
assert (gap_orig > 0) != (gap_rot > 0), "The tree-vs-MLP ranking should reverse sign under rotation."
print(f"Task 3 ok — MLP moved {mlp_change:+.3f} (invariant) while the tree lost {tree_change:+.3f}; "
      f"the ranking reversed. Tabular data is NOT rotation-invariant, so a good learner should not be either.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why does randomly rotating the feature space collapse the tree
but not the MLP, and what does that prove about *why* trees win on typical tabular data?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 026 ===")
print(f"experiment : apply the same random orthogonal Q to train+test features; refit tree/GBT/MLP")
print(f"original   : tree {acc_orig['tree']:.3f} vs MLP {acc_orig['mlp']:.3f}  (gap {gap_orig:+.3f}, tree wins)")
print(f"rotated    : tree {acc_rot['tree']:.3f} vs MLP {acc_rot['mlp']:.3f}  (gap {gap_rot:+.3f})")
print(f"invariance : MLP moved {acc_rot['mlp']-acc_orig['mlp']:+.3f}, tree moved {acc_rot['tree']-acc_orig['tree']:+.3f}")
print(f"verdict    : the ranking reversed — rotation destroyed the axis-alignment the tree exploited")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A decision tree splits on one feature at a time at axis-aligned thresholds, so it exploits the '
    'fact that tabular columns are individually meaningful; an MLP has a linear first layer and is '
    'rotationally invariant, so a random rotation — which is lossless but scrambles the axes — leaves '
    'the MLP unmoved while it destroys the tree\'s axis alignment, reversing the ranking; this proves '
    'trees win on typical tabular data because the original column basis carries the signal, not because '
    'trees are simply stronger models."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 026 — The rotation inductive bias: a random rotation reverses the ranking

**Lesson:** [`lessons/0026-inductive-bias-rotation.html`](../lessons/0026-inductive-bias-rotation.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022 D&B ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) — §5.4 (Finding 3: data are non-invariant by rotation, so learning procedures should not be either). Theory backing: Andrew Ng 2004, *Feature selection, L1 vs L2 regularization, and rotational invariance* ([ICML 2004](https://icml.cc/Conferences/2004/proceedings/papers/354.pdf)).

**Dataset tier:** C — synthetic axis-aligned classification, to *isolate the mechanism* (per `lab-authoring`: Tier C is for mechanism isolation). No real dataset lets you rotate its own feature basis on and off while holding the information content fixed; a synthetic one does.

**Skill you are practising:** implement the **random-rotation operator** (an orthogonal `Q` from a QR decomposition, applied to train + test), then show a tree **collapses** under rotation while the MLP is **invariant** — reversing the ranking. That is the experiment that proves the rotation bias.

**Exit criteria:** EXIT TICKET prints the tree beating the MLP on the original basis (gap > +0.05), the ranking reversing under a random rotation, the MLP ~unchanged (|change| < 0.05), and the tree dropping > 0.1.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, models, the fit/score harness); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** only (no network — the data is synthetic). The comparison fits ~6 models and takes ~15 seconds.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — rotational invariance, and why a tree lacks it

**Rotate the axes.** Multiply your feature matrix by a random **orthogonal** matrix `Q` (a rotation: $Q^\top Q = I$, so it preserves every distance and angle) and apply the *same* `Q` to train and test. A learning procedure is **rotationally invariant** (Ng 2004) if training + evaluating on `X` gives the *same* score as on `X·Q`.

**MLPs are invariant; trees are not.** An MLP's first layer is a full linear map: $W(Qx) = (WQ)x$, so gradient descent can absorb any rotation into its weights and reach an equivalent solution (a ResNet too). A decision tree splits on **one feature at a time** at an axis-aligned threshold, so `feature 3 > 0.5` is meaningless once the axes are spun — its whole vocabulary is tied to the original basis. (FT-Transformers are also *not* invariant, because their per-feature tokenizer is a pointwise operation.)

**Why invariance is a liability here.** Tabular columns carry individual meaning (age, balance), so the original basis is a **privileged natural basis** that aligns the signal with the axes. A rotation blends meaningful columns into arbitrary mixtures. Ng (2004) proved the sharp version: any rotation-invariant learner has worst-case sample complexity growing **at least linearly in the number of uninformative features**, because it must first recover the lost orientation before it can drop junk columns.

**The experiment (§5.4).** Randomly rotate the features and refit. Only rotation-invariant models are unchanged; the rotation can **reverse** the ranking (trees fall below MLPs). You will implement the rotation and watch it happen.

**Toy micro-example (not this lab's answer).** For any orthogonal `Q` and two points `x, x'`: `‖xQ − x'Q‖ = ‖x − x'‖`. Distances survive, so *no information is lost* — only the alignment with the axes changes.

Full write-up + the geometry widget and the ranking-reversal bar chart: [Lesson 026](../lessons/0026-inductive-bias-rotation.html).'''),
        md("## Setup — PROVIDED (synthetic axis-aligned data, models, fit/score harness)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — probe the bias

1. **Break the invariance with an embedding.** Grinsztajn notes that per-feature embeddings (FT-Transformer, SAINT) *break* rotation invariance and recover much of the gap. Approximate this cheaply: bin each feature into quantiles (`KBinsDiscretizer`) *before* rotating. Does the MLP now also drop under rotation (i.e. did the discretization tie it to the axes)?
2. **A rotation-friendly target.** Replace the axis-aligned label with a target that depends on a *linear combination* of features (e.g. `y = (X[:, :6].sum(axis=1) > 0)`). Now the signal is not axis-aligned — does the tree's original advantage shrink, and does rotation hurt it less?
3. **Where invariance is a virtue.** Rotation invariance is *good* on images (a rotated cat is still a cat). Why is tabular data the opposite regime? Write one sentence relating it to "columns carry individual meaning."
4. **How invariant, exactly?** Fit the raw `MLPClassifier` (no `StandardScaler`) on original vs rotated. Is the change even smaller than with the scaler? (The per-feature scaler is what makes the invariance only approximate.)'''),
        code(r'''# STRETCH — ungraded.
# from sklearn.preprocessing import KBinsDiscretizer
# # idea 1: discretize before rotating and see if the MLP loses invariance
# Xd = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile").fit_transform(X)
# Xtr_d, Xte_d, _, _ = train_test_split(Xd, y, test_size=0.4, random_state=SEED)
# Q = random_rotation(Xtr_d.shape[1], seed=100)
# print("binned original:", fit_acc(Xtr_d, Xte_d))
# print("binned rotated :", fit_acc(*rotate(Xtr_d, Xte_d, Q)))'''),
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
    with open(os.path.join(HERE, "0026-inductive-bias-rotation.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0026-inductive-bias-rotation.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0026-inductive-bias-rotation.ipynb + labs/solutions/0026-inductive-bias-rotation.ipynb")


if __name__ == "__main__":
    main()
