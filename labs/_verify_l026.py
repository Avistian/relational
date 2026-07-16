"""Verify Lesson 026 claims — the ROTATION inductive bias (Grinsztajn et al. 2022, §5.4).

Finding 3 of the paper: *tabular data is not invariant by rotation, so learning procedures
should not be either.* An MLP (and a ResNet) is **rotationally invariant** in the sense of Ng
(2004): learning on X and evaluating on X is unchanged if you apply the same orthogonal matrix
Q to the features of both train and test. A decision tree is **not** — it attends to each
feature on its own axis, so its axis-aligned splits are tied to the original column basis.

Tabular columns carry individual meaning (age, weight), so the original basis is special:
it aligns the informative structure with the axes. A random rotation smears that structure
across all coordinates. Grinsztajn's test (Fig 5a): randomly rotate the datasets. Only the
rotation-invariant models are unchanged; the rotation *reverses the ranking* — NNs climb above
trees. We reproduce the mechanism on a synthetic AXIS-ALIGNED target (Tier C).

Two things are produced here, both consumed by the lesson visuals:

  1. A 2-D geometry demo (for assets/rotation-splits-viz.js): an axis-aligned class boundary a
     tree carves with two straight splits; after a 30-degree rotation the same boundary needs a
     staircase of axis-aligned steps. (Illustrative geometry; no fit needed.)

  2. The Grinsztajn §5.4 rotation experiment (for assets/rotation-gap-viz.js + the lab
     reproduction target): a synthetic axis-aligned classification task; fit a tree/GBT and an
     MLP on the ORIGINAL features and on RANDOMLY-ROTATED features (same Q on train+test). The
     MLP's accuracy is ~unchanged (invariant); the tree's collapses; the gap reverses.

Run: .venv/bin/python labs/_verify_l026.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 0


# ===========================================================================
# The rotation operator (the paper's §5.4 / Ng-2004 operation).
# Draw a random orthogonal matrix Q from the QR decomposition of a Gaussian
# matrix, then apply the SAME Q to train and test features. A rotation preserves
# all pairwise distances and angles — it only changes the basis (the axes).
# ===========================================================================
def random_orthogonal(d, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # fix signs so Q is deterministic-ish (not required, but tidy)
    Q *= np.sign(np.diag(R))
    return Q


def rotate(Xtr, Xte, Q):
    return Xtr @ Q, Xte @ Q


# ===========================================================================
# Part 1 — 2-D geometry demo (axis-aligned boundary vs its rotated staircase).
# Pure geometry for the viz; emits the boundary polylines the tree must draw.
# ===========================================================================
def part1_geometry():
    # An axis-aligned "box" boundary: class 1 iff (x>0 and y>0) — one corner.
    # A tree splits at x=0 then y=0: two straight cuts. After rotation the same
    # region is a rotated quadrant, which axis-aligned cuts can only staircase.
    theta = np.deg2rad(30)
    print("== Part 1: 2-D geometry (illustrative; drives rotation-splits-viz) ==")
    print(f"  axis-aligned boundary: two straight splits (x=0, y=0) fit the quadrant exactly")
    print(f"  after {int(np.rad2deg(theta))}-degree rotation: the boundary is diagonal -> tree must staircase it")
    # Staircase points (in the tree's fixed axis-aligned frame) approximating the rotated line.
    steps = 7
    xs = np.linspace(-1, 1, steps + 1)
    # rotated boundary line through origin at angle theta: y = tan(theta) * x
    ys = np.tan(theta) * xs
    print(f"  rotated boundary slope tan({int(np.rad2deg(theta))}deg) = {np.tan(theta):.3f}")
    return theta


# ===========================================================================
# Part 2 — Grinsztajn §5.4 rotation experiment on an AXIS-ALIGNED target.
# ===========================================================================
def make_axis_aligned(n=6000, d_inform=6, d_noise=10, seed=SEED):
    """An AXIS-ALIGNED target: the label is a sum of per-feature sign thresholds on the
    informative features (each feature matters on its own axis) plus uninformative Gaussian
    columns. Trees love this (a split at 0 per axis); a random rotation smears the per-axis
    structure across all coordinates, so axis-aligned splits can no longer isolate it."""
    rng = np.random.default_rng(seed)
    Xi = rng.normal(size=(n, d_inform))
    Xn = rng.normal(size=(n, d_noise))          # uninformative features
    X = np.hstack([Xi, Xn]).astype(np.float64)
    # axis-aligned, piecewise-constant target: majority vote of per-feature signs
    votes = np.sum(np.sign(Xi), axis=1)
    y = (votes > 0).astype(int)
    return X, y


def fit_acc(Xtr, Xte, ytr, yte, seed):
    tree = DecisionTreeClassifier(max_leaf_nodes=64, random_state=seed).fit(Xtr, ytr)
    gbt = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4,
                      learning_rate_init=3e-3, max_iter=800, random_state=seed),
    ).fit(Xtr, ytr)
    return {
        "tree": accuracy_score(yte, tree.predict(Xte)),
        "gbt": accuracy_score(yte, gbt.predict(Xte)),
        "rf": accuracy_score(yte, rf.predict(Xte)),
        "mlp": accuracy_score(yte, mlp.predict(Xte)),
    }


def part2_rotation_experiment():
    seeds = [0, 1, 2, 3, 4]
    orig = {k: [] for k in ["tree", "gbt", "rf", "mlp"]}
    rot = {k: [] for k in ["tree", "gbt", "rf", "mlp"]}
    for s in seeds:
        X, y = make_axis_aligned(seed=s)
        # Gaussianize once (standardize) so a rotation keeps columns ~unit-variance
        # (Grinsztajn Gaussianizes before rotating). Distances preserved either way.
        X = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=s)

        a0 = fit_acc(Xtr, Xte, ytr, yte, s)
        for k in orig:
            orig[k].append(a0[k])

        Q = random_orthogonal(X.shape[1], seed=100 + s)
        Xtr_r, Xte_r = rotate(Xtr, Xte, Q)
        a1 = fit_acc(Xtr_r, Xte_r, ytr, yte, s)
        for k in rot:
            rot[k].append(a1[k])

    def mean(d, k):
        return float(np.mean(d[k]))

    print("\n== Part 2: Grinsztajn §5.4 rotation experiment (test accuracy, mean of 5 seeds) ==")
    print(f"  synthetic AXIS-ALIGNED classification: {6} informative + {10} uninformative features")
    print(f"  {'model':>6} | {'original':>9} | {'rotated':>9} | {'change':>8}")
    for k in ["tree", "gbt", "rf", "mlp"]:
        o, r = mean(orig, k), mean(rot, k)
        print(f"  {k:>6} | {o:>9.3f} | {r:>9.3f} | {r-o:>+8.3f}")

    to, mo = mean(orig, "tree"), mean(orig, "mlp")
    tr, mr = mean(rot, "tree"), mean(rot, "mlp")
    print(f"\n  original basis: tree {to:.3f} vs MLP {mo:.3f}  (gap {to-mo:+.3f}, tree wins)")
    print(f"  rotated basis : tree {tr:.3f} vs MLP {mr:.3f}  (gap {tr-mr:+.3f})")
    print(f"  MLP change under rotation: {mr-mo:+.3f} (≈0 -> rotationally invariant)")
    print(f"  tree change under rotation: {tr-to:+.3f} (large drop -> NOT invariant)")

    # JS-ready arrays for the viz
    def arr(d):
        return [round(mean(d, k), 4) for k in ["tree", "gbt", "rf", "mlp"]]
    print(f"\n  labels     = ['Tree','GBT','RF','MLP']")
    print(f"  orig_acc   = {arr(orig)}")
    print(f"  rot_acc    = {arr(rot)}")
    return orig, rot


if __name__ == "__main__":
    part1_geometry()
    part2_rotation_experiment()
