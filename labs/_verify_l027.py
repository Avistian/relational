"""Verify Lesson 027 claims — the UNINFORMATIVE-FEATURES inductive bias
(Grinsztajn et al. 2022, §5.3, Finding 2: "MLP-like architectures are not robust to
uninformative features").

The mechanism. A decision tree does *implicit feature selection*: at each node it splits on
the feature with the highest gain, so a pure-noise column — which by construction has ~no gain —
is almost never chosen. An MLP has no such gate: every input feeds the first linear layer, and
(because it is rotationally invariant, Ng 2004) it must spend samples learning to ignore each
junk direction. So adding uninformative features degrades an MLP much faster than a tree, and
removing them helps the MLP much more than the tree.

Three things are produced here, each consumed by a lesson visual and/or the lab:

  Part A  — ADD junk (the §5.3 Fig-a experiment). Fix the informative features; add k pure-noise
            columns. Fit tree/GBT/MLP; the tree holds, the MLP falls, the gap WIDENS with k.
            (Drives assets/uninformative-add-viz.js and the lab reproduction target.)

  Part B  — REMOVE junk (the §5.3 Fig-b experiment). Start from informative+junk, rank all
            features by a Random Forest's importance, keep the top-k; as junk is removed the MLP
            recovers more than the tree, so the gap SHRINKS. (Drives a lesson table.)

  Part C  — the MECHANISM (gating). On one dataset, compute the ROOT-SPLIT GAIN (Gini decrease of
            the best single threshold) for each feature: on the full data a junk column has ~0 gain,
            so the tree's greedy selection never picks it at the top — the tree is "anchored" by the
            informative features. Contrast the MLP's first-layer weight magnitude per input: it has
            no gate, so junk inputs receive comparable weight. (Honest caveat: trees DO make some
            spurious low-gain noise splits deep down — MDI overstates junk — but they barely move
            accuracy; the root-gain picture is the clean mechanism.) Drives
            assets/uninformative-mechanism-viz.js.

Run: .venv/bin/python labs/_verify_l027.py
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
D_INFORM = 5


def make_data(n=6000, d_inform=D_INFORM, d_noise=0, seed=SEED):
    """Informative features drive a smooth, mildly-nonlinear target that an MLP fits at least as
    well as a tree when the data is CLEAN — so any later divergence is attributable to the junk
    columns, not to smoothness/rotation. The noise columns are pure i.i.d. Gaussian: zero signal."""
    rng = np.random.default_rng(seed)
    Xi = rng.normal(size=(n, d_inform))
    # smooth additive-plus-mild-interaction score; both model families handle it well when clean
    s = (1.3 * Xi[:, 0] + 1.0 * Xi[:, 1] + 0.8 * Xi[:, 2]
         + 0.6 * Xi[:, 3] * Xi[:, 4])
    y = (s > np.median(s)).astype(int)
    if d_noise > 0:
        Xn = rng.normal(size=(n, d_noise))
    else:
        Xn = np.empty((n, 0))
    X = np.hstack([Xi, Xn]).astype(np.float64)
    return X, y


def fit_acc(Xtr, Xte, ytr, yte, seed=SEED):
    tree = DecisionTreeClassifier(max_leaf_nodes=64, random_state=seed).fit(Xtr, ytr)
    gbt = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4,
                      learning_rate_init=3e-3, max_iter=800, random_state=seed),
    ).fit(Xtr, ytr)
    return {
        "tree": accuracy_score(yte, tree.predict(Xte)),
        "gbt": accuracy_score(yte, gbt.predict(Xte)),
        "mlp": accuracy_score(yte, mlp.predict(Xte)),
    }


def part_a_add_junk():
    ks = [0, 5, 15, 30, 50, 100]
    seeds = [0, 1, 2, 3, 4]
    tree_c, gbt_c, mlp_c = [], [], []
    for k in ks:
        t, g, m = [], [], []
        for s in seeds:
            X, y = make_data(d_noise=k, seed=s)
            X = StandardScaler().fit_transform(X)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=s)
            a = fit_acc(Xtr, Xte, ytr, yte, s)
            t.append(a["tree"]); g.append(a["gbt"]); m.append(a["mlp"])
        tree_c.append(float(np.mean(t))); gbt_c.append(float(np.mean(g))); mlp_c.append(float(np.mean(m)))

    print("== Part A: ADD uninformative features (mean of 5 seeds; test accuracy) ==")
    print(f"  {D_INFORM} informative features fixed; add k pure-noise columns")
    print(f"  {'k_noise':>7} | {'tree':>6} | {'gbt':>6} | {'mlp':>6} | {'gbt-mlp':>8}")
    for i, k in enumerate(ks):
        print(f"  {k:>7} | {tree_c[i]:>6.3f} | {gbt_c[i]:>6.3f} | {mlp_c[i]:>6.3f} | {gbt_c[i]-mlp_c[i]:>+8.3f}")
    print(f"\n  gap (GBT-MLP) at k=0: {gbt_c[0]-mlp_c[0]:+.3f}   at k={ks[-1]}: {gbt_c[-1]-mlp_c[-1]:+.3f}")
    print(f"  MLP change from k=0 to k={ks[-1]}: {mlp_c[-1]-mlp_c[0]:+.3f}   "
          f"GBT change: {gbt_c[-1]-gbt_c[0]:+.3f}")
    print(f"\n  ks       = {ks}")
    print(f"  tree_acc = {[round(v,4) for v in tree_c]}")
    print(f"  gbt_acc  = {[round(v,4) for v in gbt_c]}")
    print(f"  mlp_acc  = {[round(v,4) for v in mlp_c]}")
    return ks, tree_c, gbt_c, mlp_c


def part_b_remove_junk():
    # one big dataset with lots of junk; rank features by RF importance, keep top-k
    keep = [5, 10, 20, 35, 55]
    seeds = [0, 1, 2, 3, 4]
    gbt_by_keep = {k: [] for k in keep}
    mlp_by_keep = {k: [] for k in keep}
    for s in seeds:
        X, y = make_data(d_noise=50, seed=s)  # 5 informative + 50 junk = 55
        X = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=s)
        rf = RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=-1).fit(Xtr, ytr)
        order = np.argsort(rf.feature_importances_)[::-1]  # most important first
        for k in keep:
            cols = order[:k]
            a = fit_acc(Xtr[:, cols], Xte[:, cols], ytr, yte, s)
            gbt_by_keep[k].append(a["gbt"]); mlp_by_keep[k].append(a["mlp"])

    print("\n== Part B: REMOVE uninformative features (rank by RF importance, keep top-k; 5 seeds) ==")
    print(f"  {'kept':>5} | {'gbt':>6} | {'mlp':>6} | {'gbt-mlp':>8}")
    gbt_m, mlp_m, gaps = [], [], []
    for k in keep:
        g = float(np.mean(gbt_by_keep[k])); m = float(np.mean(mlp_by_keep[k]))
        gbt_m.append(g); mlp_m.append(m); gaps.append(g - m)
        print(f"  {k:>5} | {g:>6.3f} | {m:>6.3f} | {g-m:>+8.3f}")
    print(f"\n  keep top-5 (junk removed) gap {gaps[0]:+.3f}  vs  keep all 55 gap {gaps[-1]:+.3f}")
    print(f"  keep     = {keep}")
    print(f"  gbt_acc  = {[round(v,4) for v in gbt_m]}")
    print(f"  mlp_acc  = {[round(v,4) for v in mlp_m]}")
    return keep, gbt_m, mlp_m


def _gini(y):
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y))
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def best_split_gain(col, y):
    """Best single-threshold Gini decrease for one feature on the full node (the 'root gain').
    Scans candidate thresholds at deciles for speed; gain = parent_gini - weighted_child_gini."""
    n = len(y)
    parent = _gini(y)
    order = np.argsort(col)
    cs = col[order]
    ys = y[order]
    best = 0.0
    # candidate thresholds at 5th..95th percentile positions
    for q in np.linspace(0.05, 0.95, 19):
        idx = int(q * n)
        if idx <= 0 or idx >= n:
            continue
        left, right = ys[:idx], ys[idx:]
        g = parent - (len(left) / n) * _gini(left) - (len(right) / n) * _gini(right)
        if g > best:
            best = g
    return best


def part_c_mechanism():
    d_noise = 20
    X, y = make_data(d_noise=d_noise, seed=SEED)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=SEED)
    inform = np.arange(D_INFORM)
    junk = np.arange(D_INFORM, D_INFORM + d_noise)

    # (1) tree gating: root-split gain per feature (junk ~ 0 -> greedy tree never picks it up top)
    gains = np.array([best_split_gain(Xtr[:, j], ytr) for j in range(Xtr.shape[1])])

    # (2) MLP has no gate: first-layer weight magnitude per input feature
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4,
                      learning_rate_init=3e-3, max_iter=800, random_state=SEED),
    ).fit(Xtr, ytr)
    W1 = mlp.named_steps["mlpclassifier"].coefs_[0]      # shape (n_features, 256)
    w_mag = np.abs(W1).mean(axis=1)                       # mean |weight| per input feature

    print(f"\n== Part C: gating mechanism — {D_INFORM} informative vs {d_noise} junk features ==")
    print(f"  tree ROOT-SPLIT GAIN (Gini decrease of best threshold, full data):")
    print(f"     informative mean {gains[inform].mean():.4f} (max {gains[inform].max():.4f})  |  "
          f"junk mean {gains[junk].mean():.4f} (max {gains[junk].max():.4f})")
    print(f"     ratio informative/junk root-gain: {gains[inform].mean()/max(gains[junk].mean(),1e-9):.1f}x")
    print(f"  MLP first-layer |weight| per input:")
    print(f"     informative mean {w_mag[inform].mean():.4f}  |  junk mean {w_mag[junk].mean():.4f}  "
          f"ratio {w_mag[inform].mean()/max(w_mag[junk].mean(),1e-9):.1f}x")
    print(f"\n  gain_inform = {[round(float(gains[j]),4) for j in inform]}")
    print(f"  gain_junk(mean over {d_noise}) = {round(float(gains[junk].mean()),4)}")
    print(f"  wmag_inform = {[round(float(w_mag[j]),4) for j in inform]}")
    print(f"  wmag_junk(mean over {d_noise}) = {round(float(w_mag[junk].mean()),4)}")
    return gains, w_mag, inform, junk


if __name__ == "__main__":
    part_a_add_junk()
    part_b_remove_junk()
    part_c_mechanism()
