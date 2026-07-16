"""Verify Lesson 025 claims — the SMOOTHNESS inductive bias (Grinsztajn et al. 2022, §5.2).

Finding 1 of the paper: *neural networks are biased toward overly smooth (low-frequency)
solutions*. Tabular target functions are often irregular, so an MLP over-smooths them while a
tree (piecewise-constant) tracks the jags. Grinsztajn's test: **smooth the target function** on
the train set with a Gaussian kernel of growing length-scale. Smoothing removes exactly the
high-frequency structure the tree exploits — so it should *close the tree-vs-NN gap*.

Two things are produced here, both consumed by the lesson visuals:

  1. A 1-D mechanism demo (for assets/smoothness-viz.js): an irregular target on [0,1], a
     depth-limited tree fit (staircase) and an MLP fit (smooth), at three target-smoothing
     bandwidths. The tree tracks the jags at bandwidth 0; the MLP catches up only once the
     target is smoothed.

  2. The Grinsztajn gap experiment (for assets/smoothness-gap-viz.js + the lab reproduction
     target): a synthetic irregular multi-D regression; Gaussian-smooth the TARGET at growing
     length-scales; train a GBT and an MLP on the smoothed target and report test R^2. The gap
     GBT - MLP shrinks toward 0 as the target is smoothed.

Run: .venv/bin/python labs/_verify_l025.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SEED = 0


# ===========================================================================
# Gaussian kernel target smoothing (the paper's §5.2 operation).
# Smoothed target at point i = weighted average of neighbours' targets,
# weights ~ exp(-||x_i - x_j||^2 / (2 h^2)). h -> 0 gives the raw target;
# h -> large gives the constant mean. Distances are in standardized space.
# ===========================================================================
def gaussian_smooth_target(X, y, h):
    if h <= 0:
        return y.copy()
    Xs = StandardScaler().fit_transform(X)
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        d2 = np.sum((Xs - Xs[i]) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * h * h))
        out[i] = np.dot(w, y) / w.sum()
    return out


# ===========================================================================
# Part 1 — 1-D mechanism demo (staircase tree vs smooth MLP).
# ===========================================================================
def target_1d(x):
    # An irregular target: a low-frequency trend + a high-frequency wiggle + a step.
    return (np.sin(2 * np.pi * x)
            + 0.55 * np.sin(11 * np.pi * x)
            + 0.6 * (x > 0.55).astype(float))


def smooth_1d(x_grid, y_grid, h):
    if h <= 0:
        return y_grid.copy()
    out = np.empty_like(y_grid)
    for i in range(len(x_grid)):
        w = np.exp(-((x_grid - x_grid[i]) ** 2) / (2.0 * h * h))
        out[i] = np.dot(w, y_grid) / w.sum()
    return out


def part1_one_d():
    rng = np.random.default_rng(SEED)
    grid = np.linspace(0, 1, 81)
    f_grid = target_1d(grid)

    n = 300
    xtr = np.sort(rng.uniform(0, 1, n))
    ftr = target_1d(xtr)

    states = {"raw": 0.0, "medium": 0.045, "smooth": 0.12}
    result = {}
    for name, h in states.items():
        # smooth the (true) target, then sample noisy train labels from the smoothed target
        f_grid_h = smooth_1d(grid, f_grid, h)
        # smooth the training target the same way (via grid interpolation of the smoothed fn)
        ftr_h = np.interp(xtr, grid, f_grid_h)
        ytr = ftr_h + rng.normal(0, 0.12, n)

        tree = DecisionTreeRegressor(max_leaf_nodes=28, random_state=SEED).fit(xtr[:, None], ytr)
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128), alpha=1e-4,
                         learning_rate_init=5e-3, max_iter=4000,
                         random_state=SEED),
        ).fit(xtr[:, None], ytr)

        tree_grid = tree.predict(grid[:, None])
        mlp_grid = mlp.predict(grid[:, None])
        tree_mse = float(np.mean((tree_grid - f_grid_h) ** 2))
        mlp_mse = float(np.mean((mlp_grid - f_grid_h) ** 2))
        result[name] = dict(h=h, target=f_grid_h, tree=tree_grid, mlp=mlp_grid,
                            tree_mse=tree_mse, mlp_mse=mlp_mse)

    print("== Part 1: 1-D mechanism (MSE of each fit vs the smoothed target) ==")
    print(f"  {'state':>8} | {'h':>5} | {'tree MSE':>9} | {'MLP MSE':>9} | MLP/tree")
    for name, r in result.items():
        print(f"  {name:>8} | {r['h']:>5.3f} | {r['tree_mse']:>9.4f} | {r['mlp_mse']:>9.4f} | "
              f"{r['mlp_mse']/max(r['tree_mse'],1e-9):>6.2f}x")
    print("  reading: on the raw jagged target the MLP's error dwarfs the tree's; smoothing the")
    print("           target shrinks the MLP's disadvantage (its smoothness bias now matches).")

    grid_js = np.round(grid, 4).tolist()
    for name in states:
        r = result[name]
        print(f"\n  // {name} (h={r['h']})")
        print(f"  target_{name} = {np.round(r['target'],3).tolist()}")
        print(f"  tree_{name}   = {np.round(r['tree'],3).tolist()}")
        print(f"  mlp_{name}    = {np.round(r['mlp'],3).tolist()}")
    print(f"\n  grid = {grid_js}")
    return result


# ===========================================================================
# Part 2 — Grinsztajn §5.2 gap experiment on irregular multi-D data.
# ===========================================================================
def make_irregular_regression(n=3000, d=8, seed=SEED):
    """An irregular (non-smooth) target: thresholded interactions + a high-frequency term.
    This is the regime where trees beat MLPs; smoothing should erase that edge."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, d))
    # non-smooth: sign/threshold interactions + a high-frequency sinusoid on a feature
    y = (np.sign(X[:, 0] * X[:, 1])                 # XOR-like sign flip (sharp boundary)
         + (X[:, 2] > 0.3).astype(float) * 1.5      # step
         + np.sin(4.0 * X[:, 3])                     # high-frequency wiggle
         + 0.5 * X[:, 4])                            # a smooth linear part
    return X, y


def _one_gap_run(seed, scales):
    X, y = make_irregular_regression(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=seed)
    gbt_r, mlp_r, var_ret = [], [], []
    for h in scales:
        ytr_h = gaussian_smooth_target(Xtr, ytr, h)
        yte_h = gaussian_smooth_target(Xte, yte, h)
        gbt = HistGradientBoostingRegressor(random_state=seed).fit(Xtr, ytr_h)
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4,
                         learning_rate_init=3e-3, max_iter=2000, random_state=seed),
        ).fit(Xtr, ytr_h)
        gbt_r.append(r2_score(yte_h, gbt.predict(Xte)))
        mlp_r.append(r2_score(yte_h, mlp.predict(Xte)))
        var_ret.append(np.var(yte_h) / np.var(yte))
    return np.array(gbt_r), np.array(mlp_r), np.array(var_ret)


def part2_gap_curve():
    # scales chosen so the target keeps meaningful variance (not collapsed to its mean);
    # averaged over seeds so the trend is the mechanism, not sampling noise.
    scales = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    seeds = [0, 1, 2, 3, 4]
    G = np.zeros(len(scales)); M = np.zeros(len(scales)); V = np.zeros(len(scales))
    for s in seeds:
        g, m, v = _one_gap_run(s, scales)
        G += g; M += m; V += v
    G /= len(seeds); M /= len(seeds); V /= len(seeds)
    gbt_scores = [round(float(x), 4) for x in G]
    mlp_scores = [round(float(x), 4) for x in M]

    print("\n== Part 2: Grinsztajn §5.2 gap experiment (test R^2 vs target smoothing) ==")
    print(f"  (mean over {len(seeds)} seeds; irregular {8}-D regression)")
    print(f"  {'length-scale h':>14} | {'GBT R2':>7} | {'MLP R2':>7} | {'GBT-MLP':>7} | var kept")
    for i, h in enumerate(scales):
        print(f"  {h:>14.2f} | {G[i]:>7.3f} | {M[i]:>7.3f} | {G[i]-M[i]:>+7.3f} | {V[i]:>6.2f}")

    gap0 = gbt_scores[0] - mlp_scores[0]
    gapL = gbt_scores[-1] - mlp_scores[-1]
    print(f"\n  raw target (h=0): GBT beats MLP by {gap0:+.3f} R^2 (the tree exploits the jags)")
    print(f"  smoothed (h={scales[-1]}): gap {gapL:+.3f} — smoothing the target closes the gap")
    print(f"  scales     = {scales}")
    print(f"  gbt_scores = {gbt_scores}")
    print(f"  mlp_scores = {mlp_scores}")
    return scales, gbt_scores, mlp_scores


if __name__ == "__main__":
    part1_one_d()
    part2_gap_curve()
