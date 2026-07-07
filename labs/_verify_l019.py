"""Verify the three Grinsztajn-2022 inductive-bias demos for Lesson 019 (preview).

Tier-C synthetic (mechanism isolation): RandomForest (tree, axis-aligned, not rotation
invariant) vs MLP (NN, biased to smooth, ~rotation invariant). Metric: test accuracy.

Run:  .venv/bin/python labs/_verify_l019.py
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RS = 0
rng = np.random.RandomState(RS)


def tree():
    return RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=RS)


def nn():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-4,
                      max_iter=400, early_stopping=True, random_state=RS),
    )


def score(model_fn, Xtr, Xte, ytr, yte):
    m = model_fn().fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))


# ---------------------------------------------------------------- Bias 1: irregular functions
def checkerboard(n=6000, tiles=2, noise=0.0, seed=RS):
    r = np.random.RandomState(seed)
    X = r.uniform(0, 1, size=(n, 2))
    gx = np.floor(X[:, 0] * tiles).astype(int)
    gy = np.floor(X[:, 1] * tiles).astype(int)
    y = (gx + gy) % 2
    if noise:
        flip = r.uniform(size=n) < noise
        y = np.where(flip, 1 - y, y)
    return X, y


print("=== Bias 1: irregular (non-smooth) target — checkerboard, rising frequency ===")
print(f"{'tiles/side':>10} {'cells':>6} {'tree':>7} {'nn':>7} {'gap':>7}")
b1 = {}
for tiles in (2, 4, 8, 16):
    X, y = checkerboard(tiles=tiles)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RS, stratify=y)
    t = score(tree, Xtr, Xte, ytr, yte)
    n_ = score(nn, Xtr, Xte, ytr, yte)
    b1[tiles] = (t, n_)
    print(f"{tiles:>10} {tiles*tiles:>6} {t:>7.3f} {n_:>7.3f} {t-n_:>+7.3f}")


# ---------------------------------------------------------------- Bias 2: uninformative features
print("\n=== Bias 2: robustness to uninformative (noise) features ===")
Xb, yb = make_classification(n_samples=6000, n_features=10, n_informative=10,
                             n_redundant=0, n_repeated=0, n_clusters_per_class=2,
                             class_sep=1.0, flip_y=0.02, random_state=RS)
print(f"{'#noise':>7} {'total':>6} {'tree':>7} {'nn':>7} {'gap':>7}")
b2 = {}
for k in (0, 20, 50, 100):
    if k:
        Xj = np.hstack([Xb, rng.normal(size=(Xb.shape[0], k))])
    else:
        Xj = Xb
    Xtr, Xte, ytr, yte = train_test_split(Xj, yb, test_size=0.3, random_state=RS, stratify=yb)
    t = score(tree, Xtr, Xte, ytr, yte)
    n_ = score(nn, Xtr, Xte, ytr, yte)
    b2[k] = (t, n_)
    print(f"{k:>7} {10+k:>6} {t:>7.3f} {n_:>7.3f} {t-n_:>+7.3f}")


# ---------------------------------------------------------------- Bias 3: rotation
# Axis-aligned XOR on 2 features + 6 pure-noise features. Tree solves XOR with 2 splits;
# a random rotation turns the axis-aligned boundary into a diagonal the tree must staircase,
# while the (approximately rotation-invariant) MLP is unaffected.
print("\n=== Bias 3: preserve orientation (rotation (in)variance) — axis-aligned XOR ===")
d = 8
Xr = rng.normal(size=(6000, d))
yr = ((Xr[:, 0] > 0) ^ (Xr[:, 1] > 0)).astype(int)
A = rng.normal(size=(d, d))
Q, _ = np.linalg.qr(A)               # random orthogonal rotation
Xrot = Xr @ Q

print(f"{'variant':>10} {'tree':>7} {'nn':>7}")
b3 = {}
for label, Xv in (("original", Xr), ("rotated", Xrot)):
    Xtr, Xte, ytr, yte = train_test_split(Xv, yr, test_size=0.3, random_state=RS, stratify=yr)
    t = score(tree, Xtr, Xte, ytr, yte)
    n_ = score(nn, Xtr, Xte, ytr, yte)
    b3[label] = (t, n_)
    print(f"{label:>10} {t:>7.3f} {n_:>7.3f}")

print("\n--- summary for lesson/lab tolerances ---")
print("B1 tiles=16:", b1[16])
print("B2 k=100:", b2[100])
print("B3 tree orig->rot:", b3["original"][0], "->", b3["rotated"][0],
      f"(drop {b3['original'][0]-b3['rotated'][0]:+.3f})")
print("B3 nn   orig->rot:", b3["original"][1], "->", b3["rotated"][1],
      f"(drop {b3['rotated'][1]-b3['original'][1]:+.3f})")
