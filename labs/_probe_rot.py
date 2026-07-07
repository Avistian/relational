"""Probe: find an axis-aligned target where rotation clearly hurts the tree but not the NN."""
from __future__ import annotations
import warnings
import numpy as np
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
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=400,
                                       early_stopping=True, random_state=RS))


def run(name, X, y):
    A = rng.normal(size=(X.shape[1], X.shape[1]))
    Q, _ = np.linalg.qr(A)
    Xrot = X @ Q
    out = {}
    for lab, Xv in (("orig", X), ("rot", Xrot)):
        Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.3, random_state=RS, stratify=y)
        t = accuracy_score(yte, tree().fit(Xtr, ytr).predict(Xte))
        n_ = accuracy_score(yte, nn().fit(Xtr, ytr).predict(Xte))
        out[lab] = (t, n_)
    print(f"{name:26s} tree {out['orig'][0]:.3f}->{out['rot'][0]:.3f} "
          f"(d{out['orig'][0]-out['rot'][0]:+.3f})   "
          f"nn {out['orig'][1]:.3f}->{out['rot'][1]:.3f} (d{out['orig'][1]-out['rot'][1]:+.3f})")


n = 6000
# A: sum-of-thresholds (axis-aligned linear-ish)
d = 8
X = rng.normal(size=(n, d))
y = ((X > 0).sum(1) > d / 2).astype(int)
run("A sum-thresholds d=8", X, y)

# B: 2-feature XOR + noise dims
d = 8
X = rng.normal(size=(n, d))
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
run("B xor2 + 6 noise", X, y)

# C: 3-feature parity + noise
d = 8
X = rng.normal(size=(n, d))
y = (((X[:, 0] > 0).astype(int) + (X[:, 1] > 0) + (X[:, 2] > 0)) % 2).astype(int)
run("C parity3 + 5 noise", X, y)

# D: axis-aligned box rule (all first 4 features positive)
d = 8
X = rng.normal(size=(n, d))
y = ((X[:, :4] > 0).all(1)).astype(int)
run("D box-rule 4 + 4 noise", X, y)

# E: high-dim checkerboard on 4 features (2 tiles each -> parity of floor)
d = 6
X = rng.uniform(-1, 1, size=(n, d))
y = ((X[:, :4] > 0).sum(1) % 2).astype(int)
run("E parity4 d=6", X, y)

# F: sum-thresholds larger sep, fewer dims
d = 4
X = rng.normal(size=(n, d))
y = ((X > 0.3).sum(1) >= 2).astype(int)
run("F sum-thresh d=4", X, y)
