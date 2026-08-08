"""Synthetic instance-wise feature-selection datasets used by TabNet's Table 1 / Fig. 5 (Lesson 043).

TabNet (Arik & Pfister 2019) evaluates its masks on "6 synthetic datasets from (Chen et al. 2018)",
describing Syn2 as depending on X3-X6 and Syn4 as depending on *either* X1-X2 *or* X3-X6 according to
X11. Those are the 11-dimensional Syn1-Syn6 variants (INVASE, Yoon et al. 2019) of the generators
defined in L2X (Chen et al. 2018, arXiv:1802.07814 §4.1):

    XOR              P(Y=1|X) ~ exp(X_1 X_2)
    orange skin      P(Y=1|X) ~ exp(sum_i X_i^2 - 4)
    nonlinear add.   P(Y=1|X) ~ exp(-10 sin(2 X_a) + 2|X_b| + X_c + exp(-X_d))

Tier C (synthetic) is the right tier here and matches the paper: these exist to isolate *which features
a model selects*, which real data cannot tell you because the ground-truth relevant set is unknown.

HONEST SCOPE: this reproduces the paper's **mask-reading claim** (does the aggregate mask land on the
truly-relevant features, and does instance-wise selection switch with X11?), not TabNet's exact Table 1
AUC values — those used a much larger training budget and 10k/10M-sample regimes.
"""
from __future__ import annotations

import numpy as np


def _bernoulli(logit, rng):
    p = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(len(p)) < p).astype(np.float32)


def make_syn2(n=4000, seed=0):
    """Syn2 — 'orange skin' on X3-X6. Relevant features are the SAME for every row (global).

    Returns (X, y, relevant_index_list).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 11)).astype(np.float32)
    logit = (X[:, 2:6] ** 2).sum(1) - 4.0
    return X, _bernoulli(logit, rng), [2, 3, 4, 5]


def make_syn4(n=4000, seed=0):
    """Syn4 — X11 switches which group matters: X1-X2 (XOR) when X11 < 0, else X3-X6 (orange skin).

    This is the dataset that separates *instance-wise* from *global* feature selection: no single fixed
    feature subset is right for all rows, so a global selector must waste capacity on both groups.

    Returns (X, y, relevant_per_row) where relevant_per_row is a boolean n x 11 mask of true relevance
    (the switch feature X11 itself counts as relevant — a model must read it to know where to look).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 11)).astype(np.float32)
    left = X[:, 10] < 0
    logit = np.where(left, X[:, 0] * X[:, 1], (X[:, 2:6] ** 2).sum(1) - 4.0)
    y = _bernoulli(logit, rng)

    rel = np.zeros((n, 11), dtype=bool)
    rel[:, 10] = True                 # X11 is always needed: it selects the group
    rel[left, 0] = rel[left, 1] = True
    rel[~left, 2] = rel[~left, 3] = rel[~left, 4] = rel[~left, 5] = True
    return X, y, rel
