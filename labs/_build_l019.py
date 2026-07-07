"""Build labs/0019-when-trees-win.ipynb (student, blanks) and solutions/ (filled).

Run:  python labs/_build_l019.py
Then execute the solution to verify:  jupyter nbconvert --execute ...
"""
from __future__ import annotations

import json
from pathlib import Path

LABS = Path(__file__).resolve().parent
SOL = LABS / "solutions"
SOL.mkdir(exist_ok=True)

CELLS = []


def md(src):
    CELLS.append(("md", src))


def code(src):
    CELLS.append(("code", src))


def todo(sol, stu):
    CELLS.append(("todo", sol, stu))


# ---------------------------------------------------------------- header
md("""# Lab 019 — When trees win: see the three inductive biases move

**Lesson:** [`lessons/0019-when-trees-win.html`](../lessons/0019-when-trees-win.html) · **Phase / Year:** Year 1 · Q2

**Paper (preview):** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep learning on typical tabular data?*, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) — abstract + §1 (the three challenges); §5 (the controlled experiments you reproduce in miniature here).

**Dataset tier:** C — synthetic (mechanism isolation). Each experiment turns exactly one knob so a single bias is visible in isolation; this is *not* a benchmark.

**Skill you are practising:** construct the three controlled experiments (an irregular target, a pile of uninformative features, a random rotation) and read off *why* a tree beats a plain neural net on each — then state the three tree strengths in your own words.

**Exit criteria:** EXIT TICKET prints (1) tree vs MLP accuracy across checkerboard frequencies, (2) the tree/MLP gap as noise features grow, (3) tree vs MLP before/after a random rotation, plus your one-line-each statement of the three biases.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** and **numpy** only (no boosters here). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — three biases, one knob each

Every learner has an **inductive bias** — the kind of function it reaches for first. A neural net (MLP) is biased toward **smooth** functions; a decision tree toward **axis-aligned, piecewise-constant** ones. Neither is universally better — what matters is whether the bias *matches the data*. Grinsztajn et al. (2022) found tree-based models still beat deep nets on typical ~10K-row tabular data, and isolated three reasons. This lab reproduces each in miniature with a **RandomForest** (the tree) vs a **2×256 MLP** (the net), scored by test accuracy.

**1. Irregular targets.** Tabular targets often jump at thresholds (a fee at 30 days late). A tree places a hard step anywhere; a smooth-biased net rounds it off. We make the target progressively rougher with a **checkerboard**: split each axis into $t$ tiles and label a point by the parity of its tile,
$$y = \left(\lfloor t\,x_0 \rfloor + \lfloor t\,x_1 \rfloor\right) \bmod 2.$$
Bigger $t$ = finer, more irregular target.

**2. Uninformative features.** Real tables carry many junk columns. A tree does **implicit feature selection** (it never splits on noise); an MLP feeds *every* column into every neuron. We start from an informative task and **append pure-noise columns**, watching the MLP degrade faster.

**3. Orientation.** Each tabular column means something on its own, so respecting the axes helps. A tree splits one feature at a time → **not rotation-invariant**. An MLP is ~rotation-invariant → it can't tell a meaningful column from a scrambled mixture. We apply a **random orthogonal rotation** $Q$ (so $X' = XQ$) to an axis-aligned XOR target and watch only the tree collapse.

**Toy micro-example (not this lab's answer).** Target $y = \mathbb{1}[x > 3]$ on $x \in \{1,2,3,4,5\}$ → labels $0,0,0,1,1$. A tree needs *one* split at $x=3.5$ to be perfect. A smooth line through those points must slope gently and will be wrong near the jump — the smoothness bias in one picture.

Full write-up + the interactive checkerboard/rotation widget: [Lesson 019](../lessons/0019-when-trees-win.html).""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED — models + a scorer. Same RandomForest (tree) and MLP (net) throughout.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

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

def score(model_fn, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RS, stratify=y)
    m = model_fn().fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))

print("tree + nn + scorer ready")""")

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — irregular targets: the checkerboard — TODO (crucial fragment)

**Goal:** build a checkerboard target of frequency $t$ and score the tree vs the MLP as $t$ rises.

**Why it matters:** this is Grinsztajn's smoothness finding in one experiment — as the target gets rougher, the smooth-biased net falls behind while the tree keeps carving cells.

**You implement:** the checkerboard label. For points `X` in the unit square and `t` tiles per side, a point's tile index on an axis is `np.floor(t * X[:, axis])`; the label is the **parity** (mod 2) of the sum of the two tile indices. Cast to `int`.

**Hint boundary:** use `np.floor`, integer cast, and `% 2`. Do not change the loop or the models.""")

todo(
    """# TODO — implement the checkerboard label
def checkerboard(t, n=6000, seed=RS):
    r = np.random.RandomState(seed)
    X = r.uniform(0, 1, size=(n, 2))
    y = ((np.floor(t * X[:, 0]) + np.floor(t * X[:, 1])).astype(int)) % 2
    return X, y

irr = {}
for t in (2, 4, 8, 16):
    X, y = checkerboard(t)
    irr[t] = (score(tree, X, y), score(nn, X, y))
    print(f"  {t:2d}x{t:<2d} ({t*t:3d} cells): tree {irr[t][0]:.3f}  nn {irr[t][1]:.3f}  gap {irr[t][0]-irr[t][1]:+.3f}")""",
    """# TODO — implement the checkerboard label
def checkerboard(t, n=6000, seed=RS):
    r = np.random.RandomState(seed)
    X = r.uniform(0, 1, size=(n, 2))
    y = ____            # parity (mod 2) of the sum of the two tile indices floor(t*x); cast to int
    return X, y

irr = {}
for t in (2, 4, 8, 16):
    X, y = checkerboard(t)
    irr[t] = (score(tree, X, y), score(nn, X, y))
    print(f"  {t:2d}x{t:<2d} ({t*t:3d} cells): tree {irr[t][0]:.3f}  nn {irr[t][1]:.3f}  gap {irr[t][0]-irr[t][1]:+.3f}")""",
)

code("""# CHECK — do not edit
assert set(np.unique(list(checkerboard(4)[1]))) == {0, 1}, "Checkerboard labels must be 0/1."
assert irr[2][0] > 0.95 and irr[2][1] > 0.95, "At 2x2 both models should be near-perfect."
assert irr[8][0] - irr[8][1] > 0.05, "By 8x8 the tree should clearly lead the smooth-biased net."
assert (irr[16][0] - irr[16][1]) > (irr[2][0] - irr[2][1]), "The gap should GROW as the target gets more irregular."
print(f"Task 1 ok — gap widens {irr[2][0]-irr[2][1]:+.3f} (2x2) -> {irr[16][0]-irr[16][1]:+.3f} (16x16). "
      f"Irregular targets favour the tree.")""")

# ---------------------------------------------------------------- Task 2
md(r"""## Task 2 — robustness to uninformative features — TODO (crucial fragment)

**Goal:** start from an informative 10-feature task and append growing numbers of pure-noise columns; compare how fast the tree vs the MLP degrades.

**Why it matters:** a tree ignores junk (implicit feature selection); an MLP does not. This is why trees shine on wide real tables full of weak/dead columns — *but note the honest twist in row `k=0`.*

**You implement:** append `k` columns of standard Gaussian noise to the base matrix `Xb`. Use the PROVIDED `rng` so it is reproducible, and `np.hstack` to concatenate.

**Hint boundary:** `rng.normal(size=(n_rows, k))` makes the junk; `np.hstack([...])` glues it to `Xb`. When `k == 0`, just use `Xb` unchanged.""")

code("""# PROVIDED — one fixed informative task; we only add junk columns to it.
Xb, yb = make_classification(n_samples=6000, n_features=10, n_informative=10,
                             n_redundant=0, n_repeated=0, n_clusters_per_class=2,
                             class_sep=1.0, flip_y=0.02, random_state=RS)
print("base task:", Xb.shape)""")

todo(
    """# TODO — add k noise features, then score both models
def add_noise(k):
    if k == 0:
        return Xb
    return np.hstack([Xb, rng.normal(size=(Xb.shape[0], k))])

junk = {}
for k in (0, 20, 50, 100):
    Xj = add_noise(k)
    junk[k] = (score(tree, Xj, yb), score(nn, Xj, yb))
    print(f"  +{k:3d} noise ({10+k:3d} feats): tree {junk[k][0]:.3f}  nn {junk[k][1]:.3f}  gap {junk[k][0]-junk[k][1]:+.3f}")""",
    """# TODO — add k noise features, then score both models
def add_noise(k):
    if k == 0:
        return Xb
    return ____        # np.hstack Xb with an (n_rows x k) block of rng.normal noise

junk = {}
for k in (0, 20, 50, 100):
    Xj = add_noise(k)
    junk[k] = (score(tree, Xj, yb), score(nn, Xj, yb))
    print(f"  +{k:3d} noise ({10+k:3d} feats): tree {junk[k][0]:.3f}  nn {junk[k][1]:.3f}  gap {junk[k][0]-junk[k][1]:+.3f}")""",
)

code("""# CHECK — do not edit
assert add_noise(20).shape == (Xb.shape[0], 30), "add_noise(20) should give 10 + 20 = 30 columns."
assert junk[0][1] >= junk[0][0] - 0.005, "Honest twist: with no junk the MLP should be competitive/better."
assert (junk[100][0] - junk[100][1]) > (junk[0][0] - junk[0][1]), "The tree should get RELATIVELY better as junk grows."
nn_drop = junk[0][1] - junk[100][1]
tree_drop = junk[0][0] - junk[100][0]
assert nn_drop > tree_drop, "The MLP should lose more accuracy to junk than the tree does."
print(f"Task 2 ok — junk cost the MLP {nn_drop:.3f} but the tree only {tree_drop:.3f}. "
      f"Trees are robust to uninformative features.")""")

# ---------------------------------------------------------------- Task 3
md(r"""## Task 3 — orientation: a random rotation — TODO (crucial fragment)

**Goal:** take an axis-aligned XOR target, apply a **random orthogonal rotation**, and show the tree collapses while the (rotation-invariant) MLP does not.

**Why it matters:** this is the sharpest evidence in the paper. A tree exploits the fact that individual columns are meaningful (axis-aligned splits); rotating mixes the columns into meaningless combinations, and the tree's edge vanishes. An MLP is ~rotation-invariant, so it is unmoved — which on tables is a *weakness*, not a virtue.

**You implement:** a random orthogonal matrix `Q` and the rotated design `Xrot = Xr @ Q`. Get an orthogonal `Q` from the **QR decomposition** of a random Gaussian matrix: `np.linalg.qr(rng.normal(size=(d, d)))` returns `(Q, R)`.

**Hint boundary:** build a `(d, d)` Gaussian matrix with `rng.normal`, take `Q` from `np.linalg.qr(...)`, then matrix-multiply `Xr @ Q`. `d` is the number of columns of `Xr`.""")

code("""# PROVIDED — axis-aligned XOR on 2 features + 6 pure-noise features.
d = 8
Xr = rng.normal(size=(6000, d))
yr = ((Xr[:, 0] > 0) ^ (Xr[:, 1] > 0)).astype(int)   # XOR of two axis-aligned thresholds
print("XOR task:", Xr.shape, "positives:", int(yr.sum()))""")

todo(
    """# TODO — rotate the feature space, then compare orig vs rotated
A = rng.normal(size=(d, d))
Q, _ = np.linalg.qr(A)          # random orthogonal rotation
Xrot = Xr @ Q

rot = {"original": (score(tree, Xr, yr),   score(nn, Xr, yr)),
       "rotated":  (score(tree, Xrot, yr), score(nn, Xrot, yr))}
for k, (t, n_) in rot.items():
    print(f"  {k:9s}: tree {t:.3f}  nn {n_:.3f}")
print(f"  rotation cost -> tree {rot['original'][0]-rot['rotated'][0]:+.3f}   nn {rot['original'][1]-rot['rotated'][1]:+.3f}")""",
    """# TODO — rotate the feature space, then compare orig vs rotated
A = ____                        # a (d, d) matrix of Gaussian noise (use rng)
Q, _ = ____                     # orthogonal Q from the QR decomposition of A
Xrot = ____                     # rotate the design: X @ Q

rot = {"original": (score(tree, Xr, yr),   score(nn, Xr, yr)),
       "rotated":  (score(tree, Xrot, yr), score(nn, Xrot, yr))}
for k, (t, n_) in rot.items():
    print(f"  {k:9s}: tree {t:.3f}  nn {n_:.3f}")
print(f"  rotation cost -> tree {rot['original'][0]-rot['rotated'][0]:+.3f}   nn {rot['original'][1]-rot['rotated'][1]:+.3f}")""",
)

code("""# CHECK — do not edit
assert np.allclose(Q @ Q.T, np.eye(d), atol=1e-8), "Q must be orthogonal (a rotation)."
assert np.allclose(Xrot, Xr @ Q), "Xrot should be the rotated design X @ Q."
assert rot["original"][0] > 0.9, "Axis-aligned, the tree should nail the XOR."
assert (rot["original"][0] - rot["rotated"][0]) > 0.1, "Rotation should clearly hurt the tree."
assert abs(rot["original"][1] - rot["rotated"][1]) < 0.03, "Rotation should barely move the ~invariant MLP."
print(f"Task 3 ok — rotation dropped the tree by {rot['original'][0]-rot['rotated'][0]:.3f} "
      f"but the MLP by only {abs(rot['original'][1]-rot['rotated'][1]):.3f}. Orientation was the tree's edge.")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** name the three tree strengths (inductive biases) in your own words, in one sentence each.""")

todo(
    '''# TODO: complete the three-strengths string
print("=== EXIT TICKET — Lesson 019 ===")
print(f"1. Irregular target : 2x2 gap {irr[2][0]-irr[2][1]:+.3f} -> 16x16 gap {irr[16][0]-irr[16][1]:+.3f}")
print(f"2. Uninformative    : +0 gap {junk[0][0]-junk[0][1]:+.3f} -> +100 gap {junk[100][0]-junk[100][1]:+.3f}")
print(f"3. Orientation      : tree {rot['original'][0]:.3f}->{rot['rotated'][0]:.3f}, nn {rot['original'][1]:.3f}->{rot['rotated'][1]:.3f}")
print()
print("three tree strengths:",
      "(1) trees fit irregular/non-smooth targets because they place axis-aligned steps anywhere, "
      "while neural nets are biased toward smooth functions; "
      "(2) trees are robust to uninformative features because each split selects the most informative "
      "column and ignores junk, while an MLP feeds every column into every neuron; "
      "(3) trees preserve the orientation of the data (axis-aligned splits) so they exploit that individual "
      "columns are meaningful, while a rotation-invariant MLP cannot tell a real column from a scrambled mixture.")''',
    '''# TODO: complete the three-strengths string
print("=== EXIT TICKET — Lesson 019 ===")
print(f"1. Irregular target : 2x2 gap {irr[2][0]-irr[2][1]:+.3f} -> 16x16 gap {irr[16][0]-irr[16][1]:+.3f}")
print(f"2. Uninformative    : +0 gap {junk[0][0]-junk[0][1]:+.3f} -> +100 gap {junk[100][0]-junk[100][1]:+.3f}")
print(f"3. Orientation      : tree {rot['original'][0]:.3f}->{rot['rotated'][0]:.3f}, nn {rot['original'][1]:.3f}->{rot['rotated'][1]:.3f}")
print()
print("three tree strengths:", "____")''',
)

# ---------------------------------------------------------------- stretch
md(r"""## Stretch (optional, ungraded) — push the biases

1. **Smooth it back.** Replace the checkerboard with a *smooth* target, e.g. $y = \mathbb{1}[x_0 + x_1 > 1]$. The tree's edge should shrink toward the MLP — confirming the edge lives in the *non-smooth* regime.
2. **Rotate only the informative axes.** Rotate just columns 0–1 of the XOR task and leave the noise columns alone — does a partial rotation cost the tree less?
3. **A modern tabular net.** Swap the MLP for `sklearn.ensemble.HistGradientBoostingClassifier` (another tree) on the rotated task — two trees, same rotation weakness. (The Year-2 architectures fix this on purpose.)""")

code('''# STRETCH — ungraded.
# def smooth(n=6000, seed=RS):
#     r = np.random.RandomState(seed); X = r.uniform(0, 1, size=(n, 2))
#     return X, (X[:, 0] + X[:, 1] > 1).astype(int)
# Xs, ys = smooth()
# print("smooth target: tree", round(score(tree, Xs, ys), 3), " nn", round(score(nn, Xs, ys), 3))''')


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l019-{i:02d}"
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": entry[1].splitlines(keepends=True)})
        elif kind == "todo":
            src = entry[1] if solution else entry[2]
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})
        else:
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": entry[1].splitlines(keepends=True)})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


(LABS / "0019-when-trees-win.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0019-when-trees-win.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0019-when-trees-win.ipynb and solutions/0019-when-trees-win.ipynb")
