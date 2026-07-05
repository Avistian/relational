"""Build labs/0017-hyperparameter-search.ipynb (student, blanks) and solutions/ (filled).

Run:  python labs/_build_l017.py
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
md("""# Lab 017 — Hyperparameter search: grid vs random, tuned fairly

**Lesson:** [`lessons/0017-hyperparameter-search.html`](../lessons/0017-hyperparameter-search.html) · **Phase / Year:** Year 1 · Q2

**Paper:** Bergstra & Bengio 2012, *Random Search for Hyper-Parameter Optimization*, [JMLR 13:281–305](https://www.jmlr.org/papers/v13/bergstra12a.html) — §1 (low effective dimensionality) + Fig 1.

**Dataset tiers:** C — synthetic response surface (mechanism isolation) for Task 1; A — OpenML `credit_g` via `relkit` for Tasks 2–3 (same harness as Labs 011–016).

**Skill you are practising:** show *why* random search beats grid search at an equal budget once a few knobs are useless, run `RandomizedSearchCV` vs `GridSearchCV`, and get an **honest** score with nested CV (no selection bias).

**Exit criteria:** EXIT TICKET prints (1) grid vs random best-found on the synthetic surface as dimensions grow, (2) grid vs random tuned XGBoost CV PR-AUC on credit_g at equal budget, and (3) the optimistic search score vs the honest nested-CV score.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scipy**, **xgboost** (already in `requirements-labs.txt`). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — spend trials where they matter

A **hyperparameter** is fixed before training (a weight is learned). **Tuning** is a search: pick a config, estimate quality with **cross-validation** (never the test set), keep the best. Two things define it: the **space** (which knobs, what ranges) and the **budget** (how many configs you can CV-evaluate).

**Grid search** evaluates the full Cartesian product: $k$ values on each of $d$ axes costs $k^d$ configs — exponential in the number of knobs. To stay in budget as $d$ grows you must shrink $k$, so each axis ends up with only 2–3 values.

**Bergstra & Bengio's insight (low effective dimensionality):** only a few hyperparameters actually matter. A grid of $n$ points is a $\sqrt n \times \sqrt n$ lattice, so it tries only $\sqrt n$ **distinct values** on the important axis — every lattice column re-tests the same important value. **Random search** draws $n$ independent points, giving $\approx n$ distinct values on *every* axis at the same budget.

$$\text{grid: } k=\sqrt n \text{ distinct values / axis} \qquad \text{random: } \approx n \text{ distinct values / axis}$$

**The honest nuance (this lab reproduces it):** in 1–2 dimensions a well-placed grid can *win*. The random-search advantage appears — and then dominates — as you add knobs, because grid's per-axis resolution collapses while random's does not.

**Toy micro-example (not this lab's answer).** Budget $n=9$. A grid is $3\times3$: only **3** distinct values on the important axis. Random draws **9** points: **9** distinct values on that axis — 3× the resolution on the knob that matters, for free.

**Don't fool yourself:** `best_score_` is the CV score of the *winner of a search over many candidates* — optimistically biased (the L004 selection-bias trap). The honest number comes from **nested CV**: an outer split the search never touches.

**Why it matters for the thesis:** the undervaluation bet needs **fair baselines** — every model gets the *same* tuning budget under the *same* CV protocol, and you report a nested / held-out number. Random search is the equalizer. Full derivation + the grid-vs-random figure: [Lesson 017](../lessons/0017-hyperparameter-search.html) · Bergstra & Bengio 2012 §1.""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))  # labs/ on the path so `relkit` imports

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from relkit.data import load_tier_a

RS = 0
print("relkit + scipy + xgboost ok")""")

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — grid vs random on a synthetic surface — TODO (crucial fragment)

**Goal:** implement `random_search` and use the PROVIDED `grid_search` to reproduce the Bergstra & Bengio crossover: at an equal budget, grid can win in low dimensions but random takes over as useless dimensions are added.

**Why it matters:** this is the mechanism behind every "random search is my default" recommendation. Watching grid's best-found *collapse* when it is forced down to 2 values per axis makes the $\sqrt n$-vs-$n$ resolution argument concrete.

**You implement `random_search(d, budget, seed)`:** draw `budget` points uniformly in the $d$-dimensional unit cube, evaluate `response` at each, and return `(best_value_found, n_distinct_important_axis_values)`. The important axis is column 0, so the distinct count there is just the number of unique values in `points[:, 0]` (which, for continuous draws, equals `budget`).

**Hint boundary:** use `np.random.RandomState(seed).rand(budget, d)` for the points; `response(...)` is PROVIDED; the "distinct" count is `len(np.unique(points[:, 0]))`. Do not edit `response` or `grid_search`.""")

code("""# PROVIDED — a surface on [0,1]^d. Only column 0 matters (sharp peak at 0.5);
# the rest are near-flat "useless" knobs. Optimum value is 1.0 (at x0 = 0.5).
def response(x, d_important=1):
    x = np.atleast_2d(x)
    imp = np.exp(-((x[:, :d_important] - 0.5) ** 2).sum(1) / (2 * 0.08 ** 2))
    unimp = 0.02 * np.cos(6 * x[:, d_important:]).sum(1)
    return imp + unimp

def grid_search(d, budget):
    \"\"\"Near-`budget` full factorial grid; returns (best_found, n_distinct_axis0).\"\"\"
    k = max(1, round(budget ** (1.0 / d)))          # values per axis
    axis = np.linspace(0.05, 0.95, k)
    mesh = np.array(np.meshgrid(*([axis] * d))).reshape(d, -1).T
    return response(mesh).max(), k

print("response(): peak at x0=0.5 ->", round(float(response([[0.5, 0.5]])[0]), 3))""")

todo(
    """# TODO — implement random search over the unit cube
def random_search(d, budget, seed):
    pts = np.random.RandomState(seed).rand(budget, d)
    best = response(pts).max()
    distinct = len(np.unique(pts[:, 0]))            # distinct values tried on the important axis
    return best, distinct

print("dims | budget | grid distinct/best | random distinct/best (mean over 200 seeds)")
for d, budget in [(2, 9), (2, 25), (5, 32), (8, 64)]:
    g_best, g_dist = grid_search(d, budget)
    r = np.array([random_search(d, budget, s) for s in range(200)])
    print(f"  d={d:1d} | {budget:3d} | grid {g_dist:2d} vals -> {g_best:.3f} "
          f"| random {r[:,1].mean():.0f} vals -> {r[:,0].mean():.3f}")""",
    """# TODO — implement random search over the unit cube
def random_search(d, budget, seed):
    pts = ____                                      # np.random.RandomState(seed).rand(budget, d)
    best = ____                                     # best response value over the sampled points
    distinct = ____                                 # number of distinct values on the important axis (column 0)
    return best, distinct

print("dims | budget | grid distinct/best | random distinct/best (mean over 200 seeds)")
for d, budget in [(2, 9), (2, 25), (5, 32), (8, 64)]:
    g_best, g_dist = grid_search(d, budget)
    r = np.array([random_search(d, budget, s) for s in range(200)])
    print(f"  d={d:1d} | {budget:3d} | grid {g_dist:2d} vals -> {g_best:.3f} "
          f"| random {r[:,1].mean():.0f} vals -> {r[:,0].mean():.3f}")""",
)

code("""# CHECK — do not edit
b, dist = random_search(5, 32, 0)
assert dist == 32, "Random search should try `budget` distinct values on the important axis."
assert 0.0 <= b <= 1.05, "response() is bounded near [0, 1]."
# In LOW dim a lucky grid can win; in HIGHER dim random should dominate.
g2, _ = grid_search(2, 9)
r2 = np.mean([random_search(2, 9, s)[0] for s in range(200)])
g5, _ = grid_search(5, 32)
r5 = np.mean([random_search(5, 32, s)[0] for s in range(200)])
assert g2 >= r2 - 0.05, "In 2-D a well-placed grid should be competitive."
assert r5 > g5 + 0.3, "In 5-D random should crush grid (grid collapses to ~2 values/axis)."
print(f"Task 1 ok — 2-D: grid {g2:.3f} vs random {r2:.3f} (grid competitive); "
      f"5-D: grid {g5:.3f} vs random {r5:.3f} (random wins).")""")

# ---------------------------------------------------------------- Task 2
md(r"""## Task 2 — GridSearchCV vs RandomizedSearchCV on XGBoost — TODO (reproduction target)

**Goal:** tune XGBoost on `credit_g` two ways at an **equal budget** — a 3×3×3 grid (27 configs) and 27 random draws over a *richer* 5-knob space — and confirm random wins.

**Why it matters:** honest-baseline discipline in action. Same budget, same 5-fold CV PR-AUC (as L014). The only difference is *where the trials are spent*.

**Reproduction target (lesson table):** XGBoost default ≈ 0.883, GridSearchCV ≈ 0.891, RandomizedSearchCV ≈ 0.901.

**You implement:** the `dists` dictionary of **distributions** for the random search (note `learning_rate` is **log-uniform**), and set `n_iter` to the grid size so the budgets match. `GridSearchCV` and the CV splitter are PROVIDED.""")

code("""# PROVIDED — credit_g, label-encoded for XGBoost + a fixed inner CV.
X, y = load_tier_a("credit_g")
y = np.asarray(y)
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = [c for c in X.columns if c not in num]
Xle = X[num].copy()
for c in cat:
    Xle[c] = LabelEncoder().fit_transform(X[c].astype(str))

base = XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist")
inner = StratifiedKFold(5, shuffle=True, random_state=RS)

default_score = cross_val_score(base, Xle, y, cv=inner, scoring="average_precision").mean()
grid = {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.1, 0.3], "n_estimators": [200, 400, 600]}
grid_n = int(np.prod([len(v) for v in grid.values()]))
gs = GridSearchCV(base, grid, scoring="average_precision", cv=inner, n_jobs=4).fit(Xle, y)
print(f"XGBoost default       CV PR-AUC = {default_score:.3f}")
print(f"GridSearchCV ({grid_n})    CV PR-AUC = {gs.best_score_:.3f}")""")

todo(
    """# TODO — random search over a richer space at the SAME budget
dists = {
    "max_depth": randint(3, 8),
    "learning_rate": loguniform(0.01, 0.3),     # log-uniform: a rate spanning orders of magnitude
    "n_estimators": randint(150, 700),
    "subsample": uniform(0.6, 0.4),             # range 0.6 .. 1.0
    "colsample_bytree": uniform(0.6, 0.4),
}
rs_search = RandomizedSearchCV(
    base, dists, n_iter=grid_n, scoring="average_precision",
    cv=inner, n_jobs=4, random_state=RS).fit(Xle, y)

print(f"RandomizedSearchCV ({grid_n}) CV PR-AUC = {rs_search.best_score_:.3f}")""",
    """# TODO — random search over a richer space at the SAME budget
dists = {
    "max_depth": randint(3, 8),
    "learning_rate": ____,                      # log-uniform over 0.01 .. 0.3 (use loguniform)
    "n_estimators": randint(150, 700),
    "subsample": uniform(0.6, 0.4),             # range 0.6 .. 1.0
    "colsample_bytree": uniform(0.6, 0.4),
}
rs_search = RandomizedSearchCV(
    base, dists, n_iter=____, scoring="average_precision",   # equal budget: match the grid size
    cv=inner, n_jobs=4, random_state=RS).fit(Xle, y)

print(f"RandomizedSearchCV ({grid_n}) CV PR-AUC = {rs_search.best_score_:.3f}")""",
)

code("""# CHECK — do not edit
assert abs(default_score - 0.883) < 0.03, "XGBoost default should reproduce ~0.883 on credit_g."
assert rs_search.n_iter == grid_n, "Random search budget must equal the grid size for a fair comparison."
for v in (gs.best_score_, rs_search.best_score_):
    assert 0.85 < v < 0.93, "Tuned PR-AUC should land in a sensible range for credit_g."
assert rs_search.best_score_ >= gs.best_score_ - 0.005, \\
    "At an equal budget over a richer space, random search should match or beat the grid."
print(f"Task 2 ok — default {default_score:.3f} | grid {gs.best_score_:.3f} "
      f"| random {rs_search.best_score_:.3f}.")""")

# ---------------------------------------------------------------- Task 3
md("""## Task 3 — the honest number: nested CV — TODO (leakage discipline)

**Goal:** `best_score_` is the CV score of the *winner of a 27-way search* — optimistically biased (L004 selection bias). Get the honest estimate with **nested CV**: an *outer* 5-fold loop wraps the whole `RandomizedSearchCV`, so each outer test fold is scored by a search that never saw it.

**Why it matters:** this is the number you are allowed to report. Quoting `best_score_` as "our performance" is the single most common way tuned models look better than they are.

**You implement:** wrap the search in `cross_val_score` with an **outer** `StratifiedKFold`. The inner search (same `dists`, same `n_iter`) is the estimator being cross-validated.""")

todo(
    """# TODO — nested CV: outer loop scores a search that never saw the outer test fold
outer = StratifiedKFold(5, shuffle=True, random_state=RS)
search = RandomizedSearchCV(base, dists, n_iter=grid_n, scoring="average_precision",
                            cv=inner, n_jobs=4, random_state=RS)
nested = cross_val_score(search, Xle, y, cv=outer, scoring="average_precision").mean()

print(f"best_score_ (optimistic) = {rs_search.best_score_:.3f}")
print(f"nested CV   (honest)     = {nested:.3f}")
print(f"optimism gap             = {rs_search.best_score_ - nested:+.3f}")""",
    """# TODO — nested CV: outer loop scores a search that never saw the outer test fold
outer = StratifiedKFold(5, shuffle=True, random_state=RS)
search = RandomizedSearchCV(base, dists, n_iter=grid_n, scoring="average_precision",
                            cv=inner, n_jobs=4, random_state=RS)
nested = ____                       # cross_val_score(search, Xle, y, cv=outer, scoring="average_precision").mean()

print(f"best_score_ (optimistic) = {rs_search.best_score_:.3f}")
print(f"nested CV   (honest)     = {nested:.3f}")
print(f"optimism gap             = {rs_search.best_score_ - nested:+.3f}")""",
)

code("""# CHECK — do not edit
assert 0.80 < nested < 0.93, "Nested CV score should be a sensible PR-AUC."
assert nested <= rs_search.best_score_ + 1e-9, \\
    "Nested CV is an out-of-search estimate; it should not exceed the (optimistic) best_score_."
print(f"Task 3 ok — optimistic {rs_search.best_score_:.3f} >= honest nested {nested:.3f}.")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, say why random search covers the important knob better than a grid at equal budget, and why you must report the nested-CV number instead of `best_score_`.""")

todo(
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 017 ===")
print(f"Synthetic 2-D: grid {g2:.3f} vs random {r2:.3f} (grid competitive)")
print(f"Synthetic 5-D: grid {g5:.3f} vs random {r5:.3f} (random wins)")
print(f"credit_g tuned XGB: default {default_score:.3f} | grid {gs.best_score_:.3f} | random {rs_search.best_score_:.3f}")
print(f"Honest score: nested CV {nested:.3f} vs optimistic best_score_ {rs_search.best_score_:.3f}")
print()
print("takeaway:", "a grid of n points reuses only sqrt(n) distinct values on each axis while random "
      "search tries ~n, so random covers the few knobs that matter far better once useless knobs are added; "
      "report the nested-CV score because best_score_ is the optimistic max over the searched candidates.")''',
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 017 ===")
print(f"Synthetic 2-D: grid {g2:.3f} vs random {r2:.3f} (grid competitive)")
print(f"Synthetic 5-D: grid {g5:.3f} vs random {r5:.3f} (random wins)")
print(f"credit_g tuned XGB: default {default_score:.3f} | grid {gs.best_score_:.3f} | random {rs_search.best_score_:.3f}")
print(f"Honest score: nested CV {nested:.3f} vs optimistic best_score_ {rs_search.best_score_:.3f}")
print()
print("takeaway:", "____")''',
)

# ---------------------------------------------------------------- stretch
md("""## Stretch (optional, ungraded) — successive halving

sklearn's `HalvingRandomSearchCV` starts many configs on a small budget (few estimators / few rows), kills the losers, and reinvests compute in the survivors. Try it and compare wall-clock + best score to the plain random search above — often similar quality, much faster.""")

code('''# STRETCH — ungraded. Successive halving (needs the experimental import).
# from sklearn.experimental import enable_halving_search_cv  # noqa: F401
# from sklearn.model_selection import HalvingRandomSearchCV
# hs = HalvingRandomSearchCV(base, dists, scoring="average_precision", cv=inner,
#                            factor=3, random_state=RS, n_jobs=4).fit(Xle, y)
# print(f"HalvingRandomSearchCV best CV PR-AUC = {hs.best_score_:.3f}  (n_candidates seen: {len(hs.cv_results_['params'])})")''')


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l017-{i:02d}"
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


(LABS / "0017-hyperparameter-search.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0017-hyperparameter-search.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0017-hyperparameter-search.ipynb and solutions/0017-hyperparameter-search.ipynb")
