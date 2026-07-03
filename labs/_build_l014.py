"""Build labs/0014-xgboost.ipynb (student, blanks) and solutions/0014-xgboost.ipynb (filled).

Run:  python labs/_build_l014.py     (from repo root or labs/)
Then execute the solution to verify:  jupyter nbconvert --execute ...
"""
from __future__ import annotations

import json
from pathlib import Path

LABS = Path(__file__).resolve().parent
SOL = LABS / "solutions"
SOL.mkdir(exist_ok=True)

# Each entry: ("md"|"provided"|"check"|"exit", source)  -> shared source
# TODO entries: ("todo", solution_source, student_source)
CELLS = []


def md(src):
    CELLS.append(("md", src))


def code(src):
    CELLS.append(("code", src))


def todo(sol, stu):
    CELLS.append(("todo", sol, stu))


# ---------------------------------------------------------------- header
md("""# Lab 014 — XGBoost from the formula up

**Lesson:** [`lessons/0014-xgboost.html`](../lessons/0014-xgboost.html) · **Phase / Year:** Year 1 · Q2

**Paper:** Chen & Guestrin 2016, *XGBoost: A Scalable Tree Boosting System* (KDD) — §2, Eq 5–7.

**Dataset tiers:** C — synthetic squared-error toy (mechanism isolation) for Tasks 1–2; A — OpenML German credit (`credit_g`) via `relkit` for Task 3 (same harness as Labs 011–013).

**Skill you are practising:** implement XGBoost's **regularized leaf weight** and **structure-score split gain** by hand and check them against XGBoost's own leaf outputs; then fit and **tune** XGBoost with the shared CV harness and compare honestly to the L013 baseline.

**Exit criteria:** EXIT TICKET prints (1) your by-hand vs XGBoost leaf weights, (2) how λ shrinks weights and γ prunes a split, and (3) the credit_g CV PR-AUC for the L013 GBDT, default XGBoost, and tuned XGBoost.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. This lab needs **xgboost** (already in `requirements-labs.txt`). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — the two formulas that define XGBoost

Gradient boosting (Lesson 013) fits a tree to the residuals each round. XGBoost keeps that stagewise recipe but writes down a **regularized objective** and keeps the **second-order** term of the loss. For each row it uses the gradient $g_i$ and the hessian $h_i$; for a whole leaf it sums them: $G_j = \sum g_i$, $H_j = \sum h_i$.

The tree is penalized by $\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2$ (T = number of leaves). Minimizing the resulting quadratic gives two things:

- **Optimal leaf weight** (the value the leaf outputs):
$$w_j^{*} = -\frac{G_j}{H_j + \lambda}$$
- **Structure score** of a node, whose per-leaf term measures leaf quality: $\dfrac{G_j^2}{H_j + \lambda}$. The gain of a split is then
$$\text{Gain} = \tfrac{1}{2}\Big[\tfrac{G_L^2}{H_L+\lambda} + \tfrac{G_R^2}{H_R+\lambda} - \tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big] - \gamma$$

**Callback (L011):** that bracket is *left child + right child − parent* — the same `ΔG = children − parent` shape as a Gini split, now in units of $G^2/(H+\lambda)$. The $-\gamma$ is a flat toll: if children don't beat the parent by more than $\gamma$, the gain is negative and the split is **pruned**.

**Toy micro-example (not this lab's answer):** a leaf holding rows with gradient sum $G=-6$, hessian sum $H=4$, and $\lambda=2$ outputs $w^{*} = -(-6)/(4+2) = +1.0$. Raise $\lambda$ to 10 and it shrinks to $6/14 \approx 0.43$ — that is L2 regularization pulling the leaf toward 0.

**Why it matters for the thesis:** a *tuned* XGBoost is *the* single-table baseline the relational thesis must beat fairly (Grinsztajn 2022). "We beat a default XGBoost" is not a result. Full derivation and the interactive λ/γ viz: [Lesson 014](../lessons/0014-xgboost.html) · Chen & Guestrin 2016 §2.""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))  # labs/ on the path so `relkit` imports

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0
print("relkit + xgboost ok")""")

# ---------------------------------------------------------------- Task 1
md("""## Task 1 — the leaf weight & split gain, by hand — TODO (crucial fragment)

**Goal:** implement XGBoost's three §2 formulas — node score, regularized leaf weight, and split gain — and reproduce XGBoost's *own* leaf outputs exactly.

**Why it matters:** these formulas *are* XGBoost. Once you can compute the leaf value XGBoost prints, the model stops being a black box: every knob (`reg_lambda`, `gamma`) is just a term you wrote here.

**You implement (3 one-line returns):**
1. `node_score(G, H, lam)` → $G^2/(H+\\lambda)$
2. `reg_leaf_weight(G, H, lam)` → $-G/(H+\\lambda)$
3. `split_gain(GL, HL, GR, HR, lam, gamma)` → $\\tfrac{1}{2}[\\text{score}(L)+\\text{score}(R)-\\text{score(parent)}]-\\gamma$

**Hint boundary:** the parent's stats are the sums $G_L+G_R$ and $H_L+H_R$; reuse `node_score` inside `split_gain`. Do **not** read the leaf value off XGBoost — compute it, then compare.""")

code("""# PROVIDED — a squared-error toy so g = pred - y and h = 1 (base_score 0.5).
# We fit a single XGBoost stump, read its split, and gather leaf gradient/hessian sums.
rng = np.random.RandomState(0)
Xr = rng.uniform(0, 1, size=(40, 1))
yr = (Xr[:, 0] > 0.5).astype(float) * 4.0 + rng.normal(0, 0.2, 40)

LAMBDA = 3.0
stump = XGBRegressor(
    n_estimators=1, max_depth=1, learning_rate=1.0,
    reg_lambda=LAMBDA, reg_alpha=0.0, gamma=0.0, base_score=0.5,
    objective="reg:squarederror", random_state=RS,
).fit(Xr, yr)

g = np.full(len(yr), 0.5) - yr      # gradient of ½(pred - y)^2 at pred = 0.5
h = np.ones_like(yr)                # hessian = 1

df = stump.get_booster().trees_to_dataframe()
thr = float(df[df["Feature"] != "Leaf"].iloc[0]["Split"])
left = Xr[:, 0] < thr
GL, HL = g[left].sum(), h[left].sum()
GR, HR = g[~left].sum(), h[~left].sum()
xgb_leaf_vals = df[df["Feature"] == "Leaf"].sort_values("Node")["Gain"].to_numpy()
print(f"split threshold = {thr:.4f}   GL,HL = {GL:.2f},{HL:.0f}   GR,HR = {GR:.2f},{HR:.0f}")""")

todo(
    """# TODO — implement the three XGBoost formulas (Chen & Guestrin 2016, §2)
def node_score(G, H, lam):
    return (G * G) / (H + lam)

def reg_leaf_weight(G, H, lam):
    return -G / (H + lam)

def split_gain(GL, HL, GR, HR, lam, gamma):
    children = node_score(GL, HL, lam) + node_score(GR, HR, lam)
    parent = node_score(GL + GR, HL + HR, lam)
    return 0.5 * (children - parent) - gamma

wL = reg_leaf_weight(GL, HL, LAMBDA)
wR = reg_leaf_weight(GR, HR, LAMBDA)
print(f"by-hand leaf weights  L/R = {wL:+.4f} / {wR:+.4f}")
print(f"XGBoost leaf weights      = {np.sort(xgb_leaf_vals)}")""",
    """# TODO — implement the three XGBoost formulas (Chen & Guestrin 2016, §2)
def node_score(G, H, lam):
    return ____                      # G^2 / (H + lam)

def reg_leaf_weight(G, H, lam):
    return ____                      # -G / (H + lam)

def split_gain(GL, HL, GR, HR, lam, gamma):
    children = ____                  # node_score(L) + node_score(R)
    parent = ____                    # node_score of the combined parent
    return 0.5 * (children - parent) - gamma

wL = reg_leaf_weight(GL, HL, LAMBDA)
wR = reg_leaf_weight(GR, HR, LAMBDA)
print(f"by-hand leaf weights  L/R = {wL:+.4f} / {wR:+.4f}")
print(f"XGBoost leaf weights      = {np.sort(xgb_leaf_vals)}")""",
)

code("""# CHECK — do not edit
assert np.isclose(node_score(4.0, 20.0, 0.0), 16.0 / 20.0), "node_score should be G^2/(H+lam)."
assert np.allclose(np.sort([wL, wR]), np.sort(xgb_leaf_vals), atol=1e-3), \\
    "Your leaf weights should match XGBoost's own leaf outputs (they are the same formula)."
print("Task 1 ok — your formula reproduces XGBoost's leaf weights exactly.")""")

# ---------------------------------------------------------------- Task 2
md("""## Task 2 — feel the regularizers λ and γ — TODO

**Goal:** show numerically that (a) raising **λ** shrinks a leaf weight toward 0, and (b) raising **γ** past the raw gain flips a split from KEEP to PRUNE.

**Why it matters:** these are the two knobs that make XGBoost hard to overfit on small, noisy data — exactly the regime where L013 showed a Random Forest can tie a booster. Tuning them is most of the game.

**You implement:** two list comprehensions that reuse your Task-1 functions —
`weights_by_lambda` (leaf weight of the LEFT leaf at each λ) and `net_gains` (split gain at each γ toll, λ fixed at 1.0).""")

todo(
    """# TODO — sweep λ (shrinkage) and γ (pruning toll)
lam_grid = [0.0, 1.0, 3.0, 10.0]
weights_by_lambda = [reg_leaf_weight(GL, HL, l) for l in lam_grid]

raw_gain = split_gain(GL, HL, GR, HR, lam=1.0, gamma=0.0)
gammas = [0.0, raw_gain * 0.5, raw_gain * 1.5]
net_gains = [split_gain(GL, HL, GR, HR, lam=1.0, gamma=gm) for gm in gammas]

for l, w in zip(lam_grid, weights_by_lambda):
    print(f"  λ={l:5.1f}  left-leaf w* = {w:+.4f}")
print(f"raw gain (γ=0) = {raw_gain:.3f}")
for gm, ng in zip(gammas, net_gains):
    print(f"  γ={gm:6.3f}  net gain = {ng:+.3f}  ->  {'KEEP' if ng > 0 else 'PRUNE'}")""",
    """# TODO — sweep λ (shrinkage) and γ (pruning toll)
lam_grid = [0.0, 1.0, 3.0, 10.0]
weights_by_lambda = [____ for l in lam_grid]        # reg_leaf_weight of LEFT leaf at each λ

raw_gain = split_gain(GL, HL, GR, HR, lam=1.0, gamma=0.0)
gammas = [0.0, raw_gain * 0.5, raw_gain * 1.5]
net_gains = [____ for gm in gammas]                 # split_gain at each γ (λ=1.0)

for l, w in zip(lam_grid, weights_by_lambda):
    print(f"  λ={l:5.1f}  left-leaf w* = {w:+.4f}")
print(f"raw gain (γ=0) = {raw_gain:.3f}")
for gm, ng in zip(gammas, net_gains):
    print(f"  γ={gm:6.3f}  net gain = {ng:+.3f}  ->  {'KEEP' if ng > 0 else 'PRUNE'}")""",
)

code("""# CHECK — do not edit
assert abs(weights_by_lambda[0]) > abs(weights_by_lambda[-1]), "λ should shrink |w| toward 0."
assert net_gains[0] > 0 and net_gains[-1] < 0, \\
    "γ below the raw gain keeps the split; γ above it should prune (net gain < 0)."
print("Task 2 ok — λ shrinks weights; γ past the raw gain prunes the split.")""")

# ---------------------------------------------------------------- Task 3
md("""## Task 3 — fit & tune XGBoost on real data — TODO (reproduction target)

**Goal:** score XGBoost's defaults against the L013 sklearn GBDT on `credit_g`, then tune it with a small random search and see how much tuning buys.

**Why it matters:** this is the honest-baseline discipline the mission demands. Out of the box XGBoost barely edges sklearn's GBDT here; the real gain comes from tuning — a cost you must pay before claiming any relational model "beat XGBoost".

**Reproduction target (from the lesson table):** sklearn GBDT ≈ 0.879, XGBoost default ≈ 0.883, XGBoost tuned ≈ 0.896.

**You implement:**
1. `xgb_default` — `cv_pr_auc` of a plain `XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist")`.
2. the `estimator` passed to the PROVIDED `RandomizedSearchCV` (same XGBClassifier).""")

code("""# PROVIDED — Tier A: credit_g via relkit (label-encoded for bare estimators, as in Labs 012-013)
X, y = load_tier_a("credit_g")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
Xn = X[num_cols].copy()
for c in cat_cols:
    Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
y = np.asarray(y)

gb_ref = cv_pr_auc(
    GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RS),
    Xn, y,
)
param_dist = {
    "n_estimators": [200, 400, 600],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
    "min_child_weight": [1, 3, 5],
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
print(f"rows={len(y)} pos_rate={y.mean():.3f} features={Xn.shape[1]}  |  L013 GBDT ref = {gb_ref:.3f}")""")

todo(
    """# TODO — default XGBoost, then a 40-iteration random search
xgb_default = cv_pr_auc(
    XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
    Xn, y,
)

search = RandomizedSearchCV(
    XGBClassifier(random_state=RS, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
    param_dist, n_iter=40, scoring="average_precision", cv=cv, random_state=RS, n_jobs=-1,
)
search.fit(Xn, y)
xgb_tuned = float(search.best_score_)

print(f"XGBoost default  CV PR-AUC = {xgb_default:.3f}")
print(f"XGBoost tuned    CV PR-AUC = {xgb_tuned:.3f}")
print(f"best params: {search.best_params_}")""",
    """# TODO — default XGBoost, then a 40-iteration random search
xgb_default = cv_pr_auc(
    ____,                          # a plain XGBClassifier (see hint above)
    Xn, y,
)

search = RandomizedSearchCV(
    ____,                          # the estimator to tune (same XGBClassifier)
    param_dist, n_iter=40, scoring="average_precision", cv=cv, random_state=RS, n_jobs=-1,
)
search.fit(Xn, y)
xgb_tuned = float(search.best_score_)

print(f"XGBoost default  CV PR-AUC = {xgb_default:.3f}")
print(f"XGBoost tuned    CV PR-AUC = {xgb_tuned:.3f}")
print(f"best params: {search.best_params_}")""",
)

code("""# CHECK — do not edit
for v in (xgb_default, xgb_tuned):
    assert 0.5 < v < 1.0, "PR-AUC should be a sensible probability-ranking score."
assert xgb_tuned >= xgb_default - 0.01, "Tuning should not meaningfully hurt CV PR-AUC."
print(f"Task 3 ok — default {xgb_default:.3f} -> tuned {xgb_tuned:.3f} (Δ {xgb_tuned - xgb_default:+.3f}).")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, name the two XGBoost regularizers you implemented (λ and γ) and what each does, and state whether tuning was needed to beat the L013 GBDT here.""")

todo(
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 014 ===")
print(f"leaf weight  by hand {np.sort([wL, wR])}  vs XGBoost {np.sort(xgb_leaf_vals)}")
print(f"λ sweep (left leaf w*): " + "  ".join(f"λ{l}={w:+.3f}" for l, w in zip(lam_grid, weights_by_lambda)))
print(f"γ prune: raw gain {raw_gain:.2f} -> net gains {[round(n, 2) for n in net_gains]}")
print(f"credit_g PR-AUC: GBDT {gb_ref:.3f} | XGB default {xgb_default:.3f} | XGB tuned {xgb_tuned:.3f}")
print()
print("takeaway:", "λ (reg_lambda) shrinks leaf weights and γ (gamma) is a per-split toll that prunes weak splits; "
      "on this small noisy dataset XGBoost only pulled clearly ahead of the L013 GBDT after tuning.")''',
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 014 ===")
print(f"leaf weight  by hand {np.sort([wL, wR])}  vs XGBoost {np.sort(xgb_leaf_vals)}")
print(f"λ sweep (left leaf w*): " + "  ".join(f"λ{l}={w:+.3f}" for l, w in zip(lam_grid, weights_by_lambda)))
print(f"γ prune: raw gain {raw_gain:.2f} -> net gains {[round(n, 2) for n in net_gains]}")
print(f"credit_g PR-AUC: GBDT {gb_ref:.3f} | XGB default {xgb_default:.3f} | XGB tuned {xgb_tuned:.3f}")
print()
print("takeaway:", "____")''',
)

# ---------------------------------------------------------------- stretch
md("""## Stretch (optional, ungraded)

XGBoost's edge grows with data and signal. Swap to `adult` (~49k rows, ~24% prevalence) and compare the L013 GBDT vs XGBoost default — the booster should pull ahead more clearly than on `credit_g`. Or set `missing=np.nan` and drop values into a column to watch **sparsity-aware split finding** route them without imputation (Lesson 006 callback).""")

code("""# STRETCH — ungraded. Boosting where it shines: more data + real signal.
# Xa, ya = load_tier_a("adult")
# na = Xa.select_dtypes(include=[np.number]).columns.tolist()
# Xan = Xa[na].copy()
# for c in [c for c in Xa.columns if c not in na]:
#     Xan[c] = LabelEncoder().fit_transform(Xa[c].astype(str))
# ya = np.asarray(ya)
# print("adult GBDT:", round(cv_pr_auc(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RS), Xan, ya), 3))
# print("adult XGB :", round(cv_pr_auc(XGBClassifier(random_state=RS, n_jobs=-1, eval_metric='logloss', tree_method='hist'), Xan, ya), 3))""")


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l014-{i:02d}"
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": entry[1].splitlines(keepends=True)})
        elif kind == "todo":
            src = entry[1] if solution else entry[2]
            nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})
        else:  # code / provided / check / exit
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


(LABS / "0014-xgboost.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0014-xgboost.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0014-xgboost.ipynb and solutions/0014-xgboost.ipynb")
