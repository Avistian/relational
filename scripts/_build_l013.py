import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Lab 013 — Build boosting from residuals

**Lesson:** [`lessons/0013-boosting-intuition.html`](../lessons/0013-boosting-intuition.html) · **Phase / Year:** Year 1 · Q2

**Dataset tiers:** C — synthetic toy (mechanism isolation) for Task 1; A — OpenML German credit (`credit_g`) via `relkit` for Task 3 (same harness as Labs 011–012).

**Skill you are practising:** implement the stagewise **residual-fitting loop** of gradient boosting by hand, feel the **learning-rate** trade-off, then compare a single tree, a Random Forest, and `GradientBoostingClassifier` with the shared CV harness.

**Exit criteria:** EXIT TICKET prints the toy MSE collapse (round 0 → final), the learning-rate comparison, and the single-tree / RF / GBDT CV PR-AUC on `credit_g`.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. No local install? Open this notebook from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder** to execute it in the browser."""))

cells.append(md(r"""## Concept recap — boosting fits residuals

Bagging (Lesson 012) trains trees **independently** and averages them to cut **variance**. Boosting does the opposite: it trains trees **sequentially**, each one fitting what the ensemble still gets wrong, to cut **bias**.

The recipe (gradient boosting, squared-error loss):

1. Start with a constant model: `F₀(x) = mean(y)`.
2. For each round `m = 1 … M`:
   - Compute the **residuals** `rᵢ = yᵢ − F₋₁(xᵢ)` — for squared error, the residual **is** the negative gradient of the loss.
   - Fit a small tree `hₘ` to `(xᵢ, rᵢ)`.
   - Update `Fₘ(x) = F₋₁(x) + η · hₘ(x)`, where `η` is the **learning rate** (shrinkage).

**Learning rate `η`:** small steps (0.01–0.1) generalize better but need more rounds; big steps can overshoot. Shrinkage is regularization by taking small steps.

**Toy micro-example (not this lab's answer):** suppose `y = [10, 20]` and `F₀ = 15` (the mean). Residuals are `[-5, +5]`. A stump that predicts `-5` on the left row and `+5` on the right, added with `η = 1`, lands exactly on `y`. With `η = 0.3` you move only 30% of the way — hence more rounds.

**Why it matters for the thesis:** a competently-built gradient-boosted tree is *the* single-table baseline the relational thesis must beat fairly (Grinsztajn 2022). You have to know boosting reduces bias by fitting residuals — not by magic — to argue honestly that a relational model wins on *structure*, not on a weak baseline.

Full viz (residuals shrinking as rounds grow) and reading: [Lesson 013](../lessons/0013-boosting-intuition.html) · Friedman 2001."""))

cells.append(md("## Setup — PROVIDED"))

cells.append(code("""# PROVIDED
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))  # labs/ on the path so `relkit` imports

import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0
print("relkit ok")"""))

cells.append(md("""## Task 1 — the residual loop, by hand — TODO (crucial fragment)

**Goal:** implement gradient boosting's core loop on a 1-D toy and record the MSE against the true function each round.

**Why it matters:** this five-line loop *is* gradient boosting — XGBoost and LightGBM are fast, regularized versions of exactly this. Writing it once removes the mystery for every boosting model you'll ever tune.

**You implement (the loop body has 3 blanks):**
1. `resid` = current residuals, `y_toy − pred`.
2. fit a depth-1 tree (`stump`) to `(X_toy, resid)`, then update `pred += lr * stump.predict(X_toy)`.
3. append the MSE of `pred` vs the **true** function `f_true` to `mse_hist`.

**Hint boundary:** the residual is truth minus current prediction; the stump is a `DecisionTreeRegressor(max_depth=1)`; shrink its prediction by `lr` before adding. Do **not** call `GradientBoostingRegressor` here — the point is to build it."""))

cells.append(code("""# PROVIDED — toy data (Tier C): a smooth truth + noise
rng = np.random.RandomState(0)
x = np.sort(rng.uniform(0, 1, 200))
f_true = np.sin(2 * np.pi * x) + 0.5 * x        # the signal we want to recover
y_toy = f_true + rng.normal(0, 0.1, len(x))     # noisy observations
X_toy = x.reshape(-1, 1)

lr = 0.3
n_rounds = 60
pred = np.full_like(y_toy, y_toy.mean())         # F0 = mean(y)
mse_hist = [mean_squared_error(f_true, pred)]
print(f"round 0  MSE vs truth = {mse_hist[0]:.3f}")"""))

cells.append(code("""# TODO — fill the 3 blanks in the loop body
for m in range(n_rounds):
    resid = ____                                  # y_toy - pred
    stump = DecisionTreeRegressor(max_depth=1, random_state=m).fit(X_toy, ____)  # fit to residuals
    pred = pred + ____                            # add the shrunken stump: lr * stump.predict(X_toy)
    mse_hist.append(mean_squared_error(f_true, pred))

print(f"round {n_rounds}  MSE vs truth = {mse_hist[-1]:.3f}")
print("first 6 rounds:", [round(v, 3) for v in mse_hist[:6]])"""))

cells.append(code("""# CHECK — do not edit
assert len(mse_hist) == n_rounds + 1, "Record one MSE per round plus the round-0 baseline."
assert mse_hist[-1] < mse_hist[0], "Boosting should reduce error vs the mean baseline."
assert mse_hist[-1] < 0.05, "After 60 rounds the fit should be close to the truth."
assert all(mse_hist[i + 1] <= mse_hist[i] + 1e-9 for i in range(len(mse_hist) - 1)), \\
    "Training MSE vs truth should fall (or hold) each round for this clean toy."
print("Task 1 ok — you built gradient boosting; bias falls round by round.")"""))

cells.append(md("""## Task 2 — the learning-rate trade-off — TODO

**Goal:** show that a smaller learning rate needs more rounds to reach the same fit.

**Why it matters:** `η` and `n_estimators` trade off directly — this is the single most important pair of knobs on every boosting model, and the reason small-`η` runs are slower but usually generalize better.

**You implement:** for each `lr` in `lrs`, run the same residual loop for `n_rounds` rounds and store the **final** MSE-vs-truth in `final_mse[lr]`. (Reuse the Task-1 loop body.)

**Hint boundary:** re-initialize `pred` to the mean before each learning rate; the only thing that changes between runs is the shrinkage factor."""))

cells.append(code("""# TODO — sweep the learning rate (same 40-round budget for each)
lrs = [0.05, 0.3, 1.0]
n_rounds2 = 40
final_mse = {}
for cur_lr in lrs:
    pred_lr = np.full_like(y_toy, y_toy.mean())
    for m in range(n_rounds2):
        resid = ____                              # y_toy - pred_lr
        stump = DecisionTreeRegressor(max_depth=1, random_state=m).fit(X_toy, resid)
        pred_lr = pred_lr + ____                  # cur_lr * stump.predict(X_toy)
    final_mse[cur_lr] = mean_squared_error(f_true, pred_lr)

for k in lrs:
    print(f"lr={k:<5} final MSE after {n_rounds2} rounds = {final_mse[k]:.3f}")"""))

cells.append(code("""# CHECK — do not edit
assert set(final_mse) == set(lrs)
assert final_mse[0.05] > final_mse[0.3], \\
    "With a fixed round budget, the smallest lr should still be further from the truth."
print("Task 2 ok — small lr under-fits at a fixed budget; it needs more rounds.")"""))

cells.append(md("""## Task 3 — tree vs forest vs boosting on real data — TODO

**Goal:** score a single deep tree, a Random Forest, and a `GradientBoostingClassifier` on `credit_g` with the shared CV harness.

**Why it matters:** this is the honest baseline bake-off. On this small, noisy dataset the forest's variance reduction is hard to beat with an untuned booster — seeing that yourself (rather than assuming boosting always wins) is the discipline the mission demands.

**You implement:**
1. `DecisionTreeClassifier(random_state=RS)` → `tree_pr`.
2. `RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1)` → `rf_pr`.
3. `GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RS)` → `gb_pr`.

Score each with `cv_pr_auc(estimator, Xn, y)`."""))

cells.append(code("""# PROVIDED — Tier A: credit_g via relkit (label-encoded for bare estimators, as in Lab 012)
X, y = load_tier_a("credit_g")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
Xn = X[num_cols].copy()
for c in cat_cols:
    Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
y = np.asarray(y)
print(f"rows={len(y)}  pos_rate={y.mean():.3f}  features={Xn.shape[1]}")"""))

cells.append(code("""# TODO
tree_pr = ____   # cv_pr_auc(DecisionTreeClassifier(random_state=RS), Xn, y)
rf_pr   = ____   # cv_pr_auc(RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1), Xn, y)
gb_pr   = ____   # cv_pr_auc(GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RS), Xn, y)
print(f"single tree      CV PR-AUC = {tree_pr:.3f}")
print(f"random forest    CV PR-AUC = {rf_pr:.3f}")
print(f"gradient boost   CV PR-AUC = {gb_pr:.3f}")"""))

cells.append(code("""# CHECK — do not edit
for v in (tree_pr, rf_pr, gb_pr):
    assert 0.0 <= v <= 1.0, "PR-AUC must be in [0, 1]."
assert gb_pr > tree_pr, "Boosting should beat a lone deep tree."
assert rf_pr > tree_pr, "The forest should beat a lone deep tree."
print("Task 3 ok — both ensembles beat the single tree; compare RF vs GBDT below.")"""))

cells.append(md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, say what boosting reduces (bias vs variance) and *how* (fitting residuals), and note which model won on `credit_g` here and why that does **not** contradict "boosting usually wins on tabular"."""))

cells.append(code("""# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 013 ===")
print(f"toy MSE vs truth : round 0 {mse_hist[0]:.3f}  ->  round {n_rounds} {mse_hist[-1]:.3f}")
print(f"lr sweep (40 rounds): " + "  ".join(f"lr{ k }={v:.3f}" for k, v in final_mse.items()))
print(f"credit_g PR-AUC  : tree {tree_pr:.3f}  |  RF {rf_pr:.3f}  |  GBDT {gb_pr:.3f}")
print()
print("takeaway:", "____")"""))

cells.append(md("""## Stretch (optional, ungraded)

Boosting's edge grows with data and signal. Swap the dataset to `adult` (larger, ~24% prevalence) and re-score RF vs GBDT — you should see boosting pull ahead. Or vary `max_depth` (1 vs 3) and `learning_rate` on `credit_g` and watch the CV PR-AUC move. This previews the tuning you'll do properly with XGBoost in Lesson 014."""))

cells.append(code("""# STRETCH — ungraded. Uncomment to try boosting where it shines.
# Xa, ya = load_tier_a("adult")
# na = Xa.select_dtypes(include=[np.number]).columns.tolist()
# Xan = Xa[na].copy()
# for c in [c for c in Xa.columns if c not in na]:
#     Xan[c] = LabelEncoder().fit_transform(Xa[c].astype(str))
# ya = np.asarray(ya)
# print("adult RF  :", round(cv_pr_auc(RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=-1), Xan, ya), 3))
# print("adult GBDT:", round(cv_pr_auc(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RS), Xan, ya), 3))"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("labs/0013-boosting-intuition.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote labs/0013-boosting-intuition.ipynb with", len(cells), "cells")
