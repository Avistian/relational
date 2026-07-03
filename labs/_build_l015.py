"""Build labs/0015-lightgbm.ipynb (student, blanks) and solutions/0015-lightgbm.ipynb (filled).

Run:  python labs/_build_l015.py
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
md("""# Lab 015 — LightGBM: leaf-wise, GOSS, and speed

**Lesson:** [`lessons/0015-lightgbm.html`](../lessons/0015-lightgbm.html) · **Phase / Year:** Year 1 · Q2

**Paper:** Ke et al. 2017, *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (NeurIPS 30) — §3 (GOSS), §4 (EFB). No arXiv; use the [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).

**Dataset tiers:** C — synthetic gradients (mechanism isolation) for Task 1; A — OpenML `credit_g` / `adult` via `relkit` for Tasks 2–3 (same harness as Labs 011–014).

**Skill you are practising:** implement **GOSS's amplified gradient estimate** by hand and show it is (nearly) unbiased against the full-data sum; then *see* leaf-wise overfitting via a `num_leaves` sweep, and race LightGBM against XGBoost on real data with the shared CV harness.

**Exit criteria:** EXIT TICKET prints (1) your GOSS estimate vs the true gradient sum and how many rows it used, (2) the `num_leaves` sweep on credit_g, and (3) LightGBM vs XGBoost CV PR-AUC on credit_g.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. This lab needs **lightgbm** (added to `requirements-labs.txt`). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — how LightGBM makes GBDT fast

LightGBM keeps XGBoost's exact split-gain math (Lesson 014) and re-engineers the *cost* of computing it. Four ideas:

- **Histograms** — bin each feature into ~255 discrete buckets, so a split scan is $O(\#\text{bins})$, not $O(\#\text{rows})$. A child's histogram is the parent's minus its sibling's (free subtraction).
- **Leaf-wise (best-first) growth** — instead of splitting a whole depth at a time (level-wise, XGBoost's default), split the one frontier leaf with the largest loss reduction. For a fixed number of leaves this reaches lower training loss, but grows deep, unbalanced trees that **overfit small data** — so `num_leaves` (not `max_depth`) is the main capacity knob.
- **GOSS (Gradient-based One-Side Sampling)** — a row's $|g_i|$ measures how badly the model still fits it. Keep the top $a\cdot n$ rows by $|g|$, then randomly draw a further $b\cdot n$ rows from the small-gradient remainder and **amplify** them by $\tfrac{1-a}{b}$ when summing gradients/hessians so the totals stay unbiased (Ke et al. 2017, Algorithm 2).
- **EFB (Exclusive Feature Bundling)** — merge sparse features that are never nonzero together into one bundle, cutting histogram cost from $O(\#\text{data}\times\#\text{feature})$ to $O(\#\text{data}\times\#\text{bundle})$.

**Why the GOSS factor is $\tfrac{1-a}{b}$:** you sampled $b\cdot n$ of the $(1-a)\cdot n$ small-gradient rows — a fraction $\tfrac{b}{1-a}$ of them — so each survivor must "stand in" for the rest, i.e. be weighted by the reciprocal $\tfrac{1-a}{b}$, to keep the small-gradient sum unbiased.

**Toy micro-example (not this lab's answer):** with $a=0.2, b=0.1$ and $n$ rows, you keep the top $0.2n$ by $|g|$ and sample another $0.1n$ from the rest — so you use $\approx 30\%$ of the data; each sampled small-gradient row is weighted $\tfrac{1-0.2}{0.1}=8\times$.

**Why it matters for the thesis:** LightGBM is the GBDT baseline that still *finishes* when a relational join explodes into millions of sparse rows/columns. A fair "we beat GBDT" must include a tuned LightGBM at full scale — not just XGBoost on a subsample. Full derivation + the leaf-wise viz: [Lesson 015](../lessons/0015-lightgbm.html) · Ke et al. 2017 §3–4.""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))  # labs/ on the path so `relkit` imports

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from relkit.data import load_tier_a
from relkit.metrics import cv_pr_auc

RS = 0
print("relkit + lightgbm + xgboost ok")""")

# ---------------------------------------------------------------- Task 1
md("""## Task 1 — GOSS's amplified gradient estimate, by hand — TODO (crucial fragment)

**Goal:** implement the GOSS row weighting and show its gradient-sum estimate over a region is close to the full-data sum — with far fewer rows.

**Why it matters:** this is the theorem that justifies "up to 20× faster at almost the same accuracy." Once you see the $\\tfrac{1-a}{b}$ factor keep the sum unbiased, GOSS stops being magic — it is just importance sampling that favours the rows the model still gets wrong.

**You implement `goss_weights(g, a, b, rng)`** returning a per-row weight vector:
1. sort rows by **descending |g|**; the top `int(a*n)` rows get weight **1.0**;
2. from the remaining rows, randomly pick `int(b*n)` of them (use `rng.choice(..., replace=False)`) and give them weight **(1−a)/b**;
3. everyone else gets weight **0.0**.

**Hint boundary:** build a zero vector and fill the two groups; the sample size is `int(b*n)` (a fraction of *all* rows, per the paper), not a fraction of the remainder. Do not touch the estimate/CHECK cells.""")

code("""# PROVIDED — synthetic per-row gradients and a candidate split region.
rng = np.random.RandomState(0)
n = 2000
g = rng.standard_normal(n)          # gradient of each row (sign + magnitude)
region = rng.rand(n) < 0.5          # rows that fall on the LEFT of a candidate split
A, B = 0.2, 0.1                     # keep top-20% by |g|, sample 10% of the rest
true_left = g[region].sum()
print(f"true left-region gradient sum (all {region.sum()} rows) = {true_left:.3f}")""")

todo(
    """# TODO — implement GOSS row weights (Ke et al. 2017, Algorithm 2)
def goss_weights(g, a, b, rng):
    n = len(g)
    w = np.zeros(n)
    order = np.argsort(-np.abs(g))          # descending |gradient|
    top_n = int(a * n)
    top_idx = order[:top_n]
    rest_idx = order[top_n:]
    samp = rng.choice(rest_idx, size=int(b * n), replace=False)
    w[top_idx] = 1.0
    w[samp] = (1 - a) / b
    return w

w = goss_weights(g, A, B, np.random.RandomState(1))
est_left = (w[region] * g[region]).sum()    # amplified sum over the region
used = int((w > 0).sum())
print(f"rows used = {used}/{n} ({used/n:.0%})   amplify factor = {(1-A)/B:.1f}")
print(f"GOSS estimate = {est_left:.3f}   vs true {true_left:.3f}   abs err = {abs(est_left-true_left):.3f}")""",
    """# TODO — implement GOSS row weights (Ke et al. 2017, Algorithm 2)
def goss_weights(g, a, b, rng):
    n = len(g)
    w = np.zeros(n)
    order = ____                            # indices sorted by DESCENDING |g|
    top_n = int(a * n)
    top_idx = order[:top_n]
    rest_idx = order[top_n:]
    samp = rng.choice(rest_idx, size=int(b * n), replace=False)
    w[top_idx] = ____                       # weight for kept large-gradient rows
    w[samp] = ____                          # amplification for sampled small-gradient rows
    return w

w = goss_weights(g, A, B, np.random.RandomState(1))
est_left = (w[region] * g[region]).sum()    # amplified sum over the region
used = int((w > 0).sum())
print(f"rows used = {used}/{n} ({used/n:.0%})   amplify factor = {(1-A)/B:.1f}")
print(f"GOSS estimate = {est_left:.3f}   vs true {true_left:.3f}   abs err = {abs(est_left-true_left):.3f}")""",
)

code("""# CHECK — do not edit
assert w[np.argmax(np.abs(g))] == 1.0, "The largest-|g| row must be kept with weight 1.0."
frac_used = (w > 0).mean()
assert abs(frac_used - (A + B)) < 0.02, "Used fraction should be about a + b (top a*n plus b*n sampled)."
# averaged over several seeds, the GOSS estimate is close to the true sum (unbiased)
errs = []
for s in range(200):
    ws = goss_weights(g, A, B, np.random.RandomState(s))
    errs.append((ws[region] * g[region]).sum() - true_left)
assert abs(np.mean(errs)) < 0.15 * abs(true_left) + 1.0, "GOSS estimate should be ~unbiased on average."
print(f"Task 1 ok — GOSS uses ~{frac_used:.0%} of rows; mean estimate error over 200 seeds = {np.mean(errs):+.3f}.")""")

# ---------------------------------------------------------------- Task 2
md("""## Task 2 — watch leaf-wise overfit: the num_leaves sweep — TODO

**Goal:** sweep `num_leaves` on `credit_g` and show CV PR-AUC does **not** keep rising — leaf-wise trees overfit this small, noisy dataset as they get bigger.

**Why it matters:** migrating from XGBoost, people set a big `max_depth` and leave `num_leaves` huge, then overfit. Seeing the curve turn over here builds the instinct to tune `num_leaves` **down** first.

**You implement:** the list comprehension that scores an `LGBMClassifier(num_leaves=nl, n_estimators=200, learning_rate=0.05, random_state=RS, n_jobs=1, verbose=-1)` with `cv_pr_auc` at each `nl`.""")

code("""# PROVIDED — Tier A: credit_g via relkit (label-encoded, as in Labs 012-014)
X, y = load_tier_a("credit_g")
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
Xn = X[num_cols].copy()
for c in cat_cols:
    Xn[c] = LabelEncoder().fit_transform(X[c].astype(str))
y = np.asarray(y)
leaves_grid = [7, 15, 31, 63, 127]
print(f"rows={len(y)} pos_rate={y.mean():.3f} features={Xn.shape[1]}")""")

todo(
    """# TODO — score LightGBM at each num_leaves
scores_by_leaves = [
    cv_pr_auc(
        LGBMClassifier(num_leaves=nl, n_estimators=200, learning_rate=0.05,
                       random_state=RS, n_jobs=1, verbose=-1),
        Xn, y,
    )
    for nl in leaves_grid
]
for nl, s in zip(leaves_grid, scores_by_leaves):
    print(f"  num_leaves={nl:4d}  CV PR-AUC = {s:.3f}")""",
    """# TODO — score LightGBM at each num_leaves
scores_by_leaves = [
    ____                                   # cv_pr_auc of an LGBMClassifier with num_leaves=nl (see hint)
    for nl in leaves_grid
]
for nl, s in zip(leaves_grid, scores_by_leaves):
    print(f"  num_leaves={nl:4d}  CV PR-AUC = {s:.3f}")""",
)

code("""# CHECK — do not edit
assert len(scores_by_leaves) == len(leaves_grid)
assert all(0.5 < s < 1.0 for s in scores_by_leaves), "PR-AUC should be a sensible score."
assert scores_by_leaves[-1] <= max(scores_by_leaves) - 1e-9 or scores_by_leaves[0] >= scores_by_leaves[-1] - 1e-9, \\
    "The largest num_leaves should not be the sole winner — leaf-wise overfits small data."
print("Task 2 ok — bigger num_leaves does not keep improving; small trees win on small data.")""")

# ---------------------------------------------------------------- Task 3
md("""## Task 3 — LightGBM vs XGBoost on real data — TODO (reproduction target)

**Goal:** compare default LightGBM and default XGBoost on `credit_g` with the shared CV harness.

**Why it matters:** honest-baseline discipline. On this small set LightGBM's defaults *edge* XGBoost, but the two are close — the real LightGBM advantage is speed at scale (see the stretch cell), not a magic accuracy jump on tiny data.

**Reproduction target (lesson table):** XGBoost default ≈ 0.883, LightGBM default ≈ 0.889.

**You implement:** `lgbm_score` and `xgb_score` — `cv_pr_auc` of a default `LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1)` and a default `XGBClassifier(random_state=RS, n_jobs=1, eval_metric="logloss", tree_method="hist")`.""")

todo(
    """# TODO — default LightGBM vs default XGBoost on credit_g
lgbm_score = cv_pr_auc(LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1), Xn, y)
xgb_score = cv_pr_auc(
    XGBClassifier(random_state=RS, n_jobs=1, eval_metric="logloss", tree_method="hist"), Xn, y
)
print(f"LightGBM default  CV PR-AUC = {lgbm_score:.3f}")
print(f"XGBoost  default  CV PR-AUC = {xgb_score:.3f}")""",
    """# TODO — default LightGBM vs default XGBoost on credit_g
lgbm_score = ____              # cv_pr_auc of a default LGBMClassifier (see hint)
xgb_score = ____               # cv_pr_auc of a default XGBClassifier (see hint)
print(f"LightGBM default  CV PR-AUC = {lgbm_score:.3f}")
print(f"XGBoost  default  CV PR-AUC = {xgb_score:.3f}")""",
)

code("""# CHECK — do not edit
for v in (lgbm_score, xgb_score):
    assert 0.5 < v < 1.0, "PR-AUC should be a sensible probability-ranking score."
assert abs(lgbm_score - xgb_score) < 0.05, "On credit_g the two boosters should be close (near-tie)."
print(f"Task 3 ok — LightGBM {lgbm_score:.3f} vs XGBoost {xgb_score:.3f} (close, as expected on small data).")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, name LightGBM's leaf-wise complexity knob and say what GOSS keeps vs samples — and whether LightGBM beat XGBoost by much on this small dataset.""")

todo(
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 015 ===")
print(f"GOSS: used {used}/{n} rows ({used/n:.0%}); estimate {est_left:.2f} vs true {true_left:.2f}")
print("num_leaves sweep (credit_g): " + "  ".join(f"{nl}={s:.3f}" for nl, s in zip(leaves_grid, scores_by_leaves)))
print(f"credit_g PR-AUC: LightGBM {lgbm_score:.3f} | XGBoost {xgb_score:.3f}")
print()
print("takeaway:", "num_leaves is LightGBM's leaf-wise capacity knob (lower it on small data); "
      "GOSS keeps the large-|gradient| rows and subsamples the small-gradient rest (amplified by (1-a)/b); "
      "on this small credit_g set LightGBM only edged XGBoost — the real win is speed at scale.")''',
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 015 ===")
print(f"GOSS: used {used}/{n} rows ({used/n:.0%}); estimate {est_left:.2f} vs true {true_left:.2f}")
print("num_leaves sweep (credit_g): " + "  ".join(f"{nl}={s:.3f}" for nl, s in zip(leaves_grid, scores_by_leaves)))
print(f"credit_g PR-AUC: LightGBM {lgbm_score:.3f} | XGBoost {xgb_score:.3f}")
print()
print("takeaway:", "____")''',
)

# ---------------------------------------------------------------- stretch
md("""## Stretch (optional, ungraded) — whose speedup was it, really?

The paper's famous "up to 20× faster" was measured against **conventional, pre-histogram** GBDT (like sklearn's `GradientBoostingClassifier`) — *not* against modern XGBoost, which later adopted the same histogram trick. Time all three on a moderate synthetic set and see for yourself: both histogram boosters crush the conventional one, but LightGBM and XGBoost-`hist` are roughly on par. (Also note: LightGBM's default `boosting_type` is `'gbdt'`, so **GOSS is opt-in** via `boosting_type='goss'`.)""")

code("""# STRETCH — ungraded. The honest speedup: hist boosters vs conventional GBDT.
# from sklearn.datasets import make_classification
# from sklearn.ensemble import GradientBoostingClassifier
# Xb, yb = make_classification(n_samples=50_000, n_features=50, n_informative=15,
#                              n_redundant=0, random_state=RS)
# def ft(est):
#     t0 = time.perf_counter(); est.fit(Xb, yb); return time.perf_counter() - t0
# print("sklearn GBDT (pre-hist):", round(ft(GradientBoostingClassifier(n_estimators=100, random_state=RS)), 1), "s")
# print("XGBoost (hist)         :", round(ft(XGBClassifier(n_estimators=100, random_state=RS, n_jobs=4, eval_metric='logloss', tree_method='hist')), 1), "s")
# print("LightGBM (gbdt)        :", round(ft(LGBMClassifier(n_estimators=100, random_state=RS, n_jobs=4, verbose=-1)), 1), "s")
# print("LightGBM (goss)        :", round(ft(LGBMClassifier(n_estimators=100, random_state=RS, n_jobs=4, verbose=-1, boosting_type='goss')), 1), "s")""")


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l015-{i:02d}"
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


(LABS / "0015-lightgbm.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0015-lightgbm.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0015-lightgbm.ipynb and solutions/0015-lightgbm.ipynb")
