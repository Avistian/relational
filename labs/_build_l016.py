"""Build labs/0016-catboost.ipynb (student, blanks) and solutions/0016-catboost.ipynb (filled).

Run:  python labs/_build_l016.py
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
md("""# Lab 016 — CatBoost: ordered target statistics, no leak

**Lesson:** [`lessons/0016-catboost.html`](../lessons/0016-catboost.html) · **Phase / Year:** Year 1 · Q2

**Paper:** Prokhorenkova et al. 2018, *CatBoost: unbiased boosting with categorical features* (NeurIPS 31), arXiv [1706.09516](https://arxiv.org/abs/1706.09516) — §3 (target statistics / prediction shift), §4 (ordered boosting).

**Dataset tiers:** C — synthetic pure-noise category (mechanism isolation) for Task 1; A — OpenML `credit_g` via `relkit` for Task 2 (same harness as Labs 011–015).

**Skill you are practising:** implement the **ordered target statistic** by hand and prove it is leakage-free exactly where the greedy (whole-column) statistic leaks; then run CatBoost with native categoricals against label-encoded XGBoost and native LightGBM on real data.

**Exit criteria:** EXIT TICKET prints (1) the greedy vs ordered AUC on a pure-noise category (leak vs no leak) and (2) CatBoost vs XGBoost vs LightGBM CV PR-AUC on credit_g.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. This lab needs **catboost** (added to `requirements-labs.txt`). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — "only use the past"

A categorical column must become a number for a tree to split on it. The strongest encoding is a **target statistic (TS)**: replace each category with the mean target for that category. Computed naively it **leaks**, because to encode row $i$ you average the target over *all* rows of that category **including row $i$'s own label** $y_i$ — for a rare category, the encoding basically *is* $y_i$.

**CatBoost's fix (ordered TS).** Draw a random **permutation** of the rows and encode row $i$ using only the rows *before* it in that permutation (its **prefix**), plus a smoothing prior:

$$\text{TS}_i = \frac{\sum_{j<i,\; \text{cat}_j=\text{cat}_i} y_j \;+\; a\cdot p}{\#\{j<i:\text{cat}_j=\text{cat}_i\} \;+\; a}$$

where $p$ is the global target mean (the prior) and $a$ its strength. Row $i$'s own label is never used — the same **point-in-time** discipline as Lesson 002. The first time a category appears (empty prefix) the TS is just the prior $p$.

**Ordered boosting (§4)** applies the identical idea to the boosting loop: the gradient for row $i$ is computed by a model trained only on rows *before* $i$, so it never sees row $i$ — removing the **prediction shift** bias of classic boosting. It is the default (`boosting_type="Ordered"`) on small data, slower than `"Plain"`.

**Oblivious trees.** CatBoost's base learner reuses one split (feature+threshold) per level, so a depth-6 tree is a **decision table** with $2^6=64$ leaves: weaker per tree, but strongly regularized and very fast to score.

**Toy micro-example (not this lab's answer).** Permutation of category labels `[A, A, B, A]` with targets `[1, 0, 1, 1]` and prior $p=0.75, a=1$. The ordered TS of the *last* A (targets of earlier A's are `1, 0`) is $\frac{(1+0)+1\cdot0.75}{2+1}=\frac{1.75}{3}=0.583$ — it never touches that row's own $y=1$.

**Why it matters for the thesis:** relational databases are built from high-cardinality categorical foreign keys (`user_id`, `product_id`). Ordered TS is the leakage-free single-table way to encode entity identity — a baseline RDL must beat, and the same "use only the past" rule RDL enforces with timestamps on the entity graph. Full derivation + viz: [Lesson 016](../lessons/0016-catboost.html) · Prokhorenkova et al. 2018 §3–4.""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))  # labs/ on the path so `relkit` imports

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from relkit.data import load_tier_a

RS = 0
print("relkit + catboost + lightgbm + xgboost ok")""")

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — the ordered target statistic, by hand — TODO (crucial fragment)

**Goal:** implement `ordered_ts(cat, y, a, prior, seed)` and show that on a **pure-noise** category (target independent of the category) the greedy encoding *invents* signal (AUC ≫ 0.5) while your ordered encoding does not (AUC ≈ 0.5).

**Why it matters:** this is the leak that quietly inflates models trained on high-cardinality categoricals — the single most common way a "great" offline score fails in production. Seeing greedy TS score 0.8+ AUC on a column that is literally random makes the danger concrete.

**You implement `ordered_ts`:** walk the rows in the order given by a random `permutation`. For each row, *first* compute its encoding from what you have seen so far, `(running_sum + a*prior) / (running_count + a)`, *then* add its own `y` to the running totals for its category. (Encode before you update — that is what keeps the row's own label out.)

**Hint boundary:** keep two dicts (sum and count) keyed by category; the order of "encode, then update" is the whole point. The greedy baseline is PROVIDED for contrast — do not edit it or the CHECK.""")

code("""# PROVIDED — a pure-noise, high-cardinality category. Target is INDEPENDENT of cat.
rng = np.random.RandomState(0)
n, K = 2000, 700                       # ~3 rows per category
cat = rng.randint(0, K, size=n)
y = (rng.rand(n) < 0.3).astype(int)    # no relationship to `cat`
prior = y.mean()

def greedy_ts(cat, y):
    \"\"\"Whole-category mean, INCLUDING each row's own y -> leak.\"\"\"
    out = np.empty(len(y), float)
    for k in np.unique(cat):
        m = cat == k
        out[m] = y[m].mean()
    return out

print(f"pure-noise category: n={n}, cardinality={K}, prevalence={prior:.3f}")""")

todo(
    """# TODO — implement ordered target statistics (Prokhorenkova et al. 2018, §3)
def ordered_ts(cat, y, a, prior, seed):
    perm = np.random.RandomState(seed).permutation(len(y))
    running_sum, running_cnt = {}, {}
    enc = np.empty(len(y), float)
    for i in perm:
        c = cat[i]
        s = running_sum.get(c, 0.0)
        m = running_cnt.get(c, 0)
        enc[i] = (s + a * prior) / (m + a)     # encode from the PREFIX only
        running_sum[c] = s + y[i]              # THEN update with this row's y
        running_cnt[c] = m + 1
    return enc

auc_greedy = roc_auc_score(y, greedy_ts(cat, y))
auc_ordered = np.mean([roc_auc_score(y, ordered_ts(cat, y, 1.0, prior, s)) for s in range(20)])
print(f"greedy  TS AUC(enc, y) = {auc_greedy:.3f}   (leak)")
print(f"ordered TS AUC(enc, y) = {auc_ordered:.3f}   (mean over 20 perms; ~0.5 = no leak)")""",
    """# TODO — implement ordered target statistics (Prokhorenkova et al. 2018, §3)
def ordered_ts(cat, y, a, prior, seed):
    perm = np.random.RandomState(seed).permutation(len(y))
    running_sum, running_cnt = {}, {}
    enc = np.empty(len(y), float)
    for i in perm:
        c = cat[i]
        s = running_sum.get(c, 0.0)
        m = running_cnt.get(c, 0)
        enc[i] = ____                          # encode from the PREFIX only: (s + a*prior)/(m + a)
        running_sum[c] = ____                  # THEN update the running sum with this row's y
        running_cnt[c] = ____                  # ...and its count
    return enc

auc_greedy = roc_auc_score(y, greedy_ts(cat, y))
auc_ordered = np.mean([roc_auc_score(y, ordered_ts(cat, y, 1.0, prior, s)) for s in range(20)])
print(f"greedy  TS AUC(enc, y) = {auc_greedy:.3f}   (leak)")
print(f"ordered TS AUC(enc, y) = {auc_ordered:.3f}   (mean over 20 perms; ~0.5 = no leak)")""",
)

code("""# CHECK — do not edit
enc0 = ordered_ts(cat, y, 1.0, prior, 0)
assert enc0.shape == y.shape
assert auc_greedy > 0.70, "Greedy TS should leak on a rare-category noise column (AUC well above 0.5)."
assert 0.40 < auc_ordered < 0.60, "Ordered TS should be ~0.5 on pure noise (no leak)."
# The first occurrence of any category in a permutation must equal the prior (empty prefix).
perm = np.random.RandomState(0).permutation(len(y))
seen = set(); first_ok = True
for i in perm:
    if cat[i] not in seen:
        first_ok = first_ok and abs(enc0[i] - prior) < 1e-9
        seen.add(cat[i])
assert first_ok, "The first time a category appears, its ordered TS must equal the prior."
print(f"Task 1 ok — greedy leaks (AUC {auc_greedy:.3f}); ordered is clean (AUC {auc_ordered:.3f}).")""")

# ---------------------------------------------------------------- Task 2
md("""## Task 2 — CatBoost vs the field on real categorical data — TODO (reproduction target)

**Goal:** compare CatBoost (native categoricals) against label-encoded XGBoost and native-categorical LightGBM on `credit_g`, all with a shared 5-fold CV PR-AUC.

**Why it matters:** honest-baseline discipline. `credit_g` has 13 categorical columns on only 1,000 rows — CatBoost's home turf. But a fair claim needs all three under the *same* protocol.

**Reproduction target (lesson table):** XGBoost ≈ 0.883, LightGBM ≈ 0.880, CatBoost ≈ 0.896.

**You implement:** the categorical-column **indices** `cat_idx` that CatBoost needs, and pass them to the CatBoost factory via `cat_features=...`. (A manual 5-fold helper is PROVIDED because CatBoost's `cat_features` param does not survive `sklearn.clone`, which `cross_val_score` uses.)""")

code("""# PROVIDED — credit_g in three forms + a clone-free CV helper.
X, y = load_tier_a("credit_g")
y = np.asarray(y)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# (a) native strings for CatBoost
Xs = X.copy()
for c in cat_cols:
    col = Xs[c].astype("object")
    Xs[c] = col.where(col.notna(), "missing").astype(str)
# (b) label-encoded for XGBoost
Xle = Xs[num_cols].copy()
for c in cat_cols:
    Xle[c] = LabelEncoder().fit_transform(Xs[c])
# (c) pandas 'category' dtype for LightGBM native mode
Xlg = Xs.copy()
for c in cat_cols:
    Xlg[c] = Xlg[c].astype("category")

def cv_pr_auc_factory(make_estimator, Xdf):
    \"\"\"5-fold PR-AUC. `make_estimator` is a 0-arg factory (avoids sklearn.clone).\"\"\"
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    Xa = Xdf.reset_index(drop=True)
    out = []
    for tr, te in cv.split(Xa, y):
        m = make_estimator()
        m.fit(Xa.iloc[tr], y[tr])
        out.append(average_precision_score(y[te], m.predict_proba(Xa.iloc[te])[:, 1]))
    return float(np.mean(out))

print(f"rows={len(y)} pos_rate={y.mean():.3f}  #cat={len(cat_cols)} #num={len(num_cols)}")""")

todo(
    """# TODO — CatBoost (native cats) vs XGBoost (label-enc) vs LightGBM (native)
cat_idx = [Xs.columns.get_loc(c) for c in cat_cols]     # positions of categorical columns

cb_score = cv_pr_auc_factory(lambda: CatBoostClassifier(
    iterations=400, learning_rate=0.05, depth=6, random_seed=RS,
    cat_features=cat_idx, verbose=0, allow_writing_files=False), Xs)
xgb_score = cv_pr_auc_factory(lambda: XGBClassifier(
    random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist"), Xle)
lgbm_score = cv_pr_auc_factory(lambda: LGBMClassifier(
    random_state=RS, n_jobs=4, verbose=-1), Xlg)

print(f"XGBoost  (label-enc) CV PR-AUC = {xgb_score:.3f}")
print(f"LightGBM (native)    CV PR-AUC = {lgbm_score:.3f}")
print(f"CatBoost (native)    CV PR-AUC = {cb_score:.3f}")""",
    """# TODO — CatBoost (native cats) vs XGBoost (label-enc) vs LightGBM (native)
cat_idx = ____     # list of column POSITIONS for the categorical columns (see hint: .get_loc)

cb_score = cv_pr_auc_factory(lambda: CatBoostClassifier(
    iterations=400, learning_rate=0.05, depth=6, random_seed=RS,
    cat_features=____, verbose=0, allow_writing_files=False), Xs)   # pass the categorical indices
xgb_score = cv_pr_auc_factory(lambda: XGBClassifier(
    random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist"), Xle)
lgbm_score = cv_pr_auc_factory(lambda: LGBMClassifier(
    random_state=RS, n_jobs=4, verbose=-1), Xlg)

print(f"XGBoost  (label-enc) CV PR-AUC = {xgb_score:.3f}")
print(f"LightGBM (native)    CV PR-AUC = {lgbm_score:.3f}")
print(f"CatBoost (native)    CV PR-AUC = {cb_score:.3f}")""",
)

code("""# CHECK — do not edit
for v in (cb_score, xgb_score, lgbm_score):
    assert 0.5 < v < 1.0, "PR-AUC should be a sensible probability-ranking score."
assert abs(xgb_score - 0.883) < 0.03, "XGBoost should reproduce ~0.883 on credit_g."
assert cb_score >= max(xgb_score, lgbm_score) - 0.01, \\
    "CatBoost's native categoricals should be at least competitive on this categorical-rich set."
print(f"Task 2 ok — CatBoost {cb_score:.3f} | XGBoost {xgb_score:.3f} | LightGBM {lgbm_score:.3f}.")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, say what the greedy target statistic leaks and how ordered TS removes the leak — and whether CatBoost's native categoricals beat the field on this small, categorical-rich dataset.""")

todo(
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 016 ===")
print(f"Pure-noise category: greedy TS AUC {auc_greedy:.3f} (leak) vs ordered TS AUC {auc_ordered:.3f} (clean)")
print(f"credit_g PR-AUC: CatBoost {cb_score:.3f} | XGBoost {xgb_score:.3f} | LightGBM {lgbm_score:.3f}")
print()
print("takeaway:", "greedy target encoding leaks each row's own label into its feature (worst on rare "
      "categories); ordered TS encodes each row from only earlier rows in a random permutation plus a prior, "
      "so the label never leaks; on categorical-rich credit_g CatBoost's native categoricals beat XGBoost/LightGBM.")''',
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 016 ===")
print(f"Pure-noise category: greedy TS AUC {auc_greedy:.3f} (leak) vs ordered TS AUC {auc_ordered:.3f} (clean)")
print(f"credit_g PR-AUC: CatBoost {cb_score:.3f} | XGBoost {xgb_score:.3f} | LightGBM {lgbm_score:.3f}")
print()
print("takeaway:", "____")''',
)

# ---------------------------------------------------------------- stretch
md("""## Stretch (optional, ungraded) — is *ordered boosting* itself worth it?

CatBoost's second permutation trick is **ordered boosting** (`boosting_type="Ordered"`, the default on small data) vs classic **Plain** boosting. It fixes *prediction shift* but is slower — and on a single small dataset the accuracy difference can be a wash. Toggle it and see for yourself, then note that most of CatBoost's edge above came from the *ordered target statistics*, not from ordered boosting.""")

code('''# STRETCH — ungraded. Ordered vs Plain boosting on credit_g (expect a near-wash).
# common = dict(iterations=400, learning_rate=0.05, depth=6, random_seed=RS,
#               cat_features=cat_idx, verbose=0, allow_writing_files=False)
# ordered = cv_pr_auc_factory(lambda: CatBoostClassifier(boosting_type="Ordered", **common), Xs)
# plain   = cv_pr_auc_factory(lambda: CatBoostClassifier(boosting_type="Plain",   **common), Xs)
# print(f"Ordered {ordered:.3f}  |  Plain {plain:.3f}")''')


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l016-{i:02d}"
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


(LABS / "0016-catboost.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0016-catboost.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0016-catboost.ipynb and solutions/0016-catboost.ipynb")
