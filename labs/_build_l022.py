"""Build Lab 022 (label leakage patterns) — emits the blank student notebook and a
filled solution notebook. Mirrors the L021 concept-lab structure.

Run: .venv/bin/python labs/_build_l022.py
Then execute the solution: .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0022-label-leakage-patterns.ipynb labs/solutions/0022-label-leakage-patterns.ipynb
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# ---- shared PROVIDED setup (identical to labs/_verify_l022.py) ----
SETUP = r'''# PROVIDED — three data generators (mechanism isolation) + models. Just run.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 0

def make_legit(n=4000, d=8, seed=SEED):
    """A weak, mostly-linear signal. RF and LR both land near 0.72 — an honest tie."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([0.9, -0.8, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0])
    logit = X @ w + 1.8 * rng.normal(size=n)
    return X, (logit > 0).astype(int)

def leaked_nonmonotone(y, flip=0.06, seed=SEED):
    """ILLEGITIMATE feature: encodes y NON-MONOTONICALLY.
       y==1 -> outer bands [0,0.33) U [0.66,1);  y==0 -> middle band [0.33,0.66).
    A tree carves it out with two splits; a linear model sees class means ~0.5 (no
    usable direction). This is a downstream/proxy column absent at prediction time."""
    rng = np.random.default_rng(seed + 1)
    noisy = np.where(rng.random(len(y)) < flip, 1 - y, y)
    val = np.empty(len(y))
    for i, yi in enumerate(noisy):
        if yi == 1:
            val[i] = rng.uniform(0.66, 1.0) if rng.random() < 0.5 else rng.uniform(0.0, 0.33)
        else:
            val[i] = rng.uniform(0.33, 0.66)
    return val.reshape(-1, 1)

def make_with_dups(n_unique=1500, dup_frac=0.6, d=6, seed=SEED):
    """Unique records + near-duplicate copies (same record, tiny jitter).
    Returns X, y, groups (groups = the underlying record id)."""
    rng = np.random.default_rng(seed + 2)
    Xu = rng.normal(size=(n_unique, d))
    w = np.array([1.2, -1.0, 0.8, 0.0, 0.0, 0.0])
    yu = ((Xu @ w + 0.5 * rng.normal(size=n_unique)) > 0).astype(int)
    gu = np.arange(n_unique)
    nd = int(n_unique * dup_frac)
    pick = rng.integers(0, n_unique, size=nd)
    Xd = Xu[pick] + 0.01 * rng.normal(size=(nd, d))
    X = np.vstack([Xu, Xd]); y = np.concatenate([yu, yu[pick]]); g = np.concatenate([gu, gu[pick]])
    p = rng.permutation(len(y))
    return X[p], y[p], g[p]

def rf():
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

X, y = make_legit()
leak = leaked_nonmonotone(y)
cv5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"legit data: n={len(X)}, d={X.shape[1]}, prevalence={y.mean():.3f}; leaked column shape {leak.shape}")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — reproduce the collapse — TODO (crucial fragment)

**Goal:** measure a complex model (Random Forest) and logistic regression **with** and **without** the illegitimate leaked column, and watch the apparent gap open then collapse.

**Why it matters:** this is the Kapoor & Narayanan civil-war finding in miniature — a leak can *manufacture* a "complex ML beats the simple model" conclusion. Remove one column and the win evaporates.

**You implement:** (1) build the leaked design matrix by column-stacking `X` and `leak`; (2) score each model with `cross_val_score(..., cv=cv5, scoring="accuracy").mean()` on both the honest `X` and the leaked matrix.

**Hint boundary:** `np.hstack([...])` glues columns; call the provided `rf()` / `lr()` factories fresh for each score. Do not change `cv5`.'''

T1_CODE = r'''# TODO — measure the collapse
def acc(model_fn, XX):
    return cross_val_score(model_fn(), XX, y, cv=cv5, scoring="accuracy").mean()

Xleak = ____                              # column-stack the honest X with the leaked column

rf_honest = acc(rf, ____)                 # RF on the honest features
lr_honest = acc(lr, ____)                 # LR on the honest features
rf_leak   = acc(rf, ____)                 # RF WITH the leaked column
lr_leak   = acc(lr, ____)                 # LR WITH the leaked column

print(f"WITH leak:     RF {rf_leak:.3f}   LR {lr_leak:.3f}   gap(RF-LR) {rf_leak-lr_leak:+.3f}")
print(f"WITHOUT leak:  RF {rf_honest:.3f}   LR {lr_honest:.3f}   gap(RF-LR) {rf_honest-lr_honest:+.3f}")'''

T1_SOL = T1_CODE.replace(
    "Xleak = ____                              # column-stack the honest X with the leaked column",
    "Xleak = np.hstack([X, leak])              # column-stack the honest X with the leaked column",
).replace("acc(rf, ____)                 # RF on the honest features", "acc(rf, X)                    # RF on the honest features") \
 .replace("acc(lr, ____)                 # LR on the honest features", "acc(lr, X)                    # LR on the honest features") \
 .replace("acc(rf, ____)                 # RF WITH the leaked column", "acc(rf, Xleak)                # RF WITH the leaked column") \
 .replace("acc(lr, ____)                 # LR WITH the leaked column", "acc(lr, Xleak)                # LR WITH the leaked column")

T1_CHECK = r'''# CHECK — do not edit
assert rf_leak - lr_leak > 0.15, "WITH the leak, RF should appear to crush LR (gap > 0.15)."
assert abs(rf_honest - lr_honest) < 0.05, "WITHOUT the leak, RF and LR should be ~tied."
assert rf_leak - rf_honest > 0.15, "The leak should inflate RF's score substantially."
print(f"Task 1 ok — leak inflates the RF-vs-LR gap from {rf_honest-lr_honest:+.3f} (honest) "
      f"to {rf_leak-lr_leak:+.3f} (leaked). The 'complex model wins' was the leak.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — the near-duplicate leak — TODO (crucial fragment)

**Goal:** show that when the *same underlying record* appears twice (near-duplicates), a random CV leaks — a training twin gives away its test copy. Fix it with a **grouped** split so no record straddles a fold.

**Why it matters:** the unit of independence is the *record*, not the row (taxonomy L1.4 / L3.2). This is the L004 grouped-split discipline, now recognized as a named leak type.

**You implement:** score `rf()` two ways — a naïve `KFold(shuffle=True)` and a `GroupKFold(n_splits=5)` that receives `groups=g` so copies of a record stay on one side.

**Hint boundary:** `GroupKFold(n_splits=5)`; pass `groups=g` to `cross_val_score`. The naïve one uses `cv5` (no groups).'''

T2_CODE = r'''# TODO — measure the duplicate leak
Xd, yd, g = make_with_dups()

naive   = cross_val_score(rf(), Xd, yd, cv=____, scoring="accuracy").mean()          # random KFold (leaks twins)
grouped = cross_val_score(rf(), Xd, yd, cv=____, groups=____, scoring="accuracy").mean()  # GroupKFold on record id

print(f"naive random 5-fold CV   {naive:.3f}   (copies straddle train/test -> optimistic)")
print(f"GroupKFold on record id  {grouped:.3f}   (no record straddles -> honest)")
print(f"leak size = {naive - grouped:+.3f}")'''

T2_SOL = T2_CODE.replace(
    "cv=____, scoring=\"accuracy\").mean()          # random KFold (leaks twins)",
    "cv=cv5, scoring=\"accuracy\").mean()           # random KFold (leaks twins)",
).replace(
    "cv=____, groups=____, scoring=\"accuracy\").mean()  # GroupKFold on record id",
    "cv=GroupKFold(n_splits=5), groups=g, scoring=\"accuracy\").mean()  # GroupKFold on record id",
)

T2_CHECK = r'''# CHECK — do not edit
assert naive - grouped > 0.03, "The naive random CV should be clearly optimistic vs the grouped CV."
assert grouped < naive, "GroupKFold (no record straddling a fold) is the honest, lower number."
print(f"Task 2 ok — duplicates inflate the score by {naive - grouped:+.3f}. "
      f"Group by the record id (L004 discipline) and the phantom disappears.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — spot the leak in feature engineering — TODO (the core skill)

**Goal:** classify each feature-engineering choice into the Kapoor & Narayanan taxonomy code. This is the lesson's headline skill: *name the leak before it inflates your score.*

**Taxonomy codes:**
- `L1.1` no held-out test set · `L1.2` preprocessing on all data · `L1.3` feature selection on all data · `L1.4` duplicates across the split
- `L2` illegitimate feature (proxy for / unavailable-at-prediction-time)
- `L3.1` temporal leakage · `L3.2` non-independence (same entity both sides) · `L3.3` sampling bias in the test set

**You implement:** fill each value in `answers` with the correct code string.

**Hint boundary:** map each scenario to exactly one code. Re-read the taxonomy widget in the lesson if unsure — do not guess the whole set.'''

T3_CODE = r'''# TODO — assign the taxonomy code (a string like "L2") to each scenario
scenarios = {
    "a": "Impute missing values with the column mean of the ENTIRE dataset before splitting",
    "b": "Keep the 20 features most correlated with y, computed over the whole dataset",
    "c": "Predict hospital readmission using days_in_hospital (known only at discharge)",
    "d": "Random split when the same patient contributes many visits",
    "e": "Random split of daily prices to predict tomorrow's move",
    "f": "Report accuracy on the exact rows the model trained on",
}
answers = {
    "a": ____,   # which code?
    "b": ____,
    "c": ____,
    "d": ____,
    "e": ____,
    "f": ____,
}
for k, v in scenarios.items():
    print(f"  ({k}) [{answers[k]}]  {v}")'''

T3_SOL = T3_CODE.replace('"a": ____,   # which code?', '"a": "L1.2",  # preprocessing on all data') \
    .replace('"b": ____,', '"b": "L1.3",  # feature selection on all data') \
    .replace('"c": ____,', '"c": "L2",    # illegitimate feature (post-outcome)') \
    .replace('"d": ____,', '"d": "L3.2",  # non-independence (same patient)') \
    .replace('"e": ____,', '"e": "L3.1",  # temporal leakage') \
    .replace('"f": ____,', '"f": "L1.1",  # no held-out test set')

T3_CHECK = r'''# CHECK — do not edit
key = {"a": "L1.2", "b": "L1.3", "c": "L2", "d": "L3.2", "e": "L3.1", "f": "L1.1"}
wrong = {k: answers[k] for k in key if answers[k] != key[k]}
assert not wrong, f"Reclassify these: {wrong}"
print(f"Task 3 ok — all 6 scenarios classified correctly. You can spot the leak in FE.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why is a large complex-vs-simple performance gap a *leak hypothesis first*, and what single practice would have caught every civil-war error?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 022 ===")
print(f"collapse : WITH leak  RF {rf_leak:.3f} / LR {lr_leak:.3f}  gap {rf_leak-lr_leak:+.3f}")
print(f"           NO   leak  RF {rf_honest:.3f} / LR {lr_honest:.3f}  gap {rf_honest-lr_honest:+.3f}")
print(f"dup leak : random {naive:.3f} -> grouped {grouped:.3f}  ({naive-grouped:+.3f})")
print(f"FE audit : 6/6 scenarios classified into the taxonomy")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A big complex-vs-simple gap usually means the flexible model exploited a leak harder, '
    'not that it models the real signal better; a model info sheet with a question per leak type '
    'would have surfaced every civil-war error before publication."',
)


def build(solution: bool):
    cells = [
        md(r'''# Lab 022 — Label leakage patterns: spot it, reproduce the collapse

**Lesson:** [`lessons/0022-label-leakage-patterns.html`](../lessons/0022-label-leakage-patterns.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Kapoor & Narayanan 2022, *Leakage and the Reproducibility Crisis in ML-based Science*, [arXiv:2207.07048](https://arxiv.org/abs/2207.07048) — abstract + §2 (taxonomy of 8 leak types), §5 (civil-war reproduction), §6 (model info sheet).

**Dataset tier:** C — synthetic (mechanism isolation). We isolate two leaks on controlled data so the effect is real and reproducible; this is *not* a benchmark.

**Skill you are practising:** classify a feature-engineering choice into the leak taxonomy, and reproduce how an **illegitimate feature** manufactures a false "complex model wins" result (the *reproducibility collapse*).

**Exit criteria:** EXIT TICKET prints the collapse (with/without leak), the duplicate leak, the FE-audit result, and your one-sentence takeaway. Reproduction targets: leaked gap ≈ +0.217, honest gap ≈ 0.000; dup leak 0.948 → 0.876.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** and **numpy** only (no boosters, no network).'''),
        md(r'''## Concept recap — leakage is a family with a map

**Data leakage** = anything the model learns at training time that it could not legitimately know at prediction time. Kapoor & Narayanan (2022) found it in 17 fields (329 papers) and gave it a **taxonomy of 8 types in 3 families**:

- **Family 1 — no clean train/test separation:** `L1.1` no test set · `L1.2` preprocessing on all data · `L1.3` feature selection on all data · `L1.4` duplicates across the split.
- **Family 2 — illegitimate features:** `L2` a column that proxies the label or would not exist at prediction time.
- **Family 3 — test set ≠ distribution of interest:** `L3.1` temporal leakage · `L3.2` non-independence (same entity both sides) · `L3.3` sampling bias.

**The dangerous one (L2).** An illegitimate feature passes every pipeline and split check — it just looks *very predictive*, because it is a stand-in for the answer. A flexible model exploits it harder than a linear one, so a leak can make a complex model look like it crushes logistic regression.

**Toy micro-example (not this lab's answer).** Predicting whether a loan *defaults*, you include `recovered_amount` — only ever nonzero after a default. Train accuracy 0.99, deploy accuracy 0.60: the column is a proxy for the label. The fix is provenance, not statistics: could the value be known strictly *before* the label event?

Full write-up + the taxonomy map and collapse widgets: [Lesson 022](../lessons/0022-label-leakage-patterns.html).'''),
        md("## Setup — PROVIDED"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — push the mechanism

1. **Make the leak monotone.** Replace `leaked_nonmonotone` with a plain noisy copy of `y` (e.g. `y + 0.3*randn`). Now *both* RF and LR jump — the complex-vs-LR gap does **not** open, because a linear model can use a monotone leak too. The apparent-win effect needs a leak the flexible model exploits more.
2. **Leak strength.** Sweep the `flip` rate in `leaked_nonmonotone` from 0.0 to 0.5 and plot the leaked RF accuracy — the leak fades smoothly into the honest level.
3. **Dup fraction.** Sweep `dup_frac` in `make_with_dups`; the naïve/grouped gap grows with more duplicates.
4. **Write the sheet.** Fill in the 7-line model info sheet from the lesson for a project you actually have — the first box you cannot honestly tick is your leak.'''),
        code(r'''# STRETCH — ungraded.
# monotone = (y + 0.3 * np.random.default_rng(0).normal(size=len(y))).reshape(-1, 1)
# Xm = np.hstack([X, monotone])
# print("monotone leak: RF", round(acc(rf, Xm),3), " LR", round(acc(lr, Xm),3), " (gap barely opens)")'''),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    for c in nb["cells"]:
        if isinstance(c["source"], str):
            c["source"] = c["source"].splitlines(keepends=True)

    return nb


def main():
    student = build(solution=False)
    sol = build(solution=True)
    with open(os.path.join(HERE, "0022-label-leakage-patterns.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0022-label-leakage-patterns.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0022-label-leakage-patterns.ipynb + labs/solutions/0022-label-leakage-patterns.ipynb")


if __name__ == "__main__":
    main()
