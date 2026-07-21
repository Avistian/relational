"""Build Lab 033 (When to stop feature engineering — measure the diminishing-returns curve) — student + solution.

Tier A (real credit_g via relkit). Domingos 2012 is an essay, not an architecture, so the "implement the
paper" scope (standard #18) is to OPERATIONALISE its three load-bearing claims as one controlled experiment:
  * "feature engineering is the key"           -> a fixed-order feature-budget loop
  * "overfitting has many faces"               -> the curve peaks then DECLINES below baseline
  * "more data beats a cleverer algorithm" +   -> the honest stopping rule: stop when the marginal gain
    L023 significance                             sinks into the CV noise band (Delta < sigma)

Three tasks:
  * Task 1 (crucial fragment) — the feature-budget loop: add hand features one at a time (fixed order) and
    record mean +/- std 5-fold CV ROC-AUC at each k. Model held fixed => movement is features alone.
  * Task 2 (crucial fragment) — the stopping rule: first k>=1 whose marginal gain is smaller than the CV
    std at that k (i.e. within noise); report where you'd stop and the peak.
  * Task 3 — model-dependence: rerun the SAME budget with a linear model; see the L009 asymmetry (a strong
    tree already synthesises interactions, so it gains ~nothing; both stay inside their noise bands).

Numbers reproduce labs/_verify_l033.py exactly (seed 0, 5-fold): gbdt peaks 0.7911 at k=3 then falls to
0.7659 at k=8 (below the 0.7865 baseline); linear drifts ~+0.006, also within noise.

Run: .venv/bin/python labs/_build_l033.py
Then execute the solution (threads pinned so HistGB doesn't oversubscribe):
  OMP_NUM_THREADS=4 .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0033-when-to-stop-feature-engineering.ipynb \
    labs/solutions/0033-when-to-stop-feature-engineering.ipynb
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from _colab import bootstrap_cells  # noqa: E402


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


SETUP = r'''# PROVIDED — imports, real credit_g (Tier A), the fixed feature list, and a leak-safe scorer. Just run.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 0
K = 5

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
X_raw, y = load_tier_a("credit_g")
y = np.asarray(y)
print(f"credit_g (Tier A): X {X_raw.shape}, P(good) {y.mean():.3f}")

# PROVIDED — the analyst's fixed feature list (domain knowledge is NOT the skill here; the EXPERIMENT is).
# Ordered in advance by "what I'd try first", NOT greedily on the label — so the curve is an honest
# effort trace. Each entry: (name, function of the raw frame -> a Series).
FEATURES = [
    ("monthly_payment",     lambda d: d["credit_amount"] / d["duration"].clip(lower=1)),
    ("amount_per_age",      lambda d: d["credit_amount"] / d["age"].clip(lower=1)),
    ("installment_burden",  lambda d: d["installment_commitment"] * d["credit_amount"] / 100.0),
    ("amount_x_duration",   lambda d: d["credit_amount"] * d["duration"]),
    ("credit_per_existing", lambda d: d["credit_amount"] / d["existing_credits"].clip(lower=1)),
    ("age_x_residence",     lambda d: d["age"] * d["residence_since"]),
    ("log_amount",          lambda d: np.log1p(d["credit_amount"])),
    ("dur_per_age",         lambda d: d["duration"] / d["age"].clip(lower=1)),
    ("deps_per_credit",     lambda d: d["num_dependents"] / d["existing_credits"].clip(lower=1)),
    ("amount_sq",           lambda d: d["credit_amount"].astype(float) ** 2),
]
MAXK = len(FEATURES)

def cv_auc_std(X, model="gbdt"):
    """Leak-safe 5-fold CV ROC-AUC -> (mean, std). Preprocessing is fit INSIDE each fold (Pipeline)."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if model == "gbdt":
        num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        clf = HistGradientBoostingClassifier(random_state=SEED)
    else:
        num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
    pre = ColumnTransformer([("num", Pipeline(num_steps), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(K, shuffle=True, random_state=SEED)
    s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std())

def with_k_features(k):
    """Return X_raw plus the FIRST k hand features (fixed order)."""
    Xk = X_raw.copy()
    for name, fn in FEATURES[:k]:
        Xk[name] = fn(X_raw)
    return Xk

print(f"ready — {MAXK} candidate hand features; helpers: cv_auc_std(X, 'gbdt'|'linear'), with_k_features(k)")'''


# ---- Task 1: the feature-budget loop ----
T1_MD = r'''## Task 1 — the feature-budget loop (crucial fragment) — TODO

**Goal:** trace the diminishing-returns curve. For `k = 0, 1, …, 10`, build the design matrix with the
**first k** hand features (use the provided `with_k_features(k)`), measure 5-fold CV ROC-AUC **with its
standard deviation**, and collect `(k, mean, std)` into `curve`.

**Why it matters:** this is Domingos' essay turned into a measurement. The model is held fixed, so every
change in the curve is attributable to *features alone* — the only honest way to see whether more feature
effort is still paying (and the setup that lets Task 2 decide when to stop).

**Hint boundary:** loop `k` from `0` to `MAXK` inclusive; inside, call `cv_auc_std(with_k_features(k), "gbdt")`
to get `(mean, std)` and append `(k, mean, std)`.'''

T1_CODE = r'''# TODO — fill the loop body (the crucial fragment)
curve = []   # list of (k, mean_auc, std_auc) for the GBDT
for k in range(0, MAXK + 1):
    mean, std = ____                       # measure CV AUC (+/- std) with the first k hand features, GBDT
    curve.append((k, mean, std))
    print(f"k={k:>2}  AUC={mean:.4f}  +/-{std:.4f}")

means = [m for _, m, _ in curve]
stds  = [s for _, _, s in curve]'''

T1_SOL = T1_CODE.replace(
    '    mean, std = ____                       # measure CV AUC (+/- std) with the first k hand features, GBDT',
    '    mean, std = cv_auc_std(with_k_features(k), "gbdt")')

T1_CHECK = r'''# CHECK — do not edit
assert len(curve) == MAXK + 1, "curve should have one row per k = 0..10."
assert 0.78 <= means[0] <= 0.80, "GBDT baseline (k=0) should be ~0.7865 on credit_g."
peak_k = int(np.argmax(means))
assert peak_k in (2, 3), f"the GBDT curve should peak at ~3 features, got k={peak_k}."
assert means[8] < means[0], "by k=8 the curve should fall BELOW the no-feature baseline (the overfitting tax)."
print(f"Task 1 ok — curve traced. Baseline {means[0]:.4f}, peak {max(means):.4f} at k={peak_k}, "
      f"k=8 {means[8]:.4f} (below baseline).")'''


# ---- Task 2: the stopping rule ----
T2_MD = r'''## Task 2 — the stopping rule (crucial fragment) — TODO

**Goal:** decide where to stop. Write `first_within_noise(means, stds)` that returns the smallest `k >= 1`
whose **marginal gain** `means[k] - means[k-1]` is **smaller than the CV standard deviation** `stds[k]` —
i.e. the first feature whose contribution has sunk into the noise band. Then report the recommended stop
and the peak.

**Why it matters:** this is the lesson's whole skill. You do NOT stop when you run out of ideas; you stop
when the held-out curve flattens into the noise (the L023/L030 discipline: a gain smaller than sigma is
indistinguishable from luck of the split). Past that point extra features only add the overfitting tax.

**Hint boundary:** walk `k` from `1` upward; return the first `k` where `abs(means[k]-means[k-1]) < stds[k]`.
If none qualifies, return `len(means)-1`.'''

T2_CODE = r'''# TODO — implement the stopping rule (the crucial fragment)
def first_within_noise(means, stds):
    for k in range(1, len(means)):
        if ____:                              # marginal gain smaller than the CV std at this k?
            return k
    return len(means) - 1

stop_k = first_within_noise(means, stds)
peak_k = int(np.argmax(means))
print(f"recommended stop: k={stop_k} (marginal gain within noise)")
print(f"peak AUC {means[peak_k]:.4f} at k={peak_k}; gain over baseline "
      f"{means[peak_k]-means[0]:+.4f} vs noise band +/-{stds[peak_k]:.4f} "
      f"-> {'WITHIN noise' if abs(means[peak_k]-means[0]) < stds[peak_k] else 'exceeds 1 sigma'}")'''

T2_SOL = T2_CODE.replace(
    '        if ____:                              # marginal gain smaller than the CV std at this k?',
    '        if abs(means[k] - means[k - 1]) < stds[k]:')

T2_CHECK = r'''# CHECK — do not edit
assert 1 <= stop_k <= 3, f"on this curve you should stop very early (k<=3), got k={stop_k}."
assert abs(means[peak_k] - means[0]) < stds[peak_k], \
    "the peak gain over baseline should be WITHIN the CV noise band (not a real, provable win)."
print(f"Task 2 ok — stop at k={stop_k}; even the best point (+{means[peak_k]-means[0]:.4f}) is inside the "
      f"+/-{stds[peak_k]:.4f} band. On credit_g + a strong GBDT, single-table FE buys nothing provable.")'''


# ---- Task 3: model-dependence ----
T3_MD = r'''## Task 3 — model-dependence: the same features, a weaker model — TODO

**Goal:** rerun the SAME feature budget with a **linear** model (logistic regression) instead of the GBDT,
and compare the two curves' total drift (`means[-1] - means[0]`).

**Why it matters:** "diminishing returns" is always *for a given model*. A tree already synthesises
interactions through its splits (L009), so hand ratios/products add ~nothing; a linear model cannot, so in
principle FE should help it more — but on this fairly-linear dataset both stay inside their noise bands.
The takeaway is the **asymmetry**, not a winner: always ask "returns for which model?"

**Hint boundary:** reuse the Task 1 pattern but pass `"linear"` to `cv_auc_std`.'''

T3_CODE = r'''# TODO — trace the linear-model curve over the same budget
lin_means = []
for k in range(0, MAXK + 1):
    mean, _ = ____                            # CV AUC with the first k features, LINEAR model
    lin_means.append(mean)

print(f"GBDT   drift over 0->10 features: {means[-1]-means[0]:+.4f}")
print(f"Linear drift over 0->10 features: {lin_means[-1]-lin_means[0]:+.4f}")
print(f"GBDT   baseline {means[0]:.4f} | Linear baseline {lin_means[0]:.4f}")'''

T3_SOL = T3_CODE.replace(
    '    mean, _ = ____                            # CV AUC with the first k features, LINEAR model',
    '    mean, _ = cv_auc_std(with_k_features(k), "linear")')

T3_CHECK = r'''# CHECK — do not edit
assert len(lin_means) == MAXK + 1, "linear curve should have one row per k."
assert lin_means[0] > 0.78, "the linear baseline should already be strong (~0.79) — credit_g is fairly linear."
assert abs(lin_means[-1] - lin_means[0]) < 0.02, "linear total drift from FE should also be small (within noise)."
print(f"Task 3 ok — both models stay inside their noise bands (GBDT {means[-1]-means[0]:+.4f}, "
      f"linear {lin_means[-1]-lin_means[0]:+.4f}). Diminishing returns is model-relative; here neither buys a "
      f"provable single-table win.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** print your deliverable — the returns curve, where you'd stop, the peak vs the noise band, the
model-dependence — and a one-line takeaway.

**Takeaway prompt:** in one sentence, state the feature-engineering trade-off, give the rule for when to
stop adding hand features, and say why these single-table diminishing returns point at the relational thesis.'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 033 (when to stop feature engineering) ===")
print("GBDT returns curve (k: AUC +/- std):")
for k, m, s in curve:
    flag = "  <- peak" if k == int(np.argmax(means)) else ("  <- below baseline" if m < means[0] else "")
    print(f"  k={k:>2}  {m:.4f} +/-{s:.4f}{flag}")
print(f"recommended stop: k={stop_k}   peak: k={int(np.argmax(means))} ({max(means):.4f}, "
      f"{max(means)-means[0]:+.4f} vs baseline, within +/-{stds[int(np.argmax(means))]:.4f})")
print(f"model-dependence: GBDT drift {means[-1]-means[0]:+.4f} | linear drift {lin_means[-1]-lin_means[0]:+.4f}")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"Feature engineering is the key EARLY (it supplies structure a weak model cannot synthesise, L009) but '
    'its returns diminish and then go negative as each feature adds variance (Domingos: overfitting has many '
    'faces) -- so you STOP when the next feature\'s marginal held-out gain falls below the CV std (enters the '
    'noise band), not when you run out of ideas: on credit_g a strong GBDT peaked at just k=3 (+0.005, inside '
    'the +/-0.03 band) then fell BELOW baseline by k=8, and the linear model barely moved; the value of '
    'reshaping ONE table is nearly exhausted, so the features that would still pay are aggregates ACROSS '
    'related tables (DFS by hand, L009), which relational deep learning aims to LEARN end-to-end -- the '
    'returns moved across the join."')


STRETCH = r'''# STRETCH (optional, ungraded) — is the peak real? A corrected paired significance test (L023).
# Compares the k=0 baseline vs the k=peak model with the corrected resampled t-test (Nadeau & Bengio 2003),
# which inflates the naive variance by (1/n + n_test/n_train) to account for overlapping CV training sets.
# Expect p >> 0.05: the peak "gain" is not significant — the honest report is "no measurable FE gain".
#
# from scipy import stats
# def paired_scores(Xa, Xb, reps=5):
#     da, db = [], []
#     for r in range(reps):
#         cv = StratifiedKFold(K, shuffle=True, random_state=r)
#         # (reuse cv_auc_std internals per fold; left as an exercise — see L023 lab for the full harness)
#     return da, db
print("Stretch: see the L023 lab for the corrected resampled t-test harness; apply it to k=0 vs k=peak here.")'''


def build(solution: bool):
    cells = [
        md(r'''# Lab 033 — When to stop feature engineering: measure the diminishing-returns curve

**Lesson:** [`lessons/0033-when-to-stop-feature-engineering.html`](../lessons/0033-when-to-stop-feature-engineering.html) · **Phase / Year:** Year 1 · Q4

**Primary reading:** Domingos 2012, *A Few Useful Things to Know about Machine Learning*, CACM 55(10) — the sections *"Feature engineering is the key"*, *"More data beats a cleverer algorithm"*, *"Overfitting has many faces"* ([PDF](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)).

**Dataset tier:** **A** (real `credit_g` via `relkit`).

**Implementation scope (standard #18):** Domingos' paper is an *essay*, not an architecture — so instead of building a model, you **operationalise its three load-bearing claims as one controlled experiment**: a feature-budget loop (FE is the key), a curve that peaks then declines (overfitting has many faces), and an honest stopping rule tied to the CV noise band (don't celebrate noise — L023).

**Skill you are practising:** allocate a fixed modeling budget between feature engineering and everything else — trace the held-out returns curve, and stop adding hand features when the marginal gain sinks into the CV noise band.

**Exit criteria:** EXIT TICKET prints the GBDT returns curve, your recommended stop `k`, the peak vs the noise band, the GBDT-vs-linear model-dependence, and your one-line takeaway.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, the fixed feature list, a leak-safe scorer); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs scikit-learn (in `requirements-labs.txt`). `credit_g` is fetched from OpenML on first run then cached. The full lab runs in ~1–2 minutes on CPU (it fits ~130 small models across the budget × two model families).'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — effort, returns, and the noise band

A model can only combine features in the ways its hypothesis space allows, so **feature engineering** —
hand-crafting ratios, products, logs, aggregates — can hand a weak model structure it could never
synthesise (L009: a Ridge regressor jumped R² 0.64 → 1.00 once a ratio-of-differences was engineered in).
That is Domingos' *"feature engineering is the key."*

But two of his other lessons pull back. *"Overfitting has many faces":* every added feature raises
**variance** (more ways to fit noise), so past a point the held-out score **falls**. *"More data beats a
cleverer algorithm":* extra rows usually cut variance more per unit effort than extra hand features. So the
returns curve of *held-out score vs. features added* rises, flattens, and can turn **negative**.

**The stopping rule.** A CV score is a mean over folds with a standard deviation **σ** (the noise band,
mean ± σ). A change smaller than σ is **within noise** — indistinguishable from luck of the split (L023).
So: add features while each one's **marginal gain exceeds σ**; **stop** at the first feature whose gain sinks
into the band. Not "stop when out of ideas" — "stop when the held-out curve flattens into the noise."

**Toy micro-example (not this lab's answer).** Suppose adding a feature moves CV AUC from
`0.800 ± 0.030` to `0.804 ± 0.031`. The marginal gain is `+0.004`, far **smaller than σ ≈ 0.03`** → within
noise → stop. Reporting "0.804 > 0.800, the feature helped" would be exactly the L023/L030 mistake of
celebrating noise.

Full write-up + the interactive returns curve: [Lesson 033](../lessons/0033-when-to-stop-feature-engineering.html).'''),
        md("## Setup — PROVIDED (data + fixed feature list + leak-safe scorer)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — is the peak actually significant?

The peak "gain" at k=3 is inside the noise band, but let's prove it the L023 way: run the **corrected
resampled t-test** (Nadeau & Bengio 2003) on the k=0 baseline vs the k=peak model across repeated CV, which
inflates the naive variance to account for overlapping training folds. Expect **p ≫ 0.05** — the honest
report is "no measurable single-table FE gain on credit_g." (See the L023 lab for the full harness.)'''),
        code(STRETCH),
    ]

    nb_obj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for c in nb_obj["cells"]:
        if isinstance(c["source"], str):
            c["source"] = c["source"].splitlines(keepends=True)
    return nb_obj


def main():
    with open(os.path.join(HERE, "0033-when-to-stop-feature-engineering.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0033-when-to-stop-feature-engineering.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0033-when-to-stop-feature-engineering.ipynb + solution")


if __name__ == "__main__":
    main()
