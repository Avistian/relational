"""Build Lab 029 (Manual FE vs AutoML) — emits the blank student notebook and a filled solution.

Tier A (real credit_g via relkit). The learner implements the two mechanisms that ARE AutoML
(Feurer et al. 2015, Auto-sklearn):
  * CASH selection — search a COMBINED (algorithm, hyperparameter) space and keep the config with the
    best VALIDATION score;
  * greedy ENSEMBLE selection (Caruana 2004) — the argmax pick that grows a weighted blend from the
    pool of already-trained models.
Then the honest bake-off: default XGBoost vs a tuned XGBoost vs the tiny AutoML.

Random search stands in for SMAC (installs anywhere; Bergstra & Bengio 2012 show it is a strong search
baseline). The mechanism taught is CASH + select-by-validation + ensemble, not the exact optimizer.

Run: .venv/bin/python labs/_build_l029.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0029-manual-fe-vs-automl.ipynb \
    labs/solutions/0029-manual-fe-vs-automl.ipynb
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


# ---- shared PROVIDED setup ----
SETUP = r'''# PROVIDED — imports, real credit_g (Tier A), the CASH search space + a train/score harness. Just run.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 0
np.random.seed(SEED)

# --- Tier A: real credit_g via relkit ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
X, y = load_tier_a("credit_g")
print(f"credit_g (Tier A): X {X.shape}, positive rate {y.mean():.2f}")

# --- shared preprocessing: impute+scale numeric, impute+one-hot categorical ---
def make_prep(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])

# --- the CASH search space: each draw is (algorithm name, a classifier with sampled hyperparameters) ---
def sample_config(rng):
    algo = rng.choice(["logreg", "rf", "extratrees", "histgb"])
    if algo == "logreg":
        clf = LogisticRegression(C=float(10 ** rng.uniform(-3, 2)), max_iter=2000,
                                 class_weight=rng.choice([None, "balanced"]))
    elif algo == "rf":
        clf = RandomForestClassifier(n_estimators=int(rng.choice([100, 200])),
                                     max_depth=int(rng.choice([4, 8, 16])),
                                     max_features=float(rng.uniform(0.3, 1.0)),
                                     min_samples_leaf=int(rng.choice([1, 2, 4])), n_jobs=1)
    elif algo == "extratrees":
        clf = ExtraTreesClassifier(n_estimators=int(rng.choice([100, 200])),
                                   max_depth=int(rng.choice([4, 8, 16])),
                                   max_features=float(rng.uniform(0.3, 1.0)),
                                   min_samples_leaf=int(rng.choice([1, 2, 4])), n_jobs=1)
    else:  # histgb
        clf = HistGradientBoostingClassifier(learning_rate=float(10 ** rng.uniform(-2, -0.3)),
                                             max_leaf_nodes=int(rng.choice([15, 31, 63])),
                                             l2_regularization=float(10 ** rng.uniform(-3, 1)),
                                             max_iter=int(rng.choice([100, 200])))
    return algo, clf

def fit_eval(clf, Xtr, ytr, Xva, yva):
    pipe = Pipeline([("prep", make_prep(X)), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe, roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])

# --- one 60/20/20 train/val/test split per seed ---
def split(seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    return Xtr, Xva, Xte, ytr, yva, yte

print("ready — helpers: make_prep, sample_config(rng), fit_eval(clf, Xtr,ytr, Xva,yva) -> (pipe, val_auc), split(seed)")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — the greedy ensemble-selection pick (crucial fragment) — TODO

**Goal:** complete the inner step of **greedy ensemble selection** (Caruana et al. 2004). The ensemble
is a *sum* of chosen models' validation-probability vectors; each round you add the pool model whose
addition most improves the ensemble's **validation** AUC. You implement the candidate the loop scores:
the ensemble mean **if model `j` were added**.

**Why it matters:** this one line is Auto-sklearn's second extension over Auto-WEKA. It turns the pool
of models the search already trained into a diverse weighted blend — free accuracy, no meta-learner to
overfit (contrast L018 stacking). Selecting on *validation* is what keeps it leak-free.

**Hint boundary:** the running sum of picked probability vectors is `ens_sum`, and `n_added` models
have been picked so far. Adding model `j` makes the new ensemble mean `(ens_sum + val_probs[j])` divided
by `(n_added + 1)`.'''

T1_CODE = r'''# TODO — implement the greedy ensemble-selection candidate (the crucial fragment)
def ensemble_selection(val_probs, yva, n_picks=25):
    n = len(val_probs)
    weights = np.zeros(n)
    ens_sum = np.zeros(len(yva), dtype=float)
    n_added = 0
    for _ in range(n_picks):
        best_j, best_auc = -1, -1.0
        for j in range(n):
            cand = ____                                # the ensemble MEAN if model j were added now
            a = roc_auc_score(yva, cand)
            if a > best_auc:
                best_auc, best_j = a, j
        ens_sum += val_probs[best_j]                   # commit the greedy pick
        weights[best_j] += 1
        n_added += 1
    return weights / weights.sum()

print("ensemble_selection defined")'''

T1_SOL = T1_CODE.replace(
    "            cand = ____                                # the ensemble MEAN if model j were added now",
    "            cand = (ens_sum + val_probs[j]) / (n_added + 1)   # the ensemble MEAN if model j were added now")

T1_CHECK = r'''# CHECK — do not edit
rng = np.random.RandomState(0)
_va = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
# three fake models: good, mediocre, and the good one's complement
_pool = np.array([[0.1,0.8,0.2,0.7,0.9,0.3,0.6,0.2],
                  [0.5,0.5,0.4,0.6,0.5,0.5,0.5,0.4],
                  [0.2,0.9,0.1,0.8,0.7,0.2,0.9,0.3]])
_w = ensemble_selection(_pool, _va, n_picks=10)
assert abs(_w.sum() - 1.0) < 1e-9, "weights must be normalized to sum to 1."
single_best = max(roc_auc_score(_va, _pool[j]) for j in range(len(_pool)))
ens = roc_auc_score(_va, (_w[:, None] * _pool).sum(0))
assert ens >= single_best - 1e-9, "The greedy ensemble's VALIDATION AUC must be >= the single best model (the first pick IS the best single)."
print(f"Task 1 ok — greedy ensemble val AUC {ens:.3f} >= single best {single_best:.3f}. The argmax pick is the whole mechanism.")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — the CASH search: select by validation, then ensemble — TODO

**Goal:** run a random search over the **combined (algorithm, hyperparameter)** space (`sample_config`),
scoring each config on the **validation** split. You implement the selection rule — the index of the
config with the best *validation* AUC — the single rule that makes any search work (L004/L017). Then we
compare the single best config against the greedy ensemble on the untouched **test** set.

**Why it matters:** this is CASH — Combined Algorithm Selection and Hyperparameter optimization. The
search visits every algorithm lane; "keep the best-validation config" is what turns a pile of trials
into a model. The ensemble from Task 1 then squeezes out a little more, for free.

**Hint boundary:** `val_aucs` is a list of validation AUCs in pool order; the best config's index is
`int(np.argmax(val_aucs))`.'''

T2_CODE = r'''# TODO — CASH random search over the combined space (~1 min)
Xtr, Xva, Xte, ytr, yva, yte = split(SEED)
rng = np.random.RandomState(SEED)

algos, val_aucs, val_probs, test_probs = [], [], [], []
for i in range(30):
    algo, clf = sample_config(rng)
    try:
        pipe, va = fit_eval(clf, Xtr, ytr, Xva, yva)
    except Exception:
        continue
    algos.append(algo); val_aucs.append(va)
    val_probs.append(pipe.predict_proba(Xva)[:, 1])
    test_probs.append(pipe.predict_proba(Xte)[:, 1])
val_probs, test_probs = np.array(val_probs), np.array(test_probs)

best_idx = ____                                        # index of the config with the highest VALIDATION AUC
single_test = roc_auc_score(yte, test_probs[best_idx])

w = ensemble_selection(val_probs, yva, n_picks=25)     # uses your Task-1 function
ens_test = roc_auc_score(yte, (w[:, None] * test_probs).sum(0))

print(f"algorithms tried: {dict(Counter(algos))}")
print(f"single best-val config: algo={algos[best_idx]}, TEST AUC {single_test:.3f}")
print(f"greedy ensemble ({int((w>0).sum())} members): TEST AUC {ens_test:.3f}  ({ens_test-single_test:+.3f})")'''

T2_SOL = T2_CODE.replace(
    "best_idx = ____                                        # index of the config with the highest VALIDATION AUC",
    "best_idx = int(np.argmax(val_aucs))                    # index of the config with the highest VALIDATION AUC")

T2_CHECK = r'''# CHECK — do not edit
assert len(set(algos)) >= 3, "A CASH search should evaluate several DIFFERENT algorithms (combined space)."
assert algos[best_idx] == algos[int(np.argmax(val_aucs))], "best_idx must be the argmax of the VALIDATION AUCs."
assert ens_test >= single_test - 0.02, "The greedy ensemble should be about as good as, or better than, the single best on test."
print(f"Task 2 ok — searched {len(set(algos))} algorithms; selected by validation; ensemble {ens_test:.3f} vs single {single_test:.3f}.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — the honest bake-off: default vs tuned XGB vs tiny AutoML — TODO

**Goal:** race three things by ROC-AUC over a few seeds: an **untuned default XGBoost**, a **tuned
XGBoost** (random search over *its own* hyperparameters, selected by validation), and the **tiny
AutoML** (CASH search + your ensemble). You implement the tuned-XGB selection rule (keep the best
*validation* pipe).

**Why it matters:** this is the lesson's punchline. The big jump is **tuning at all** (default → tuned);
the four-algorithm AutoML then only **ties** the tuned XGB, at far higher compute. AutoML buys
automation and robustness, not new accuracy — and it never touches the *representation*, which is where
the relational thesis says the remaining upside lives.

**Hint boundary:** inside the tuned-XGB loop, when a config's validation AUC `va` beats `best_va`, keep
it: `best_va, best_pipe = va, pipe`.'''

T3_CODE = r'''# TODO — bake-off over 3 seeds (~2 min)
def tuned_xgb_test(Xtr, ytr, Xva, yva, Xte, rng, budget=20):
    best_va, best_pipe = -1.0, None
    for _ in range(budget):
        clf = XGBClassifier(n_estimators=int(rng.choice([100, 200])),
                            max_depth=int(rng.choice([2, 3, 4, 6])),
                            learning_rate=float(10 ** rng.uniform(-2, -0.3)),
                            subsample=float(rng.uniform(0.6, 1.0)),
                            colsample_bytree=float(rng.uniform(0.6, 1.0)),
                            reg_lambda=float(10 ** rng.uniform(-2, 1)),
                            eval_metric="logloss", tree_method="hist", n_jobs=1, verbosity=0)
        pipe, va = fit_eval(clf, Xtr, ytr, Xva, yva)
        if va > best_va:
            best_va, best_pipe = ____                  # keep this config as the new best-validation pipe
    return best_pipe.predict_proba(Xte)[:, 1]

def automl_test(Xtr, ytr, Xva, yva, Xte, rng, budget=20):
    vp, tp = [], []
    for _ in range(budget):
        algo, clf = sample_config(rng)
        try:
            pipe, _ = fit_eval(clf, Xtr, ytr, Xva, yva)
        except Exception:
            continue
        vp.append(pipe.predict_proba(Xva)[:, 1]); tp.append(pipe.predict_proba(Xte)[:, 1])
    vp, tp = np.array(vp), np.array(tp)
    w = ensemble_selection(vp, yva, n_picks=25)
    return (w[:, None] * tp).sum(0)

res = {"default": [], "tuned": [], "automl": []}
for s in [0, 1, 2]:
    Xtr, Xva, Xte, ytr, yva, yte = split(s)
    dft = Pipeline([("prep", make_prep(X)),
                    ("clf", XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1, verbosity=0))]).fit(Xtr, ytr)
    res["default"].append(roc_auc_score(yte, dft.predict_proba(Xte)[:, 1]))
    res["tuned"].append(roc_auc_score(yte, tuned_xgb_test(Xtr, ytr, Xva, yva, Xte, np.random.RandomState(s+100))))
    res["automl"].append(roc_auc_score(yte, automl_test(Xtr, ytr, Xva, yva, Xte, np.random.RandomState(s))))

for k in ["default", "tuned", "automl"]:
    print(f"{k:>8} XGB/AutoML ROC-AUC: {np.mean(res[k]):.3f} ± {np.std(res[k]):.3f}")'''

T3_SOL = T3_CODE.replace(
    "            best_va, best_pipe = ____                  # keep this config as the new best-validation pipe",
    "            best_va, best_pipe = va, pipe             # keep this config as the new best-validation pipe")

T3_CHECK = r'''# CHECK — do not edit
d, t, a = np.mean(res["default"]), np.mean(res["tuned"]), np.mean(res["automl"])
assert t > d, "Tuning should beat the untuned default (the big jump is searching at all)."
assert abs(a - t) < 0.05, "The tiny AutoML should TIE the hand-tuned XGB (bands overlap), not crush it."
print(f"Task 3 ok — default {d:.3f} < tuned {t:.3f} ~ AutoML {a:.3f}. Tuning is the jump; AutoML ties it at higher compute.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, what does AutoML automate (CASH + ensemble), what does it NOT do,
and what does the tie with a tuned XGBoost mean for the relational thesis?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 029 ===")
print(f"CASH search (seed 0): single best TEST {single_test:.3f} -> greedy ensemble {ens_test:.3f} ({ens_test-single_test:+.3f})")
print(f"bake-off (3 seeds)  : default {np.mean(res['default']):.3f} < tuned XGB {np.mean(res['tuned']):.3f} "
      f"~ tiny AutoML {np.mean(res['automl']):.3f}")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"AutoML automates the CASH problem (jointly search which algorithm AND its hyperparameters, keeping '
    'the best-validation config) plus meta-learning warm-start and Caruana greedy ensemble selection over '
    'the models it already trained (free diversity, +ensemble here) — but it does NOT do domain feature '
    'engineering or change the representation; on credit_g it only TIED a hand-tuned XGBoost (both far above '
    'the untuned default, whose gap is the real payoff of tuning at all), which is exactly why the remaining '
    'upside for the thesis must come from learning over relational structure, not from smarter single-table search."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 029 — Manual FE vs AutoML: build a tiny AutoML, race a tuned XGBoost

**Lesson:** [`lessons/0029-manual-fe-vs-automl.html`](../lessons/0029-manual-fe-vs-automl.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Feurer, Klein, Eggensperger, Springenberg, Blum & Hutter 2015, *Efficient and Robust Automated Machine Learning*, NeurIPS 2015 (**Auto-sklearn**). Background: Thornton et al. 2013 (Auto-WEKA — the CASH framing) and Caruana et al. 2004 (ensemble selection).

**Dataset tier:** **A** (real `credit_g` via `relkit`).

**Skill you are practising:** implement the two mechanisms that ARE AutoML — the **CASH selection** (keep the config with the best validation score over a combined algorithm+hyperparameter space) and the **greedy ensemble-selection pick** (the Caruana argmax) — then run the honest bake-off: default XGBoost vs tuned XGBoost vs your tiny AutoML.

**Exit criteria:** EXIT TICKET prints the greedy ensemble matching/beating the single best config, and — over seeds — the tuned XGBoost and the tiny AutoML tying while both clear the untuned default.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, the CASH search space, the train/score harness); just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs scikit-learn + xgboost (already in `requirements-labs.txt`). `credit_g` is fetched from OpenML on first run (needs network) then cached. The search fits many small models — budget ~3–4 minutes on CPU.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — what AutoML automates (and what it does not)

**CASH — Combined Algorithm Selection and Hyperparameter optimization.** Treat *which algorithm* as a top-level categorical hyperparameter sitting above each model's own knobs, and search the whole space at once, keeping the config with the best **validation** score. That is the core of Auto-sklearn.

**Bayesian optimization (SMAC).** Auto-sklearn picks the next config with a surrogate model (a random forest) that predicts a config's score + uncertainty, trading exploitation vs exploration. Here we use **random search** as an installable stand-in — the mechanism to learn is *CASH + select-by-validation*, not the optimizer.

**Two extensions over Auto-WEKA.** (1) **Meta-learning warm-start**: use cheap dataset *meta-features* to start the search from configs that won on similar past datasets. (2) **Ensemble selection** (Caruana 2004): greedily blend the pool of already-trained models — add the one that most improves validation AUC, with replacement — for free diversity, no meta-learner to overfit (contrast L018 stacking).

**What it does NOT do.** It does not invent **domain features** (it searches models + generic preprocessing) and it does not change the **representation** — it tunes a model on top of an already-flattened single table. The honest finding: on typical tabular data AutoML *ties* a well-tuned GBDT; it buys automation/robustness, not new accuracy.

Full write-up + the CASH scatter, the ensemble toggle, and the bake-off: [Lesson 029](../lessons/0029-manual-fe-vs-automl.html).'''),
        md("## Setup — PROVIDED (data + CASH search space + harness)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — probe the AutoML claim

1. **Where does the ensemble stop helping?** Vary `n_picks` in `ensemble_selection` (5, 25, 100). Does
   the validation gain keep rising while the test gain plateaus (over-selection on validation)?
2. **Budget accounting.** Time the tuned-XGB vs the AutoML runs. AutoML tied at what compute multiple?
   On a *fixed* time budget, does spending it all on one strong family (XGB) beat spreading it?
3. **Add a genuinely bad algorithm** (e.g. a 1-nearest-neighbour) to `sample_config`. Does the CASH
   search waste budget on it, and does ensemble selection correctly give it ~0 weight?
4. **Manual FE, the thing AutoML skips.** Hand-engineer 2–3 domain features on `credit_g` (ratios of
   existing columns). Do they move the tuned XGB more than switching to AutoML did? That gap is the
   lesson: representation > search.'''),
        code(r'''# STRETCH — ungraded.
# for k in [5, 25, 100]:
#     w = ensemble_selection(val_probs, yva, n_picks=k)
#     print(k, "ensemble TEST", round(roc_auc_score(yte, (w[:,None]*test_probs).sum(0)), 3))'''),
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
    student = build(solution=False)
    sol = build(solution=True)
    with open(os.path.join(HERE, "0029-manual-fe-vs-automl.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0029-manual-fe-vs-automl.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0029-manual-fe-vs-automl.ipynb + solution")


if __name__ == "__main__":
    main()
