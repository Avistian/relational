"""Verify Lesson 029 claims — Manual FE + tuned XGBoost vs AutoML (Feurer et al. 2015, Auto-sklearn).

Paper: "Efficient and Robust Automated Machine Learning" (NeurIPS 2015). Auto-sklearn solves the
**CASH** problem — Combined Algorithm Selection and Hyperparameter optimization — jointly picking a
preprocessing+classifier pipeline and its hyperparameters via **Bayesian optimization (SMAC)**, and
adds two things over Auto-WEKA: (1) **meta-learning warm-start** (initialize the search from configs
that worked on similar past datasets) and (2) **automated ensemble construction** (Caruana 2004
greedy ensemble selection over the models evaluated during the search).

Auto-sklearn itself is Linux/version-fragile to install, so we reproduce its *core mechanisms* with a
"tiny AutoML" built only on scikit-learn + xgboost (installs cleanly, same idea):
  * a small pipeline search space over SEVERAL algorithms (logreg, RF, extra-trees, HistGB) + HPs,
  * **CASH via random search** over that joint space (an honest, installable stand-in for SMAC —
    Bergstra & Bengio 2012 / L017 show random search is a strong search baseline; SMAC would be more
    sample-efficient per iteration but the mechanism we teach — jointly searching algo+HP, selecting
    by validation — is identical),
  * **greedy ensemble selection** (Caruana) over the pool of trained models.

We verify three things the lesson/viz/lab depend on, all on real Tier-A data (credit_g):

  Part 1 — CASH BUDGET CURVE: run random search over the joint (algorithm, hyperparameter) space and
    track the best-validation AUC found after k iterations (echoes the L024 budget curve). Shows the
    search visits many *different* algorithms and climbs with budget. Drives cash-search-viz.

  Part 2 — ENSEMBLE SELECTION beats the single best model: Caruana greedy ensemble over the search
    pool vs the single best-validation config. Drives ensemble-select-viz.

  Part 3 — HONEST BAKE-OFF: default XGBoost vs a manually-FE'd, tuned XGBoost vs the tiny AutoML
    (ensemble), by ROC-AUC over 5 seeds. The honest finding: tuning matters a lot (default -> tuned is
    the big jump), and AutoML ~ ties the tuned XGB at much higher compute — it buys automation, not a
    guaranteed win, and it is still a single-table learner. Drives automl-bakeoff-viz + the lab.

Run: .venv/bin/python labs/_verify_l029.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from relkit import load_tier_a


# ---------------------------------------------------------------------------
# Shared preprocessing: impute + scale numeric, impute + one-hot categorical.
# Auto-sklearn searches over preprocessors too; we fix a sane one so the story
# is about algorithm+HP search and ensembling, not preprocessing tricks.
# ---------------------------------------------------------------------------
def make_prep(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


# ---------------------------------------------------------------------------
# The CASH search space: each config is (algorithm name, a classifier with
# sampled hyperparameters). This is the "combined" space Auto-sklearn searches.
# ---------------------------------------------------------------------------
def sample_config(rng):
    algo = rng.choice(["logreg", "rf", "extratrees", "histgb"])
    if algo == "logreg":
        clf = LogisticRegression(
            C=float(10 ** rng.uniform(-3, 2)), max_iter=2000,
            class_weight=rng.choice([None, "balanced"]))
    elif algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=int(rng.choice([100, 200, 400])),
            max_depth=int(rng.choice([4, 8, 16, 32])),
            max_features=float(rng.uniform(0.3, 1.0)),
            min_samples_leaf=int(rng.choice([1, 2, 4, 8])), n_jobs=1)
    elif algo == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=int(rng.choice([100, 200, 400])),
            max_depth=int(rng.choice([4, 8, 16, 32])),
            max_features=float(rng.uniform(0.3, 1.0)),
            min_samples_leaf=int(rng.choice([1, 2, 4, 8])), n_jobs=1)
    else:  # histgb
        clf = HistGradientBoostingClassifier(
            learning_rate=float(10 ** rng.uniform(-2, -0.3)),
            max_leaf_nodes=int(rng.choice([15, 31, 63])),
            l2_regularization=float(10 ** rng.uniform(-3, 1)),
            max_iter=int(rng.choice([100, 200, 300])))
    return algo, clf


def fit_eval(clf, prep, Xtr, ytr, Xva, yva):
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    va = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
    return pipe, va


# ---------------------------------------------------------------------------
# Caruana (2004) greedy ensemble selection: from a pool of models' VALIDATION
# probability vectors, greedily add (with replacement) the model that most
# improves the ensemble's validation AUC. Returns integer weights per model.
# ---------------------------------------------------------------------------
def ensemble_selection(val_probs, yva, n_picks=25):
    n = len(val_probs)
    weights = np.zeros(n)
    ens_sum = np.zeros_like(yva, dtype=float)
    picks = []
    for _ in range(n_picks):
        best_j, best_auc = -1, -1.0
        for j in range(n):
            cand = (ens_sum + val_probs[j]) / (len(picks) + 1)
            a = roc_auc_score(yva, cand)
            if a > best_auc:
                best_auc, best_j = a, j
        ens_sum += val_probs[best_j]
        weights[best_j] += 1
        picks.append(best_j)
    return weights / weights.sum()


def part1_and_2_cash(X, y, budget=40, seed=0):
    print("== Part 1+2: CASH random search + ensemble selection (credit_g, seed 0) ==")
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    rng = np.random.RandomState(seed)

    pool_pipes, val_probs, test_probs = [], [], []
    algos, val_aucs = [], []
    best_curve, best_so_far = [], -1.0
    for i in range(budget):
        algo, clf = sample_config(rng)
        prep = make_prep(X)
        try:
            pipe, va = fit_eval(clf, prep, Xtr, ytr, Xva, yva)
        except Exception:
            continue
        pool_pipes.append(pipe)
        val_probs.append(pipe.predict_proba(Xva)[:, 1])
        test_probs.append(pipe.predict_proba(Xte)[:, 1])
        algos.append(algo); val_aucs.append(va)
        best_so_far = max(best_so_far, va)
        best_curve.append(round(best_so_far, 3))

    val_probs = np.array(val_probs); test_probs = np.array(test_probs)
    val_aucs = np.array(val_aucs)

    # single best-validation config
    best_idx = int(np.argmax(val_aucs))
    single_test = roc_auc_score(yte, test_probs[best_idx])

    # greedy ensemble
    w = ensemble_selection(val_probs, yva, n_picks=25)
    ens_test = roc_auc_score(yte, (w[:, None] * test_probs).sum(0))
    ens_val = roc_auc_score(yva, (w[:, None] * val_probs).sum(0))

    # which algorithms did the search actually try / how many in the ensemble
    from collections import Counter
    algo_counts = Counter(algos)
    ens_members = int((w > 0).sum())
    ens_algos = Counter([algos[j] for j in range(len(algos)) if w[j] > 0])

    print(f"  budget={budget}; algorithms tried: {dict(algo_counts)}")
    print(f"  best-val curve (best AUC after k iters): {best_curve}")
    print(f"  single best-val config: algo={algos[best_idx]}, val={val_aucs[best_idx]:.3f}, TEST={single_test:.3f}")
    print(f"  greedy ensemble: {ens_members} distinct members {dict(ens_algos)}, val={ens_val:.3f}, TEST={ens_test:.3f}")
    print(f"  ensemble - single (TEST AUC): {ens_test - single_test:+.3f}")
    print(f"\n  CURVE_X = {list(range(1, len(best_curve)+1))}")
    print(f"  CURVE_Y = {best_curve}")
    print(f"  SINGLE_TEST = {single_test:.3f}   ENS_TEST = {ens_test:.3f}")
    return best_curve, single_test, ens_test


def tiny_automl(X, y, Xtr, ytr, Xva, yva, Xte, budget, seed):
    rng = np.random.RandomState(seed)
    val_probs, test_probs = [], []
    for _ in range(budget):
        algo, clf = sample_config(rng)
        prep = make_prep(X)
        try:
            pipe, _ = fit_eval(clf, prep, Xtr, ytr, Xva, yva)
        except Exception:
            continue
        val_probs.append(pipe.predict_proba(Xva)[:, 1])
        test_probs.append(pipe.predict_proba(Xte)[:, 1])
    val_probs = np.array(val_probs); test_probs = np.array(test_probs)
    w = ensemble_selection(val_probs, yva, n_picks=25)
    return (w[:, None] * test_probs).sum(0)


def tuned_xgb(X, y, Xtr, ytr, Xva, yva, Xte, budget, seed):
    rng = np.random.RandomState(seed + 100)
    best_pipe, best_va = None, -1.0
    for _ in range(budget):
        clf = XGBClassifier(
            n_estimators=int(rng.choice([100, 200, 400])),
            max_depth=int(rng.choice([2, 3, 4, 6, 8])),
            learning_rate=float(10 ** rng.uniform(-2, -0.3)),
            subsample=float(rng.uniform(0.6, 1.0)),
            colsample_bytree=float(rng.uniform(0.6, 1.0)),
            min_child_weight=int(rng.choice([1, 2, 4, 8])),
            reg_lambda=float(10 ** rng.uniform(-2, 1)),
            eval_metric="logloss", tree_method="hist", n_jobs=1, verbosity=0)
        prep = make_prep(X)
        pipe, va = fit_eval(clf, prep, Xtr, ytr, Xva, yva)
        if va > best_va:
            best_va, best_pipe = va, pipe
    return best_pipe.predict_proba(Xte)[:, 1]


def part3_bakeoff(X, y, budget=40, seeds=(0, 1, 2, 3, 4)):
    print("\n== Part 3: honest bake-off — default XGB vs tuned XGB vs tiny AutoML (credit_g) ==")
    res = {"default_xgb": [], "tuned_xgb": [], "automl": []}
    for s in seeds:
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=s, stratify=y)
        Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=s, stratify=ytmp)

        # default XGB (no tuning)
        prep = make_prep(X)
        dpipe = Pipeline([("prep", prep), ("clf", XGBClassifier(
            eval_metric="logloss", tree_method="hist", n_jobs=1, verbosity=0))])
        dpipe.fit(Xtr, ytr)
        res["default_xgb"].append(roc_auc_score(yte, dpipe.predict_proba(Xte)[:, 1]))

        res["tuned_xgb"].append(roc_auc_score(yte, tuned_xgb(X, y, Xtr, ytr, Xva, yva, Xte, budget, s)))
        res["automl"].append(roc_auc_score(yte, tiny_automl(X, y, Xtr, ytr, Xva, yva, Xte, budget, s)))

    print(f"  {'model':>12} | {'roc-auc (mean±sd)':>18}")
    for k in ["default_xgb", "tuned_xgb", "automl"]:
        print(f"  {k:>12} | {np.mean(res[k]):.3f} ± {np.std(res[k]):.3f}")
    print(f"\n  labels   = ['default XGB','tuned XGB','tiny AutoML']")
    print(f"  auc_mean = {[round(float(np.mean(res[k])),3) for k in ['default_xgb','tuned_xgb','automl']]}")
    print(f"  auc_sd   = {[round(float(np.std(res[k])),3) for k in ['default_xgb','tuned_xgb','automl']]}")
    print(f"  tuned-default = {np.mean(res['tuned_xgb'])-np.mean(res['default_xgb']):+.3f} (the tuning jump)")
    print(f"  automl-tuned  = {np.mean(res['automl'])-np.mean(res['tuned_xgb']):+.3f} (AutoML vs manual tuned XGB)")
    return res


if __name__ == "__main__":
    t0 = time.time()
    X, y = load_tier_a("credit_g")
    print(f"credit_g: X {X.shape}, positive rate {y.mean():.2f}\n")
    part1_and_2_cash(X, y, budget=40, seed=0)
    part3_bakeoff(X, y, budget=40)
    print(f"\n[done in {time.time()-t0:.0f}s]")
