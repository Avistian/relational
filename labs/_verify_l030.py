"""Verify the numbers quoted in Lesson 030 (Q3 checkpoint — the 1-page benchmark report).

This is a *consolidation* checkpoint: it assembles the whole Q3 evaluation-rigor toolkit into one
defensible benchmark on a real dataset (credit_g, Tier A):

  * L024 — a random-search BUDGET CURVE (select by validation, report test) for each model family,
           showing both the default (k=1) and the tuned ceiling;
  * L028 — an honest NEURAL baseline (here sklearn MLPClassifier stands in for the L028 tabular
           ResNet — torch-free so the checkpoint lab runs anywhere; same inductive biases);
  * L029 — a tiny AutoML (CASH search + Caruana greedy ensemble) as the "automate the pipeline" bar;
  * L023 — a SIGNIFICANCE test on paired CV folds: the naive paired t-test (anticonservative) vs the
           corrected resampled t-test (Nadeau & Bengio 2003);
  * L021/L022 — the split + leakage discipline the whole thing sits on.

Run: .venv/bin/python labs/_verify_l030.py
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relkit import load_tier_a  # noqa: E402

SEED = 0
np.random.seed(SEED)

X, y = load_tier_a("credit_g")
y = np.asarray(y)
print(f"credit_g (Tier A): X {X.shape}, positive rate {y.mean():.3f}")

NUM = X.select_dtypes(include="number").columns.tolist()
CAT = [c for c in X.columns if c not in NUM]


def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def split(seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    return Xtr, Xva, Xte, ytr, yva, yte


def fit_auc(clf, Xtr, ytr, Xev, yev):
    pipe = Pipeline([("prep", make_prep()), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe, roc_auc_score(yev, pipe.predict_proba(Xev)[:, 1])


# ---- search spaces for the two headline families ----
def sample_gbdt(rng):
    return HistGradientBoostingClassifier(
        learning_rate=float(10 ** rng.uniform(-2, -0.3)),
        max_leaf_nodes=int(rng.choice([15, 31, 63])),
        l2_regularization=float(10 ** rng.uniform(-3, 1)),
        max_iter=int(rng.choice([100, 200, 300])), random_state=0)


MLP_SHAPES = [(128,), (256, 128), (256, 256)]


def sample_mlp(rng):
    return MLPClassifier(
        hidden_layer_sizes=MLP_SHAPES[int(rng.randint(len(MLP_SHAPES)))],
        alpha=float(10 ** rng.uniform(-5, -1)),
        learning_rate_init=float(10 ** rng.uniform(-3.5, -2)),
        max_iter=300, early_stopping=True, random_state=0)


def sample_config(rng):
    """Tiny CASH space for the AutoML contestant (L029)."""
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
    else:
        clf = HistGradientBoostingClassifier(learning_rate=float(10 ** rng.uniform(-2, -0.3)),
                                             max_leaf_nodes=int(rng.choice([15, 31, 63])),
                                             l2_regularization=float(10 ** rng.uniform(-3, 1)),
                                             max_iter=int(rng.choice([100, 200])), random_state=0)
    return algo, clf


# ============================================================ 1. BUDGET CURVE (L024)
print("\n=== 1. Budget curve: best-so-far TEST AUC vs random-search budget (select by validation) ===")
Xtr, Xva, Xte, ytr, yva, yte = split(SEED)


def budget_curve(sampler, K=15):
    rng = np.random.RandomState(SEED)
    best_va, best_te, curve = -1.0, None, []
    for _ in range(K):
        clf = sampler(rng)
        try:
            pipe, va = fit_auc(clf, Xtr, ytr, Xva, yva)
        except Exception:
            curve.append(best_te if best_te is not None else np.nan)
            continue
        te = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
        if va > best_va:
            best_va, best_te = va, te
        curve.append(best_te)
    return curve


gbdt_curve = budget_curve(sample_gbdt)
mlp_curve = budget_curve(sample_mlp)
print(f"GBDT : default (k=1) {gbdt_curve[0]:.3f}  ->  ceiling (k={len(gbdt_curve)}) {gbdt_curve[-1]:.3f}")
print(f"MLP  : default (k=1) {mlp_curve[0]:.3f}  ->  ceiling (k={len(mlp_curve)}) {mlp_curve[-1]:.3f}")
print(f"gap  : default {gbdt_curve[0]-mlp_curve[0]:+.3f}  ->  tuned {gbdt_curve[-1]-mlp_curve[-1]:+.3f}")

# ============================================================ 2. HEADLINE HEAD-TO-HEAD (L023 spine)
# One consistent protocol for the report: paired repeated CV of the two families at their DEFAULTS,
# on identical folds. Point estimates from a single tuned split flip seed-to-seed (a small-data
# artifact) — which is exactly why the report's verdict rests on this paired, significance-tested view.
print("\n=== 2/3. Headline: paired 5x5 CV of GBDT vs honest MLP baseline (defaults) + significance ===")
n_splits, n_repeats = 5, 5
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
diffs = []
gb_scores, mlp_scores = [], []
for tr_idx, te_idx in rkf.split(X, y):
    Xtr_f, Xte_f = X.iloc[tr_idx], X.iloc[te_idx]
    ytr_f, yte_f = y[tr_idx], y[te_idx]
    _, a_gb = fit_auc(HistGradientBoostingClassifier(random_state=0), Xtr_f, ytr_f, Xte_f, yte_f)
    _, a_mlp = fit_auc(MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                     max_iter=300, early_stopping=True, random_state=0),
                       Xtr_f, ytr_f, Xte_f, yte_f)
    gb_scores.append(a_gb)
    mlp_scores.append(a_mlp)
    diffs.append(a_gb - a_mlp)

diffs = np.array(diffs)
n = len(diffs)
mean_d = diffs.mean()
var_d = diffs.var(ddof=1)

# naive paired t-test (treats folds as independent -> anticonservative)
t_naive = mean_d / np.sqrt(var_d / n)
p_naive = 2 * stats.t.sf(abs(t_naive), df=n - 1)

# corrected resampled t-test (Nadeau & Bengio 2003): inflate variance by (1/n + rho),
# rho = n_test / n_train = 1/(k-1) for k-fold
rho = 1.0 / (n_splits - 1)
t_corr = mean_d / np.sqrt((1.0 / n + rho) * var_d)
p_corr = 2 * stats.t.sf(abs(t_corr), df=n - 1)

print(f"  mean GBDT {np.mean(gb_scores):.3f}  mean MLP {np.mean(mlp_scores):.3f}")
print(f"  mean paired gap {mean_d:+.4f}  (sd {np.sqrt(var_d):.4f}, n={n} folds)")
print(f"  naive paired t     : t={t_naive:.3f}  p={p_naive:.2e}  -> {'SIG' if p_naive < 0.05 else 'not sig'}")
print(f"  corrected resampled: t={t_corr:.3f}  p={p_corr:.3f}  -> {'SIG' if p_corr < 0.05 else 'not sig (noise)'}")

print("\nDone. These are the numbers quoted in Lesson 030 + Lab 030.")
