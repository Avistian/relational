"""Verify Lesson 018 (Ensembling & stacking) numbers. Run from labs/ so `relkit` imports:

    cd labs && OMP_NUM_THREADS=3 ../.venv/bin/python _verify_l018.py

Produces the real numbers used in the lesson HTML and lab:
  1. Base learners on credit_g (label-encoded), 5-fold CV PR-AUC (relkit) -- the
     pool we will combine (XGB/LGBM/CatBoost/RF/Logistic).
  2. Diversity: correlation of out-of-fold probability predictions. GBDTs are
     highly correlated with each other; the linear model is the odd one out.
  3. Simple average blend of OOF probabilities vs the best single model.
  4. Stacking (Wolpert 1992 stacked generalization): a logistic meta-learner
     trained on OOF base predictions (sklearn StackingClassifier), honest outer
     5-fold CV PR-AUC. Contrast: stack of ONLY the correlated GBDTs vs the full
     diverse pool.
  5. The leakage trap: a meta-learner trained on IN-SAMPLE base predictions
     over-trusts a base model that memorizes (RF), inflating the apparent score
     and hurting the held-out one -- OOF predictions are mandatory (L004/L005).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore")

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from relkit.data import load_tier_a

RS = 0


def section(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ------------------------------------------------------------- data (L017 style)
X, y = load_tier_a("credit_g")
y = np.asarray(y)
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = [c for c in X.columns if c not in num]
Xle = X[num].copy()
for c in cat:
    Xle[c] = LabelEncoder().fit_transform(X[c].astype(str))
Xle = Xle.to_numpy(dtype=float)


def base_models():
    """Fresh, clone-safe base learners on the label-encoded matrix."""
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RS)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=4),
        "xgboost": XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist"),
        "lightgbm": LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1),
        "catboost": CatBoostClassifier(random_state=RS, verbose=0),
    }


cv = StratifiedKFold(5, shuffle=True, random_state=RS)

# -------------------------------------------------------------- 1. base learners
section("1. Base learners on credit_g -- 5-fold CV PR-AUC")
single = {}
for name, est in base_models().items():
    single[name] = float(cross_val_score(est, Xle, y, cv=cv, scoring="average_precision").mean())
    print(f"  {name:16s} CV PR-AUC = {single[name]:.3f}")
best_name = max(single, key=single.get)
print(f"  best single model: {best_name} = {single[best_name]:.3f}")

# ---------------------------------------------------------------- 2. diversity
section("2. Diversity -- correlation of out-of-fold probability predictions")
oof = {}
for name, est in base_models().items():
    oof[name] = cross_val_predict(est, Xle, y, cv=cv, method="predict_proba")[:, 1]
names = list(oof)
P = np.column_stack([oof[n] for n in names])
corr = np.corrcoef(P.T)
print("        " + " ".join(f"{n[:6]:>7s}" for n in names))
for i, n in enumerate(names):
    print(f"  {n[:6]:>6s} " + " ".join(f"{corr[i, j]:7.3f}" for j in range(len(names))))
gbdt_idx = [names.index(n) for n in ("xgboost", "lightgbm", "catboost")]
gbdt_corr = np.mean([corr[i, j] for i in gbdt_idx for j in gbdt_idx if i < j])
log_i = names.index("logistic")
log_corr = np.mean([corr[log_i, j] for j in gbdt_idx])
print(f"  mean GBDT<->GBDT corr = {gbdt_corr:.3f} | mean logistic<->GBDT corr = {log_corr:.3f}")

# -------------------------------------------------------------- 3. simple blend
section("3. Simple average blend of OOF probabilities")
blend_all = P.mean(axis=1)
blend_all_ap = average_precision_score(y, blend_all)
gbdt_blend = P[:, gbdt_idx].mean(axis=1)
gbdt_blend_ap = average_precision_score(y, gbdt_blend)
print(f"  blend (all 5 models)   PR-AUC = {blend_all_ap:.3f}")
print(f"  blend (3 GBDTs only)   PR-AUC = {gbdt_blend_ap:.3f}")
print(f"  best single ({best_name}) = {single[best_name]:.3f}  ->  diverse-blend lift = {blend_all_ap - single[best_name]:+.3f}")

# ------------------------------------------------------------------ 4. stacking
section("4. Stacking (Wolpert 1992) -- logistic meta-learner on OOF base preds")


def stack(estimators):
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RS),
        cv=5,
        n_jobs=4,
        stack_method="predict_proba",
    )


full = [(n, base_models()[n]) for n in names]
gbdt_only = [(n, base_models()[n]) for n in ("xgboost", "lightgbm", "catboost")]

stack_full_ap = float(cross_val_score(stack(full), Xle, y, cv=cv, scoring="average_precision").mean())
stack_gbdt_ap = float(cross_val_score(stack(gbdt_only), Xle, y, cv=cv, scoring="average_precision").mean())
print(f"  stack (diverse pool, 5 models) CV PR-AUC = {stack_full_ap:.3f}")
print(f"  stack (3 correlated GBDTs)     CV PR-AUC = {stack_gbdt_ap:.3f}")
print(f"  best single = {single[best_name]:.3f} | blend = {blend_all_ap:.3f} | stack = {stack_full_ap:.3f}")

# --------------------------------------------------------------- 5. leakage trap
section("5. Leakage trap -- in-sample vs OOF base predictions for the meta-learner")
from sklearn.neighbors import KNeighborsClassifier

Xtr, Xte, ytr, yte = train_test_split(Xle, y, test_size=0.3, stratify=y, random_state=RS)
inner = StratifiedKFold(5, shuffle=True, random_state=RS)


def leak_models():
    m = base_models()
    # a pure memorizer: 1-NN is perfect in-sample (its own row is the neighbour),
    # near-useless out-of-fold. It exposes the trap dramatically.
    m["one_nn"] = KNeighborsClassifier(n_neighbors=1)
    return m


leak_names = list(leak_models())
# base models fit on the full training set
fitted = {n: leak_models()[n].fit(Xtr, ytr) for n in leak_names}
names = leak_names  # extend the meta-feature table with the memorizer for this section

# (a) NAIVE meta features: base predictions on the SAME rows they were trained on
naive_tr = np.column_stack([fitted[n].predict_proba(Xtr)[:, 1] for n in names])
# (b) HONEST meta features: out-of-fold base predictions on the training set
oof_tr = np.column_stack(
    [cross_val_predict(leak_models()[n], Xtr, ytr, cv=inner, method="predict_proba")[:, 1] for n in names]
)
# test meta features are the same for both: bases (fit on all train) predict on test
te_feat = np.column_stack([fitted[n].predict_proba(Xte)[:, 1] for n in names])

meta_naive = LogisticRegression(max_iter=2000, random_state=RS).fit(naive_tr, ytr)
meta_oof = LogisticRegression(max_iter=2000, random_state=RS).fit(oof_tr, ytr)

naive_train_ap = average_precision_score(ytr, meta_naive.predict_proba(naive_tr)[:, 1])
oof_train_ap = average_precision_score(ytr, meta_oof.predict_proba(oof_tr)[:, 1])
naive_test_ap = average_precision_score(yte, meta_naive.predict_proba(te_feat)[:, 1])
oof_test_ap = average_precision_score(yte, meta_oof.predict_proba(te_feat)[:, 1])

print("  meta-learner coefficients (weight given to each base):")
print("        " + " ".join(f"{n[:6]:>7s}" for n in names))
print("  naive " + " ".join(f"{c:7.2f}" for c in meta_naive.coef_[0]))
print("   oof  " + " ".join(f"{c:7.2f}" for c in meta_oof.coef_[0]))
knn_j = names.index("one_nn")
print(f"\n  in-sample 1-NN PR-AUC on train = {average_precision_score(ytr, naive_tr[:, knn_j]):.3f} "
      f"(memorizes) vs OOF 1-NN on train = {average_precision_score(ytr, oof_tr[:, knn_j]):.3f}")
print(f"  NAIVE stack: train PR-AUC = {naive_train_ap:.3f} (inflated) -> test PR-AUC = {naive_test_ap:.3f}")
print(f"  OOF   stack: train PR-AUC = {oof_train_ap:.3f}           -> test PR-AUC = {oof_test_ap:.3f}")
print(f"  held-out gap (OOF - naive) = {oof_test_ap - naive_test_ap:+.3f}")
print("\nDONE")
