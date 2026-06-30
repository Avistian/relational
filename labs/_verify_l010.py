"""Scratch verification for Lesson 010 Q1 checkpoint capstone. Deleted after use.

Assembles the whole leakage spine on one mixed-type, imbalanced, missing-laden
dataset and prints an honest comparison: dummy vs logistic vs HistGBDT, with
PR-AUC (and prevalence baseline), ROC-AUC, and Brier raw-vs-calibrated.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RS = 0
rng = np.random.RandomState(RS)
n = 8000

# --- raw mixed-type data ---
a = rng.uniform(1, 9, n); b = rng.uniform(1, 9, n)
c = rng.uniform(1, 9, n); d = rng.uniform(9.5, 11, n)   # c-d bounded off zero
x_extra = rng.normal(0, 1, n)
region = rng.choice(["north", "south", "east", "west"], size=n, p=[.4, .3, .2, .1])
region_effect = pd.Series(region).map({"north": 0.0, "south": 0.4, "east": -0.3, "west": 0.2}).to_numpy()

# signal: irregular threshold INTERACTIONS (what trees win on; Grinsztajn 2022),
# plus a smaller ratio-of-differences term and region. Linear models can't carve
# the axis-aligned steps; HistGBDT can.
inter1 = ((a > 6) & (b < 3)).astype(float)
inter2 = ((c > 5) & (d > 10.3)).astype(float)
nonmono = (np.abs(x_extra) > 1.0).astype(float)        # U-shaped in x_extra
ratio = (a - b) / (c - d)
logit = (2.4 * inter1 + 2.0 * inter2 + 1.1 * nonmono
         + 0.6 * ratio + 1.0 * region_effect)
logit = (logit - logit.mean()) / logit.std()
p = 1 / (1 + np.exp(-(logit * 1.4 - 2.3)))     # shift to make positives rare
y = rng.binomial(1, p)

df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "x_extra": x_extra, "region": region})

# inject MAR missingness into x_extra (depends on observed 'a')
miss = (a > 6) & (rng.uniform(size=n) < 0.5)
df.loc[miss, "x_extra"] = np.nan

print(f"n={n}, positives={y.mean():.3f}, x_extra missing={df['x_extra'].isna().mean():.3f}")

num_cols = ["a", "b", "c", "d", "x_extra"]
cat_cols = ["region"]

def add_ratio(X):
    X = X.copy()
    X["ratio"] = (X["a"] - X["b"]) / (X["c"] - X["d"])
    return X

ratio_step = FunctionTransformer(add_ratio)

def make_pre(scale):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols + ["ratio"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, stratify=y, random_state=RS)
prevalence = yte.mean()
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

models = {
    "Dummy(prior)": (DummyClassifier(strategy="prior"), False),
    "Logistic":     (LogisticRegression(max_iter=1000), True),
    "HistGBDT":     (HistGradientBoostingClassifier(random_state=RS), False),
}

print(f"\n{'model':14s} {'CV PR-AUC':>10s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Brier':>8s}")
print(f"{'(baseline)':14s} {prevalence:10.3f} {'':8s} {'':8s} {'':8s}  <- prevalence")
for name, (clf, scale) in models.items():
    pipe = Pipeline([("ratio", ratio_step), ("pre", make_pre(scale)), ("clf", clf)])
    cv_pr = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="average_precision").mean()
    pipe.fit(Xtr, ytr)
    if hasattr(pipe, "predict_proba"):
        pp = pipe.predict_proba(Xte)[:, 1]
    else:
        pp = pipe.decision_function(Xte)
    pr = average_precision_score(yte, pp)
    roc = roc_auc_score(yte, pp)
    brier = brier_score_loss(yte, np.clip(pp, 0, 1)) if pp.min() >= 0 else float("nan")
    print(f"{name:14s} {cv_pr:10.3f} {pr:8.3f} {roc:8.3f} {brier:8.3f}")

# calibration on the winning HistGBDT
base = Pipeline([("ratio", ratio_step), ("pre", make_pre(False)),
                 ("clf", HistGradientBoostingClassifier(random_state=RS))])
base.fit(Xtr, ytr)
p_raw = base.predict_proba(Xte)[:, 1]
cal = CalibratedClassifierCV(base, method="sigmoid", cv=5).fit(Xtr, ytr)
p_cal = cal.predict_proba(Xte)[:, 1]
print(f"\nHistGBDT Brier raw {brier_score_loss(yte, p_raw):.4f} -> calibrated {brier_score_loss(yte, p_cal):.4f}")
print(f"HistGBDT ROC raw {roc_auc_score(yte, p_raw):.3f} -> calibrated {roc_auc_score(yte, p_cal):.3f}")
