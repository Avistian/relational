"""Builds labs/0010-baseline-checkpoint.ipynb from cell specs. Deleted after use."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Lab 010 — Q1 Checkpoint: a reproducible baseline (capstone)

**Lesson:** [`lessons/0010-baseline-checkpoint.html`](../lessons/0010-baseline-checkpoint.html) · **Phase / Year:** Phase 1 / Year 1 · **capstone**

**Skill you are practising:** assemble Lessons 001–009 into one leak-free `Pipeline`, benchmark it against the dummy floor with the right metric, and calibrate the winner — reproducibly.

**Exit criteria:** the EXIT TICKET prints the model comparison (CV PR-AUC, test PR-AUC vs prevalence, ROC-AUC, Brier) and the calibration delta for the winner.

---

### How this notebook works
- **PROVIDED** cells are complete — just run them.
- **TODO** cells have blanks (`____`). Fill them in.
- **CHECK** cells auto-run and give you immediate feedback — don't edit them.
- Run top to bottom. When the **EXIT TICKET** prints cleanly, paste it back to your teacher (or say *"lab done"*).

### Environment
One-time setup from the repo root: `bash labs/setup-env.sh`.  
Then select kernel **Relational Labs (.venv)** or interpreter `.venv/bin/python`.""")

md("## Setup — PROVIDED (run me)")
code("""# PROVIDED
import numpy as np, pandas as pd, sklearn
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
print("sklearn", sklearn.__version__, "| seed", RS)   # reproducibility: pinned env + fixed seed""")

md("""## Data — PROVIDED

One realistic dataset that exercises the whole spine: **mixed-type** (numeric + a `region`
category), **imbalanced** (~15% positive), **MAR missingness** in `x_extra`, and an **irregular
threshold-interaction** signal (the regime where trees win — Grinsztajn 2022) plus a small
ratio-of-differences term.""")
code("""# PROVIDED
rng = np.random.RandomState(RS)
n = 8000
a = rng.uniform(1, 9, n); b = rng.uniform(1, 9, n)
c = rng.uniform(1, 9, n); d = rng.uniform(9.5, 11, n)
x_extra = rng.normal(0, 1, n)
region = rng.choice(["north", "south", "east", "west"], size=n, p=[.4, .3, .2, .1])
region_effect = pd.Series(region).map({"north":0.0,"south":0.4,"east":-0.3,"west":0.2}).to_numpy()

inter1 = ((a > 6) & (b < 3)).astype(float)
inter2 = ((c > 5) & (d > 10.3)).astype(float)
nonmono = (np.abs(x_extra) > 1.0).astype(float)
logit = 2.4*inter1 + 2.0*inter2 + 1.1*nonmono + 0.6*((a-b)/(c-d)) + 1.0*region_effect
logit = (logit - logit.mean()) / logit.std()
y = rng.binomial(1, 1/(1+np.exp(-(logit*1.4 - 2.3))))

df = pd.DataFrame({"a":a,"b":b,"c":c,"d":d,"x_extra":x_extra,"region":region})
df.loc[(a > 6) & (rng.uniform(size=n) < 0.5), "x_extra"] = np.nan   # MAR missingness

num_cols = ["a","b","c","d","x_extra"]
cat_cols = ["region"]
print(f"n={n}  positives={y.mean():.3f}  x_extra missing={df['x_extra'].isna().mean():.3f}")""")

# Task 1: split + dummy floor
md("""## Task 1 — Hold out a test set & set the floor — TODO

Split off a **stratified** 30% test set (L003), then record the **prevalence** (the PR-AUC
no-skill floor, L008). Stratify so the rare class is represented in both halves.

Hint: `train_test_split(..., test_size=0.3, stratify=y, random_state=RS)`.""")
code("""# TODO: stratified split, then prevalence.
Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, stratify=____, random_state=RS)
prevalence = yte.mean()
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
print(f"train {len(Xtr)}  test {len(Xte)}  prevalence (PR floor) {prevalence:.3f}")""")
code("""# CHECK — do not edit.
assert abs(ytr.mean() - yte.mean()) < 0.02, "Stratify on y so train/test prevalence match."
assert 0.10 < prevalence < 0.20, "Prevalence should be ~0.15 here."
print(f"Task 1 ok — stratified split; PR-AUC floor is prevalence = {prevalence:.3f}.")""")

# Task 2: build the pipeline
md("""## Task 2 — Build the leak-free pipeline — TODO

Assemble one `Pipeline`: stateless ratio engineering (L009) → a `ColumnTransformer` that imputes
**with a missing-indicator** (L006) and one-hot-encodes `region` → the classifier. Every
fit-bearing step lives **inside** the pipeline so it only sees training folds (L005).

Fill: the `add_indicator=True` imputer, and the `ColumnTransformer` numeric column list (numeric
cols **plus** the engineered `"ratio"`).""")
code('''def add_ratio(X):
    X = X.copy()
    X["ratio"] = (X["a"] - X["b"]) / (X["c"] - X["d"])     # L009 stateless feature
    return X

def make_pipe(clf, scale):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=____))]  # TODO: indicator on
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), ____),                 # TODO: num_cols + ["ratio"]
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("engineer", FunctionTransformer(add_ratio)), ("pre", pre), ("clf", clf)])

print("pipeline factory ready")''')
code("""# CHECK — do not edit.
_p = make_pipe(LogisticRegression(max_iter=1000), scale=True)
_p.fit(Xtr, ytr)
assert _p.named_steps["pre"].transformers_[0][2] == num_cols + ["ratio"], "Numeric branch must include the engineered 'ratio'."
imp = _p.named_steps["pre"].named_transformers_["num"].named_steps["impute"]
assert imp.add_indicator is True, "Imputer must add a missing-indicator (L006)."
print("Task 2 ok — ratio engineered in-pipeline; impute+indicator; OHE for region.")""")

# Task 3: bake-off
md("""## Task 3 — Bake-off vs the dummy floor — TODO

Score three models with **CV PR-AUC** (L007/L008) on the training set, then evaluate on test.
The scorer for PR-AUC / average precision is `"average_precision"`.

Fill the `scoring=` argument.""")
code('''models = {
    "Dummy(prior)": (DummyClassifier(strategy="prior"), False),
    "Logistic":     (LogisticRegression(max_iter=1000), True),
    "HistGBDT":     (HistGradientBoostingClassifier(random_state=RS), False),
}
rows = {}
for name, (clf, scale) in models.items():
    pipe = make_pipe(clf, scale)
    cv_pr = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=____).mean()   # TODO: PR-AUC scorer
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    rows[name] = (cv_pr, average_precision_score(yte, p), roc_auc_score(yte, p), brier_score_loss(yte, p))

print(f"{'model':14s}{'CV PR':>8s}{'PR':>8s}{'ROC':>8s}{'Brier':>8s}")
print(f"{'prevalence':14s}{prevalence:8.3f}")
for name,(cvp,pr,roc,br) in rows.items():
    print(f"{name:14s}{cvp:8.3f}{pr:8.3f}{roc:8.3f}{br:8.3f}")''')
code("""# CHECK — do not edit.
assert abs(rows["Dummy(prior)"][1] - prevalence) < 0.02, "Dummy PR-AUC should equal prevalence."
assert rows["HistGBDT"][0] > rows["Logistic"][0], "HistGBDT should win CV PR-AUC on this irregular signal."
assert rows["HistGBDT"][0] > prevalence + 0.2, "The winner should clearly beat the floor."
print(f"Task 3 ok — HistGBDT wins (CV PR {rows['HistGBDT'][0]:.3f}) and crushes the dummy floor {prevalence:.3f}.")""")

# Task 4: calibrate winner
md("""## Task 4 — Calibrate the winner — TODO

Wrap the winning HistGBDT pipeline in `CalibratedClassifierCV(method="sigmoid", cv=5)`, fit on
train, and compare Brier raw vs calibrated (L008). Get calibrated positive-class probabilities
with `predict_proba(Xte)[:, 1]`.""")
code('''winner = make_pipe(HistGradientBoostingClassifier(random_state=RS), scale=False).fit(Xtr, ytr)
p_raw = winner.predict_proba(Xte)[:, 1]

cal = CalibratedClassifierCV(make_pipe(HistGradientBoostingClassifier(random_state=RS), scale=False),
                             method="sigmoid", cv=5).fit(Xtr, ytr)
p_cal = ____                                   # TODO: calibrated positive-class probabilities

brier_raw, brier_cal = brier_score_loss(yte, p_raw), brier_score_loss(yte, p_cal)
print(f"HistGBDT Brier raw {brier_raw:.4f} -> calibrated {brier_cal:.4f}")
print(f"HistGBDT ROC   raw {roc_auc_score(yte, p_raw):.3f} -> calibrated {roc_auc_score(yte, p_cal):.3f}")''')
code("""# CHECK — do not edit.
assert p_cal.shape == yte.shape, "p_cal should be one probability per test row."
assert brier_cal <= brier_raw + 1e-6, "Calibration should not worsen Brier here."
print(f"Task 4 ok — calibration moved Brier {brier_raw:.4f} -> {brier_cal:.4f} with ranking ~unchanged.")""")

# Exit ticket
md("""## Exit ticket — TODO

Fill the one-sentence takeaway, run, and paste the output back to your teacher.""")
code('''# TODO: complete the takeaway.
print("=== EXIT TICKET — Lesson 010 (Q1 checkpoint) ===")
print(f"prevalence (PR floor) {prevalence:.3f}")
for name,(cvp,pr,roc,br) in rows.items():
    print(f"  {name:14s} CV-PR {cvp:.3f}  PR {pr:.3f}  ROC {roc:.3f}  Brier {br:.3f}")
print(f"winner calibration: Brier {brier_raw:.4f} -> {brier_cal:.4f}")
print()
print("takeaway:", "____")   # which model is the baseline to beat, why did it win, and what did calibration change?''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Relational Labs (.venv)", "language": "python", "name": "relational-labs"},
    "language_info": {"name": "python"},
}
out = "/home/avist/Projects/relational/labs/0010-baseline-checkpoint.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
