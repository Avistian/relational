"""Build labs/0018-ensembling-stacking.ipynb (student, blanks) and solutions/ (filled).

Run:  python labs/_build_l018.py
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
md("""# Lab 018 — Ensembling & stacking: blend, then stack without leaking

**Lesson:** [`lessons/0018-ensembling-stacking.html`](../lessons/0018-ensembling-stacking.html) · **Phase / Year:** Year 1 · Q2

**Paper:** Wolpert 1992, *Stacked Generalization*, [Neural Networks 5(2):241–259](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) — §1–3 (a level-1 model trained on level-0 predictions of held-out data).

**Dataset tier:** A — OpenML `credit_g` via `relkit` (same harness as Labs 011–017).

**Skill you are practising:** build **out-of-fold (OOF) meta-features** by hand, see why the meta-learner must never see in-sample base predictions (the leak that crowns a memorizer), and run sklearn's `StackingClassifier`.

**Exit criteria:** EXIT TICKET prints (1) the simple OOF blend vs the best single model, (2) the leaky in-sample stack vs the honest OOF stack on a held-out split, and (3) `StackingClassifier` CV PR-AUC.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____` / `# TODO`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn**, **xgboost**, **lightgbm**, **catboost** (already in `requirements-labs.txt`). No local install? Open from [`notebooks.html`](../notebooks.html) — **View** to read it rendered, or **Run on Binder**.""")

# ---------------------------------------------------------------- recap
md(r"""## Concept recap — combine diverse models, without leaking

You have several tuned single-table models. **Ensembling** asks: given each model's prediction for a row, what is my final prediction? Three levels:

- **Averaging / blending** — mean of the base models' predicted probabilities. Zero extra parameters.
- **Weighted blend** — a convex combination $\sum_i w_i p_i$ with chosen weights.
- **Stacking (Wolpert 1992)** — treat the base predictions as *features* and train a small **meta-learner** (level-1, e.g. logistic regression) to combine them; it *learns* the weights.

**Why it can beat the best single model — diversity.** If two models make *independent* errors, averaging halves the error variance; if they make the *same* errors, averaging does nothing. So the fuel is **diverse, individually-good** base models. On `credit_g` the three GBDTs correlate $\approx 0.89$ (nearly the same model), while the logistic model correlates only $\approx 0.68$ with them — it carries the diversity.

**The one hard part — out-of-fold predictions.** You cannot train the meta-learner on predictions a base model made on its *own* training rows: those are memorized copies of the label, not forecasts. The fix is **OOF**: split into $K$ folds; for each fold, train the base on the other $K-1$ folds and predict the held-out fold. Stitch the held-out predictions together and every row has a prediction from a model that *never saw it*. `sklearn.model_selection.cross_val_predict(est, X, y, cv=K, method="predict_proba")` does exactly this.

**The leak, made concrete.** Add a pure memorizer — a **1-nearest-neighbour** classifier, which is *perfect* in-sample (each row's nearest neighbour is itself) but useless out-of-fold. Train the meta-learner on **in-sample** predictions and it hands the 1-NN the biggest weight; its apparent score is a mirage and it collapses on new data. Train on **OOF** predictions and the meta-learner sees the 1-NN's true (weak) signal and ignores it.

**Toy micro-example (not this lab's answer).** Two models, each 80% accurate. If their mistakes are on the *same* rows, the average is still 80%. If their mistakes are on *disjoint* rows, a soft average can be right whenever *either* is confident — pushing accuracy above 80%. Independent errors are what a stack monetizes.

**Why it matters for the thesis:** tabular leaderboards (TabArena, Kaggle) are won by *leak-free stacked ensembles* of tuned models — not single defaults. The RDL undervaluation bet must beat that stronger baseline, under the same CV protocol. Full derivation + the OOF widget: [Lesson 018](../lessons/0018-ensembling-stacking.html) · Wolpert 1992.""")

# ---------------------------------------------------------------- setup
md("## Setup — PROVIDED")

code("""# PROVIDED
import sys
from pathlib import Path
_here = Path(".").resolve()
for _p in (_here, _here.parent):          # put labs/ on the path (works from labs/ or labs/solutions/)
    sys.path.insert(0, str(_p))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from relkit.data import load_tier_a

RS = 0

# credit_g, label-encoded so every model shares one numeric matrix (as in L017).
X, y = load_tier_a("credit_g")
y = np.asarray(y)
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = [c for c in X.columns if c not in num]
Xle = X[num].copy()
for c in cat:
    Xle[c] = LabelEncoder().fit_transform(X[c].astype(str))
Xle = Xle.to_numpy(dtype=float)

def base_models():
    \"\"\"Fresh, clone-safe base learners on the label-encoded matrix.\"\"\"
    return {
        "logistic":      make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RS)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=4),
        "xgboost":       XGBClassifier(random_state=RS, n_jobs=4, eval_metric="logloss", tree_method="hist"),
        "lightgbm":      LGBMClassifier(random_state=RS, n_jobs=4, verbose=-1),
        "catboost":      CatBoostClassifier(random_state=RS, verbose=0),
    }

cv = StratifiedKFold(5, shuffle=True, random_state=RS)
print("relkit + 5 base models ready; X:", Xle.shape)""")

# ---------------------------------------------------------------- Task 1
md(r"""## Task 1 — out-of-fold meta-features + a blend — TODO (crucial fragment)

**Goal:** build the **out-of-fold probability** for each base model with `cross_val_predict`, average the five columns into a **blend**, and score it against the best single model.

**Why it matters:** the OOF prediction is *the* object stacking is built on — a prediction for every row from a model that never trained on that row. Averaging them is the simplest ensemble; the meta-learner in Task 3 just learns the weights instead of fixing them at $1/5$.

**You implement:** for each base model, `cross_val_predict(est, Xle, y, cv=cv, method="predict_proba")[:, 1]` gives its OOF positive-class probability; stack the five columns and take the row-wise **mean** for the blend.

**Hint boundary:** use the PROVIDED `cv` and `method="predict_proba"`; take column `[:, 1]` (positive class); `np.column_stack` then `.mean(axis=1)`. Do not fit on the full data and predict the same rows — that is the leak Task 2 exposes.""")

todo(
    """# TODO — build OOF probabilities for each base model, then blend
single = {}
oof = {}
for name, est in base_models().items():
    single[name] = float(cross_val_score(est, Xle, y, cv=cv, scoring="average_precision").mean())
    oof[name] = cross_val_predict(est, Xle, y, cv=cv, method="predict_proba")[:, 1]

P = np.column_stack([oof[n] for n in base_models()])   # rows x 5 OOF columns
blend = P.mean(axis=1)
blend_ap = average_precision_score(y, blend)

best_name = max(single, key=single.get)
for n, v in single.items():
    print(f"  {n:16s} {v:.3f}")
print(f"  best single = {best_name} {single[best_name]:.3f} | blend = {blend_ap:.3f}")""",
    """# TODO — build OOF probabilities for each base model, then blend
single = {}
oof = {}
for name, est in base_models().items():
    single[name] = float(cross_val_score(est, Xle, y, cv=cv, scoring="average_precision").mean())
    oof[name] = ____          # cross_val_predict(...): OOF positive-class prob for this model

P = ____                      # column-stack the five OOF probability vectors -> rows x 5
blend = ____                  # row-wise mean of the five columns
blend_ap = average_precision_score(y, blend)

best_name = max(single, key=single.get)
for n, v in single.items():
    print(f"  {n:16s} {v:.3f}")
print(f"  best single = {best_name} {single[best_name]:.3f} | blend = {blend_ap:.3f}")""",
)

code("""# CHECK — do not edit
assert P.shape == (len(y), 5), "P should be one OOF probability column per base model."
assert ((P >= 0) & (P <= 1)).all(), "OOF values are probabilities in [0, 1]."
assert abs(blend_ap - 0.899) < 0.02, "Blend PR-AUC should reproduce ~0.899 on credit_g."
assert abs(single["random_forest"] - 0.901) < 0.02, "RF (best single) should be ~0.901."
print(f"Task 1 ok — blend {blend_ap:.3f} vs best single {single[best_name]:.3f} "
      f"(on tiny credit_g the blend barely moves — the mechanism is the point).")""")

# ---------------------------------------------------------------- Task 2
md(r"""## Task 2 — the leakage trap: in-sample vs OOF meta-features — TODO (leakage discipline)

**Goal:** add a **1-NN memorizer** to the pool, then fit a logistic meta-learner two ways on a 70/30 split — once on **in-sample** base predictions (leak) and once on **OOF** base predictions (correct) — and compare held-out PR-AUC.

**Why it matters:** this is the single most common stacking bug. The 1-NN is perfect in-sample and useless out-of-fold; the leaky meta-learner crowns it and collapses on new data. Reproducing the collapse cements *why* OOF is mandatory.

**Reproduction target (lesson table):** naïve stack train PR-AUC ≈ 1.000 (mirage) → test ≈ 0.895; OOF stack train ≈ 0.885 → test ≈ 0.930 (held-out gap ≈ +0.035).

**You implement:** the **OOF** training meta-features `oof_tr` — for each model, `cross_val_predict` on the *training* split with the PROVIDED `inner` splitter. The naïve (in-sample) meta-features and the test meta-features are PROVIDED.""")

code("""# PROVIDED — pool + a pure memorizer; fit bases on the full training split.
def leak_models():
    m = base_models()
    m["one_nn"] = KNeighborsClassifier(n_neighbors=1)   # perfect in-sample, useless OOF
    return m

leak_names = list(leak_models())
Xtr, Xte, ytr, yte = train_test_split(Xle, y, test_size=0.3, stratify=y, random_state=RS)
inner = StratifiedKFold(5, shuffle=True, random_state=RS)

fitted = {n: leak_models()[n].fit(Xtr, ytr) for n in leak_names}
# NAIVE meta-features: each base predicts the SAME rows it trained on (the leak)
naive_tr = np.column_stack([fitted[n].predict_proba(Xtr)[:, 1] for n in leak_names])
# test meta-features (same for both): bases (fit on all train) predict the held-out test rows
te_feat  = np.column_stack([fitted[n].predict_proba(Xte)[:, 1] for n in leak_names])
print("naive_tr and te_feat ready; columns =", leak_names)""")

todo(
    """# TODO — build HONEST out-of-fold training meta-features, then compare both meta-learners
oof_tr = np.column_stack(
    [cross_val_predict(leak_models()[n], Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
     for n in leak_names]
)

meta_naive = LogisticRegression(max_iter=2000, random_state=RS).fit(naive_tr, ytr)
meta_oof   = LogisticRegression(max_iter=2000, random_state=RS).fit(oof_tr,   ytr)

naive_train = average_precision_score(ytr, meta_naive.predict_proba(naive_tr)[:, 1])
oof_train   = average_precision_score(ytr, meta_oof.predict_proba(oof_tr)[:, 1])
naive_test  = average_precision_score(yte, meta_naive.predict_proba(te_feat)[:, 1])
oof_test    = average_precision_score(yte, meta_oof.predict_proba(te_feat)[:, 1])

j = leak_names.index("one_nn")
print(f"1-NN weight  naive={meta_naive.coef_[0][j]:+.2f}  oof={meta_oof.coef_[0][j]:+.2f}")
print(f"NAIVE stack: train {naive_train:.3f} (mirage) -> test {naive_test:.3f}")
print(f"OOF   stack: train {oof_train:.3f}          -> test {oof_test:.3f}")
print(f"held-out gap (oof - naive) = {oof_test - naive_test:+.3f}")""",
    """# TODO — build HONEST out-of-fold training meta-features, then compare both meta-learners
oof_tr = ____   # column-stack cross_val_predict(model, Xtr, ytr, cv=inner, method="predict_proba")[:,1]
                # for each name in leak_names

meta_naive = LogisticRegression(max_iter=2000, random_state=RS).fit(naive_tr, ytr)
meta_oof   = LogisticRegression(max_iter=2000, random_state=RS).fit(oof_tr,   ytr)

naive_train = average_precision_score(ytr, meta_naive.predict_proba(naive_tr)[:, 1])
oof_train   = average_precision_score(ytr, meta_oof.predict_proba(oof_tr)[:, 1])
naive_test  = average_precision_score(yte, meta_naive.predict_proba(te_feat)[:, 1])
oof_test    = average_precision_score(yte, meta_oof.predict_proba(te_feat)[:, 1])

j = leak_names.index("one_nn")
print(f"1-NN weight  naive={meta_naive.coef_[0][j]:+.2f}  oof={meta_oof.coef_[0][j]:+.2f}")
print(f"NAIVE stack: train {naive_train:.3f} (mirage) -> test {naive_test:.3f}")
print(f"OOF   stack: train {oof_train:.3f}          -> test {oof_test:.3f}")
print(f"held-out gap (oof - naive) = {oof_test - naive_test:+.3f}")""",
)

code("""# CHECK — do not edit
assert oof_tr.shape == naive_tr.shape, "OOF meta-features must have the same shape as the naive ones."
assert naive_train > 0.99, "In-sample meta-features should look near-perfect (the mirage) because of the 1-NN."
assert oof_train < naive_train, "OOF training score must be lower/honest than the leaked in-sample score."
assert oof_test >= naive_test, "Not leaking should NOT hurt held-out performance (here it clearly helps)."
print(f"Task 2 ok — leaking cost {oof_test - naive_test:+.3f} held-out PR-AUC; "
      f"OOF meta-learner ignored the 1-NN memorizer.")""")

# ---------------------------------------------------------------- Task 3
md(r"""## Task 3 — sklearn `StackingClassifier` — TODO (reproduction target)

**Goal:** let scikit-learn do the OOF bookkeeping. Configure a `StackingClassifier` over the five base models with a **logistic** meta-learner and `cv=5`, and score the whole stack with an outer 5-fold CV.

**Why it matters:** `StackingClassifier` generates the level-0 features by internal cross-validation, so you never touch a leaked prediction — and wrapping it in an outer `cross_val_score` gives the honest number you are allowed to report (the L017 rule: score the *whole* fitting procedure on unseen data).

**Reproduction target (lesson table):** stack CV PR-AUC ≈ 0.902 — edging the best single model (~0.901).

**You implement:** the `final_estimator` (a `LogisticRegression`) and `cv=5` in the `StackingClassifier`; the estimator list and the outer scoring are PROVIDED.""")

todo(
    """# TODO — configure and score the stack
estimators = [(n, base_models()[n]) for n in base_models()]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, random_state=RS),
    cv=5,
    stack_method="predict_proba",
    n_jobs=4,
)
stack_ap = float(cross_val_score(stack, Xle, y, cv=cv, scoring="average_precision").mean())
print(f"StackingClassifier CV PR-AUC = {stack_ap:.3f}  (best single {single[best_name]:.3f}, blend {blend_ap:.3f})")""",
    """# TODO — configure and score the stack
estimators = [(n, base_models()[n]) for n in base_models()]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=____,        # the level-1 meta-learner: a simple LogisticRegression
    cv=____,                     # internal CV that generates leak-free OOF meta-features
    stack_method="predict_proba",
    n_jobs=4,
)
stack_ap = float(cross_val_score(stack, Xle, y, cv=cv, scoring="average_precision").mean())
print(f"StackingClassifier CV PR-AUC = {stack_ap:.3f}  (best single {single[best_name]:.3f}, blend {blend_ap:.3f})")""",
)

code("""# CHECK — do not edit
assert 0.88 < stack_ap < 0.92, "Stack PR-AUC should land near the base models' range."
assert stack_ap >= single[best_name] - 0.01, "A leak-free stack should match or edge the best single model."
print(f"Task 3 ok — stack {stack_ap:.3f} >= best single {single[best_name]:.3f} - tol.")""")

# ---------------------------------------------------------------- exit
md("""## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, say why the stacking meta-learner must be trained on out-of-fold predictions, and what makes an ensemble beat the best single model.""")

todo(
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 018 ===")
print(f"Blend (OOF avg of 5)     PR-AUC = {blend_ap:.3f}  (best single {single[best_name]:.3f})")
print(f"Leak trap: naive test {naive_test:.3f} vs OOF test {oof_test:.3f} (gap {oof_test - naive_test:+.3f})")
print(f"StackingClassifier CV    PR-AUC = {stack_ap:.3f}")
print()
print("takeaway:", "the meta-learner must train on out-of-fold base predictions because in-sample "
      "predictions are memorized copies of the label, so it would over-trust whichever base overfits; "
      "an ensemble beats the best single model only when its base models are diverse (make uncorrelated "
      "errors) and individually good, so their errors cancel when combined.")''',
    '''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 018 ===")
print(f"Blend (OOF avg of 5)     PR-AUC = {blend_ap:.3f}  (best single {single[best_name]:.3f})")
print(f"Leak trap: naive test {naive_test:.3f} vs OOF test {oof_test:.3f} (gap {oof_test - naive_test:+.3f})")
print(f"StackingClassifier CV    PR-AUC = {stack_ap:.3f}")
print()
print("takeaway:", "____")''',
)

# ---------------------------------------------------------------- stretch
md("""## Stretch (optional, ungraded) — diversity & passthrough

1. **Drop the boosters, keep diversity.** Score a stack of just `{logistic, random_forest, catboost}` — fewer, more diverse models often match all five.
2. **`passthrough=True`.** Let the meta-learner also see the raw features, not only base predictions. Does it help here, or just add variance on 1 000 rows?
3. **Correlation → gain.** Recompute `np.corrcoef(P.T)` from Task 1 and confirm the three GBDTs are the most correlated block — the reason stacking only GBDTs barely helps.""")

code('''# STRETCH — ungraded.
# diverse = [(n, base_models()[n]) for n in ("logistic", "random_forest", "catboost")]
# s = StackingClassifier(diverse, final_estimator=LogisticRegression(max_iter=2000),
#                        cv=5, stack_method="predict_proba", n_jobs=4)
# print("diverse-3 stack:", round(float(cross_val_score(s, Xle, y, cv=cv, scoring="average_precision").mean()), 3))
# print("OOF correlation matrix:\\n", np.round(np.corrcoef(P.T), 2))''')


def build(cells, *, solution):
    nb_cells = []
    for i, entry in enumerate(cells):
        kind = entry[0]
        cid = f"l018-{i:02d}"
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


(LABS / "0018-ensembling-stacking.ipynb").write_text(json.dumps(build(CELLS, solution=False), indent=1) + "\n")
(SOL / "0018-ensembling-stacking.ipynb").write_text(json.dumps(build(CELLS, solution=True), indent=1) + "\n")
print("wrote labs/0018-ensembling-stacking.ipynb and solutions/0018-ensembling-stacking.ipynb")
