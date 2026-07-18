"""Build Lab 030 (Q3 checkpoint — write a one-page benchmark report) — student + solution.

Tier A (real credit_g via relkit). This is a CONSOLIDATION capstone: the learner assembles the whole
Q3 evaluation-rigor toolkit into one defensible benchmark report. The two crucial fragments are the
Q3 evaluation skills (not model internals):
  * Task 1 — the L024 budget-curve selection rule: keep the config with the best VALIDATION score,
    report its TEST score (best-so-far), for the GBDT and the honest neural baseline.
  * Task 2 — the L023 corrected resampled t-test: inflate the paired-fold variance by (1/n + 1/(k-1))
    so overlapping folds don't make the naive test over-reject.
  * Task 3 — assemble the one-page report (leakage audit + honest verdict).

The neural contestant is sklearn's MLPClassifier (torch-free, runs anywhere) standing in for the L028
tabular ResNet — same inductive biases; the checkpoint is about the report, not the architecture.

Run: .venv/bin/python labs/_build_l030.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0030-q3-checkpoint.ipynb labs/solutions/0030-q3-checkpoint.ipynb
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


SETUP = r'''# PROVIDED — imports, real credit_g (Tier A), preprocessing, split, and the two family samplers. Just run.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 0
np.random.seed(SEED)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))          # labs/ when run from there
sys.path.insert(0, str(Path(".").resolve().parent))   # labs/ when run from labs/solutions/
from relkit import load_tier_a
X, y = load_tier_a("credit_g")
y = np.asarray(y)
NUM = X.select_dtypes(include="number").columns.tolist()
CAT = [c for c in X.columns if c not in NUM]
print(f"credit_g (Tier A): X {X.shape}, positive rate {y.mean():.3f}, {len(NUM)} numeric + {len(CAT)} categorical cols")

def make_prep():
    # per-fold-safe: impute+scale numeric, impute+one-hot categorical
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])

def fit_auc(clf, Xtr, ytr, Xev, yev):
    pipe = Pipeline([("prep", make_prep()), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe, roc_auc_score(yev, pipe.predict_proba(Xev)[:, 1])

def split(seed):
    # one 60/20/20 train / validation / test split
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    return Xtr, Xva, Xte, ytr, yva, yte

# --- the two headline families (defaults + a small random search space) ---
def sample_gbdt(rng):
    return HistGradientBoostingClassifier(
        learning_rate=float(10 ** rng.uniform(-2, -0.3)), max_leaf_nodes=int(rng.choice([15, 31, 63])),
        l2_regularization=float(10 ** rng.uniform(-3, 1)), max_iter=int(rng.choice([100, 200, 300])),
        random_state=0)

_MLP_SHAPES = [(128,), (256, 128), (256, 256)]
def sample_mlp(rng):
    return MLPClassifier(
        hidden_layer_sizes=_MLP_SHAPES[int(rng.randint(len(_MLP_SHAPES)))],
        alpha=float(10 ** rng.uniform(-5, -1)), learning_rate_init=float(10 ** rng.uniform(-3.5, -2)),
        max_iter=300, early_stopping=True, random_state=0)

print("ready — helpers: make_prep, fit_auc(clf, Xtr,ytr, Xev,yev), split(seed), sample_gbdt(rng), sample_mlp(rng)")'''


# ---- Task 1: budget curve ----
T1_MD = r'''## Task 1 — the budget curve: select by validation, report test (crucial fragment) — TODO

**Goal:** for each family, run a random search and record the **best-so-far TEST AUC** — where "best"
is decided on the **validation** split (never the test set). You implement the one line that makes this
honest: when a config's *validation* AUC beats the incumbent, adopt it and record *its* test AUC.

**Why it matters:** this is Grinsztajn's §4 protocol (L024). A single tuned number hides how good the
*default* was and how much tuning bought. The budget curve shows both — and, on a small dataset, it can
even *dip* when the tuning overfits a tiny validation slice (a finding your report must disclose).

**Hint boundary:** inside the `if va > best_va:` branch, set `best_va` to `va` and `best_te` to the
current config's test AUC `te` — the best-so-far is chosen by validation but *reported* on test.'''

T1_CODE = r'''# TODO — the budget-curve best-so-far selection (the crucial fragment)
Xtr, Xva, Xte, ytr, yva, yte = split(SEED)

def budget_curve(sampler, K=12):
    rng = np.random.RandomState(SEED)
    best_va, best_te, curve = -1.0, None, []
    for _ in range(K):
        clf = sampler(rng)
        pipe, va = fit_auc(clf, Xtr, ytr, Xva, yva)          # score THIS config on validation
        te = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
        if va > best_va:
            best_va, best_te = ____                            # adopt this config: best-VAL, report its TEST
        curve.append(best_te)
    return curve

gbdt_curve = budget_curve(sample_gbdt)
mlp_curve = budget_curve(sample_mlp)
print(f"GBDT : default {gbdt_curve[0]:.3f}  ->  ceiling {gbdt_curve[-1]:.3f}")
print(f"MLP  : default {mlp_curve[0]:.3f}  ->  ceiling {mlp_curve[-1]:.3f}")'''

T1_SOL = T1_CODE.replace(
    "            best_va, best_te = ____                            # adopt this config: best-VAL, report its TEST",
    "            best_va, best_te = va, te                          # adopt this config: best-VAL, report its TEST")

T1_CHECK = r'''# CHECK — do not edit
# NOTE: best-so-far is chosen by VALIDATION but reported on TEST, so the TEST curve need NOT be
# monotone — it can dip when a higher-validation config has a lower test score (small-data overfitting).
for name, c in [("GBDT", gbdt_curve), ("MLP", mlp_curve)]:
    assert len(c) == 12, "curve should have one point per budget step."
    assert all(v is not None for v in c), f"{name}: every step must record the best-so-far TEST AUC."
    assert 0.70 <= c[0] <= 0.87 and 0.70 <= c[-1] <= 0.87, f"{name}: AUCs should be in a sane range for credit_g."
print(f"Task 1 ok — budget curves built. GBDT {gbdt_curve[0]:.3f}->{gbdt_curve[-1]:.3f}, "
      f"MLP {mlp_curve[0]:.3f}->{mlp_curve[-1]:.3f}. Both families land ~0.80; the curve shows default AND ceiling "
      f"(and may dip if tuning overfits a small validation slice — a finding to disclose).")'''


# ---- Task 2: corrected resampled t-test ----
T2_MD = r'''## Task 2 — is the gap significant? the corrected resampled t-test (crucial fragment) — TODO

**Goal:** run a paired **5×5 repeated stratified CV** of the two families at their defaults (same folds
for both), then test whether the mean paired gap is distinguishable from noise. You implement the
**correction** that makes the test honest.

**Why it matters:** k-fold training sets **overlap**, so the per-fold scores are positively correlated.
A naive paired t-test treats them as independent and **over-rejects** (L023). Nadeau & Bengio's fix:
inflate the variance by a factor `(1/n + rho)`, where `rho = n_test/n_train = 1/(k-1)` for k-fold. The
corrected statistic can only make a gap *harder* to call significant, never easier.

**Hint boundary:** the corrected denominator is `sqrt( (1/n + 1/(k-1)) * var_d )`, where `var_d` is the
sample variance of the paired differences (`ddof=1`), `n` is the number of folds, and `k = n_splits`.
Fill in the corrected variance term (everything under the square root).'''

T2_CODE = r'''# TODO — paired 5x5 CV, then the corrected resampled t-test (~40s)
n_splits, n_repeats = 5, 5
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
diffs, gb, mlp = [], [], []
for tr, te in rkf.split(X, y):
    Xa, Xb, ya, yb = X.iloc[tr], X.iloc[te], y[tr], y[te]
    _, a_gb  = fit_auc(HistGradientBoostingClassifier(random_state=0), Xa, ya, Xb, yb)
    _, a_mlp = fit_auc(MLPClassifier(hidden_layer_sizes=(256,128), alpha=1e-3,
                                     max_iter=300, early_stopping=True, random_state=0), Xa, ya, Xb, yb)
    gb.append(a_gb); mlp.append(a_mlp); diffs.append(a_gb - a_mlp)

diffs = np.array(diffs)
n = len(diffs); mean_d = diffs.mean(); var_d = diffs.var(ddof=1)

# naive paired t-test (treats folds as independent -> anticonservative)
t_naive = mean_d / np.sqrt(var_d / n)
p_naive = 2 * stats.t.sf(abs(t_naive), df=n - 1)

# corrected resampled t-test (Nadeau & Bengio 2003)
corrected_var = ____                                   # (1/n + 1/(k-1)) * var_d  with k = n_splits
t_corr = mean_d / np.sqrt(corrected_var)
p_corr = 2 * stats.t.sf(abs(t_corr), df=n - 1)

print(f"mean GBDT {np.mean(gb):.3f}  mean MLP {np.mean(mlp):.3f}  gap {mean_d:+.4f} (n={n} folds)")
print(f"naive     t={t_naive:.2f}  p={p_naive:.3f}  -> {'SIG' if p_naive<0.05 else 'not sig'}")
print(f"corrected t={t_corr:.2f}  p={p_corr:.3f}  -> {'SIG' if p_corr<0.05 else 'not sig (noise)'}")'''

T2_SOL = T2_CODE.replace(
    "corrected_var = ____                                   # (1/n + 1/(k-1)) * var_d  with k = n_splits",
    "corrected_var = (1.0 / n + 1.0 / (n_splits - 1)) * var_d   # inflate variance for overlapping folds")

T2_CHECK = r'''# CHECK — do not edit
_ref = (1.0 / n + 1.0 / (n_splits - 1)) * var_d
assert abs(corrected_var - _ref) < 1e-12, "corrected_var must be (1/n + 1/(k-1)) * var_d."
assert p_corr >= p_naive - 1e-9, "The correction can only RAISE the p-value (never lower it)."
assert 0.0 <= p_corr <= 1.0, "p must be a probability."
_verdict = "TIE (not significant)" if p_corr >= 0.05 else "significant"
print(f"Task 2 ok — corrected p={p_corr:.3f} >= naive p={p_naive:.3f}. Verdict on credit_g: {_verdict}. "
      f"A mean gap a correct test cannot separate from noise is not a win.")'''


# ---- Task 3: assemble the report ----
T3_MD = r'''## Task 3 — assemble the one-page report — TODO

**Goal:** print the deliverable — a single, self-contained benchmark report. Two things to fill:

1. **The leakage audit** (`model_info_sheet`): answer the provenance questions honestly for this
   benchmark. (`credit_g` is a static UCI table with no timestamps and no obvious illegitimate feature,
   so the honest answers here are all `"yes"` / `"no leak"` — but a *real* report answers them per column.)
2. **The verdict** — one honest sentence: the effect size, the significance result, and whether there is
   a winner.

**Why it matters:** the report is the artifact a skeptic reads. Its credibility rests on the audit being
filled (not assumed) and the verdict being honest — including the honest verdict of a *tie*.

**Hint boundary:** set each audit value, and write `verdict` as one sentence naming the gap (`mean_d`),
the corrected p-value (`p_corr`), and the conclusion ("no significant winner" here).'''

T3_CODE = r'''# TODO — fill the leakage audit and the honest verdict
model_info_sheet = {
    "split matches deployment (temporal if timestamped)?": ____,   # credit_g has no timestamps -> "yes (i.i.d., no time order)"
    "every feature knowable before the label?":            ____,   # -> "yes"
    "any illegitimate / target-derived feature?":          ____,   # -> "no"
    "test set touched only once?":                         ____,   # -> "yes"
}

verdict = ____   # ONE honest sentence: name the gap (mean_d), the corrected p (p_corr), and the conclusion

# --- PROVIDED: print the one-page report ---
print("="*64)
print("BENCHMARK REPORT — credit_g (Q3 checkpoint, Lesson 030)")
print("="*64)
print(f"dataset   : credit_g  |  {X.shape[0]} rows x {X.shape[1]} cols  |  prevalence {y.mean():.3f}")
print(f"metric    : ROC-AUC   |  split: stratified (no timestamps)  |  seed {SEED}")
print("-"*64)
print("leakage audit (model info sheet):")
for q, a in model_info_sheet.items():
    print(f"  - {q:52s} {a}")
print("-"*64)
print("budget curve (default -> tuned ceiling):")
print(f"  GBDT (incumbent)      {gbdt_curve[0]:.3f} -> {gbdt_curve[-1]:.3f}")
print(f"  MLP  (neural baseline){mlp_curve[0]:.3f} -> {mlp_curve[-1]:.3f}")
print("  AutoML bar            ~ ties a tuned GBDT (L029)")
print("-"*64)
print("head-to-head (paired 5x5 CV, defaults):")
print(f"  GBDT {np.mean(gb):.3f}  vs  MLP {np.mean(mlp):.3f}  gap {mean_d:+.4f}")
print(f"  significance: naive p={p_naive:.3f}, corrected p={p_corr:.3f}")
print("  explanation : small heterogeneous single table -> the three biases (L025-27) lean to trees")
print("-"*64)
print("VERDICT:", verdict)
print("="*64)'''

T3_SOL = (T3_CODE
    .replace('"split matches deployment (temporal if timestamped)?": ____,   # credit_g has no timestamps -> "yes (i.i.d., no time order)"',
             '"split matches deployment (temporal if timestamped)?": "yes (no timestamps; i.i.d. assumed)",')
    .replace('"every feature knowable before the label?":            ____,   # -> "yes"',
             '"every feature knowable before the label?":            "yes",')
    .replace('"any illegitimate / target-derived feature?":          ____,   # -> "no"',
             '"any illegitimate / target-derived feature?":          "no",')
    .replace('"test set touched only once?":                         ____,   # -> "yes"',
             '"test set touched only once?":                         "yes",')
    .replace('verdict = ____   # ONE honest sentence: name the gap (mean_d), the corrected p (p_corr), and the conclusion',
             'verdict = (f"On credit_g the GBDT leads the honest MLP baseline by {mean_d:+.4f} ROC-AUC over 25 paired "\n'
             '           f"folds, but the corrected resampled t-test gives p={p_corr:.3f} (>0.05), so the gap is not "\n'
             '           f"significant -- NO SIGNIFICANT WINNER (a tie within noise); the GBDT is the cheapest strong "\n'
             '           f"default and the three inductive biases explain its small lean.")'))

T3_CHECK = r'''# CHECK — do not edit
assert all(isinstance(v, str) and v and v != "____" for v in model_info_sheet.values()), "Fill every audit answer."
assert isinstance(verdict, str) and len(verdict) > 40, "Write a real one-sentence verdict."
_v = verdict.lower()
assert ("not significant" in _v) or ("no significant" in _v) or ("tie" in _v), \
    "On credit_g the honest verdict is a tie / no significant winner — say so."
print("Task 3 ok — one-page report assembled with a filled audit and an honest verdict.")'''


EXIT_MD = r'''## Exit ticket — TODO

**Goal:** re-print the report as your deliverable and add a one-line takeaway.

**Takeaway prompt:** in one sentence, what makes a benchmark report defensible, and why is "no
significant winner" a legitimate — even valuable — conclusion?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 030 (Q3 checkpoint) ===")
print(f"credit_g  |  GBDT {np.mean(gb):.3f} vs MLP {np.mean(mlp):.3f}  gap {mean_d:+.4f}  corrected p={p_corr:.3f}")
print(f"budget curve: GBDT {gbdt_curve[0]:.3f}->{gbdt_curve[-1]:.3f}, MLP {mlp_curve[0]:.3f}->{mlp_curve[-1]:.3f}")
print("verdict:", "no significant winner (a tie within noise)" if p_corr >= 0.05 else "significant difference")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A benchmark report is defensible when the comparison is fair (deployment-matched split, audited '
    'features, identical protocol, disclosed budget), the winner is proven by a corrected significance '
    'test rather than a bigger mean, and the ranking is explained by the inductive biases -- and reporting '
    '\'no significant winner\' when a correct test cannot separate the gap from noise (credit_g: +0.008, '
    'corrected p>0.05) is exactly the honesty that will make an eventual RDL win from the same protocol '
    'believable."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 030 — Q3 checkpoint: write a one-page benchmark report

**Lesson:** [`lessons/0030-q3-checkpoint.html`](../lessons/0030-q3-checkpoint.html) · **Phase / Year:** Year 1 · Q3 (capstone)

**Primary reading:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep learning on typical tabular data?* ([arXiv:2207.08815](https://arxiv.org/abs/2207.08815)) — read in full.

**Dataset tier:** **A** (real `credit_g` via `relkit`).

**Skill you are practising:** assemble the whole Q3 evaluation-rigor toolkit into one defensible benchmark report — an honest split + leakage audit, a random-search **budget curve** (default AND ceiling) over a tuned GBDT and an honest neural baseline, a **corrected resampled t-test** on the gap, and an honest verdict (including "no significant winner").

**Exit criteria:** EXIT TICKET prints the one-page report — dataset + protocol, the budget-curve defaults/ceilings, the paired-CV gap with naive vs corrected p-values, and your honest one-line verdict.

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate (data, preprocessing, split, the two family samplers); just run.
- **TODO** cells — blanks (`____`); you implement the Q3 skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs scikit-learn + scipy (already in `requirements-labs.txt`). `credit_g` is fetched from OpenML on first run (needs network) then cached. The paired 5×5 CV fits ~50 small models — budget ~1 minute on CPU.

> **Note on the neural baseline.** We use sklearn's `MLPClassifier` (torch-free, runs anywhere) as the honest neural contestant. It stands in for the L028 tabular **ResNet** — same inductive biases; this checkpoint is about the *report*, not the architecture.'''),
        *bootstrap_cells(),
        md(r'''## Concept recap — the benchmark-report contract

A benchmark report is the artifact a skeptic reads before believing "model A beats model B". It is trustworthy only when three things hold:

**Fair** — same dataset (curated by explicit criteria), a **deployment-matched split** (temporal if a time order/drift exists, random only if truly i.i.d. — L021), one headline metric with prevalence, a disclosed tuning budget, and every fit-bearing transform inside a per-fold pipeline (L010). Features are **audited** for provenance with a model info sheet (L022): a suspiciously large win is a leak hypothesis first.

**Proven** — a reported gap is a random variable, so a "win" needs a **significance test**. k-fold scores overlap and are positively correlated, so the naive paired t-test over-rejects; use the **corrected resampled t-test** (Nadeau & Bengio 2003): inflate the variance by `(1/n + 1/(k-1))`. The correction can only *raise* the p-value.

**Explained + honest** — report the effect size, not a bare mean; explain the ranking with the three inductive biases (smoothness, rotation, uninformative-feature robustness — L025-27); and state the verdict honestly, **including "no significant winner"**.

**The contestants** (all under one identical protocol): a tuned **GBDT** (the Q2 incumbent), an honest tuned **neural baseline** (so DL isn't flattered by a weak bar — L028), and the **AutoML** bar (which only ties a tuned GBDT — L029).

**Toy micro-example (not this lab's answer).** Two models score 0.902 and 0.898 mean over 10 folds. That +0.004 looks like a win — but if the corrected resampled t-test gives p=0.4, the report must say *no significant difference*. A bigger mean is not a win.

Full write-up: [Lesson 030](../lessons/0030-q3-checkpoint.html).'''),
        md("## Setup — PROVIDED (data + preprocessing + split + family samplers)"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_SOL if solution else T3_CODE), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — stress the report

1. **Average the budget curve over orderings.** Re-run `budget_curve` with several `np.random.RandomState`
   seeds and average — does the MLP's non-monotone dip smooth out? (Grinsztajn averages over draw orderings
   exactly for this reason.)
2. **Add AutoML as a third contestant** under the *same* 5×5 CV (carve a validation slice from each fold's
   train for the CASH selection). Does it beat either family significantly, or tie both?
3. **Manufacture a temporal split.** Sort `credit_g` by one numeric column as a fake "time", split
   past/future instead of randomly, and re-run. Does the gap or the verdict change? (This rehearses the
   L021 discipline you would use for real timestamped data.)
4. **Effect size.** Add Cohen's d (`mean_d / sd(diffs)`) to the report. Is the standardized effect small
   even where a mean gap exists?'''),
        code(r'''# STRETCH — ungraded.
# import numpy as np
# print("Cohen's d:", round(mean_d / diffs.std(ddof=1), 3), " (small effect if |d| < 0.2)")'''),
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
    with open(os.path.join(HERE, "0030-q3-checkpoint.ipynb"), "w") as f:
        json.dump(build(solution=False), f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0030-q3-checkpoint.ipynb"), "w") as f:
        json.dump(build(solution=True), f, indent=1)
    print("wrote labs/0030-q3-checkpoint.ipynb + solution")


if __name__ == "__main__":
    main()
