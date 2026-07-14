"""Build Lab 023 (statistical comparison of classifiers) — emits the blank student
notebook and a filled solution notebook. Mirrors the L021/L022 concept-lab structure.

Run: .venv/bin/python labs/_build_l023.py
Then execute the solution:
  .venv/bin/jupyter nbconvert --to notebook --execute \
    --output solutions/0023-statistical-comparison.ipynb labs/solutions/0023-statistical-comparison.ipynb
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# ---- shared PROVIDED setup (mirrors labs/_verify_l023.py) ----
SETUP = r'''# PROVIDED — data, models, and the multi-dataset score matrix. Just run.
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0

def make_data(n=1000, d=12, n_informative=6, seed=SEED):
    """A moderate, mostly-linear signal — logistic regression does well."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[:n_informative] = rng.normal(size=n_informative)
    logit = X @ w + 1.5 * rng.normal(size=n)
    y = (logit > np.median(logit)).astype(int)
    return X, y

def lr():   return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
def nb():   return GaussianNB()
def tree(): return DecisionTreeClassifier(max_depth=3, random_state=SEED)

X, y = make_data()
rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)  # 100 paired folds
N_SPLITS = 10

# Nemenyi q_alpha (alpha=0.05) for a handful of k, from Demšar 2006 Table 5(a).
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}

print(f"data: n={len(X)}, d={X.shape[1]}, prevalence={y.mean():.3f}; CV = RepeatedStratifiedKFold(10x10)")'''


# ---- Task 1 ----
T1_MD = r'''## Task 1 — the paired fold differences + the naive test — TODO (crucial fragment)

**Goal:** score logistic regression and Gaussian Naive Bayes on the **same** 100 CV folds, form the per-fold differences, and run a naive paired t-test.

**Why it matters:** a mean-accuracy gap is a random variable. Pairing fold-by-fold cancels "some folds are just harder" and is the setup for asking *is this gap real?*

**You implement:** (1) `cross_val_score` for each model on `rkf`; (2) the per-fold difference `a - b`; (3) `stats.ttest_rel(a, b)` for the naive paired t-test.

**Hint boundary:** `cross_val_score(model(), X, y, cv=rkf, scoring="accuracy")` returns one score per fold. `stats.ttest_rel` returns `(t, p)`. Do not change `rkf`.'''

T1_CODE = r'''# TODO — build the paired differences and the naive p-value
a = ____                                  # 100 fold accuracies for lr()
b = ____                                  # 100 fold accuracies for nb()
diffs = ____                              # per-fold difference a - b

t_naive, p_naive = ____                   # stats.ttest_rel on the two score arrays

print(f"LogReg mean {a.mean():.4f}   GaussianNB mean {b.mean():.4f}")
print(f"mean(A-B) = {diffs.mean():+.4f}   ({(diffs>0).mean()*100:.0f}% of folds favour LR)")
print(f"NAIVE paired t-test: t={t_naive:+.3f}  p={p_naive:.3g}  -> "
      f"{'SIGNIFICANT' if p_naive < 0.05 else 'not sig'}")'''

T1_SOL = (T1_CODE
    .replace('a = ____                                  # 100 fold accuracies for lr()',
             'a = cross_val_score(lr(), X, y, cv=rkf, scoring="accuracy")   # 100 fold accuracies for lr()')
    .replace('b = ____                                  # 100 fold accuracies for nb()',
             'b = cross_val_score(nb(), X, y, cv=rkf, scoring="accuracy")   # 100 fold accuracies for nb()')
    .replace('diffs = ____                              # per-fold difference a - b',
             'diffs = a - b                             # per-fold difference a - b')
    .replace('t_naive, p_naive = ____                   # stats.ttest_rel on the two score arrays',
             't_naive, p_naive = stats.ttest_rel(a, b)  # stats.ttest_rel on the two score arrays'))

T1_CHECK = r'''# CHECK — do not edit
assert len(diffs) == 100, "Expected 100 paired fold scores (RepeatedStratifiedKFold 10x10)."
assert 0.004 < diffs.mean() < 0.016, "Mean gap should be a small positive ~+0.0098."
assert p_naive < 1e-3, "The naive t-test should call this tiny gap wildly significant."
print(f"Task 1 ok — a +{diffs.mean():.4f} gap, and the naive test screams p={p_naive:.1e}. "
      f"Suspicious for so small a gap...")'''


# ---- Task 2 ----
T2_MD = r'''## Task 2 — the corrected resampled t-test — TODO (crucial fragment)

**Goal:** implement Nadeau & Bengio's (2003) **corrected resampled t-test** and watch the p-value balloon — then see that swapping in Wilcoxon does **not** rescue you.

**Why it matters:** k-fold training sets overlap heavily (≈8/9 of rows in 10-fold), so fold scores are *positively correlated*. The naive standard error `σ/√n` assumes independence and is far too small. The correction inflates the variance of the mean by a term that captures the train/test overlap.

**The formula** (per-fold differences `d`, `n` folds, `k = n_splits` folds per repeat):

$$t = \frac{\bar d}{\sqrt{\left(\tfrac{1}{n} + \tfrac{n_\text{test}}{n_\text{train}}\right)\,\sigma^2(d)}}, \qquad \frac{n_\text{test}}{n_\text{train}} = \frac{1}{k-1}$$

**You implement:** the `correction` factor `1/n + 1/(n_splits - 1)`. Everything else is given.

**Hint boundary:** it is a sum of two fractions; the second uses `n_splits`, not `n`. Do not change the `t`/`p` lines.'''

T2_CODE = r'''# TODO — implement the variance correction
def corrected_resampled_t(diffs, n_splits):
    n = len(diffs)
    mean = np.mean(diffs)
    var = np.var(diffs, ddof=1)
    correction = ____                       # 1/n  +  n_test/n_train, and n_test/n_train = 1/(n_splits-1)
    t = mean / np.sqrt(correction * var)
    p = 2 * stats.t.cdf(-abs(t), df=n - 1)
    return t, p

t_corr, p_corr = corrected_resampled_t(diffs, N_SPLITS)
w_stat, p_wilcoxon = stats.wilcoxon(a, b)   # PROVIDED — the "distribution-free" alternative

print(f"CORRECTED resampled t: t={t_corr:+.3f}  p={p_corr:.3g}  -> "
      f"{'SIGNIFICANT' if p_corr < 0.05 else 'not sig'}")
print(f"Wilcoxon on the folds:              p={p_wilcoxon:.3g}  -> "
      f"{'SIGNIFICANT (still! -- the trap)' if p_wilcoxon < 0.05 else 'not sig'}")
print(f"naive p {p_naive:.1e}  ->  corrected p {p_corr:.3f}  (the 'win' dissolved into noise)")'''

T2_SOL = T2_CODE.replace(
    'correction = ____                       # 1/n  +  n_test/n_train, and n_test/n_train = 1/(n_splits-1)',
    'correction = 1 / n + 1 / (n_splits - 1) # 1/n  +  n_test/n_train, and n_test/n_train = 1/(n_splits-1)')

T2_CHECK = r'''# CHECK — do not edit
assert p_corr > 0.05, "Corrected test should NOT be significant for this tiny gap."
assert 0.10 < p_corr < 0.35, "Corrected p should land around ~0.19."
assert p_corr > 100 * p_naive, "The correction should blow the p-value up by orders of magnitude."
assert p_wilcoxon < 0.05, "Wilcoxon on correlated folds ALSO over-rejects (the trap this lesson warns about)."
print(f"Task 2 ok — corrected p={p_corr:.2f} (not sig), while naive t AND Wilcoxon both over-reject on "
      f"the correlated folds. On one dataset, correct the variance; don't just swap the test.")'''


# ---- Task 3 ----
T3_MD = r'''## Task 3 — many models over many datasets: Friedman + Nemenyi CD — TODO

**Goal:** rank four models over 12 datasets, run the **Friedman** omnibus test, and compute the **Nemenyi critical difference** — Demšar's recipe for reading a benchmark.

**Why it matters:** across datasets you cannot average accuracies (incommensurable) or assume normality, so you rank per dataset and test the *ranks*. This is exactly how TabArena / RelBench-style leaderboards should be read — and how an RDL win must be argued.

**You implement:** (1) per-dataset ranks with `stats.rankdata(-row)` (negate so rank 1 = best); (2) the Friedman test `stats.friedmanchisquare(*columns)`; (3) the critical difference `CD = q_alpha * sqrt(k*(k+1) / (6*N))`.

**Hint boundary:** `stats.rankdata(x)` ranks ascending (1 = smallest), so rank the **negated** scores to make 1 = best. `friedmanchisquare` takes one argument per model (a column of the score matrix). `Q_ALPHA_05[k]` is provided.'''

T3_CODE = r'''# PROVIDED — score four models on 12 varied synthetic datasets (takes ~1-2 min)
MODELS = {
    "LogReg":       lr(),
    "GaussianNB":   nb(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "HistGBDT":     HistGradientBoostingClassifier(random_state=SEED),
}
names = list(MODELS)
N = 12
score_matrix = np.zeros((N, len(names)))
for di in range(N):
    rng = np.random.default_rng(100 + di)
    n = int(rng.integers(400, 1200)); d = int(rng.integers(8, 20)); n_inf = int(rng.integers(3, d))
    Xi = rng.normal(size=(n, d)); w = np.zeros(d); w[:n_inf] = rng.normal(size=n_inf)
    yi = ((Xi @ w + rng.uniform(0.8, 2.2) * rng.normal(size=n)) > 0).astype(int)
    for mi, nm in enumerate(names):
        score_matrix[di, mi] = cross_val_score(MODELS[nm], Xi, yi, cv=5, scoring="accuracy").mean()
print("score matrix:", score_matrix.shape, "(datasets x models)")'''

T3_CODE2 = r'''# TODO — rank, run Friedman, and compute the Nemenyi critical difference
k = len(names)

ranks = np.zeros_like(score_matrix)
for di in range(N):
    ranks[di] = ____                          # rank this row so 1 = best (hint: rankdata of the NEGATED scores)
avg_ranks = ranks.mean(axis=0)

chi2, p_friedman = ____                        # stats.friedmanchisquare on the columns of score_matrix

cd = ____                                      # Q_ALPHA_05[k] * sqrt(k*(k+1) / (6*N))

order = np.argsort(avg_ranks)
for i in order:
    print(f"  {names[i]:14s} avg-rank {avg_ranks[i]:.3f}   mean-acc {score_matrix[:,i].mean():.3f}")
print(f"Friedman chi2={chi2:.2f}  p={p_friedman:.2e}  -> {'ranks differ' if p_friedman<0.05 else 'no diff'}")
print(f"Nemenyi CD (alpha=0.05, k={k}, N={N}) = {cd:.3f}")
best = names[int(np.argmin(avg_ranks))]
for i in order:
    if names[i] == best: continue
    gap = avg_ranks[i] - avg_ranks[int(np.argmin(avg_ranks))]
    print(f"  {best} vs {names[i]:14s}: rank gap {gap:.3f}  "
          f"({'significantly better' if gap > cd else 'NOT sig. different'})")'''

T3_SOL2 = (T3_CODE2
    .replace('ranks[di] = ____                          # rank this row so 1 = best (hint: rankdata of the NEGATED scores)',
             'ranks[di] = stats.rankdata(-score_matrix[di])  # rank this row so 1 = best')
    .replace('chi2, p_friedman = ____                        # stats.friedmanchisquare on the columns of score_matrix',
             'chi2, p_friedman = stats.friedmanchisquare(*[score_matrix[:, mi] for mi in range(k)])')
    .replace('cd = ____                                      # Q_ALPHA_05[k] * sqrt(k*(k+1) / (6*N))',
             'cd = Q_ALPHA_05[k] * np.sqrt(k * (k + 1) / (6 * N))'))

T3_CHECK = r'''# CHECK — do not edit
assert p_friedman < 1e-4, "Friedman should strongly reject the null (the ranks differ)."
assert abs(cd - 1.354) < 0.01, "CD for k=4, N=12 at alpha=0.05 should be ~1.354."
assert names[int(np.argmin(avg_ranks))] == "LogReg", "LogReg should have the best (lowest) average rank here."
lr_i = names.index("LogReg"); nb_i = names.index("GaussianNB"); rf_i = names.index("RandomForest")
assert avg_ranks[nb_i] - avg_ranks[lr_i] < cd, "LogReg vs NB rank gap should be within CD (not sig different)."
assert avg_ranks[rf_i] - avg_ranks[lr_i] > cd, "LogReg vs RandomForest gap should exceed CD (significant)."
print(f"Task 3 ok — Friedman p={p_friedman:.1e}, CD={cd:.3f}. LogReg tops the ranks; it beats the trees "
      f"significantly but is NOT distinguishable from NB. That's a CD diagram in numbers.")'''


# ---- Exit ----
EXIT_MD = r'''## Exit ticket — TODO

**Goal:** one printed summary to paste back to your teacher.

**Takeaway prompt:** in one sentence, why is a naive paired t-test on k-fold CV folds anticonservative, and what is the single-dataset fix?'''

EXIT_CODE = r'''# TODO: complete the takeaway string
print("=== EXIT TICKET — Lesson 023 ===")
print(f"one dataset : gap {diffs.mean():+.4f}  naive p {p_naive:.1e} (SIG) -> corrected p {p_corr:.2f} (noise)")
print(f"trap        : Wilcoxon on folds p {p_wilcoxon:.1e} (also over-rejects)")
print(f"many datasets: Friedman p {p_friedman:.1e}, CD {cd:.3f}; ranks {dict(zip(names, np.round(avg_ranks,2)))}")
print()
print("takeaway:", "____")'''

EXIT_SOL = EXIT_CODE.replace(
    '"____"',
    '"A naive paired t-test treats the k-fold scores as independent, but overlapping training sets make them '
    'positively correlated, so it underestimates the variance of the mean and over-rejects; on one dataset the fix '
    'is the corrected resampled t-test (inflate the variance by 1/n + 1/(k-1)), and across many datasets use '
    'Friedman + Nemenyi rank tests."')


def build(solution: bool):
    cells = [
        md(r'''# Lab 023 — Statistical comparison: is the gap real, or is it noise?

**Lesson:** [`lessons/0023-statistical-comparison.html`](../lessons/0023-statistical-comparison.html) · **Phase / Year:** Year 1 · Q3

**Paper:** Demšar 2006, *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR 7:1–30 ([jmlr.org](https://jmlr.org/papers/v7/demsar06a.html)) — abstract, §3.2 (Wilcoxon), §3.5 (Friedman + Nemenyi), CD diagrams. Single-dataset fix: Nadeau & Bengio 2003 (corrected resampled t-test); Dietterich 1998 (5×2cv, McNemar).

**Dataset tier:** C — synthetic (mechanism isolation). We isolate the statistics on controlled data so the p-values are reproducible; this is *not* a benchmark.

**Skill you are practising:** given per-fold scores for two models, run a *paired* significance test to decide whether the gap is real — implement the **corrected resampled t-test** and recognize why the naive one (and Wilcoxon) over-reject on CV folds; then compare many models over many datasets with **Friedman + Nemenyi CD**.

**Exit criteria:** EXIT TICKET prints the single-dataset collapse (naive p ≈ 1.2e−5 → corrected p ≈ 0.19), the Wilcoxon trap, and the multi-dataset Friedman/CD verdict (p ≈ 4e−6, CD ≈ 1.354).

---

### How this notebook works
- **PROVIDED** cells — complete boilerplate; just run.
- **TODO** cells — blanks (`____`); you implement the skill.
- **CHECK** cells — immediate feedback; do not edit.
- Run top to bottom. When **EXIT TICKET** prints cleanly, paste it to your teacher or say *"lab done"*.

### Environment
One-time: `bash labs/setup-env.sh` from repo root → kernel **Relational Labs (.venv)**. Needs **scikit-learn** and **scipy** only (no boosters, no network).'''),
        md(r'''## Concept recap — a reported gap is a random variable

The cross-validated score is an *estimate* of generalization error, and estimates wobble with the particular fold partition. So "model A beats B by 0.4%" is not a fact until you test whether that gap could have arisen by chance. Because both models are scored on the **same folds**, use a **paired** test on the per-fold differences `d_i = score_A(i) − score_B(i)`.

**The failure mode (single dataset).** A paired t-test assumes the `n` differences are *independent*, giving standard error `σ/√n`. But k-fold training sets overlap heavily (≈ 8/9 of the rows in 10-fold), so the fold scores are **positively correlated** and the true variance of the mean is much larger than `σ²/n`. The naive test underestimates variance and **over-rejects** — it manufactures "significant" wins. Nadeau & Bengio's **corrected resampled t-test** fixes it by inflating the variance:

$$\text{corrected } t = \frac{\bar d}{\sqrt{\left(\frac{1}{n} + \frac{1}{k-1}\right)\sigma^2(d)}}$$

**Toy micro-example (not this lab's answer).** Ten fold-differences all equal to `+0.01` with tiny scatter look "impossible by chance" to a naive t-test; but if those ten fits shared 90% of their data, you really have *far less* than ten independent measurements — the correction encodes that.

**Across datasets.** You cannot average accuracies or assume normality, so Demšar (2006) ranks the `k` models per dataset, tests the ranks with **Friedman**, and if it rejects, uses the **Nemenyi critical difference** `CD = q_α √(k(k+1)/6N)` to say which rank gaps are real — the CD diagram. Note: Wilcoxon/Friedman are for *independent datasets*; on the correlated folds of one dataset they over-reject just like the t-test.

Full write-up + the paired-difference and CD-diagram widgets: [Lesson 023](../lessons/0023-statistical-comparison.html).'''),
        md("## Setup — PROVIDED"),
        code(SETUP),
        md(T1_MD), code(T1_SOL if solution else T1_CODE), code(T1_CHECK),
        md(T2_MD), code(T2_SOL if solution else T2_CODE), code(T2_CHECK),
        md(T3_MD), code(T3_CODE), code(T3_SOL2 if solution else T3_CODE2), code(T3_CHECK),
        md(EXIT_MD), code(EXIT_SOL if solution else EXIT_CODE),
        md(r'''## Stretch (optional, ungraded) — push the mechanism

1. **Fold count vs the correction.** Re-run Task 1/2 with `RepeatedStratifiedKFold(5, 20)` and `(10, 20)`. The naive p keeps shrinking as you add folds; the corrected p barely moves — the `1/(k−1)` term is the floor the overlap imposes.
2. **5×2cv (Dietterich 1998).** Score the LR-vs-NB pair with five 2-fold repeats (`RepeatedStratifiedKFold(2, 5)`) and a plain paired t-test. Within each repeat the train/test sets are disjoint, so the correlation is tamed — compare its p-value to the naive 10×10 one.
3. **Effect size.** Even when a gap is significant, report Cohen's d = `mean(diffs) / std(diffs)`. A significant but tiny d is a warning that "significant" ≠ "worth the complexity".
4. **CD diagram by hand.** Sketch the Task-3 ranks on a number line and draw the bar joining models within `cd` — confirm it matches the widget in the lesson.'''),
        code(r'''# STRETCH — ungraded.
# for ns, nr in [(5,20),(10,20)]:
#     rk = RepeatedStratifiedKFold(n_splits=ns, n_repeats=nr, random_state=SEED)
#     aa = cross_val_score(lr(), X, y, cv=rk, scoring="accuracy")
#     bb = cross_val_score(nb(), X, y, cv=rk, scoring="accuracy")
#     _, pn = stats.ttest_rel(aa, bb); _, pc = corrected_resampled_t(aa-bb, ns)
#     print(ns, nr, "naive", round(pn,5), "corrected", round(pc,3))'''),
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
    with open(os.path.join(HERE, "0023-statistical-comparison.ipynb"), "w") as f:
        json.dump(student, f, indent=1)
    os.makedirs(os.path.join(HERE, "solutions"), exist_ok=True)
    with open(os.path.join(HERE, "solutions", "0023-statistical-comparison.ipynb"), "w") as f:
        json.dump(sol, f, indent=1)
    print("wrote labs/0023-statistical-comparison.ipynb + labs/solutions/0023-statistical-comparison.ipynb")


if __name__ == "__main__":
    main()
